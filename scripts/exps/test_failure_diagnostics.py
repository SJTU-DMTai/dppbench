import ast
import json
import pathlib
import tempfile
import unittest
from types import SimpleNamespace

from baselines.common.pipeline import Pipeline, PipelineStep
from scripts.exps.run_llm_all_tasks import failure_kind


ROOT = pathlib.Path(__file__).resolve().parents[2]


def run_body(path):
    tree = ast.parse((ROOT / path).read_text())
    return next(node.body for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'run')


def execute(nodes, namespace):
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), 'actual_diagnostic_branch', 'exec'), namespace)


class FailureDiagnosticsTest(unittest.TestCase):
    def test_haipipe_budget_branch_writes_matching_artifacts(self):
        body = run_body('baselines/HAIPipe/haipipe.py')
        initialize = next(node for node in body if isinstance(node, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'ai' for t in node.targets))
        branch = next(node for node in body if isinstance(node, ast.If) and isinstance(node.test, ast.Attribute)
                      and node.test.attr == 'evaluation_budget_exhausted')
        log = next(node for node in body if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'log' for t in node.targets))
        write_yaml = next(node for node in body if isinstance(node, ast.With)
                          and any(isinstance(n, ast.Name) and n.id == 'yaml_path' for n in ast.walk(node)))
        pipeline = Pipeline([PipelineStep('HandleMV')])
        with tempfile.TemporaryDirectory() as folder:
            namespace = dict(self=SimpleNamespace(data_name='unit', hi_source='test', training_mode='finetune'),
                             time=SimpleNamespace(time=lambda: 1.0), started=0.0, provenance={},
                             training_seconds=0.0, scorer_adaptation_seconds=0.0, agent_training={},
                             aipipe_path='policy.pt', al_path='scorer.pt',
                             evaluator=SimpleNamespace(evaluation_budget_exhausted=True, best_pipeline=pipeline,
                                                       n_unique_evaluations=100), hi=pipeline, output=pipeline,
                             selected=SimpleNamespace(fitness=0.7, metrics={'auc': 0.7}, candidates=[]),
                             full_fitness=None, full_metrics={}, report=SimpleNamespace(valid=True, errors=[]),
                             yaml_path=str(pathlib.Path(folder) / 'best_pipeline.yaml'))
            execute([initialize, branch, write_yaml, log], namespace)
            record = namespace['log']
            self.assertIsNone(record['ai_ops'])
            self.assertEqual(record['ai_source'], 'budget_exhausted_best_candidate')
            self.assertEqual(record['aipipe_trace'], [])
            json.dumps(record)
            saved = Pipeline.from_yaml(pathlib.Path(namespace['yaml_path']).read_text())
            self.assertEqual(saved.hash(), pipeline.hash())
            namespace.update(ai=pipeline)
            execute([log], namespace)
            self.assertEqual(namespace['log']['ai_source'], 'rollout')

    def test_flowpipe_successful_but_illegal_candidate_is_diagnosed(self):
        body = run_body('baselines/FlowPipe/flowpipe.py')
        start = next(i for i, node in enumerate(body) if isinstance(node, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'diagnostics' for t in node.targets))
        pipeline = Pipeline([PipelineStep('HandleMV')])
        candidate = dict(pipeline=pipeline, success=True, fitness=0.89, error_messages=[])
        namespace = dict(candidates=[candidate], ctx=None,
                         check_pipeline=lambda *args: SimpleNamespace(valid=False, errors=['missing JoinTable'], warnings=[]))
        execute(body[start:start + 3], namespace)
        self.assertEqual(namespace['eligible'], [])
        record = namespace['diagnostics'][0]
        self.assertTrue(record['success'])
        self.assertFalse(record['is_legal'])
        self.assertEqual(record['check_errors'], ['missing JoinTable'])
        self.assertEqual(Pipeline.from_yaml(record['pipeline_yaml']).hash(), pipeline.hash())
        json.dumps(record)

    def test_failure_classes_do_not_mask_failed_tasks(self):
        self.assertEqual(failure_kind('RuntimeError: graph tasks are unsupported.'), 'unsupported_task')
        self.assertEqual(failure_kind('FileNotFoundError: Pretrained agent missing: /graph/aipipe.pt'), 'missing_checkpoint')
        self.assertEqual(failure_kind('BudgetExceededError: memory budget exceeded: 190889MB'), 'memory_budget')
        self.assertEqual(failure_kind('ValueError: invalid pipeline'), 'execution_error')
        self.assertIsNone(failure_kind(None))

    def test_graph_requirements_separate_capability_from_checkpoint(self):
        from scripts.exps.run_llm_all_tasks import graph_requirements
        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(pretrained_dir=pathlib.Path(folder), flow_pretrained_dir=pathlib.Path(folder),
                                   training_mode='finetune')
            plan = [(method, 42, 'elliptic_bitcoin') for method in ('HAIPipe', 'CtxPipe', 'FlowPipe', 'BAT')]
            issues = graph_requirements(args, plan)
            self.assertEqual(sum(i['failure_kind'] == 'missing_checkpoint' for i in issues), 3)
            self.assertFalse(any(i['failure_kind'] == 'unsupported_task' for i in issues))
            self.assertFalse(any(i['baseline'] == 'BAT' for i in issues))
            self.assertEqual(graph_requirements(args, [('HAIPipe', 42, 'amazon_beauty')]), [])
            for method in ('HAIPipe', 'CtxPipe', 'FlowPipe'):
                target = pathlib.Path(folder) / method / 'LightGBM'
                target.mkdir(parents=True)
                names = ('aipipe.pt', 'alternative_evaluator.pt') if method == 'HAIPipe' else ('checkpoint.pt',)
                for name in names:
                    (target / name).write_bytes(b'checkpoint')
            self.assertEqual(graph_requirements(args, plan), [])
            args.training_mode = 'scratch'
            self.assertEqual(graph_requirements(args, plan), [])


if __name__ == '__main__':
    unittest.main()
