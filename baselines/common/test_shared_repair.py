import unittest

from baselines.common.data_ctx import DataContext, TableContext
from baselines.common.pipeline import Pipeline, PipelineStep, check_pipeline
from baselines.common.shared_repair import complete_pipeline


class SharedRepairTest(unittest.TestCase):
    def context(self):
        ctx = DataContext('tabular', 'unit', 'main')
        ctx.tables['main'] = TableContext('main', 'primary', dtypes={'id': 'int64', 'target': 'int64', 'x': 'float64'},
                                          configured_target_col='target', configured_id_col='id')
        return ctx

    def test_preserves_instances_and_dag_and_is_idempotent(self):
        ctx = self.context()
        pipeline = Pipeline(steps=[PipelineStep('HandleMV', params={'cols': ['x']}, id='a', prev=['s']),
                                   PipelineStep('HandleMV', params={'cols': ['x'], 'fill_value': 7}, id='b', prev=['a'])],
                            sources=[{'id': 's', 'table': 'main'}], train_prev=['b'])
        effective, report = complete_pipeline(pipeline, ctx)
        self.assertEqual([step.to_dict() for step in effective.steps[:2]], [step.to_dict() for step in pipeline.steps])
        self.assertEqual(effective.steps[-1].op, 'LabelEncode')
        self.assertEqual(effective.steps[-1].prev, ['b'])
        self.assertEqual(len(report['added']), 1)
        again, diff = complete_pipeline(effective, ctx)
        self.assertEqual(again.hash(), effective.hash())
        self.assertEqual(diff['added'], [])
        self.assertTrue(check_pipeline(effective, ctx).valid)

    def test_rejects_empty_unsupported_and_overlength(self):
        ctx = self.context()
        with self.assertRaises(ValueError):
            complete_pipeline(Pipeline(), ctx)
        with self.assertRaises(ValueError):
            complete_pipeline(Pipeline([PipelineStep('Unknown')]), ctx)
        with self.assertRaisesRegex(ValueError, 'max_pipeline_len'):
            complete_pipeline(Pipeline([PipelineStep('HandleMV')]), ctx, max_len=1)

    def test_deepprep_executes_completed_candidate_without_changing_tree_path(self):
        from unittest.mock import Mock
        from baselines.DeepPrep.tree_agent import TreeAgent
        from baselines.DeepPrep.sandbox import ObsState, ExecutionResult
        ctx = self.context()
        obs = ObsState((2, 3), ['id', 'target', 'x'], {'id': 'int64', 'target': 'int64', 'x': 'float64'},
                       [], 0., 'main', tables={'main': {'columns': ['id', 'target', 'x']}})
        sandbox = Mock()
        sandbox.restore.return_value = obs
        sandbox.snapshot.return_value = b'partial'
        sandbox.execute_chain.return_value = ExecutionResult(True, obs=obs, snapshot=b'completed')
        evaluator = Mock()
        evaluator.evaluate_for_agent.return_value = (0.7, {'auc': 0.7}, None)
        agent = TreeAgent(Mock(), sandbox, ctx, shared_repair=True, downstream_evaluator=evaluator)
        agent.tree.add_root(b'root', 'main')
        child = agent.tree.add_child([PipelineStep('HandleMV')], b'partial', 'main')
        self.assertEqual(agent._answer_errors(child), [])
        effective = agent._pipeline_for_node(child)
        self.assertEqual(effective.op_names(), ['HandleMV', 'LabelEncode'])
        self.assertEqual([step.op for step in child.accumulated_steps], ['HandleMV'])
        self.assertEqual(sandbox.execute_chain.call_args.kwargs['pipeline'].hash(), effective.hash())
        feedback = agent._downstream_feedback(child)
        self.assertIn('shared-repair-v1', feedback)
        self.assertEqual(evaluator.evaluate_for_agent.call_args.args[0].hash(), effective.hash())
        self.assertIn(child.node_id, agent.repaired_candidates)

    def test_bat_end_reward_and_downstream_use_same_completed_dag(self):
        import random
        from unittest.mock import Mock, patch
        from baselines.BAT.action import EndAction, ActionContext
        from baselines.BAT.node import MCTSNode, MCTSNodeType
        from baselines.BAT.reward import BATReward
        ctx = self.context()
        ctx.shared_repair, ctx.repair_seed, ctx.repair_max_len = True, 42, 10
        parent = MCTSNode(MCTSNodeType.TRANSFORMATION, pipeline_steps=[PipelineStep('HandleMV')])
        sandbox = Mock()
        sandbox._initial_snapshot = b'root'
        sandbox.execute_chain.return_value.success = True
        action_ctx = ActionContext(Mock(), ctx, sandbox, random.Random(42))
        child = EndAction().create_children_nodes(parent, action_ctx)[0]
        self.assertIsNone(child.exec_error)
        self.assertEqual(child.effective_pipeline.op_names(), ['HandleMV', 'LabelEncode'])
        self.assertEqual(parent.pipeline_steps[0].op, 'HandleMV')
        evaluator = Mock()
        evaluator.metric_key = 'auc'
        evaluator.evaluate_for_agent.return_value = (0.8, {}, None)
        reward = BATReward(ctx, sandbox, downstream_evaluator=evaluator)
        with patch('baselines.BAT.reward.expected_target_columns', return_value=['x']), \
             patch('baselines.BAT.reward._sandbox_columns', return_value=['x']):
            value, report = reward.score(child)
        self.assertGreater(value, 0)
        self.assertEqual(sandbox.execute_chain.call_args.kwargs['pipeline'].hash(), child.effective_pipeline.hash())
        self.assertEqual(evaluator.evaluate_for_agent.call_args.args[0].hash(), child.effective_pipeline.hash())
        self.assertEqual(report['repair']['effective_hash'], child.effective_pipeline.hash())

    def test_llmautodp_scores_and_saves_effective_candidate(self):
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock
        from baselines.LLMAutoDP.llm_autodp import LLMAutoDP
        method = LLMAutoDP.__new__(LLMAutoDP)
        method.shared_repair = True
        method.verbose = False
        method.cfg = dict(seed=42, max_pipeline_len=10, n_candidates=1, n_rounds=1)
        raw = Pipeline([PipelineStep('HandleMV')])
        llm = Mock()
        import yaml
        llm.chat.return_value = json.dumps({'candidates': [yaml.safe_load(raw.to_yaml())], 'stop': False})
        evaluator = Mock()
        evaluator.evaluate.side_effect = [SimpleNamespace(success=True, fitness=0.5, error=None, metrics={}),
                                           SimpleNamespace(success=True, fitness=0.8, error=None, metrics={})]
        best, fitness, metrics, history, messages, reason = method._search(evaluator, llm, self.context())
        self.assertEqual(fitness, 0.8)
        record = history[1]['group'][0]
        self.assertEqual(record['repair']['effective_hash'], best.hash())
        self.assertEqual(evaluator.evaluate.call_args.args[0].hash(), best.hash())
        self.assertEqual(best.op_names(), ['HandleMV', 'LabelEncode'])

    def test_missing_join_is_inserted_before_primary_operations(self):
        ctx = self.context()
        ctx.secondary_tables = ['lookup']
        ctx.tables['lookup'] = TableContext('lookup', 'secondary', dtypes={'id': 'int64', 'value': 'float64'},
                                            join_keys_to_primary=[{'primary': 'id', 'secondary': 'id'}],
                                            join_key_unique={'id': True})
        original = Pipeline(steps=[PipelineStep('HandleMV', id='clean', prev=['raw'])],
                            sources=[{'id': 'raw', 'table': 'main'}], train_prev=['clean'])
        effective, report = complete_pipeline(original, ctx)
        join = effective.steps[0]
        self.assertEqual(join.op, 'JoinTable')
        self.assertEqual(join.prev[0], 'raw')
        self.assertEqual(len(join.prev), 2)
        self.assertEqual(effective.steps[1].prev, [join.id])
        self.assertEqual(original.steps[0].prev, ['raw'])
        self.assertTrue(check_pipeline(effective, ctx).valid, check_pipeline(effective, ctx).errors)


if __name__ == '__main__':
    unittest.main()
