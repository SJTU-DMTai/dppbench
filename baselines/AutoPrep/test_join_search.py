"""Join coverage, feedback integrity and reproducibility regressions."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from baselines.common.data_ctx import DataContext, TableContext
from baselines.AutoPrep.auto_prep import AutoPrep
from baselines.AutoPrep.solver import pipeline_signature, solve
from baselines.AutoPrep.transformation_model import JoinModel, TransformationModel
from baselines.AutoPrep.transformation_tree import build_transformation_tree


def context():
    primary = TableContext('loan', 'primary',
        dtypes={'loan_id': 'int64', 'account_id': 'int64', 'amount': 'float64', 'target': 'int64'},
        configured_target_col='target', configured_id_col='loan_id')
    tables = {'loan': primary}
    for name in ['account', 'trans', 'disp']:
        tables[name] = TableContext(name, 'secondary',
            dtypes={'account_id': 'int64', name + '_value': 'float64'},
            join_keys_to_primary=[{'primary': 'account_id', 'secondary': 'account_id'}],
            join_key_unique={'account_id': name == 'account'})
    return DataContext('tabular', 'fixture', 'loan', ['account', 'trans', 'disp'], tables,
                       prediction_task='binary')


class JoinSearchTests(unittest.TestCase):
    def setUp(self):
        self.ctx = context()
        self.transforms = TransformationModel(self.ctx)
        self.joins = JoinModel(self.ctx)

    def candidates(self, **kwargs):
        return solve(self.ctx, self.transforms, self.joins, seed=43, **kwargs)

    def test_initial_probes_cover_joins_with_matching_preprocessing(self):
        candidates = self.candidates(n_candidates=8)
        probes = [c for c in candidates if c.coverage]
        self.assertEqual(len(probes), 3)
        tails = []
        for candidate in candidates:
            joins = [s for s in candidate.pipeline.steps if s.op == 'JoinTable']
            self.assertEqual(len(joins), 1)
            table = joins[0].params['aux_df'][1:]
            self.assertEqual(candidate.join_set, (f'JoinTable({table})',))
        for candidate in probes:
            tails.append([(s.op, s.params) for s in candidate.pipeline.steps if s.op != 'JoinTable'])
        self.assertTrue(all(tail == tails[0] for tail in tails))
        self.assertLessEqual(len(candidates), 8)

    def test_coverage_continues_across_small_rounds(self):
        evaluated = set()
        tails = []
        for round_index in range(3):
            candidates = solve(self.ctx, self.transforms, self.joins, n_candidates=1,
                               evaluated_joins=evaluated, seed=43 + 31 * round_index, coverage_seed=43)
            tails.append([(s.op, s.params) for s in candidates[0].pipeline.steps[1:]])
            self.assertTrue(candidates[0].coverage)
            evaluated.update(candidates[0].join_set)
        self.assertEqual(evaluated, {e.name for e in self.joins.all_edges()})
        self.assertTrue(all(tail == tails[0] for tail in tails))
        self.assertFalse(self.candidates(n_candidates=1, evaluated_joins=evaluated)[0].coverage)

    def test_feedback_guides_search_after_coverage(self):
        self.joins.logp['JoinTable(trans)'] = 5.
        covered = {edge.name for edge in self.joins.all_edges()}
        candidate = self.candidates(n_candidates=1, evaluated_joins=covered)[0]
        self.assertEqual(candidate.join_set, ('JoinTable(trans)',))

    def test_full_signatures_distinguish_tables_and_ignore_dict_key_order(self):
        candidates = self.candidates(n_candidates=3)
        signatures = [pipeline_signature(c.pipeline) for c in candidates]
        self.assertEqual(len(set(signatures)), 3)
        copied = candidates[0].pipeline.copy()
        copied.steps[0].params = dict(reversed(list(copied.steps[0].params.items())))
        self.assertEqual(pipeline_signature(copied), signatures[0])
        copied.steps[1:] = reversed(copied.steps[1:])
        self.assertNotEqual(pipeline_signature(copied), signatures[0])
        self.assertEqual(len({pipeline_signature(c.pipeline) for c in self.candidates()}),
                         len(self.candidates()))

    def test_tree_does_not_generate_joins(self):
        import random
        leaves = build_transformation_tree(self.ctx, self.transforms, 'main_tabular', rng=random.Random(43))
        self.assertTrue(all('JoinTable' not in leaf.op_chain for leaf in leaves))

    def test_infeasible_auxiliary_table_is_excluded(self):
        self.ctx.secondary_tables.append('unrelated')
        self.ctx.tables['unrelated'] = TableContext('unrelated', 'secondary', dtypes={'x': 'float64'})
        self.assertNotIn('JoinTable(unrelated)', {e.name for e in JoinModel(self.ctx).all_edges()})

    def test_no_auxiliary_tables(self):
        self.ctx.secondary_tables = []
        self.ctx.tables = {'loan': self.ctx.tables['loan']}
        candidates = solve(self.ctx, TransformationModel(self.ctx), JoinModel(self.ctx), seed=43)
        self.assertTrue(candidates)
        self.assertTrue(all(not c.join_set for c in candidates))

    def test_recommendation_mandatory_join_keeps_its_feedback_edge(self):
        interaction = TableContext('interaction', 'primary',
            dtypes={'user_id': 'int64', 'item_id': 'int64', 'timestamp': 'int64', 'rating': 'int64'},
            configured_target_col='rating', configured_time_col='timestamp')
        user = TableContext('user', 'secondary', dtypes={'user_id': 'int64', 'age': 'int64'})
        ctx = DataContext('rec', 'fixture', 'interaction', ['user'],
                          {'interaction': interaction, 'user': user})
        candidates = solve(ctx, TransformationModel(ctx), JoinModel(ctx), seed=43)
        self.assertTrue(candidates)
        for candidate in candidates:
            self.assertEqual(candidate.join_set, ('JoinTable(user+item)',))
            self.assertEqual(candidate.pipeline.op_names().count('JoinTable'), 1)
            self.assertIn('CreateSequence', candidate.pipeline.op_names())

    def test_repair_cannot_silently_change_feedback_join(self):
        from baselines.AutoPrep import solver
        original = solver.repair_pipeline
        def changed(pipe, ctx, **kwargs):
            pipe = original(pipe, ctx, **kwargs)
            pipe.steps[0].params['aux_df'] = '$trans'
            return pipe
        with patch.object(solver, 'repair_pipeline', side_effect=changed):
            with self.assertRaisesRegex(ValueError, 'feedback edge'):
                self.candidates()

    def test_outer_loop_finishes_coverage_before_early_stop(self):
        class Evaluator:
            def __init__(self, **kwargs):
                pass
            def set_evaluate_std_test(self, enabled):
                pass
            def set_fast_mode(self, enabled):
                pass
            def evaluate_for_agent(self, pipeline):
                table = pipeline.steps[0].params['aux_df'][1:]
                score = {'account': .6, 'disp': .7, 'trans': .9}[table]
                return score, {'auc': score}, None
        with tempfile.TemporaryDirectory() as folder:
            runner = AutoPrep(task_dir='.', data_name='fixture', n_iters=1, n_candidates=1,
                              eval_full=False, output_dir=folder, verbose=False,
                              early_stop_patience=1)
            with patch('baselines.AutoPrep.auto_prep.AutoPrepEvaluator', Evaluator), \
                 patch.object(runner, '_build_context', return_value=self.ctx):
                result = runner.run()
            self.assertTrue(result['join_coverage']['complete'])
            self.assertEqual(len(result['history']), 4)
            self.assertEqual(len(result['join_coverage']['attempted']), 3)
            for row in result['history'][:2]:
                self.assertEqual(set(row['join_probs'].values()), {.5})
            self.assertGreater(result['history'][2]['join_probs']['JoinTable(trans)'], .5)
            saved = json.loads((Path(folder) / 'auto_prep_log.json').read_text())
            self.assertEqual(saved['search_protocol'], 'coverage_first_joins_v1')

    def test_candidates_match_across_process_hash_seeds(self):
        code = ("import json; from baselines.AutoPrep.test_join_search import context; "
                "from baselines.AutoPrep.solver import solve,pipeline_signature; "
                "from baselines.AutoPrep.transformation_model import JoinModel,TransformationModel; "
                "c=context(); print(json.dumps([pipeline_signature(x.pipeline) for x in "
                "solve(c,TransformationModel(c),JoinModel(c),seed=43)]))")
        results = []
        for hash_seed in ['1', '73']:
            env = dict(os.environ, PYTHONHASHSEED=hash_seed)
            result = subprocess.run([sys.executable, '-c', code], env=env,
                                    capture_output=True, text=True, check=True, timeout=60)
            results.append(result.stdout.strip().splitlines()[-1])
        self.assertEqual(results[0], results[1])

    def test_budget_rejection_leaves_coverage_pending(self):
        class Evaluator:
            def __init__(self, **kwargs):
                self.calls = 0
            def set_evaluate_std_test(self, enabled):
                pass
            def set_fast_mode(self, enabled):
                pass
            def evaluate_for_agent(self, pipeline):
                self.calls += 1
                if self.calls == 1:
                    return .7, {'auc': .7}, None
                return float('-inf'), {}, 'BudgetExceededError: evaluation budget exhausted'
        with tempfile.TemporaryDirectory() as folder:
            runner = AutoPrep(task_dir='.', data_name='fixture', n_iters=1, n_candidates=1,
                              eval_full=False, output_dir=folder, verbose=False)
            with patch('baselines.AutoPrep.auto_prep.AutoPrepEvaluator', Evaluator), \
                 patch.object(runner, '_build_context', return_value=self.ctx):
                result = runner.run()
            self.assertFalse(result['join_coverage']['complete'])
            self.assertEqual(len(result['join_coverage']['attempted']), 1)
            self.assertEqual(len(result['join_coverage']['pending']), 2)
            self.assertTrue(result['history'][1]['candidates'][0]['error'])


if __name__ == '__main__':
    unittest.main()
