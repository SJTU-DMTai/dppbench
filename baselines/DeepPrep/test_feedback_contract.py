import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd

from baselines.common.data_ctx import DataContext, TableContext
from baselines.DeepPrep.sandbox import Sandbox, ObsState, ExecutionResult
from baselines.DeepPrep.tree_agent import TreeAgent


SEQUENCE = 'CreateSequence(user_col="user_id", item_col="item_id", time_col="timestamp", seq_col="item_id_seq", max_len=20)'
BASE = 'JoinTable(method="rec", how="left", user_col="user_id", item_col="item_id", item_df="$item_df") --> ' + SEQUENCE


class MaterializingSandbox:
    def __init__(self):
        self.ops = ()

    def snapshot(self):
        return self.ops

    def observe(self):
        types = {'user_id': 'categorical', 'item_id': 'categorical', 'rating': 'numeric'}
        if 'JoinTable' in self.ops:
            types['brand'] = 'categorical'
        if 'CreateSequence' in self.ops:
            types['item_id_seq'] = 'categorical_list'
        if 'HashEncode' in self.ops:
            types['h_brand'] = 'categorical'
        return ObsState((3, len(types)), list(types), {c: 'int64' for c in types}, [], 0.,
                        'observed_columns=' + ','.join(types),
                        tables={'interaction': {}, 'item': {}}, col_types=types)

    def restore(self, snapshot):
        self.ops = snapshot
        return self.observe()

    def reset(self):
        self.ops = ()
        return ExecutionResult(True, self.observe(), snapshot=self.snapshot())

    def execute_chain(self, steps, pipeline=None):
        self.ops = tuple(s.op for s in steps) if pipeline is not None else self.ops + tuple(s.op for s in steps)
        return ExecutionResult(True, self.observe(), snapshot=self.snapshot())


class FeedbackContractTest(unittest.TestCase):
    def agent(self, responses, scores):
        ctx = DataContext('rec', 'unit', 'interaction', secondary_tables=['item'])
        ctx.tables['interaction'] = TableContext('interaction', 'primary',
            dtypes={'user_id': 'int64', 'item_id': 'int64', 'timestamp': 'int64', 'rating': 'int64'},
            configured_target_col='rating', configured_time_col='timestamp', configured_id_col='user_id')
        ctx.tables['item'] = TableContext('item', 'secondary', dtypes={'item_id': 'int64', 'brand': 'object'})
        llm = Mock()
        llm.chat.side_effect = responses
        evaluator = Mock(metric_key='auc')
        evaluator.evaluate_for_agent.side_effect = [(score, {'auc': score}, None) for score in scores]
        agent = TreeAgent(llm, MaterializingSandbox(), ctx, shared_repair=True, verbose=False,
                          max_explore_turn=10, downstream_evaluator=evaluator,
                          model_config={'model': 'DIN', 'feature': {'seq_cols': {'item_id_seq': 'item_id'}}})
        return agent, evaluator

    def test_effective_observation_path_remaining_and_answer_agree(self):
        agent, evaluator = self.agent([SEQUENCE + ' </execute>', '<answer>' + BASE + '</answer>'], [.6])
        result = agent.run()
        self.assertTrue(result.success)
        self.assertEqual(result.termination_reason, 'model_answer')
        first = next(e['content'] for e in result.transcript if e['role'] == 'environment')
        self.assertIn('materialized_path: JoinTable', first)
        self.assertIn('observed_columns=user_id,item_id,rating,brand,item_id_seq', first)
        self.assertIn('mandatory_ops_remaining: []', first)
        self.assertEqual(evaluator.evaluate_for_agent.call_count, 1)
        self.assertEqual(result.n_turns, 2)
        self.assertEqual(result.pipeline.op_names(), ['JoinTable', 'CreateSequence'])
        self.assertTrue(any(n.repaired_from for n in agent.tree.nodes.values()))

    def test_only_repeats_use_remaining_turns_and_keep_best(self):
        agent, evaluator = self.agent([SEQUENCE + ' </execute>'] + [BASE + ' </execute>'] * 9, [.6])
        result = agent.run()
        self.assertEqual(result.termination_reason, 'turn_limit')
        self.assertEqual(agent.llm.chat.call_count, 10)
        self.assertEqual(evaluator.evaluate_for_agent.call_count, 1)
        self.assertEqual(result.pipeline.op_names(), ['JoinTable', 'CreateSequence'])

    def test_extension_keeps_repaired_side_source_and_best_survives_answer(self):
        changed = BASE + ' --> HashEncode(cols=["brand"])'
        agent, evaluator = self.agent([SEQUENCE + ' </execute>', changed + ' </execute>',
                                      *([BASE + ' </execute>'] * 4), '<answer>' + BASE + '</answer>'], [.6, .7])
        result = agent.run()
        self.assertTrue(result.success)
        self.assertEqual(result.n_turns, 7)
        self.assertEqual(result.termination_reason, 'model_answer')
        self.assertIn('HashEncode', result.pipeline.op_names())
        self.assertEqual(evaluator.evaluate_for_agent.call_count, 2)
        self.assertIn('item', [s['table'] for s in result.pipeline.sources])
        summaries = [json.loads(e['content'].split('Sampled downstream feedback: ')[1].splitlines()[0])
                     for e in result.transcript if 'Sampled downstream feedback: ' in e['content']]
        self.assertAlmostEqual(summaries[1]['delta_vs_previous_best'], .1)
        self.assertEqual(summaries[1]['comparison'], 'improved')

    def test_profiles_exclude_validation_and_test_and_measure_sequence_lengths(self):
        sandbox = Sandbox.__new__(Sandbox)
        sandbox.data = SimpleNamespace(col_types={'seq': 'categorical_list'},
                                       model_cfg={'feature': {'target_col': 'label', 'seq_cols': {'seq': 'item'}}})
        frame = pd.DataFrame({'seq': [[], [], [1, 2], [1] * 100, [1] * 200],
                              'x': [1., None, 2., None, None], 'label': [1, 0, 1, 0, 0],
                              '__split__': ['train', 'train', 'train', 'val', 'std_test']})
        profile = sandbox._training_profile(sandbox._visible_frame(frame))
        self.assertEqual(profile['rows'], 3)
        self.assertAlmostEqual(profile['sequences']['seq']['nonempty_ratio'], 1 / 3)
        self.assertEqual(profile['sequences']['seq']['length_quantiles']['max'], 2)
        self.assertAlmostEqual(profile['feature_missing_rates']['x'], 1 / 3)
        self.assertNotIn('label', profile['feature_missing_rates'])

    def test_worse_candidate_feedback_and_schema_minimum(self):
        changed = BASE + ' --> HashEncode(cols=["brand"])'
        agent, _ = self.agent([SEQUENCE + ' </execute>', changed + ' </execute>',
                               '<answer>' + BASE + '</answer>'], [.6, .5])
        result = agent.run()
        self.assertNotIn('HashEncode', result.pipeline.op_names())
        feedbacks = [e['content'] for e in result.transcript if e['role'] == 'environment']
        self.assertIn('worse than previous best', feedbacks[1])
        summary = json.loads(feedbacks[1].split('Sampled downstream feedback: ')[1].splitlines()[0])
        self.assertAlmostEqual(summary['delta_vs_previous_best'], -.1)
        self.assertIn('NOT a whitelist', agent.target_schema.render())
        self.assertIn('column-order', agent.target_schema.render())


if __name__ == '__main__':
    unittest.main()
