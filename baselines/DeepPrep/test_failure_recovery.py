import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd

from baselines.common.data_ctx import DataContext, TableContext
from baselines.common.pipeline import PipelineStep
from baselines.DeepPrep.operator_catalog import format_op_descriptions
from baselines.DeepPrep.sandbox import Sandbox, ObsState, ExecutionResult, categorical_dtype
from baselines.DeepPrep.target_schema import TargetSchema
from baselines.DeepPrep.tree_agent import TreeAgent, parse_response, protocol_diagnostic


def context(kind='tabular'):
    ctx = DataContext(kind, 'unit', 'main')
    ctx.tables['main'] = TableContext('main', 'primary', dtypes={'x': 'str', 'target': 'int64'},
                                      configured_target_col='target')
    return ctx


class FailureRecoveryTest(unittest.TestCase):
    def test_parser_ignores_reasoning_and_rejects_multiple_actions(self):
        self.assertEqual(parse_response('<think>HandleMV() </execute></think>').kind, 'noop')
        action = parse_response('<think>HandleMV() </execute></think>\nLabelEncode() </execute>')
        self.assertEqual(action.chain_str, 'LabelEncode()')
        self.assertEqual(parse_response('HandleMV() </execute>\nLabelEncode() </execute>').kind, 'noop')
        self.assertEqual(parse_response('1. HandleMV() </execute>').kind, 'noop')
        self.assertIn('numbered_or_bulleted', protocol_diagnostic('1. HandleMV() </execute>'))
        self.assertIn('incomplete_action', protocol_diagnostic('HandleMV()'))

    def test_string_dtype_and_revision_hint(self):
        for dtype in ['str', 'string', 'object', 'category']:
            self.assertTrue(categorical_dtype(dtype), dtype)
        self.assertFalse(categorical_dtype('float64'))
        agent = TreeAgent(Mock(), Mock(), context(), verbose=False)
        hint = agent._revision_hint()
        self.assertIn('LabelEncode(cols=None)', hint)
        self.assertIn('HandleMV(cols=null)', hint)
        self.assertIn('never include supervised target', hint)
        self.assertIn("categorical_columns=['x']", agent._initial_messages('root')[1]['content'])

    def test_parse_error_cannot_exceed_error_budget(self):
        obs = ObsState((2, 2), ['x', 'target'], {'x': 'float64', 'target': 'int64'}, [], 0., 'main',
                       tables={'main': {'columns': ['x', 'target']}})
        sandbox = Mock()
        sandbox.reset.return_value = ExecutionResult(True, obs=obs, snapshot=b'root')
        sandbox.restore.return_value = obs
        llm = Mock()
        llm.chat.return_value = '_HANDLEMV() </execute>'
        result = TreeAgent(llm, sandbox, context(), max_explore_turn=12, max_err_cnt=2, verbose=False).run()
        self.assertEqual(llm.chat.call_count, 2)
        self.assertEqual(result.n_errors, 2)
        self.assertEqual(result.termination_reason, 'error_limit')
        self.assertTrue(result.no_legal_candidate)
        self.assertFalse(result.success)
        sandbox.execute_chain.assert_not_called()

    def test_repeated_partial_path_can_be_explicitly_extended(self):
        obs = ObsState((2, 2), ['x', 'target'], {'x': 'float64', 'target': 'int64'}, [], 0., 'main',
                       tables={'main': {'columns': ['x', 'target']}})
        sandbox = Mock()
        sandbox.reset.return_value = ExecutionResult(True, obs=obs, snapshot=b'root')
        sandbox.restore.return_value = obs
        sandbox.execute_chain.side_effect = [ExecutionResult(True, obs=obs, snapshot=b'first'),
                                             ExecutionResult(True, obs=obs, snapshot=b'second')]
        llm = Mock()
        first = 'HandleMV(cols=None)'
        chain = first + ' --> LabelEncode(cols=None)'
        llm.chat.side_effect = [first + ' </execute>', first + ' </execute>',
                                chain + ' </execute>', '<answer>' + chain + '</answer>']
        result = TreeAgent(llm, sandbox, context(), max_explore_turn=4, verbose=False).run()
        self.assertTrue(result.success)
        self.assertEqual(result.pipeline.op_names(), ['HandleMV', 'LabelEncode'])
        self.assertEqual(sandbox.execute_chain.call_count, 2)
        self.assertIn('append the missing operators', '\n'.join(t['content'] for t in result.transcript))

    def test_graph_catalog_and_real_sandbox_execution(self):
        from dppbench.tasks.elliptic_bitcoin.elliptic_bitcoin_data import EllipticBitcoinData
        ctx = context('graph')
        ctx.tables['main'].configured_target_col = 'class'
        ctx.tables['main'].configured_id_col = 'txId'
        ctx.tables['main'].configured_time_col = 'time_step'
        ctx.tables['main'].dtypes = {'txId': 'int64', 'time_step': 'int64', 'class': 'int64', 'feat_0': 'float64'}
        self.assertIn('ExtractGraphFeature', format_op_descriptions('graph'))
        self.assertNotIn('Oversample', format_op_descriptions('graph'))
        schema = TargetSchema.from_context(ctx)
        self.assertEqual(set(schema.required_columns), {'txId', 'time_step', 'class'})
        data = EllipticBitcoinData()
        data.train_df = pd.DataFrame({'txId': [1, 2, 3], 'time_step': [1, 2, 3],
                                      'class': [0, 1, -1], 'feat_0': [1., 2., 3.]})
        data.auxiliary_dfs = {'edges': pd.DataFrame({'txId1': [1, 2], 'txId2': [2, 3]})}
        data.model_cfg = {'model': 'GraphSAGE', 'feature': {'target_col': 'class'}, 'train': {'seed': 42}}
        data._skip_std_test_loading = True
        sandbox = Sandbox.__new__(Sandbox)
        sandbox.task_type, sandbox.data_name, sandbox.seed = 'graph', 'unit', 42
        sandbox.data, sandbox._ctx = data, ctx
        with tempfile.TemporaryDirectory() as temp:
            sandbox._tmpdir = temp
            snapshot = sandbox.snapshot()
            obs = sandbox._observe()
            self.assertIn('main', obs.tables)
            self.assertIn('edges', obs.tables)
            self.assertEqual(schema.validate_observation(obs), [])
            result = sandbox.execute_chain([PipelineStep('HandleMV', 'main', {'cols': ['feat_0']})])
            self.assertTrue(result.success, result.error)
            self.assertEqual(sandbox.data.train_df['class'].tolist(), [0, 1, -1])
            failed = sandbox.execute_chain([PipelineStep('Oversample', 'main', {})])
            self.assertFalse(failed.success)
            restored = sandbox.restore(snapshot)
            self.assertEqual(restored.shape, (3, 4))


if __name__ == '__main__':
    unittest.main()
