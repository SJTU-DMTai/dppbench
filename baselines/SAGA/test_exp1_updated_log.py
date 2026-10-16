"""Regressions for the rerun log: metadata, metrics, graph shape and search."""
import inspect
import random
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from baselines.common.data_ctx import infer_data_context, DataContext, TableContext
from baselines.common.pipeline import build_default_params, validate_step_params
from dppbench.graph_input import validate_graph_training


class UpdatedLogTests(unittest.TestCase):
    def bike_context(self):
        data = SimpleNamespace(
            train_df=pd.DataFrame({'instant': [1, 2], '_sort_idx': [1, 2], 'x': [1., 2.], 'cnt': [3, 4]}),
            id_col='instant', target_col='cnt', _sort_col='_sort_idx',
            auxiliary_dfs={'day': pd.DataFrame({'x': [1.]})},
            model_cfg={'model_params': {'task': 'regression'}})
        return infer_data_context('bike_sharing', data, 'tabular')

    def test_configured_metadata_not_in_feature_candidates(self):
        ctx = self.bike_context()
        self.assertEqual(ctx.id_col, 'instant')
        self.assertEqual(ctx.time_col, '_sort_idx')
        self.assertEqual(ctx.numeric_cols, ['x'])
        for seed in range(20):
            params = build_default_params('ClipOutlier', ctx, random.Random(seed))
            self.assertEqual(params['cols'], ['x'])
        self.assertIsNone(build_default_params('Undersample', ctx, random.Random(1)))
        self.assertTrue(validate_step_params('Undersample', {'target_col': 'cnt'}, ctx))
        self.assertIsNone(build_default_params('ConcatTable', ctx, random.Random(1)))

    def test_autoprep_metric_is_inferred(self):
        from baselines.AutoPrep.evaluator import AutoPrepEvaluator
        self.assertIsNone(inspect.signature(AutoPrepEvaluator).parameters['metric_key'].default)
        # Shared evaluator resolves model metrics when not explicitly overridden.
        from unittest.mock import patch
        with patch.object(AutoPrepEvaluator, 'executor_class') as executor:
            executor.return_value.get_model_config.return_value = {'train': {'metrics': ['rmse']}}
            evaluator = AutoPrepEvaluator('unused', 'bike_sharing')
            self.assertEqual(evaluator.metric_key, 'rmse')

    def test_graph_resampling_rejected_before_execution(self):
        from baselines.common.op_spec import operators_for_task
        ctx = DataContext('graph', 'fixture', 'main')
        self.assertNotIn('AugmentNoise', operators_for_task('graph'))
        self.assertTrue(validate_step_params('AugmentNoise', {'cols': ['feat_0']}, ctx))
        self.assertIsNone(build_default_params('AugmentNoise', ctx, random.Random(1)))

    def test_graph_shapes_and_edge_ranges_checked_on_cpu(self):
        graph = dict(x=np.ones((4, 2)), y=np.array([0, 1, 0, 1]), num_nodes=4,
                     edge_index=np.array([[0], [1]]),
                     train_mask=np.array([1, 1, 0, 0], dtype=bool),
                     val_mask=np.array([0, 0, 1, 1], dtype=bool))
        validate_graph_training(graph, ['auc'])
        with self.assertRaisesRegex(ValueError, 'must agree'):
            validate_graph_training(dict(graph, num_nodes=3), ['auc'])
        with self.assertRaisesRegex(ValueError, 'outside'):
            validate_graph_training(dict(graph, edge_index=np.array([[0], [4]])), ['auc'])

    def test_duplicate_graph_nodes_rejected(self):
        from dppbench.tasks.elliptic_bitcoin.elliptic_bitcoin_data import EllipticBitcoinData
        data = EllipticBitcoinData()
        data.train_df = pd.DataFrame({'txId': [1, 1], 'class': [1, 1], 'time_step': [1, 1], 'feat_0': [0., 1.]})
        data.auxiliary_dfs['edges'] = pd.DataFrame({'txId1': [1], 'txId2': [1]})
        with self.assertRaisesRegex(ValueError, 'unique nonmissing'):
            data.build_graph()

    def test_alphaclean_final_encoding_follows_type_changes(self):
        from baselines.AlphaClean.repair import Repair, repairs_to_pipeline
        ctx = self.bike_context()
        pipeline = repairs_to_pipeline([
            Repair('LabelEncode', 'train', {}),
            Repair('HandleMV', 'train', {'method': 'median'}),
            Repair('CastType', 'train', {'dtype_map': {'x': 'str'}}),
        ], ctx)
        self.assertEqual(pipeline.op_names()[-3:], ['CastType', 'LabelEncode', 'HandleMV'])

    def test_zero_iqr_does_not_amplify_rare_values(self):
        import torch
        from baselines.DAGPipe.vendor.operations import RobustScaler
        x = torch.tensor([[0.], [0.], [0.], [0.], [1.]], dtype=torch.float64, requires_grad=True)
        result = RobustScaler()(x)
        self.assertEqual(result[-1].item(), 1.)
        result.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())


if __name__ == '__main__':
    unittest.main()
