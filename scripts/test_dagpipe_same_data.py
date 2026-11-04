"""Synthetic same-data audits, with no training or experiment data."""
import unittest
import numpy as np
import pandas as pd
import torch

from baselines.dagpipe_old import runtime as old
from baselines.dagpipe_old.vendor import operations as old_ops
from baselines.DAGPipe import runtime as current
from scripts.diagnose_dagpipe_nodes import trace_checkpoint
from scripts.dagpipe_scaler_comparison import compare_input_encoders
from scripts.test_diagnose_dagpipe_nodes import checkpoint


class SameDataTests(unittest.TestCase):
    def test_old_encoders_are_unreachable_and_vendor_behavior_is_not_identity(self):
        self.assertFalse({'TargetEncoder', 'FrequencyEncoder'} & set(sum(old.limited_layers(9), [])))
        for name in ('TargetEncoder', 'FrequencyEncoder'):
            with self.assertRaisesRegex(ValueError, 'Unknown DAGPipe operation'):
                old.make_model([[name]], 2)
        x = torch.tensor([[1., 10.], [1., 20.], [2., 30.]], dtype=torch.float64)
        torch.testing.assert_close(old_ops.TargetEncoder()(x), x)
        expected = torch.tensor([[2/3, 1/3], [2/3, 1/3], [1/3, 1/3]], dtype=x.dtype)
        torch.testing.assert_close(old_ops.FrequencyEncoder()(x), expected)
        self.assertFalse(torch.equal(old_ops.FrequencyEncoder()(x[:2]), expected[:2]))

    def test_same_topology_weights_and_old_scaling_replay_exactly(self):
        layers = [['StandardScaler', 'RobustScaler'], ['Identity', 'MinMaxScaler']]
        a = old.make_model(layers, 2)
        b = current.make_model(layers, 2, scale_protocol='absolute_epsilon_v2')
        with torch.no_grad():
            for p in a.parameters():
                p.copy_(torch.linspace(-.3, .7, p.numel()).reshape(p.shape))
        b.load_state_dict(a.state_dict(), strict=True)
        ref = torch.tensor([[0., 1.], [1., 2.], [3., 2.], [4., 9.]], dtype=torch.float64)
        query = torch.tensor([[10., 4.], [-1., 8.]], dtype=torch.float64)
        torch.testing.assert_close(old.reference_forward(a, query, ref),
                                   current.reference_forward(b, query, ref), rtol=0, atol=0)
        self.assertAlmostEqual(a.dag_penalty().item(), 0., places=12)
        self.assertEqual(b.dag_penalty().item(), 0.)

    def test_public_matches_current_and_old_small_iqr_differs(self):
        payload = checkpoint()
        payload['reference'] = [[0., 0.], [1e-9, 0.], [2e-9, 0.], [3e-9, 0.]]
        frame = pd.DataFrame({'feat_0': [1., 100.], 'feat_1': [1., 2.]})
        before, output = trace_checkpoint(payload, frame)
        report, compared = trace_checkpoint(payload, frame, compare_public_scalers=True)
        torch.testing.assert_close(output, compared, rtol=0, atol=0)
        rows = report['public_scaler_comparison']
        self.assertEqual(sum(r['current_public_mismatches'] for r in rows), 0)
        robust = next(r for r in rows if r['op'] == 'RobustScaler' and r['column'] == 'feat_0')
        self.assertGreater(robust['current_abs_max'], 1e8)
        self.assertGreater(robust['current_old_mismatches'], 0)
        self.assertLess(robust['old_local_abs_max'], 100.)

    def test_numeric_input_encoders_equal_unless_missing_is_preserved(self):
        training = pd.DataFrame({'feat_0': [0., 1., 2., 3.], 'feat_1': [0., 1., 2., 3.]})
        payload = checkpoint()
        payload['encoder'] = current.InputEncoder().fit(training).state
        rows = compare_input_encoders(payload, training, training * 100)
        self.assertEqual(sum(r['old_current_mismatches'] + r['refit_checkpoint_mismatches'] for r in rows), 0)
        payload['operator_space'] = 'legacy_daghie'
        query = training.copy()
        query.loc[0, 'feat_0'] = np.nan
        rows = compare_input_encoders(payload, training, query)
        self.assertEqual(sum(r['old_current_mismatches'] for r in rows), 1)


if __name__ == '__main__':
    unittest.main()
