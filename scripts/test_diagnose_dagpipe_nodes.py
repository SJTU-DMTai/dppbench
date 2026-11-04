import csv
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch

from baselines.DAGPipe.runtime import FrozenTransform, InputEncoder, make_model, reference_forward
from scripts.diagnose_dagpipe_nodes import (trace_checkpoint, write_reports,
                                         compare_training_reference, load_elliptic_training_features)
from scripts.diagnose_dagpipe_nodes import select_elliptic_training_features, restore_training_availability


def checkpoint(space='current', version='v3'):
    layers = [['StandardScaler', 'RobustScaler', 'MinMaxScaler'], ['Identity']]
    if space == 'legacy_daghie':
        layers.insert(0, ['MeanImputer', 'MedianImputer', 'KNNImputer'])
    train = pd.DataFrame({'feat_0': [0., 1., 2., 3.], 'feat_1': [0., 1., 2., None]})
    encoder = InputEncoder(preserve_missing=space == 'legacy_daghie').fit(train)
    protocol = 'public_v1' if version == 'v3' else None
    model = make_model(layers, 2, operator_space=space, scale_protocol=protocol)
    return {'format': f'dppbench.DAGPipe.reference.{version}', 'task': 'elliptic_bitcoin', 'seed': 42,
            'encoder': encoder.state, 'layers': layers, 'reference': encoder.transform(train).tolist(),
            'operator_space': space, 'scale_protocol': protocol,
            'state_dict': {name: value.tolist() for name, value in model.state_dict().items()}}


class DiagnoseNodesTests(unittest.TestCase):
    def test_legacy_classes_restore_training_membership_only(self):
        frame = pd.DataFrame({'txId': [1, 2, 3, 4, 5], 'time_step': [1, 34, 35, 2, 1],
                              'feat_0': [10, 20, 30, 40, 50],
                              '__split__': ['train', 'train', 'val', 'train', 'std_test']})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'classes.csv'
            # Non-training labels intentionally invalid: they must not be parsed.
            path.write_text('txId,class\n1,1\n2,2\n3,DO_NOT_PARSE\n4,unknown\n5,DO_NOT_PARSE\n6,DO_NOT_PARSE\n')
            restored = restore_training_availability(frame, path)
            self.assertEqual(select_elliptic_training_features(restored, ['feat_0']).feat_0.tolist(), [10, 20])
            self.assertEqual(restored['class'].tolist(), [0, 0, -1, -1, -1])
            self.assertNotIn('class', frame)
            path.write_text('txId,class\n1,1\n2,2\n')
            with self.assertRaisesRegex(ValueError, 'missing'):
                restore_training_availability(frame, path)
            path.write_text('txId,class\n1,1\n1,2\n2,2\n4,unknown\n')
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                restore_training_availability(frame, path)

    def test_training_comparison_preserves_checkpoint_and_replay(self):
        import copy
        payload = checkpoint()
        payload['transform_batch_size'] = 2
        original = copy.deepcopy(payload)
        training = pd.DataFrame({'feat_0': [0., 10., 20., 30.], 'feat_1': [0., 1., 2., 3.]})
        before, output = trace_checkpoint(payload, training)
        rows = compare_training_reference(payload, training)
        robust = next(r for r in rows if r['op'] == 'RobustScaler' and r['column'] == 'feat_0')
        self.assertAlmostEqual(robust['train_to_reference_iqr_ratio'], 10.)
        self.assertEqual(robust['outside_reference_range_fraction'], .75)
        self.assertEqual(payload, original)
        after, replay = trace_checkpoint(payload, training)
        torch.testing.assert_close(output, replay, rtol=0, atol=0)
        self.assertEqual(before, after)
        before.update(training_reference_comparison=rows, comparison_training_rows=4)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'report'
            write_reports(before, path)
            self.assertTrue((path / 'training_reference_comparison.csv').exists())

    def test_training_loader_uses_actual_masks_and_ignores_test_file(self):
        frame = pd.DataFrame({'txId': range(6), 'time_step': [1, 34, 35, 40, 2, 1],
                              'class': [0, 1, 1, 0, -1, 1], 'feat_0': range(6),
                              '__split__': ['train'] * 5 + ['std_test']})
        result = select_elliptic_training_features(frame, ['feat_0'])
        self.assertEqual(result.feat_0.tolist(), [0, 1])
        with self.assertRaisesRegex(ValueError, 'class availability'):
            select_elliptic_training_features(frame.drop(columns='class'), ['feat_0'])
        try:
            import pyarrow
        except ImportError:
            self.skipTest('Mask checks passed; Parquet integration requires pyarrow')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            frame.to_parquet(path / 'train_frozen.parquet')
            (path / 'std_test.parquet').write_text('must never be opened')
            result = load_elliptic_training_features(path, ['feat_0'])
            self.assertEqual(result.feat_0.tolist(), [0, 1])
            frame.drop(columns='class').to_parquet(path / 'train_frozen.parquet')
            with self.assertRaisesRegex(ValueError, 'lacks class'):
                load_elliptic_training_features(path, ['feat_0'])

    def test_robust_peak_keeps_global_row_and_transaction_id(self):
        payload = checkpoint()
        payload['transform_batch_size'] = 2
        frame = pd.DataFrame({'feat_0': [0., 1., 2., 1e10], 'feat_1': [0., 1., 2., 3.],
                              'txId': ['a', 'b', 'c', 'extreme']})
        report, output = trace_checkpoint(payload, frame)
        peak = next(p for p in report['robust_scaler_peaks'] if p['split'] == 'query')
        self.assertEqual((peak['row_position'], peak['txId'], peak['column']), (3, 'extreme', 'feat_0'))
        self.assertEqual(report['reference_statistics_changes_across_batches'], 0)
        self.assertEqual(report['full_query_output_abs_max'], output.abs().max().item())

    def test_first_crossing_locates_scaler_and_feature(self):
        payload = checkpoint()
        payload['layers'] = [['StandardScaler']]
        model = make_model(payload['layers'], 2, scale_protocol='public_v1')
        payload['state_dict'] = {k: v.tolist() for k, v in model.state_dict().items()}
        payload['reference'] = [[i * 2e-6, float(i)] for i in range(4)]
        spec = payload['encoder'][0]
        query = pd.DataFrame({'feat_0': [1000 * spec['scale'] + spec['mean']], 'feat_1': [1.]})
        report, _ = trace_checkpoint(payload, query)
        for threshold in (10_000, 1_000_000, 100_000_000):
            crossing = report['first_threshold_crossings'][f'query:{threshold}']
            self.assertEqual(crossing['op'], 'StandardScaler')
            self.assertEqual(crossing['stage'], 'operator_output')
            self.assertEqual(crossing['columns'], ['feat_0'])
        denominator = report['scaler_statistics'][0]['denominator']
        self.assertAlmostEqual(denominator, torch.tensor(payload['reference'], dtype=torch.float64)[:, 0].std().item())

    def test_real_replay_equivalence_and_files(self):
        frame = pd.DataFrame({'feat_0': [1e9, 0.], 'feat_1': [1., None], 'class': [9, 8]})
        for space in ('current', 'legacy_daghie'):
            with self.subTest(space=space):
                payload = checkpoint(space)
                report, actual = trace_checkpoint(payload, frame)
                frozen = FrozenTransform(payload)
                with torch.no_grad():
                    expected = reference_forward(frozen.model, frozen.reference.new_tensor(
                        frozen.encoder.transform(frame)), frozen.reference)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                stages = {event['stage'] for event in report['events']}
                self.assertTrue({'before_routing', 'after_routing', 'node_edge_contribution',
                                 'layer_edge_contribution', 'layer_mean', 'layer_aggregate',
                                 'final_output'}.issubset(stages))
                self.assertEqual(report['first_threshold_crossings']['query:100000000']['stage'], 'input_encoding')
                self.assertEqual({row['op'] for row in report['scaler_statistics']},
                                 {'StandardScaler', 'RobustScaler', 'MinMaxScaler'})
                if space == 'legacy_daghie':
                    self.assertEqual(report['events'][0]['reference']['summary']['nan_count'], 1)
                    self.assertIn('operator_output_before_routing', stages)
                with tempfile.TemporaryDirectory() as folder:
                    output_dir = Path(folder) / 'reports'
                    write_reports(report, output_dir)
                    self.assertEqual(json.loads((output_dir / 'diagnosis.json').read_text()), report)
                    with (output_dir / 'node_statistics.csv').open(newline='') as file:
                        self.assertTrue(any(row['column'] == 'feat_0' for row in csv.DictReader(file)))
                    self.assertTrue((output_dir / 'scaler_statistics.csv').is_file())
                    self.assertTrue((output_dir / 'summary.md').is_file())

    def test_historical_protocol_is_explicit(self):
        payload = checkpoint(version='v2')
        with self.assertRaisesRegex(ValueError, 'historical scaling rule'):
            trace_checkpoint(payload)
        for protocol in ('relative-floor', 'absolute-epsilon'):
            report, output = trace_checkpoint(payload, v2_protocol=protocol)
            self.assertEqual(report['scale_protocol'], protocol)
            self.assertTrue(torch.isfinite(output).all())
        with self.assertRaisesRegex(ValueError, 'Unsupported DAGPipe pipeline format'):
            trace_checkpoint(dict(payload, format='legacy.v1'))
