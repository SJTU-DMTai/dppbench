"""Regression checks for inductive DAGPipe and NYC's frozen-data contract."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import tempfile

import numpy as np
import pandas as pd
import torch

from baselines.DAGPipe.runtime import InputEncoder, limited_layers, make_model, reference_forward
from baselines.DAGPipe.pipeline import DAGPipePipeline
from baselines.DAGPipe.tabular_adapter import tabular_adapter, augment_splits
from baselines.DAGPipe.vendor.operations import FrequencyEncoder

ROOT = Path(__file__).resolve().parents[1]


class DAGPipeTests(unittest.TestCase):
    def test_reference_transform_independent_of_query_batch_and_roundtrip(self):
        torch.manual_seed(42)
        frame = pd.DataFrame({'a': [0., 1., 2., 4., 20.], 'b': [1., 1., 1., 2., 2.]})
        encoder = InputEncoder().fit(frame)
        x = torch.from_numpy(encoder.transform(frame))
        model = make_model(limited_layers(9), 2)
        reference = x[:4]
        combined = reference_forward(model, x, reference)
        single = torch.cat([reference_forward(model, row[None], reference) for row in x])
        torch.testing.assert_close(combined, single)
        perturbed = reference_forward(model, torch.cat([x[:1], x[1:]*1000]), reference)
        torch.testing.assert_close(combined[:1], perturbed[:1])
        combined.square().mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        self.assertAlmostEqual(model.dag_penalty().item(), 0., places=12)
        payload = {'format': 'dppbench.DAGPipe.reference.v2', 'task': 'test', 'seed': 42,
                   'encoder': encoder.state, 'layers': limited_layers(9),
                   'reference': reference.tolist(),
                   'state_dict': {k: v.detach().tolist() for k, v in model.state_dict().items()}}
        restored = DAGPipePipeline.from_yaml(DAGPipePipeline(payload).to_yaml())
        np.testing.assert_allclose(restored.transform(frame), combined.detach().numpy())
        payload['format'] = 'dppbench.DAGPipe.reference.v1'
        with self.assertRaisesRegex(ValueError, 'retrain'):
            DAGPipePipeline(payload)

    def test_frequency_counts_do_not_mix_columns(self):
        x = torch.tensor([[1., 1.], [1., 2.], [2., 3.]], dtype=torch.float64)
        expected = torch.tensor([[2/3, 1/3], [2/3, 1/3], [1/3, 1/3]], dtype=x.dtype)
        torch.testing.assert_close(FrequencyEncoder()(x), expected)

    def test_default_history_is_joined_without_cross_customer_or_target(self):
        data = SimpleNamespace(target_col='TARGET', auxiliary_dfs={'monthly_history': pd.DataFrame({
            'ID': [1, 1, 2], 'PAY_STATUS': [0., 2., 9.], 'TARGET': [9, 9, 9]})})
        train = pd.DataFrame({'ID': [2, 1, 3], 'TARGET': [0, 1, 0]})
        out = augment_splits(data, {'train': train, 'test': None}, tabular_adapter('default_credit'))
        self.assertEqual(out['train']['ID'].tolist(), [2, 1, 3])
        self.assertEqual(out['train']['TARGET'].tolist(), [0, 1, 0])
        self.assertEqual(out['train']['monthly_history__PAY_STATUS__mean'].iloc[:2].tolist(), [9., 1.])
        self.assertTrue(pd.isna(out['train']['monthly_history__PAY_STATUS__mean'].iloc[2]))
        self.assertIsNone(out['test'])
        self.assertFalse(any('__TARGET' in c for c in out['train']))

    def test_berka_key_join_and_duplicate_rejection(self):
        data = SimpleNamespace(target_col='target', auxiliary_dfs={'account': pd.DataFrame({
            'account_id': [2, 1], 'district_id': [20, 10]}), 'trans': pd.DataFrame({
            'account_id': [1, 2], 'date': [930101, 930101], 'amount': [10., 20.]})})
        main = pd.DataFrame({'account_id': [1, 2], 'date': [930102, 930102], 'target': [0, 1]})
        out = augment_splits(data, {'train': main}, tabular_adapter('berka'))['train']
        self.assertEqual(out['account__district_id'].tolist(), [10, 20])
        data.auxiliary_dfs['account']['account_id'] = 1
        with self.assertRaises(ValueError):
            augment_splits(data, {'train': main}, tabular_adapter('berka'))

    def test_berka_history_cutoff_row_identity_and_label_independence(self):
        from baselines.DAGPipe.tabular_adapter import _join_berka_history
        loans = pd.DataFrame({'account_id': [1, 1, 2], 'date': [930102, 930104, 930104],
                              'target': [0, 1, 0]}, index=[9, 4, 20])
        trans = pd.DataFrame({'account_id': [1, 1, 1, 1],
                              'date': [930101, 930102, 930104, 930105],
                              'amount': [10., 20., 1000., 2000.], 'balance': [100., 200., 9999., 99999.]})
        result = _join_berka_history(loans, trans)
        self.assertEqual(result['trans__count'].tolist(), [1, 2, 0])
        self.assertEqual(result['trans__amount__mean'].iloc[:2].tolist(), [10., 15.])
        self.assertTrue(pd.isna(result['trans__balance__mean'].iloc[2]))
        self.assertEqual(result['target'].tolist(), loans['target'].tolist())
        future_changed = trans.copy()
        future_changed.loc[2:, 'amount'] = -1e9
        pd.testing.assert_frame_equal(result, _join_berka_history(loans, future_changed))
        permuted = _join_berka_history(loans.assign(target=[1, 0, 1]), trans)
        pd.testing.assert_frame_equal(result.drop(columns='target'), permuted.drop(columns='target'))
        separate = pd.concat([_join_berka_history(loans.iloc[[i]], trans) for i in range(3)], ignore_index=True)
        pd.testing.assert_frame_equal(result, separate)

    def test_berka_requires_history_but_old_checkpoint_contract_replays(self):
        data = SimpleNamespace(target_col='target', auxiliary_dfs={'account': pd.DataFrame({
            'account_id': [1], 'district_id': [10]})})
        main = pd.DataFrame({'account_id': [1], 'date': [930102], 'target': [0]})
        new = tabular_adapter('berka')
        self.assertEqual(new['ops'], ['JoinTable', 'JoinTable'])
        with self.assertRaisesRegex(ValueError, 'requires trans'):
            augment_splits(data, {'train': main}, new)
        old = {k: v for k, v in new.items() if k != 'berka_history_version'}
        self.assertNotIn('trans__count', augment_splits(data, {'train': main}, old)['train'])

    def test_berka_diagnostic_uses_validation_only(self):
        from scripts.diagnose_dagpipe_berka import diagnose
        n = 40
        loans = pd.DataFrame({'loan_id': range(n), 'account_id': range(n), 'date': 930102,
                              'amount': 100., 'target': [i % 2 for i in range(n)]})
        data = SimpleNamespace(target_col='target', id_col='loan_id', auxiliary_dfs={
            'account': pd.DataFrame({'account_id': range(n), 'district_id': 1}),
            'trans': pd.DataFrame({'account_id': range(n), 'date': 930101,
                                   'balance': [i % 2 * 100. for i in range(n)]})})
        splits = {'train': loans.iloc[:30].copy(), 'val': loans.iloc[30:].copy()}
        with patch('scripts.diagnose_dagpipe_berka.training_splits', return_value=splits) as split:
            report = diagnose(data, {'model_params': {'task': 'binary', 'n_estimators': 5,
                                                     'min_child_samples': 2, 'colsample_bytree': 1.,
                                                     'subsample': 1.}})
        split.assert_called_once_with(data, {'model_params': {'task': 'binary', 'n_estimators': 5,
                                      'min_child_samples': 2, 'colsample_bytree': 1., 'subsample': 1.}, 'train': {'seed': 42}}, include_std_test=False)
        self.assertEqual(report['selection_split'], 'validation_only')
        self.assertEqual(len(report['results']), 8)
        self.assertGreater(report['results']['loan_account_preloan_history']['n_features'],
                           report['results']['loan_account']['n_features'])
        self.assertIn('loan_shared_trans_join', report['results'])

    def test_training_and_checkpoint_replay(self):
        from baselines.DAGPipe.dagpipe import DAGPipe
        rng = np.random.default_rng(42)
        for regression in (False, True):
            frame = pd.DataFrame({'x': rng.normal(size=40), 'z': rng.normal(size=40)})
            frame['target'] = frame.x * 2 if regression else (frame.x > 0).astype(int)
            data = SimpleNamespace(target_col='target', id_col=None)
            executor = SimpleNamespace(
                task_type='tabular', get_model_config=lambda: {'model_params': {'task': 'regression' if regression else 'binary'}},
                load_data=lambda: data, cleanup=lambda: None)
            with tempfile.TemporaryDirectory() as directory, patch('baselines.DAGPipe.dagpipe.TrainingExecutor', return_value=executor), patch(
                'baselines.DAGPipe.dagpipe.training_splits', return_value={'train': frame.iloc[:30], 'val': frame.iloc[30:]}):
                search = DAGPipe('unused', 'synthetic', output_dir=directory, max_pipeline_len=10, verbose=False)
                search.cfg.update(epochs=2, batch_size=8, reference_rows=8)
                result = search.run()
                self.assertEqual(result['n_unique_evaluations'], 0)
                pipeline = DAGPipePipeline.from_yaml(result['best_pipeline_yaml'])
                self.assertTrue(np.isfinite(pipeline.transform(frame[['x', 'z']])).all())
                self.assertLessEqual(len(pipeline), 10)


class NYCTests(unittest.TestCase):
    def setUp(self):
        path = ROOT / 'dppbench/tasks/nyc_taxi_hourly/nyc_taxi_hourly_data.py'
        node = next(n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n, ast.ClassDef))
        namespace = {'TabularData': object, 'pd': pd, 'np': np}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
        self.data = namespace[node.name].__new__(namespace[node.name])
        self.data._sort_col = '_hour_idx'

    def test_raw_counts_do_not_depend_on_same_hour_amounts(self):
        raw = pd.DataFrame({'PULocationID': [1, 1, 2],
                            'tpep_pickup_datetime': pd.to_datetime(['2023-01-07 08:01', '2023-01-07 08:15', '2023-01-07 09:00']),
                            'trip_distance': [1., 2., 3.], 'fare_amount': [10., 20., 30.]})
        result = self.data._build_hourly_panel(raw)
        self.assertEqual(result.trip_count.tolist(), [2, 1])
        self.assertEqual(result.hour_of_day.tolist(), [8, 9])
        pd.testing.assert_frame_equal(result, self.data._build_hourly_panel(raw.assign(trip_distance=900., fare_amount=800.)))
        self.assertFalse(any(c.startswith(('fare_amount_', 'trip_distance_')) for c in result))

    def test_old_frozen_rows_labels_and_membership_preserved(self):
        old = pd.DataFrame({'PULocationID': [1, 2], '_hour_ts': pd.to_datetime(['2023-01-07', '2023-01-08']),
                            'trip_count': [2, 9], '__split__': ['train', 'std_test'],
                            'fare_amount_sum': [20., 90.], 'passenger_count_mean': [1., 2.]})
        result = self.data._normalize_frozen_frame(old)
        pd.testing.assert_frame_equal(old[['PULocationID', 'trip_count', '__split__']], result[['PULocationID', 'trip_count', '__split__']])
        self.assertNotIn('fare_amount_sum', result)
        self.assertNotIn('passenger_count_mean', result)
        pd.testing.assert_frame_equal(result, self.data._normalize_frozen_frame(result))


if __name__ == '__main__':
    unittest.main()
