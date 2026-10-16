"""Cleaning implementations, conditional search contracts and split isolation."""
import random
import unittest

import numpy as np
import pandas as pd

from dppbench.dataset import RecData, TabularData
from dppbench.operators.cleaning.handle_error import HandleError
from dppbench.operators.cleaning.handle_outlier import HandleOutlier
from baselines.common.data_ctx import DataContext, TableContext
from baselines.common.op_spec import CATALOG
from baselines.common.pipeline import (
    build_default_params, sample_handle_error_params, validate_step_params,
)


class CleaningRepairs(unittest.TestCase):
    def test_outlier_median_uses_training_inliers_and_reuses_state(self):
        op = HandleOutlier(cols=['x'], action='repair', repair_method='median', threshold=1.5)
        frame = pd.DataFrame({'x': [1., 2., 3., 4., 100.]})
        result = op.transform(frame)
        self.assertEqual(result.x.tolist(), [1., 2., 3., 4., 2.5])
        self.assertEqual(op.transform(pd.DataFrame({'x': [200.]})).x.iloc[0], 2.5)
        self.assertEqual(frame.x.iloc[-1], 100.)

    def test_outlier_all_missing_is_preserved(self):
        op = HandleOutlier(cols=['x'], action='repair', repair_method='median')
        self.assertTrue(op.transform(pd.DataFrame({'x': [np.nan, np.nan]})).x.isna().all())

    def test_regex_full_match_and_missing(self):
        frame = pd.DataFrame({'code': ['AB12', 'xAB12', None, 'AB12x']})
        op = HandleError(cols=['code'], rule='regex', pattern=r'[A-Z]{2}\d{2}')
        result = op.transform(frame)
        self.assertEqual(result.code.iloc[0], 'AB12')
        self.assertTrue(pd.isna(result.code.iloc[1]))
        op = HandleError(cols=['code'], rule='regex', pattern=r'[A-Z]{2}\d{2}',
                         action='repair', repair_method='fill_constant', fill_value='ZZ00')
        result = op.transform(frame)
        self.assertEqual(result.code.iloc[[0, 1, 3]].tolist(), ['AB12', 'ZZ00', 'ZZ00'])
        self.assertTrue(pd.isna(result.code.iloc[2]))

    def test_numeric_median_and_categorical_mode(self):
        op = HandleError(cols=['x'], rule='numeric', action='repair', repair_method='median')
        self.assertEqual(op.transform(pd.DataFrame({'x': ['1', '3', 'bad']})).x.iloc[-1], 2.)
        self.assertEqual(op.transform(pd.DataFrame({'x': ['999', 'bad']})).x.iloc[-1], 2.)
        op = HandleError(cols=['x'], rule='not_in', allowed_values=['A', 'B'],
                         action='repair', repair_method='mode')
        self.assertEqual(op.transform(pd.DataFrame({'x': ['A', 'A', 'B', 'bad']})).x.tolist(),
                         ['A', 'A', 'B', 'A'])

    def test_no_valid_values_fail_instead_of_inventing_statistics(self):
        for method in ('median', 'mode'):
            op = HandleError(cols=['x'], rule='positive', action='repair', repair_method=method)
            with self.assertRaisesRegex(ValueError, 'No valid training values'):
                op.transform(pd.DataFrame({'x': [-1, 0]}))

    def test_nullable_and_categorical_assignment(self):
        op = HandleError(cols=['x'], rule='positive', action='repair', repair_method='median')
        frame = pd.DataFrame({'x': pd.Series([1, 2, -1], dtype='Int64')})
        self.assertEqual(op.transform(frame).x.iloc[-1], 1.5)
        op = HandleError(cols=['x'], rule='not_in', allowed_values=['A', 'unknown'],
                         action='repair', repair_method='fill_constant', fill_value='unknown')
        self.assertEqual(op.transform(pd.DataFrame({'x': pd.Categorical(['A', 'bad'])})).x.iloc[-1], 'unknown')

    def test_clip_parses_numeric_and_date_values_only_at_flagged_positions(self):
        op = HandleError(cols=['x'], rule='in_range', min_value=0, max_value=10,
                         action='repair', repair_method='clip')
        frame = pd.DataFrame({'x': ['-2', '5', '12', None]})
        result = op.transform(frame)
        self.assertEqual(result.x.iloc[:3].tolist(), [0., '5', 10.])
        self.assertTrue(pd.isna(result.x.iloc[3]))
        with self.assertRaisesRegex(ValueError, 'unparseable'):
            op.transform(pd.DataFrame({'x': ['bad']}))
        op = HandleError(rules=[{'col': 'd', 'min': '2020-01-01', 'max': '2020-12-31',
                                'action': 'repair', 'repair_method': 'clip'}])
        result = op.transform(pd.DataFrame({'d': ['2019-01-01', '2021-01-01']}))
        self.assertEqual(result.d.tolist(), [pd.Timestamp('2020-01-01'), pd.Timestamp('2020-12-31')])

    def test_constructor_and_pipeline_reject_invalid_combinations(self):
        cases = [
            {'rule': 'regex'}, {'rule': 'regex', 'pattern': '['},
            {'rule': 'in_range'}, {'rule': 'not_in'},
            {'rule': 'in_range', 'min_value': 5, 'max_value': 1},
            {'rule': 'numeric', 'action': 'repair', 'repair_method': 'clip'},
            {'rule': 'regex', 'pattern': 'A+', 'action': 'repair', 'repair_method': 'median'},
            {'rule': 'numeric', 'action': 'repair', 'repair_method': 'fill_constant'},
            {'rule': 'positive', 'action': 'repair', 'repair_method': 'fill_constant', 'fill_value': 0},
            {'rules': [{'col': 'x', 'rule': 'bogus'}]},
        ]
        for params in cases:
            with self.subTest(params=params):
                params = {'cols': ['x'], **params}
                with self.assertRaises(ValueError):
                    HandleError(**params)
                self.assertTrue(validate_step_params('HandleError', params))

    def test_per_column_overrides_and_cached_statistics(self):
        op = HandleError(action='repair', repair_method='mode', rules=[
            {'col': 'x', 'rule': 'positive', 'repair_method': 'median'},
            {'col': 'c', 'rule': 'regex', 'pattern': '[AB]'},
        ])
        result = op.transform(pd.DataFrame({'x': [2, 4, -1], 'c': ['A', 'A', 'bad']}))
        self.assertEqual(result.x.tolist(), [2, 4, 3])
        self.assertEqual(result.c.tolist(), ['A', 'A', 'A'])
        result = op.transform(pd.DataFrame({'x': [-1], 'c': ['bad']}))
        self.assertEqual(result.iloc[0].tolist(), [3, 'A'])

    def test_held_out_rows_unchanged_for_both_dataset_protocols(self):
        frame = pd.DataFrame({'x': [1., 3., -1., 100., -9.],
                              '__split__': ['train', 'train', 'train', 'val', 'std_test']})
        for data, transform in [(TabularData('test'), '_transform_tabular_dag_df'),
                                (RecData('test'), '_transform_rec_dag_df')]:
            op = HandleError(cols=['x'], rule='positive', action='repair', repair_method='median')
            result = getattr(data, transform)(frame, op, 'HandleError')
            self.assertEqual(result.x.tolist(), [1., 3., 2., 100., -9.])
            self.assertEqual(op.fill_values_[0], 2.)

    def test_search_generation_and_conditional_tuning(self):
        ctx = DataContext('tabular', 'test', 'main', tables={
            'main': TableContext('main', 'primary', dtypes={'x': 'float64', 'y': 'int64'})})
        methods = set()
        outlier_methods = set()
        for seed in range(80):
            rng = random.Random(seed)
            params = build_default_params('HandleError', ctx, rng)
            self.assertFalse(validate_step_params('HandleError', params, ctx))
            methods.add(params.get('repair_method'))
            params = sample_handle_error_params(params, rng)
            self.assertEqual(params['rule'], 'numeric')
            self.assertFalse(validate_step_params('HandleError', params, ctx))
            params = build_default_params('HandleOutlier', ctx, rng)
            outlier_methods.add(params['repair_method'])
            self.assertFalse(validate_step_params('HandleOutlier', params, ctx))
            HandleOutlier(**params)
        self.assertTrue({'median', 'mode', 'fill_constant', 'set_missing'} <= methods)
        self.assertEqual(outlier_methods, set(CATALOG['HandleOutlier'].param_space['repair_method']))
        params = {'rules': [{'col': 'x', 'rule': 'regex', 'pattern': '[AB]'}]}
        for seed in range(25):
            tuned = sample_handle_error_params(params, random.Random(seed))
            self.assertEqual(tuned['rules'][0]['pattern'], '[AB]')
            self.assertFalse(validate_step_params('HandleError', tuned))

    def test_physical_search_keeps_regex_contract(self):
        from baselines.SAGA.physical_search import PhysicalSearch
        from baselines.common.evaluator import EvaluationResult
        from baselines.common.pipeline import Pipeline, PipelineStep
        from unittest.mock import Mock, patch
        from types import SimpleNamespace

        evaluator = Mock(ctx=None)
        evaluator.evaluate.return_value = EvaluationResult(
            fitness=1., success=True, metrics={}, duration_seconds=0.)
        base = Pipeline(steps=[PipelineStep(op='HandleError', target='train', params={
            'cols': ['code'], 'rule': 'regex', 'pattern': '[AB]',
        })])
        # Isolate parameter tuning from mandatory whole-pipeline tail checks.
        with patch('baselines.SAGA.physical_search.check_pipeline',
                   return_value=SimpleNamespace(valid=True)):
            PhysicalSearch(evaluator, n_trials=20, verbose=False).tune_one(base, 0., {})
        self.assertEqual(evaluator.evaluate.call_count, 20)
        for call in evaluator.evaluate.call_args_list:
            params = call.args[0].steps[0].params
            self.assertEqual(params['pattern'], '[AB]')
            self.assertEqual(params['rule'], 'regex')
            self.assertFalse(validate_step_params('HandleError', params))


if __name__ == '__main__':
    unittest.main()
