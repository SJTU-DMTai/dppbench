import contextlib
import io
import pathlib
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from dppbench.operators.integration.join_table import JoinTable
from scripts import build_std_test, evaluate_baseline


class FailureRegressionTest(unittest.TestCase):
    def test_graph_builder_uses_primary_labels_not_public_relation(self):
        frame = pd.DataFrame({'txId': range(50), 'class': [0, 1] * 20 + [-1] * 10, 'feature': range(50)})
        data = SimpleNamespace(train_df=frame, target_col='class',
                               auxiliary_dfs={'classes': frame[['txId']].copy()})
        with tempfile.TemporaryDirectory() as temp:
            meta = build_std_test.build_graph('elliptic_bitcoin', data, temp, False)
            train = pd.read_parquet(pathlib.Path(temp) / 'train_frozen.parquet')
            test = pd.read_parquet(pathlib.Path(temp) / 'std_test.parquet')
            self.assertEqual(meta['std_test_rows'], 8)
            self.assertTrue(test['class'].isin([0, 1]).all())
            self.assertEqual(int(train['class'].eq(-1).sum()), 10)
            self.assertFalse(set(train.txId) & set(test.txId))
            self.assertNotIn('class', data.auxiliary_dfs['classes'])
            pd.testing.assert_frame_equal(data.train_df, frame)

    def test_graph_builder_preserves_original_class_order(self):
        import numpy as np
        classes = pd.DataFrame({'txId': list(reversed(range(50))), 'class': ['1', '2'] * 20 + ['unknown'] * 10})
        normalized = classes.assign(**{'class': classes['class'].map({'1': 1, '2': 0, 'unknown': -1})})
        data = SimpleNamespace(train_df=normalized.sort_values('txId').reset_index(drop=True),
                               target_col='class', _class_labels=classes,
                               auxiliary_dfs={'classes': classes[['txId']]})
        labeled = classes[classes['class'].isin(['1', '2'])]
        expected = set(labeled.iloc[np.random.default_rng(42).permutation(len(labeled))[:8]].txId)
        with tempfile.TemporaryDirectory() as temp:
            build_std_test.build_graph('elliptic_bitcoin', data, temp, False)
            test = pd.read_parquet(pathlib.Path(temp) / 'std_test.parquet')
            self.assertEqual(set(test.txId), expected)
            self.assertTrue(test['class'].isin([0, 1]).all())
        data._class_labels = classes.iloc[1:]
        with self.assertRaisesRegex(ValueError, 'do not match'):
            build_std_test.build_graph('elliptic_bitcoin', data, '', True)

    def test_restrict_prepared_rec_search_keeps_partition_and_history(self):
        import numpy as np
        from dppbench.dataset import RecData
        from baselines.common.executor import TrainingExecutor
        with tempfile.TemporaryDirectory() as temp:
            directory = pathlib.Path(temp)
            (directory / 'data').mkdir()
            (directory / 'std_test').mkdir()
            data = RecData('unit')
            data.data_dir = str(directory / 'data')
            data.model_cfg = {'feature': {'target_col': 'rating', 'time_col': 'timestamp'},
                              'train': {'val_ratio': 0.25, 'seed': 42}}
            frame = pd.DataFrame({'user_id': [1, 2] * 10, 'item_id': [1, 2] * 10,
                                  'timestamp': range(20), 'rating': [0, 1] * 10})
            frame.to_parquet(directory / 'std_test/interaction_frozen.parquet', index=False)
            frame.iloc[:2].assign(timestamp=[100, 101]).to_parquet(directory / 'std_test/std_test.parquet', index=False)
            data.interaction_df = frame.copy()
            data._rec_training_only = True
            data._rec_frozen_train_only = True
            executor = TrainingExecutor.__new__(TrainingExecutor)
            executor._restrict_rec_training(data)
            original = data.interaction_df.copy(deep=True)
            reference = data._rec_history_reference
            membership = data._rec_partition_state['membership'].copy()
            executor._restrict_rec_training(data)
            pd.testing.assert_frame_equal(original, data.interaction_df)
            np.testing.assert_array_equal(membership, data._rec_partition_state['membership'])
            self.assertIs(reference, data._rec_history_reference)
            data.interaction_df.loc[0, '__split__'] = 'std_test'
            with self.assertRaisesRegex(ValueError, 'only fixed train/val'):
                executor._restrict_rec_training(data)

    def test_repeated_aggregate_joins_keep_unique_columns_and_rows(self):
        main = pd.DataFrame({'SK_ID_CURR': [1, 2], 'SK_ID_PREV_mean_x': [10, 20],
                             'SK_ID_PREV_mean_y': [30, 40]})
        auxiliary = pd.DataFrame({'SK_ID_CURR': [1, 1, 2], 'SK_ID_PREV': [5, 7, 9]})
        op = JoinTable(method='agg', aux_df=auxiliary, key_col='SK_ID_CURR')
        result = main
        for _ in range(5):
            result = op.transform(result)
        self.assertEqual(len(result), len(main))
        self.assertTrue(result.columns.is_unique)
        self.assertEqual(result['SK_ID_PREV_mean_x'].tolist(), [10, 20])
        self.assertEqual(result['SK_ID_PREV_mean'].tolist(), [6, 9])
        self.assertEqual(result['SK_ID_PREV_mean__join5'].tolist(), [6, 9])
        replay = main
        for _ in range(5):
            replay = JoinTable(method='agg', aux_df=auxiliary.copy(), key_col='SK_ID_CURR').transform(replay)
        pd.testing.assert_frame_equal(result, replay)

    def test_brand_sequence_uses_full_training_history_and_fitted_imputation(self):
        from dppbench.dataset import RecData
        from dppbench.operators.feature_engineering.create_sequence import CreateSequence
        data = RecData('unit')
        data.model_cfg = {'feature': {'time_col': 'timestamp', 'target_col': 'rating'}}
        data.col_types = {'user_id': 'categorical', 'item_id': 'categorical', 'item_brand': 'categorical'}
        reference = pd.DataFrame({'user_id': [1, 1, 1], 'item_id': [1, 2, 2],
                                  'timestamp': [1, 2, 2], '__split__': ['train'] * 3})
        data._rec_history_reference = reference.copy()
        data.item_df = pd.DataFrame({'item_id': [1, 2, 3], 'item_brand': ['A', None, 'C']})
        query = pd.DataFrame({'user_id': [1, 1], 'item_id': [2, 3], 'item_brand': [None, 'C'],
                              'timestamp': [2, 3], 'rating': [1, 0], '__split__': ['train', 'val']})
        fill = data._load_op('HandleMV')(cols=['item_brand'], method='constant', fill_value=0)
        processed = data._transform_rec_dag_df(query, fill, 'HandleMV')
        # An intermediate operator must not lose the fitted history transformation.
        processed = data._transform_rec_dag_df(processed, data._load_op('FilterSample')(subset=['rating']), 'FilterSample')
        sequence = CreateSequence(seq_col='cate_id_seq', item_col='item_brand', max_len=20)
        result = data._transform_rec_dag_df(processed, sequence, 'CreateSequence')
        self.assertEqual(result['cate_id_seq'].tolist(), [['A'], ['A', 0, 0]])
        pd.testing.assert_frame_equal(data._rec_history_reference, reference)
        self.assertEqual(data.col_types['cate_id_seq'], 'categorical_list')
        data.item_df = pd.concat([data.item_df, data.item_df.iloc[:1]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, 'unique nonmissing'):
            data._transform_rec_dag_df(processed, sequence, 'CreateSequence')
        data._rec_history_reference.loc[0, '__split__'] = 'val'
        data.item_df = data.item_df.drop_duplicates('item_id')
        with self.assertRaisesRegex(ValueError, 'training events only'):
            data._transform_rec_dag_df(processed, sequence, 'CreateSequence')

    def test_key_join_still_rejects_nonunique_keys(self):
        op = JoinTable(method='key', aux_df=pd.DataFrame({'id': [1, 1], 'x': [2, 3]}), key_col='id')
        with self.assertRaises(pd.errors.MergeError):
            op.transform(pd.DataFrame({'id': [1]}))

    def test_builder_returns_failure_status(self):
        args = SimpleNamespace(gpu_id=-1, data_names='unit', dry_run=False, data_dir=None)
        with patch.object(build_std_test, 'parse_args', return_value=args), \
             patch.object(build_std_test, 'run_for_task', side_effect=ValueError('bad input')), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(build_std_test.main(), 1)
        with patch.object(build_std_test, 'parse_args', return_value=args), \
             patch.object(build_std_test, 'run_for_task', return_value={'std_test_rows': 10}), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(build_std_test.main(), 0)

    def test_readonly_preflight_collects_all_errors(self):
        args = SimpleNamespace(baseline='BAT', gpu_id=-1, data_names='amazon_beauty,kuairec',
                               preflight_only=True, data_dir=None, skip_build=True, quiet=True)
        with patch.object(evaluate_baseline, 'parse_args', return_value=args), \
             patch.object(evaluate_baseline, '_ensure_std_test', side_effect=FileNotFoundError('missing')) as ensure, \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit):
                evaluate_baseline.main()
            self.assertEqual(ensure.call_count, 2)


if __name__ == '__main__':
    unittest.main()
