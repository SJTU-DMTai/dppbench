"""Offline reproductions of failures in exp1_1_nollm.log."""
from pathlib import Path
import random
import tempfile
import unittest
import warnings

import pandas as pd

from baselines.common.data_ctx import infer_data_context
from baselines.common.pipeline import build_default_params, get_param_space
from dppbench.tasks.bike_sharing.bike_sharing_data import BikeSharingData
from dppbench.tasks.elliptic_bitcoin.elliptic_bitcoin_data import EllipticBitcoinData
from dppbench.operators.feature_engineering.extract_text_feature import ExtractTextFeature


class LogRegressions(unittest.TestCase):
    def test_bike_sources_and_legacy_frozen_frames_exclude_outcomes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            hour = pd.DataFrame({'instant': [1, 2], 'dteday': ['2011-01-01'] * 2,
                                 'cnt': [5, 7], 'casual': [2, 3], 'registered': [3, 4]})
            hour.to_csv(root / 'hour.csv', index=False)
            pd.DataFrame({'instant': [99], 'dteday': ['2011-01-01'], 'cnt': [12],
                          'casual': [5], 'registered': [7], 'weather': [2]}).to_csv(root / 'day.csv', index=False)
            data = BikeSharingData(folder)
            data.load_data()
            self.assertFalse({'casual', 'registered'} & set(data.train_df))
            self.assertFalse({'cnt', 'casual', 'registered', 'instant'} & set(data.auxiliary_dfs['day']))
            frozen = data._normalize_frozen_frame(hour)
            self.assertEqual(frozen.cnt.tolist(), [5, 7])
            self.assertFalse({'casual', 'registered'} & set(frozen))
            ctx = infer_data_context('bike_sharing', data, 'tabular')
            params = build_default_params('JoinTable', ctx, random.Random(1))
            self.assertEqual(params['key_col'], 'dteday')
            params['aux_df'] = data.auxiliary_dfs['day']
            op = data._load_op('JoinTable')(**params)
            result = data._transform_tabular_dag_df(data.train_df, op, 'JoinTable')
            self.assertEqual(result.cnt.tolist(), [5, 7])
            self.assertEqual(result.weather.tolist(), [2, 2])

    def test_elliptic_labels_private_but_frozen_restoration_and_join_work(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            pd.DataFrame([[1, 1, .1], [2, 2, .2], [3, 40, .3]]).to_csv(
                root / 'elliptic_txs_features.csv', index=False, header=False)
            pd.DataFrame({'txId': [1, 2, 3], 'class': ['1', '2', 'unknown']}).to_csv(
                root / 'elliptic_txs_classes.csv', index=False)
            pd.DataFrame({'txId1': [1], 'txId2': [2]}).to_csv(root / 'elliptic_txs_edgelist.csv', index=False)
            data = EllipticBitcoinData(folder)
            data.load_data()
            self.assertEqual(data.train_df['class'].tolist(), [1, 0, -1])
            self.assertNotIn('class', data.auxiliary_dfs['classes'])
            restored = data._normalize_frozen_frame(data.train_df.drop(columns='class'))
            self.assertEqual(restored['class'].tolist(), [1, 0, -1])
            op = data._load_op('JoinTable')(aux_df=data.auxiliary_dfs['classes'], key_col='txId')
            result = data._transform_tabular_dag_df(data.train_df, op, 'JoinTable')
            pd.testing.assert_frame_equal(result, data.train_df)
            # Explicitly feeding labels remains forbidden, not silently allowed.
            op.aux_df = data._class_labels
            with self.assertRaisesRegex(ValueError, 'supervised target'):
                data._transform_tabular_dag_df(data.train_df, op, 'JoinTable')

    def test_imputer_tuning_preserves_logical_action(self):
        from baselines.common.data_ctx import DataContext
        ctx = DataContext('tabular', 'fixture', 'main')
        for action in ('impute', 'delete', 'mark'):
            space = get_param_space('HandleMV', ctx, {'action': action})
            self.assertEqual(space['action'], [action])
            self.assertEqual('method' in space, action == 'impute')

    def test_text_features_are_batched_and_vocabulary_is_reused(self):
        frame = pd.DataFrame({'text': [' '.join(f'word{i}' for i in range(180)), 'word1 word2']}, index=[7, 7])
        op = ExtractTextFeature(['text'], method='bow', max_features=180)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', pd.errors.PerformanceWarning)
            result = op.transform(frame)
        self.assertEqual(result.shape, (2, 181))
        self.assertEqual(result['text_bow_word1'].tolist(), [1, 1])
        self.assertFalse(any(issubclass(w.category, pd.errors.PerformanceWarning) for w in caught))
        held = op.transform(pd.DataFrame({'text': ['unseenword']}))
        self.assertEqual(held.columns.tolist(), result.columns.tolist())
        self.assertEqual(held.filter(like='text_bow_').sum().sum(), 0)
        pd.testing.assert_frame_equal(op.transform(result), result)


if __name__ == '__main__':
    unittest.main()
