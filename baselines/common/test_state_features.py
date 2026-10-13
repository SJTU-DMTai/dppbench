"""CPU regressions for sequence-valued search-state columns."""
import copy
import unittest

import numpy as np
import pandas as pd

from .state_features import dataframe_state_features, unique_value_count


class StateFeatureTest(unittest.TestCase):
    def test_sequence_counts_preserve_order_and_missing_values(self):
        values = [[1, 2], [1, 2], [2, 1], [], None]
        original = copy.deepcopy(values)
        frame = pd.DataFrame({'sequence': values, 'numeric': [1., 2., 3., 4., 5.]})
        features = dataframe_state_features(frame)
        self.assertEqual(unique_value_count(frame['sequence']), 3)
        self.assertEqual(features.shape, (3300,))
        self.assertTrue(np.isfinite(features).all())
        self.assertAlmostEqual(float(features[3]), 0.2)
        self.assertAlmostEqual(float(features[7]), 0.6)
        self.assertEqual(frame['sequence'].tolist(), original)

    def test_nested_containers_and_arrays(self):
        series = pd.Series([[{'x': [1, 2]}], [{'x': [1, 2]}], [{'x': [2, 1]}]])
        self.assertEqual(unique_value_count(series), 2)
        arrays = pd.Series([np.array([1, 2]), np.array([1, 2]), np.array([2, 1])])
        self.assertEqual(unique_value_count(arrays), 2)

    def test_scalar_counts_unchanged(self):
        for values in ([1, 1, 2, None], ['a', 'a', 'b', None], [None, None]):
            series = pd.Series(values)
            self.assertEqual(unique_value_count(series), series.nunique())


if __name__ == '__main__':
    unittest.main()
