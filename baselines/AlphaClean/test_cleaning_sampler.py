"""AlphaClean must preserve conditional cleaning parameter contracts."""
import unittest

from baselines.AlphaClean.parameter_sampler import ParameterSampler
from baselines.common.data_ctx import DataContext, TableContext
from baselines.common.pipeline import validate_step_params
from dppbench.operators.cleaning.handle_error import HandleError
from dppbench.operators.cleaning.handle_outlier import HandleOutlier


class CleaningSamplerTests(unittest.TestCase):
    def test_cleaning_samples_are_valid_and_include_new_repairs(self):
        ctx = DataContext('tabular', 'fixture', 'main', tables={
            'main': TableContext('main', 'primary', dtypes={'x': 'float64', 'y': 'int64'})})
        seen = {'HandleError': set(), 'HandleOutlier': set()}
        for seed in range(100):
            sampler = ParameterSampler(ctx, seed=seed)
            for name, cls in [('HandleError', HandleError), ('HandleOutlier', HandleOutlier)]:
                repair = sampler.sample_for_op(name)
                self.assertIsNotNone(repair)
                self.assertFalse(validate_step_params(name, repair.params, ctx))
                cls(**repair.params)
                if repair.params['action'] == 'repair':
                    seen[name].add(repair.params['repair_method'])
                if name == 'HandleError':
                    self.assertEqual(repair.params['rule'], 'numeric')
        self.assertTrue({'median', 'mode', 'fill_constant'} <= seen['HandleError'])
        self.assertIn('median', seen['HandleOutlier'])


if __name__ == '__main__':
    unittest.main()
