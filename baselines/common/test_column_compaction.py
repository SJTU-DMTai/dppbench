import copy
import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from baselines.common.pipeline import Pipeline, PipelineStep
from baselines.common.column_compaction import compact_column_steps
from dppbench.operators.cleaning.handle_m_v import HandleMV


class ColumnCompactionTests(unittest.TestCase):
    def step(self, cols, **params):
        return PipelineStep('HandleMV', params=dict(cols=cols, **params))

    def test_40_nodes_become_three_without_dropping_columns(self):
        p = Pipeline([PipelineStep('JoinTable')] +
                     [self.step([f'x{i}']) for i in range(38)] +
                     [PipelineStep('CreateSequence')])
        self.assertEqual(len(p), 3)
        self.assertEqual(p.steps[1].params['cols'], [f'x{i}' for i in range(38)])
        self.assertEqual(len(Pipeline.from_yaml(p.to_yaml())), 3)

    def test_train_and_heldout_values_and_dtypes(self):
        train = pd.DataFrame({'a': [1., None, 3.], 'b': [None, 2., 4.],
                              'category': ['x', None, 'y'], 'untouched': [7, 8, 9]})
        heldout = pd.DataFrame({'a': [None, 99.], 'b': [100., None],
                                'category': [None, 'unseen'], 'untouched': [4, 5]})
        for method in ('median', 'mean', 'mode', 'constant'):
            with self.subTest(method=method):
                raw = [self.step([c], method=method) for c in ('a', 'b', 'category')]
                merged = compact_column_steps(Pipeline(copy.deepcopy(raw))).steps
                def replay(steps):
                    tr, va = train.copy(), heldout.copy()
                    for step in steps:
                        op = HandleMV(**step.params)
                        tr = op.transform(tr)
                        va = op.transform(va)
                    return tr, va
                for before, after in zip(replay(raw), replay(merged)):
                    assert_frame_equal(before, after)

    def test_non_equivalent_steps_stay_separate(self):
        for steps in ([self.step(['a'], method='knn'), self.step(['b'], method='knn')],
                      [self.step(['a']), self.step(['a'])],
                      [self.step(['a']), self.step(['b'], method='mean')],
                      [self.step(['a']), PipelineStep('ScaleFeature'), self.step(['b'])]):
            self.assertEqual(len(Pipeline(steps)), len(steps))

    def test_dag_fork_preserved_and_chain_rewired(self):
        a, b = self.step(['a']), self.step(['b'])
        a.id, a.prev = 'a', ['s0']
        b.id, b.prev = 'b', ['a']
        p = Pipeline([a, b], train_prev=['b'])
        self.assertEqual(len(p), 1)
        self.assertEqual(p.steps[0].id, 'b')
        self.assertEqual(p.steps[0].prev, ['s0'])
        fork = Pipeline([a, b], train_prev=['a', 'b'])
        self.assertEqual(len(fork), 2)


if __name__ == '__main__':
    unittest.main()
