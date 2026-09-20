"""Offline contract tests with real task loader/executor/LightGBM and HPO.

Synthetic fixture has the bike task's file/schema protocol; it is not a
measurement on the UCI Bike Sharing dataset.
"""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from baselines.common.budget import Budget, budget_scope, BudgetExceededError
from baselines.DagPipe.dagpipe import DagPipe
from baselines.DagPipe.pipeline import DagPipePipeline
from baselines.DagPipe.evaluator import DagPipeEvaluator
from scripts.final_model_hpo import tune_final_model

ROOT = Path(__file__).resolve().parents[2]


def make_fixture(root):
    rng = np.random.default_rng(123)
    data_dir = Path(root)/'bike_sharing'/'data'
    std_dir = data_dir.parent/'std_test'
    data_dir.mkdir(parents=True)
    std_dir.mkdir()
    n = 100
    frame = pd.DataFrame({'instant': np.arange(n), '_sort_idx': np.arange(n),
                          'temp': rng.normal(size=n), 'humidity': rng.uniform(size=n),
                          'weather': np.tile(['sun', 'rain'], n//2)})
    frame['cnt'] = 20 + 5*frame.temp + rng.normal(size=n)
    frame.to_csv(data_dir/'hour.csv', index=False)
    frame.head(10).to_csv(data_dir/'day.csv', index=False)
    frame.iloc[:80].to_parquet(std_dir/'train_frozen.parquet', index=False)
    frame.iloc[80:].to_parquet(std_dir/'std_test.parquet', index=False)
    return data_dir, std_dir


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.tmp = tempfile.TemporaryDirectory(prefix='dagpipe_contract_')
        cls.data_dir, cls.std_dir = make_fixture(cls.tmp.name)
        cls.kwargs = dict(task_dir=str(ROOT/'dppbench/tasks/bike_sharing'), data_name='bike_sharing',
                          data_dir=str(cls.data_dir), seed=42, model_name='LightGBM', device='cpu', verbose=False)
        cls.budget = Budget(max_evaluations=2, max_pipeline_length=10)
        original = pd.read_parquet
        def no_test(path, *args, **kwargs):
            if Path(path).name == 'std_test.parquet':
                raise AssertionError('Search read standard test!')
            return original(path, *args, **kwargs)
        with patch.object(pd, 'read_parquet', side_effect=no_test), budget_scope(cls.budget):
            cls.runner = DagPipe(**cls.kwargs, max_pipeline_len=10, output_dir=Path(cls.tmp.name)/'run')
            cls.result = cls.runner.run()
        cls.pipeline = cls.runner.get_pipeline()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_budget_and_serialization(self):
        self.assertEqual(self.budget.evaluations_used, 2)
        self.assertEqual(self.result['n_unique_evaluations'], 2)
        self.assertEqual(len(self.pipeline), 10)
        reloaded = DagPipePipeline.from_yaml(self.result['best_pipeline_yaml'])
        frame = pd.read_csv(self.data_dir/'hour.csv').head(3)
        np.testing.assert_array_equal(self.pipeline.transform(frame), reloaded.transform(frame))

    def test_query_independence(self):
        frame = pd.read_csv(self.data_dir/'hour.csv').head(4)
        batch = self.pipeline.transform(frame)
        rows = np.vstack([self.pipeline.transform(frame.iloc[[i]]) for i in range(len(frame))])
        np.testing.assert_array_equal(batch, rows)
        poisoned = frame.copy()
        poisoned.loc[1:, 'temp'] = 1e4
        np.testing.assert_array_equal(batch[0], self.pipeline.transform(poisoned)[0])

    def test_seed_reproducibility(self):
        with budget_scope(Budget(max_evaluations=2)):
            again = DagPipe(**self.kwargs, max_pipeline_len=10, output_dir=Path(self.tmp.name)/'repeat').run()
        self.assertEqual(again['best_pipeline_yaml'], self.result['best_pipeline_yaml'])

    def test_hpo_never_reads_test_and_reuses_frozen_pipeline(self):
        original = pd.read_parquet
        def no_test(path, *args, **kwargs):
            if Path(path).name == 'std_test.parquet':
                raise AssertionError('HPO read standard test!')
            return original(path, *args, **kwargs)
        before = self.pipeline.hash()
        with patch.object(pd, 'read_parquet', side_effect=no_test), contextlib.redirect_stdout(io.StringIO()):
            evaluator, summary = tune_final_model(self.pipeline, output_dir=self.tmp.name,
                                                 evaluator_class=DagPipeEvaluator, **self.kwargs)
        self.assertEqual(len(summary['trials']), 9)
        self.assertTrue(all(t['success'] for t in summary['trials']))
        self.assertEqual(self.pipeline.hash(), before)
        try:
            result = evaluator.evaluate(self.pipeline)
            self.assertTrue(result.success, result.error)
            self.assertEqual(result.metrics['std_test_n_rows'], 20)
            self.assertTrue(np.isfinite(result.metrics['std_test_rmse']))
        finally:
            evaluator._executor.cleanup()

    def test_resource_failure_not_swallowed(self):
        with budget_scope(Budget(max_memory_mb=.0001)):
            with self.assertRaises(BudgetExceededError):
                self.pipeline.transform(pd.read_csv(self.data_dir/'hour.csv').head(2))

    def test_unsupported_task(self):
        kwargs = dict(self.kwargs, task_dir=str(ROOT/'dppbench/tasks/movielens'), data_name='movielens', model_name=None)
        with self.assertRaisesRegex(ValueError, 'unsupported_task_type=rec'):
            DagPipe(**kwargs).run()

    def test_too_short_budget(self):
        with self.assertRaisesRegex(ValueError, 'max_pipeline_len'):
            DagPipe(**self.kwargs, max_pipeline_len=1)


if __name__ == '__main__':
    unittest.main()
