import unittest
from unittest.mock import Mock
from types import SimpleNamespace

from baselines.common.budget import Budget, BudgetExceededError
from baselines.common.evaluator import PipelineEvaluator
from baselines.common.pipeline import Pipeline


class SearchBudgetTest(unittest.TestCase):
    def evaluator(self, limit):
        evaluator = PipelineEvaluator.__new__(PipelineEvaluator)
        evaluator.budget = Budget(max_evaluations=limit)
        evaluator.scoring_method = 'full_train'
        evaluator.metric_key = 'auc'
        evaluator.verbose = False
        evaluator._cache = {}
        evaluator.evaluation_count = 0
        evaluator.best_pipeline = evaluator.best_result = None
        evaluator._get_data_context = Mock(return_value=None)
        evaluator._check_pipeline = Mock(return_value=SimpleNamespace(valid=True, errors=[]))
        evaluator._executor = Mock()
        evaluator._executor._fast_mode = False
        evaluator._executor.get_model_config.return_value = {'train': {'metrics': ['auc']}}
        evaluator._run_scoring_method = Mock(return_value={'success': True, 'metrics': {'auc': 0.7}})
        return evaluator

    def test_attempts_cache_and_budget_rejections(self):
        evaluator = self.evaluator(1)
        pipeline = Pipeline([])
        result = evaluator.evaluate(pipeline)
        self.assertTrue(result.success)
        self.assertEqual(evaluator.evaluate(pipeline), result)
        rejected = evaluator.evaluate(pipeline, scoring_method='static_quality')
        self.assertFalse(rejected.success)
        self.assertEqual(evaluator.n_unique_evaluations, 1)
        self.assertEqual(len(evaluator._cache), 1)
        self.assertTrue(evaluator.evaluation_budget_exhausted)
        self.assertEqual(evaluator.best_result, result)
        self.assertIsNot(evaluator.best_pipeline, pipeline)
        with self.assertRaises(BudgetExceededError):
            evaluator.budget.check_evaluation_available()
        self.assertEqual(evaluator.budget.evaluations_used, 1)

    def test_failed_admitted_attempt_counts_but_is_not_best(self):
        evaluator = self.evaluator(2)
        evaluator._run_scoring_method.return_value = {'success': False, 'error': 'training failed'}
        self.assertFalse(evaluator.evaluate(Pipeline([])).success)
        self.assertEqual(evaluator.n_unique_evaluations, 1)
        self.assertIsNone(evaluator.best_pipeline)
        evaluator._cache.clear()
        self.assertEqual(evaluator.n_unique_evaluations, 1)

    def test_ctxpipe_does_not_score_after_limit(self):
        from baselines.CtxPipe.trainer import CtxPipeTrainer
        trainer = CtxPipeTrainer.__new__(CtxPipeTrainer)
        evaluator = self.evaluator(1)
        evaluator.evaluate(Pipeline([]))
        trainer.env = SimpleNamespace(evaluator=evaluator)
        trainer.agent = SimpleNamespace()
        trainer.frames = 2000
        trainer.history = []
        trainer._epsilon = Mock(return_value=0.1)
        trainer._start_episode = Mock(return_value={})
        result = trainer.train()
        self.assertEqual(result['history'], [])
        self.assertEqual(evaluator.n_unique_evaluations, 1)

    def test_haipipe_does_not_start_episode_after_limit(self):
        import tempfile
        from baselines.HAIPipe.pretrain import CrossTaskPretrainer
        trainer = CrossTaskPretrainer.__new__(CrossTaskPretrainer)
        evaluator = self.evaluator(1)
        evaluator.data_name = 'unit'
        evaluator.evaluate(Pipeline([]))
        env = Mock(evaluator=evaluator)
        trainer.environments = [env]
        trainer.frames = 2000
        trainer.trainer = SimpleNamespace(frames=0, training_statistics=lambda: {})
        trainer.agent = Mock()
        with tempfile.TemporaryDirectory() as folder:
            trainer.train(folder + '/policy.pt')
        env.episode.assert_not_called()
        trainer.agent.save.assert_called_once()

    def test_prefix_resource_error_is_not_wrapped(self):
        evaluator = self.evaluator(1)
        evaluator.budget.check_resources = Mock(side_effect=BudgetExceededError('RAM'))
        with self.assertRaises(BudgetExceededError):
            evaluator.run_preprocessing(Pipeline([]))
        evaluator._executor.run_preprocessing.assert_not_called()


if __name__ == '__main__':
    unittest.main()
