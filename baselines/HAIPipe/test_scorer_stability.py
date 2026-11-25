import ast
import contextlib
import io
import pathlib
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch

from baselines.HAIPipe.active_learning import ActiveCandidateSelector
from baselines.common.pipeline import Pipeline


class ScorerStabilityTest(unittest.TestCase):
    def test_fit_rolls_back_parameters_and_batchnorm_buffers(self):
        class Unstable(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = torch.nn.BatchNorm1d(2)
                self.weight = torch.nn.Parameter(torch.ones(2))
            def forward(self, dataset, features):
                return (self.bn(dataset) * self.weight).sum(dim=1) * float('nan')
        model = Unstable().eval()
        state = {key: value.clone() for key, value in model.state_dict().items()}
        with contextlib.redirect_stdout(io.StringIO()):
            ActiveCandidateSelector._fit(model, np.array([[1., 2.], [3., 4.]]),
                                         np.zeros((2, 10), dtype=np.int64), [0, 1], [0., 1.])
        self.assertFalse(model.training)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, state[key]), key)

    def test_finite_fit_updates_and_restores_mode(self):
        class Stable(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(2, 1)
            def forward(self, dataset, features):
                return self.layer(dataset)
        model = Stable().eval()
        before = model.layer.weight.detach().clone()
        ActiveCandidateSelector._fit(model, np.ones((2, 2)), np.zeros((2, 10)), [0, 1], [0., 1.])
        self.assertFalse(model.training)
        self.assertFalse(torch.equal(before, model.layer.weight))
        self.assertTrue(all(torch.isfinite(value).all() for value in model.state_dict().values()))

    def test_latent_nan_rejected_before_clustering(self):
        selector = ActiveCandidateSelector(torch.nn.Linear(2, 1))
        with self.assertRaisesRegex(FloatingPointError, 'latent'):
            selector._representatives(np.array([[float('nan')], [1.]]), [0, 1], 1)

    def test_ctxpipe_actual_post_training_branch(self):
        from baselines.CtxPipe import ctxpipe
        tree = ast.parse(pathlib.Path(ctxpipe.__file__).read_text())
        run = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'run')
        branch = next(node for node in run.body if isinstance(node, ast.If)
                      and isinstance(node.test, ast.Attribute) and node.test.attr == 'evaluation_budget_exhausted')
        code = compile(ast.fix_missing_locations(ast.Module(body=[branch], type_ignores=[])), 'ctxpipe_branch', 'exec')
        pipeline = Pipeline([])
        evaluator = SimpleNamespace(evaluation_budget_exhausted=True, best_pipeline=pipeline,
                                    best_result=SimpleNamespace(fitness=0.7, metrics={'auc': 0.7}))
        tester = Mock()
        namespace = dict(train_evaluator=evaluator, tester=tester, agent=None, env=None, full_eval=None)
        exec(code, namespace)
        self.assertEqual(namespace['test_out']['fitness'], 0.7)
        tester.run.assert_not_called()
        evaluator.evaluation_budget_exhausted = False
        exec(code, namespace)
        tester.run.assert_called_once()
        evaluator.evaluation_budget_exhausted = True
        evaluator.best_pipeline = None
        with self.assertRaisesRegex(RuntimeError, 'without a successful'):
            exec(code, namespace)


if __name__ == '__main__':
    unittest.main()
