"""CPU regressions for the merged four-baseline snapshot (no downloads)."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import os
import tempfile
import unittest

import pandas as pd
import torch

from baselines.common.search_reward import fitness_to_reward
from baselines.common.data_ctx import DataContext, TableContext
from baselines.common.pipeline import Pipeline, PipelineStep
from baselines.common.llm_quantization import fp8_quantization_config, local_torch_dtype
from baselines.common.budget import BudgetExceededError
from baselines.DeepPrep.evaluator import DeepPrepEvaluator
from baselines.BAT.evaluator import BATEvaluator
from baselines.BAT.reward import BATReward
from baselines.BAT.node import MCTSNode, MCTSNodeType
from baselines.DeepPrep.sandbox import Sandbox, ObsState, ExecutionResult
from baselines.DeepPrep.tree_agent import TreeAgent
from baselines.CtxPipe.context import GTEEmbedder, CONTEXT_DIM
from baselines.FlowPipe.env import FlowPipeEnv
from baselines.FlowPipe.agent import GFNAgent
from baselines.FlowPipe.context import SemanticContextLoader
from dppbench.tasks.nyc_taxi_hourly.nyc_taxi_hourly_data import NycTaxiHourlyData

ROOT = Path(__file__).resolve().parents[1]


def context():
    return DataContext('tabular', 'toy', 'main', tables={
        'main': TableContext('main', 'primary', dtypes={'x': 'float64', 'target': 'float64'})
    }, prediction_task='regression')


class MergeCompatibility(unittest.TestCase):
    def test_nyc_preserves_frozen_partitions(self):
        data = NycTaxiHourlyData()
        data.train_df = pd.DataFrame({'_hour_idx': range(10), 'trip_count': range(10),
                                     '__split__': ['train'] * 6 + ['val'] * 3 + ['std_test']})
        data._validation_prepared = True
        data.test_df = None
        result = data.split()
        self.assertEqual(result['train']['_hour_idx'].tolist(), list(range(6)))
        self.assertEqual(result['val']['_hour_idx'].tolist(), [6, 7, 8])
        self.assertEqual(result['std_test']['_hour_idx'].tolist(), [9])

    def test_both_evaluators_resolve_regression_and_classification(self):
        for evaluator in (DeepPrepEvaluator, BATEvaluator):
            for task, metric in [('nyc_taxi_hourly', 'rmse'), ('default_credit', 'auc')]:
                with self.subTest(evaluator=evaluator.__name__, task=task):
                    instance = evaluator(str(ROOT / 'dppbench/tasks' / task), task)
                    self.assertEqual(instance.metric_key, metric)

    def test_reward_orders_errors_and_preserves_auc(self):
        values = [fitness_to_reward(-v, 'rmse') for v in (5, 50, 500)]
        self.assertGreater(values[0], values[1])
        self.assertGreater(values[1], values[2])
        self.assertGreater(values[2], 0)
        self.assertEqual(fitness_to_reward(.8, 'auc'), .8)
        with self.assertRaises(ValueError):
            fitness_to_reward(float('nan'), 'auc')

    def test_bat_regression_reward_and_raw_metrics(self):
        sandbox = Mock()
        sandbox._initial_snapshot = b'initial'
        sandbox.execute_chain.return_value.success = True
        evaluator = Mock(metric_key='rmse')
        reward = BATReward(context(), sandbox, downstream_evaluator=evaluator)
        scores = []
        with patch('baselines.BAT.reward.expected_target_columns', return_value=['x']), patch(
            'baselines.BAT.reward._sandbox_columns', return_value=['x']
        ):
            for error in (5., 50., 500.):
                evaluator.evaluate_for_agent.return_value = (-error, {'rmse': error}, None)
                node = MCTSNode(MCTSNodeType.END,
                                final_pipeline_steps=[PipelineStep('LabelEncode', 'both', {})])
                score, info = reward.score(node)
                scores.append(score)
                self.assertEqual(info['downstream_fitness'], -error)
                self.assertEqual(info['downstream_metrics'], {'rmse': error})
        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[1], scores[2])

    def test_flowpipe_regression_rollout_updates_real_policy(self):
        frame = pd.DataFrame({'x': [1., 2., 3.], 'target': [1., 4., 9.]})
        ctx = context()
        evaluator = Mock(metric_key='rmse')
        evaluator.load_data.return_value = SimpleNamespace(train_df=frame)
        evaluator.run_preprocessing.return_value = frame
        rewards = []
        trajectory = None
        torch.manual_seed(42)
        env = FlowPipeEnv(evaluator, ctx, max_steps=4)
        agent = GFNAgent(env.state_dim, env.action_dim, hidden_dim=8, ctx_dim=4)
        agent.set_context(torch.zeros(4))
        with patch('baselines.FlowPipe.env.check_pipeline', return_value=SimpleNamespace(valid=True)):
            for rmse in (5., 50.):
                evaluator.evaluate.return_value = SimpleNamespace(success=True, fitness=-rmse,
                                                                    metrics={'rmse': rmse})
                env.reset()
                value, trajectory = agent.sample_trajectory(env, greedy_rank=0)
                self.assertTrue(trajectory['success'])
                self.assertEqual(trajectory['fitness'], -rmse)
                rewards.append(value)
        self.assertGreater(rewards[0], rewards[1])
        before = [parameter.detach().clone() for parameter in agent.model.parameters()]
        losses = agent.learn_from_batch([trajectory], updates=1)
        self.assertEqual(len(losses), 1)
        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, agent.model.parameters())))

    def test_deepprep_selects_smallest_rmse_without_auc_label(self):
        obs = ObsState((3, 2), ['x', 'target'], {'x': 'float64', 'target': 'float64'},
                       [], 0., 'main', {'main': {'columns': ['x', 'target']}})
        sandbox = Mock()
        sandbox.reset.return_value = ExecutionResult(True, obs=obs, snapshot=b'root')
        sandbox.restore.return_value = obs
        sandbox.execute_chain.return_value = ExecutionResult(True, obs=obs, snapshot=b'child')
        llm = Mock()
        llm.chat.side_effect = [
            'HandleMV(method="median", action="impute") --> LabelEncode() </execute>',
            'HandleMV(method="mean", action="impute") --> LabelEncode() </execute>',
        ]
        evaluator = Mock(metric_key='rmse')
        evaluator.evaluate_for_agent.side_effect = [(-50., {'rmse': 50.}, None), (-5., {'rmse': 5.}, None)]
        result = TreeAgent(llm, sandbox, context(), downstream_evaluator=evaluator,
                           max_explore_turn=2, verbose=False).run()
        self.assertTrue(result.success)
        self.assertEqual(result.pipeline.steps[0].params['method'], 'mean')
        self.assertEqual([r['fitness'] for r in result.solution_attempts], [-50., -5.])
        self.assertTrue(all('auc' not in r for r in result.solution_attempts))

        # A timeout while evaluating another candidate preserves the previous
        # successful candidate, not the unscored path at the current cursor.
        llm.chat.side_effect = [
            'HandleMV(method="median", action="impute") --> LabelEncode() </execute>',
            'HandleMV(method="mean", action="impute") --> LabelEncode() </execute>',
        ]
        evaluator.evaluate_for_agent.side_effect = [
            (-50., {'rmse': 50.}, None), BudgetExceededError('test budget exhausted'),
        ]
        result = TreeAgent(llm, sandbox, context(), downstream_evaluator=evaluator,
                           max_explore_turn=2, verbose=False).run()
        self.assertTrue(result.success)
        self.assertEqual(result.pipeline.steps[0].params['method'], 'median')
        self.assertEqual(len(result.solution_attempts), 1)

    def test_model_path_cli_preserves_other_baseline_registrations(self):
        from scripts.evaluate_baseline import parse_args, _apply_baseline_model_paths, BASELINE_CONFIGS
        with patch('sys.argv', ['evaluate_baseline.py', '--ctxpipe_model_path', '/models/Qwen3-8B']):
            args = parse_args()
        kwargs = {}
        _apply_baseline_model_paths('CtxPipe', kwargs, args)
        self.assertEqual(kwargs['gte_model_path'], '/models/Qwen3-8B')
        self.assertTrue({'DAGPipe', 'ReAct', 'DataMaster', 'SPIO'} <= set(BASELINE_CONFIGS))

    def test_ctxpipe_bfloat_embedding_supports_cache_and_policy(self):
        class Inputs(dict):
            def to(self, device):
                return self

        with tempfile.TemporaryDirectory() as directory:
            encoder = GTEEmbedder(cache_dir=directory)
            encoder._load = Mock()
            encoder._tokenizer = Mock(return_value=Inputs(attention_mask=torch.ones(1, 2)))
            encoder._model = Mock(return_value=SimpleNamespace(
                last_hidden_state=torch.ones(1, 2, CONTEXT_DIM, dtype=torch.bfloat16)))
            value = encoder.embed_frame(pd.DataFrame({'x': [1, 2]}))
            self.assertEqual(value.dtype, torch.float32)
            self.assertTrue(torch.isfinite(torch.nn.Linear(CONTEXT_DIM, 1)(value)).all())
            encoder.embed_frame(pd.DataFrame({'x': [1, 2]}))
            self.assertEqual(encoder._model.call_count, 1)

    def test_fp8_optional_and_explicit_modes(self):
        with patch('torch.cuda.is_available', return_value=False):
            with patch.dict(os.environ, {'DPPBENCH_LLM_QUANTIZATION': 'auto'}):
                with self.assertWarns(RuntimeWarning):
                    self.assertIsNone(fp8_quantization_config())
            with patch.dict(os.environ, {'DPPBENCH_LLM_QUANTIZATION': 'fp8'}):
                with self.assertRaisesRegex(RuntimeError, 'Explicit FP8'):
                    fp8_quantization_config()
            with patch.dict(os.environ, {'DPPBENCH_LLM_QUANTIZATION': 'none'}):
                self.assertIsNone(fp8_quantization_config())
            self.assertEqual(local_torch_dtype('cpu'), torch.float32)

    def test_sandbox_hides_frozen_rows_from_observations(self):
        sandbox = Sandbox.__new__(Sandbox)
        sandbox.task_type = 'tabular'
        sandbox.data = SimpleNamespace(train_df=pd.DataFrame({
            'x': ['heldout_secret', 'visible'], 'target': [99., 1.],
            '__split__': ['std_test', 'train'],
        }), auxiliary_dfs={})
        obs = sandbox._observe()
        self.assertNotIn('heldout_secret', obs.text)
        self.assertEqual(obs.shape[0], 1)
        self.assertNotIn('__split__', obs.columns)
        self.assertEqual(len(sandbox.data.train_df), 2)

    def test_flowpipe_does_not_reuse_generic_other_model_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.save(torch.zeros(4), Path(directory) / 'toy.pt')
            loader = SemanticContextLoader(directory, 'Qwen3-8B', 4)
            with self.assertRaises(FileNotFoundError):
                loader.load('toy')
            prompt = loader.build_prompt('toy', pd.DataFrame({'sequence': [[1], [2]]}), context())
            self.assertIn('unique=2', prompt)


if __name__ == '__main__':
    unittest.main()
