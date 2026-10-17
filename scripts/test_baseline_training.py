"""CPU contract/transfer regressions; no raw datasets or LLM weights required."""
from contextlib import ExitStack
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from baselines.common.data_ctx import DataContext, TableContext
from baselines.common.evaluator import EvaluationResult
from baselines.common.pipeline import Pipeline, PipelineStep
from baselines.common.training_utils import contract, freeze_agent, load_pretrained, isolate_training_data
from baselines.CtxPipe.agent import DQNAgent
from baselines.CtxPipe.env import PipelineEnv
from baselines.CtxPipe.model import DQN, ForwardMode
from baselines.FlowPipe.agent import GFNAgent
from baselines.FlowPipe.env import FlowPipeEnv
from baselines.HAIPipe.env import HAIPipeEnv


class Evaluator:
    """Keep the actual agent/environment; substitute only downstream training."""
    def __init__(self, task_dir=None, data_name="toy", **kwargs):
        self.data_name = data_name
        self.task_type = "tabular"
        self.metric_key = "auc"
        self.n_unique_evaluations = 0
        self.frame = pd.DataFrame({"x": [1., 2., 3., 4.], "target": [0, 1, 0, 1]})
        self._executor = SimpleNamespace(get_model_config=lambda: {"model": "LightGBM"}, cleanup=lambda: None)

    def load_data(self):
        return SimpleNamespace(train_df=self.frame.copy(), target_col="target", id_col=None,
                               auxiliary_dfs={}, model_cfg={"model_params": {"task": "binary"}})

    def set_evaluate_std_test(self, enabled):
        assert not enabled

    def set_fast_mode(self, *args):
        pass

    def run_preprocessing(self, pipeline):
        return self.frame.copy()

    def evaluate(self, pipeline):
        self.n_unique_evaluations += 1
        return EvaluationResult(.7, True, {"auc": .7}, None, .01)


def context(name="toy", numeric=True):
    return DataContext("tabular", name, "main", tables={"main": TableContext(
        "main", "primary", dtypes={"x": "float64" if numeric else "object", "target": "int64"},
        configured_target_col="target")})


class BaselineTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_global_ctx_actions_survive_schema_changes(self):
        a = PipelineEnv(Evaluator(), context(), device="cpu")
        b = PipelineEnv(Evaluator(), context("other", False), device="cpu")
        self.assertEqual(contract("CtxPipe", a, "LightGBM"), contract("CtxPipe", b, "LightGBM"))
        self.assertNotEqual(a.applicable_ops, b.applicable_ops)
        for key in a.component_ids:
            self.assertEqual(a.component(key).candidates, b.component(key).candidates)

    def test_global_hai_actions_survive_task_types(self):
        a = HAIPipeEnv(Evaluator(), context(), model_name="LightGBM")
        rec = context("rec")
        rec.task_type = "rec"
        b = HAIPipeEnv(Evaluator(), rec, model_name="LightGBM")
        self.assertEqual(a.component_actions, b.component_actions)
        self.assertEqual(a.primitive_vocabulary, b.primitive_vocabulary)

    def test_ctx_gate_is_optimized_and_roundtrips(self):
        net = DQN(4, 3, "cpu")
        self.assertIn("context_gate", dict(net.named_parameters()))
        old = net.context_gate.detach().clone()
        opt = torch.optim.Adam(net.parameters(), lr=.01)
        state, ctx = torch.ones(2, 4), torch.ones(2, 4096)
        loss = net(state, ctx, ForwardMode.GATED).square().mean()
        loss.backward(); opt.step()
        self.assertFalse(torch.equal(old, net.context_gate))
        other = DQN(4, 3, "cpu")
        other.load_state_dict(net.state_dict())
        torch.testing.assert_close(net(state, ctx), other(state, ctx), rtol=0, atol=0)

    def test_agent_rejects_changed_action_contract_and_missing_weights(self):
        with tempfile.TemporaryDirectory() as root:
            path = str(Path(root) / "checkpoint.pt")
            agent = GFNAgent(4, 3, hidden_dim=8, ctx_dim=5)
            agent.checkpoint_metadata = {"actions": ["a", "b"]}
            agent.save(path)
            with self.assertRaises(ValueError):
                load_pretrained(agent, path, {"actions": ["b", "a"]})
            with self.assertRaises(FileNotFoundError):
                load_pretrained(agent, str(Path(root) / "missing"), agent.checkpoint_metadata)

    def test_agent_profiles_and_execution_never_load_frozen_test(self):
        for kind in ("tabular", "rec"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as root:
                filename = "interaction_frozen.parquet" if kind == "rec" else "train_frozen.parquet"
                path = Path(root) / filename; path.touch()
                data = SimpleNamespace(data_dir=root, _std_test_dir=lambda _: root,
                    load_data=lambda: None, load_std_test_frozen=lambda: self.fail("test was loaded"),
                    _normalize_frozen_frame=lambda f: f)
                ev = SimpleNamespace(task_type=kind, small_n=0, _orig_make_data_instance=lambda: data,
                    _executor=SimpleNamespace(clear_data_cache=lambda: None))
                isolate_training_data(ev)
                training = pd.DataFrame({"x": [1., 2.], "target": [0, 1]})
                with patch("pandas.read_parquet", return_value=training) as read:
                    loaded = ev._executor._make_data_instance()
                    loaded.load_data()
                    loaded.load_std_test_frozen()
                read.assert_called_once_with(path)
                self.assertTrue(loaded._skip_std_test_loading)
                result = loaded.interaction_df if kind == "rec" else loaded.train_df
                pd.testing.assert_frame_equal(result, training)

    def test_ctx_shared_training_visits_two_sources_and_updates_gate(self):
        from baselines.CtxPipe.context import GTEEmbedder
        from baselines.CtxPipe.trainer import CtxPipeTrainer
        first, second = Evaluator(data_name="first"), Evaluator(data_name="second")
        envs = [PipelineEnv(first, context("first")), PipelineEnv(second, context("second"))]
        agent = DQNAgent(envs[0].component_action_dims, envs[0].primitive_vocab_size,
                         envs[0].n_logical_pipelines)
        before = agent.logical_model.network.context_gate.detach().clone()
        with patch.object(GTEEmbedder, "embed_frame", return_value=torch.ones(4096)):
            result = CtxPipeTrainer(envs, agent, frames=16, batch_size=4, logical_batch_size=2,
                                   backpropagate_interval=1, verbose=False).train()
        self.assertGreater(first.n_unique_evaluations, 0)
        self.assertGreater(second.n_unique_evaluations, 0)
        self.assertGreater(result["training_stats"]["logical_optimizer_steps"], 0)
        self.assertFalse(torch.equal(before, agent.logical_model.network.context_gate))

    def test_ctx_runner_modes(self):
        import baselines.CtxPipe.ctxpipe as module
        from baselines.CtxPipe.ctxpipe import CtxPipe
        from baselines.CtxPipe.trainer import CtxPipeTrainer
        from baselines.CtxPipe.tester import CtxPipeTester
        from baselines.common.utils import get_data_ctx
        pipeline = Pipeline([PipelineStep("LabelEncode"), PipelineStep("HandleMV")])
        inference = {"pipeline": pipeline, "pipeline_yaml": pipeline.to_yaml(), "fitness": .7,
                     "metrics": {"auc": .7}, "error": None}
        with tempfile.TemporaryDirectory() as root:
            env = PipelineEnv(Evaluator(), get_data_ctx("toy", Evaluator().load_data(), "tabular"))
            a = DQNAgent(env.component_action_dims, env.primitive_vocab_size, env.n_logical_pipelines)
            runner = CtxPipe("unused", "toy", output_dir=root)
            a.checkpoint_metadata = contract("CtxPipe", env, "LightGBM", {
                "path": str(Path(runner.gte_model_path).resolve()), "dim": 4096})
            path = str(Path(root) / "base.pt"); a.save(path)
            for mode in ("scratch", "pretrained", "finetune"):
                with self.subTest(mode=mode), ExitStack() as stack:
                    stack.enter_context(patch.object(module, "SearchPipelineEvaluator", Evaluator))
                    training = stack.enter_context(patch.object(CtxPipeTrainer, "train", return_value={
                        "history": [], "training_stats": {"frames": 12}}))
                    stack.enter_context(patch.object(CtxPipeTester, "run", return_value=inference))
                    runner = CtxPipe("unused", "toy", training_mode=mode, pretrained_checkpoint=path,
                                     output_dir=str(Path(root) / mode), verbose=False, eval_full=False)
                    result = runner.run()
                    self.assertEqual(result["training_mode"], mode)
                    self.assertEqual(training.call_count, 0 if mode == "pretrained" else 1)

    def test_hai_runner_modes_load_both_networks(self):
        import baselines.HAIPipe.haipipe as module
        from baselines.HAIPipe.haipipe import HAIPipe
        from baselines.HAIPipe.pretrain import CrossTaskPretrainer, save_alternative_evaluator
        from baselines.HAIPipe.active_learning import ScoreNN
        from baselines.common.utils import get_data_ctx
        pipeline = Pipeline([PipelineStep("LabelEncode"), PipelineStep("HandleMV")])
        selection = SimpleNamespace(pipeline=pipeline, fitness=.7, metrics={"auc": .7}, candidates=[])
        with tempfile.TemporaryDirectory() as root:
            env = HAIPipeEnv(Evaluator(), get_data_ctx("toy", Evaluator().load_data(), "tabular"), model_name="LightGBM")
            trainer = CrossTaskPretrainer([env], frames=4)
            expected = contract("HAIPipe", env, "LightGBM")
            trainer.agent.checkpoint_metadata = expected
            path = str(Path(root) / "aipipe.pt"); trainer.agent.save(path)
            score = ScoreNN(env.data_dim, len(env.primitive_vocabulary))
            save_alternative_evaluator(score, str(Path(root) / "alternative_evaluator.pt"),
                data_dim=env.data_dim, primitive_vocabulary=env.primitive_vocabulary, checkpoint_metadata=expected)
            del score, trainer
            for mode in ("scratch", "pretrained", "finetune"):
                with self.subTest(mode=mode), ExitStack() as stack:
                    stack.enter_context(patch.object(module, "SearchPipelineEvaluator", Evaluator))
                    stack.enter_context(patch.object(module, "build_hi_pipeline", return_value=pipeline))
                    training = stack.enter_context(patch.object(CrossTaskPretrainer, "train", return_value={"frames": 4}))
                    stack.enter_context(patch.object(HAIPipeEnv, "rollout", return_value=(pipeline, [])))
                    stack.enter_context(patch.object(module, "combine_pipelines", return_value=[pipeline]))
                    stack.enter_context(patch.object(module, "select_best_candidate", return_value=selection))
                    runner = HAIPipe("unused", "toy", training_mode=mode, pretrained_checkpoint=path,
                                     output_dir=str(Path(root) / mode), verbose=False, eval_full=False)
                    result = runner.run()
                    self.assertEqual(result["training_mode"], mode)
                    self.assertEqual(training.call_count, 0 if mode == "pretrained" else 1)

    def test_finetuning_copies_base_and_direct_inference_is_frozen(self):
        with tempfile.TemporaryDirectory() as root:
            path = str(Path(root) / "checkpoint.pt")
            source = GFNAgent(4, 3, hidden_dim=8, ctx_dim=5)
            source.checkpoint_metadata = {"model": "toy"}
            source.save(path)
            original = Path(path).read_bytes()
            agent = GFNAgent(4, 3, hidden_dim=8, ctx_dim=5)
            load_pretrained(agent, path, source.checkpoint_metadata)
            x, ctx = torch.ones(1, 4), torch.ones(1, 5)
            torch.testing.assert_close(source.model(x, ctx)[0], agent.model(x, ctx)[0])
            loss = agent.model(x, ctx)[0].square().sum()
            agent.optimizer.zero_grad(); loss.backward(); agent.optimizer.step()
            self.assertFalse(torch.equal(source.model(x, ctx)[0], agent.model(x, ctx)[0]))
            self.assertEqual(original, Path(path).read_bytes())
            freeze_agent(agent)
            self.assertTrue(all(not p.requires_grad for p in agent.model.parameters()))
            self.assertFalse(agent.model.training)

    def test_flow_runner_modes_load_and_train_independently(self):
        from baselines.FlowPipe.flowpipe import FlowPipe
        from baselines.FlowPipe.trainer import FlowPipeTrainer
        from baselines.FlowPipe.context import SemanticContextLoader
        from baselines.common.utils import get_data_ctx
        from baselines.common.budget import BudgetExceededError
        pipeline = Pipeline(steps=[PipelineStep("LabelEncode", params={}),
                                   PipelineStep("HandleMV", params={"method": "median"})])
        with tempfile.TemporaryDirectory() as root:
            eval_ = Evaluator()
            ctx = get_data_ctx("toy", eval_.load_data(), "tabular")
            env = FlowPipeEnv(eval_, ctx)
            base = FlowPipe("unused", "toy", output_dir=str(Path(root) / "setup"),
                            training_mode="scratch", hidden_dim=8)
            source = GFNAgent(env.state_dim, env.action_dim, hidden_dim=8, ctx_dim=4096)
            source.checkpoint_metadata = contract("FlowPipe", env, "LightGBM", {
                "path": str(Path(base.llama_model_path).resolve()), "name": base.llm_name, "dim": 4096})
            source_path = str(Path(root) / "base.pt")
            source.save(source_path)
            base_bytes = Path(source_path).read_bytes()
            for mode in ("scratch", "pretrained", "finetune"):
                out = str(Path(root) / mode)
                with self.subTest(mode=mode), ExitStack() as stack:
                    stack.enter_context(patch.object(FlowPipe, "_build_evaluator", return_value=Evaluator()))
                    stack.enter_context(patch.object(SemanticContextLoader, "load", return_value=torch.zeros(4096)))
                    training = stack.enter_context(patch.object(FlowPipeTrainer, "train", return_value=[]))
                    stack.enter_context(patch.object(GFNAgent, "sample_trajectory", return_value=(.7, {
                        "actions": [0], "errors": [], "success": True, "fitness": .7, "metrics": {"auc": .7}})))
                    stack.enter_context(patch.object(FlowPipeEnv, "current_pipeline", return_value=pipeline))
                    runner = FlowPipe("unused", "toy", training_mode=mode, pretrained_checkpoint=source_path,
                                      hidden_dim=8, output_dir=out, finetune_epochs=3, train_epochs=7)
                    result = runner.run()
                    self.assertEqual(result["training_mode"], mode)
                    if mode == "pretrained":
                        training.assert_not_called()
                    else:
                        training.assert_called_once_with(num_epochs=3 if mode == "finetune" else 7)
                    self.assertEqual(base_bytes, Path(source_path).read_bytes())


if __name__ == "__main__":
    unittest.main()
