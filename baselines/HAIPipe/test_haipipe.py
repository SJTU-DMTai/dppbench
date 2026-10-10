from __future__ import annotations

import numpy as np
import pandas as pd

from baselines.common.pipeline import Pipeline, PipelineStep
from baselines.HAIPipe.active_learning import pipeline_features
from baselines.HAIPipe.combiner import _subsequences
from baselines.HAIPipe.features import dataset_features


def test_dataset_features_match_upstream_shape_and_order():
    frame = pd.DataFrame({"number": [1.0, 2.0, np.nan], "text": ["a", "a", "b"]})
    features = dataset_features(frame)
    assert features.shape == (100 * 19,)
    assert features[1] == 3
    assert np.isclose(features[2], 1 / 3)
    assert features[3] == 1
    assert features[19 + 3] == 2


def test_all_nonempty_ordered_subsequences_are_enumerated():
    steps = [PipelineStep("a"), PipelineStep("b"), PipelineStep("c")]
    names = [tuple(step.op for step in seq) for seq in _subsequences(steps)]
    assert len(names) == 7
    assert ("a", "c") in names
    assert ("c", "a") not in names


def test_pipeline_features_use_checkpoint_vocabulary():
    pipe = Pipeline([PipelineStep("b"), PipelineStep("unknown")])
    encoded = pipeline_features(pipe, ["a", "b"], sequence_length=4)
    assert encoded.tolist() == [2, 3, 0, 0]


def test_models_match_released_checkpoint_shapes():
    """Golden shapes come from checkpoint metadata, not our model definitions."""
    import json
    from pathlib import Path
    import torch
    from .active_learning import ScoreNN
    from .model import LogicalDQN, PrimitiveRnnDQN

    reference = json.loads(Path(__file__).with_name("upstream_checkpoint_shapes.json").read_text())
    with torch.device("meta"):
        models = {
            "logical": LogicalDQN(1901, 6),
            "primitive": PrimitiveRnnDQN(
                data_dim=1900, sequence_length=6, primitive_vocab_size=27,
                predictor_vocab_size=19, logical_pipeline_count=6, action_dim=9,
            ),
            "score": ScoreNN(1900, 19),
        }
    for name, model in models.items():
        assert {k: list(v.shape) for k, v in model.state_dict().items()} == reference["checkpoints"][name]["shapes"]


def test_features_exclude_target_test_rows_and_missing_mode():
    from types import SimpleNamespace
    from .features import frame_from_loaded_data

    frame = pd.DataFrame({
        "x": [1.0, np.nan, np.nan, 100.0], "label": [0, 1, 1, 9],
        "__split__": ["train", "train", "train", "std_test"],
    })
    predictors = frame_from_loaded_data(SimpleNamespace(train_df=frame, target_col="label"))
    assert list(predictors) == ["x"]
    features = dataset_features(predictors)
    assert features[1] == 3
    assert np.isclose(features[15], 1 / 3)
    assert features[6] == 1
    assert np.isfinite(dataset_features(pd.DataFrame({"x": [np.nan, np.nan]}))).all()


def test_environment_updates_state_preserves_skip_and_resets(monkeypatch):
    from types import SimpleNamespace
    from . import env as module
    from baselines.common.op_spec import OpCategory

    specs = [SimpleNamespace(op_name="ScaleFeature", category=OpCategory.PREPROCESSING),
             SimpleNamespace(op_name="SelectFeature", category=OpCategory.FEATURE_ENGINEERING)]
    monkeypatch.setattr(module, "get_available_ops", lambda _: specs)
    monkeypatch.setattr(module, "assign_dag_structure", lambda pipe, *_: pipe)
    monkeypatch.setattr(module, "check_pipeline", lambda *_, **__: SimpleNamespace(valid=True))
    monkeypatch.setattr(module, "make_step", lambda name, *_: PipelineStep(name))
    raw = pd.DataFrame({"x": [1., 2.], "label": [0, 1]})
    processed = pd.DataFrame({"x": [10., 20.], "label": [0, 1]})
    evaluator = SimpleNamespace(
        load_data=lambda: SimpleNamespace(train_df=raw, target_col="label"),
        run_preprocessing=lambda _: processed,
    )
    env = module.HAIPipeEnv(evaluator, SimpleNamespace(task_type="tabular", target_col="label"), model_name="m")
    agent = SimpleNamespace(select_logical=lambda *_: 0,
                            select_component=lambda component, *_: 0 if component == "Preprocessing" else 1)
    for _ in range(2):
        episode = env.episode(agent)
        first, last = episode["transitions"]
        assert first["state"][6] == 1.5
        assert first["next_state"][6] == 15
        assert last["state"][6] == 15
        assert first["next_component"] == "FeatureEngineering"
        assert last["done"] and last["next_component"] is None
        assert first["state"][1900:1902].tolist() == [3, 3]
        assert last["next_state"][1900:1902].tolist() == [1, 0]
        assert episode["logical_state"][-1] == 1
        assert first["state"][-2] == 0
        assert env.data_feature[6] == 1.5
    # A failed execution must not replace the last accepted state.
    evaluator.run_preprocessing = lambda _: None
    assert not env._step_executes([PipelineStep("ScaleFeature")])
    assert env.current_data_feature[6] == 15


def test_dqn_bootstraps_from_next_component_and_stops_at_terminal():
    import random
    import torch
    from .agent import HAIPipeAgent, ReplayBuffer, Transition

    agent = HAIPipeAgent.__new__(HAIPipeAgent)
    agent.device = torch.device("cpu")
    agent.rng = random.Random(42)
    agent.buffer = ReplayBuffer(10, 42)
    agent.models = {key: torch.nn.Linear(1, 1) for key in ("first", "next")}
    with torch.no_grad():
        for model in agent.models.values():
            model.weight.zero_()
            model.bias.zero_()
        agent.models["next"].bias.fill_(0.75)
    agent.optimizers = {key: torch.optim.SGD(model.parameters(), lr=0) for key, model in agent.models.items()}
    state = np.array([0.], dtype=np.float32)
    agent.buffer.add(Transition("first", state, 0, 0., state, False, "next"))
    agent.buffer.add(Transition("next", state, 0, 0.25, state, True))
    loss = agent.learn()
    assert np.isclose(loss["first"], 0.75 ** 2)
    assert np.isclose(loss["next"], (0.75 - 0.25) ** 2)


def test_trainer_does_not_miss_interval_inside_episode():
    from types import SimpleNamespace
    from .agent import ReplayBuffer
    from .trainer import HAIPipeTrainer

    buffer = ReplayBuffer(20, 42)
    calls = []
    agent = SimpleNamespace(buffer=buffer, add_transition=buffer.add,
                            learn=lambda *args: calls.append(args) or {"loss": 1.})
    trainer = HAIPipeTrainer(agent, batch_size=1, train_interval=3)
    transition = dict(component="a", state=np.zeros(1), action=0, reward=0., next_state=np.zeros(1), done=True)
    trainer.add_episode([transition] * 4, np.zeros(1), 0, 0.)
    assert trainer.frames == 4
    assert calls == [(1, 1.0, 20)]


def test_score_pretrain_and_checkpoint_round_trip(tmp_path):
    import torch
    from .active_learning import ScoreNN, pretrain_score_model
    from .pretrain import save_alternative_evaluator

    torch.manual_seed(42)
    model = ScoreNN(4, 2, sequence_length=3)
    before = model.dataset_nn[0].weight.detach().clone()
    history = pretrain_score_model(
        model, np.arange(20, dtype=np.float32).reshape(5, 4),
        np.array([[1, 2, 0]] * 5), np.array([-0.2, -0.1, 0., 0.1, 0.2]),
        epochs=1, batch_size=2,
    )
    assert len(history) == 1 and np.isfinite(history).all()
    assert not torch.equal(before, model.dataset_nn[0].weight)
    model.eval()
    with torch.no_grad():
        # Released checkpoint uses running statistics, so singleton inference works.
        output = model(torch.ones(1, 4), torch.tensor([[1, 2, 0]]))
    assert output.shape == (1, 1) and torch.isfinite(output).all()
    path = tmp_path / "score.pt"
    save_alternative_evaluator(model, str(path), data_dim=4, primitive_vocabulary=["a", "b"], provenance={"held_out_tasks": ["target"]})
    state = torch.load(path, weights_only=True)
    assert state["metadata"] == model.metadata(["a", "b"])
    restored = ScoreNN(4, 2, sequence_length=3)
    restored.load_state_dict(state["model"])
    restored.eval()
    with torch.no_grad():
        assert torch.equal(output, restored(torch.ones(1, 4), torch.tensor([[1, 2, 0]])))


def test_old_policy_checkpoint_is_rejected_before_loading_weights(tmp_path):
    import torch
    from .agent import HAIPipeAgent
    import pytest

    agent = HAIPipeAgent.__new__(HAIPipeAgent)
    agent.device = torch.device("cpu")
    agent.component_actions = {"one": ["a"]}
    agent.logical_pipelines = [["one"]]
    agent.primitive_vocabulary = ["a"]
    agent.model_vocabulary = ["m"]
    agent.data_dim = 1900
    agent.sequence_length = 1
    metadata = agent.metadata()
    metadata["format_version"] = 1
    path = tmp_path / "old.pt"
    torch.save({"metadata": metadata}, path)
    with pytest.raises(ValueError, match="incompatible"):
        agent.load(str(path))


def test_pretraining_cli_uses_aligned_defaults(monkeypatch):
    import sys
    from baselines.HAIPipe.pretrain_haipipe import parse_args

    monkeypatch.setattr(sys, "argv", ["pretrain_haipipe.py", "--data_names", "train", "--held_out_data_names", "test",
                                     "--model", "m", "--data_dir", "/unused", "--output_dir", "/unused"])
    args = parse_args()
    assert (args.gamma, args.batch_size, args.logic_batch_size) == (1.0, 100, 20)
    assert (args.al_epochs, args.al_learning_rate, args.al_batch_size) == (10, 0.01, 16)


def test_partial_state_uses_shared_dag_validation():
    from types import SimpleNamespace
    from .env import HAIPipeEnv
    from baselines.common.data_ctx import infer_data_context
    from baselines.common.pipeline import check_pipeline

    raw = pd.DataFrame({"x": [1., 2.], "label": [0, 1]})
    data = SimpleNamespace(train_df=raw, target_col="label")
    ctx = infer_data_context("synthetic", data, "tabular")
    seen = []

    def execute(pipe):
        assert check_pipeline(pipe, ctx, require_mandatory=False).valid
        assert pipe.op_names() == ["ScaleFeature"]
        seen.append(pipe)
        return raw.assign(x=[0., 1.])

    env = HAIPipeEnv(SimpleNamespace(load_data=lambda: data, run_preprocessing=execute), ctx, model_name="m")
    step = PipelineStep("ScaleFeature", params={"method": "standard", "cols": ["x"]})
    assert env._step_executes([step])
    assert len(seen) == 1
    assert env.current_data_feature[6] == 0.5
    # Running the prefix must not add mandatory operators or mutate the action.
    assert step.id is None


def test_policy_round_trip_and_inference_disables_dropout(monkeypatch, tmp_path):
    import torch
    from . import agent as module

    modes = []

    class TinyPolicy(torch.nn.Module):
        def __init__(self, inputs, outputs):
            super().__init__()
            self.linear = torch.nn.Linear(inputs, outputs)
            self.dropout = torch.nn.Dropout(0.5)

        def forward(self, state):
            modes.append(self.training)
            return self.linear(self.dropout(state))

    monkeypatch.setattr(module, "LogicalDQN", TinyPolicy)
    monkeypatch.setattr(module, "PrimitiveRnnDQN", lambda **kw: TinyPolicy(kw["data_dim"] + kw["sequence_length"] + 2, kw["action_dim"]))
    options = dict(component_actions={"one": ["op"]}, logical_pipelines=[["one"]],
                   primitive_vocabulary=["op"], model_vocabulary=["m"], data_dim=2)
    agent = module.HAIPipeAgent(**options)
    state = np.array([1., 2., 2., 0., 0.], dtype=np.float32)
    action = agent.select_component("one", state)
    assert modes == [False]
    assert agent.models["one"].training
    path = tmp_path / "policy.pt"
    provenance = {"training_tasks": ["train"], "held_out_tasks": ["test"]}
    agent.save(str(path), provenance)
    restored = module.HAIPipeAgent(**options, seed=99)
    assert restored.load(str(path)) == provenance
    assert restored.select_component("one", state) == action
    for key, value in agent.models["one"].state_dict().items():
        assert torch.equal(value, restored.models["one"].state_dict()[key])


def test_runner_trains_fresh_models_per_task_model(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from . import haipipe as module

    calls = {}
    hi = Pipeline([PipelineStep("DropMissingRows")])
    ai = Pipeline([PipelineStep("ScaleFeature")])

    class FakeEvaluator:
        task_type = "tabular"
        n_unique_evaluations = 3

        def __init__(self, **kwargs):
            calls["evaluator"] = kwargs

        def set_evaluate_std_test(self, value):
            calls["std_test"] = value

        def set_fast_mode(self, value):
            calls["fast_mode"] = value

        def load_data(self):
            return SimpleNamespace()

    class FakeEnv:
        data_dim = 4
        primitive_vocabulary = ["DropMissingRows", "ScaleFeature"]
        data_feature = np.ones(4, dtype=np.float32)

        def __init__(self, evaluator, ctx, model_name, seed):
            calls["env_model"] = model_name

        def rollout(self, agent, epsilon):
            calls["rollout"] = (agent, epsilon)
            return ai, [{"action": 0}]

    class FakeTrainer:
        agent = object()

        def __init__(self, environments, **kwargs):
            calls["trainer_envs"] = environments
            calls["trainer_kwargs"] = kwargs

        def train(self, path, **kwargs):
            calls["policy_save"] = (path, kwargs)
            return {"frames": 56000, "episodes": 1, "checkpoint": path}

    class FakeSelector:
        def __init__(self, model, **kwargs):
            calls["selector"] = kwargs

    model = object()
    monkeypatch.setattr(module, "SearchPipelineEvaluator", FakeEvaluator)
    monkeypatch.setattr(module, "get_data_ctx", lambda *args: SimpleNamespace())
    monkeypatch.setattr(module, "build_hi_pipeline", lambda *args, **kwargs: hi)
    monkeypatch.setattr(module, "HAIPipeEnv", FakeEnv)
    monkeypatch.setattr(module, "CrossTaskPretrainer", FakeTrainer)
    monkeypatch.setattr(module, "combine_pipelines", lambda *args, **kwargs: [hi])
    monkeypatch.setattr(module, "ScoreNN", lambda *args: model)
    monkeypatch.setattr(module, "ActiveCandidateSelector", FakeSelector)
    monkeypatch.setattr(
        module, "select_best_candidate",
        lambda *args, **kwargs: SimpleNamespace(
            pipeline=hi, fitness=0.75, metrics={"auc": 0.75},
            candidates=[{"candidate": 0}], error=None,
        ),
    )
    monkeypatch.setattr(
        module, "save_alternative_evaluator",
        lambda current, path, **kwargs: calls.update(
            al_save=(current, path, kwargs),
        ),
    )
    monkeypatch.setattr(
        module, "check_pipeline",
        lambda *args: SimpleNamespace(valid=True, errors=[]),
    )

    output_dir = tmp_path / "task_model"
    result = module.HAIPipe(
        task_dir="/task", data_name="task", data_dir="/data",
        model_name="LightGBM", output_dir=str(output_dir),
        eval_full=False, verbose=False,
    ).run()

    assert calls["evaluator"]["model_name"] == "LightGBM"
    assert calls["env_model"] == "LightGBM"
    assert calls["trainer_kwargs"]["frames"] == 56000
    assert calls["policy_save"] == (
        str(output_dir / "aipipe.pt"),
        {"training_mode": "target_task_online"},
    )
    assert calls["rollout"] == (FakeTrainer.agent, 0.0)
    assert calls["al_save"][1] == str(output_dir / "alternative_evaluator.pt")
    provenance = calls["al_save"][2]["provenance"]
    assert provenance["training_tasks"] == ["task"]
    assert provenance["model_name"] == "LightGBM"
    assert result["training_mode"] == "per_task_model"
    assert result["policy_training"]["frames"] == 56000
