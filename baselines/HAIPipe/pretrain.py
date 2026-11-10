"""AIPipe policy training for one or more HAIPipe environments."""
from __future__ import annotations

import os
import random
import math
from baselines.common.search_reward import fitness_to_reward

from baselines.common.pipeline import finalize_pipeline

from .agent import HAIPipeAgent
from .env import HAIPipeEnv
from .trainer import HAIPipeTrainer


class CrossTaskPretrainer:
    """Train one operator-layout checkpoint over compatible environments."""

    def __init__(
        self, environments: list[HAIPipeEnv], *, frames: int = 100000,
        learning_rate: float = 1e-5, buffer_capacity: int = 2000,
        batch_size: int = 100, gamma: float = 1.0, train_interval: int = 100,
        logic_batch_size: int = 20, epsilon_start: float = 1.0,
        epsilon_end: float = 0.4, epsilon_decay: float = 500.0,
        seed: int = 42, device: str = "cpu",
    ) -> None:
        if not environments:
            raise ValueError("HAIPipe pretraining requires at least one training task")
        first = environments[0]
        layout = (
            first.component_actions, first.logical_pipelines,
            first.primitive_vocabulary, first.model_vocabulary, first.data_dim,
        )
        for env in environments[1:]:
            current = (
                env.component_actions, env.logical_pipelines,
                env.primitive_vocabulary, env.model_vocabulary, env.data_dim,
            )
            if current != layout:
                raise ValueError(
                    "All tasks in one HAIPipe pretraining run must share task type, "
                    "model vocabulary, components, and operator layout"
                )
        self.environments = environments
        if frames < 1 or train_interval < 1 or batch_size < 1 or logic_batch_size < 1 or epsilon_decay <= 0:
            raise ValueError("HAIPipe training budgets, batch sizes and epsilon decay must be positive")
        self.frames = int(frames)
        self.epsilon_parameters = (epsilon_start, epsilon_end, epsilon_decay)
        self.seed = int(seed)
        self.rng = random.Random(seed)
        self.agent = HAIPipeAgent(
            component_actions=first.component_actions,
            logical_pipelines=first.logical_pipelines,
            primitive_vocabulary=first.primitive_vocabulary,
            model_vocabulary=first.model_vocabulary,
            data_dim=first.data_dim, learning_rate=learning_rate,
            buffer_capacity=buffer_capacity, seed=seed, device=device,
        )
        self.trainer = HAIPipeTrainer(
            self.agent, batch_size=batch_size, gamma=gamma,
            train_interval=train_interval, logic_batch_size=logic_batch_size,
        )

    def train(
        self, checkpoint_path: str, *, held_out_tasks: list[str] | None = None,
        training_mode: str = "cross_task_offline",
    ) -> dict:
        episode_count = 0
        while self.trainer.frames < self.frames:
            env = self.environments[episode_count % len(self.environments)]
            epsilon = self.agent.epsilon(self.trainer.frames, *self.epsilon_parameters)
            episode = env.episode(self.agent, epsilon=epsilon)
            executable, report = finalize_pipeline(
                episode["pipeline"], env.ctx, self.rng, sort_steps=False,
            )
            result = env.evaluator.evaluate(executable) if report.valid else None
            reward = (fitness_to_reward(result.fitness, getattr(env.evaluator, "metric_key", None))
                      if result is not None and result.success and math.isfinite(result.fitness) else 0.0)
            if not episode["transitions"]:
                raise RuntimeError("HAIPipe episode produced no transitions; training cannot progress")
            for transition in episode["transitions"]:
                transition["reward"] = reward if transition["done"] else 0.0
            self.trainer.add_episode(
                episode["transitions"], episode["logical_state"],
                episode["logical_action"], reward,
            )
            episode_count += 1
        os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
        self.agent.save(checkpoint_path, provenance={
            "training_mode": training_mode,
            "training_tasks": [env.evaluator.data_name for env in self.environments],
            "held_out_tasks": list(held_out_tasks or []),
            "frames": self.trainer.frames,
        })
        return {"frames": self.trainer.frames, "episodes": episode_count, "checkpoint": checkpoint_path}


__all__ = ["CrossTaskPretrainer"]


def save_alternative_evaluator(
    model, path: str, *, data_dim: int, primitive_vocabulary: list[str],
    provenance: dict | None = None, checkpoint_metadata: dict | None = None,
) -> None:
    """Save the strict vocabulary-bearing checkpoint consumed by HAIPipe."""
    import torch
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if model.dataset_dim != int(data_dim):
        raise ValueError("ScoreNN dataset dimension differs from checkpoint metadata")
    torch.save({
        "checkpoint_metadata": checkpoint_metadata,
        "metadata": model.metadata(primitive_vocabulary),
        "provenance": dict(provenance or {}),
        "model": model.state_dict(),
    }, path)


__all__.append("save_alternative_evaluator")


def load_alternative_evaluator(model, path, vocabulary, checkpoint_metadata):
    import torch
    checkpoint = torch.load(path, map_location=next(model.parameters()).device, weights_only=True)
    if checkpoint.get("metadata") != model.metadata(vocabulary) or checkpoint.get("checkpoint_metadata") != checkpoint_metadata:
        raise ValueError("Incompatible HAIPipe alternative-evaluator checkpoint; retrain both weights together")
    model.load_state_dict(checkpoint["model"])
    return checkpoint.get("provenance", {})
