"""Cross-task pretraining for HAIPipe's AIPipe policy."""
from __future__ import annotations

import os
import random

from baselines.common.pipeline import finalize_pipeline

from .agent import HAIPipeAgent
from .env import HAIPipeEnv
from .trainer import HAIPipeTrainer


class CrossTaskPretrainer:
    """Train one operator-layout checkpoint without using target tasks."""

    def __init__(
        self, environments: list[HAIPipeEnv], *, frames: int = 100000,
        learning_rate: float = 1e-5, buffer_capacity: int = 2000,
        batch_size: int = 100, gamma: float = 0.0, train_interval: int = 100,
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
        self.frames = int(frames)
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
            train_interval=train_interval,
        )

    def train(self, checkpoint_path: str, *, held_out_tasks: list[str] | None = None) -> dict:
        episode_count = 0
        while self.trainer.frames < self.frames:
            env = self.environments[episode_count % len(self.environments)]
            epsilon = self.agent.epsilon(self.trainer.frames)
            episode = env.episode(self.agent, epsilon=epsilon)
            executable, report = finalize_pipeline(
                episode["pipeline"], env.ctx, self.rng,
            )
            result = env.evaluator.evaluate(executable) if report.valid else None
            reward = float(result.fitness) if result is not None and result.success else 0.0
            for transition in episode["transitions"]:
                transition["reward"] = reward if transition["done"] else 0.0
            self.trainer.add_episode(
                episode["transitions"], episode["logical_state"],
                episode["logical_action"], reward,
            )
            episode_count += 1
        os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
        self.agent.save(checkpoint_path, provenance={
            "training_tasks": [env.evaluator.data_name for env in self.environments],
            "held_out_tasks": list(held_out_tasks or []),
            "frames": self.frames,
        })
        return {"frames": self.trainer.frames, "episodes": episode_count, "checkpoint": checkpoint_path}


__all__ = ["CrossTaskPretrainer"]


def save_alternative_evaluator(
    model, path: str, *, data_dim: int, primitive_vocabulary: list[str],
    provenance: dict | None = None,
) -> None:
    """Save the strict vocabulary-bearing checkpoint consumed by HAIPipe."""
    import torch
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({
        "metadata": {
            "data_dim": int(data_dim),
            "primitive_vocabulary": list(primitive_vocabulary),
            "sequence_length": 30,
        },
        "provenance": dict(provenance or {}),
        "model": model.state_dict(),
    }, path)


__all__.append("save_alternative_evaluator")
