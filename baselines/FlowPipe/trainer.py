"""Offline cross-task FlowPipe trainer following the official combo schedule."""
from __future__ import annotations

import math
import os
import random


class FlowPipeTrainer:
    """Train one shared agent over disjoint source-task environments.

    Each combo is ``(environment, frozen_context_embedding)``. This keeps the
    official offline/amortized training semantics while allowing DPPBench's
    task loaders to construct the environments.
    """

    def __init__(
        self, agent, combos, *, output_dir: str, combos_per_epoch: int = 10,
        trajectories_per_combo: int = 8, updates_per_combo: int = 1,
        trajectory_batch_size: int = 16, checkpoint_interval: int = 20,
        seed: int = 42,
    ) -> None:
        self.agent = agent
        self.combos = list(combos)
        self.output_dir = os.path.abspath(output_dir)
        self.combos_per_epoch = int(combos_per_epoch)
        self.trajectories_per_combo = int(trajectories_per_combo)
        self.updates_per_combo = int(updates_per_combo)
        self.trajectory_batch_size = int(trajectory_batch_size)
        self.checkpoint_interval = int(checkpoint_interval)
        self.rng = random.Random(seed)
        os.makedirs(self.output_dir, exist_ok=True)
        self.candidates = []
        self.history = []
        self.optimizer_updates = 0

    def train(self, num_epochs: int | None = None) -> list[dict]:
        if not self.combos:
            raise ValueError("FlowPipe offline training requires at least one source-task combo")
        shuffled = list(self.combos)
        self.rng.shuffle(shuffled)
        coverage_epochs = math.ceil(len(shuffled) / max(1, self.combos_per_epoch))
        epochs = max(int(num_epochs or coverage_epochs), coverage_epochs)
        history = self.history
        combo_counter = 0
        for epoch in range(epochs):
            if epoch < coverage_epochs:
                start = epoch * self.combos_per_epoch
                active = shuffled[start:start + self.combos_per_epoch]
            else:
                active = self.rng.sample(shuffled, min(self.combos_per_epoch, len(shuffled)))
            rewards, losses = [], []
            for env, context in active:
                trajectory_pool = []
                self.agent.set_context(context)
                for _ in range(self.trajectories_per_combo):
                    env.reset()
                    reward, trajectory = self.agent.sample_trajectory(env)
                    rewards.append(reward)
                    trajectory_pool.append(trajectory)
                    self.candidates.append({
                        'reward': float(reward), 'pipeline': env.current_pipeline().copy(),
                        'actions': list(trajectory['actions']), 'errors': list(trajectory['errors']),
                        'logic_pipeline_id': env.logic_id,
                        'success': trajectory.get('success', False),
                        'fitness': trajectory.get('fitness'), 'metrics': trajectory.get('metrics', {}),
                        'error_messages': trajectory.get('error_messages', []),
                    })
                update_losses = self.agent.learn_from_batch(
                    trajectory_pool, self.updates_per_combo, self.trajectory_batch_size,
                )
                losses.extend(update_losses)
                self.optimizer_updates += len(update_losses)
                combo_counter += 1
                if combo_counter % self.checkpoint_interval == 0:
                    self.agent.save(os.path.join(self.output_dir, f"combo_{combo_counter}_gfn.pt"))
            history.append({
                "epoch": epoch + 1,
                "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
                "mean_loss": sum(losses) / len(losses) if losses else None,
                "combo_counter": combo_counter,
            })
        self.agent.save(os.path.join(self.output_dir, "gfn.pt"))
        return history


__all__ = ["FlowPipeTrainer"]
