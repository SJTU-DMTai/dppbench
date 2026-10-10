"""AIPipe replay trainer for target-task and optional cross-task training."""
from __future__ import annotations

from .agent import Transition


class HAIPipeTrainer:
    def __init__(self, agent, *, batch_size: int = 100, gamma: float = 1.0, train_interval: int = 100, logic_batch_size: int = 20) -> None:
        self.agent = agent
        self.batch_size = int(batch_size)
        self.gamma = float(gamma)
        self.train_interval = int(train_interval)
        self.logic_batch_size = int(logic_batch_size)
        self.frames = 0

    def add_episode(self, component_transitions, logical_state, logical_action: int, reward: float) -> dict:
        self.agent.buffer.add_logical(logical_state, logical_action, reward)
        losses = {}
        for item in component_transitions:
            self.agent.add_transition(Transition(**item))
            self.frames += 1
            if len(self.agent.buffer.items) > self.batch_size and self.frames % self.train_interval == 0:
                losses = self.agent.learn(self.batch_size, self.gamma, self.logic_batch_size)
        return losses


__all__ = ["HAIPipeTrainer"]
