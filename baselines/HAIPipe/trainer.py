"""Cross-task AIPipe pretraining utilities.

Target-task HAIPipe inference never calls this module.  A pretrainer feeds
successful task episodes into the public HAIPipe replay semantics and saves a
frozen checkpoint for later inference.
"""
from __future__ import annotations

from .agent import Transition


class HAIPipeTrainer:
    def __init__(self, agent, *, batch_size: int = 100, gamma: float = 0.0, train_interval: int = 100) -> None:
        self.agent = agent
        self.batch_size = int(batch_size)
        self.gamma = float(gamma)
        self.train_interval = int(train_interval)
        self.frames = 0

    def add_episode(self, component_transitions, logical_state, logical_action: int, reward: float) -> dict:
        for item in component_transitions:
            self.agent.add_transition(Transition(**item))
            self.frames += 1
        self.agent.buffer.add_logical(logical_state, logical_action, reward)
        if self.frames and self.frames % self.train_interval == 0:
            return self.agent.learn(self.batch_size, self.gamma)
        return {}


__all__ = ["HAIPipeTrainer"]
