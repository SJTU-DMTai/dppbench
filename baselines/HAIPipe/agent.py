"""HAIPipe-specific DQN agent and checkpoint contract."""
from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .model import LogicalDQN, PrimitiveRnnDQN


@dataclass
class Transition:
    component: str
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    next_component: str | None = None


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int) -> None:
        self.items = deque(maxlen=int(capacity))
        self.logical_items = deque(maxlen=int(capacity))
        self.rng = random.Random(seed)

    def add(self, transition: Transition) -> None:
        self.items.append(transition)

    def add_logical(self, state: np.ndarray, action: int, reward: float) -> None:
        self.logical_items.append((state, int(action), float(reward)))

    def sample(self, size: int) -> list[Transition]:
        return self.rng.sample(list(self.items), min(int(size), len(self.items)))


class HAIPipeAgent:
    FORMAT_VERSION = 2

    def __init__(
        self,
        *,
        component_actions: dict[str, list[str]],
        logical_pipelines: list[list[str]],
        primitive_vocabulary: list[str],
        model_vocabulary: list[str],
        data_dim: int,
        learning_rate: float = 1e-5,
        buffer_capacity: int = 2000,
        seed: int = 42,
        device: str = "cpu",
    ) -> None:
        self.component_actions = {k: list(v) for k, v in component_actions.items()}
        self.logical_pipelines = [list(v) for v in logical_pipelines]
        self.primitive_vocabulary = list(primitive_vocabulary)
        self.model_vocabulary = list(model_vocabulary) or ["model"]
        self.data_dim = int(data_dim)
        self.sequence_length = len(self.component_actions)
        self.device = torch.device(device)
        self.rng = random.Random(seed)
        torch.manual_seed(seed)
        self.buffer = ReplayBuffer(buffer_capacity, seed)
        self.logical_model = LogicalDQN(data_dim + 1, len(logical_pipelines)).to(self.device)
        self.logical_optimizer = torch.optim.Adam(self.logical_model.parameters(), lr=learning_rate)
        self.models = {}
        self.optimizers = {}
        for component, actions in self.component_actions.items():
            model = PrimitiveRnnDQN(
                data_dim=data_dim,
                sequence_length=self.sequence_length,
                primitive_vocab_size=len(self.primitive_vocabulary) + 2,
                predictor_vocab_size=len(self.model_vocabulary),
                logical_pipeline_count=len(self.logical_pipelines),
                action_dim=len(actions) + 1,
            ).to(self.device)
            self.models[component] = model
            self.optimizers[component] = torch.optim.Adam(model.parameters(), lr=learning_rate)

    @staticmethod
    def epsilon(frame: int, start: float = 1.0, end: float = 0.4, decay: float = 500.0) -> float:
        return end + (start - end) * math.exp(-float(frame) / float(decay))

    def _choose(self, values: torch.Tensor, epsilon: float, tried: set[int]) -> int:
        available = [i for i in range(values.numel()) if i not in tried]
        if not available:
            raise ValueError("no untried HAIPipe action remains")
        if self.rng.random() < epsilon:
            return self.rng.choice(available)
        scores = values.detach().cpu().numpy()
        return max(available, key=lambda i: float(scores[i]))

    def select_logical(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        was_training = self.logical_model.training
        self.logical_model.eval()
        with torch.no_grad():
            values = self.logical_model(torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0))[0]
        self.logical_model.train(was_training)
        return self._choose(values, epsilon, set())

    def select_component(self, component: str, state: np.ndarray, epsilon: float = 0.0, tried=None) -> int:
        model = self.models[component]
        was_training = model.training
        model.eval()
        with torch.no_grad():
            values = model(torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0))[0]
        model.train(was_training)
        return self._choose(values, epsilon, set(tried or []))

    def add_transition(self, transition: Transition) -> None:
        self.buffer.add(transition)

    def learn(self, batch_size: int = 100, gamma: float = 1.0, logic_batch_size: int = 20) -> dict[str, float]:
        samples = self.buffer.sample(batch_size)
        losses = {}
        for component, model in self.models.items():
            selected = [s for s in samples if s.component == component]
            if not selected:
                continue
            states = torch.as_tensor(np.stack([s.state for s in selected]), dtype=torch.float32, device=self.device)
            actions = torch.as_tensor([s.action for s in selected], dtype=torch.long, device=self.device)
            rewards = torch.as_tensor([s.reward for s in selected], dtype=torch.float32, device=self.device)
            current = model(states).gather(1, actions[:, None]).squeeze(1)
            next_states = torch.as_tensor(np.stack([s.next_state for s in selected]), dtype=torch.float32, device=self.device)
            with torch.no_grad():
                future = torch.zeros_like(rewards)
                # Each component has a different action space. Bootstrap from
                # the next component in this episode's logical pipeline.
                for next_component in sorted({s.next_component for s in selected if not s.done}):
                    ids = [i for i, s in enumerate(selected) if not s.done and s.next_component == next_component]
                    future[ids] = self.models[next_component](next_states[ids]).max(1).values
                expected = rewards + float(gamma) * future
            loss = F.mse_loss(current, expected)
            self.optimizers[component].zero_grad()
            loss.backward()
            self.optimizers[component].step()
            losses[component] = float(loss.item())
        if self.buffer.logical_items:
            logical = self.rng.sample(
                list(self.buffer.logical_items),
                min(int(logic_batch_size), len(self.buffer.logical_items)),
            )
            states = torch.as_tensor(
                np.stack([item[0] for item in logical]),
                dtype=torch.float32,
                device=self.device,
            )
            actions = torch.as_tensor(
                [item[1] for item in logical], dtype=torch.long, device=self.device,
            )
            rewards = torch.as_tensor(
                [item[2] for item in logical], dtype=torch.float32, device=self.device,
            )
            values = self.logical_model(states).gather(1, actions[:, None]).squeeze(1)
            logical_loss = F.mse_loss(values, rewards)
            self.logical_optimizer.zero_grad()
            logical_loss.backward()
            self.logical_optimizer.step()
            losses["LogicPipeline"] = float(logical_loss.item())
        return losses

    def metadata(self) -> dict:
        return {
            "format_version": self.FORMAT_VERSION,
            "component_actions": self.component_actions,
            "logical_pipelines": self.logical_pipelines,
            "primitive_vocabulary": self.primitive_vocabulary,
            "model_vocabulary": self.model_vocabulary,
            "data_dim": self.data_dim,
            "state_encoding": "dynamic_predictors_blank0_padding_last",
            "sequence_length": self.sequence_length,
        }

    def save(self, path: str, provenance: Optional[dict] = None) -> None:
        torch.save({
            "metadata": self.metadata(),
            "provenance": dict(provenance or {}),
            "logical_model": self.logical_model.state_dict(),
            "component_models": {k: v.state_dict() for k, v in self.models.items()},
        }, path)

    def load(self, path: str) -> dict:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        if checkpoint.get("metadata") != self.metadata():
            raise ValueError("HAIPipe checkpoint is incompatible with the current DPPBench search space")
        self.logical_model.load_state_dict(checkpoint["logical_model"])
        for component, model in self.models.items():
            model.load_state_dict(checkpoint["component_models"][component])
        return dict(checkpoint.get("provenance") or {})


__all__ = ["HAIPipeAgent", "ReplayBuffer", "Transition"]
