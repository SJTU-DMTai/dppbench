"""Four-component CtxPipe agent aligned with upstream DQN/OCG training."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .model import DATA_DIM, DQN, ForwardMode, RnnDQN


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    component: str
    next_component: Optional[str]
    logical_pipeline_id: int
    context: np.ndarray


class ReplayBuffer:
    """Upstream-style shared FIFO buffers for component and logical samples."""

    def __init__(self, capacity: int, seed: int = 42) -> None:
        self.capacity = int(capacity)
        self.buffer = deque(maxlen=self.capacity)
        self.logical_buffer = deque(maxlen=self.capacity)
        self.rng = np.random.RandomState(seed)

    def add(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def add_logical(self, state, action, reward, context) -> None:
        self.logical_buffer.append((
            np.asarray(state, dtype=np.float32), int(action), float(reward),
            np.asarray(context, dtype=np.float32),
        ))

    def sample(self, batch_size: int) -> list[Transition]:
        indices = self.rng.choice(len(self.buffer), batch_size, replace=False)
        values = list(self.buffer)
        return [values[int(i)] for i in indices]

    def sample_logical(self, batch_size: int):
        indices = self.rng.choice(len(self.logical_buffer), batch_size, replace=False)
        values = list(self.logical_buffer)
        return [values[int(i)] for i in indices]

    def __len__(self) -> int:
        return len(self.buffer)


class AgentModel:
    def __init__(self, name: str, network, action_dim: int, lr: float, device: str) -> None:
        self.name = name
        self.action_dim = int(action_dim)
        self.network = network.to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

    def __call__(self, *args, **kwargs):
        return self.network(*args, **kwargs)


class DQNAgent:
    """Four category RnnDQNs plus the upstream logical-pipeline DQN."""

    def __init__(
        self,
        component_action_dims: dict[str, int],
        primitive_vocab_size: int,
        logical_pipeline_count: int,
        sequence_length: int = 4,
        lr: float = 1e-5,
        buffer_capacity: int = 5000,
        device: str = "cpu",
        seed: int = 42,
        **_compat,
    ) -> None:
        if len(component_action_dims) != 4:
            raise ValueError(
                f"CtxPipe requires exactly four category DQNs, got {len(component_action_dims)}"
            )
        self.device = torch.device(device)
        self.component_ids = list(component_action_dims)
        self.sequence_length = int(sequence_length)
        self.logical_pipeline_count = int(logical_pipeline_count)
        self.rng = np.random.RandomState(seed)
        torch.manual_seed(seed)
        self.buffer = ReplayBuffer(buffer_capacity, seed=seed)
        self.models: dict[str, AgentModel] = {}
        for component, action_dim in component_action_dims.items():
            network = RnnDQN(
                actions_dim=action_dim,
                primitive_vocab_size=primitive_vocab_size,
                sequence_length=self.sequence_length,
                logical_pipeline_count=self.logical_pipeline_count,
                device=device,
            )
            self.models[component] = AgentModel(component, network, action_dim, lr, device)
        self.logical_model = AgentModel(
            "LogicPipeline",
            DQN(DATA_DIM + 1, self.logical_pipeline_count, device),
            self.logical_pipeline_count,
            lr,
            device,
        )

    @staticmethod
    def _tensor(value, device, dtype=torch.float32):
        return torch.as_tensor(value, dtype=dtype, device=device)

    def _epsilon_action(
        self, model: AgentModel, state, context, epsilon, tried=None,
        context_factory=None,
    ):
        tried = set(tried or [])
        available = np.asarray([i for i in range(model.action_dim) if i not in tried])
        if not len(available):
            raise ValueError(f"no untried action remains for {model.name}")
        used_model = bool(self.rng.random_sample() > epsilon)
        if not used_model:
            return int(self.rng.choice(available)), False
        action_context = context_factory() if context_factory is not None else context
        with torch.no_grad():
            q_values = model(
                self._tensor(state, self.device).unsqueeze(0),
                self._tensor(action_context, self.device).unsqueeze(0),
                ForwardMode.GATED,
            ).squeeze(0).cpu().numpy()
        masked = np.full(model.action_dim, -np.inf, dtype=np.float32)
        masked[available] = q_values[available]
        return int(np.argmax(masked)), True

    def select_logical(self, state, context, epsilon: float, context_factory=None):
        return self._epsilon_action(
            self.logical_model, state, context, epsilon,
            context_factory=context_factory,
        )

    def select_component(
        self, component, state, context, epsilon: float, tried=None,
        context_factory=None,
    ):
        return self._epsilon_action(
            self.models[component], state, context, epsilon, tried,
            context_factory=context_factory,
        )

    def add_transition(self, **kwargs) -> None:
        self.buffer.add(Transition(**kwargs))

    def add_logical_transition(self, state, action, reward, context) -> None:
        self.buffer.add_logical(state, action, reward, context)

    def _component_loss(self, samples: list[Transition], component: str, mode: ForwardMode):
        selected = [sample for sample in samples if sample.component == component]
        if not selected:
            return None
        model = self.models[component]
        states = self._tensor(np.stack([s.state for s in selected]), self.device)
        actions = self._tensor([s.action for s in selected], self.device, torch.long)
        rewards = self._tensor([s.reward for s in selected], self.device)
        contexts = self._tensor(np.stack([s.context for s in selected]), self.device)
        q_value = model(states, contexts, mode).gather(1, actions[:, None]).squeeze(1)
        next_values = []
        for sample in selected:
            if sample.done or sample.next_component is None:
                next_values.append(0.0)
                continue
            next_model = self.models[sample.next_component]
            next_q = next_model(
                self._tensor(sample.next_state, self.device).unsqueeze(0),
                self._tensor(sample.context, self.device).unsqueeze(0),
                mode,
            )
            next_values.append(float(next_q.max(1)[0].detach().item()))
        expected = rewards + self._tensor(next_values, self.device)
        return F.mse_loss(q_value, expected.detach())

    def learn_components(self, batch_size: int) -> dict[str, float]:
        if len(self.buffer) < batch_size:
            return {}
        samples = self.buffer.sample(batch_size)
        losses: dict[str, float] = {}
        for mode in ForwardMode:
            for component, model in self.models.items():
                loss = self._component_loss(samples, component, mode)
                if loss is None or torch.isnan(loss):
                    continue
                model.optimizer.zero_grad()
                loss.backward()
                probe = list(model.network.parameters())[-1]
                before = probe.detach().clone()
                model.optimizer.step()
                self.component_updates = getattr(self, 'component_updates', 0) + 1
                self.component_parameter_changes = getattr(self, 'component_parameter_changes', 0) + int(not torch.equal(before, probe.detach()))
                if mode == ForwardMode.GATED:
                    losses[component] = float(loss.item())
        return losses

    def learn_logical(self, batch_size: int) -> float | None:
        if len(self.buffer.logical_buffer) < batch_size:
            return None
        samples = self.buffer.sample_logical(batch_size)
        states = self._tensor(np.stack([s[0] for s in samples]), self.device)
        actions = self._tensor([s[1] for s in samples], self.device, torch.long)
        rewards = self._tensor([s[2] for s in samples], self.device)
        contexts = self._tensor(np.stack([s[3] for s in samples]), self.device)
        gated_loss = None
        for mode in ForwardMode:
            q_value = self.logical_model(states, contexts, mode).gather(
                1, actions[:, None],
            ).squeeze(1)
            loss = F.mse_loss(q_value, rewards.detach())
            if torch.isnan(loss):
                continue
            self.logical_model.optimizer.zero_grad()
            loss.backward()
            probe = list(self.logical_model.network.parameters())[-1]
            before = probe.detach().clone()
            self.logical_model.optimizer.step()
            self.logical_updates = getattr(self, 'logical_updates', 0) + 1
            self.logical_parameter_changes = getattr(self, 'logical_parameter_changes', 0) + int(not torch.equal(before, probe.detach()))
            if mode == ForwardMode.GATED:
                gated_loss = float(loss.item())
        return gated_loss

    def save(self, path: str) -> None:
        torch.save({
            "component_models": {
                name: model.network.state_dict() for name, model in self.models.items()
            },
            "logical_model": self.logical_model.network.state_dict(),
            "component_ids": self.component_ids,
            "logical_pipeline_count": self.logical_pipeline_count,
        }, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        if checkpoint["component_ids"] != self.component_ids:
            raise ValueError("checkpoint category layout does not match")
        for name, state in checkpoint["component_models"].items():
            self.models[name].network.load_state_dict(state)
        self.logical_model.network.load_state_dict(checkpoint["logical_model"])


__all__ = ["DQNAgent", "ReplayBuffer", "Transition"]
