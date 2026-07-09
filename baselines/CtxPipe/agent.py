"""Component-level DQN agents for CtxPipe.

Original CtxPipe trains component-specific policies with a gated context
plug-in and OCG-style replay.  This module keeps one Q-network and replay
buffer per component slot, gates hidden state with the context prefix, and
samples from success/failure/recent replay partitions.
"""
from __future__ import annotations

import random as _random
from collections import deque
from typing import Deque, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


Transition = tuple[np.ndarray, int, float, np.ndarray, bool, Optional[str], np.ndarray]


class QNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden: int = 128,
        context_dim: int = 32,
    ) -> None:
        super().__init__()
        self.context_dim = max(0, min(int(context_dim), int(state_dim)))
        rest_dim = int(state_dim) - self.context_dim
        self.ctx_proj = nn.Linear(self.context_dim, hidden) if self.context_dim else None
        self.ctx_gate = nn.Linear(self.context_dim, hidden) if self.context_dim else None
        self.rest_proj = nn.Linear(rest_dim if rest_dim > 0 else state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.context_dim:
            ctx = x[:, : self.context_dim]
            rest = x[:, self.context_dim :]
            if rest.shape[1] == 0:
                rest = x
            h = F.relu(self.rest_proj(rest))
            gate = torch.sigmoid(self.ctx_gate(ctx))
            ctx_h = F.relu(self.ctx_proj(ctx))
            h = h * gate + ctx_h
        else:
            h = F.relu(self.rest_proj(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h)


class ReplayBuffer:
    def __init__(self, capacity: int = 10_000, seed: int = 42) -> None:
        self._all: Deque[Transition] = deque(maxlen=capacity)
        self._open: Deque[Transition] = deque(maxlen=capacity)
        self._closed: Deque[Transition] = deque(maxlen=capacity)
        self._rng = _random.Random(seed)

    def push(
        self,
        s: np.ndarray,
        a: int,
        r: float,
        s2: np.ndarray,
        done: bool,
        next_component: Optional[str],
        next_mask: np.ndarray,
        episode_success: bool,
    ) -> None:
        transition = (
            np.asarray(s, dtype=np.float32),
            int(a),
            float(r),
            np.asarray(s2, dtype=np.float32),
            bool(done),
            next_component,
            np.asarray(next_mask, dtype=bool),
        )
        self._all.append(transition)
        if episode_success:
            self._open.append(transition)
        else:
            self._closed.append(transition)

    def _draw(self, source: Deque[Transition], n: int) -> list[Transition]:
        if n <= 0 or not source:
            return []
        n = min(n, len(source))
        return self._rng.sample(list(source), n)

    def sample(self, batch_size: int) -> list[Transition]:
        # OCG-style balance: successful (open), failed (closed), and recent/all
        # experiences are mixed whenever available.
        n_open = batch_size // 3
        n_closed = batch_size // 3
        out = self._draw(self._open, n_open)
        out += self._draw(self._closed, n_closed)
        out += self._draw(self._all, batch_size - len(out))
        if len(out) < batch_size:
            out += self._draw(self._all, batch_size - len(out))
        self._rng.shuffle(out)
        return out[:batch_size]

    def __len__(self) -> int:
        return len(self._all)


class DQNAgent:
    """A bank of component-specific DQN policies."""

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        component_ids: list[str],
        hidden: int = 128,
        lr: float = 1e-3,
        buffer_capacity: int = 10_000,
        device: str = "cpu",
        seed: int = 42,
        context_dim: int = 32,
    ) -> None:
        self.state_dim = int(state_dim)
        self.n_actions = int(n_actions)
        self.context_dim = int(context_dim)
        self.component_ids = list(component_ids)
        if not self.component_ids:
            raise ValueError("CtxPipe requires at least one component policy")
        self.device = torch.device(device)

        torch.manual_seed(seed)
        np.random.seed(seed)
        self._np_rng = np.random.RandomState(seed)

        self._key_by_component = {
            component_id: f"c{i}" for i, component_id in enumerate(self.component_ids)
        }
        self._component_by_key = {v: k for k, v in self._key_by_component.items()}

        self.q_nets = nn.ModuleDict({
            self._key_by_component[c]: QNetwork(
                self.state_dim, self.n_actions, hidden, self.context_dim
            )
            for c in self.component_ids
        }).to(self.device)
        self.target_nets = nn.ModuleDict({
            self._key_by_component[c]: QNetwork(
                self.state_dim, self.n_actions, hidden, self.context_dim
            )
            for c in self.component_ids
        }).to(self.device)
        self.target_nets.load_state_dict(self.q_nets.state_dict())
        for p in self.target_nets.parameters():
            p.requires_grad_(False)

        self.optimizer = torch.optim.Adam(self.q_nets.parameters(), lr=lr)
        self.buffers = {
            c: ReplayBuffer(capacity=buffer_capacity, seed=seed + i)
            for i, c in enumerate(self.component_ids)
        }

    # ------------------------------------------------------------------
    def _net_key(self, component_id: str) -> str:
        if component_id not in self._key_by_component:
            raise KeyError(f"Unknown CtxPipe component: {component_id}")
        return self._key_by_component[component_id]

    def select_action(
        self,
        component_id: str,
        state: np.ndarray,
        eps: float,
        action_mask: np.ndarray,
    ) -> int:
        mask = np.asarray(action_mask, dtype=bool)
        valid = np.flatnonzero(mask)
        if len(valid) == 0:
            raise ValueError(f"No valid actions for component {component_id}")
        if self._np_rng.rand() < eps:
            return int(self._np_rng.choice(valid))
        key = self._net_key(component_id)
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.q_nets[key](s).cpu().numpy()[0]
        q = q.copy()
        q[~mask] = -np.inf
        return int(np.argmax(q))

    # ------------------------------------------------------------------
    def push(
        self,
        component_id: str,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_component: Optional[str],
        next_mask: np.ndarray,
        episode_success: bool,
    ) -> None:
        self.buffers[component_id].push(
            state,
            action,
            reward,
            next_state,
            done,
            next_component,
            next_mask,
            episode_success,
        )

    def buffer_size(self, component_id: Optional[str] = None) -> int:
        if component_id is not None:
            return len(self.buffers[component_id])
        return sum(len(buf) for buf in self.buffers.values())

    def train_step(self, component_id: str, batch_size: int, gamma: float) -> float:
        buf = self.buffers[component_id]
        if len(buf) < batch_size:
            return 0.0
        batch = buf.sample(batch_size)
        s = np.stack([t[0] for t in batch])
        a = np.array([t[1] for t in batch], dtype=np.int64)
        r = np.array([t[2] for t in batch], dtype=np.float32)
        s2 = np.stack([t[3] for t in batch])
        d = np.array([t[4] for t in batch], dtype=np.float32)
        next_components = [t[5] for t in batch]
        next_masks = [t[6] for t in batch]

        key = self._net_key(component_id)
        s_t = torch.as_tensor(s, device=self.device)
        a_t = torch.as_tensor(a, device=self.device)
        r_t = torch.as_tensor(r, device=self.device)
        d_t = torch.as_tensor(d, device=self.device)

        q_pred = self.q_nets[key](s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next_vals: list[float] = []
            for idx, next_component in enumerate(next_components):
                if d[idx] or next_component is None:
                    q_next_vals.append(0.0)
                    continue
                next_key = self._net_key(next_component)
                ns = torch.as_tensor(
                    s2[idx], dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                q_next = self.target_nets[next_key](ns).squeeze(0)
                mask_t = torch.as_tensor(next_masks[idx], dtype=torch.bool, device=self.device)
                if mask_t.any():
                    q_next_vals.append(float(q_next.masked_fill(~mask_t, -torch.inf).max().item()))
                else:
                    q_next_vals.append(0.0)
            q_next_t = torch.as_tensor(q_next_vals, dtype=torch.float32, device=self.device)
            q_target = r_t + gamma * q_next_t * (1.0 - d_t)

        loss = F.mse_loss(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_nets.parameters(), max_norm=10.0)
        self.optimizer.step()
        return float(loss.item())

    # ------------------------------------------------------------------
    def update_target(self) -> None:
        self.target_nets.load_state_dict(self.q_nets.state_dict())

    def save(self, path: str) -> None:
        torch.save({
            "q_nets": self.q_nets.state_dict(),
            "state_dim": self.state_dim,
            "n_actions": self.n_actions,
            "context_dim": self.context_dim,
            "component_ids": self.component_ids,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        if ckpt.get("component_ids") != self.component_ids:
            raise ValueError("Checkpoint component layout does not match this env")
        self.q_nets.load_state_dict(ckpt["q_nets"])
        self.target_nets.load_state_dict(ckpt["q_nets"])
