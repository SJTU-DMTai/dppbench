"""DQN agent for CtxPipe.

Implements a compact DQN with:
  * 3-layer MLP Q-network
  * fixed-size replay buffer
  * ε-greedy action selection
  * MSE Bellman loss with a target network
"""
from __future__ import annotations

import random as _random
from collections import deque
from typing import Deque, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden: int = 128) -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, capacity: int = 10_000, seed: int = 42) -> None:
        self._buf: Deque[Transition] = deque(maxlen=capacity)
        self._rng = _random.Random(seed)

    def push(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, done: bool) -> None:
        self._buf.append((np.asarray(s, dtype=np.float32), int(a), float(r),
                          np.asarray(s2, dtype=np.float32), bool(done)))

    def sample(self, batch_size: int) -> tuple:
        batch = self._rng.sample(self._buf, batch_size)
        s = np.stack([t[0] for t in batch])
        a = np.array([t[1] for t in batch], dtype=np.int64)
        r = np.array([t[2] for t in batch], dtype=np.float32)
        s2 = np.stack([t[3] for t in batch])
        d = np.array([t[4] for t in batch], dtype=np.float32)
        return s, a, r, s2, d

    def __len__(self) -> int:
        return len(self._buf)


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden: int = 128,
        lr: float = 1e-3,
        buffer_capacity: int = 10_000,
        device: str = "cpu",
        seed: int = 42,
    ) -> None:
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.device = torch.device(device)

        torch.manual_seed(seed)
        np.random.seed(seed)
        self._np_rng = np.random.RandomState(seed)

        self.q_net = QNetwork(state_dim, n_actions, hidden).to(self.device)
        self.target_net = QNetwork(state_dim, n_actions, hidden).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        for p in self.target_net.parameters():
            p.requires_grad_(False)

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(capacity=buffer_capacity, seed=seed)

    # ------------------------------------------------------------------
    def select_action(self, state: np.ndarray, eps: float) -> int:
        if self._np_rng.rand() < eps:
            return int(self._np_rng.randint(self.n_actions))
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32,
                                device=self.device).unsqueeze(0)
            q = self.q_net(s).cpu().numpy()[0]
        return int(np.argmax(q))

    # ------------------------------------------------------------------
    def train_step(self, batch_size: int, gamma: float) -> float:
        if len(self.buffer) < batch_size:
            return 0.0
        s, a, r, s2, d = self.buffer.sample(batch_size)
        s_t = torch.as_tensor(s, device=self.device)
        a_t = torch.as_tensor(a, device=self.device)
        r_t = torch.as_tensor(r, device=self.device)
        s2_t = torch.as_tensor(s2, device=self.device)
        d_t = torch.as_tensor(d, device=self.device)

        q_pred = self.q_net(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self.target_net(s2_t).max(dim=1).values
            q_target = r_t + gamma * q_next * (1.0 - d_t)
        loss = F.mse_loss(q_pred, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        return float(loss.item())

    # ------------------------------------------------------------------
    def update_target(self) -> None:
        self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, path: str) -> None:
        torch.save({"q_net": self.q_net.state_dict(),
                    "state_dim": self.state_dim,
                    "n_actions": self.n_actions}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["q_net"])
