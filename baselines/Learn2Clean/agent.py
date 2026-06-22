"""Tabular Q-learning agent with Boltzmann (softmax) exploration.

Implements the original Learn2Clean algorithm exactly:

* dict-based Q-table keyed by ``(frozenset(applied_op_idx), last_op_idx)``;
* Boltzmann policy ``P(a|s) = exp(Q(s,a)/T) / Sigma_j exp(Q(s,a_j)/T)`` with
  numerical-overflow guard (subtract the max);
* tabular update ``Q(s,a) <- Q(s,a) + lr * (r + gamma * max_a' Q(s',a') - Q(s,a))``.
"""
from __future__ import annotations

import json
import os
import random as _random
from typing import Dict, Tuple

import numpy as np

from .env import StateKey


class TabularQAgent:
    def __init__(
        self,
        n_actions: int,
        gamma: float = 0.9,
        lr: float = 0.5,
        temperature_init: float = 2.0,
        temperature_final: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.n_actions = int(n_actions)
        self.gamma = float(gamma)
        self.lr = float(lr)
        self.temperature_init = float(temperature_init)
        self.temperature_final = float(temperature_final)
        self._rng = _random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

        self.Q: Dict[StateKey, np.ndarray] = {}

    # ------------------------------------------------------------------
    def _q_row(self, state: StateKey) -> np.ndarray:
        row = self.Q.get(state)
        if row is None:
            row = np.zeros(self.n_actions, dtype=np.float64)
            self.Q[state] = row
        return row

    # ------------------------------------------------------------------
    def select_action(self, state: StateKey, temperature: float) -> int:
        row = self._q_row(state)
        T = max(float(temperature), 1e-6)
        scaled = row / T
        # Softmax with overflow guard.
        scaled = scaled - scaled.max()
        weights = np.exp(scaled)
        s = weights.sum()
        if not np.isfinite(s) or s <= 0.0:
            # Fallback: uniform random.
            return int(self._rng.randrange(self.n_actions))
        probs = weights / s
        return int(self._np_rng.choice(self.n_actions, p=probs))

    # ------------------------------------------------------------------
    def update(
        self,
        state: StateKey,
        action: int,
        reward: float,
        next_state: StateKey,
        done: bool,
    ) -> None:
        row = self._q_row(state)
        if done:
            target = float(reward)
        else:
            next_row = self._q_row(next_state)
            target = float(reward) + self.gamma * float(np.max(next_row))
        row[action] += self.lr * (target - row[action])

    def greedy(self, state: StateKey) -> int:
        row = self._q_row(state)
        return int(np.argmax(row))

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Serialise with string keys: state -> "<sorted_idx_csv>|<last_idx>".
        payload = {
            "n_actions": self.n_actions,
            "gamma": self.gamma,
            "lr": self.lr,
            "temperature_init": self.temperature_init,
            "temperature_final": self.temperature_final,
            "Q": {
                self._encode_key(state): row.tolist()
                for state, row in self.Q.items()
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @staticmethod
    def _encode_key(state: StateKey) -> str:
        applied, last = state
        applied_csv = ",".join(str(i) for i in sorted(applied))
        return f"{applied_csv}|{last}"

    @staticmethod
    def _decode_key(s: str) -> StateKey:
        applied_csv, last = s.split("|")
        applied = (
            frozenset(int(x) for x in applied_csv.split(",") if x != "")
            if applied_csv
            else frozenset()
        )
        return (applied, int(last))

    @classmethod
    def load(cls, path: str) -> "TabularQAgent":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        agent = cls(
            n_actions=int(payload["n_actions"]),
            gamma=float(payload.get("gamma", 0.9)),
            lr=float(payload.get("lr", 0.5)),
            temperature_init=float(payload.get("temperature_init", 2.0)),
            temperature_final=float(payload.get("temperature_final", 0.1)),
        )
        for s, row in payload.get("Q", {}).items():
            agent.Q[cls._decode_key(s)] = np.asarray(row, dtype=np.float64)
        return agent

    # ------------------------------------------------------------------
    @property
    def n_states(self) -> int:
        return len(self.Q)


__all__ = ["TabularQAgent"]
