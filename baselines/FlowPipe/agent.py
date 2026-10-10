"""Official FlowPipe trajectory sampling and failure-aware TB learning."""
from __future__ import annotations

import math
import random
from collections import deque

import numpy as np
import torch

from .model import FlowNetwork


class TrajectoryBuffer:
    def __init__(self, capacity: int) -> None:
        self.data = deque(maxlen=int(capacity))

    def extend(self, trajectories) -> None:
        self.data.extend(trajectories)

    def sample(self, size: int):
        return random.sample(list(self.data), min(int(size), len(self.data)))


class GFNAgent:
    def __init__(
        self, state_dim: int, action_dim: int, *, hidden_dim: int = 512,
        ctx_dim: int = 4096, learning_rate: float = 5e-4,
        buffer_capacity: int = 80000, failure_weight: float = 0.2,
        reward_temperature: float = 2.0, device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.ctx_dim = int(ctx_dim)
        self.model = FlowNetwork(state_dim, action_dim, hidden_dim, ctx_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(learning_rate))
        self.buffer = TrajectoryBuffer(buffer_capacity)
        self.failure_weight = float(failure_weight)
        self.reward_temperature = float(reward_temperature)
        self.context: torch.Tensor | None = None
        self.action_dim = int(action_dim)

    def set_context(self, context: torch.Tensor) -> None:
        context = torch.as_tensor(context, dtype=torch.float32, device=self.device).reshape(1, -1)
        if context.shape[1] != self.ctx_dim:
            raise ValueError(f"expected context dim {self.ctx_dim}, got {context.shape[1]}")
        self.context = context.detach()

    def encode_state(self, state: dict, n_logic: int, n_stages: int) -> torch.Tensor:
        logic_flag = np.asarray([float(state["logic_pipeline_selected"])], dtype=np.float32)
        logic = np.zeros(n_logic, dtype=np.float32)
        if state["logic_pipeline_selected"]:
            logic[state["logic_pipeline_id"]] = 1.0
        stage = np.zeros(n_stages, dtype=np.float32)
        if 0 <= state["stage_index"] < n_stages:
            stage[state["stage_index"]] = 1.0
        predictor = np.asarray([float(state["predictor_id"])], dtype=np.float32)
        gids = np.zeros(self.action_dim, dtype=np.float32)
        for gid in state["selected_gids"]:
            if 0 <= gid < self.action_dim:
                gids[gid] = 1.0
        flat = np.concatenate([logic_flag, logic, stage, predictor, gids, state["data_feature"]])
        flat = np.nan_to_num(flat, nan=0.0, posinf=1e6, neginf=-1e6)
        return torch.tensor(flat, dtype=torch.float32, device=self.device)

    @staticmethod
    def apply_mask(logits: torch.Tensor, mask: np.ndarray) -> torch.Tensor:
        mask_tensor = torch.as_tensor(mask, dtype=logits.dtype, device=logits.device)
        return torch.where(mask_tensor > 0, logits, torch.full_like(logits, float("-inf")))

    def sample_trajectory(self, env, *, greedy_rank: int | None = None):
        if self.context is None:
            raise RuntimeError("FlowPipe context is not set")
        states, actions, masks, errors = [], [], [], []
        error_messages = []
        info = {}
        state = env.get_state_repr()
        done, reward = False, 1e-6
        while not done and len(actions) < 1 + env.max_steps:
            encoded = self.encode_state(state, env.n_logic, len(env.categories))
            logits, _ = self.model(encoded.unsqueeze(0), self.context)
            logits = self.apply_mask(logits.squeeze(0), env.get_action_mask())
            if torch.isinf(logits).all():
                errors.append(-1)
                error_messages.append('No feasible action remains under the mandatory contract and step budget')
                break
            if greedy_rank is None:
                action = torch.distributions.Categorical(logits=logits).sample()
            else:
                legal_count = int(torch.isfinite(logits).sum().item())
                rank = min(int(greedy_rank), max(0, legal_count - 1))
                action = torch.topk(logits, k=rank + 1).indices[rank]
            masks.append(env.get_action_mask().copy())
            states.append(encoded.detach().cpu())
            actions.append(int(action.item()))
            state, reward, done, info = env.step(int(action.item()))
            errors.append(int(info.get("err_code", 0)))
            if info.get('error'):
                error_messages.append(info['error'])
            if errors[-1] != 0:
                reward = 1e-6
                break
        raw_reward = max(float(reward or 0.0), 1e-6)
        reward = raw_reward ** (1.0 / max(self.reward_temperature, 1e-2))
        trajectory = {
            "states": states, "actions": actions, "masks": masks,
            "errors": errors, "reward": max(reward, 1e-6),
            "ctx": self.context.detach().cpu().squeeze(0),
            "error_messages": error_messages,
            "fitness": info.get('fitness'), "metrics": info.get('metrics') or {},
            "success": bool(done and info.get('fitness') is not None and not any(errors)),
        }
        return max(reward, 1e-6), trajectory

    def learn_from_batch(self, trajectories, updates: int = 1, batch_size: int = 16):
        if not trajectories:
            return []
        self.buffer.extend(trajectories)
        losses = []
        for _ in range(int(updates)):
            batch = self.buffer.sample(batch_size)
            trajectory_losses = []
            for trajectory in batch:
                if not trajectory["states"]:
                    continue
                states = torch.stack(trajectory["states"]).to(self.device)
                ctx = trajectory["ctx"].to(self.device).reshape(1, -1).expand(len(states), -1)
                logits, log_z = self.model(states, ctx)
                masks = torch.as_tensor(trajectory["masks"], dtype=logits.dtype, device=self.device)
                logits = torch.where(masks > 0, logits, torch.full_like(logits, float("-inf")))
                actions = torch.as_tensor(trajectory["actions"], dtype=torch.long, device=self.device)
                rows = torch.arange(len(actions), device=self.device)
                sum_log_pf = torch.log_softmax(logits, dim=-1)[rows, actions].sum()
                log_reward = torch.log(torch.tensor(trajectory["reward"], device=self.device).clamp(min=1e-6))
                # Preserve the executable official implementation's sign convention.
                loss = (sum_log_pf - log_z[0] - log_reward) ** 2
                failed = trajectory["reward"] <= 1e-6 or any(trajectory["errors"])
                trajectory_losses.append(loss * (self.failure_weight if failed else 1.0))
            if trajectory_losses:
                batch_loss = torch.stack(trajectory_losses).mean()
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
                losses.append(float(batch_loss.item()))
        return losses

    def save(self, path: str) -> None:
        torch.save({"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict()}, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])


__all__ = ["GFNAgent", "TrajectoryBuffer"]
