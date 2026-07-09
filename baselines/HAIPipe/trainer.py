"""Training and greedy rollout for the HAIPipe DQN adapter."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from baselines.CtxPipe.agent import DQNAgent
    from .env import HAIPipeEnv


@dataclass
class HAIPipeEpisodeRecord:
    episode: int
    reward: float
    success: bool
    fitness: Optional[float]
    metrics: dict
    ai_ops: list[str]
    hai_ops: list[str]
    n_steps: int
    duration_seconds: float
    trace: list[dict]


class HAIPipeTrainer:
    def __init__(
        self,
        env: "HAIPipeEnv",
        agent: "DQNAgent",
        *,
        n_episodes: int = 20,
        eps_start: float = 1.0,
        eps_end: float = 0.1,
        eps_decay_episodes: Optional[int] = None,
        gamma: float = 0.95,
        batch_size: int = 32,
        target_update_every: int = 10,
        min_buffer: int = 64,
        verbose: bool = True,
    ) -> None:
        self.env = env
        self.agent = agent
        self.n_episodes = int(n_episodes)
        self.eps_start = float(eps_start)
        self.eps_end = float(eps_end)
        self.eps_decay_episodes = int(eps_decay_episodes or max(1, n_episodes - 1))
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.target_update_every = int(target_update_every)
        self.min_buffer = int(min_buffer)
        self.verbose = bool(verbose)
        self.history: list[HAIPipeEpisodeRecord] = []

    def _eps_at(self, ep: int) -> float:
        if ep >= self.eps_decay_episodes:
            return self.eps_end
        frac = ep / max(1, self.eps_decay_episodes)
        return self.eps_start + (self.eps_end - self.eps_start) * frac

    def train(self) -> dict:
        best_record: HAIPipeEpisodeRecord | None = None
        best_pipeline_yaml: str | None = None

        for ep in range(self.n_episodes):
            t0 = time.time()
            eps = self._eps_at(ep)
            state = self.env.reset()
            done = False
            episode_reward = 0.0
            terminal_info: dict = {}
            trace: list[dict] = []
            transitions: list[tuple] = []
            n_steps = 0

            while not done:
                component = self.env.current_component_id()
                if component is None:
                    break
                mask = self.env.action_mask()
                action = self.agent.select_action(component, state, eps, mask)
                next_state, reward, done, info = self.env.step(action)
                next_component = self.env.current_component_id() if not done else None
                next_mask = self.env.action_mask() if not done else np.zeros(
                    self.env.n_actions, dtype=bool
                )
                transitions.append((
                    component,
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                    next_component,
                    next_mask,
                ))
                trace.append({
                    "component": component,
                    "component_name": info.get("component_name"),
                    "action": int(action),
                    "selected": info.get("op_added") or info.get("action_name"),
                    "reward": float(reward),
                })
                state = next_state
                episode_reward += float(reward)
                n_steps += 1
                if done:
                    terminal_info = info

            success = bool(terminal_info.get("success", False))
            touched_components: set[str] = set()
            for transition in transitions:
                component = transition[0]
                self.agent.push(*transition, episode_success=success)
                touched_components.add(component)
            for component in touched_components:
                if self.agent.buffer_size(component) >= self.min_buffer:
                    self.agent.train_step(component, self.batch_size, self.gamma)
            if (ep + 1) % self.target_update_every == 0:
                self.agent.update_target()

            rec = HAIPipeEpisodeRecord(
                episode=ep + 1,
                reward=episode_reward,
                success=success,
                fitness=terminal_info.get("fitness"),
                metrics=dict(terminal_info.get("metrics") or {}),
                ai_ops=list(terminal_info.get("ai_ops") or []),
                hai_ops=list(terminal_info.get("hai_ops") or self.env.current_pipeline().op_names()),
                n_steps=n_steps,
                duration_seconds=time.time() - t0,
                trace=trace,
            )
            self.history.append(rec)
            if rec.success and rec.fitness is not None:
                if best_record is None or rec.fitness > (best_record.fitness or float("-inf")):
                    best_record = rec
                    best_pipeline_yaml = self.env.current_pipeline().to_yaml()

            if self.verbose:
                fit_str = f"{rec.fitness:.4f}" if rec.fitness is not None else "n/a"
                print(
                    f"[HAIPipe] ep={rec.episode:>3}/{self.n_episodes}  "
                    f"eps={eps:.2f}  reward={rec.reward:+.3f}  "
                    f"fit={fit_str}  ai={rec.ai_ops}  hai={rec.hai_ops}  "
                    f"({rec.duration_seconds:.1f}s)"
                )

        return {
            "history": self.history,
            "best_record": best_record,
            "best_pipeline_yaml": best_pipeline_yaml,
        }


def greedy_rollout(agent: "DQNAgent", env: "HAIPipeEnv", verbose: bool = True) -> dict:
    state = env.reset()
    done = False
    total_reward = 0.0
    terminal_info: dict = {}
    trace: list[dict] = []
    while not done:
        component = env.current_component_id()
        if component is None:
            break
        mask = env.action_mask()
        action = agent.select_action(component, state, eps=0.0, action_mask=mask)
        state, reward, done, info = env.step(action)
        total_reward += float(reward)
        trace.append({
            "component": component,
            "component_name": info.get("component_name"),
            "action": int(action),
            "selected": info.get("op_added") or info.get("action_name"),
            "reward": float(reward),
        })
        if done:
            terminal_info = info
    pipe = env.current_pipeline()
    out = {
        "pipeline": pipe,
        "pipeline_yaml": pipe.to_yaml(),
        "fitness": terminal_info.get("fitness"),
        "metrics": terminal_info.get("metrics") or {},
        "error": terminal_info.get("error"),
        "success": bool(terminal_info.get("success", False)),
        "ai_ops": terminal_info.get("ai_ops") or [],
        "hai_ops": pipe.op_names(),
        "reward": total_reward,
        "trace": trace,
        "candidates": terminal_info.get("candidates") or [],
    }
    if verbose:
        fit = out.get("fitness")
        fit_str = f"{fit:.4f}" if isinstance(fit, float) else "n/a"
        print(f"[HAIPipe] greedy final ops={out['hai_ops']} fitness={fit_str}")
    return out


__all__ = ["HAIPipeEpisodeRecord", "HAIPipeTrainer", "greedy_rollout"]
