"""Learn2Clean Q-learning training loop.

Mirrors the structure of :class:`baselines.CtxPipe.trainer.CtxPipeTrainer` but
uses tabular Q-learning + Boltzmann exploration (Learn2Clean paper §4).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

from baselines.common.pipeline import Pipeline, assign_dag_structure
from baselines.common.pipeline_constraints import repair as repair_pipeline

if TYPE_CHECKING:
    from .agent import TabularQAgent
    from .env import Learn2CleanEnv


@dataclass
class EpisodeRecord:
    episode: int
    reward: float
    success: bool
    fitness: Optional[float]
    metrics: dict
    ops: list
    n_steps: int
    temperature: float
    duration_seconds: float


@dataclass
class TrainResult:
    best_pipeline: Pipeline
    best_fitness: float
    best_metrics: dict
    history: List[EpisodeRecord] = field(default_factory=list)
    top_k: List[dict] = field(default_factory=list)


class Learn2CleanTrainer:
    def __init__(
        self,
        env: "Learn2CleanEnv",
        agent: "TabularQAgent",
        n_episodes: int = 12,
        verbose: bool = True,
    ) -> None:
        self.env = env
        self.agent = agent
        self.n_episodes = int(n_episodes)
        self.verbose = bool(verbose)
        self.history: List[EpisodeRecord] = []

    # ------------------------------------------------------------------
    def _temperature_at(self, ep: int) -> float:
        if self.n_episodes <= 1:
            return self.agent.temperature_final
        frac = ep / max(1, self.n_episodes - 1)
        T0 = self.agent.temperature_init
        T1 = self.agent.temperature_final
        return T0 + (T1 - T0) * frac

    # ------------------------------------------------------------------
    def train(self) -> TrainResult:
        best_fit = -math.inf
        best_pipe: Optional[Pipeline] = None
        best_metrics: dict = {}
        seen_pipes: dict[str, dict] = {}

        if self.verbose:
            print(f"[Learn2Clean] baseline fitness = {self.env.baseline_fitness:.4f}  "
                  f"n_actions={self.agent.n_actions}  n_episodes={self.n_episodes}")

        for ep in range(self.n_episodes):
            t0 = time.time()
            T = self._temperature_at(ep)
            state = self.env.reset()
            episode_reward = 0.0
            episode_ops: List[str] = []
            n_steps = 0

            done = False
            while not done:
                action = self.agent.select_action(state, T)
                next_state, reward, done, info = self.env.step(action)
                self.agent.update(state, action, reward, next_state, done)
                episode_reward += float(reward)
                n_steps += 1
                if info.get("op_added"):
                    episode_ops.append(info["op_added"])
                state = next_state

            # Re-fetch terminal info: pipeline is already repaired by the env
            # at episode end (STOP / max_steps), so just snapshot it.
            final_pipe = self.env.pipeline.copy()
            repair_pipeline(final_pipe, self.env.task_type, self.env.ctx)
            assign_dag_structure(final_pipe, self.env.ctx)

            fitness: Optional[float] = None
            metrics: dict = {}
            success = False
            if "fitness" in info:
                fitness = info["fitness"]
                metrics = info.get("metrics", {}) or {}
                success = bool(info.get("success", False))
            else:
                ev = self.env.evaluator.evaluate(final_pipe)
                if ev.success and ev.fitness is not None:
                    fitness = float(ev.fitness)
                    metrics = dict(ev.metrics or {})
                    success = True

            record = EpisodeRecord(
                episode=ep + 1,
                reward=float(episode_reward),
                success=success,
                fitness=float(fitness) if fitness is not None else None,
                metrics=metrics,
                ops=list(final_pipe.op_names()),
                n_steps=n_steps,
                temperature=float(T),
                duration_seconds=time.time() - t0,
            )
            self.history.append(record)

            # Track best.
            if fitness is not None and fitness > best_fit:
                best_fit = float(fitness)
                best_pipe = final_pipe
                best_metrics = metrics

            # Maintain top-K dedup'd by hash.
            h = final_pipe.hash()
            if h not in seen_pipes and fitness is not None:
                seen_pipes[h] = {
                    "fitness": float(fitness),
                    "metrics": metrics,
                    "ops": list(final_pipe.op_names()),
                    "yaml": final_pipe.to_yaml(),
                }

            if self.verbose:
                fit_str = f"{fitness:.4f}" if fitness is not None else "n/a"
                print(
                    f"[Learn2Clean] ep={ep+1:>3}/{self.n_episodes}  "
                    f"T={T:.3f}  reward={episode_reward:+.3f}  "
                    f"fitness={fit_str}  steps={n_steps}  "
                    f"ops={record.ops}  qstates={self.agent.n_states}"
                )

        if best_pipe is None:
            # Fall back to whatever the last episode produced.
            best_pipe = self.env.pipeline.copy()
            repair_pipeline(best_pipe, self.env.task_type, self.env.ctx)
            assign_dag_structure(best_pipe, self.env.ctx)
            best_fit = float(self.env.baseline_fitness)
            best_metrics = {}

        top_k = sorted(seen_pipes.values(), key=lambda d: d["fitness"], reverse=True)[:5]
        return TrainResult(
            best_pipeline=best_pipe,
            best_fitness=float(best_fit) if math.isfinite(best_fit) else float("nan"),
            best_metrics=dict(best_metrics),
            history=list(self.history),
            top_k=top_k,
        )


__all__ = ["Learn2CleanTrainer", "EpisodeRecord", "TrainResult"]
