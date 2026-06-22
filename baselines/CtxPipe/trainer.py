"""RL training loop for CtxPipe."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .agent import DQNAgent
    from .env import PipelineEnv


@dataclass
class EpisodeRecord:
    episode: int
    reward: float
    success: bool
    fitness: Optional[float]
    metrics: dict
    ops: list[str]
    n_steps: int
    duration_seconds: float


class CtxPipeTrainer:
    def __init__(
        self,
        env: "PipelineEnv",
        agent: "DQNAgent",
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

        self.history: list[EpisodeRecord] = []

    # ------------------------------------------------------------------
    def _eps_at(self, ep: int) -> float:
        if ep >= self.eps_decay_episodes:
            return self.eps_end
        frac = ep / max(1, self.eps_decay_episodes)
        return self.eps_start + (self.eps_end - self.eps_start) * frac

    # ------------------------------------------------------------------
    def train(self) -> dict:
        best_record: Optional[EpisodeRecord] = None
        best_pipeline_yaml: Optional[str] = None

        for ep in range(self.n_episodes):
            t0 = time.time()
            eps = self._eps_at(ep)
            state = self.env.reset()
            done = False
            episode_reward = 0.0
            terminal_info: dict = {}
            n_transitions = 0

            while not done:
                action = self.agent.select_action(state, eps)
                next_state, reward, done, info = self.env.step(action)
                self.agent.buffer.push(state, action, reward, next_state, done)
                if len(self.agent.buffer) >= self.min_buffer:
                    self.agent.train_step(self.batch_size, self.gamma)
                state = next_state
                episode_reward += float(reward)
                n_transitions += 1
                if done:
                    terminal_info = info

            # Sync target network periodically.
            if (ep + 1) % self.target_update_every == 0:
                self.agent.update_target()

            pipeline = self.env.current_pipeline()
            ops = [s.op for s in pipeline.steps]
            success = bool(terminal_info.get("success", False))
            fitness = terminal_info.get("fitness")
            metrics = terminal_info.get("metrics") or {}

            rec = EpisodeRecord(
                episode=ep + 1,
                reward=episode_reward,
                success=success,
                fitness=float(fitness) if fitness is not None else None,
                metrics=dict(metrics),
                ops=list(ops),
                n_steps=n_transitions,
                duration_seconds=time.time() - t0,
            )
            self.history.append(rec)

            # Track best successful pipeline by fitness.
            if success and fitness is not None:
                if best_record is None or fitness > (best_record.fitness or float("-inf")):
                    best_record = rec
                    best_pipeline_yaml = pipeline.to_yaml()

            if self.verbose:
                fit_str = f"{rec.fitness:.4f}" if rec.fitness is not None else "n/a"
                print(
                    f"[CtxPipe] ep={rec.episode:>3}/{self.n_episodes}  "
                    f"eps={eps:.2f}  reward={rec.reward:+.3f}  "
                    f"fit={fit_str}  steps={rec.n_steps}  "
                    f"ops={ops}  ({rec.duration_seconds:.1f}s)"
                )

        return {
            "history": self.history,
            "best_record": best_record,
            "best_pipeline_yaml": best_pipeline_yaml,
        }
