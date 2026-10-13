"""Frame-based CtxPipe training loop aligned with the upstream trainer."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional


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
    trace: list[dict]
    logical_pipeline_id: int
    logical_order: list[str]


class CtxPipeTrainer:
    def __init__(
        self,
        env,
        agent,
        frames: int = 50000,
        eps_start: float = 1.0,
        eps_end: float = 0.2,
        eps_decay: float = 2000.0,
        batch_size: int = 200,
        logical_batch_size: int = 40,
        backpropagate_interval: int = 50,
        checkpoint_interval: int = 200,
        output_dir: str | None = None,
        verbose: bool = True,
        **_compat,
    ) -> None:
        self.env = env
        self.agent = agent
        self.frames = int(frames)
        self.eps_start = float(eps_start)
        self.eps_end = float(eps_end)
        self.eps_decay = float(eps_decay)
        self.batch_size = int(batch_size)
        self.logical_batch_size = int(logical_batch_size)
        self.backpropagate_interval = int(backpropagate_interval)
        self.checkpoint_interval = int(checkpoint_interval)
        self.output_dir = output_dir
        self.verbose = bool(verbose)
        self.history: list[EpisodeRecord] = []

    def _epsilon(self, frame: int) -> float:
        return self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -float(frame) / self.eps_decay,
        )

    def _start_episode(self, epsilon: float):
        self.env.reset(0)
        raw_context = self.env.context_embedding()
        logical_id, _ = self.agent.select_logical(
            self.env.logical_state(), raw_context, epsilon,
            context_factory=self.env.context_embedding,
        )
        self.env.set_logical_pipeline(logical_id)
        return {
            "started": time.time(), "raw_context": raw_context,
            "logical_state": self.env.logical_state(), "logical_id": logical_id,
            "trace": [], "steps": 0,
        }

    def train(self) -> dict:
        episode = self._start_episode(self._epsilon(0))
        best_record = None
        best_pipeline_yaml = None

        for frame in range(1, self.frames + 1):
            epsilon = self._epsilon(frame)
            component = self.env.current_component_id()
            if component is None:
                episode = self._start_episode(epsilon)
                component = self.env.current_component_id()
            state = self.env.get_state()
            replay_context = self.env.context_embedding()
            tried: list[int] = []
            while True:
                action, used_model = self.agent.select_component(
                    component, state, replay_context, epsilon, tried,
                    context_factory=self.env.context_embedding,
                )
                tried.append(action)
                next_state, reward, done, info = self.env.step(action)
                if not info.get("retry"):
                    break
                if len(tried) >= self.env.component(component).action_dim:
                    done = True
                    reward = self.env.failure_reward
                    info = {**info, "success": False, "error": "all component actions failed"}
                    break

            next_component = self.env.current_component_id() if not done else None
            if not info.get("retry"):
                self.agent.add_transition(
                    state=state, action=action, reward=float(reward),
                    next_state=next_state, done=bool(done), component=component,
                    next_component=next_component,
                    logical_pipeline_id=episode["logical_id"],
                    context=replay_context,
                )
            episode["steps"] += 1
            episode["trace"].append({
                "frame": frame, "component": component, "action": int(action),
                "selected": info.get("op_added") or info.get("action_name"),
                "used_model": bool(used_model), "reward": float(reward),
                "error": info.get("error"),
            })

            if (
                len(self.agent.buffer) >= self.batch_size
                and len(self.agent.buffer.logical_buffer) >= self.logical_batch_size
                and frame % self.backpropagate_interval == 0
            ):
                self.agent.learn_components(self.batch_size)
                self.agent.learn_logical(self.logical_batch_size)

            if self.output_dir and frame % self.checkpoint_interval == 0:
                self.agent.save(f"{self.output_dir}/q_network_frame_{frame}.pt")

            if done:
                pipeline = self.env.current_pipeline().copy()
                success = bool(info.get("success", False))
                fitness = info.get("fitness")
                self.agent.add_logical_transition(
                    episode["logical_state"], episode["logical_id"],
                    float(reward), episode["raw_context"],
                )
                record = EpisodeRecord(
                    episode=len(self.history) + 1,
                    reward=float(reward), success=success,
                    fitness=float(fitness) if fitness is not None else None,
                    metrics=dict(info.get("metrics") or {}),
                    ops=pipeline.op_names(), n_steps=episode["steps"],
                    duration_seconds=time.time() - episode["started"],
                    trace=list(episode["trace"]),
                    logical_pipeline_id=episode["logical_id"],
                    logical_order=list(self.env.current_order),
                )
                self.history.append(record)
                if success and fitness is not None and (
                    best_record is None or fitness > best_record.fitness
                ):
                    best_record = record
                    best_pipeline_yaml = pipeline.to_yaml()
                if self.verbose:
                    value = f"{record.fitness:.4f}" if record.fitness is not None else "n/a"
                    print(
                        f"[CtxPipe] frame={frame}/{self.frames} episode={record.episode} "
                        f"eps={epsilon:.3f} lp={record.logical_pipeline_id} "
                        f"fitness={value} ops={record.ops}"
                    )
                episode = self._start_episode(epsilon)

        return {
            "history": self.history,
            "best_record": best_record,
            "best_pipeline_yaml": best_pipeline_yaml,
            "training_stats": {
                "component_optimizer_steps": getattr(self.agent, 'component_updates', 0),
                "logical_optimizer_steps": getattr(self.agent, 'logical_updates', 0),
                "component_parameter_changes": getattr(self.agent, 'component_parameter_changes', 0),
                "logical_parameter_changes": getattr(self.agent, 'logical_parameter_changes', 0),
                "frames": self.frames,
            },
        }


__all__ = ["CtxPipeTrainer", "EpisodeRecord"]
