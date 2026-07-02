"""MDP environment for CtxPipe.

Wraps the operator catalog, the pipeline factory, and the evaluator into a
gym-style ``PipelineEnv`` that exposes ``reset()`` / ``step()``.

State layout (concatenated, fixed length):
    [ context_vector(32) | current_pipeline_one_hot(n_ops) | step_count(1) ]

Action layout:
    actions 0..n_ops-1 -> append the corresponding operator
    action n_ops       -> STOP (terminate the episode)

Reward layout:
    + intermediate steps : 0
    + illegal action     : ``illegal_penalty`` (default -0.05); pipeline is
      NOT mutated, but the step counter advances
    + terminal           : downstream model AUC (failure -> -1.0)
"""
from __future__ import annotations

import random as _random
from typing import TYPE_CHECKING, Optional

import numpy as np

from baselines.common.pipeline import Pipeline, make_step
from baselines.common.pipeline_constraints import is_legal, repair

from .context import ContextEncoder
from .operator_catalog import operators_for_task

if TYPE_CHECKING:
    from baselines.common.pipeline import DataContext

    from .evaluator import CtxPipeEvaluator


class PipelineEnv:
    """An RL environment for incremental pipeline construction."""

    def __init__(
        self,
        evaluator: "CtxPipeEvaluator",
        ctx: "DataContext",
        max_steps: int = 8,
        illegal_penalty: float = -0.05,
        failure_reward: float = -1.0,
        seed: int = 42,
    ) -> None:
        self.evaluator = evaluator
        self.ctx = ctx
        self.max_steps = int(max_steps)
        self.illegal_penalty = float(illegal_penalty)
        self.failure_reward = float(failure_reward)
        self._rng = _random.Random(seed)

        self.task_type = ctx.task_type
        self.op_names: list[str] = list(operators_for_task(self.task_type))
        self.n_ops: int = len(self.op_names)
        self.stop_action: int = self.n_ops
        self.n_actions: int = self.n_ops + 1

        # Encode dataset context once. The context is dataset-level so it does
        # not change across episodes.
        self._context_encoder = ContextEncoder()
        self._ctx_vec: np.ndarray = self._context_encoder.encode(ctx, evaluator._executor)
        self.context_dim: int = self._ctx_vec.shape[0]

        self.state_dim: int = self.context_dim + self.n_ops + 1

        # episode state
        self.pipeline = Pipeline()
        self.step_count: int = 0
        self._done: bool = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _state_vec(self) -> np.ndarray:
        one_hot = np.zeros(self.n_ops, dtype=np.float32)
        for s in self.pipeline.steps:
            if s.op in self.op_names:
                one_hot[self.op_names.index(s.op)] = 1.0
        step_feat = np.array(
            [self.step_count / max(1, self.max_steps)], dtype=np.float32
        )
        return np.concatenate([self._ctx_vec, one_hot, step_feat]).astype(np.float32)

    def _try_apply(self, action: int) -> tuple[bool, str]:
        """Try to append the operator chosen by ``action`` to the current
        pipeline. Returns ``(ok, reason)``.
        """
        if action < 0 or action >= self.n_ops:
            return False, "out_of_range"
        op_name = self.op_names[action]
        # repeat operator -> reject (matches SAGA's repair() de-dup behaviour)
        if op_name in {s.op for s in self.pipeline.steps}:
            return False, "duplicate"
        step = make_step(op_name, self.ctx, self._rng)
        if step is None:
            return False, "no_default_params"
        # tentatively append, then check legality
        self.pipeline.steps.append(step)
        if not is_legal(self.pipeline, self.task_type):
            # rollback
            self.pipeline.steps.pop()
            return False, "illegal_order"
        return True, op_name

    def _terminal_reward(self) -> tuple[float, dict]:
        # Repair the pipeline to make sure mandatory operators are present.
        repair(self.pipeline, self.task_type, self.ctx)
        if len(self.pipeline) == 0:
            return self.failure_reward, {"empty": True}
        ev = self.evaluator.evaluate(self.pipeline)
        if not ev.success:
            return self.failure_reward, {
                "success": False,
                "error": ev.error,
                "metrics": ev.metrics,
            }
        return float(ev.fitness), {
            "success": True,
            "metrics": ev.metrics,
            "fitness": ev.fitness,
        }

    # ------------------------------------------------------------------
    # Gym-style API
    # ------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        self.pipeline = Pipeline()
        self.step_count = 0
        self._done = False
        return self._state_vec()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        if self._done:
            raise RuntimeError("step() called on a terminated env. Call reset().")

        info: dict = {"action": int(action)}
        # ---- STOP action ----
        if action == self.stop_action:
            self._done = True
            reward, terminal_info = self._terminal_reward()
            info.update(terminal_info)
            info["reason"] = "stop"
            return self._state_vec(), reward, True, info

        # ---- Operator action ----
        ok, reason = self._try_apply(int(action))
        self.step_count += 1
        info["op_added"] = reason if ok else None
        info["reason"] = reason

        if not ok:
            reward = self.illegal_penalty
            # If we run out of budget, terminate the episode.
            if self.step_count >= self.max_steps:
                self._done = True
                term_reward, terminal_info = self._terminal_reward()
                # Combine penalty for the illegal step with terminal reward.
                reward = reward + term_reward
                info.update(terminal_info)
                info["reason"] = f"{reason}+max_steps"
            return self._state_vec(), reward, self._done, info

        # legal append
        if self.step_count >= self.max_steps:
            self._done = True
            term_reward, terminal_info = self._terminal_reward()
            info.update(terminal_info)
            info["reason"] = "max_steps"
            return self._state_vec(), term_reward, True, info

        # intermediate transition (no reward signal)
        return self._state_vec(), 0.0, False, info

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def current_pipeline(self) -> Pipeline:
        return self.pipeline

    def action_to_op(self, action: int) -> Optional[str]:
        if action == self.stop_action:
            return "STOP"
        if 0 <= action < self.n_ops:
            return self.op_names[action]
        return None
