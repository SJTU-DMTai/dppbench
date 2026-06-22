"""MDP environment for Learn2Clean.

Wraps the operator catalog, the pipeline factory, and the evaluator so that a
``TabularQAgent`` can construct pipelines via Q-learning.

State key (used as a Q-table dict key):
    (frozenset(applied_op_idx), last_op_idx)
This collapses state into the *set* of operators applied so far + the most
recent operator. This corresponds to the paper's "current cleaning node" but
with order-aware augmentation via ``last_op_idx``.

Action layout:
    actions 0..n_ops-1 -> append the corresponding operator
    action n_ops       -> STOP (terminate the episode)

Reward shaping (matches Learn2Clean paper §4 with AUC as the quality metric):
    + legal step that improves AUC      :  +reward_max
    + legal step that hurts AUC         :  -reward_max
    + legal step with |delta| <= eps    :  0
    + illegal / duplicate / no-params   :  illegal_reward (defaults to -reward_max)
    + evaluator failure                 :  illegal_reward
    + terminal AUC delta vs baseline    :  +reward_max * (final_auc - baseline_auc)
"""
from __future__ import annotations

import math
import random as _random
from typing import TYPE_CHECKING, FrozenSet, Optional, Tuple

from baselines.SAGA.pipeline import Pipeline, make_step
from baselines.SAGA.pipeline_constraints import is_legal, repair

from .operator_catalog import operators_for_task

if TYPE_CHECKING:
    from baselines.SAGA.pipeline import DataContext
    from baselines.CtxPipe.evaluator import CtxPipeEvaluator


StateKey = Tuple[FrozenSet[int], int]


class Learn2CleanEnv:
    """Q-learning environment for incremental pipeline construction."""

    def __init__(
        self,
        evaluator: "CtxPipeEvaluator",
        ctx: "DataContext",
        max_steps: int = 8,
        reward_max: float = 1.0,
        illegal_reward: float = -1.0,
        improvement_eps: float = 1e-3,
        seed: int = 42,
    ) -> None:
        self.evaluator = evaluator
        self.ctx = ctx
        self.max_steps = int(max_steps)
        self.reward_max = float(reward_max)
        self.illegal_reward = float(illegal_reward)
        self.improvement_eps = float(improvement_eps)
        self._rng = _random.Random(seed)

        self.task_type = ctx.task_type
        self.op_names: list[str] = list(operators_for_task(self.task_type))
        self.op_names.sort()  # stable index for state keys
        self.n_ops: int = len(self.op_names)
        self.stop_action: int = self.n_ops
        self.n_actions: int = self.n_ops + 1

        # Episode state
        self.pipeline: Pipeline = Pipeline()
        self._applied_idx: set[int] = set()
        self._last_idx: int = -1
        self.step_count: int = 0
        self._done: bool = False
        self._last_fitness: float = float("nan")

        # Compute baseline fitness once: empty pipeline + repair() (mandatory ops).
        self._baseline_fitness: float = self._eval_baseline()

    # ------------------------------------------------------------------
    def _eval_baseline(self) -> float:
        empty = Pipeline(steps=[])
        repair(empty, self.task_type, self.ctx)
        ev = self.evaluator.evaluate(empty)
        if ev.success and ev.fitness is not None:
            return float(ev.fitness)
        return 0.5  # neutral AUC if baseline cannot be evaluated

    # ------------------------------------------------------------------
    def _state_key(self) -> StateKey:
        return (frozenset(self._applied_idx), self._last_idx)

    # ------------------------------------------------------------------
    def _try_apply(self, action: int) -> tuple[bool, str]:
        if action < 0 or action >= self.n_ops:
            return False, "out_of_range"
        op_name = self.op_names[action]
        if op_name in {s.op for s in self.pipeline.steps}:
            return False, "duplicate"
        step = make_step(op_name, self.ctx, self._rng)
        if step is None:
            return False, "no_default_params"
        self.pipeline.steps.append(step)
        if not is_legal(self.pipeline, self.task_type):
            self.pipeline.steps.pop()
            return False, "illegal_order"
        return True, op_name

    # ------------------------------------------------------------------
    def _evaluate_current(self) -> float:
        """Evaluate the *current* pipeline (without final repair) -- used
        between intermediate steps. Returns NaN on failure.
        """
        # Repair is non-destructive at this point because it only injects
        # mandatory ops; for intermediate scoring we make a temporary copy so
        # repair-injected ops don't accumulate in the agent's view.
        tmp = Pipeline(steps=list(self.pipeline.steps))
        repair(tmp, self.task_type, self.ctx)
        ev = self.evaluator.evaluate(tmp)
        if ev.success and ev.fitness is not None:
            return float(ev.fitness)
        return float("nan")

    def _terminal_reward(self) -> tuple[float, dict]:
        """Apply repair() to ``self.pipeline`` and return (reward, info)."""
        repair(self.pipeline, self.task_type, self.ctx)
        if len(self.pipeline) == 0:
            return self.illegal_reward, {"empty": True, "success": False}
        ev = self.evaluator.evaluate(self.pipeline)
        if not ev.success or ev.fitness is None:
            return self.illegal_reward, {
                "success": False,
                "error": getattr(ev, "error", None),
                "metrics": getattr(ev, "metrics", {}),
            }
        delta = float(ev.fitness) - self._baseline_fitness
        bonus = self.reward_max * delta
        return float(bonus), {
            "success": True,
            "metrics": dict(ev.metrics or {}),
            "fitness": float(ev.fitness),
            "delta_vs_baseline": delta,
        }

    # ------------------------------------------------------------------
    # Gym-style API
    # ------------------------------------------------------------------
    def reset(self) -> StateKey:
        self.pipeline = Pipeline()
        self._applied_idx = set()
        self._last_idx = -1
        self.step_count = 0
        self._done = False
        self._last_fitness = self._baseline_fitness
        return self._state_key()

    def step(self, action: int) -> tuple[StateKey, float, bool, dict]:
        if self._done:
            raise RuntimeError("step() called on a terminated env. Call reset().")

        info: dict = {"action": int(action)}

        # ---- STOP action ----
        if action == self.stop_action:
            self._done = True
            reward, terminal_info = self._terminal_reward()
            info.update(terminal_info)
            info["reason"] = "stop"
            return self._state_key(), reward, True, info

        # ---- Operator action ----
        ok, reason = self._try_apply(int(action))
        self.step_count += 1
        info["op_added"] = reason if ok else None
        info["reason"] = reason

        if not ok:
            reward = self.illegal_reward
            if self.step_count >= self.max_steps:
                self._done = True
                term_reward, term_info = self._terminal_reward()
                reward = reward + term_reward
                info.update(term_info)
                info["reason"] = f"{reason}+max_steps"
            return self._state_key(), reward, self._done, info

        # legal append: update state bookkeeping
        self._applied_idx.add(int(action))
        self._last_idx = int(action)

        # AUC delta reward for this step
        cur_fit = self._evaluate_current()
        if math.isnan(cur_fit):
            step_reward = self.illegal_reward
            # roll back: this op caused evaluation failure
            self.pipeline.steps.pop()
            self._applied_idx.discard(int(action))
            if self.pipeline.steps:
                last_op = self.pipeline.steps[-1].op
                self._last_idx = (
                    self.op_names.index(last_op)
                    if last_op in self.op_names else -1
                )
            else:
                self._last_idx = -1
            info["reason"] = "eval_failed"
        else:
            delta = cur_fit - self._last_fitness
            if delta > self.improvement_eps:
                step_reward = self.reward_max
            elif delta < -self.improvement_eps:
                step_reward = -self.reward_max
            else:
                step_reward = 0.0
            self._last_fitness = cur_fit
            info["fitness"] = cur_fit
            info["delta"] = delta

        # End of budget => add terminal reward and finish.
        if self.step_count >= self.max_steps:
            self._done = True
            term_reward, term_info = self._terminal_reward()
            info.update(term_info)
            info["reason"] = (info.get("reason") or "ok") + "+max_steps"
            return self._state_key(), step_reward + term_reward, True, info

        return self._state_key(), step_reward, False, info

    # ------------------------------------------------------------------
    def current_pipeline(self) -> Pipeline:
        return self.pipeline

    def action_to_op(self, action: int) -> Optional[str]:
        if action == self.stop_action:
            return "STOP"
        if 0 <= action < self.n_ops:
            return self.op_names[action]
        return None

    @property
    def baseline_fitness(self) -> float:
        return self._baseline_fitness


__all__ = ["Learn2CleanEnv", "StateKey"]
