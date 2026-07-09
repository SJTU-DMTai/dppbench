"""Component-level MDP environment for CtxPipe.

The original CtxPipe formulates pipeline construction as a sequence of
component-level decisions rather than a flat action over every possible
operator.  This adapter keeps that semantics while using DPPBench's shared
operator space and evaluator: each component corresponds to one canonical
operator category, a component-specific policy chooses one operator (or SKIP),
and the executable DAG is produced by common pipeline APIs.
"""
from __future__ import annotations

import random as _random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

from baselines.common.op_spec import OpCategory, is_mandatory
from baselines.common.pipeline import (
    Pipeline,
    assign_dag_structure,
    check_pipeline,
    get_category_order,
    make_step,
    repair,
    truncate_optional_steps,
)
from baselines.common.utils import get_available_ops

from .context import ContextEncoder

if TYPE_CHECKING:
    from baselines.common.data_ctx import DataContext
    from baselines.common.search_evaluator import SearchPipelineEvaluator


@dataclass(frozen=True)
class ComponentSlot:
    """One CtxPipe component/stage."""

    key: str
    category: OpCategory
    candidates: list[str]
    mandatory: bool = False


class PipelineEnv:
    """RL environment for component-level pipeline construction."""

    def __init__(
        self,
        evaluator: "SearchPipelineEvaluator",
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
        self.op_names: list[str] = [
            spec.op_name for spec in get_available_ops(self.task_type)
        ]
        self._op_index = {name: idx for idx, name in enumerate(self.op_names)}
        self.n_ops = len(self.op_names)

        self.slots = self._build_slots()
        self.component_ids = [slot.key for slot in self.slots]
        self.n_components = len(self.slots)
        self.skip_action = max((len(slot.candidates) for slot in self.slots), default=0)
        self.n_actions = self.skip_action + 1

        self._context_encoder = ContextEncoder()
        self._ctx_vec = self._context_encoder.encode(ctx, evaluator)
        self.context_dim = int(self._ctx_vec.shape[0])

        # Ordered sequence state: max_steps positions, each with n_ops + 1
        # choices (0 = empty, op index + 1 = selected op). This preserves the
        # pipeline order that the previous unordered one-hot state lost.
        self.state_dim = (
            self.context_dim
            + self.n_components
            + self.max_steps * (self.n_ops + 1)
            + self.n_ops
            + 2
        )

        self.pipeline = Pipeline()
        self.current_slot_idx = 0
        self.step_count = 0
        self._done = False
        self._last_report_errors: list[str] = []

    # ------------------------------------------------------------------
    # Slot/action construction
    # ------------------------------------------------------------------
    def _build_slots(self) -> list[ComponentSlot]:
        specs = list(get_available_ops(self.task_type))
        order = list(get_category_order(self.task_type))
        known = set(order)
        for spec in specs:
            if spec.category not in known:
                order.append(spec.category)
                known.add(spec.category)

        slots: list[ComponentSlot] = []
        for cat in order:
            raw = [spec.op_name for spec in specs if spec.category == cat]
            if not raw:
                continue
            applicable = [name for name in raw if self._can_make_step(name)]
            if not applicable:
                continue
            mandatory = [
                name for name in applicable
                if is_mandatory(name, self.task_type)
            ]
            candidates = mandatory if mandatory else applicable
            slots.append(
                ComponentSlot(
                    key=f"c{len(slots):02d}_{cat.value}",
                    category=cat,
                    candidates=candidates,
                    mandatory=bool(mandatory),
                )
            )
        return slots

    def _can_make_step(self, op_name: str) -> bool:
        # ``make_step`` is the common context-aware constructor. Try a few
        # deterministic draws because some defaults depend on random column
        # sampling, but never duplicate its parameter logic here.
        for i in range(3):
            if make_step(op_name, self.ctx, _random.Random(10_000 + i)) is not None:
                return True
        return False

    def _current_slot(self) -> ComponentSlot | None:
        if self.current_slot_idx >= len(self.slots):
            return None
        return self.slots[self.current_slot_idx]

    def current_component_id(self) -> Optional[str]:
        slot = self._current_slot()
        return slot.key if slot is not None else None

    def current_component_name(self) -> Optional[str]:
        slot = self._current_slot()
        return slot.category.value if slot is not None else None

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(self.n_actions, dtype=bool)
        slot = self._current_slot()
        if slot is None or self._done:
            return mask

        used = {step.op for step in self.pipeline.steps}
        allow_optional_op = len(self.pipeline.steps) < self.max_steps
        for idx, op_name in enumerate(slot.candidates):
            if op_name in used:
                continue
            if slot.mandatory or allow_optional_op:
                mask[idx] = True

        if not slot.mandatory:
            mask[self.skip_action] = True

        # Robust fallback: if a mandatory slot has no valid candidate because
        # it was already selected earlier, allow SKIP so the episode can move on;
        # terminal repair will preserve mandatory post-conditions.
        if not mask.any():
            mask[self.skip_action] = True
        return mask

    # ------------------------------------------------------------------
    # State and pipeline helpers
    # ------------------------------------------------------------------
    def _state_vec(self) -> np.ndarray:
        comp = np.zeros(self.n_components, dtype=np.float32)
        if self.current_slot_idx < self.n_components:
            comp[self.current_slot_idx] = 1.0

        seq = np.zeros((self.max_steps, self.n_ops + 1), dtype=np.float32)
        for pos in range(self.max_steps):
            seq[pos, 0] = 1.0
        for pos, step in enumerate(self.pipeline.steps[: self.max_steps]):
            seq[pos, :] = 0.0
            seq[pos, self._op_index.get(step.op, -1) + 1] = 1.0

        used = np.zeros(self.n_ops, dtype=np.float32)
        for step in self.pipeline.steps:
            idx = self._op_index.get(step.op)
            if idx is not None:
                used[idx] = 1.0

        scalars = np.array(
            [
                len(self.pipeline.steps) / max(1, self.max_steps),
                self.current_slot_idx / max(1, self.n_components),
            ],
            dtype=np.float32,
        )
        return np.concatenate([
            self._ctx_vec,
            comp,
            seq.reshape(-1),
            used,
            scalars,
        ]).astype(np.float32)

    def _remaining_mandatory_slots(self) -> bool:
        for slot in self.slots[self.current_slot_idx:]:
            if slot.mandatory:
                return True
        return False

    def _should_finish(self) -> bool:
        if self.current_slot_idx >= len(self.slots):
            return True
        if len(self.pipeline.steps) >= self.max_steps and not self._remaining_mandatory_slots():
            return True
        return False

    def _finalize_pipeline(self) -> None:
        repair(self.pipeline, self.task_type, self.ctx)
        truncate_optional_steps(self.pipeline, self.task_type, self.max_steps)
        repair(self.pipeline, self.task_type, self.ctx)
        assign_dag_structure(self.pipeline, self.ctx, self._rng)

    def _terminal_reward(self) -> tuple[float, dict]:
        self._finalize_pipeline()
        report = check_pipeline(self.pipeline, self.ctx)
        self._last_report_errors = list(report.errors)
        if not report.valid:
            return self.failure_reward, {
                "success": False,
                "error": "; ".join(report.errors),
                "check_errors": report.errors,
            }
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
        self.current_slot_idx = 0
        self.step_count = 0
        self._done = False
        self._last_report_errors = []
        return self._state_vec()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        if self._done:
            raise RuntimeError("step() called on a terminated env. Call reset().")

        slot = self._current_slot()
        if slot is None:
            self._done = True
            reward, terminal_info = self._terminal_reward()
            info = {"action": int(action), "reason": "no_component"}
            info.update(terminal_info)
            return self._state_vec(), reward, True, info

        action = int(action)
        mask = self.action_mask()
        if action == self.skip_action:
            action_name = "SKIP"
        elif 0 <= action < len(slot.candidates):
            action_name = slot.candidates[action]
        else:
            action_name = None
        info: dict = {
            "action": action,
            "action_name": action_name,
            "component": slot.key,
            "category": slot.category.value,
            "op_added": None,
        }

        reward = 0.0
        if action < 0 or action >= self.n_actions or not mask[action]:
            reward = self.illegal_penalty
            info["reason"] = "masked_or_invalid_action"
        elif action == self.skip_action:
            if slot.mandatory:
                reward = self.illegal_penalty
                info["reason"] = "skip_mandatory"
            else:
                info["reason"] = "skip"
        else:
            op_name = slot.candidates[action]
            step = make_step(op_name, self.ctx, self._rng)
            if step is None:
                reward = self.illegal_penalty
                info["reason"] = "no_default_params"
            elif op_name in {s.op for s in self.pipeline.steps}:
                reward = self.illegal_penalty
                info["reason"] = "duplicate"
            else:
                self.pipeline.steps.append(step)
                info["op_added"] = op_name
                info["reason"] = op_name

        self.current_slot_idx += 1
        self.step_count += 1

        if self._should_finish():
            self._done = True
            terminal_reward, terminal_info = self._terminal_reward()
            info.update(terminal_info)
            if info.get("reason") in {"masked_or_invalid_action", "skip_mandatory", "no_default_params", "duplicate"}:
                terminal_reward += reward
            return self._state_vec(), terminal_reward, True, info

        return self._state_vec(), reward, False, info

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def current_pipeline(self) -> Pipeline:
        self._finalize_pipeline()
        return self.pipeline

    def action_to_op(self, action: int) -> Optional[str]:
        slot = self._current_slot()
        if slot is None:
            return None
        if action == self.skip_action:
            return "SKIP"
        if 0 <= action < len(slot.candidates):
            return slot.candidates[action]
        return None
