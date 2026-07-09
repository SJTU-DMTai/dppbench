"""Component-level AI-pipeline environment for HAIPipe."""
from __future__ import annotations

import random as _random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

from baselines.common.op_spec import CATALOG, OpCategory, is_mandatory
from baselines.common.pipeline import Pipeline, PipelineStep, get_category_order, make_step
from baselines.common.utils import get_available_ops
from baselines.CtxPipe.context import ContextEncoder

from .combiner import combine_pipelines, select_best_candidate
from .operator_mapping import COMPONENT_CATEGORY_GROUPS

if TYPE_CHECKING:
    from baselines.common.data_ctx import DataContext
    from baselines.common.search_evaluator import SearchPipelineEvaluator


@dataclass(frozen=True)
class HAIPipeComponent:
    key: str
    name: str
    categories: tuple[OpCategory, ...]
    candidates: list[str]


class HAIPipeEnv:
    """Generate AI-pipelines, combine with HI scaffold, and reward by best HAI."""

    def __init__(
        self,
        evaluator: "SearchPipelineEvaluator",
        ctx: "DataContext",
        hi_pipeline: Pipeline,
        *,
        max_ai_steps: int = 6,
        max_pipeline_len: int = 8,
        n_combinations: int = 32,
        failure_reward: float = -1.0,
        seed: int = 42,
    ) -> None:
        self.evaluator = evaluator
        self.ctx = ctx
        self.hi_pipeline = hi_pipeline.copy()
        self.max_ai_steps = int(max_ai_steps)
        self.max_pipeline_len = int(max_pipeline_len)
        self.n_combinations = int(n_combinations)
        self.failure_reward = float(failure_reward)
        self._rng = _random.Random(seed)

        self.task_type = ctx.task_type
        self.op_names = [spec.op_name for spec in get_available_ops(self.task_type)]
        self._op_index = {name: i for i, name in enumerate(self.op_names)}
        self.n_ops = len(self.op_names)

        self.components = self._build_components()
        self.component_ids = [c.key for c in self.components]
        self.n_components = len(self.components)
        self.skip_action = max((len(c.candidates) for c in self.components), default=0)
        self.n_actions = self.skip_action + 1

        self._context_encoder = ContextEncoder()
        self._ctx_vec = self._context_encoder.encode(ctx, evaluator)
        self.context_dim = int(self._ctx_vec.shape[0])
        self.state_dim = (
            self.context_dim
            + self.n_components
            + self.max_ai_steps * (self.n_ops + 1)
            + self.n_ops
            + self.n_ops
            + 2
        )

        self.ai_steps: list[PipelineStep] = []
        self.current_component_idx = 0
        self._done = False
        self._best_hai_pipeline: Pipeline | None = None
        self._terminal_info: dict = {}

    # ------------------------------------------------------------------
    def _can_make_step(self, op_name: str) -> bool:
        for i in range(3):
            if make_step(op_name, self.ctx, _random.Random(20_000 + i)) is not None:
                return True
        return False

    def _build_components(self) -> list[HAIPipeComponent]:
        specs = list(get_available_ops(self.task_type))
        by_cat: dict[OpCategory, list[str]] = {}
        for spec in specs:
            if is_mandatory(spec.op_name, self.task_type):
                continue
            if not self._can_make_step(spec.op_name):
                continue
            by_cat.setdefault(spec.category, []).append(spec.op_name)

        canonical = {cat: i for i, cat in enumerate(get_category_order(self.task_type))}
        components: list[HAIPipeComponent] = []
        for name, categories in COMPONENT_CATEGORY_GROUPS:
            cat_set = set(categories)
            candidates: list[str] = []
            for cat in sorted(cat_set, key=lambda c: canonical.get(c, 10_000)):
                candidates.extend(sorted(by_cat.get(cat, [])))
            if not candidates:
                continue
            components.append(HAIPipeComponent(
                key=f"h{len(components):02d}_{name}",
                name=name,
                categories=tuple(categories),
                candidates=candidates,
            ))
        return components

    def _current_component(self) -> HAIPipeComponent | None:
        if self.current_component_idx >= len(self.components):
            return None
        return self.components[self.current_component_idx]

    def current_component_id(self) -> Optional[str]:
        comp = self._current_component()
        return comp.key if comp else None

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(self.n_actions, dtype=bool)
        comp = self._current_component()
        if comp is None or self._done:
            return mask
        used = {step.op for step in self.ai_steps}
        can_add = len(self.ai_steps) < self.max_ai_steps
        if can_add:
            for idx, op_name in enumerate(comp.candidates):
                if op_name not in used:
                    mask[idx] = True
        mask[self.skip_action] = True
        return mask

    # ------------------------------------------------------------------
    def _state_vec(self) -> np.ndarray:
        comp_vec = np.zeros(self.n_components, dtype=np.float32)
        if self.current_component_idx < self.n_components:
            comp_vec[self.current_component_idx] = 1.0

        seq = np.zeros((self.max_ai_steps, self.n_ops + 1), dtype=np.float32)
        for pos in range(self.max_ai_steps):
            seq[pos, 0] = 1.0
        for pos, step in enumerate(self.ai_steps[: self.max_ai_steps]):
            seq[pos, :] = 0.0
            seq[pos, self._op_index.get(step.op, -1) + 1] = 1.0

        ai_used = np.zeros(self.n_ops, dtype=np.float32)
        for step in self.ai_steps:
            idx = self._op_index.get(step.op)
            if idx is not None:
                ai_used[idx] = 1.0

        hi_used = np.zeros(self.n_ops, dtype=np.float32)
        for step in self.hi_pipeline.steps:
            idx = self._op_index.get(step.op)
            if idx is not None:
                hi_used[idx] = 1.0

        scalars = np.array([
            len(self.ai_steps) / max(1, self.max_ai_steps),
            self.current_component_idx / max(1, self.n_components),
        ], dtype=np.float32)
        return np.concatenate([
            self._ctx_vec,
            comp_vec,
            seq.reshape(-1),
            ai_used,
            hi_used,
            scalars,
        ]).astype(np.float32)

    def _ai_pipeline(self) -> Pipeline:
        return Pipeline(steps=[PipelineStep.from_dict(s.to_dict()) for s in self.ai_steps])

    def _finish(self) -> tuple[float, dict]:
        candidates = combine_pipelines(
            self.hi_pipeline,
            self._ai_pipeline(),
            self.ctx,
            self._rng,
            max_len=self.max_pipeline_len,
            n_combinations=self.n_combinations,
        )
        selected = select_best_candidate(self.evaluator, candidates)
        self._best_hai_pipeline = selected.pipeline.copy()
        success = selected.fitness != float("-inf")
        reward = float(selected.fitness) if success else self.failure_reward
        info = {
            "success": success,
            "fitness": float(selected.fitness) if success else None,
            "metrics": selected.metrics,
            "error": selected.error,
            "ai_ops": self._ai_pipeline().op_names(),
            "hai_ops": self._best_hai_pipeline.op_names(),
            "n_hai_candidates": len(candidates),
            "candidates": selected.candidates,
        }
        self._terminal_info = info
        return reward, info

    # ------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        self.ai_steps = []
        self.current_component_idx = 0
        self._done = False
        self._best_hai_pipeline = None
        self._terminal_info = {}
        return self._state_vec()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        if self._done:
            return self._state_vec(), 0.0, True, dict(self._terminal_info)
        comp = self._current_component()
        if comp is None:
            self._done = True
            reward, info = self._finish()
            return self._state_vec(), reward, True, info

        action_name = "SKIP" if action == self.skip_action else None
        op_added = None
        if 0 <= action < len(comp.candidates):
            action_name = comp.candidates[action]
            if len(self.ai_steps) < self.max_ai_steps and action_name not in {s.op for s in self.ai_steps}:
                step = make_step(action_name, self.ctx, self._rng)
                if step is not None:
                    self.ai_steps.append(step)
                    op_added = action_name
        elif action != self.skip_action:
            action_name = "INVALID"

        self.current_component_idx += 1
        done = self.current_component_idx >= self.n_components or len(self.ai_steps) >= self.max_ai_steps
        info = {
            "component": comp.key,
            "component_name": comp.name,
            "action": int(action),
            "action_name": action_name,
            "op_added": op_added,
            "reason": "selected" if op_added else "skip",
        }
        reward = 0.0
        if done:
            self._done = True
            reward, terminal_info = self._finish()
            info.update(terminal_info)
        return self._state_vec(), float(reward), self._done, info

    def current_pipeline(self) -> Pipeline:
        if self._best_hai_pipeline is not None:
            return self._best_hai_pipeline.copy()
        return self._ai_pipeline()

    def current_ai_pipeline(self) -> Pipeline:
        return self._ai_pipeline()


__all__ = ["HAIPipeComponent", "HAIPipeEnv"]
