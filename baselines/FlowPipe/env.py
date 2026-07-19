"""Conditional-GFlowNet environment adapted to DPPBench's shared DSL."""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass

import numpy as np

from baselines.common.op_spec import CATALOG, OpCategory
from baselines.common.pipeline import Pipeline, assign_dag_structure, check_pipeline, make_step
from baselines.common.state_features import dataframe_state_features, primary_feature_frame


@dataclass(frozen=True)
class ActionSpec:
    gid: int
    name: str
    category: OpCategory | None


class FlowPipeEnv:
    """Two-level action graph: logical order, then one action per category."""

    def __init__(self, evaluator, ctx, max_steps: int = 5, seed: int = 42) -> None:
        self.evaluator = evaluator
        self.ctx = ctx
        self.max_steps = int(max_steps)
        self.rng = random.Random(seed)
        self.categories = tuple(OpCategory)
        self.logical_pipelines = tuple(itertools.permutations(self.categories))
        self.n_logic = len(self.logical_pipelines)

        # Keep global gids stable across tasks, as required by an amortized
        # cross-dataset checkpoint; task applicability is enforced by masking.
        specs = list(CATALOG.values())
        self.actions: list[ActionSpec] = [
            ActionSpec(i + self.n_logic, spec.op_name, spec.category)
            for i, spec in enumerate(specs)
        ]
        self.applicable_gids = {
            action.gid for action in self.actions
            if CATALOG[action.name].task_type in ("both", ctx.task_type)
            and self._can_instantiate(action.name)
        }
        self.action_by_gid = {action.gid: action for action in self.actions}
        self.skip_gid = self.n_logic + len(self.actions)
        self.action_dim = self.skip_gid + 1
        self.state_dim = (
            dataframe_state_features(self._raw_frame()).size
            + 1 + self.n_logic + len(self.categories) + 1 + self.action_dim
        )
        self.reset()

    def _can_instantiate(self, name: str) -> bool:
        return any(make_step(name, self.ctx, random.Random(10_000 + i)) is not None for i in range(3))

    def _raw_frame(self):
        data = self.evaluator.load_data()
        frame = getattr(data, "interaction_df", None) if self.ctx.task_type == "rec" else getattr(data, "train_df", None)
        if frame is None:
            raise ValueError("FlowPipe could not locate the primary dataframe")
        return primary_feature_frame(frame, self.ctx.target_col)

    def reset(self):
        self.pipeline = Pipeline()
        self.current_frame = self._raw_frame().copy(deep=True)
        self.logic_selected = False
        self.logic_id = -1
        self.stage_index = 0
        self.selected_gids: list[int] = []
        self.done = False
        self.last_info: dict = {}
        return self.get_state_repr()

    @property
    def current_order(self):
        return self.logical_pipelines[self.logic_id] if self.logic_selected else ()

    def get_state_repr(self) -> dict:
        return {
            "logic_pipeline_selected": self.logic_selected,
            "logic_pipeline_id": self.logic_id,
            "stage_index": self.stage_index,
            "predictor_id": 0,
            "selected_gids": list(self.selected_gids),
            "data_feature": dataframe_state_features(self.current_frame),
        }

    def get_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_dim, dtype=np.float32)
        if self.done:
            return mask
        if not self.logic_selected:
            mask[:self.n_logic] = 1.0
            return mask
        category = self.current_order[self.stage_index]
        for action in self.actions:
            if action.category == category and action.gid in self.applicable_gids:
                mask[action.gid] = 1.0
        mask[self.skip_gid] = 1.0
        return mask

    def step(self, gid: int):
        if self.done:
            raise RuntimeError("FlowPipe trajectory is already terminal")
        gid = int(gid)
        if self.get_action_mask()[gid] <= 0:
            self.done = True
            return self.get_state_repr(), 1e-6, True, {"error": "masked action", "err_code": -1}
        if not self.logic_selected:
            self.logic_selected = True
            self.logic_id = gid
            self.selected_gids.append(gid)
            return self.get_state_repr(), 0.0, False, {"logic_pipeline_id": gid, "err_code": 0}

        info = {"category": self.current_order[self.stage_index].value, "err_code": 0}
        if gid != self.skip_gid:
            action = self.action_by_gid[gid]
            step = make_step(action.name, self.ctx, self.rng)
            if step is None:
                self.done = True
                return self.get_state_repr(), 1e-6, True, {**info, "error": "parameter instantiation failed", "err_code": -1}
            candidate = self.pipeline.copy()
            candidate.steps.append(step)
            try:
                assign_dag_structure(candidate, self.ctx, self.rng)
                processed = self.evaluator.run_preprocessing(candidate)
                if processed is None:
                    raise ValueError("operator produced no dataframe")
            except Exception as exc:
                self.done = True
                return self.get_state_repr(), 1e-6, True, {
                    **info, "error": f"{type(exc).__name__}: {exc}", "err_code": -1,
                }
            self.pipeline = candidate
            self.current_frame = primary_feature_frame(processed, self.ctx.target_col)
            info["op"] = action.name
        else:
            info["op"] = "NOOP"

        self.selected_gids.append(gid)
        self.stage_index += 1
        self.done = self.stage_index >= len(self.categories) or self.stage_index >= self.max_steps
        if not self.done:
            return self.get_state_repr(), 0.0, False, info
        assign_dag_structure(self.pipeline, self.ctx, self.rng)
        report = check_pipeline(self.pipeline, self.ctx)
        if not report.valid:
            return self.get_state_repr(), 1e-6, True, {
                **info, "error": "; ".join(report.errors), "err_code": -1,
            }
        result = self.evaluator.evaluate(self.pipeline)
        if not result.success or result.fitness is None:
            return self.get_state_repr(), 1e-6, True, {
                **info, "error": result.error or "evaluation failed", "err_code": -1,
            }
        return self.get_state_repr(), max(float(result.fitness), 1e-6), True, {
            **info, "fitness": float(result.fitness), "metrics": dict(result.metrics or {}),
        }

    def current_pipeline(self) -> Pipeline:
        assign_dag_structure(self.pipeline, self.ctx, self.rng)
        return self.pipeline.copy()


__all__ = ["ActionSpec", "FlowPipeEnv"]
