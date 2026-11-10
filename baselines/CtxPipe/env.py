"""DPPBench environment with the original CtxPipe state/action protocol."""
from __future__ import annotations

import itertools
import random as _random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

from baselines.common.op_spec import OpCategory, CATALOG
from baselines.common.pipeline import (
    Pipeline, assign_dag_structure, make_step, repair, truncate_optional_steps,
)
from baselines.common.utils import get_available_ops

from .context import GTEEmbedder, data_feature, feature_frame

if TYPE_CHECKING:
    from baselines.common.data_ctx import DataContext
    from baselines.common.search_evaluator import SearchPipelineEvaluator


@dataclass(frozen=True)
class Component:
    key: str
    category: OpCategory
    candidates: tuple[str, ...]

    @property
    def skip_action(self) -> int:
        return len(self.candidates)

    @property
    def action_dim(self) -> int:
        return len(self.candidates) + 1


class PipelineEnv:
    """Select a logical category order, then one primitive per category."""

    def __init__(
        self,
        evaluator: "SearchPipelineEvaluator",
        ctx: "DataContext",
        max_steps: int = 8,
        illegal_penalty: float = -1.0,
        failure_reward: float = -1.0,
        seed: int = 42,
        gte_model_path: str = "/mnt/sdb/dengjiale/dppbench/llm_ckpt/Qwen3-8B",
        context_cache_dir: str = "/mnt/sdb/dengjiale/dppbench/outputs/CtxPipe/context_cache",
        device: str = "cpu",
    ) -> None:
        self.evaluator = evaluator
        self.ctx = ctx
        self.task_type = ctx.task_type
        self.max_steps = int(max_steps)
        self.illegal_penalty = float(illegal_penalty)
        self.failure_reward = float(failure_reward)
        if self.failure_reward >= 0:
            raise ValueError("CtxPipe failure_reward must be negative")
        self.rng = _random.Random(seed)
        self.embedder = GTEEmbedder(
            gte_model_path,
            device=device,
            cache_dir=context_cache_dir,
            task_name=ctx.data_name,
            sample_seed=seed,
        )

        specs = list(CATALOG.values())
        applicable = {s.op_name for s in get_available_ops(self.task_type)}
        self.applicable_ops = {n for n in applicable if self._can_make_step(n)}
        self.op_names = [spec.op_name for spec in specs]
        self.op_gid = {name: index + 1 for index, name in enumerate(self.op_names)}
        self.components: dict[str, Component] = {}
        for category in OpCategory:
            candidates = tuple(
                spec.op_name for spec in specs
                if spec.category == category
            )
            if not candidates:
                raise ValueError(f"CtxPipe category {category.value} has no applicable operators")
            self.components[category.value] = Component(
                key=category.value, category=category, candidates=candidates,
            )
        if len(self.components) != 4:
            raise ValueError(f"CtxPipe expected four categories, got {len(self.components)}")

        self.component_ids = list(self.components)
        self.logical_pipelines = list(itertools.permutations(self.component_ids))
        self.n_components = len(self.component_ids)
        self.n_logical_pipelines = len(self.logical_pipelines)
        self.primitive_vocab_size = len(self.op_names) + 1
        self.component_action_dims = {
            key: component.action_dim for key, component in self.components.items()
        }

        self.target_col = ctx.target_col
        self.raw_frame = self._load_raw_feature_frame()
        self.pipeline = Pipeline()
        self.current_frame = self.raw_frame.copy(deep=True)
        self.sequence_ids = np.zeros(self.n_components, dtype=np.int64)
        self.logical_pipeline_id = 0
        self.position = 0
        self.done = False
        self.last_info: dict = {}

    def _load_raw_feature_frame(self):
        data = self.evaluator.load_data()
        self.target_col = getattr(data, "target_col", None) or self.ctx.target_col
        frame = (
            getattr(data, "interaction_df", None)
            if self.task_type == "rec" else getattr(data, "train_df", None)
        )
        if frame is None:
            raise ValueError("CtxPipe could not locate the task primary dataframe")
        return feature_frame(frame, self.target_col)

    def _can_make_step(self, op_name: str) -> bool:
        return any(
            make_step(op_name, self.ctx, _random.Random(10_000 + i)) is not None
            for i in range(3)
        )

    @property
    def current_order(self) -> tuple[str, ...]:
        return self.logical_pipelines[self.logical_pipeline_id]

    def reset(self, logical_pipeline_id: int = 0) -> np.ndarray:
        self.pipeline = Pipeline()
        self.current_frame = self.raw_frame.copy(deep=True)
        self.sequence_ids = np.zeros(self.n_components, dtype=np.int64)
        self.position = 0
        self.done = False
        self.last_info = {}
        self.set_logical_pipeline(logical_pipeline_id)
        return self.get_state()

    def set_logical_pipeline(self, logical_pipeline_id: int) -> None:
        if not 0 <= int(logical_pipeline_id) < self.n_logical_pipelines:
            raise ValueError(f"invalid logical pipeline id {logical_pipeline_id}")
        self.logical_pipeline_id = int(logical_pipeline_id)

    def invalid_actions(self, component_id):
        component = self.components[component_id]
        return [i for i, name in enumerate(component.candidates) if name not in self.applicable_ops]

    def current_component_id(self) -> Optional[str]:
        if self.done or self.position >= self.n_components:
            return None
        return self.current_order[self.position]

    def component(self, component_id: str) -> Component:
        return self.components[component_id]

    def logical_state(self) -> np.ndarray:
        return np.concatenate([
            data_feature(self.current_frame), np.asarray([0.0], dtype=np.float32),
        ]).astype(np.float32)

    def get_state(self) -> np.ndarray:
        return np.concatenate([
            data_feature(self.current_frame),
            self.sequence_ids.astype(np.float32),
            np.asarray([0.0, float(self.logical_pipeline_id)], dtype=np.float32),
        ]).astype(np.float32)

    def context_embedding(self) -> np.ndarray:
        return self.embedder.embed_frame(self.current_frame).numpy().astype(np.float32)

    def _partial_execute(self, candidate: Pipeline):
        assign_dag_structure(candidate, self.ctx, self.rng)
        return self.evaluator.run_preprocessing(candidate)

    def step(self, action: int):
        component_id = self.current_component_id()
        if component_id is None:
            raise RuntimeError("CtxPipe episode is already complete")
        component = self.components[component_id]
        state = self.get_state()
        info = {
            "component": component_id,
            "category": component.category.value,
            "action": int(action),
        }

        if action in self.invalid_actions(component_id):
            return state, self.illegal_penalty, False, {**info, "error": "masked action", "retry": True}
        if action == component.skip_action:
            info["action_name"] = "SKIP"
            info["reason"] = "skip"
        elif 0 <= action < len(component.candidates):
            op_name = component.candidates[action]
            step = make_step(op_name, self.ctx, self.rng)
            if step is None:
                return state, self.illegal_penalty, False, {
                    **info, "error": "operator has no valid parameters", "retry": True,
                }
            candidate = self.pipeline.copy()
            candidate.steps.append(step)
            try:
                processed = self._partial_execute(candidate)
                if processed is None:
                    raise ValueError("operator produced no dataframe")
            except Exception as exc:
                return state, self.illegal_penalty, False, {
                    **info, "error": f"{type(exc).__name__}: {exc}", "retry": True,
                }
            self.pipeline = candidate
            self.current_frame = feature_frame(processed, self.target_col)
            self.sequence_ids[self.position] = self.op_gid[op_name]
            info["op_added"] = op_name
            info["reason"] = op_name
        else:
            return state, self.illegal_penalty, False, {
                **info, "error": "action index out of range", "retry": True,
            }

        self.position += 1
        self.done = self.position >= self.n_components
        next_state = self.get_state()
        reward = 0.0
        if self.done:
            reward, terminal = self._terminal_reward()
            info.update(terminal)
        self.last_info = info
        return next_state, float(reward), self.done, info

    def _finalize_pipeline(self) -> Pipeline:
        repair(self.pipeline, self.task_type, self.ctx, sort_steps=False)
        truncate_optional_steps(self.pipeline, self.task_type, self.max_steps)
        repair(self.pipeline, self.task_type, self.ctx, sort_steps=False)
        assign_dag_structure(self.pipeline, self.ctx, self.rng)
        return self.pipeline

    def _terminal_reward(self):
        pipeline = self._finalize_pipeline()
        result = self.evaluator.evaluate(pipeline)
        if not result.success:
            return self.failure_reward, {
                "success": False, "error": result.error, "metrics": result.metrics,
            }
        from baselines.common.search_reward import fitness_to_reward
        return fitness_to_reward(result.fitness, self.evaluator.metric_key), {
            "success": True, "fitness": float(result.fitness),
            "metrics": dict(result.metrics or {}),
        }

    def current_pipeline(self) -> Pipeline:
        return self._finalize_pipeline()

    def action_to_op(self, component_id: str, action: int) -> str:
        component = self.components[component_id]
        if action in self.invalid_actions(component_id):
            return state, self.illegal_penalty, False, {**info, "error": "masked action", "retry": True}
        if action == component.skip_action:
            return "SKIP"
        return component.candidates[action]


__all__ = ["Component", "PipelineEnv"]
