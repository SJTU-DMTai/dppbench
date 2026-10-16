"""AIPipe inference environment aligned with HAIPipe's two-level policy."""
from __future__ import annotations

import itertools
import random

import numpy as np

from baselines.common.op_spec import is_mandatory
from baselines.common.budget import BudgetExceededError
from baselines.common.pipeline import Pipeline, assign_dag_structure, check_pipeline, make_step
from baselines.common.utils import get_available_ops

from .features import dataset_features, feature_frame, frame_from_loaded_data
from .operator_mapping import COMPONENT_CATEGORY_GROUPS


class HAIPipeEnv:
    def __init__(self, evaluator, ctx, *, model_name: str | None, seed: int = 42) -> None:
        self.evaluator = evaluator
        self.ctx = ctx
        self.rng = random.Random(seed)
        specs = list(get_available_ops(ctx.task_type))
        by_category = {}
        for spec in specs:
            if not is_mandatory(spec.op_name, ctx.task_type):
                by_category.setdefault(spec.category, []).append(spec.op_name)
        self.component_actions = {}
        for component, categories in COMPONENT_CATEGORY_GROUPS:
            actions = sorted({name for category in categories for name in by_category.get(category, [])})
            if actions:
                self.component_actions[component] = actions
        components = list(self.component_actions)
        self.logical_pipelines = [list(order) for order in itertools.permutations(components)]
        self.primitive_vocabulary = sorted({name for values in self.component_actions.values() for name in values})
        self.model_vocabulary = [model_name or "model"]
        self.model_id = 0
        loaded = evaluator.load_data()
        self.data_feature = dataset_features(frame_from_loaded_data(loaded, ctx.target_col))
        self.current_data_feature = self.data_feature.copy()
        # Operators are 1..N, blank is 0, and unvisited slots use N+1.
        self.padding_id = len(self.primitive_vocabulary) + 1
        self.data_dim = int(self.data_feature.size)

    def logical_state(self) -> np.ndarray:
        return np.concatenate((self.data_feature, np.asarray([self.model_id + 1], dtype=np.float32)))

    def primitive_state(self, sequence_ids: list[int], logical_id: int) -> np.ndarray:
        padded = list(sequence_ids[: len(self.component_actions)])
        padded.extend([self.padding_id] * (len(self.component_actions) - len(padded)))
        return np.concatenate((
            self.current_data_feature,
            np.asarray(padded, dtype=np.float32),
            np.asarray([self.model_id, logical_id], dtype=np.float32),
        ))

    def _step_executes(self, steps) -> bool:
        candidate = assign_dag_structure(Pipeline(steps=list(steps)).copy(), self.ctx, self.rng)
        referenced = set(candidate.train_prev or [])
        for step in candidate.steps:
            referenced.update(step.prev)
        candidate.sources = [s for s in candidate.sources or [] if s["id"] in referenced]
        report = check_pipeline(candidate, self.ctx, require_mandatory=False)
        if not report.valid:
            return False
        try:
            frame = self.evaluator.run_preprocessing(candidate)
            if frame is None:
                return False
            features = dataset_features(feature_frame(frame, self.ctx.target_col))
        except BudgetExceededError:
            raise
        except Exception:
            return False
        self.current_data_feature = features
        return True

    def episode(self, agent, *, epsilon: float = 0.0) -> dict:
        self.current_data_feature = self.data_feature.copy()
        logical_id = agent.select_logical(self.logical_state(), epsilon)
        order = self.logical_pipelines[logical_id]
        steps = []
        sequence_ids = []
        trace = [{"logical_pipeline_id": logical_id, "logical_pipeline": order}]
        transitions = []
        vocabulary_ids = {name: i + 1 for i, name in enumerate(self.primitive_vocabulary)}
        for position, component in enumerate(order):
            tried = set()
            chosen = None
            while len(tried) < len(self.component_actions[component]) + 1:
                state = self.primitive_state(sequence_ids, logical_id)
                action = agent.select_component(component, state, epsilon, tried)
                tried.add(action)
                if action == len(self.component_actions[component]):
                    sequence_ids.append(0)
                    break
                op_name = self.component_actions[component][action]
                step = make_step(op_name, self.ctx, self.rng)
                if step is not None and self._step_executes([*steps, step]):
                    chosen = step
                    steps.append(step)
                    sequence_ids.append(vocabulary_ids[op_name])
                    break
            next_state = self.primitive_state(sequence_ids, logical_id)
            transitions.append({
                "component": component, "state": state, "action": action,
                "reward": 0.0, "next_state": next_state,
                "done": component == order[-1],
                "next_component": order[position + 1] if position + 1 < len(order) else None,
            })
            trace.append({
                "component": component,
                "selected": chosen.op if chosen is not None else "blank",
                "tried_actions": sorted(tried),
            })
        return {
            "pipeline": Pipeline(steps=steps), "trace": trace,
            "transitions": transitions, "logical_state": self.logical_state(),
            "logical_action": logical_id,
        }

    def rollout(self, agent, *, epsilon: float = 0.0) -> tuple[Pipeline, list[dict]]:
        episode = self.episode(agent, epsilon=epsilon)
        return episode["pipeline"], episode["trace"]


__all__ = ["HAIPipeEnv"]
