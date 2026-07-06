"""MDP environment for CtxPipe.

Wraps the operator catalog, the pipeline factory, and the evaluator into a
gym-style ``PipelineEnv`` that exposes ``reset()`` / ``step()``.

State layout (concatenated, fixed length):
    [ context_vector(32) | current_pipeline_one_hot(n_ops) |
      selected_source_flags(n_sources) | primary_parent_slots(max_steps) |
      step_count(1) ]

Action layout:
    actions 0..n_ops-1 -> append the corresponding operator
    next source actions -> toggle optional source tables
    next parent actions -> set the latest operator's primary predecessor
    next param action   -> resample the latest operator's params
    final action        -> STOP (terminate the episode)

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

from baselines.common.pipeline import (
    Pipeline,
    _source_entries_for_context,
    _strip_df_params,
    _table_to_source_id,
    make_step,
)
from baselines.common.pipeline import is_legal, repair

from .context import ContextEncoder
from .operator_catalog import operators_for_task

if TYPE_CHECKING:
    from baselines.common.data_ctx import DataContext

    from .evaluator import CtxPipeEvaluator


class PipelineEnv:
    """An RL environment for incremental DAG pipeline construction."""

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

        if self.task_type == "rec":
            self._mandatory_sources = [ctx.primary_table] + list(ctx.secondary_tables)
            self._optional_sources: list[str] = []
        else:
            self._mandatory_sources = [ctx.primary_table]
            self._optional_sources = list(ctx.aux_dfs)
        self._source_names = self._mandatory_sources + self._optional_sources
        self._selected_optional_sources: set[str] = set()
        self.n_sources = len(self._source_names)

        self.source_action_start = self.n_ops
        self.parent_action_start = self.source_action_start + len(self._optional_sources)
        self.n_parent_actions = self.max_steps + self.n_sources
        self.param_action: int = self.parent_action_start + self.n_parent_actions
        self.stop_action: int = self.param_action + 1
        self.n_actions: int = self.stop_action + 1

        # Encode dataset context once. The context is dataset-level so it does
        # not change across episodes.
        self._context_encoder = ContextEncoder()
        self._ctx_vec: np.ndarray = self._context_encoder.encode(ctx, evaluator._executor)
        self.context_dim: int = self._ctx_vec.shape[0]

        self.state_dim: int = (
            self.context_dim + self.n_ops + self.n_sources + self.max_steps + 1
        )

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
        source_flags = np.zeros(self.n_sources, dtype=np.float32)
        active_sources = set(self._active_source_names())
        for idx, name in enumerate(self._source_names):
            if name in active_sources:
                source_flags[idx] = 1.0
        parent_slots = np.zeros(self.max_steps, dtype=np.float32)
        denom = max(1, self.n_sources + self.max_steps - 1)
        for idx, step in enumerate(self.pipeline.steps[:self.max_steps]):
            ref = step.prev[0] if step.prev else None
            parent_slots[idx] = self._parent_ref_to_slot(ref) / denom
        step_feat = np.array(
            [self.step_count / max(1, self.max_steps)], dtype=np.float32
        )
        return np.concatenate([
            self._ctx_vec, one_hot, source_flags, parent_slots, step_feat
        ]).astype(np.float32)

    def _active_source_names(self) -> list[str]:
        out = list(self._mandatory_sources)
        for name in self._optional_sources:
            if name in self._selected_optional_sources:
                out.append(name)
        for step in self.pipeline.steps:
            _params, refs = _strip_df_params(step.params or {})
            for name in refs:
                if name in self._source_names and name not in out:
                    out.append(name)
        return out

    def _context_for_selected_sources(self):
        if self.task_type != "tabular":
            return self.ctx
        from dataclasses import replace
        return replace(self.ctx, aux_dfs=sorted(self._selected_optional_sources))

    def _sources(self) -> list[dict]:
        if self.task_type == "rec":
            return _source_entries_for_context(self.ctx)
        return _source_entries_for_context(self.ctx, list(self._active_source_names()[1:]))

    def _parent_ref_to_slot(self, ref: str | None) -> int:
        if ref is None:
            return 0
        if isinstance(ref, str) and ref.startswith("s"):
            for source in self._sources():
                if source.get("id") == ref:
                    table = source.get("table")
                    if table in self._source_names:
                        return self._source_names.index(table)
            return 0
        if isinstance(ref, str) and ref.startswith("o"):
            try:
                return self.n_sources + max(0, int(ref[1:]) - 1)
            except ValueError:
                return 0
        return 0

    def _parent_slot_to_ref(self, slot: int, latest_step_idx: int) -> str | None:
        if slot < self.n_sources:
            table = self._source_names[slot]
            if table not in self._active_source_names():
                return None
            source_by_table = _table_to_source_id(self._sources())
            return source_by_table.get(table)
        op_idx = slot - self.n_sources + 1
        if 1 <= op_idx < latest_step_idx:
            return f"o{op_idx}"
        return None

    def _assign_dag_structure(self) -> None:
        sources = self._sources()
        source_by_table = _table_to_source_id(sources)
        root_table = self.ctx.primary_table
        current = source_by_table[root_table]
        known_refs = set(source_by_table.values())

        for idx, step in enumerate(self.pipeline.steps, start=1):
            step.id = f"o{idx}"
            params, refs = _strip_df_params(step.params or {})
            primary = current
            if idx == 1 and step.prev and step.prev[0] in set(source_by_table.values()):
                primary = step.prev[0]
            prev = [primary]
            if step.op in {"JoinTable", "ConcatTable"}:
                for table in refs:
                    sid = source_by_table.get(table)
                    if sid and sid not in prev:
                        prev.append(sid)
                if step.op == "JoinTable" and self.task_type == "rec" and params.get("method") == "rec":
                    for table in ("user", "item"):
                        sid = source_by_table.get(table)
                        if sid and sid not in prev:
                            prev.append(sid)
            step.prev = prev
            current = step.id
            known_refs.add(step.id)

        self.pipeline.sources = sources
        self.pipeline.train_prev = [current] if self.pipeline.steps else [source_by_table[root_table]]

    def _try_toggle_source(self, action: int) -> tuple[bool, str]:
        idx = action - self.source_action_start
        if idx < 0 or idx >= len(self._optional_sources):
            return False, "source_out_of_range"
        name = self._optional_sources[idx]
        for step in self.pipeline.steps:
            _params, refs = _strip_df_params(step.params or {})
            if name in refs:
                return False, "source_in_use"
        if name in self._selected_optional_sources:
            self._selected_optional_sources.remove(name)
            reason = f"deselect_source:{name}"
        else:
            self._selected_optional_sources.add(name)
            reason = f"select_source:{name}"
        self._assign_dag_structure()
        return True, reason

    def _try_rewire_latest(self, action: int) -> tuple[bool, str]:
        if not self.pipeline.steps:
            return False, "no_step_to_rewire"
        slot = action - self.parent_action_start
        latest_idx = len(self.pipeline.steps)
        ref = self._parent_slot_to_ref(slot, latest_idx)
        if ref is None:
            return False, "invalid_parent"
        if latest_idx > 1 and ref != f"o{latest_idx - 1}":
            return False, "would_disconnect_dag"
        latest = self.pipeline.steps[-1]
        latest.prev = [ref]
        self._assign_dag_structure()
        return True, f"parent:{ref}"

    def _try_resample_latest_params(self) -> tuple[bool, str]:
        if not self.pipeline.steps:
            return False, "no_step_to_resample"
        latest = self.pipeline.steps[-1]
        replacement = make_step(latest.op, self._context_for_selected_sources(), self._rng)
        if replacement is None:
            return False, "no_default_params"
        latest.target = replacement.target
        latest.params = replacement.params
        self._assign_dag_structure()
        return True, f"params:{latest.op}"

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
        step = make_step(op_name, self._context_for_selected_sources(), self._rng)
        if step is None:
            return False, "no_default_params"
        # tentatively append, then check legality
        self.pipeline.steps.append(step)
        if not is_legal(self.pipeline, self.task_type):
            # rollback
            self.pipeline.steps.pop()
            return False, "illegal_order"
        self._assign_dag_structure()
        return True, op_name

    def _terminal_reward(self) -> tuple[float, dict]:
        # Repair the pipeline to make sure mandatory operators are present.
        repair(self.pipeline, self.task_type, self.ctx)
        self._assign_dag_structure()
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
        self._selected_optional_sources = set()
        self.step_count = 0
        self._done = False
        self._assign_dag_structure()
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

        # ---- Source-table action ----
        if self.source_action_start <= action < self.parent_action_start:
            ok, reason = self._try_toggle_source(int(action))
            self.step_count += 1
            info["op_added"] = None
            info["reason"] = reason
            reward = 0.0 if ok else self.illegal_penalty
            if self.step_count >= self.max_steps:
                self._done = True
                term_reward, terminal_info = self._terminal_reward()
                reward += term_reward
                info.update(terminal_info)
                info["reason"] = f"{reason}+max_steps"
            return self._state_vec(), reward, self._done, info

        # ---- DAG parent action ----
        if self.parent_action_start <= action < self.param_action:
            ok, reason = self._try_rewire_latest(int(action))
            self.step_count += 1
            info["op_added"] = None
            info["reason"] = reason
            reward = 0.0 if ok else self.illegal_penalty
            if self.step_count >= self.max_steps:
                self._done = True
                term_reward, terminal_info = self._terminal_reward()
                reward += term_reward
                info.update(terminal_info)
                info["reason"] = f"{reason}+max_steps"
            return self._state_vec(), reward, self._done, info

        # ---- Parameter action ----
        if action == self.param_action:
            ok, reason = self._try_resample_latest_params()
            self.step_count += 1
            info["op_added"] = None
            info["reason"] = reason
            reward = 0.0 if ok else self.illegal_penalty
            if self.step_count >= self.max_steps:
                self._done = True
                term_reward, terminal_info = self._terminal_reward()
                reward += term_reward
                info.update(terminal_info)
                info["reason"] = f"{reason}+max_steps"
            return self._state_vec(), reward, self._done, info

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
        self._assign_dag_structure()
        return self.pipeline

    def action_to_op(self, action: int) -> Optional[str]:
        if action == self.stop_action:
            return "STOP"
        if 0 <= action < self.n_ops:
            return self.op_names[action]
        if self.source_action_start <= action < self.parent_action_start:
            return f"SOURCE:{self._optional_sources[action - self.source_action_start]}"
        if self.parent_action_start <= action < self.param_action:
            return f"PARENT:{action - self.parent_action_start}"
        if action == self.param_action:
            return "PARAMS"
        return None
