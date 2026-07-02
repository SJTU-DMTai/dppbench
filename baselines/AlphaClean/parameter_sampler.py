"""ParameterSampler -- AlphaClean's "Library + ParameterSampler" stage.

For each requested repair, picks an operator at random from the task-applicable
subset of the dppbench catalog, then samples a concrete parameter set using
:func:`baselines.DiffPrep.slot_planner.diffprep_make_step`, which already has
context-aware default-params logic for the shared operator catalog.

For ops with a non-empty ``param_space`` we additionally perturb a single key
to mimic the paper's "threshold parameter sweep" (§6.1).
"""
from __future__ import annotations

import copy
import random as _random
from typing import List, Optional

from baselines.DiffPrep.slot_planner import diffprep_make_step
from baselines.common.pipeline import DataContext

from .operator_catalog import CATALOG, operators_for_task
from .repair import Repair


class ParameterSampler:
    def __init__(self, ctx: DataContext, seed: int = 42) -> None:
        self.ctx = ctx
        self.rng = _random.Random(seed)
        self._ops = operators_for_task(ctx.task_type)
        # Cheap reproducibility: stable order for one-hot indexing.
        self._ops.sort()

    # ------------------------------------------------------------------
    @property
    def operator_names(self) -> List[str]:
        return list(self._ops)

    @property
    def op_index(self) -> dict:
        return {n: i for i, n in enumerate(sorted(CATALOG.keys()))}

    # ------------------------------------------------------------------
    def sample(self, n: int) -> List[Repair]:
        """Return up to ``n`` valid repairs. Invalid samples (op not
        applicable in the current context) are skipped."""
        out: List[Repair] = []
        attempts = 0
        max_attempts = max(n * 4, 8)
        while len(out) < n and attempts < max_attempts:
            attempts += 1
            op_name = self.rng.choice(self._ops)
            r = self._sample_one(op_name)
            if r is not None:
                out.append(r)
        return out

    def sample_for_op(self, op_name: str) -> Optional[Repair]:
        return self._sample_one(op_name)

    # ------------------------------------------------------------------
    def _sample_one(self, op_name: str) -> Optional[Repair]:
        step = diffprep_make_step(op_name, self.ctx, self.rng)
        if step is None:
            return None
        params = copy.deepcopy(step.params)
        if op_name == "JoinTable" and self.ctx.task_type != "rec" and self.ctx.aux_dfs:
            aux = self.rng.choice(self.ctx.aux_dfs)
            params["aux_df"] = f"${aux}"
            if params.get("prefix"):
                params["prefix"] = aux.upper()[:8]
        elif op_name == "ConcatTable" and self.ctx.aux_dfs:
            aux = self.rng.choice(self.ctx.aux_dfs)
            params["other_dfs"] = [f"${aux}"]
        spec = CATALOG[op_name]

        # Optionally perturb one key in param_space (threshold sweep).
        if spec.param_space:
            keys = [k for k in spec.param_space.keys() if k in params or True]
            if keys:
                key = self.rng.choice(keys)
                choices = spec.param_space[key]
                if isinstance(choices, list) and choices:
                    params[key] = copy.deepcopy(self.rng.choice(choices))

        return Repair(op_name=op_name, target=step.target, params=params)


__all__ = ["ParameterSampler"]
