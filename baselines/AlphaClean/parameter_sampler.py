"""ParameterSampler -- AlphaClean's "Library + ParameterSampler" stage.

For each requested repair, picks an operator at random from the task-applicable
subset of the shared DPPBench catalog, then samples a concrete parameter set
through ``baselines.common.pipeline``.

For ops with a non-empty context-aware parameter space we additionally perturb
a single key to mimic the paper's parameter sweep.
"""
from __future__ import annotations

import copy
import random as _random
from typing import List, Optional

from baselines.common.data_ctx import DataContext
from baselines.common.pipeline import get_param_space, make_step
from baselines.common.utils import get_available_ops, get_op

from .operator_catalog import CATALOG
from .repair import Repair


class ParameterSampler:
    def __init__(self, ctx: DataContext, seed: int = 42) -> None:
        self.ctx = ctx
        self.rng = _random.Random(seed)
        self._ops = [spec.op_name for spec in get_available_ops(ctx.task_type)]
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
        step = make_step(op_name, self.ctx, self.rng)
        if step is None:
            return None
        params = copy.deepcopy(step.params)
        if op_name == "JoinTable" and self.ctx.task_type != "rec" and self.ctx.aux_dfs:
            aux = self.rng.choice(self.ctx.aux_dfs)
            params["aux_df"] = f"${aux}"
            join_keys = self.ctx.tables.get(aux).join_keys_to_primary if aux in self.ctx.tables else []
            params["key_col"] = join_keys[0]["primary"] if join_keys else self.ctx.id_col
            if params.get("prefix"):
                params["prefix"] = aux.upper()[:8]
        elif op_name == "ConcatTable" and self.ctx.aux_dfs:
            aux = self.rng.choice(self.ctx.aux_dfs)
            params["other_dfs"] = [f"${aux}"]
        param_space = get_param_space(get_op(op_name), self.ctx)

        # Optionally perturb one key in the shared context-aware parameter space.
        keys = [k for k, v in param_space.items() if k != "source_tables" and v]
        if keys:
            key = self.rng.choice(keys)
            choices = param_space[key]
            if isinstance(choices, list) and choices:
                if key in {"cols", "source_cols", "group_cols", "agg_cols", "feature_cols"}:
                    flat_choices = [c for c in choices if isinstance(c, str)]
                    if flat_choices:
                        n_cols = self.rng.randint(1, min(3, len(flat_choices)))
                        params[key] = self.rng.sample(flat_choices, n_cols)
                else:
                    params[key] = copy.deepcopy(self.rng.choice(choices))

        return Repair(op_name=op_name, target=step.target, params=params)


__all__ = ["ParameterSampler"]
