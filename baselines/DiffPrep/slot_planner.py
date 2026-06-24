"""Slot planner for the DiffPrep continuous pipeline.

A *slot* is a pipeline position belonging to a single :class:`OpCategory`. Each
slot has a list of candidate operator specs (drawn from the DiffPrep catalog)
plus a synthetic ``Identity`` candidate so the search can choose to skip the
slot. The planner uses the same canonical category orderings as SAGA
(``_TABULAR_ORDER`` and ``_REC_ORDER``) so the discrete pipelines emitted by
DiffPrep match SAGA's pipeline layout convention.
"""
from __future__ import annotations

import random as _random
from dataclasses import dataclass, field
from typing import List, Optional

from baselines.SAGA.pipeline import DataContext, PipelineStep, build_default_params
from baselines.SAGA.pipeline_constraints import _TABULAR_ORDER, _REC_ORDER

from .operator_catalog import CATALOG, OpCategory, slot_kind_of


# ---------------------------------------------------------------------------
# Slot dataclass
# ---------------------------------------------------------------------------
IDENTITY = "Identity"  # sentinel name; never appears in CATALOG.


@dataclass
class Slot:
    index: int                         # position in the pipeline
    category: OpCategory               # category this slot fills
    candidates: List[str]              # operator names (may include IDENTITY)
    kind: str                          # "soft" | "hard" (slot-level)
    mandatory: bool = False            # if True, the slot is forced to a fixed op
    forced_op: Optional[str] = None    # name of the forced op (only when mandatory)

    @property
    def n_candidates(self) -> int:
        return len(self.candidates)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def make_slots(task_type: str, ctx: DataContext) -> List[Slot]:
    """Build the canonical list of pipeline slots for ``task_type``.

    For tabular the order matches :data:`_TABULAR_ORDER`. For rec it matches
    :data:`_REC_ORDER`. Categories with no applicable operator under the
    current :class:`DataContext` are skipped.
    """
    order = _REC_ORDER if task_type == "rec" else _TABULAR_ORDER
    slots: List[Slot] = []
    idx = 0

    for cat in order:
        cands = _candidates_in_category(cat, task_type, ctx)
        if not cands:
            continue

        # Mandatory categories: pick the unique mandatory op as the forced
        # candidate (still keep IDENTITY if user disables forcing).
        mandatory_ops = [n for n in cands if CATALOG[n].mandatory]
        forced_op = mandatory_ops[0] if mandatory_ops else None
        is_mandatory = forced_op is not None

        # Add IDENTITY as a no-op for non-mandatory slots.
        if not is_mandatory:
            cands_with_id = cands + [IDENTITY]
        else:
            # mandatory: still include the forced op only (keeps shape stable)
            cands_with_id = cands

        # Determine slot kind.  Mandatory rec ops (JoinTable, CreateSequence)
        # are structural -> hard slot.  Otherwise
        # the slot is hard if *any* candidate is hard (mixed slots can't be
        # safely averaged).
        if is_mandatory:
            kind = "hard"
        elif all(slot_kind_of(n) == "soft" for n in cands):
            kind = "soft"
        else:
            kind = "hard"

        slots.append(Slot(
            index=idx,
            category=cat,
            candidates=cands_with_id,
            kind=kind,
            mandatory=is_mandatory,
            forced_op=forced_op,
        ))
        idx += 1

    return slots


def diffprep_make_step(op_name: str, ctx: DataContext, rng: _random.Random) -> Optional[PipelineStep]:
    """Construct a :class:`PipelineStep` for ``op_name``.

    Uses :func:`baselines.SAGA.pipeline.build_default_params` for most
    operators and keeps a few DiffPrep-specific context defaults for operators
    that need a tensor-friendly or slot-planner-specific parameterization.
    """
    if op_name == IDENTITY:
        return None
    spec = CATALOG.get(op_name)
    if spec is None:
        return None

    if op_name in _DIFFPREP_CUSTOM_DEFAULT_OPS:
        params = _diffprep_custom_default_params(op_name, ctx)
        if params is None:
            return None
    else:
        params = build_default_params(op_name, ctx, rng)
        if params is None:
            return None

    target = _default_target_for(op_name, ctx.task_type)
    return PipelineStep(op=op_name, target=target, params=params)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
_DIFFPREP_CUSTOM_DEFAULT_OPS = {
    "ScaleFeature", "HandleOutlier", "HandleError", "JoinTable", "ParseDate",
    "CreateLagFeature", "CreateRollingFeature", "ResampleTimeSeries",
}


def _candidates_in_category(cat: OpCategory, task_type: str, ctx: DataContext) -> List[str]:
    """All operators in ``cat`` that *could* be applied given the context."""
    out: List[str] = []
    rng = _random.Random(0)
    for name, spec in CATALOG.items():
        if spec.category != cat:
            continue
        if spec.task_type not in (task_type, "both"):
            continue
        # Probe whether the op produces a valid step under this context.
        step = diffprep_make_step(name, ctx, rng)
        if step is None:
            continue
        out.append(name)
    return out


def _default_target_for(op_name: str, task_type: str) -> str:
    spec = CATALOG[op_name]
    if spec.valid_targets == ("interaction",):
        return "interaction"
    if task_type == "tabular":
        return "both"
    return "interaction"


def _diffprep_custom_default_params(op_name: str, ctx: DataContext) -> Optional[dict]:
    """DiffPrep-specific defaults for selected shared-catalog operators."""
    if op_name == "ScaleFeature":
        bad = {ctx.target_col, ctx.id_col, ctx.user_col, ctx.item_col, ctx.time_col}
        cols = [c for c in ctx.numeric_cols if c not in bad]
        if not cols:
            return None
        return {"cols": list(cols), "auto_numeric": False, "eps": 1e-6, "out_dtype": "float32"}

    if op_name == "HandleOutlier":
        # Need at least one numeric column to define outlier bounds.
        cols = [c for c in ctx.numeric_cols if c != ctx.target_col][:2]
        if not cols:
            return None
        return {"cols": cols, "action": "delete"}

    if op_name == "HandleError":
        cols = [c for c in ctx.numeric_cols if c != ctx.target_col][:2]
        if not cols:
            return None
        return {"cols": cols, "rule": "numeric", "action": "delete"}

    if op_name == "JoinTable":
        if not ctx.aux_dfs or ctx.id_col is None:
            return None
        # Pick the first aux table; user may override later.
        return {
            "aux_df": f"${ctx.aux_dfs[0]}",
            "key_col": ctx.id_col,
            "how": "left",
        }

    if op_name == "ParseDate":
        # Looks for an integer-typed time column. Without one, skip.
        if ctx.time_col is None:
            return None
        return {
            "cols": [ctx.time_col],
            "mode": "date",
            "out_features": ["year", "month", "day", "days_since_epoch"],
            "drop_original": False,
        }

    if op_name == "CreateLagFeature":
        if ctx.target_col is None or ctx.time_col is None:
            return None
        return {
            "target_col": ctx.target_col,
            "lags": [1],
            "group_cols": None,
            "time_col": ctx.time_col,
        }

    if op_name == "CreateRollingFeature":
        if ctx.target_col is None or ctx.time_col is None:
            return None
        return {
            "target_col": ctx.target_col,
            "windows": [3],
            "aggs": ["mean"],
            "group_cols": None,
            "time_col": ctx.time_col,
        }

    if op_name == "ResampleTimeSeries":
        # Needs a real datetime column.
        if ctx.time_col is None:
            return None
        return {
            "time_col": ctx.time_col,
            "freq": "H",
            "aggs": {},
            "group_cols": [],
            "count_col": None,
        }

    return None


__all__ = [
    "Slot",
    "IDENTITY",
    "make_slots",
    "diffprep_make_step",
]
