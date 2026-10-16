"""Slot planner for the DiffPrep continuous pipeline.

A slot corresponds to one preprocessing-function family in the original
DiffPrep formulation.  In DPPBench the families are the public operator
categories exposed by :mod:`baselines.common.pipeline`; each slot contains the
operators from the shared DPPBench operator space that are applicable to the
current task/context, plus an ``Identity`` no-op for optional slots.
"""
from __future__ import annotations

import random as _random
from dataclasses import dataclass
from typing import List, Optional

from baselines.common.data_ctx import DataContext
from baselines.common.pipeline import PipelineStep, get_category_order, make_step
from baselines.common.utils import get_available_ops

from .operator_catalog import OpCategory, is_mandatory, slot_kind_of
from .soft_ops import has_soft_impl


IDENTITY = "Identity"


@dataclass
class Slot:
    index: int
    category: OpCategory
    candidates: List[str]
    kind: str
    mandatory: bool = False
    forced_op: Optional[str] = None

    @property
    def n_candidates(self) -> int:
        return len(self.candidates)


def make_slots(
    task_type: str,
    ctx: DataContext,
    max_pipeline_len: Optional[int] = None,
) -> List[Slot]:
    """Build DiffPrep slots from the shared DPPBench operator space."""
    slots: List[Slot] = []
    max_len = int(max_pipeline_len) if max_pipeline_len else None

    for cat in get_category_order(task_type):
        cands = _candidates_in_category(cat, task_type, ctx)
        if not cands:
            continue

        mandatory_ops = [n for n in cands if is_mandatory(n, ctx.task_type)]
        forced_op = mandatory_ops[0] if mandatory_ops else None
        mandatory_slot = forced_op is not None

        # Unsupported dataframe operations cannot be learned by a tensor gate.
        # Retain required structural adapters only; learn supported soft ops.
        if mandatory_slot and slot_kind_of(forced_op) == 'hard':
            cands = [forced_op]
        else:
            cands = [n for n in cands if slot_kind_of(n) == 'soft' and has_soft_impl(n)]
            if not cands:
                continue

        if max_len is not None and len(slots) >= max_len and not mandatory_slot:
            continue

        candidates = cands if mandatory_slot else cands + [IDENTITY]
        kind = "hard" if any(slot_kind_of(n) == "hard" for n in cands) else "soft"

        slots.append(Slot(
            index=len(slots),
            category=cat,
            candidates=candidates,
            kind=kind,
            mandatory=mandatory_slot,
            forced_op=forced_op,
        ))

    return slots


def diffprep_make_step(op_name: str, ctx: DataContext, rng: _random.Random) -> Optional[PipelineStep]:
    """Construct a concrete pipeline step through the common factory.

    Keeping this as a tiny wrapper preserves DiffPrep's local API while avoiding
    duplicated operator-specific parameter logic.  It also guarantees that the
    generated parameters match the real DPPBench operator constructors.
    """
    if op_name == IDENTITY:
        return None
    return make_step(op_name, ctx, rng)


def _candidates_in_category(cat: OpCategory, task_type: str, ctx: DataContext) -> List[str]:
    out: List[str] = []
    rng = _random.Random(0)
    for spec in get_available_ops(task_type):
        if spec.category != cat:
            continue
        if diffprep_make_step(spec.op_name, ctx, rng) is None:
            continue
        out.append(spec.op_name)
    return out


__all__ = [
    "Slot",
    "IDENTITY",
    "make_slots",
    "diffprep_make_step",
]
