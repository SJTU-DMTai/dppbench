"""Discretize the trained continuous pipeline into a YAML-serialisable
:class:`baselines.common.pipeline.Pipeline`.

For each slot we ``argmax`` over the slot's beta row to pick a single
candidate operator. Mandatory slots already have one-hot beta so the
argmax simply returns the forced op. ``IDENTITY`` picks become no-ops.

After collecting the per-slot ops we apply :func:`baselines.common.
pipeline_constraints.repair` to inject mandatory operators that may have
been omitted (e.g. ``LabelEncode`` + ``HandleMV`` for tabular tail) and
to reorder according to the canonical category ranks.
"""
from __future__ import annotations

import random as _random
from typing import List, Optional

import torch

from baselines.common.pipeline import DataContext, Pipeline
from baselines.common.pipeline_constraints import repair

from .search_space import ContinuousPipeline
from .slot_planner import IDENTITY, Slot, diffprep_make_step


def slot_order(continuous: ContinuousPipeline) -> List[int]:
    """Return the discrete slot order implied by DiffPrep-Flex alpha."""
    if not continuous.flex:
        return list(range(len(continuous.slots)))
    with torch.no_grad():
        alpha = continuous.alpha().detach().cpu()
    remaining = set(range(alpha.shape[1]))
    order: List[int] = []
    for row in range(alpha.shape[0]):
        col = max(remaining, key=lambda c: float(alpha[row, c]))
        order.append(int(col))
        remaining.remove(col)
    order.extend(sorted(remaining))
    return order


def argmax_op_names(continuous: ContinuousPipeline) -> List[str]:
    """Return one operator name per slot via argmax of the beta row."""
    with torch.no_grad():
        beta = continuous.beta()
    out: List[str] = []
    for i in slot_order(continuous):
        slot = continuous.slots[i]
        idx = int(beta[i, : slot.n_candidates].argmax().item())
        out.append(slot.candidates[idx])
    return out


def discretize(
    continuous: ContinuousPipeline,
    slots: List[Slot],
    ctx: DataContext,
    rng: Optional[_random.Random] = None,
) -> Pipeline:
    """Materialise a :class:`Pipeline` from the continuous parameters."""
    rng = rng or _random.Random(0)
    order = slot_order(continuous)
    op_names = argmax_op_names(continuous)

    pipe = Pipeline(steps=[])
    for op_name in op_names:
        if op_name == IDENTITY:
            continue
        step = diffprep_make_step(op_name, ctx, rng)
        if step is not None:
            pipe.steps.append(step)

    repair(pipe, ctx.task_type, ctx)
    pipe.metadata = dict(getattr(pipe, "metadata", {}) or {})
    pipe.metadata["diffprep_slot_order"] = order
    return pipe


__all__ = ["argmax_op_names", "slot_order", "discretize"]
