"""Discretize the trained continuous pipeline into a YAML-serialisable
:class:`baselines.common.pipeline.Pipeline`.

For each slot we ``argmax`` over the slot's beta row to pick a single
candidate operator. Mandatory slots already have one-hot beta so the
argmax simply returns the forced op. ``IDENTITY`` picks become no-ops.

After collecting the per-slot ops we apply :func:`baselines.common.
pipeline.repair` to inject mandatory operators that may have
been omitted (e.g. ``LabelEncode`` + ``HandleMV`` for tabular data).
"""
from __future__ import annotations

import random as _random
import copy
from collections import Counter
from typing import Dict, List, Optional

import torch

from baselines.common.data_ctx import DataContext
from baselines.common.pipeline import Pipeline
from baselines.common.pipeline import assign_dag_structure, check_pipeline, repair

from .search_space import ContinuousPipeline
from .slot_planner import IDENTITY, Slot, diffprep_make_step


def slot_order(continuous: ContinuousPipeline) -> List[int]:
    """Return the discrete slot order implied by DiffPrep-Flex alpha."""
    if not continuous.flex:
        return list(range(len(continuous.slots)))
    with torch.no_grad():
        alpha = continuous.alpha().detach().cpu()
        if alpha.dim() == 3:
            alpha = alpha.mean(dim=0)
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
        if beta.dim() == 3:
            beta = beta.mean(dim=1)
    out: List[str] = []
    for i in slot_order(continuous):
        slot = continuous.slots[i]
        idx = int(beta[i, : slot.n_candidates].argmax().item())
        out.append(slot.candidates[idx])
    return out


def featurewise_op_names(continuous: ContinuousPipeline) -> Dict[int, List[str]]:
    """Return the official DiffPrep-style argmax choice for every feature."""
    with torch.no_grad():
        beta = continuous.beta().detach().cpu()
    choices: Dict[int, List[str]] = {}
    for slot_idx, slot in enumerate(continuous.slots):
        choices[slot_idx] = [
            slot.candidates[int(beta[slot_idx, feature_idx, :slot.n_candidates].argmax())]
            for feature_idx in range(beta.shape[1])
        ]
    return choices


def discretize(
    continuous: ContinuousPipeline,
    slots: List[Slot],
    ctx: DataContext,
    rng: Optional[_random.Random] = None,
    structural_steps=None,
) -> Pipeline:
    """Materialise a :class:`Pipeline` from the continuous parameters."""
    rng = rng or _random.Random(0)
    order = slot_order(continuous)
    per_feature = featurewise_op_names(continuous)
    feature_names = list(continuous.feature_names)

    pipe = Pipeline(steps=copy.deepcopy(structural_steps or []))
    projection: dict[str, object] = {}
    for slot_idx in order:
        slot = slots[slot_idx]
        selected = per_feature[slot_idx]
        groups: dict[str, list[str]] = {}
        for feature_idx, op_name in enumerate(selected):
            if op_name == IDENTITY:
                continue
            col = feature_names[feature_idx] if feature_idx < len(feature_names) else None
            if col is not None:
                groups.setdefault(op_name, []).append(col)

        materialized = False
        for op_name, cols in groups.items():
            if op_name == "ExtractDateTimeFeature":
                raise ValueError("Legacy DiffPrep datetime relaxation is unsupported; rerun pipeline search")
            step = diffprep_make_step(op_name, ctx, rng)
            if step is None or "cols" not in step.params:
                continue
            step.params["cols"] = cols
            pipe.steps.append(step)
            materialized = True

        if materialized:
            projection[str(slot_idx)] = groups
            continue

        # Some shared operators are necessarily dataframe-wide. For those,
        # project feature-wise choices deterministically by majority vote and
        # retain the complete choices in metadata.
        global_choices = [name for name in selected if name != IDENTITY]
        if not global_choices:
            continue
        counts = Counter(global_choices)
        op_name = max(counts, key=lambda name: (counts[name], -slot.candidates.index(name)))
        step = diffprep_make_step(op_name, ctx, rng)
        if step is not None:
            pipe.steps.append(step)
            projection[str(slot_idx)] = {
                "mode": op_name,
                "feature_choices": selected,
            }

    repair(pipe, ctx.task_type, ctx, sort_steps=not continuous.flex, rng=rng)
    assign_dag_structure(pipe, ctx, rng)

    flex_order_repaired = False
    if continuous.flex:
        report = check_pipeline(pipe, ctx)
        if not report.valid:
            # Operator-level and DAG legality remain part of the benchmark
            # contract. Preserve metadata if structural repair is needed.
            repair(pipe, ctx.task_type, ctx, sort_steps=True, rng=rng)
            assign_dag_structure(pipe, ctx, rng)
            flex_order_repaired = True

    pipe.metadata = dict(getattr(pipe, "metadata", {}) or {})
    pipe.metadata["diffprep_slot_order"] = order
    pipe.metadata["diffprep_flex_order_repaired"] = flex_order_repaired
    pipe.metadata["diffprep_featurewise_projection"] = projection
    return pipe


__all__ = ["argmax_op_names", "featurewise_op_names", "slot_order", "discretize"]
