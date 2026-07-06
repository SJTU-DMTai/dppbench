"""ReAct operator catalog adapter over the shared operator catalog."""
from __future__ import annotations

from baselines.common.op_spec import (
    CATALOG,
    OpCategory,
    OpSpec,
    operators_by_category,
    is_mandatory,
    needs_context,
    operators_for_task,
    valid_targets_for,
)
from baselines.DeepPrep.operator_catalog import format_op_descriptions

# Invariant: this adapter re-exports the full shared catalog.

__all__ = [
    "CATALOG",
    "OpCategory",
    "OpSpec",
    "operators_by_category",
    "operators_for_task",
    "format_op_descriptions",
]
