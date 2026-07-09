"""ReAct operator catalog adapter over the shared operator catalog."""
from __future__ import annotations

from baselines.common.op_spec import (
    CATALOG,
    OpCategory,
    OpSpec,
    operators_by_category,
    is_mandatory,
    needs_context,
    valid_targets_for,
)
from baselines.common.utils import get_available_ops
from baselines.DeepPrep.operator_catalog import format_op_descriptions


def operators_for_task(task_type: str, include_custom: bool = False) -> list[str]:
    return [
        spec.op_name
        for spec in get_available_ops(task_type, include_custom=include_custom)
    ]

# Invariant: this adapter re-exports the full shared catalog.

__all__ = [
    "CATALOG",
    "OpCategory",
    "OpSpec",
    "operators_by_category",
    "operators_for_task",
    "format_op_descriptions",
]
