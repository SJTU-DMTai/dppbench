"""AlphaClean operator catalog adapter.

AlphaClean uses the shared DPPBench operator universe exposed by
``baselines.common.op_spec`` so it sees the same search space and metadata as
other baselines.
"""
from __future__ import annotations

from baselines.common.op_spec import (
    CATALOG,
    OpCategory,
    OpSpec,
    operators_by_category,
    valid_targets_for,
)
from baselines.common.utils import get_available_ops


def operators_for_task(task_type: str, include_custom: bool = False) -> list[str]:
    return [
        spec.op_name
        for spec in get_available_ops(task_type, include_custom=include_custom)
    ]

__all__ = [
    "CATALOG",
    "OpCategory",
    "OpSpec",
    "operators_for_task",
    "operators_by_category",
    "valid_targets_for",
]
