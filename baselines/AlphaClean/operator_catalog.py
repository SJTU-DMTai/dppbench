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
    operators_for_task,
    valid_targets_for,
)

__all__ = [
    "CATALOG",
    "OpCategory",
    "OpSpec",
    "operators_for_task",
    "operators_by_category",
    "valid_targets_for",
]
