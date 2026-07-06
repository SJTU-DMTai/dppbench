"""SAGA operator catalog adapter.

The shared operator metadata now lives in
``baselines.common.op_spec``. SAGA re-exports it for backward
compatibility and to keep older intra-SAGA imports stable.
"""
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

__all__ = [
    "CATALOG",
    "OpCategory",
    "OpSpec",
    "operators_by_category",
    "is_mandatory",
    "needs_context",
    "operators_for_task",
    "valid_targets_for",
]
