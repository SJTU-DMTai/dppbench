"""DataMaster operator catalog.

DataMaster MUST have access to every operator under
``dppbench/operators/``. We re-export the shared operator catalog to keep
this baseline in lock-step with the rest of dppbench.
"""
from __future__ import annotations

from baselines.common.operator_catalog import (
    CATALOG,
    OpCategory,
    OpSpec,
    operators_by_category,
    operators_for_task,
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
