"""ReAct operator catalog adapter over the shared 58-op catalog."""
from __future__ import annotations

from baselines.common.operator_catalog import (
    CATALOG,
    OpCategory,
    OpSpec,
    operators_by_category,
    operators_for_task,
)
from baselines.DeepPrep.operator_catalog import format_op_descriptions

assert len(CATALOG) == 52, (
    f"ReAct expected 52 operators in SAGA CATALOG, got {len(CATALOG)}"
)


__all__ = [
    "CATALOG",
    "OpCategory",
    "OpSpec",
    "operators_by_category",
    "operators_for_task",
    "format_op_descriptions",
]
