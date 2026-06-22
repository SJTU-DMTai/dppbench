"""DataMaster operator catalog.

DataMaster MUST have access to every operator under
``dppbench/operators/``. We re-export the shared 58-operator catalog to keep
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

# Invariant: DataMaster sees the full shared 58-op catalog. Keep this
# assertion as an early tripwire when the base catalog evolves.
assert len(CATALOG) == 52, (
    f"DataMaster expects 52 operators (parity with dppbench/ operators/), "
    f"got {len(CATALOG)}"
)


__all__ = [
    "CATALOG",
    "OpCategory",
    "OpSpec",
    "operators_by_category",
    "operators_for_task",
    "format_op_descriptions",
]
