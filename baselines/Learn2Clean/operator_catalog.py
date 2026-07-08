"""Learn2Clean operator catalog adapter.

Learn2Clean uses the shared operator universe directly from
:mod:`baselines.common.op_spec`.
"""
from __future__ import annotations

from baselines.common.op_spec import (
    CATALOG,
    OpCategory,
    OpSpec,
    operators_by_category,
    operators_for_task,
)

__all__ = [
    "CATALOG",
    "OpCategory",
    "OpSpec",
    "operators_for_task",
    "operators_by_category",
]
