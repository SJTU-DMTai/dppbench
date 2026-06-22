"""Learn2Clean operator catalog adapter.

Learn2Clean uses the shared 58-operator universe through DiffPrep's adapter because
that layer also exposes the ``SLOT_KIND`` annotation used by shared sampling and
repair utilities.
"""
from __future__ import annotations

from baselines.DiffPrep.operator_catalog import (
    CATALOG,
    OpCategory,
    OpSpec,
    SLOT_KIND,
    operators_by_category,
    operators_for_task,
    slot_kind_of,
)

__all__ = [
    "CATALOG",
    "OpCategory",
    "OpSpec",
    "operators_for_task",
    "operators_by_category",
    "SLOT_KIND",
    "slot_kind_of",
]
