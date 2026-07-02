"""SAGA operator catalog adapter.

The shared operator metadata now lives in
``baselines.common.operator_catalog``. SAGA re-exports it for backward
compatibility and to keep older intra-SAGA imports stable.
"""
from __future__ import annotations

from baselines.common.operator_catalog import (
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
    "operators_by_category",
    "operators_for_task",
]
