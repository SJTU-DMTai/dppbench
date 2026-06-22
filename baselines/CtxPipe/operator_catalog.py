"""CtxPipe operator catalog.

CtxPipe re-exports the shared 58-operator catalog so all baselines share an
identical executable operator universe. This guarantees that the
RL agent can choose from **all** operator categories (cleaning, encoding,
feature generation, sequence, split, sampling, ...) and that both tabular and
recommendation tasks are covered.
"""
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
