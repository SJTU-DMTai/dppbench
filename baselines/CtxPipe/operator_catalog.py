"""CtxPipe operator catalog.

CtxPipe re-exports the shared operator catalog so all baselines share an
identical executable operator universe. This guarantees that the
RL agent can choose from **all** operator categories (cleaning, encoding,
feature generation, sequence, split, sampling, ...) and that both tabular and
recommendation tasks are covered.
"""
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
