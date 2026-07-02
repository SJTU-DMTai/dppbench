"""Compatibility wrapper for shared pipeline legality utilities."""
from baselines.common.pipeline_constraints import (
    _REC_ORDER,
    _TABULAR_ORDER,
    ensure_tabular_tail,
    is_legal,
    repair,
)

__all__ = [
    "_REC_ORDER",
    "_TABULAR_ORDER",
    "ensure_tabular_tail",
    "is_legal",
    "repair",
]
