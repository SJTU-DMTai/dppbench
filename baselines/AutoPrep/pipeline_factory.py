"""Auto-Prep parameter factory adapter.

AutoPrep must use the same executable operator contract as every other
baseline.  This module intentionally re-exports the common context-aware
parameter builders instead of maintaining a private copy, so searched operator
parameters are checked against the shared ``OpSpec`` metadata and the real
operator constructors.
"""
from __future__ import annotations

from baselines.common.data_ctx import DataContext
from baselines.common.pipeline import (
    Pipeline,
    PipelineStep,
    build_default_params,
    default_target_for,
    make_step,
    validate_step_params,
)

__all__ = [
    "build_default_params",
    "make_step",
    "default_target_for",
    "validate_step_params",
    "Pipeline",
    "PipelineStep",
    "DataContext",
]
