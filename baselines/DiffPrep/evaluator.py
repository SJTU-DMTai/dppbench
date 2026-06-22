"""Pipeline evaluator for DiffPrep.

Re-exports :class:`baselines.CtxPipe.evaluator.CtxPipeEvaluator` so DiffPrep
gets the same ``small_n`` subsampling helper. We keep a thin alias so
downstream code can import :class:`DiffPrepEvaluator` symmetrically with
SAGA / CtxPipe.
"""
from __future__ import annotations

from baselines.CtxPipe.evaluator import CtxPipeEvaluator as DiffPrepEvaluator
from baselines.SAGA.evaluator import EvaluationResult

__all__ = ["DiffPrepEvaluator", "EvaluationResult"]
