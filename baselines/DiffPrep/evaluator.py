"""Pipeline evaluator for DiffPrep.

Re-exports :class:`baselines.common.search_evaluator.SearchPipelineEvaluator` so DiffPrep
gets the same ``small_n`` subsampling helper. We keep a thin alias so
downstream code can import :class:`DiffPrepEvaluator` symmetrically with
SAGA / CtxPipe.
"""
from __future__ import annotations

from baselines.common.search_evaluator import SearchPipelineEvaluator as DiffPrepEvaluator
from baselines.common.evaluator import EvaluationResult

__all__ = ["DiffPrepEvaluator", "EvaluationResult"]
