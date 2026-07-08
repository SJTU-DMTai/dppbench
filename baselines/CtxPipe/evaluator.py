"""Backward-compatible CtxPipe evaluator alias.

The reusable search-time evaluator lives in ``baselines.common``. This module
keeps older imports working for CtxPipe-specific code.
"""
from baselines.common.evaluator import EvaluationResult
from baselines.common.search_evaluator import SearchPipelineEvaluator


class CtxPipeEvaluator(SearchPipelineEvaluator):
    """Compatibility alias for the common search-time evaluator."""


__all__ = ["CtxPipeEvaluator", "EvaluationResult", "SearchPipelineEvaluator"]
