"""Evaluator for Auto-Prep candidates.

Subclasses ``SearchPipelineEvaluator`` so we get the small-data subsampling and the
same downstream training stack and task-configured metric (e.g. RMSE or AUC)
used elsewhere in dppbench. Adds a single ``evaluate_for_agent`` wrapper
that returns ``(fitness, metrics, error_str)`` triples and never raises.
"""
from __future__ import annotations

from typing import Optional

from baselines.common.evaluator import EvaluationResult
from baselines.common.search_evaluator import SearchPipelineEvaluator
from baselines.common.pipeline import Pipeline


class AutoPrepEvaluator(SearchPipelineEvaluator):
    """Pipeline evaluator with small-N subsampling and a safe agent wrapper."""

    def __init__(
        self,
        task_dir: str,
        data_name: str,
        data_dir=None,
        metric_key: Optional[str] = None,
        verbose: bool = False,
        small_n: Optional[int] = 3000,
        seed: int = 42,
        device: str = "cpu",
        model_name: Optional[str] = None,
    ) -> None:
        super().__init__(
            task_dir=task_dir,
            data_name=data_name,
            data_dir=data_dir,
            metric_key=metric_key,
            verbose=verbose,
            small_n=small_n,
            seed=seed,
            device=device,
            model_name=model_name,
        )

    def evaluate_for_agent(
        self,
        pipeline: Pipeline,
    ) -> tuple[float, dict, Optional[str]]:
        """Return ``(fitness, metrics, error)``.

        ``fitness`` follows the task metric (loss metrics are negated). Failures are caught
        and returned as ``-inf`` fitness with a short error string so the
        outer search loop can continue without crashing.
        """
        try:
            ev = self.evaluate(pipeline)
        except Exception as e:  # pragma: no cover - safety net
            return float("-inf"), {}, f"{type(e).__name__}: {e}"
        if not ev.success:
            return float("-inf"), ev.metrics or {}, ev.error
        return float(ev.fitness), dict(ev.metrics), None


__all__ = ["AutoPrepEvaluator", "EvaluationResult"]
