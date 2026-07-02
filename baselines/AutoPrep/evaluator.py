"""Evaluator for Auto-Prep candidates.

Subclasses ``CtxPipeEvaluator`` so we get the small-data subsampling and the
same downstream training stack (LightGBM AUC for tabular, DIN AUC for rec)
used elsewhere in dppbench. Adds a single ``evaluate_for_agent`` wrapper
that returns ``(fitness, metrics, error_str)`` triples and never raises.
"""
from __future__ import annotations

from typing import Optional

from baselines.CtxPipe.evaluator import CtxPipeEvaluator, EvaluationResult
from baselines.common.pipeline import Pipeline


class AutoPrepEvaluator(CtxPipeEvaluator):
    """Pipeline evaluator with small-N subsampling and a safe agent wrapper."""

    def __init__(
        self,
        task_dir: str,
        data_name: str,
        data_dir=None,
        metric_key: str = "auc",
        verbose: bool = False,
        small_n: Optional[int] = 3000,
        seed: int = 42,
        device: str = "cpu",
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
        )

    def evaluate_for_agent(
        self,
        pipeline: Pipeline,
    ) -> tuple[float, dict, Optional[str]]:
        """Return ``(fitness, metrics, error)``.

        ``fitness`` is the chosen metric (default AUC). Failures are caught
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
