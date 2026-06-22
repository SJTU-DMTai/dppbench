"""Final-pipeline evaluator for SPIO.

Re-uses :class:`CtxPipeEvaluator` so SPIO shares identical downstream
training (LightGBM/binary AUC for tabular, DIN/AUC for rec) with the other
baselines. ``evaluate_for_agent`` is the contract used by the per-stage
candidate selection loop; it never raises so a bad candidate just costs an
error string instead of crashing the search.
"""
from __future__ import annotations

from typing import Optional

from baselines.CtxPipe.evaluator import CtxPipeEvaluator, EvaluationResult
from baselines.SAGA.pipeline import Pipeline


class SPIOEvaluator(CtxPipeEvaluator):
    """Thin alias of :class:`CtxPipeEvaluator` for the SPIO baseline."""

    def __init__(
        self,
        task_dir: str,
        data_name: str,
        data_dir: Optional[str] = None,
        metric_key: str = "auc",
        verbose: bool = False,
        small_n: Optional[int] = None,
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

    # ------------------------------------------------------------------
    def evaluate_for_agent(
        self, pipeline: Pipeline
    ) -> tuple[Optional[float], dict, Optional[str]]:
        """Run downstream training for a candidate pipeline.

        Returns ``(fitness, metrics, error)``. ``fitness`` is ``None`` if
        the pipeline failed structurally or numerically.
        """
        try:
            res = self.evaluate(pipeline)
        except Exception as e:  # pragma: no cover - safety net
            return None, {}, f"{type(e).__name__}: {e}"
        if not res.success:
            return None, dict(res.metrics or {}), res.error
        return float(res.fitness), dict(res.metrics or {}), None


__all__ = ["SPIOEvaluator", "EvaluationResult"]
