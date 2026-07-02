"""BAT downstream evaluator.

Thin alias of :class:`CtxPipeEvaluator` exposing
``evaluate_for_agent(pipeline) -> (fitness, metrics, error)``, used inside
the MCTS reward model. Identical mechanics to DeepPrep/DataMaster
evaluators (``small_n`` subsampling + LightGBM/DIN AUC), renamed for
log-readability.
"""
from __future__ import annotations

from typing import Optional

from baselines.CtxPipe.evaluator import CtxPipeEvaluator, EvaluationResult
from baselines.common.pipeline import Pipeline


class BATEvaluator(CtxPipeEvaluator):
    """Downstream evaluator for BAT's reward model."""

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
        """Run downstream training and return ``(fitness, metrics, error)``.

        Wraps :meth:`evaluate` with try/except so any failure surfaces as
        an error string rather than aborting the search.
        """
        try:
            res = self.evaluate(pipeline)
        except Exception as e:  # pragma: no cover -- safety net
            return None, {}, f"{type(e).__name__}: {e}"
        if not res.success:
            return None, dict(res.metrics or {}), res.error
        return float(res.fitness), dict(res.metrics or {}), None


__all__ = ["BATEvaluator", "EvaluationResult"]
