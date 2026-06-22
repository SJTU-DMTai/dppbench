"""Final-pipeline evaluator for DeepPrep.

Used both:
* at the very end of ``DeepPrep.run()`` to score the pipeline (full data),
* and (when downstream feedback is enabled) inside the agent loop after
  every successful ``<solution>`` to give the LLM a real downstream metric.
"""
from __future__ import annotations

from typing import Optional

from baselines.CtxPipe.evaluator import CtxPipeEvaluator, EvaluationResult
from baselines.SAGA.pipeline import Pipeline


class DeepPrepEvaluator(CtxPipeEvaluator):
    """Thin alias of :class:`CtxPipeEvaluator` exposed under a DeepPrep-named
    class so users (and external benchmark harnesses) can identify the
    baseline easily. Functionally identical: it runs the downstream
    LightGBM (tabular) or DIN (rec) training and returns the AUC.
    """

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
        """Run downstream training for the agent loop.

        Wraps :meth:`evaluate` with a try/except so any failure surfaces as
        an ``error`` string rather than aborting the agent. Returns
        ``(fitness, metrics, error)``. ``fitness`` is ``None`` on failure.
        """
        try:
            res = self.evaluate(pipeline)
        except Exception as e:  # pragma: no cover - safety net
            return None, {}, f"{type(e).__name__}: {e}"
        if not res.success:
            return None, dict(res.metrics or {}), res.error
        return float(res.fitness), dict(res.metrics or {}), None


__all__ = ["DeepPrepEvaluator", "EvaluationResult"]
