"""Shared downstream evaluator for DeepPrep.

Used both:
* at the very end of ``DeepPrep.run()`` to score the pipeline (full data),
* after DeepPrep inference for DPPBench reporting.
* during search on a fixed-size sample for validation fitness feedback.
"""
from __future__ import annotations

from typing import Optional

from baselines.common.evaluator import EvaluationResult
from baselines.common.search_evaluator import SearchPipelineEvaluator
from baselines.common.pipeline import Pipeline
from baselines.common.budget import BudgetExceededError


class DeepPrepEvaluator(SearchPipelineEvaluator):
    """Thin alias of :class:`SearchPipelineEvaluator` exposed under a DeepPrep-named
    class so users (and external benchmark harnesses) can identify the
    baseline easily. Functionally identical: it runs the downstream
    LightGBM (tabular) or DIN (rec) training and returns task-specific validation fitness.
    """

    def __init__(
        self,
        task_dir: str,
        data_name: str,
        data_dir: Optional[str] = None,
        metric_key: str | None = None,
        verbose: bool = False,
        small_n: Optional[int] = None,
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
        except BudgetExceededError:
            raise
        except Exception as e:  # pragma: no cover - safety net
            return None, {}, f"{type(e).__name__}: {e}"
        if not res.success:
            return None, dict(res.metrics or {}), res.error
        return float(res.fitness), dict(res.metrics or {}), None


__all__ = ["DeepPrepEvaluator", "EvaluationResult"]
