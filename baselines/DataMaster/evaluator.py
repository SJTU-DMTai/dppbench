"""Final-pipeline evaluator for DataMaster.

Re-uses :class:`SearchPipelineEvaluator` so all baselines share identical downstream
training (LightGBM/binary AUC for tabular, DIN/AUC for rec). The
``evaluate_for_agent`` wrapper is the contract the agent loop uses to score
black-node pipelines; it never raises so a bad node only costs ``visits=1``
plus an error string in the GlobalMemory.
"""
from __future__ import annotations

from typing import Optional

from baselines.common.evaluator import EvaluationResult
from baselines.common.search_evaluator import SearchPipelineEvaluator
from baselines.common.pipeline import Pipeline


class DataMasterEvaluator(SearchPipelineEvaluator):
    """Thin alias of :class:`SearchPipelineEvaluator` for the DataMaster baseline."""

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

        Returns ``(fitness, metrics, error)``; ``fitness`` is ``None`` when
        the pipeline failed structurally or numerically.
        """
        try:
            res = self.evaluate(pipeline)
        except Exception as e:  # pragma: no cover - safety net
            return None, {}, f"{type(e).__name__}: {e}"
        if not res.success:
            return None, dict(res.metrics or {}), res.error
        return float(res.fitness), dict(res.metrics or {}), None


__all__ = ["DataMasterEvaluator", "EvaluationResult"]
