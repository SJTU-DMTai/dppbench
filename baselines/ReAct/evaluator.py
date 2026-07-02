"""Final-pipeline evaluator for ReAct.

Re-uses :class:`CtxPipeEvaluator` so all baselines share identical downstream
training (LightGBM/binary AUC for tabular, DIN/AUC for rec). The
``evaluate_for_agent`` wrapper is the contract the agent loop uses to score
each per-turn pipeline; it never raises so a bad turn only costs an error
string in the observation.
"""
from __future__ import annotations

from typing import Optional

from baselines.CtxPipe.evaluator import CtxPipeEvaluator, EvaluationResult
from baselines.common.pipeline import Pipeline


class ReActEvaluator(CtxPipeEvaluator):
    """Thin alias of :class:`CtxPipeEvaluator` for the ReAct baseline."""

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

        Returns ``(fitness, metrics, error)``; ``fitness`` is ``None`` when
        the pipeline failed structurally or numerically. ``metrics`` is the
        full dict returned by the trainer (val-set keys for tabular,
        held-out eval split keys for rec) so the agent can reflect on every
        downstream signal it has, not just the primary fitness.
        """
        try:
            res = self.evaluate(pipeline)
        except Exception as e:  # pragma: no cover - safety net
            return None, {}, f"{type(e).__name__}: {e}"
        if not res.success:
            return None, dict(res.metrics or {}), res.error
        return float(res.fitness), dict(res.metrics or {}), None


__all__ = ["ReActEvaluator", "EvaluationResult"]
