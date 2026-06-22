"""Pipeline evaluator that drives downstream model training.

Reuses ``baselines.common.executor.TrainingExecutor`` so that SAGA's fitness
signal is exactly the same metric that the rest of the project optimises
(LightGBM AUC for tabular tasks, DIN AUC for recommendation tasks).
"""
from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

from .pipeline import Pipeline

# Ensure the repo root is importable so ``baselines.common.executor`` resolves
# regardless of how this module is loaded (script vs. package).
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from baselines.common.executor import TrainingExecutor  # noqa: E402


@dataclass
class EvaluationResult:
    fitness: float
    success: bool
    metrics: dict = field(default_factory=dict)
    error: Optional[str] = None
    duration_seconds: float = 0.0


class PipelineEvaluator:
    """Evaluate pipelines by running the actual downstream training.

    Caches results by pipeline YAML hash so identical pipelines (e.g. produced
    by GA crossover) are not retrained.
    """

    def __init__(self, task_dir: str, data_name: str, data_dir=None,
                 metric_key: str = "auc", verbose: bool = False,
                 device: str = "cpu"):
        self.task_dir = task_dir
        self.data_name = data_name
        self.data_dir = data_dir
        self.metric_key = metric_key
        self.verbose = verbose
        self.device = device
        self._executor = TrainingExecutor(task_dir, data_name=data_name, data_dir=data_dir, device=device)
        self._cache: dict[str, EvaluationResult] = {}

    @property
    def task_type(self) -> str:
        return self._executor.task_type

    def get_data_summary(self) -> dict:
        return self._executor.get_data_summary()

    def evaluate(self, pipeline: Pipeline) -> EvaluationResult:
        key = pipeline.hash()
        if key in self._cache:
            return self._cache[key]

        yaml_text = pipeline.to_yaml()
        self._executor.write_yaml(yaml_text)

        t0 = time.time()
        try:
            result = self._executor.run_training()
        except Exception as e:  # pragma: no cover - safety net
            result = {
                "success": False,
                "metrics": {},
                "error": f"{type(e).__name__}: {e}",
            }
        dur = time.time() - t0

        success = bool(result.get("success"))
        metrics = result.get("metrics") or {}
        if success:
            fitness = float(metrics.get(self.metric_key, metrics.get("auc", 0.0)))
        else:
            fitness = -math.inf

        ev = EvaluationResult(
            fitness=fitness,
            success=success,
            metrics=metrics,
            error=result.get("error"),
            duration_seconds=dur,
        )
        self._cache[key] = ev
        if self.verbose:
            if success:
                print(f"  [eval] fitness={fitness:.4f}  ({dur:.1f}s)  steps={len(pipeline)}")
            else:
                err_short = (result.get("error") or "")[:200].replace("\n", " ")
                print(f"  [eval] FAILED ({dur:.1f}s)  steps={len(pipeline)}  err={err_short}")
        return ev

    @property
    def n_unique_evaluations(self) -> int:
        return len(self._cache)

    def set_fast_mode(self, enabled: bool, overrides: dict = None) -> None:
        """Toggle search-time fast training on the underlying executor.

        Clears the result cache so identical pipelines are re-evaluated under
        the new mode rather than returning a stale (e.g. fast) score during a
        full evaluation pass.
        """
        self._executor.set_fast_mode(enabled, overrides)
        self._cache.clear()
