"""Pipeline evaluator that drives downstream model training.

Reuses ``baselines.common.executor.TrainingExecutor`` so that each baseline fitness
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

from baselines.common.data_ctx import infer_data_context
from baselines.common.budget import Budget, BudgetExceededError, get_active_budget
from baselines.common.pipeline import Pipeline, PipelineCheckReport, check_pipeline

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
    by search operators) are not retrained.
    """

    executor_class = TrainingExecutor

    def _check_pipeline(self, pipeline, ctx, **kwargs):
        provider = getattr(pipeline, 'get_evaluator_class', None)
        if callable(provider):
            required = provider()
            if not isinstance(self, required):
                return PipelineCheckReport(False, [
                    f'{type(pipeline).__name__} requires {required.__module__}.{required.__name__}; '
                    f'got {type(self).__module__}.{type(self).__name__}'
                ])
        return check_pipeline(pipeline, ctx, **kwargs)

    def __init__(self, task_dir: str, data_name: str, data_dir=None,
                 metric_key: str | None = None, verbose: bool = False,
                 device: str = "cpu", model_name: Optional[str] = None,
                 scoring_method: str = "full_train", budget: Budget | None = None,
                 seed: int | None = None, model_config: dict | None = None):
        self.task_dir = task_dir
        self.data_name = data_name
        self.data_dir = data_dir
        self.metric_key = metric_key
        self.verbose = verbose
        self.device = device
        self.model_name = model_name
        self.scoring_method = scoring_method
        self.budget = budget or get_active_budget()
        self._executor = self.executor_class(
            task_dir,
            data_name=data_name,
            data_dir=data_dir,
            device=device,
            model_name=model_name,
            seed=seed,
            model_config=model_config,
        )
        if self.metric_key is None:
            self.metric_key = self._executor.get_model_config().get("train", {}).get("metrics", ["auc"])[0]
        self._cache: dict[str, EvaluationResult] = {}
        self._data_context = None

    def _get_data_context(self):
        if self._data_context is None:
            data = self.load_data()
            self._data_context = infer_data_context(
                self.data_name,
                data,
                self.task_type,
            )
            # Context contains schema metadata only. Training loads its own
            # instance, so retaining this raw dataset doubles resident data.
            self._executor.clear_data_cache()
        return self._data_context

    @property
    def task_type(self) -> str:
        return self._executor.task_type

    def load_data(self):
        return self._executor.load_data()

    def run_preprocessing(self, pipeline: Pipeline):
        """Run a partial pipeline without downstream model evaluation."""
        report = self._check_pipeline(
            pipeline,
            self._get_data_context(),
            require_mandatory=False,
            budget=self.budget,
        )
        if not report.valid:
            raise ValueError("; ".join(report.errors))
        self._executor.write_yaml(pipeline.to_yaml())
        return self._executor.run_preprocessing()

    def evaluate(self, pipeline: Pipeline, scoring_method: Optional[str] = None) -> EvaluationResult:
        method = scoring_method or self.scoring_method
        key = f"{method}:{pipeline.hash()}"
        if key in self._cache:
            return self._cache[key]

        try:
            if self.budget is not None:
                self.budget.consume_evaluation()
        except BudgetExceededError as exc:
            ev = EvaluationResult(
                fitness=-math.inf, success=False, metrics={},
                error=f"BudgetExceededError: {exc}", duration_seconds=0.0,
            )
            self._cache[key] = ev
            return ev

        report = self._check_pipeline(
            pipeline, self._get_data_context(), budget=self.budget,
        )
        if not report.valid:
            ev = EvaluationResult(
                fitness=-math.inf,
                success=False,
                metrics={},
                error="; ".join(report.errors),
                duration_seconds=0.0,
            )
            self._cache[key] = ev
            if self.verbose:
                err_short = ev.error[:200].replace("\n", " ") if ev.error else ""
                print(f"  [eval:precheck] FAILED  steps={len(pipeline)}  err={err_short}")
            return ev

        yaml_text = pipeline.to_yaml()
        self._executor.write_yaml(yaml_text)

        t0 = time.time()
        try:
            result = self._run_scoring_method(method)
        except Exception as e:  # pragma: no cover - safety net
            result = {
                "success": False,
                "metrics": {},
                "error": f"{type(e).__name__}: {e}",
            }
        dur = time.time() - t0

        post_report = self._check_pipeline(
            pipeline,
            self._get_data_context(),
            budget=self.budget,
            execution_data=result.get("execution_data"),
        )
        if not post_report.valid:
            result["success"] = False
            prior = result.get("error")
            post_error = "; ".join(post_report.errors)
            result["error"] = f"{prior}; {post_error}" if prior else post_error

        success = bool(result.get("success"))
        metrics = result.get("metrics") or {}
        if success:
            metric = self._metric_key_for_method(method)
            value = metrics.get(metric)
            if value is None or not math.isfinite(float(value)):
                success = False
                result["error"] = f"Missing or non-finite evaluation metric {metric!r}: {value!r}"
                fitness = -math.inf
            else:
                fitness = float(value)
                if metric in {"rmse", "mse", "mae", "logloss"}:
                    fitness = -fitness
        else:
            fitness = -math.inf

        label = self._display_method(method)
        ev = EvaluationResult(
            fitness=fitness,
            success=success,
            metrics={**metrics, "scoring_method": label},
            error=result.get("error"),
            duration_seconds=dur,
        )
        self._cache[key] = ev
        if self.verbose:
            if success:
                print(f"  [eval:{label}] fitness={fitness:.4f}  ({dur:.1f}s)  steps={len(pipeline)}")
            else:
                err_short = (result.get("error") or "")[:200].replace("\n", " ")
                print(f"  [eval:{label}] FAILED ({dur:.1f}s)  steps={len(pipeline)}  err={err_short}")
                print(result.get("error") or "")
        return ev

    def _display_method(self, method: str) -> str:
        if method == "full_train" and getattr(self._executor, "_fast_mode", False):
            return "fast_train"
        return method

    def _metric_key_for_method(self, method: str) -> str:
        if method == "static_quality":
            return "static_quality"
        if method == "error_detection":
            return "error_detection"
        return self.metric_key

    def _run_scoring_method(self, method: str) -> dict:
        if method == "full_train":
            return self._executor.run_training()
        if method == "fast_train":
            return self._run_fast_train()
        if method == "static_quality":
            return self._executor.run_static_quality()
        if method == "error_detection":
            return self._executor.run_error_detection_quality()
        raise ValueError(
            f"Unknown scoring_method '{method}'. Expected one of "
            "full_train, fast_train, static_quality, error_detection."
        )

    def _run_fast_train(self) -> dict:
        original_make = self._executor._make_data_instance
        original_fast = self._executor._fast_mode
        original_overrides = self._executor._fast_overrides

        def patched_make_data_instance():
            data = original_make()
            original_load_data = data.load_data
            original_load_std = getattr(data, "load_std_test_frozen", None)

            def patched_load_data(*args, **kwargs):
                ret = original_load_data(*args, **kwargs)
                _subsample_loaded_data(data, fraction=0.1, seed=42)
                return ret

            data.load_data = patched_load_data  # type: ignore[assignment]
            if callable(original_load_std):
                def patched_load_std_test(*args, **kwargs):
                    ret = original_load_std(*args, **kwargs)
                    _subsample_loaded_data(data, fraction=0.1, seed=42)
                    return ret

                data.load_std_test_frozen = patched_load_std_test  # type: ignore[assignment]
            return data

        fast_overrides = {
            "lightgbm": {"n_estimators": 20},
            "rec": {"epochs": 2, "early_stopping_patience": 1},
            "graph": {"epochs": 2},
            "seq": {"epochs": 2, "early_stopping_patience": 1},
            "tabular_deep": {"epochs": 2, "max_train_rows": 5000, "max_features": 64, "batch_size": 256},
        }
        try:
            self._executor._make_data_instance = patched_make_data_instance
            self._executor.set_fast_mode(True, fast_overrides)
            return self._executor.run_training()
        finally:
            self._executor._make_data_instance = original_make
            self._executor.set_fast_mode(original_fast, original_overrides)
            self._executor.clear_data_cache()

    @property
    def n_unique_evaluations(self) -> int:
        return len(self._cache)

    def set_evaluate_std_test(self, enabled: bool) -> None:
        """Toggle frozen std-test evaluation on the underlying executor."""
        self._executor.set_evaluate_std_test(enabled)
        self._cache.clear()

    def set_fast_mode(self, enabled: bool, overrides: dict = None) -> None:
        """Toggle search-time fast training on the underlying executor.

        Clears the result cache so identical pipelines are re-evaluated under
        the new mode rather than returning a stale (e.g. fast) score during a
        full evaluation pass.
        """
        self._executor.set_fast_mode(enabled, overrides)
        self._cache.clear()

def _subsample_loaded_data(data, fraction: float, seed: int) -> None:
    """Best-effort 10% sampling used by fast_train scoring."""
    fraction = max(0.0, min(1.0, float(fraction)))
    if fraction <= 0.0 or fraction >= 1.0:
        return

    def sample_df(df):
        if df is None or len(df) <= 1:
            return df
        if "__split__" in df.columns:
            parts = []
            for _, group in df.groupby("__split__", sort=False):
                n = max(1, int(len(group) * fraction))
                parts.append(group.sample(n=min(n, len(group)), random_state=seed))
            return type(df)(pd.concat(parts, ignore_index=True))
        n = max(1, int(len(df) * fraction))
        return df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)

    # Local import avoids making pandas a hard import at module import time.
    global pd
    import pandas as pd  # type: ignore

    for attr in ("train_df", "test_df", "interaction_df", "std_test_negatives_df"):
        df = getattr(data, attr, None)
        if df is not None:
            setattr(data, attr, sample_df(df))
