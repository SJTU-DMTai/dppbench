"""Random pipeline-search baseline.

This is the executable counterpart of the random baseline example in
``papers/DPPBench实现.md``. It samples candidate DAG pipelines from the shared
operator space, validates them through ``baselines.common.pipeline``, and scores
them with the same downstream evaluator used by the other baselines.
"""
from __future__ import annotations

import math
import os
import random
import time
from typing import Optional

from baselines.CtxPipe.evaluator import CtxPipeEvaluator
from baselines.common.config import (
    default_config_path,
    load_baseline_config,
    resolve_config_value,
)
from baselines.common.data_ctx import (
    _infer_graph_context,
    _infer_rec_context,
    _infer_tabular_context,
)
from baselines.common.pipeline import check_pipeline, random_pipeline


CONFIG_KEYS = (
    "n_trials",
    "random_pipeline_p_optional",
    "small_n",
    "eval_full",
    "seed",
    "fast_train",
)


class RandomBaseline:
    def __init__(
        self,
        task_dir: str,
        data_name: str,
        data_dir: Optional[str] = None,
        n_trials: Optional[int] = None,
        random_pipeline_p_optional: Optional[float] = None,
        small_n: Optional[int] = None,
        eval_full: Optional[bool] = None,
        seed: Optional[int] = None,
        output_dir: Optional[str] = None,
        verbose: bool = True,
        device: str = "cpu",
        fast_train: Optional[bool] = None,
        model_name: Optional[str] = None,
        config_path: Optional[str] = None,
    ) -> None:
        cfg = load_baseline_config(
            config_path or default_config_path(__file__), CONFIG_KEYS
        )
        n_trials = resolve_config_value(cfg, "n_trials", n_trials)
        random_pipeline_p_optional = resolve_config_value(
            cfg, "random_pipeline_p_optional", random_pipeline_p_optional
        )
        small_n = resolve_config_value(cfg, "small_n", small_n)
        eval_full = resolve_config_value(cfg, "eval_full", eval_full)
        seed = resolve_config_value(cfg, "seed", seed)
        fast_train = resolve_config_value(cfg, "fast_train", fast_train)

        self.task_dir = task_dir
        self.data_name = data_name
        self.data_dir = data_dir
        self.n_trials = int(n_trials)
        self.random_pipeline_p_optional = float(random_pipeline_p_optional)
        self.small_n = int(small_n) if small_n else 0
        self.eval_full = bool(eval_full)
        self.seed = int(seed)
        self.output_dir = os.path.abspath(
            output_dir
            or os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "Random", data_name)
        )
        self.verbose = bool(verbose)
        self.device = device
        self.fast_train = bool(fast_train)
        self.model_name = model_name
        os.makedirs(self.output_dir, exist_ok=True)

    def _build_context(self, evaluator: CtxPipeEvaluator):
        data = evaluator.load_data()
        if evaluator.task_type == "rec":
            return _infer_rec_context(self.data_name, data)
        if evaluator.task_type == "graph":
            return _infer_graph_context(self.data_name, data)
        return _infer_tabular_context(self.data_name, data)

    def run(self) -> dict:
        rng = random.Random(self.seed)
        t0 = time.time()

        evaluator = CtxPipeEvaluator(
            task_dir=self.task_dir,
            data_name=self.data_name,
            data_dir=self.data_dir,
            verbose=self.verbose,
            small_n=self.small_n,
            seed=self.seed,
            device=self.device,
            model_name=self.model_name,
        )
        if self.fast_train:
            evaluator.set_fast_mode(True)
        ctx = self._build_context(evaluator)
        evaluator.ctx = ctx

        if ctx.task_type == "graph":
            error = "unsupported_task_type=graph: Random baseline graph search is not implemented"
            return {
                "best_pipeline_yaml": None,
                "best_pipeline_path": None,
                "best_fitness": None,
                "best_metrics": {},
                "eval_error": error,
                "is_legal": False,
                "final_pipeline_ops": [],
                "top_k": [],
                "n_unique_evaluations": evaluator.n_unique_evaluations,
                "duration_seconds": time.time() - t0,
                "output_dir": self.output_dir,
                "unsupported_task_type": "graph",
                "error": error,
            }

        if self.verbose:
            print("=" * 60)
            print(f"[Random] dataset={self.data_name}  task={ctx.task_type}")
            print(f"[Random] trials={self.n_trials}  p_optional={self.random_pipeline_p_optional}")
            print("=" * 60)

        candidates = []
        best = None
        for trial in range(1, self.n_trials + 1):
            pipe = random_pipeline(
                ctx.task_type,
                ctx,
                rng,
                p_optional=self.random_pipeline_p_optional,
            )
            report = check_pipeline(pipe, ctx)
            if not report.valid:
                record = {
                    "trial": trial,
                    "fitness": -math.inf,
                    "metrics": {"legal_errors": report.errors},
                    "pipeline": pipe,
                }
            else:
                ev = evaluator.evaluate(pipe)
                record = {
                    "trial": trial,
                    "fitness": float(ev.fitness),
                    "metrics": dict(ev.metrics or {}),
                    "error": ev.error,
                    "pipeline": pipe,
                }
            candidates.append(record)
            if best is None or record["fitness"] > best["fitness"]:
                best = record
            if self.verbose:
                fit = record["fitness"]
                fit_str = f"{fit:.4f}" if fit > -math.inf else "-inf"
                print(f"[Random] trial={trial}/{self.n_trials} fit={fit_str} ops={pipe.op_names()}")

        if best is None:
            raise RuntimeError("Random baseline did not generate any candidate pipeline.")

        best_pipe = best["pipeline"]
        best_fitness = float(best["fitness"]) if best["fitness"] > -math.inf else None
        best_metrics = dict(best.get("metrics") or {})
        eval_error = best.get("error")

        out_yaml = os.path.join(self.output_dir, "best_pipeline.yaml")
        with open(out_yaml, "w", encoding="utf-8") as f:
            f.write(best_pipe.to_yaml())

        if self.eval_full:
            try:
                full_eval = CtxPipeEvaluator(
                    task_dir=self.task_dir,
                    data_name=self.data_name,
                    data_dir=self.data_dir,
                    verbose=self.verbose,
                    small_n=0,
                    seed=self.seed,
                    device=self.device,
                    model_name=self.model_name,
                )
                ev = full_eval.evaluate(best_pipe)
                if ev.success:
                    best_fitness = float(ev.fitness)
                    best_metrics = dict(ev.metrics or {})
                else:
                    eval_error = ev.error
            except Exception as exc:
                eval_error = f"{type(exc).__name__}: {exc}"

        candidates.sort(key=lambda item: item["fitness"], reverse=True)
        duration = time.time() - t0
        if self.verbose:
            fit_str = f"{best_fitness:.4f}" if isinstance(best_fitness, float) else "n/a"
            print("=" * 60)
            print(f"[Random] DONE in {duration:.1f}s")
            print(f"[Random] best fitness = {fit_str}")
            print(f"[Random] best yaml    -> {out_yaml}")
            print("=" * 60)

        return {
            "best_pipeline_yaml": best_pipe.to_yaml(),
            "best_pipeline_path": out_yaml,
            "best_fitness": best_fitness,
            "best_metrics": best_metrics,
            "eval_error": eval_error,
            "is_legal": check_pipeline(best_pipe, ctx).valid,
            "final_pipeline_ops": best_pipe.op_names(),
            "top_k": [
                {
                    "trial": item["trial"],
                    "fitness": item["fitness"],
                    "metrics": item.get("metrics") or {},
                    "ops": item["pipeline"].op_names(),
                    "yaml": item["pipeline"].to_yaml(),
                }
                for item in candidates[:5]
            ],
            "n_unique_evaluations": evaluator.n_unique_evaluations,
            "duration_seconds": duration,
            "output_dir": self.output_dir,
        }


Random = RandomBaseline

__all__ = ["RandomBaseline", "Random"]
