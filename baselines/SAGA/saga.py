"""SAGA orchestrator: combines logical search and physical tuning.

Usage:
    saga = SAGA(task_dir, data_name, ...).run()
"""
from __future__ import annotations

import os
import time
from dataclasses import asdict
from typing import Optional

from baselines.common.config import (
    default_config_path,
    load_baseline_config,
    resolve_config_value,
)

from baselines.common.evaluator import PipelineEvaluator
from .logical_search import LogicalSearch
from .physical_search import PhysicalSearch, TunedPipeline
from baselines.common.pipeline import DataContext, Pipeline
from baselines.common.pipeline_constraints import is_legal

# CtxPipeEvaluator extends PipelineEvaluator with row-level subsampling so the
# search loop can iterate cheaply; SAGA reuses it when small_n is set.
# Imported lazily inside ``run`` to avoid a circular import:
# ``baselines.CtxPipe.__init__`` imports ``CtxPipe`` which in turn imports
# helpers from this module, so a top-level import here would re-enter
# ``baselines.SAGA.saga`` while it is still partially initialized.


from baselines.common.context import (
    _infer_graph_context,
    _infer_rec_context,
    _infer_tabular_context,
)


# -----------------------------------------------------------------------------
# SAGA main class
# -----------------------------------------------------------------------------
CONFIG_KEYS = (
    "population_size",
    "n_generations",
    "top_k",
    "n_physical_trials",
    "early_stop_patience",
    "mutation_rate",
    "crossover_rate",
    "tournament_size",
    "elitism_size",
    "random_pipeline_p_optional",
    "small_n",
    "seed",
    "fast_train",
)


class SAGA:
    def __init__(
        self,
        task_dir: str,
        data_name: str,
        data_dir: Optional[str] = None,
        population_size: Optional[int] = None,
        n_generations: Optional[int] = None,
        top_k: Optional[int] = None,
        n_physical_trials: Optional[int] = None,
        early_stop_patience: Optional[int] = None,
        mutation_rate: Optional[float] = None,
        crossover_rate: Optional[float] = None,
        tournament_size: Optional[int] = None,
        elitism_size: Optional[int] = None,
        random_pipeline_p_optional: Optional[float] = None,
        seed: Optional[int] = None,
        output_dir: Optional[str] = None,
        verbose: bool = True,
        device: str = "cpu",
        small_n: Optional[int] = None,
        fast_train: Optional[bool] = None,
        model_name: Optional[str] = None,
        config_path: Optional[str] = None,
        skip_final_eval: bool = False,
    ):
        cfg = load_baseline_config(
            config_path or default_config_path(__file__), CONFIG_KEYS
        )
        population_size = resolve_config_value(
            cfg, "population_size", population_size
        )
        n_generations = resolve_config_value(cfg, "n_generations", n_generations)
        top_k = resolve_config_value(cfg, "top_k", top_k)
        n_physical_trials = resolve_config_value(
            cfg, "n_physical_trials", n_physical_trials
        )
        early_stop_patience = resolve_config_value(
            cfg, "early_stop_patience", early_stop_patience
        )
        mutation_rate = resolve_config_value(cfg, "mutation_rate", mutation_rate)
        crossover_rate = resolve_config_value(cfg, "crossover_rate", crossover_rate)
        tournament_size = resolve_config_value(cfg, "tournament_size", tournament_size)
        elitism_size = resolve_config_value(cfg, "elitism_size", elitism_size)
        random_pipeline_p_optional = resolve_config_value(
            cfg, "random_pipeline_p_optional", random_pipeline_p_optional
        )
        small_n = resolve_config_value(cfg, "small_n", small_n)
        seed = resolve_config_value(cfg, "seed", seed)
        fast_train = resolve_config_value(cfg, "fast_train", fast_train)

        self.task_dir = task_dir
        self.data_name = data_name
        self.data_dir = data_dir
        self.population_size = int(population_size)
        self.n_generations = int(n_generations)
        self.top_k = int(top_k)
        self.n_physical_trials = int(n_physical_trials)
        self.early_stop_patience = int(early_stop_patience)
        self.mutation_rate = float(mutation_rate)
        self.crossover_rate = float(crossover_rate)
        self.tournament_size = int(tournament_size)
        self.elitism_size = int(elitism_size)
        self.random_pipeline_p_optional = float(random_pipeline_p_optional)
        self.seed = int(seed)
        self.verbose = bool(verbose)
        self.device = device
        self.small_n = int(small_n) if small_n else 0
        self.fast_train = bool(fast_train)
        self.model_name = model_name
        self.skip_final_eval = bool(skip_final_eval)
        self.output_dir = output_dir or os.path.join(
            os.path.dirname(__file__), "..", "..", "outputs", "SAGA", data_name
        )
        self.output_dir = os.path.abspath(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    def _build_context(self, evaluator: PipelineEvaluator) -> DataContext:
        # Force the executor to load data so we can inspect the schema.
        data = evaluator._executor._load_data()
        summary = evaluator.get_data_summary()
        if evaluator.task_type == "rec":
            return _infer_rec_context(self.data_name, summary, data)
        if evaluator.task_type == "graph":
            return _infer_graph_context(self.data_name, summary, data)
        return _infer_tabular_context(self.data_name, summary, data)

    # ------------------------------------------------------------------
    def run(self) -> dict:
        # Lazy import to break a circular dependency with baselines.CtxPipe.
        from baselines.CtxPipe.evaluator import CtxPipeEvaluator

        t0 = time.time()
        evaluator = CtxPipeEvaluator(
            task_dir=self.task_dir,
            data_name=self.data_name,
            data_dir=self.data_dir,
            verbose=self.verbose,
            device=self.device,
            small_n=self.small_n,
            seed=self.seed,
            model_name=self.model_name,
        )
        if self.fast_train:
            evaluator.set_fast_mode(True)
        ctx = self._build_context(evaluator)
        evaluator.ctx = ctx
        if self.verbose:
            print("=" * 60)
            print(f"[SAGA] dataset={self.data_name}  task={ctx.task_type}")
            print(f"[SAGA] numeric_cols={len(ctx.numeric_cols)}  "
                  f"categorical_cols={len(ctx.categorical_cols)}  "
                  f"list_cols={len(ctx.list_cols)}")
            print(f"[SAGA] target={ctx.target_col}  time={ctx.time_col}  "
                  f"id={ctx.id_col}  aux={ctx.aux_dfs}")
            print("=" * 60)

        if ctx.task_type == "graph":
            duration = time.time() - t0
            error = "unsupported_task_type=graph: SAGA graph search is not implemented"
            if self.verbose:
                print(f"[SAGA] {error}")
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
                "logical_history": [],
                "duration_seconds": duration,
                "output_dir": self.output_dir,
                "unsupported_task_type": "graph",
                "error": error,
            }

        # ---- Logical search ----
        logical = LogicalSearch(
            evaluator=evaluator,
            ctx=ctx,
            population_size=self.population_size,
            n_generations=self.n_generations,
            mutation_rate=self.mutation_rate,
            crossover_rate=self.crossover_rate,
            tournament_size=self.tournament_size,
            elitism_size=self.elitism_size,
            early_stop_patience=self.early_stop_patience,
            random_pipeline_p_optional=self.random_pipeline_p_optional,
            seed=self.seed,
            verbose=self.verbose,
        )
        top_logical = logical.run(top_k=self.top_k)
        if self.verbose:
            print(f"[SAGA] logical phase done. top-{len(top_logical)}:")
            for i, ind in enumerate(top_logical):
                print(f"  #{i+1}  fit={ind.fitness:.4f}  steps={[s.op for s in ind.pipeline.steps]}")

        # ---- Physical tuning ----
        physical = PhysicalSearch(
            evaluator=evaluator,
            n_trials=self.n_physical_trials,
            seed=self.seed,
            verbose=self.verbose,
        )
        candidates = [(ind.pipeline, ind.fitness, ind.metrics) for ind in top_logical]
        tuned: list[TunedPipeline] = physical.tune(candidates)

        # ---- Save best ----
        best = tuned[0]
        out_yaml = os.path.join(self.output_dir, "best_pipeline.yaml")
        with open(out_yaml, "w", encoding="utf-8") as f:
            f.write(best.pipeline.to_yaml())

        # also persist per-rank
        for i, t in enumerate(tuned):
            path = os.path.join(self.output_dir, f"top{i+1}_pipeline.yaml")
            with open(path, "w", encoding="utf-8") as f:
                f.write(t.pipeline.to_yaml())

        # ---- Final full-data evaluation ----
        # When fast/small-n was used during search, the recorded fitness reflects
        # the cheap proxy run. Re-evaluate the best pipeline once with full data
        # and full training rounds so the reported metrics match the rest of the
        # benchmark.
        best_fitness = float(best.fitness)
        best_metrics = dict(best.metrics or {})
        eval_error: Optional[str] = None
        if not self.skip_final_eval and (self.fast_train or self.small_n > 0):
            try:
                full_eval = CtxPipeEvaluator(
                    task_dir=self.task_dir,
                    data_name=self.data_name,
                    data_dir=self.data_dir,
                    verbose=self.verbose,
                    device=self.device,
                    small_n=0,
                    seed=self.seed,
                    model_name=self.model_name,
                )
                full_eval.ctx = ctx
                ev = full_eval.evaluate(best.pipeline)
                if ev.success:
                    best_fitness = float(ev.fitness)
                    best_metrics = dict(ev.metrics or {})
                else:
                    eval_error = ev.error
                    if self.verbose:
                        print(f"[SAGA] full eval failed: {ev.error}")
            except Exception as e:
                eval_error = f"{type(e).__name__}: {e}"
                if self.verbose:
                    print(f"[SAGA] full eval raised: {e}")

        duration = time.time() - t0
        if self.verbose:
            print("=" * 60)
            print(f"[SAGA] DONE in {duration:.1f}s")
            print(f"[SAGA] best fitness = {best_fitness:.4f}")
            print(f"[SAGA] best metrics = {best_metrics}")
            print(f"[SAGA] best pipeline saved to: {out_yaml}")
            print(f"[SAGA] unique evaluations: {evaluator.n_unique_evaluations}")
            print("=" * 60)

        return {
            "best_pipeline_yaml": best.pipeline.to_yaml(),
            "best_pipeline_path": out_yaml,
            "best_fitness": best_fitness,
            "best_metrics": best_metrics,
            "eval_error": eval_error,
            "is_legal": is_legal(best.pipeline, ctx.task_type),
            "final_pipeline_ops": best.pipeline.op_names(),
            "top_k": [
                {
                    "fitness": t.fitness,
                    "metrics": t.metrics,
                    "ops": [s.op for s in t.pipeline.steps],
                    "yaml": t.pipeline.to_yaml(),
                }
                for t in tuned
            ],
            "n_unique_evaluations": evaluator.n_unique_evaluations,
            "logical_history": logical.history,
            "duration_seconds": duration,
            "output_dir": self.output_dir,
        }
