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

from .evaluator import PipelineEvaluator
from .logical_search import LogicalSearch
from .physical_search import PhysicalSearch, TunedPipeline
from .pipeline import DataContext, Pipeline
from .pipeline_constraints import is_legal

# CtxPipeEvaluator extends PipelineEvaluator with row-level subsampling so the
# search loop can iterate cheaply; SAGA reuses it when small_n is set.
# Imported lazily inside ``run`` to avoid a circular import:
# ``baselines.CtxPipe.__init__`` imports ``CtxPipe`` which in turn imports
# helpers from this module, so a top-level import here would re-enter
# ``baselines.SAGA.saga`` while it is still partially initialized.


# -----------------------------------------------------------------------------
# Context inference helpers
# -----------------------------------------------------------------------------
def _infer_rec_context(data_name: str, summary: dict, data) -> DataContext:
    interaction_df = data.interaction_df
    col_types = data.col_types or {}

    numeric_cols = [c for c in interaction_df.columns if col_types.get(c) == "numeric"]
    categorical_cols = [c for c in interaction_df.columns if col_types.get(c) == "categorical"]
    list_cols = [c for c in interaction_df.columns
                 if col_types.get(c) in ("numeric_list", "categorical_list")]
    text_cols = [c for c in interaction_df.columns if col_types.get(c) == "text"]

    # Add columns from user_df / item_df schemas as well so JoinTable downstream
    # operators can reference them by name.
    for side_df in (data.user_df, data.item_df):
        if side_df is None:
            continue
        for c in side_df.columns:
            if c in (data._user_id_col, data._item_id_col):
                continue
            t = col_types.get(c)
            if t == "numeric":
                numeric_cols.append(c)
            elif t == "categorical":
                categorical_cols.append(c)
            elif t in ("numeric_list", "categorical_list"):
                list_cols.append(c)
            elif t == "text":
                text_cols.append(c)

    # target column
    target_col = None
    for cand in ("rating", "stars", "label", "click", "is_click"):
        if cand in interaction_df.columns:
            target_col = cand
            break
    # time column
    time_col = None
    for cand in ("timestamp", "time", "ts"):
        if cand in interaction_df.columns:
            time_col = cand
            break

    return DataContext(
        task_type="rec",
        data_name=data_name,
        numeric_cols=sorted(set(numeric_cols)),
        categorical_cols=sorted(set(categorical_cols)),
        list_cols=sorted(set(list_cols)),
        text_cols=sorted(set(text_cols)),
        target_col=target_col,
        id_col=None,
        time_col=time_col,
        user_col=data._user_id_col,
        item_col=data._item_id_col,
        has_user_df=data.user_df is not None,
        has_item_df=data.item_df is not None,
        aux_dfs=[],
    )


def _infer_tabular_context(data_name: str, summary: dict, data) -> DataContext:
    train_df = data.train_df
    target_col = data.target_col
    id_col = data.id_col

    num_cols, cat_cols = [], []
    for c in train_df.columns:
        if c in (target_col, id_col):
            continue
        if train_df[c].dtype.kind in ("i", "u", "f"):
            num_cols.append(c)
        else:
            cat_cols.append(c)

    # Detect a temporal column heuristically (used by ExtractDateTimeFeature)
    time_col = None
    for cand in ("TransactionDT", "timestamp", "time"):
        if cand in train_df.columns:
            time_col = cand
            break

    aux_names = list((data.auxiliary_dfs or {}).keys())
    aux_names = [a for a in aux_names if data.auxiliary_dfs.get(a) is not None]

    # Sentinel rules used by CustomClean (heuristic per dataset)
    sentinel_rules: list[dict] = []
    if data_name == "home_credit":
        if "DAYS_EMPLOYED" in train_df.columns:
            sentinel_rules.append({"col": "DAYS_EMPLOYED", "value": 365243})
        if "CODE_GENDER" in train_df.columns:
            sentinel_rules.append({"col": "CODE_GENDER", "value": "XNA"})
        if "ORGANIZATION_TYPE" in train_df.columns:
            sentinel_rules.append({"col": "ORGANIZATION_TYPE", "value": "XNA"})

    return DataContext(
        task_type="tabular",
        data_name=data_name,
        numeric_cols=num_cols,
        categorical_cols=cat_cols,
        list_cols=[],
        text_cols=[],
        target_col=target_col,
        id_col=id_col,
        time_col=time_col,
        aux_dfs=aux_names,
        sentinel_rules=sentinel_rules,
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
        )
        if self.fast_train:
            evaluator.set_fast_mode(True)
        ctx = self._build_context(evaluator)
        if self.verbose:
            print("=" * 60)
            print(f"[SAGA] dataset={self.data_name}  task={ctx.task_type}")
            print(f"[SAGA] numeric_cols={len(ctx.numeric_cols)}  "
                  f"categorical_cols={len(ctx.categorical_cols)}  "
                  f"list_cols={len(ctx.list_cols)}")
            print(f"[SAGA] target={ctx.target_col}  time={ctx.time_col}  "
                  f"id={ctx.id_col}  aux={ctx.aux_dfs}")
            print("=" * 60)

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
                )
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
