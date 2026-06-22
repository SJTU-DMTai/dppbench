"""Top-level AlphaClean orchestrator.

Public entry: :class:`AlphaClean`.run() returns a dict whose schema is aligned
with :class:`baselines.CtxPipe.ctxpipe.CtxPipe`.run() so the AlphaClean
baseline plugs into existing benchmark harnesses without changes.
"""
from __future__ import annotations

import json
import os
import random as _random
import time
import warnings
from typing import Optional, Tuple

import numpy as np

from baselines.common.config import (
    default_config_path,
    load_baseline_config,
    resolve_config_value,
)
from baselines.CtxPipe.evaluator import CtxPipeEvaluator
from baselines.SAGA.pipeline import DataContext, Pipeline
from baselines.SAGA.pipeline_constraints import is_legal
from baselines.SAGA.saga import _infer_rec_context, _infer_tabular_context

from .operator_catalog import CATALOG
from .parameter_sampler import ParameterSampler
from .pruner import LearnedPruner
from .searcher import AlphaCleanSearcher


CONFIG_KEYS = (
    "beam_width",
    "n_iters",
    "batch_per_iter",
    "gamma",
    "patience",
    "small_n",
    "eval_full",
    "learned_pruning",
    "pruner_min_samples",
    "pruner_refit_every",
    "seed",
    "fast_train",
)


class AlphaClean:
    def __init__(
        self,
        task_dir: str,
        data_name: str,
        data_dir: Optional[str] = None,
        beam_width: Optional[int] = None,
        n_iters: Optional[int] = None,
        batch_per_iter: Optional[int] = None,
        gamma: Optional[int] = None,
        patience: Optional[int] = None,
        small_n: Optional[int] = None,
        eval_full: Optional[bool] = None,
        learned_pruning: Optional[bool] = None,
        pruner_min_samples: Optional[int] = None,
        pruner_refit_every: Optional[int] = None,
        seed: Optional[int] = None,
        output_dir: Optional[str] = None,
        verbose: bool = True,
        device: str = "cpu",
        fast_train: Optional[bool] = None,
        config_path: Optional[str] = None,
    ) -> None:
        cfg = load_baseline_config(
            config_path or default_config_path(__file__), CONFIG_KEYS
        )
        beam_width = resolve_config_value(cfg, "beam_width", beam_width)
        n_iters = resolve_config_value(cfg, "n_iters", n_iters)
        batch_per_iter = resolve_config_value(cfg, "batch_per_iter", batch_per_iter)
        gamma = resolve_config_value(cfg, "gamma", gamma)
        patience = resolve_config_value(cfg, "patience", patience)
        small_n = resolve_config_value(cfg, "small_n", small_n)
        eval_full = resolve_config_value(cfg, "eval_full", eval_full)
        learned_pruning = resolve_config_value(cfg, "learned_pruning", learned_pruning)
        pruner_min_samples = resolve_config_value(
            cfg, "pruner_min_samples", pruner_min_samples
        )
        pruner_refit_every = resolve_config_value(
            cfg, "pruner_refit_every", pruner_refit_every
        )
        seed = resolve_config_value(cfg, "seed", seed)
        fast_train = resolve_config_value(cfg, "fast_train", fast_train)

        self.task_dir = task_dir
        self.data_name = data_name
        self.data_dir = data_dir
        self.beam_width = int(beam_width)
        self.n_iters = int(n_iters)
        self.batch_per_iter = int(batch_per_iter)
        self.gamma = int(gamma)
        self.patience = int(patience)
        self.small_n = int(small_n) if small_n else 0
        self.eval_full = bool(eval_full)
        self.learned_pruning = bool(learned_pruning)
        self.pruner_min_samples = int(pruner_min_samples)
        self.pruner_refit_every = int(pruner_refit_every)
        self.seed = int(seed)
        self.verbose = bool(verbose)
        self.device = device
        self.fast_train = bool(fast_train)

        self.output_dir = output_dir or os.path.join(
            os.path.dirname(__file__), "..", "..", "outputs", "AlphaClean", data_name
        )
        self.output_dir = os.path.abspath(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    def _build_context(self, evaluator: CtxPipeEvaluator) -> Tuple[DataContext, object]:
        data = evaluator._executor._load_data()
        if evaluator.task_type == "rec":
            ctx = _infer_rec_context(self.data_name, {}, data)
        else:
            ctx = _infer_tabular_context(self.data_name, {}, data)
        return ctx, data

    # ------------------------------------------------------------------
    def run(self) -> dict:
        np.random.seed(self.seed)
        _random.seed(self.seed)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        t0 = time.time()

        # 1) Search-time evaluator (subsampled).
        train_evaluator = CtxPipeEvaluator(
            task_dir=self.task_dir,
            data_name=self.data_name,
            data_dir=self.data_dir,
            verbose=self.verbose,
            small_n=self.small_n,
            seed=self.seed,
            device=self.device,
        )
        if self.fast_train:
            train_evaluator.set_fast_mode(True)
        ctx, _data = self._build_context(train_evaluator)

        if self.verbose:
            print("=" * 60)
            print(f"[AlphaClean] dataset={self.data_name}  task={ctx.task_type}")
            print(f"[AlphaClean] numeric={len(ctx.numeric_cols)}  "
                  f"categorical={len(ctx.categorical_cols)}  "
                  f"target={ctx.target_col}")
            print(f"[AlphaClean] beam_width={self.beam_width}  "
                  f"n_iters={self.n_iters}  batch={self.batch_per_iter}  "
                  f"gamma={self.gamma}  small_n={self.small_n or 'OFF'}  "
                  f"learned_pruning={self.learned_pruning}")
            print("=" * 60)

        # 2) Sampler / Pruner / Searcher
        sampler = ParameterSampler(ctx, seed=self.seed)
        op_index = {n: i for i, n in enumerate(sorted(CATALOG.keys()))}
        pruner = LearnedPruner(
            op_index=op_index,
            ctx=ctx,
            min_samples=self.pruner_min_samples,
            refit_every=self.pruner_refit_every,
            enabled=self.learned_pruning,
        )
        searcher = AlphaCleanSearcher(
            evaluator=train_evaluator,
            ctx=ctx,
            sampler=sampler,
            pruner=pruner,
            beam_width=self.beam_width,
            n_iters=self.n_iters,
            batch_per_iter=self.batch_per_iter,
            gamma=self.gamma,
            patience=self.patience,
            verbose=self.verbose,
        )

        # 3) Run search
        frontier = searcher.run()
        if not frontier:
            raise RuntimeError("AlphaClean search produced no scored pipelines.")

        best = frontier[0]

        # 4) Persist artefacts
        out_yaml_path = os.path.join(self.output_dir, "best_pipeline.yaml")
        with open(out_yaml_path, "w", encoding="utf-8") as f:
            f.write(best.pipeline.to_yaml())

        history_path = os.path.join(self.output_dir, "search_history.json")
        try:
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(searcher.history, f, indent=2)
        except Exception as e:
            if self.verbose:
                print(f"[AlphaClean] warning: failed to save history: {e}")

        pruner_path = os.path.join(self.output_dir, "pruner.pkl")
        try:
            pruner.save(pruner_path)
        except Exception as e:
            if self.verbose:
                print(f"[AlphaClean] warning: failed to save pruner: {e}")
            pruner_path = None

        # also persist top-K
        for i, sp in enumerate(frontier[:5]):
            path = os.path.join(self.output_dir, f"top{i+1}_pipeline.yaml")
            with open(path, "w", encoding="utf-8") as f:
                f.write(sp.pipeline.to_yaml())

        # 5) Final full-data evaluation
        best_fitness: Optional[float] = float(best.fitness) if best.fitness > -1e30 else None
        best_metrics: dict = dict(best.metrics or {})
        eval_error: Optional[str] = None
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
                )
                ev = full_eval.evaluate(best.pipeline)
                if ev.success:
                    best_fitness = float(ev.fitness)
                    best_metrics = dict(ev.metrics or {})
                else:
                    eval_error = ev.error
                    if self.verbose:
                        print(f"[AlphaClean] full eval failed: {ev.error}")
            except Exception as e:
                eval_error = f"{type(e).__name__}: {e}"
                if self.verbose:
                    print(f"[AlphaClean] full eval raised: {e}")

        duration = time.time() - t0

        if self.verbose:
            print("=" * 60)
            print(f"[AlphaClean] DONE in {duration:.1f}s")
            fit_str = f"{best_fitness:.4f}" if isinstance(best_fitness, float) else "n/a"
            print(f"[AlphaClean] best fitness = {fit_str}")
            print(f"[AlphaClean] best metrics = {best_metrics}")
            print(f"[AlphaClean] best ops     = {best.pipeline.op_names()}")
            print(f"[AlphaClean] best yaml    -> {out_yaml_path}")
            print(f"[AlphaClean] history      -> {history_path}")
            print(f"[AlphaClean] pruner       -> {pruner_path}")
            print(f"[AlphaClean] unique evals = {train_evaluator.n_unique_evaluations}")
            print("=" * 60)

        return {
            "best_pipeline_yaml": best.pipeline.to_yaml(),
            "best_pipeline_path": out_yaml_path,
            "best_fitness": best_fitness,
            "best_metrics": best_metrics,
            "eval_error": eval_error,
            "is_legal": is_legal(best.pipeline, ctx.task_type),
            "final_pipeline_ops": best.pipeline.op_names(),
            "search_history": list(searcher.history),
            "n_unique_evaluations": train_evaluator.n_unique_evaluations,
            "duration_seconds": duration,
            "output_dir": self.output_dir,
            "pruner_path": pruner_path,
            "top_k": [
                {
                    "fitness": sp.fitness,
                    "metrics": sp.metrics,
                    "ops": sp.pipeline.op_names(),
                    "yaml": sp.pipeline.to_yaml(),
                }
                for sp in frontier[:5]
            ],
        }


__all__ = ["AlphaClean"]
