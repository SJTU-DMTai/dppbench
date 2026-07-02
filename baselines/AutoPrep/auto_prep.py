"""Top-level Auto-Prep orchestrator.

Implements the outer search loop:

  1. Use ``AutoPrepEvaluator(small_n=...)`` to load the dataset, then
     reuse SAGA's context-inference helpers to populate ``DataContext``.
  2. Initialise ``TransformationModel`` (M_T+) and ``JoinModel`` (M_J).
  3. Repeat for ``n_iters`` rounds:
       a. Call ``solve(...)`` to get ``n_candidates`` candidate pipelines.
       b. Evaluate each candidate via ``evaluate_for_agent`` (small-data AUC).
       c. Update ``best_pipeline`` and apply multiplicative-weights updates
          to both M_T+ and M_J using mean-centred AUC reward.
       d. Optional early stopping if best fitness has not improved for two
          consecutive rounds.
  4. Optionally re-evaluate the winning pipeline on the full dataset
     (``small_n=0``) for an honest final report.
  5. Persist ``best_pipeline.yaml`` and a structured ``auto_prep_log.json``.
"""
from __future__ import annotations

import json
import math
import os
import time
from typing import Optional

from baselines.common.config import (
    default_config_path,
    load_baseline_config,
    resolve_config_value,
)
from baselines.CtxPipe.evaluator import CtxPipeEvaluator
from baselines.common.pipeline import DataContext, Pipeline
from baselines.common.pipeline_constraints import is_legal
from baselines.common.context import _infer_rec_context, _infer_tabular_context

from .evaluator import AutoPrepEvaluator
from .operator_catalog import CATALOG
from .solver import Candidate, repair_pipeline, solve
from .transformation_model import JoinModel, TransformationModel


CONFIG_KEYS = (
    "n_iters",
    "beam",
    "n_candidates",
    "max_depth",
    "small_n",
    "eval_full",
    "eta",
    "early_stop_patience",
    "seed",
    "fast_train",
)


class AutoPrep:
    """Auto-Prep baseline driver."""

    def __init__(
        self,
        task_dir: str,
        data_name: str,
        data_dir: Optional[str] = None,
        n_iters: Optional[int] = None,
        beam: Optional[int] = None,
        n_candidates: Optional[int] = None,
        max_depth: Optional[int] = None,
        small_n: Optional[int] = None,
        eval_full: Optional[bool] = None,
        eta: Optional[float] = None,
        early_stop_patience: Optional[int] = None,
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
        n_iters = resolve_config_value(cfg, "n_iters", n_iters)
        beam = resolve_config_value(cfg, "beam", beam)
        n_candidates = resolve_config_value(cfg, "n_candidates", n_candidates)
        max_depth = resolve_config_value(cfg, "max_depth", max_depth)
        small_n = resolve_config_value(cfg, "small_n", small_n)
        eval_full = resolve_config_value(cfg, "eval_full", eval_full)
        eta = resolve_config_value(cfg, "eta", eta)
        early_stop_patience = resolve_config_value(
            cfg, "early_stop_patience", early_stop_patience
        )
        seed = resolve_config_value(cfg, "seed", seed)
        fast_train = resolve_config_value(cfg, "fast_train", fast_train)

        self.task_dir = task_dir
        self.data_name = data_name
        self.data_dir = data_dir
        self.n_iters = int(n_iters)
        self.beam = int(beam)
        self.n_candidates = int(n_candidates)
        self.max_depth = int(max_depth)
        self.small_n = small_n
        self.eval_full = bool(eval_full)
        self.eta = float(eta)
        self.early_stop_patience = int(early_stop_patience)
        self.seed = int(seed)
        self.verbose = bool(verbose)
        self.device = device
        self.fast_train = bool(fast_train)
        self.output_dir = output_dir or os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..", "outputs", "AutoPrep", data_name,
        ))
        os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    def _build_context(self, evaluator: CtxPipeEvaluator) -> DataContext:
        # Force the executor to load data so we can introspect the schema.
        data = evaluator._executor._load_data()
        summary = evaluator.get_data_summary()
        if evaluator.task_type == "rec":
            return _infer_rec_context(self.data_name, summary, data)
        return _infer_tabular_context(self.data_name, summary, data)

    # ------------------------------------------------------------------
    def run(self) -> dict:
        t0 = time.time()
        evaluator = AutoPrepEvaluator(
            task_dir=self.task_dir,
            data_name=self.data_name,
            data_dir=self.data_dir,
            small_n=self.small_n,
            seed=self.seed,
            verbose=self.verbose,
            device=self.device,
        )
        if self.fast_train:
            evaluator.set_fast_mode(True)
        ctx = self._build_context(evaluator)
        if self.verbose:
            print("=" * 60)
            print(f"[AutoPrep] dataset={self.data_name} task={ctx.task_type}")
            print(f"[AutoPrep] numeric={len(ctx.numeric_cols)} cat={len(ctx.categorical_cols)} "
                  f"list={len(ctx.list_cols)} text={len(ctx.text_cols)}")
            print(f"[AutoPrep] target={ctx.target_col} time={ctx.time_col} "
                  f"id={ctx.id_col} aux={ctx.aux_dfs}")
            print(f"[AutoPrep] catalog ops = {len(CATALOG)}")
            print("=" * 60)

        t_model = TransformationModel(ctx, eta=self.eta)
        j_model = JoinModel(ctx, eta=self.eta)

        best_candidate: Optional[Candidate] = None
        best_fitness = float("-inf")
        best_metrics: dict = {}
        no_improve_streak = 0
        history: list[dict] = []

        for it in range(self.n_iters):
            iter_seed = self.seed + 31 * it
            candidates = solve(
                ctx=ctx,
                t_model=t_model,
                j_model=j_model,
                beam=self.beam,
                n_candidates=self.n_candidates,
                max_depth=self.max_depth,
                seed=iter_seed,
            )
            if not candidates:
                if self.verbose:
                    print(f"[AutoPrep] iter {it+1}: no feasible candidates, stopping.")
                break

            # Evaluate each candidate with downstream feedback
            results: list[tuple[Candidate, float, dict, Optional[str]]] = []
            for ci, cand in enumerate(candidates):
                fit, metrics, err = evaluator.evaluate_for_agent(cand.pipeline)
                results.append((cand, fit, metrics, err))
                if self.verbose:
                    print(f"  [iter {it+1}] cand {ci+1}/{len(candidates)} "
                          f"ops={list(cand.op_set)} fit={fit:.4f} "
                          f"err={(err or '')[:80]}")

            # Best of this iter
            iter_best = max(results, key=lambda r: r[1])
            iter_best_fit = iter_best[1]
            if iter_best_fit > best_fitness + 1e-6:
                best_fitness = iter_best_fit
                best_candidate = iter_best[0]
                best_metrics = iter_best[2]
                no_improve_streak = 0
            else:
                no_improve_streak += 1

            # Multiplicative-weights update (mean-centred reward)
            valid = [r for r in results if math.isfinite(r[1])]
            if valid:
                mean_fit = sum(r[1] for r in valid) / len(valid)
                for cand, fit, _metrics, _err in valid:
                    reward = fit - mean_fit
                    for op_name in cand.op_set:
                        t_model.update(op_name, reward)
                    for jname in cand.join_set:
                        j_model.update(jname, reward)

            # Logging
            history.append({
                "iter": it + 1,
                "iter_best_fit": float(iter_best_fit) if math.isfinite(iter_best_fit) else None,
                "best_fit_so_far": float(best_fitness) if math.isfinite(best_fitness) else None,
                "candidates": [
                    {
                        "ops": list(c.op_set),
                        "joins": list(c.join_set),
                        "fitness": float(f) if math.isfinite(f) else None,
                        "metrics": m,
                        "error": e,
                    }
                    for c, f, m, e in results
                ],
                "op_probs": t_model.snapshot(),
                "join_probs": j_model.snapshot(),
            })

            if no_improve_streak >= self.early_stop_patience and it + 1 < self.n_iters:
                if self.verbose:
                    print(f"[AutoPrep] early stop at iter {it+1}: no improvement.")
                break

        # ----- Full-data re-evaluation -----
        full_metrics: dict = {}
        full_fitness: Optional[float] = None
        eval_error: Optional[str] = None
        if self.eval_full and best_candidate is not None:
            evaluator.set_small_n(0)
            if self.fast_train:
                evaluator.set_fast_mode(False)
            full_pipe = repair_pipeline(best_candidate.pipeline.copy(), ctx)
            full_fit, full_metrics, err = evaluator.evaluate_for_agent(full_pipe)
            eval_error = err
            full_fitness = float(full_fit) if math.isfinite(full_fit) else None
            if self.verbose:
                print(f"[AutoPrep] full-data eval: fit={full_fit:.4f} err={(err or '')[:80]}")
            best_metrics = full_metrics or best_metrics

        # ----- Persist outputs -----
        out_yaml = os.path.join(self.output_dir, "best_pipeline.yaml")
        out_log = os.path.join(self.output_dir, "auto_prep_log.json")
        output_pipe = (
            repair_pipeline(best_candidate.pipeline.copy(), ctx)
            if best_candidate is not None else repair_pipeline(Pipeline(), ctx)
        )
        with open(out_yaml, "w", encoding="utf-8") as f:
            f.write(output_pipe.to_yaml())

        log = {
            "data_name": self.data_name,
            "task_type": ctx.task_type,
            "n_catalog_ops": len(CATALOG),
            "best_small_fitness": float(best_fitness) if math.isfinite(best_fitness) else None,
            "best_full_fitness": full_fitness,
            "best_metrics": best_metrics,
            "best_pipeline_yaml": output_pipe.to_yaml(),
            "best_pipeline_path": out_yaml,
            "best_fitness": full_fitness if full_fitness is not None else (
                float(best_fitness) if math.isfinite(best_fitness) else None
            ),
            "eval_error": eval_error,
            "is_legal": is_legal(output_pipe, ctx.task_type),
            "final_pipeline_ops": output_pipe.op_names(),
            "best_ops": list(best_candidate.op_set) if best_candidate else [],
            "best_join_edges": list(best_candidate.join_set) if best_candidate else [],
            "history": history,
            "duration_seconds": time.time() - t0,
            "config": {
                "n_iters": self.n_iters, "beam": self.beam,
                "n_candidates": self.n_candidates, "max_depth": self.max_depth,
                "small_n": self.small_n, "eval_full": self.eval_full,
                "eta": self.eta, "seed": self.seed,
            },
        }
        with open(out_log, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2)

        if self.verbose:
            print("=" * 60)
            print(f"[AutoPrep] DONE in {time.time()-t0:.1f}s")
            print(f"[AutoPrep] best_small_fit = {log['best_small_fitness']}")
            print(f"[AutoPrep] best_full_fit  = {log['best_full_fitness']}")
            print(f"[AutoPrep] saved best pipeline -> {out_yaml}")
            print(f"[AutoPrep] saved log           -> {out_log}")
            print("=" * 60)

        return log


__all__ = ["AutoPrep"]
