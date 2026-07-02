"""AlphaCleanSearcher -- best-first beam search over pipeline compositions.

Each iteration:
  1. **Generate** a batch of candidate repairs via :class:`ParameterSampler`,
     optionally filtered by :class:`LearnedPruner`.
  2. **Expand** the current beam: for each frontier pipeline ``p`` and each
     candidate repair ``c``, form ``p' = p ∘ c`` (append + repair-rebalance).
  3. **Evaluate** every new pipeline through the supplied evaluator.
  4. **Frontier update** -- keep top-``gamma`` distinct (by hash) pipelines.
  5. **Learn** -- emit (repair, label) examples to the pruner using "label = 1
     iff op appears in best pipeline's op set" and refit periodically.
  6. **Early stop** if ``patience`` consecutive iters had no improvement.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional

from baselines.common.pipeline import DataContext, Pipeline
from baselines.common.evaluator import EvaluationResult, PipelineEvaluator

from .parameter_sampler import ParameterSampler
from .pruner import LearnedPruner
from .repair import Repair, repairs_to_pipeline


@dataclass
class ScoredPipeline:
    pipeline: Pipeline
    fitness: float = -math.inf
    metrics: dict = field(default_factory=dict)

    def hash(self) -> str:
        return self.pipeline.hash()


class AlphaCleanSearcher:
    def __init__(
        self,
        evaluator: PipelineEvaluator,
        ctx: DataContext,
        sampler: ParameterSampler,
        pruner: LearnedPruner,
        beam_width: int = 4,
        n_iters: int = 8,
        batch_per_iter: int = 6,
        gamma: int = 8,
        patience: int = 3,
        verbose: bool = True,
    ) -> None:
        self.evaluator = evaluator
        self.ctx = ctx
        self.sampler = sampler
        self.pruner = pruner
        self.beam_width = int(beam_width)
        self.n_iters = int(n_iters)
        self.batch_per_iter = int(batch_per_iter)
        # ``beam_width`` is the actual number of states kept in the beam.
        # ``gamma`` remains as a compatibility/candidate-budget knob.
        self.gamma = max(int(gamma), self.beam_width)
        self.patience = int(patience)
        self.verbose = bool(verbose)

        self.frontier: List[ScoredPipeline] = []
        self.history: List[dict] = []

    # ------------------------------------------------------------------
    def _evaluate(self, pipe: Pipeline) -> ScoredPipeline:
        ev: EvaluationResult = self.evaluator.evaluate(pipe)
        return ScoredPipeline(
            pipeline=pipe,
            fitness=float(ev.fitness) if ev.fitness is not None else -math.inf,
            metrics=dict(ev.metrics or {}),
        )

    # ------------------------------------------------------------------
    def _seed_frontier(self) -> None:
        # Start with the empty pipeline (which repair() will populate with
        # mandatory ops) and one fully-random pipeline of length 3 to spice
        # up the initial beam.
        empty = repairs_to_pipeline([], self.ctx)
        seeds = [empty]

        warm_reps = self.sampler.sample(3)
        if warm_reps:
            seeds.append(repairs_to_pipeline(warm_reps, self.ctx))

        scored: List[ScoredPipeline] = []
        seen = set()
        for p in seeds:
            h = p.hash()
            if h in seen:
                continue
            seen.add(h)
            sp = self._evaluate(p)
            if sp.fitness > -math.inf:
                scored.append(sp)

        scored.sort(key=lambda s: s.fitness, reverse=True)
        self.frontier = scored[: self.beam_width]

        if not self.frontier:
            # Last-resort fallback: keep the empty pipeline even if eval failed,
            # so the search has a base to expand.
            self.frontier = [ScoredPipeline(pipeline=empty)]

    # ------------------------------------------------------------------
    def run(self) -> List[ScoredPipeline]:
        t0 = time.time()
        self._seed_frontier()

        if self.verbose:
            best_init = self.frontier[0].fitness if self.frontier else float("nan")
            print(f"[AlphaClean] seed frontier  size={len(self.frontier)}  "
                  f"beam_width={self.beam_width}  gamma={self.gamma}  "
                  f"best={best_init:.4f}")

        best_so_far = self.frontier[0].fitness if self.frontier else -math.inf
        no_improve = 0

        for it in range(1, self.n_iters + 1):
            t_iter = time.time()

            # 1) Generate
            reps = self.sampler.sample(self.batch_per_iter)

            # 2) Pruner filter
            n_before = len(reps)
            kept_reps = [r for r in reps if self.pruner.predict(r)]
            n_pruned = n_before - len(kept_reps)
            if not kept_reps:
                kept_reps = reps  # never go empty

            # 3) Expand beam
            base_beam = self.frontier[: self.beam_width]
            candidates: List[Pipeline] = []
            seen = {sp.hash() for sp in self.frontier}
            for sp in base_beam:
                base_repairs = [Repair.from_pipeline_step(s) for s in sp.pipeline.steps]
                for r in kept_reps:
                    new_pipe = repairs_to_pipeline(base_repairs + [r], self.ctx)
                    h = new_pipe.hash()
                    if h in seen:
                        continue
                    seen.add(h)
                    candidates.append(new_pipe)

            # 4) Evaluate
            scored_candidates: List[ScoredPipeline] = []
            for p in candidates:
                sp = self._evaluate(p)
                if sp.fitness > -math.inf:
                    scored_candidates.append(sp)

            # 5) Frontier update
            merged = list(self.frontier) + scored_candidates
            merged_unique: dict = {}
            for sp in merged:
                merged_unique.setdefault(sp.hash(), sp)
            merged_list = list(merged_unique.values())
            merged_list.sort(key=lambda s: s.fitness, reverse=True)
            self.frontier = merged_list[: self.beam_width]

            cur_best = self.frontier[0].fitness if self.frontier else -math.inf
            improved = cur_best > best_so_far + 1e-6
            if improved:
                best_so_far = cur_best
                no_improve = 0
            else:
                no_improve += 1

            # 6) Learn pruner
            best_ops = set(self.frontier[0].pipeline.op_names()) if self.frontier else set()
            self.pruner.add_pipeline_examples(best_ops, kept_reps)
            self.pruner.fit_if_ready()

            iter_record = {
                "iter": it,
                "best_fitness": cur_best,
                "mean_fitness": (
                    sum(s.fitness for s in self.frontier) / max(len(self.frontier), 1)
                ),
                "n_candidates": len(candidates),
                "n_pruned": n_pruned,
                "frontier_size": len(self.frontier),
                "best_ops": list(best_ops),
                "duration_seconds": time.time() - t_iter,
                "pruner_stats": self.pruner.stats,
            }
            self.history.append(iter_record)

            if self.verbose:
                print(
                    f"[AlphaClean] iter={it}/{self.n_iters}  "
                    f"best={cur_best:.4f}  "
                    f"frontier={len(self.frontier)}  "
                    f"cands={len(candidates)} (pruned={n_pruned})  "
                    f"no_improve={no_improve}  "
                    f"ops={list(best_ops)}"
                )

            if no_improve >= self.patience:
                if self.verbose:
                    print(f"[AlphaClean] early stop after iter {it}")
                break

        if self.verbose:
            print(f"[AlphaClean] search done in {time.time()-t0:.1f}s; "
                  f"best fitness = {best_so_far:.4f}")

        return self.frontier


__all__ = ["AlphaCleanSearcher", "ScoredPipeline"]
