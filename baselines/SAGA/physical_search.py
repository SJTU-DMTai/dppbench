"""Physical-pipeline tuning via random search.

Given the top-K *logical* pipelines from ``LogicalSearch``, this module tunes
the hyperparameters of each operator (using the ``param_space`` declared in
``operator_catalog.CATALOG``) and returns the best instance.
"""
from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass

from baselines.common.evaluator import EvaluationResult, PipelineEvaluator
from .operator_catalog import CATALOG
from baselines.common.pipeline import Pipeline, assign_dag_structure


@dataclass
class TunedPipeline:
    pipeline: Pipeline
    fitness: float
    metrics: dict


def _sample_param(space_value, rng: random.Random):
    """Sample one value from a param-space entry.

    The entry is either a list (categorical) or a 2-tuple of floats (range).
    """
    if isinstance(space_value, list):
        return rng.choice(space_value)
    if isinstance(space_value, tuple) and len(space_value) == 2:
        lo, hi = space_value
        if isinstance(lo, int) and isinstance(hi, int):
            return rng.randint(lo, hi)
        return rng.uniform(float(lo), float(hi))
    return space_value


class PhysicalSearch:
    def __init__(
        self,
        evaluator: PipelineEvaluator,
        n_trials: int = 5,
        seed: int = 1234,
        verbose: bool = True,
    ):
        self.evaluator = evaluator
        self.n_trials = n_trials
        self.rng = random.Random(seed)
        self.verbose = verbose

    def tune_one(self, base: Pipeline, base_fitness: float, base_metrics: dict) -> TunedPipeline:
        best_pipe = base.copy()
        best_fitness = base_fitness
        best_metrics = dict(base_metrics)

        if self.verbose:
            print(f"[Phys] tuning pipeline (len={len(base)})  base={base_fitness:.4f}")

        for trial in range(self.n_trials):
            cand = base.copy()
            for step in cand.steps:
                spec = CATALOG.get(step.op)
                if spec is None or not spec.param_space:
                    continue
                for k, space in spec.param_space.items():
                    new_val = _sample_param(space, self.rng)
                    step.params[k] = new_val
            # Parameter tuning may affect table-reference params in future catalogs;
            # keep the explicit DAG prev/source fields in sync before evaluation.
            ctx = getattr(self.evaluator, "ctx", None)
            if ctx is not None:
                assign_dag_structure(cand, ctx, self.rng)
            ev: EvaluationResult = self.evaluator.evaluate(cand)
            if ev.fitness > best_fitness:
                best_fitness = ev.fitness
                best_pipe = cand
                best_metrics = ev.metrics
                if self.verbose:
                    print(f"  [Phys trial {trial+1}] new best={best_fitness:.4f}")
            elif self.verbose:
                marker = "ok" if ev.success else "FAIL"
                fit_repr = f"{ev.fitness:.4f}" if ev.fitness > -math.inf else "-inf"
                print(f"  [Phys trial {trial+1}] {marker} fit={fit_repr}")

        return TunedPipeline(pipeline=best_pipe, fitness=best_fitness, metrics=best_metrics)

    def tune(self, candidates: list[tuple[Pipeline, float, dict]]) -> list[TunedPipeline]:
        out = []
        for pipe, fit, metrics in candidates:
            out.append(self.tune_one(pipe, fit, metrics))
        out.sort(key=lambda t: t.fitness, reverse=True)
        return out
