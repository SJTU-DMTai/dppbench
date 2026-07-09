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
from baselines.common.pipeline import Pipeline, assign_dag_structure, check_pipeline, get_param_space
from baselines.common.utils import get_op


@dataclass
class TunedPipeline:
    pipeline: Pipeline
    fitness: float
    metrics: dict


def _sample_param(space_value, rng: random.Random, current_value=None):
    """Sample one value from a param-space entry.

    List-valued current params such as ``cols`` stay lists: the context-aware
    space is a pool of legal columns, not a request to replace the param with a
    single string.
    """
    if isinstance(space_value, list):
        if not space_value:
            return current_value
        if isinstance(current_value, list):
            hi = min(len(space_value), max(1, len(current_value)))
            n = rng.randint(1, hi)
            return rng.sample(space_value, k=n)
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
            ctx = getattr(self.evaluator, "ctx", None)
            for step in cand.steps:
                try:
                    spec = get_op(step.op)
                except KeyError:
                    continue
                if ctx is not None:
                    context_space = get_param_space(step.op, ctx)
                    allowed_keys = set(spec.param_space or {}) | set(spec.required_params or ())
                    space_by_key = {
                        k: v for k, v in context_space.items()
                        if k in allowed_keys
                    }
                else:
                    space_by_key = dict(spec.param_space or {})
                if not space_by_key:
                    continue
                for k, space in space_by_key.items():
                    new_val = _sample_param(space, self.rng, step.params.get(k))
                    if new_val is not None:
                        step.params[k] = new_val
            # Parameter tuning may affect table-reference params in future catalogs;
            # keep the explicit DAG prev/source fields in sync before evaluation.
            if ctx is not None:
                assign_dag_structure(cand, ctx, self.rng)
            report = check_pipeline(cand, ctx) if ctx is not None else check_pipeline(cand)
            if not report.valid:
                ev = EvaluationResult(
                    fitness=-math.inf,
                    success=False,
                    metrics={"legal_errors": report.errors},
                    error="; ".join(report.errors),
                    duration_seconds=0.0,
                )
            else:
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
