"""HAIPipe enumeration, rule pruning, and active candidate selection."""
from __future__ import annotations

import itertools
import random as _random
from dataclasses import dataclass, field

from baselines.common.pipeline import Pipeline, PipelineStep, assign_dag_structure, check_pipeline

from .active_learning import ActiveCandidateSelector, pipeline_features


@dataclass
class HAISelection:
    pipeline: Pipeline
    fitness: float
    metrics: dict = field(default_factory=dict)
    error: str | None = None
    candidates: list[dict] = field(default_factory=list)


def _copy(step: PipelineStep) -> PipelineStep:
    return PipelineStep.from_dict(step.to_dict())


def _subsequences(steps: list[PipelineStep]):
    """All non-empty ordered subsets, matching upstream end-sequence enumeration."""
    for length in range(len(steps), 0, -1):
        for indices in itertools.combinations(range(len(steps)), length):
            yield [_copy(steps[i]) for i in indices]


def _candidate_from_slots(hi: Pipeline, ai: list[PipelineStep], slots: tuple[int, ...], ctx) -> Pipeline:
    buckets: dict[int, list[PipelineStep]] = {}
    for slot, step in zip(slots, ai):
        buckets.setdefault(slot, []).append(_copy(step))
    steps = []
    for slot in range(len(hi.steps) + 1):
        steps.extend(buckets.get(slot, []))
        if slot < len(hi.steps):
            steps.append(_copy(hi.steps[slot]))
    candidate = Pipeline(steps=steps, sources=hi.sources)
    return assign_dag_structure(candidate, ctx, _random.Random(0))


def combine_pipelines(
    hi_pipeline: Pipeline,
    ai_pipeline: Pipeline,
    ctx,
    rng: _random.Random,
    *,
    max_len: int | None,
    n_combinations: int | None = None,
) -> list[Pipeline]:
    """Enumerate ordered AI subsets over every before/after/end insertion slot.

    DPPBench YAML operators are atomic transformations, so an insertion slot
    cannot fall inside an estimator's fit/transform pair.  The train node is
    outside ``steps`` and therefore cannot be crossed.  These two structural
    facts implement upstream pruning rules 2 and 3 directly.
    """
    del rng, n_combinations  # Enumeration is deterministic; AL limits executions.
    candidates = [hi_pipeline.copy()]
    seen = {hi_pipeline.hash()}
    for subsequence in _subsequences(ai_pipeline.steps):
        if max_len is not None and len(hi_pipeline) + len(subsequence) > int(max_len):
            continue
        for slots in itertools.combinations_with_replacement(
            range(len(hi_pipeline.steps) + 1), len(subsequence)
        ):
            candidate = _candidate_from_slots(hi_pipeline, subsequence, slots, ctx)
            report = check_pipeline(candidate, ctx)
            if not report.valid:
                continue
            key = candidate.hash()
            if key not in seen:
                seen.add(key)
                candidates.append(candidate)
    return candidates


def select_best_candidate(
    evaluator,
    candidates: list[Pipeline],
    *,
    selector: ActiveCandidateSelector,
    dataset_feature,
    vocabulary: list[str],
) -> HAISelection:
    features = __import__("numpy").stack([
        pipeline_features(candidate, vocabulary) for candidate in candidates
    ])
    best_index, records = selector.select(dataset_feature, features, evaluator, candidates)
    best = candidates[best_index].copy()
    result = evaluator.evaluate(best)
    return HAISelection(
        pipeline=best,
        fitness=float(result.fitness) if result.success else float("-inf"),
        metrics=dict(result.metrics or {}),
        error=result.error,
        candidates=records,
    )


__all__ = ["HAISelection", "combine_pipelines", "select_best_candidate"]
