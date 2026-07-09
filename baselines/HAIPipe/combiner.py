"""HAIPipe combination and selection utilities."""
from __future__ import annotations

import random as _random
from dataclasses import dataclass, field

from baselines.common.data_ctx import DataContext
from baselines.common.pipeline import Pipeline, PipelineStep, finalize_pipeline


@dataclass
class HAISelection:
    pipeline: Pipeline
    fitness: float
    metrics: dict = field(default_factory=dict)
    error: str | None = None
    candidates: list[dict] = field(default_factory=list)


def _copy_steps(steps: list[PipelineStep]) -> list[PipelineStep]:
    return [PipelineStep.from_dict(step.to_dict()) for step in steps]


def combine_pipelines(
    hi_pipeline: Pipeline,
    ai_pipeline: Pipeline,
    ctx: DataContext,
    rng: _random.Random,
    *,
    max_len: int | None,
    n_combinations: int,
) -> list[Pipeline]:
    """Enumerate/sample legal HAI-pipeline candidates from HI + AI.

    The original HAIPipe inserts AI operations into the HI-pipeline at multiple
    positions.  DPPBench legality is expressed by common finalization, so each
    enumerated insertion is repaired, assigned a DAG, validated, and deduped.
    """
    raw: list[Pipeline] = [hi_pipeline.copy()]
    hi_steps = _copy_steps(hi_pipeline.steps)
    ai_steps = _copy_steps(ai_pipeline.steps)

    if ai_steps:
        for end in range(1, len(ai_steps) + 1):
            prefix = ai_steps[:end]
            for pos in range(len(hi_steps) + 1):
                raw.append(Pipeline(steps=_copy_steps(hi_steps[:pos] + prefix + hi_steps[pos:])))
        for idx, step in enumerate(ai_steps):
            for pos in range(len(hi_steps) + 1):
                raw.append(Pipeline(steps=_copy_steps(hi_steps[:pos] + [step] + hi_steps[pos:])))
        raw.append(Pipeline(steps=_copy_steps(hi_steps + ai_steps)))

    rng.shuffle(raw)
    out: list[Pipeline] = []
    seen: set[str] = set()
    limit = max(1, int(n_combinations))
    for pipe in raw:
        candidate, report = finalize_pipeline(pipe, ctx, rng, max_len=max_len)
        if not report.valid:
            continue
        key = candidate.hash()
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
        if len(out) >= limit:
            break
    if not out:
        fallback, report = finalize_pipeline(hi_pipeline.copy(), ctx, rng, max_len=max_len)
        if report.valid:
            out.append(fallback)
    return out


def select_best_candidate(evaluator, candidates: list[Pipeline]) -> HAISelection:
    """Evaluate sampled HAI candidates and return the best downstream result."""
    best_pipe: Pipeline | None = None
    best_fitness = float("-inf")
    best_metrics: dict = {}
    best_error: str | None = None
    records: list[dict] = []

    for pipe in candidates:
        try:
            ev = evaluator.evaluate(pipe)
            success = bool(ev.success)
            fitness = float(ev.fitness) if success else float("-inf")
            metrics = dict(ev.metrics or {})
            error = ev.error
        except Exception as exc:  # pragma: no cover - defensive safety net
            success = False
            fitness = float("-inf")
            metrics = {}
            error = f"{type(exc).__name__}: {exc}"
        records.append({
            "ops": pipe.op_names(),
            "fitness": fitness if fitness != float("-inf") else None,
            "success": success,
            "metrics": metrics,
            "error": error,
        })
        if fitness > best_fitness or best_pipe is None:
            best_pipe = pipe.copy()
            best_fitness = fitness
            best_metrics = metrics
            best_error = error

    if best_pipe is None:
        raise RuntimeError("HAIPipe combiner received no candidates")
    return HAISelection(
        pipeline=best_pipe,
        fitness=best_fitness,
        metrics=best_metrics,
        error=best_error,
        candidates=records,
    )


__all__ = ["HAISelection", "combine_pipelines", "select_best_candidate"]
