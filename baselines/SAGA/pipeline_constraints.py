"""Pipeline legality checks and structural repair.

After GA crossover/mutation, a pipeline may violate ordering constraints
(e.g. ``DiscretizeFeature`` after ``SampleNegative``).
This module enforces a canonical ordering and removes duplicates, guaranteeing
that the resulting pipeline is executable by ``dppbench.dataset``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .operator_catalog import CATALOG, OpCategory

if TYPE_CHECKING:
    from .pipeline import DataContext, Pipeline, PipelineStep


# Canonical ordering of categories used to sort steps. Lower index = earlier.
_TABULAR_ORDER = [
    OpCategory.SCHEMA,
    OpCategory.DATETIME_PARSE,
    OpCategory.CLEAN_VALUE,
    OpCategory.ERROR_DETECTION,
    OpCategory.DEDUPLICATE,
    OpCategory.JOIN,
    OpCategory.RESHAPE_PIVOT,
    OpCategory.RESHAPE_LONGWIDE,
    OpCategory.RESHAPE_STRING,
    OpCategory.GROUP_AGG,
    OpCategory.SORT_ORDER,
    OpCategory.FEATURE_GEN,
    OpCategory.FEATURE_TIME,
    OpCategory.OUTLIER,
    OpCategory.MISSING_VALUE,
    OpCategory.DISCRETIZATION,
    OpCategory.ENCODING,
    OpCategory.NORMALIZATION,
    OpCategory.SCALING,
    OpCategory.DISTRIBUTION_RESHAPE,
    OpCategory.FEATURE_REDUCTION,
    OpCategory.FEATURE_SELECTION,
    OpCategory.FILTER_COL,
    OpCategory.FILTER_ROW,
    OpCategory.IMBALANCE,
    OpCategory.AUGMENT,
]
_TABULAR_RANK = {c: i for i, c in enumerate(_TABULAR_ORDER)}

_REC_ORDER = [
    OpCategory.SCHEMA,
    OpCategory.JOIN,
    OpCategory.DATETIME_PARSE,
    OpCategory.CLEAN_VALUE,
    OpCategory.FILTER_COL,
    OpCategory.FILTER_ROW,
    OpCategory.OUTLIER,
    OpCategory.NORMALIZATION,
    OpCategory.SCALING,
    OpCategory.DISTRIBUTION_RESHAPE,
    OpCategory.FEATURE_GEN,
    OpCategory.FEATURE_TIME,
    OpCategory.FEATURE_REDUCTION,
    OpCategory.FEATURE_SELECTION,
    OpCategory.ENCODING,
    OpCategory.SEQUENCE,
    OpCategory.DISCRETIZATION,
    OpCategory.MISSING_VALUE,
    OpCategory.SAMPLING,
]
_REC_RANK = {c: i for i, c in enumerate(_REC_ORDER)}


def _rank(op_name: str, task_type: str) -> int:
    spec = CATALOG[op_name]
    if task_type == "tabular":
        return _TABULAR_RANK.get(spec.category, len(_TABULAR_ORDER))
    return _REC_RANK.get(spec.category, len(_REC_ORDER))


def is_legal(pipeline: "Pipeline", task_type: str) -> bool:
    # general rule: ranks must be non-decreasing
    ranks = [_rank(n, task_type) for n in pipeline.op_names()]
    return ranks == sorted(ranks)


def repair(pipeline: "Pipeline", task_type: str, ctx: "DataContext") -> "Pipeline":
    """In-place structural repair. Returns the same pipeline."""
    # 1. Deduplicate by op_name (keep first occurrence)
    seen = set()
    deduped: list = []
    for s in pipeline.steps:
        if s.op in seen:
            continue
        seen.add(s.op)
        deduped.append(s)
    pipeline.steps = deduped

    # 2. Remove operators that are illegal for the task type
    pipeline.steps = [
        s for s in pipeline.steps
        if CATALOG[s.op].task_type in (task_type, "both")
    ]

    # 3. Ensure mandatory operators are present
    if task_type == "rec":
        from .pipeline import make_step
        import random as _r
        rng = _r.Random()
        existing = {s.op for s in pipeline.steps}
        for mand in ("JoinTable", "CreateSequence", "SampleNegative"):
            if mand not in existing:
                step = make_step(mand, ctx, rng)
                if step is not None:
                    pipeline.steps.append(step)
    else:  # tabular
        ensure_tabular_tail(pipeline, ctx)

    # 4. Sort by canonical rank, stable
    pipeline.steps.sort(key=lambda s: _rank(s.op, task_type))
    return pipeline


def ensure_tabular_tail(pipeline: "Pipeline", ctx: "DataContext") -> None:
    """Ensure tabular pipelines end with LabelEncode then HandleMV."""
    ops = pipeline.op_names()
    from .pipeline import make_step
    import random as _r
    rng = _r.Random()
    if "LabelEncode" not in ops:
        step = make_step("LabelEncode", ctx, rng)
        if step is not None:
            pipeline.steps.append(step)
    if "HandleMV" not in pipeline.op_names():
        step = make_step("HandleMV", ctx, rng)
        if step is not None:
            pipeline.steps.append(step)
