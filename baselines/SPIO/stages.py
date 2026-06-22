"""Stage definitions and scaffolding helpers for the SPIO baseline.

SPIO uses 5 stage labels mirroring the dppbench operator subdirectories:

    1. integration            (multi-table joins; not LLM-coded)
    2. cleaning               (LLM-coded -> CustomOp)
    3. preprocessing          (LLM-coded -> CustomOp)
    4. feature_engineering    (LLM-coded -> CustomOp)
    5. transformation         (LLM-coded -> CustomOp)

The per-stage code generation only runs for the 4 single-table stages
(``CODE_STAGES``). ``integration`` and the rec-specific structural ops
(``JoinTable`` / ``CreateSequence`` /
``SampleNegative``) cannot be expressed under the ``CustomOp`` single-table
df->df contract, so they are injected structurally by the orchestrator via
``baselines.SAGA.pipeline.make_step``.
"""
from __future__ import annotations

import random as _random
from typing import Optional

from baselines.common.operator_catalog import CATALOG, OpCategory
from baselines.SAGA.pipeline import (
    DataContext,
    Pipeline,
    PipelineStep,
    make_step,
)
from baselines.SAGA.pipeline_constraints import ensure_tabular_tail


# ---------------------------------------------------------------------------
# Stage labels
# ---------------------------------------------------------------------------
STAGES: list[str] = [
    "integration",
    "cleaning",
    "preprocessing",
    "feature_engineering",
    "transformation",
]

# Stages for which SPIO actually generates ``CustomOp`` code.
CODE_STAGES: list[str] = [
    "cleaning",
    "preprocessing",
    "feature_engineering",
    "transformation",
]

# Map each stage to the OpCategory members that fall under it. Used to
# describe the stage's typical operators in the NL-plan prompt.
_STAGE_CATEGORIES: dict[str, tuple[OpCategory, ...]] = {
    "integration": (OpCategory.JOIN,),
    "cleaning": (
        OpCategory.MISSING_VALUE,
        OpCategory.OUTLIER,
        OpCategory.DEDUPLICATE,
        OpCategory.ERROR_DETECTION,
        OpCategory.CLEAN_VALUE,
        OpCategory.DATETIME_PARSE,
    ),
    "preprocessing": (
        OpCategory.SCHEMA,
        OpCategory.FILTER_COL,
        OpCategory.FILTER_ROW,
        OpCategory.ENCODING,
        OpCategory.SCALING,
        OpCategory.DISTRIBUTION_RESHAPE,
        OpCategory.NORMALIZATION,
        OpCategory.DISCRETIZATION,
        OpCategory.IMBALANCE,
        OpCategory.AUGMENT,
        OpCategory.SAMPLING,
    ),
    "feature_engineering": (
        OpCategory.FEATURE_GEN,
        OpCategory.FEATURE_TIME,
        OpCategory.FEATURE_SELECTION,
        OpCategory.FEATURE_REDUCTION,
        OpCategory.SEQUENCE,
    ),
    "transformation": (
        OpCategory.RESHAPE_PIVOT,
        OpCategory.RESHAPE_LONGWIDE,
        OpCategory.RESHAPE_STRING,
        OpCategory.SORT_ORDER,
        OpCategory.GROUP_AGG,
    ),
}


def stage_op_descriptions(stage: str, task_type: str, max_per_cat: int = 6) -> str:
    """Return a markdown bullet list of typical ops in ``stage`` for the
    given task type. Used as inspirational context inside the NL plan
    prompt; the LLM is NOT required to use these operators (it generates a
    ``CustomOp`` snippet), but knowing what falls under each stage helps it
    plan in dppbench's vocabulary.
    """
    cats = _STAGE_CATEGORIES.get(stage, ())
    if not cats:
        return "(no typical operators)"
    lines: list[str] = []
    for cat in cats:
        ops: list[str] = []
        for name, spec in CATALOG.items():
            if spec.task_type not in (task_type, "both"):
                continue
            if spec.category != cat:
                continue
            ops.append(name)
        if not ops:
            continue
        ops = sorted(ops)[:max_per_cat]
        lines.append(f"- {cat.value}: {', '.join(ops)}")
    return "\n".join(lines) if lines else "(no typical operators)"


# ---------------------------------------------------------------------------
# Structural scaffolding (integration + rec-mandatory ops)
# ---------------------------------------------------------------------------
def build_scaffolded_pipeline(
    code_steps: list[PipelineStep],
    ctx: DataContext,
    rng: Optional[_random.Random] = None,
) -> Pipeline:
    """Wrap a list of ``CustomOp`` steps with the structural operators that
    cannot be expressed under the single-table df->df contract.

    Tabular: prepend ``JoinTable``/``JoinTable`` whenever auxiliary tables
    exist, then run the user's CustomOp steps, and finally make sure the
    LightGBM-required ``LabelEncode`` + ``HandleMV`` tail is present.

    Rec: prepend ``JoinTable`` (when user/item dfs exist), then the user's
    CustomOp steps, then the mandatory structural tail
    (``CreateSequence`` ->
    ``SampleNegative``) following ``baselines.SAGA.pipeline._REC_ORDER``.
    """
    rng = rng or _random.Random(0)
    pipe = Pipeline()

    if ctx.task_type == "rec":
        # ---- Rec: integration ----
        if ctx.has_user_df or ctx.has_item_df:
            step = make_step("JoinTable", ctx, rng)
            if step is not None:
                pipe.steps.append(step)

        # ---- Rec: LLM-coded stages (interaction-only) ----
        for s in code_steps:
            pipe.steps.append(s)

        # ---- Rec: mandatory structural tail (matching _REC_ORDER) ----
        for op_name in (
            "CreateSequence",
            "SampleNegative",
        ):
            step = make_step(op_name, ctx, rng)
            if step is not None:
                pipe.steps.append(step)
        return pipe

    # ---- Tabular ----
    if ctx.aux_dfs:
        # Pick the first auxiliary table integration op that resolves with
        # the current context. ``JoinTable`` is preferred when an id_col is
        # available; otherwise fall back to ``JoinTable``.
        for op_name in ("JoinTable", "JoinTable"):
            step = make_step(op_name, ctx, rng)
            if step is not None:
                pipe.steps.append(step)
                break

    for s in code_steps:
        pipe.steps.append(s)

    ensure_tabular_tail(pipe, ctx)
    return pipe


def build_prefix_scaffolded_pipeline(
    code_steps: list[PipelineStep],
    ctx: DataContext,
    rng: Optional[_random.Random] = None,
) -> Pipeline:
    """Build the scaffold used for stage observations.

    This includes integration steps that affect the visible schema, but avoids
    final-only tail operations such as split/sampling/label encoding.
    """
    rng = rng or _random.Random(0)
    pipe = Pipeline()

    if ctx.task_type == "rec":
        if ctx.has_user_df or ctx.has_item_df:
            step = make_step("JoinTable", ctx, rng)
            if step is not None:
                pipe.steps.append(step)
        pipe.steps.extend(code_steps)
        return pipe

    if ctx.aux_dfs:
        for op_name in ("JoinTable", "JoinTable"):
            step = make_step(op_name, ctx, rng)
            if step is not None:
                pipe.steps.append(step)
                break
    pipe.steps.extend(code_steps)
    return pipe


__all__ = [
    "STAGES",
    "CODE_STAGES",
    "stage_op_descriptions",
    "build_scaffolded_pipeline",
    "build_prefix_scaffolded_pipeline",
]
