"""MPBP solver for Auto-Prep.

Builds transformation trees over the shared DPPBench operator space and combines
one selected path with schema-driven join edges.  The executable pipeline is
always finalized through ``baselines.common.pipeline`` so parameter names,
required parameters, DAG sources, and task legality are checked in one place.
"""
from __future__ import annotations

import copy
import math
import random as _random
from dataclasses import dataclass

from baselines.common.data_ctx import DataContext
from baselines.common.pipeline import (
    Pipeline,
    PipelineStep,
    finalize_pipeline,
    make_step,
    validate_step_params,
)

from .transformation_model import JoinEdge, JoinModel, TransformationModel
from .transformation_tree import TreeNode, build_transformation_tree


# ---------------------------------------------------------------------------
# Shared repair/finalization adapter
# ---------------------------------------------------------------------------
def repair_pipeline(
    pipe: Pipeline,
    ctx: DataContext,
    rng: _random.Random | None = None,
    max_len: int | None = None,
) -> Pipeline:
    """Apply the common DPPBench pipeline contract or raise on invalidity."""
    pipe, report = finalize_pipeline(pipe, ctx, rng, max_len=max_len)
    if not report.valid:
        raise ValueError("; ".join(report.errors))
    return pipe


# ---------------------------------------------------------------------------
# Candidate pipeline assembly
# ---------------------------------------------------------------------------
@dataclass
class Candidate:
    pipeline: Pipeline
    log_score: float = 0.0
    op_set: tuple[str, ...] = ()
    join_set: tuple[str, ...] = ()


def _node_to_steps(
    node: TreeNode,
    target: str,
) -> list[PipelineStep]:
    out: list[PipelineStep] = []
    for op_name, params in node.params_per_op.items():
        if op_name not in node.op_chain:
            continue
        out.append(PipelineStep(op=op_name, target=target, params=copy.deepcopy(params)))
    return out


def _join_step_for_edge(
    edge: JoinEdge,
    ctx: DataContext,
    rng: _random.Random,
) -> PipelineStep | None:
    if edge.op_name != "JoinTable":
        return None
    step = make_step("JoinTable", ctx, rng)
    if step is None:
        return None
    step.target = edge.target
    if edge.aux_ref is not None and ctx.task_type != "rec":
        params = dict(step.params or {})
        params["aux_df"] = f"${edge.aux_ref}"
        table_ctx = ctx.tables.get(edge.aux_ref)
        join_keys = table_ctx.join_keys_to_primary if table_ctx is not None else []
        params["key_col"] = join_keys[0]["primary"] if join_keys else ctx.id_col
        params["method"] = "key"
        if rng.random() < 0.5:
            params["prefix"] = edge.aux_ref.upper()[:8]
            params["max_cols"] = 20
        errors = validate_step_params("JoinTable", params, ctx)
        if errors:
            return None
        step.params = params
    return step


def _add_join_steps(
    candidate_steps: list[PipelineStep],
    join_edges: list[JoinEdge],
    ctx: DataContext,
    rng: _random.Random,
) -> tuple[list[PipelineStep], list[str]]:
    used_names: list[str] = []
    for edge in join_edges:
        step = _join_step_for_edge(edge, ctx, rng)
        if step is None:
            continue
        candidate_steps.append(step)
        used_names.append(edge.name)
    return candidate_steps, used_names


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------
def solve(
    ctx: DataContext,
    t_model: TransformationModel,
    j_model: JoinModel,
    beam: int = 4,
    n_candidates: int = 8,
    max_depth: int = 2,
    seed: int = 42,
) -> list[Candidate]:
    """Return top ``n_candidates`` executable candidate pipelines.

    Auto-Prep's original MPBP chooses one transformation path per input table
    and then connects leaf tables with a maximum-probability join tree.  Under
    the current DPPBench executable DSL, side-table transformations are not
    independently targetable; therefore this adapter searches the primary table
    path and uses schema-driven JoinTable edges as the DPPBench-equivalent join
    tree, without pretending that user/item/aux transformations are executable.
    """
    rng = _random.Random(seed)

    if ctx.task_type == "rec":
        main_leaves = build_transformation_tree(
            ctx, t_model, "interaction", max_depth=max_depth,
            branching=beam * 2, rng=rng,
        )[:beam]
    else:
        main_leaves = build_transformation_tree(
            ctx, t_model, "main_tabular", max_depth=max_depth,
            branching=beam * 2, rng=rng,
        )[:beam]

    mandatory_edges = j_model.mandatory_edges()
    optional_edges = sorted(
        j_model.optional_edges(),
        key=lambda e: -j_model.prob(e.name),
    )
    # DPPBench's shared repair de-duplicates operator names, so a
    # pipeline can execute at most one JoinTable step.  Select the best optional
    # star edge deterministically rather than sampling invalid duplicate joins.
    chosen_optional = optional_edges[:1]

    candidates: list[Candidate] = []
    for leaf_main in main_leaves:
        pipeline = Pipeline()
        main_target = "interaction" if ctx.task_type == "rec" else "both"
        pipeline.steps.extend(_node_to_steps(leaf_main, main_target))
        pipeline.steps, used_join_names = _add_join_steps(
            pipeline.steps,
            mandatory_edges + chosen_optional,
            ctx,
            rng,
        )

        try:
            pipeline = repair_pipeline(pipeline, ctx, rng=rng)
        except ValueError:
            continue

        op_set = tuple(s.op for s in pipeline.steps)
        log_score = 0.0
        for op_name in op_set:
            log_score += math.log(max(t_model.prob(op_name), 1e-9))
        for jname in used_join_names:
            log_score += math.log(max(j_model.prob(jname), 1e-9))

        candidates.append(Candidate(
            pipeline=pipeline,
            log_score=log_score,
            op_set=op_set,
            join_set=tuple(used_join_names),
        ))

    candidates.sort(key=lambda c: -c.log_score)
    seen = set()
    top: list[Candidate] = []
    for candidate in candidates:
        key = (candidate.pipeline.hash(), candidate.join_set)
        if key in seen:
            continue
        seen.add(key)
        top.append(candidate)
        if len(top) >= n_candidates:
            break
    return top


__all__ = ["solve", "Candidate", "repair_pipeline"]
