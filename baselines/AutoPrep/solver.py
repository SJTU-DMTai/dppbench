"""MPBP solver for Auto-Prep.

Builds transformation trees over the shared DPPBench operator space and combines
one selected path with schema-driven join edges.  The executable pipeline is
always finalized through ``baselines.common.pipeline`` so parameter names,
required parameters, DAG sources, and task legality are checked in one place.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random as _random
from dataclasses import dataclass

import yaml

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
    coverage: bool = False


def _node_to_steps(
    node: TreeNode,
    target: str,
) -> list[PipelineStep]:
    out: list[PipelineStep] = []
    for op_name in node.op_chain:
        out.append(PipelineStep(op=op_name, target=target,
                                params=copy.deepcopy(node.params_per_op[op_name])))
    return out


def _join_step_for_edge(
    edge: JoinEdge,
    ctx: DataContext,
    rng: _random.Random,
) -> PipelineStep | None:
    if edge.op_name != "JoinTable":
        return None
    if edge.aux_ref is None:
        step = make_step("JoinTable", ctx, rng)
        if step is not None:
            step.target = edge.target
        return step
    table = ctx.tables.get(edge.aux_ref)
    if table is None or not table.join_keys_to_primary:
        return None
    key = sorted(table.join_keys_to_primary,
                 key=lambda k: (k['primary'], k.get('secondary', k['primary'])))[0]
    params = {"aux_df": f"${edge.aux_ref}", "key_col": key['primary'],
              "method": "key" if table.join_key_unique.get(key['primary'], False) else "agg",
              "how": "left"}
    if validate_step_params("JoinTable", params, ctx):
        return None
    return PipelineStep(op="JoinTable", target=edge.target, params=params)


def pipeline_signature(pipeline: Pipeline) -> str:
    """Canonical full DAG identity, including parameters, sources and order."""
    canonical = json.dumps(yaml.safe_load(pipeline.to_yaml()), sort_keys=True,
                           separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


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
    evaluated_joins: set[str] | None = None,
    coverage_seed: int | None = None,
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

    if n_candidates < 1 or beam < 1:
        raise ValueError('AutoPrep requires positive beam and candidate counts')
    evaluated_joins = evaluated_joins or set()
    edges = sorted(j_model.mandatory_edges() + j_model.optional_edges(),
                   key=lambda e: (-j_model.prob(e.name), e.name))
    pending = sorted((edge for edge in edges if edge.name not in evaluated_joins),
                     key=lambda e: e.name)

    def assemble(leaf, edge, coverage=False):
        # Use the same preprocessing parameters for every initial join probe.
        # Integration precedes transformations that may remove a join key.
        candidate_rng = _random.Random(coverage_seed if coverage and coverage_seed is not None else seed)
        steps = []
        if edge is not None:
            step = _join_step_for_edge(edge, ctx, candidate_rng)
            if step is None:
                return None
            steps.append(step)
        target = "interaction" if ctx.task_type == "rec" else "both"
        steps.extend(_node_to_steps(leaf, target))
        expected = [copy.deepcopy(steps[0].params)] if edge is not None else []
        try:
            pipeline = repair_pipeline(Pipeline(steps=steps), ctx, rng=candidate_rng)
        except ValueError:
            return None
        actual = [step for step in pipeline.steps if step.op == 'JoinTable']
        if [step.params for step in actual] != expected:
            raise ValueError('AutoPrep finalized join differs from the selected feedback edge')
        ops = tuple(step.op for step in pipeline.steps)
        joins = (edge.name,) if edge is not None else ()
        score = sum(math.log(max(t_model.prob(op), 1e-9)) for op in ops)
        score += sum(math.log(max(j_model.prob(name), 1e-9)) for name in joins)
        return Candidate(pipeline, score, ops, joins, coverage)

    # Reserve the first slots for previously unattempted joins with an identical
    # empty transformation path. State advances only after evaluator feedback.
    probes = [assemble(TreeNode(), edge, coverage=True) for edge in pending[:n_candidates]]
    expanded = [assemble(leaf, edge) for leaf in main_leaves for edge in (edges or [None])]
    expanded = sorted((c for c in expanded if c is not None),
                      key=lambda c: (-c.log_score, pipeline_signature(c.pipeline)))
    seen = set()
    top = []
    for candidate in probes + expanded:
        if candidate is None:
            continue
        signature = pipeline_signature(candidate.pipeline)
        if signature in seen:
            continue
        seen.add(signature)
        top.append(candidate)
        if len(top) >= n_candidates:
            break
    return top


__all__ = ["solve", "Candidate", "repair_pipeline", "pipeline_signature"]
