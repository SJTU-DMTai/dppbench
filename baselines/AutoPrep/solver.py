"""MPBP solver for Auto-Prep.

Builds the global graph G(T) (one transformation tree per table + a
schema-driven join model), enumerates top-K candidate pipelines via beam
search, and applies the shared dpp-bench repair step so Auto-Prep follows the
same canonical legality/order constraints as the other baselines.
"""
from __future__ import annotations

import copy
import math
import random as _random
from dataclasses import dataclass, field
from itertools import islice
from typing import Optional

from baselines.SAGA.pipeline import DataContext, Pipeline, PipelineStep
from baselines.SAGA.pipeline_constraints import repair

from .operator_catalog import CATALOG
from .pipeline_factory import build_default_params, make_step
from .transformation_model import JoinEdge, JoinModel, TransformationModel
from .transformation_tree import TreeNode, build_transformation_tree


# ---------------------------------------------------------------------------
# Shared repair adapter
# ---------------------------------------------------------------------------
def repair_pipeline(pipe: Pipeline, ctx: DataContext) -> Pipeline:
    """Order, dedup and ensure mandatory operators via shared constraints."""
    repair(pipe, ctx.task_type, ctx)
    return pipe


# ---------------------------------------------------------------------------
# Candidate pipeline assembly
# ---------------------------------------------------------------------------
@dataclass
class Candidate:
    pipeline: Pipeline
    log_score: float = 0.0
    op_set: tuple[str, ...] = ()  # for logging/probability accounting only
    join_set: tuple[str, ...] = ()  # join edge names used


def _node_to_steps(
    node: TreeNode,
    ctx: DataContext,
    target: str,
) -> list[PipelineStep]:
    out: list[PipelineStep] = []
    for op_name in node.op_chain:
        params = node.params_per_op.get(op_name)
        if params is None:
            continue
        out.append(PipelineStep(op=op_name, target=target, params=copy.deepcopy(params)))
    return out


def _add_join_steps(
    candidate_steps: list[PipelineStep],
    join_edges: list[JoinEdge],
    ctx: DataContext,
    rng: _random.Random,
) -> tuple[list[PipelineStep], list[str]]:
    used_names: list[str] = []
    for edge in join_edges:
        if edge.op_name == "JoinTable":
            params = build_default_params("JoinTable", ctx, rng)
            if params is None:
                continue
            if edge.aux_ref is not None and ctx.task_type != "rec":
                params["aux_df"] = f"${edge.aux_ref}"
                params["key_col"] = ctx.id_col
                params["method"] = "key"
                if "prefix" not in params:
                    pass
                if rng.random() < 0.5:
                    params["prefix"] = edge.aux_ref.upper()[:8]
                    params["max_cols"] = 20
            candidate_steps.append(PipelineStep(
                op="JoinTable", target=edge.target, params=params,
            ))
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
    """Return top ``n_candidates`` candidate pipelines.

    For rec tasks we build:
      G_interaction (m=2) + G_user (m=1) + G_item (m=1) + JoinTable (mandatory).
    For tabular we build:
      G_main (m=2) + per-aux G_aux (m=1) + JoinTable / JoinTable edges.
    """
    rng = _random.Random(seed)

    # ---- 1. Build per-table transformation trees -------------------------
    if ctx.task_type == "rec":
        main_leaves = build_transformation_tree(
            ctx, t_model, "interaction", max_depth=max_depth,
            branching=beam * 2, rng=rng,
        )
        # Side tables: at most one column-level op
        side_user = build_transformation_tree(
            ctx, t_model, "user_df", max_depth=1,
            branching=beam, rng=rng,
        ) if ctx.has_user_df else [TreeNode()]
        side_item = build_transformation_tree(
            ctx, t_model, "item_df", max_depth=1,
            branching=beam, rng=rng,
        ) if ctx.has_item_df else [TreeNode()]
        # Cap to beam size
        main_leaves = main_leaves[:beam]
        side_user = side_user[:beam]
        side_item = side_item[:beam]
    else:
        main_leaves = build_transformation_tree(
            ctx, t_model, "main_tabular", max_depth=max_depth,
            branching=beam * 2, rng=rng,
        )[:beam]
        side_user = side_item = [TreeNode()]

    # ---- 2. Determine join edges (greedy max-weight spanning tree) ------
    mandatory_edges = j_model.mandatory_edges()
    optional_edges = sorted(
        j_model.optional_edges(),
        key=lambda e: -j_model.prob(e.name),
    )
    # Greedy: include all mandatory edges; optionally include up to len(aux)
    # optional edges. Since the schema-driven graph is a star (main + each
    # aux), the spanning tree is exactly: all mandatory edges plus, for each
    # aux not yet covered, the best of {JoinTable, JoinTable}.
    chosen_optional: list[JoinEdge] = []
    covered_aux: set[str] = set()
    for edge in optional_edges:
        if edge.aux_ref and edge.aux_ref in covered_aux:
            continue
        # Roll a Bernoulli with the edge probability to honour the model.
        if rng.random() < j_model.prob(edge.name):
            chosen_optional.append(edge)
            if edge.aux_ref:
                covered_aux.add(edge.aux_ref)

    # ---- 3. Combine paths into candidate pipelines ----------------------
    candidates: list[Candidate] = []
    for leaf_main in main_leaves:
        for leaf_user in side_user:
            for leaf_item in side_item:
                pipeline = Pipeline()
                # Main table ops
                main_target = (
                    "interaction" if ctx.task_type == "rec" else "both"
                )
                pipeline.steps.extend(_node_to_steps(leaf_main, ctx, main_target))
                # Side-table ops (rec only)
                if ctx.task_type == "rec":
                    pipeline.steps.extend(_node_to_steps(leaf_user, ctx, "interaction"))
                    pipeline.steps.extend(_node_to_steps(leaf_item, ctx, "interaction"))

                # Join edges (mandatory + chosen optional)
                pipeline.steps, used_join_names = _add_join_steps(
                    pipeline.steps,
                    mandatory_edges + chosen_optional,
                    ctx, rng,
                )

                # Local repair (orders, dedups, ensures mandatory tail)
                pipeline = repair_pipeline(pipeline, ctx)

                # Compute objective: sum of log probs over actually-present ops
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

    # ---- 4. Sort + dedup + take top-K -----------------------------------
    candidates.sort(key=lambda c: -c.log_score)
    seen = set()
    top: list[Candidate] = []
    for c in candidates:
        key = (c.pipeline.hash(), c.join_set)
        if key in seen:
            continue
        seen.add(key)
        top.append(c)
        if len(top) >= n_candidates:
            break
    return top


__all__ = ["solve", "Candidate", "repair_pipeline"]
