"""Per-table transformation tree (depth m=2, beam search).

Implements the construction of ``G(T_i)`` from the Auto-Prep paper. Each node
on the tree represents one operator from the dppbench library; a path from
root to leaf is a candidate transformation sequence ``S_i``.

We use beam search with branching factor ``branching`` and depth
``max_depth`` (paper default m=2). The score of each path is the product of
operator probabilities (``M_T+`` outputs).
"""
from __future__ import annotations

import random as _random
from dataclasses import dataclass, field

from baselines.SAGA.pipeline import DataContext

from .operator_catalog import CATALOG, OpCategory, operators_for_task
from .pipeline_factory import build_default_params
from .transformation_model import TransformationModel


# ---------------------------------------------------------------------------
# Whitelists per table kind
# ---------------------------------------------------------------------------
# Operators that act at row-level on the rec interaction table.
# Expanded to a broad whitelist of column-level safe ops so that AutoPrep can
# explore richer rec pipelines (issue: rec search space was too narrow).
# Excludes ops that would corrupt the (already-frozen) target column or
# require multi-table semantics that don't apply to a single interaction df.
_REC_INTERACTION = {
    # rec-only structural / mandatory
    "JoinTable", "CreateSequence", "FilterSample", "FilterKCore",
    "SampleNegative",
    # missing-value
    "HandleMV", "HandleMV", "HandleMV",
    # cleaning / value transforms
    "CustomClean", "CustomClean", "HandleError",
    # outliers
    "HandleOutlier",
    # datetime
    "ParseDate", "ParseDate", "ExtractDateTimeFeature",
    # schema / column drops
    "CastType", "RenameColumn", "CustomProcess", "CustomProcess",
    "SelectFeature",
    # encoding
    "OneHotEncode", "OrdinalEncode", "LabelEncode",
    "CustomProcess", "HashEncode", "TargetEncode",
    # scaling / distribution reshape
    "ScaleFeature", "ScaleFeature", "ScaleFeature", "ScaleFeature", "TransformPower",
    "TransformPower", "TransformPower",
    # discretization
    "DiscretizeFeature",
    # feature gen
    "CrossFeature", "CreateFeature",
}

# Side tables: only column-level cleaning / lightweight encoding.
_REC_SIDE = {
    "HandleMV", "LabelEncode", "CustomProcess",
    "CustomProcess", "CustomProcess", "ScaleFeature", "ScaleFeature",
    "SelectFeature",
}

# Tabular auxiliary table: light cleaning + later joined back via JoinTable.
_TABULAR_AUX = {
    "HandleMV", "LabelEncode", "CustomProcess",
    "CustomProcess", "CustomProcess",
    "JoinTable", "JoinTable",
}


def _whitelist_for_main_tabular() -> set[str]:
    # All tabular operators (the catalog has already been pruned to remove
    # full-table-replacing ops such as PivotTable / GroupByAggregate).
    return set(operators_for_task("tabular"))


def whitelist_for(table_kind: str) -> set[str]:
    if table_kind == "interaction":
        return set(_REC_INTERACTION)
    if table_kind in ("user_df", "item_df"):
        return set(_REC_SIDE)
    if table_kind == "main_tabular":
        return _whitelist_for_main_tabular()
    if table_kind == "aux_df":
        return set(_TABULAR_AUX)
    raise ValueError(f"Unknown table_kind: {table_kind}")


# ---------------------------------------------------------------------------
# Tree node + builder
# ---------------------------------------------------------------------------
@dataclass
class TreeNode:
    op_chain: tuple[str, ...] = ()
    log_prob: float = 0.0
    params_per_op: dict[str, dict] = field(default_factory=dict)

    @property
    def prob_product(self) -> float:
        # Approximate exp(log_prob) but keep numerically stable.
        import math
        return math.exp(max(min(self.log_prob, 30), -30))

    def extend(self, op_name: str, prob: float, params: dict) -> "TreeNode":
        import math
        log_p = math.log(max(prob, 1e-9))
        new_params = dict(self.params_per_op)
        new_params[op_name] = params
        return TreeNode(
            op_chain=self.op_chain + (op_name,),
            log_prob=self.log_prob + log_p,
            params_per_op=new_params,
        )


def build_transformation_tree(
    ctx: DataContext,
    t_model: TransformationModel,
    table_kind: str,
    max_depth: int = 2,
    branching: int = 8,
    rng: _random.Random | None = None,
) -> list[TreeNode]:
    """Return the leaf nodes of ``G(T_i)`` after beam search.

    ``max_depth`` is the number of optional operator slots; each leaf
    corresponds to a (possibly empty) chain of operators.
    """
    rng = rng or _random.Random(42)

    whitelist = whitelist_for(table_kind)
    candidate_ops: list[tuple[str, float, dict]] = []
    for op_name in whitelist:
        spec = CATALOG[op_name]
        # Skip mandatory ops here; they are appended outside the tree by the
        # solver to ensure they always end up in the final pipeline.
        if spec.mandatory:
            continue
        params = build_default_params(op_name, ctx, rng)
        if params is None:
            continue
        prob = t_model.prob(op_name)
        if prob <= 1e-3:
            continue
        candidate_ops.append((op_name, prob, params))

    # Sort once by prob descending; we re-rank within each layer below.
    candidate_ops.sort(key=lambda x: -x[1])

    # Beam search
    beam: list[TreeNode] = [TreeNode()]
    for _depth in range(max_depth):
        next_beam: list[TreeNode] = []
        for node in beam:
            used = set(node.op_chain)
            for op_name, prob, params in candidate_ops:
                if op_name in used:
                    continue
                next_beam.append(node.extend(op_name, prob, params))
            # Allow keeping the node as-is (early termination of this branch).
            next_beam.append(node)
        next_beam.sort(key=lambda n: -n.log_prob)
        # de-duplicate by op_chain
        seen = set()
        deduped = []
        for n in next_beam:
            if n.op_chain in seen:
                continue
            seen.add(n.op_chain)
            deduped.append(n)
        beam = deduped[:branching]

    return beam


__all__ = ["TreeNode", "build_transformation_tree", "whitelist_for"]
