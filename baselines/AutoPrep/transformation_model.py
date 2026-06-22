"""Transformation / join probability models for Auto-Prep.

The paper's ``M_T+`` is a boosted-decision-tree model trained on offline data
with global features (column-header / value-domain overlap). We replace the
offline training with:

  * a heuristic prior log-probability assembled from each operator's
    ``prior_features`` and the inferred ``DataContext``;
  * an online multiplicative-weights update driven by downstream AUC feedback
    (see ``AutoPrep.run`` outer loop).

This keeps the reasoning structure faithful to the paper -- every operator
gets a calibrated, normalized probability -- while making the whole stack
self-contained and reproducible inside dppbench.

``JoinModel`` is intentionally schema-driven (``M_J`` simplification): the
edges are determined by what the dppbench Integration / JoinTable operators
actually accept; the probabilities still respect the paper's
``p(T,T') = max(p_tilde, 0.5)`` floor.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from baselines.SAGA.pipeline import DataContext

from .operator_catalog import CATALOG, OpCategory


def _sigmoid(x: float) -> float:
    if x > 30:
        return 1.0
    if x < -30:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


# ---------------------------------------------------------------------------
# Single-table transformation model M_T+
# ---------------------------------------------------------------------------
class TransformationModel:
    """Online probability model over the shared 52 operators."""

    def __init__(self, ctx: DataContext, eta: float = 0.5):
        self.ctx = ctx
        self.eta = float(eta)
        self.logp: dict[str, float] = {}
        self._init_priors()

    # ------------------------------------------------------------------
    def _ctx_signals(self) -> dict[str, float]:
        ctx = self.ctx
        n_num = max(1, len(ctx.numeric_cols))
        n_cat = len(ctx.categorical_cols)
        return {
            "missing": 0.5,            # prior expectation; no per-col stat at ctx level
            "missing_max": 0.5,
            "numeric": 1.0 if ctx.numeric_cols else 0.0,
            "categorical": min(1.0, n_cat / 5.0) if n_cat else 0.0,
            "high_card": 1.0 if n_cat >= 5 else 0.0,
            "many_numeric": 1.0 if len(ctx.numeric_cols) >= 5 else 0.0,
            "list": 1.0 if ctx.list_cols else 0.0,
            "text": 1.0 if ctx.text_cols else 0.0,
            "time": 1.0 if ctx.time_col else 0.0,
            "time_target": 1.0 if (ctx.time_col and ctx.target_col) else 0.0,
            "target_classes": 1.0 if ctx.target_col else 0.0,
            "aux": 1.0 if ctx.aux_dfs else 0.0,
            "id": 1.0 if ctx.id_col else 0.0,
            "sentinel": 1.0 if ctx.sentinel_rules else 0.0,
            "imbalance": 1.0 if (ctx.task_type == "tabular" and ctx.target_col) else 0.0,
            "numeric_pairs": 1.0 if len(ctx.numeric_cols) >= 2 else 0.0,
            "outlier": 1.0 if ctx.numeric_cols else 0.0,
            "skew": 0.5 if ctx.numeric_cols else 0.0,
            "int_date": 0.0,
            "const": 1.0,
        }

    def _init_priors(self) -> None:
        signals = self._ctx_signals()
        for name, spec in CATALOG.items():
            # Mandatory ops -> always 1.0
            if spec.mandatory:
                self.logp[name] = 30.0
                continue
            # Drop ops that do not fit the task type at all -> very low prior.
            if spec.task_type not in (self.ctx.task_type, "both"):
                self.logp[name] = -10.0
                continue
            # Compute log-prior from prior_features dot signals
            base = -0.5  # default mild rejection
            for key, weight in spec.prior_features.items():
                base += weight * signals.get(key, 0.0)
            self.logp[name] = base

    # ------------------------------------------------------------------
    def prob(self, op_name: str) -> float:
        spec = CATALOG[op_name]
        if spec.mandatory:
            return 1.0
        return _sigmoid(self.logp[op_name])

    def update(self, op_name: str, delta: float) -> None:
        if op_name not in self.logp:
            return
        spec = CATALOG[op_name]
        if spec.mandatory:
            return
        self.logp[op_name] += self.eta * delta

    def snapshot(self) -> dict[str, float]:
        return {n: self.prob(n) for n in CATALOG}


# ---------------------------------------------------------------------------
# Pairwise join model M_J (schema-driven)
# ---------------------------------------------------------------------------
@dataclass
class JoinEdge:
    """One join edge in the global graph."""

    name: str            # human-friendly label
    op_name: str         # which operator implements this join
    target: str          # target slot ("interaction" / "both")
    aux_ref: Optional[str] = None  # name of side table referenced via $name


class JoinModel:
    """Schema-driven join probability model.

    For rec tasks the join is ``JoinTable`` against the user/item/context side
    tables, and is treated as mandatory (probability 1.0).

    For tabular tasks, every aux dataframe contributes one ``JoinTable`` and
    one ``JoinTable`` candidate (probabilistic, initial 0.5 floor).
    """

    def __init__(self, ctx: DataContext, eta: float = 0.5):
        self.ctx = ctx
        self.eta = float(eta)
        self.edges: list[JoinEdge] = []
        self.logp: dict[str, float] = {}
        self.mandatory_keys: set[str] = set()
        self._build_edges()

    # ------------------------------------------------------------------
    def _build_edges(self) -> None:
        ctx = self.ctx
        if ctx.task_type == "rec":
            if ctx.has_user_df or ctx.has_item_df:
                edge = JoinEdge(
                    name="JoinTable(user+item)",
                    op_name="JoinTable", target="interaction",
                )
                self.edges.append(edge)
                self.logp[edge.name] = 30.0  # mandatory -> sigmoid≈1
                self.mandatory_keys.add(edge.name)
        else:
            for aux in ctx.aux_dfs:
                e1 = JoinEdge(name=f"JoinTable({aux})", op_name="JoinTable",
                              target="both", aux_ref=aux)
                e2 = JoinEdge(name=f"JoinTable({aux})", op_name="JoinTable",
                              target="both", aux_ref=aux)
                self.edges.append(e1)
                self.edges.append(e2)
                # initial prob 0.5 -> logit 0
                self.logp[e1.name] = 0.0
                self.logp[e2.name] = 0.0

    # ------------------------------------------------------------------
    def prob(self, edge_name: str) -> float:
        if edge_name in self.mandatory_keys:
            return 1.0
        # paper: p = max(p_tilde, 0.5)
        return max(0.5, _sigmoid(self.logp.get(edge_name, 0.0)))

    def update(self, edge_name: str, delta: float) -> None:
        if edge_name in self.mandatory_keys or edge_name not in self.logp:
            return
        self.logp[edge_name] += self.eta * delta

    def all_edges(self) -> list[JoinEdge]:
        return list(self.edges)

    def mandatory_edges(self) -> list[JoinEdge]:
        return [e for e in self.edges if e.name in self.mandatory_keys]

    def optional_edges(self) -> list[JoinEdge]:
        return [e for e in self.edges if e.name not in self.mandatory_keys]

    def snapshot(self) -> dict[str, float]:
        return {e.name: self.prob(e.name) for e in self.edges}


__all__ = ["TransformationModel", "JoinModel", "JoinEdge"]
