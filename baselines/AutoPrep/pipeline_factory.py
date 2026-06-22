"""Context-aware default-parameter factory for Auto-Prep, covering all 59
dppbench operators.

For each operator we either:
  * fill the default params with column names / aux dataframe references
    inferred from ``ctx``; or
  * return ``None`` to signal the operator is **not applicable** to the
    current context (e.g. ``ReduceDimension`` without a target column).

Pipeline / PipelineStep / DataContext are reused from ``baselines.SAGA.pipeline``
(they are pure data structures and do not depend on the SAGA catalog).
"""
from __future__ import annotations

import copy
import random as _random
from typing import Optional

from baselines.SAGA.pipeline import DataContext, Pipeline, PipelineStep

from .operator_catalog import CATALOG, OpCategory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _exclude_target(cols: list[str], ctx: DataContext) -> list[str]:
    bad = {ctx.target_col, ctx.id_col, ctx.user_col, ctx.item_col, ctx.time_col}
    return [c for c in cols if c not in bad]


def _pick_some(cols: list[str], rng: _random.Random, lo: int = 1, hi: int = 3) -> list[str]:
    if not cols:
        return []
    n = min(len(cols), rng.randint(lo, hi))
    return cols[:n]


def default_target_for(op_name: str, task_type: str) -> str:
    spec = CATALOG[op_name]
    if spec.valid_targets == ("interaction",):
        return "interaction"
    if spec.valid_targets == ("train",):
        return "train"
    if task_type == "tabular":
        return "both"
    return "interaction"


# ---------------------------------------------------------------------------
# Per-operator parameter builders
# ---------------------------------------------------------------------------
def build_default_params(
    op_name: str,
    ctx: DataContext,
    rng: _random.Random,
) -> Optional[dict]:
    """Return a fully populated param dict, or ``None`` if the operator is
    inapplicable in the given ``ctx``.
    """
    if op_name not in CATALOG:
        return None
    spec = CATALOG[op_name]
    p = copy.deepcopy(spec.default_params)
    numeric_cols = _exclude_target(ctx.numeric_cols, ctx)
    categorical_cols = _exclude_target(ctx.categorical_cols, ctx)
    text_cols = list(ctx.text_cols)

    # ------------------------------------------------------------- Cleaning
    if op_name == "HandleMV":
        return p
    if op_name == "HandleMV":
        if not numeric_cols:
            return None
        p["cols"] = _pick_some(numeric_cols, rng, 1, 5)
        return p
    if op_name == "HandleMV":
        if not numeric_cols:
            return None
        p["cols"] = _pick_some(numeric_cols, rng, 1, 5)
        return p
    if op_name == "Deduplicate":
        return p
    if op_name == "HandleError":
        if not numeric_cols:
            return None
        p["cols"] = _pick_some(numeric_cols, rng, 1, 2)
        p["rule"] = "numeric"
        p["action"] = "delete"
        return p
    if op_name == "HandleOutlier":
        if not numeric_cols:
            return None
        p["cols"] = _pick_some(numeric_cols, rng, 1, 3)
        p["action"] = "delete"
        return p
    if op_name == "HandleNonIID":
        cols = list(numeric_cols[:3])
        if cols:
            p["feature_cols"] = cols
        p["action"] = "reweight"
        return p
    if op_name == "ReweightUPG":
        cols = list(numeric_cols[:2])
        if cols:
            p["feature_cols"] = cols
        return p
    if op_name == "CustomClean":
        if not text_cols:
            return None
        p["cols"] = _pick_some(text_cols, rng, 1, 1)
        p["pattern"] = r"\s+"
        p["replacement"] = " "
        p["regex"] = True
        return p
    if op_name == "CustomClean":
        if not ctx.sentinel_rules:
            return None
        converted = []
        for r in ctx.sentinel_rules:
            new_r = {k: v for k, v in r.items() if k != "value"}
            new_r["eq"] = r["value"]
            converted.append(new_r)
        p["rules"] = converted
        return p
    if op_name == "ParseDate":
        if not ctx.time_col:
            return None
        p["cols"] = [ctx.time_col]
        return p
    if op_name == "ParseDate":
        # Only useful if the dataset is known to contain YYMMDD ints; we
        # signal not-applicable by default (ctx does not encode int_date cols).
        return None

    # ------------------------------------------------------------ Integration
    if op_name == "JoinTable":
        if not ctx.aux_dfs or ctx.id_col is None:
            return None
        aux = rng.choice(ctx.aux_dfs)
        p["aux_df"] = f"${aux}"
        p["key_col"] = ctx.id_col
        p["prefix"] = aux.upper()[:8]
        p["max_cols"] = 20
        return p
    if op_name == "JoinTable":
        if not ctx.aux_dfs or ctx.id_col is None:
            return None
        aux = rng.choice(ctx.aux_dfs)
        p["aux_df"] = f"${aux}"
        p["key_col"] = ctx.id_col
        return p
    if op_name == "ConcatTable":
        if not ctx.aux_dfs:
            return None
        p["other_dfs"] = [f"${ctx.aux_dfs[0]}"]
        return p
    if op_name == "CrossFeature":
        cols = _exclude_target(ctx.categorical_cols, ctx)
        if len(cols) < 2:
            return None
        p["cols"] = cols[:2]
        p["output_col"] = "_".join(cols[:2]) + "_combo"
        return p
    if op_name == "JoinTable":
        if not (ctx.has_user_df or ctx.has_item_df):
            return None
        p["user_col"] = ctx.user_col
        p["item_col"] = ctx.item_col
        p["user_df"] = "$user_df" if ctx.has_user_df else None
        p["item_df"] = "$item_df" if ctx.has_item_df else None
        p["how"] = "left"
        return p

    # -------------------------------------------------- Preprocessing-Schema
    if op_name == "CastType":
        if not numeric_cols:
            return None
        p["col_dtypes"] = {numeric_cols[0]: "float"}
        return p
    if op_name == "RenameColumn":
        # No safe default -- skip.
        return None
    if op_name == "CustomProcess":
        cols = []
        if ctx.id_col and ctx.task_type == "tabular":
            cols.append(ctx.id_col)
        if not cols:
            return None
        p["cols"] = cols
        return p
    if op_name == "CustomProcess":
        return p
    if op_name == "SelectFeature":
        p["target_col"] = ctx.target_col
        return p

    # ------------------------------------------------- Preprocessing-Scaling
    if op_name in ("ScaleFeature", "ScaleFeature", "ScaleFeature"):
        # Never auto-pick target / id / user / item / time columns: scaling
        # them would corrupt downstream ops.
        if not numeric_cols:
            return None
        p["cols"] = list(numeric_cols)
        p["auto_numeric"] = False
        return p
    if op_name == "ScaleFeature":
        if not numeric_cols:
            return None
        p["cols"] = _pick_some(numeric_cols, rng, 1, 5)
        return p
    if op_name == "TransformPower":
        if not numeric_cols:
            return None
        p["cols"] = _pick_some(numeric_cols, rng, 1, 5)
        return p
    if op_name == "TransformPower":
        if not numeric_cols:
            return None
        p["cols"] = _pick_some(numeric_cols, rng, 1, 5)
        return p

    # ------------------------------------------------ Preprocessing-Encoding
    if op_name == "OneHotEncode":
        if not categorical_cols:
            return None
        p["cols"] = _pick_some(categorical_cols, rng, 1, 3)
        return p
    if op_name == "OrdinalEncode":
        # Needs explicit ordering -- skip.
        return None
    if op_name == "LabelEncode":
        return p
    if op_name == "CustomProcess":
        if not categorical_cols:
            return None
        p["cols"] = _pick_some(categorical_cols, rng, 1, 5)
        return p
    if op_name == "HashEncode":
        if not categorical_cols:
            return None
        p["cols"] = _pick_some(categorical_cols, rng, 1, 3)
        return p
    if op_name == "TargetEncode":
        if not ctx.target_col or not categorical_cols:
            return None
        p["cols"] = _pick_some(categorical_cols, rng, 1, 3)
        p["target_col"] = ctx.target_col
        return p
    if op_name == "DiscretizeFeature":
        if not numeric_cols:
            return None
        col = numeric_cols[0]
        p["boundaries"] = {col: [10, 50, 100]}
        return p

    # -------------------------------- Preprocessing-Imbalance / Augmentation
    if op_name == "Oversample":
        if ctx.task_type != "tabular" or not ctx.target_col:
            return None
        p["target_col"] = ctx.target_col
        return p
    if op_name == "Undersample":
        if ctx.task_type != "tabular" or not ctx.target_col:
            return None
        p["target_col"] = ctx.target_col
        return p
    if op_name == "Undersample":
        if ctx.task_type != "tabular" or not ctx.target_col:
            return None
        p["target_col"] = ctx.target_col
        return p
    if op_name == "Oversample":
        if ctx.task_type != "tabular" or not ctx.target_col:
            return None
        p["target_col"] = ctx.target_col
        return p
    if op_name == "AugmentNoise":
        if not numeric_cols:
            return None
        p["cols"] = _pick_some(numeric_cols, rng, 1, 3)
        return p

    # -------------------------------------------------------- FE-Generation
    if op_name == "CreateFeature":
        if len(numeric_cols) < 2:
            return None
        p["source_cols"] = numeric_cols[:2]
        p["output_col"] = f"{numeric_cols[0]}_mean"
        p["method"] = "mean"
        return p
    if op_name == "TransformPower":
        if not numeric_cols:
            return None
        p["cols"] = _pick_some(numeric_cols, rng, 1, 3)
        return p
    if op_name == "CreatePolynomialFeature":
        if len(numeric_cols) < 2:
            return None
        p["cols"] = numeric_cols[:5]
        return p
    if op_name == "ExtractDateTimeFeature":
        if not ctx.time_col:
            return None
        p["cols"] = [ctx.time_col]
        return p

    # -------------------------------------------------------- FE-TimeSeries
    if op_name == "CreateLagFeature":
        if not ctx.time_col or not ctx.target_col:
            return None
        p["target_col"] = ctx.target_col
        p["lags"] = [1, 2, 3]
        p["time_col"] = ctx.time_col
        return p
    if op_name == "CreateRollingFeature":
        if not ctx.time_col or not ctx.target_col:
            return None
        p["target_col"] = ctx.target_col
        p["windows"] = [3, 7]
        p["aggs"] = ["mean", "std"]
        p["time_col"] = ctx.time_col
        return p
    if op_name == "ResampleTimeSeries":
        if not ctx.time_col or not numeric_cols:
            return None
        p["time_col"] = ctx.time_col
        p["freq"] = "D"
        p["aggs"] = {numeric_cols[0]: ["mean"]}
        return p

    # ------------------------------------------------------ FE-Selection
    if op_name == "SelectFeature":
        if not ctx.target_col:
            return None
        p["target_col"] = ctx.target_col
        p["k"] = max(5, min(50, len(numeric_cols)))
        return p
    if op_name == "SelectFeature":
        if not ctx.target_col or len(numeric_cols) < 5:
            return None
        p["target_col"] = ctx.target_col
        p["n_features_to_select"] = max(5, min(20, len(numeric_cols) // 2))
        return p

    # ------------------------------------------------------ FE-Reduction
    if op_name == "ReduceDimension":
        if len(numeric_cols) < 4:
            return None
        n_comp = min(8, max(2, len(numeric_cols) // 2))
        p["cols"] = numeric_cols
        p["n_components"] = n_comp
        return p
    if op_name == "ReduceDimension":
        if len(numeric_cols) < 4:
            return None
        p["cols"] = numeric_cols
        p["n_components"] = min(4, max(2, len(numeric_cols) // 3))
        return p
    if op_name == "ReduceDimension":
        if not ctx.target_col or len(numeric_cols) < 2:
            return None
        p["cols"] = numeric_cols
        p["target_col"] = ctx.target_col
        return p
    if op_name == "ReduceDimension":
        if len(numeric_cols) < 4:
            return None
        p["cols"] = numeric_cols
        return p

    # ------------------------------------------------ Reshape / Sort / String
    if op_name == "SortRows":
        if ctx.time_col:
            p["by"] = [ctx.time_col]
            return p
        return None

    # -------------------------------------------------------- Recommendation
    if op_name == "FilterKCore":
        p["user_col"] = ctx.user_col
        p["item_col"] = ctx.item_col
        return p
    if op_name == "CreateSequence":
        if ctx.task_type != "rec" or not ctx.time_col:
            return None
        p["user_col"] = ctx.user_col
        p["item_col"] = ctx.item_col
        p["time_col"] = ctx.time_col
        return p
    if op_name == "FilterSample":
        if ctx.target_col:
            p["subset"] = [ctx.target_col]
        return p
    if op_name == "SampleNegative":
        if ctx.task_type != "rec" or not ctx.target_col:
            return None
        p["user_col"] = ctx.user_col
        p["item_col"] = ctx.item_col
        p["target_col"] = ctx.target_col
        return p

    return p


def make_step(
    op_name: str,
    ctx: DataContext,
    rng: _random.Random,
) -> Optional[PipelineStep]:
    params = build_default_params(op_name, ctx, rng)
    if params is None:
        return None
    return PipelineStep(
        op=op_name,
        target=default_target_for(op_name, ctx.task_type),
        params=params,
    )


__all__ = [
    "build_default_params", "make_step", "default_target_for",
    "Pipeline", "PipelineStep", "DataContext",
]
