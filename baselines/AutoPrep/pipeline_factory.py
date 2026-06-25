"""Context-aware default-parameter factory for Auto-Prep, covering all
dppbench operators.

For each operator we either:
  * fill the default params with column names / aux dataframe references
    inferred from ``ctx`` (with stochastic variation across calls so that
    the beam search explores multiple parameter configurations); or
  * return ``None`` to signal the operator is **not applicable** to the
    current context (e.g. ``ReduceDimension`` without enough numeric columns).

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

    Each call uses ``rng`` to choose among the operator's valid parameter
    variants (e.g. scaling method, imputation strategy, aux-table join mode),
    which gives the beam search diversity without requiring duplicate op names.
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
        # Stochastically choose among: global median/mean (default), cols+MICE, cols+KNN
        variant = rng.choice(["global", "cols_mice", "cols_knn"])
        if variant == "global":
            methods = ["median", "mean", "mode", "constant"]
            p["method"] = rng.choice(methods)
            p["action"] = "impute"
            return p
        else:
            if not numeric_cols:
                return None
            p["cols"] = _pick_some(numeric_cols, rng, 1, 5)
            p["method"] = "iterative" if variant == "cols_mice" else "knn"
            p["action"] = "impute"
            return p

    if op_name == "Deduplicate":
        return p

    if op_name == "HandleError":
        if not numeric_cols:
            return None
        p["cols"] = _pick_some(numeric_cols, rng, 1, 2)
        p["rule"] = rng.choice(["numeric", "positive"])
        p["action"] = "delete"
        return p

    if op_name == "HandleOutlier":
        if not numeric_cols:
            return None
        p["cols"] = _pick_some(numeric_cols, rng, 1, 3)
        p["method"] = rng.choice(["iqr", "zscore"])
        p["action"] = rng.choice(["delete", "repair"])
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
        # Stochastically choose between sentinel-rule cleaning and regex cleaning
        if ctx.sentinel_rules and (not text_cols or rng.random() < 0.6):
            converted = []
            for r in ctx.sentinel_rules:
                new_r = {k: v for k, v in r.items() if k != "value"}
                new_r["eq"] = r["value"]
                converted.append(new_r)
            p["rules"] = converted
            return p
        else:
            if not text_cols:
                return None
            p["cols"] = _pick_some(text_cols, rng, 1, 1)
            p["pattern"] = r"\s+"
            p["replacement"] = " "
            p["regex"] = True
            return p

    if op_name == "CorrectLabel":
        if not ctx.target_col:
            return None
        p["label_col"] = ctx.target_col
        p["strategy"] = "flag"
        p["confidence_threshold"] = rng.choice([0.8, 0.9, 0.95])
        return p

    if op_name == "CorrectTypo":
        if not text_cols:
            return None
        p["cols"] = _pick_some(text_cols, rng, 1, 2)
        return p

    if op_name == "ParseDate":
        if ctx.time_col:
            p["cols"] = [ctx.time_col]
            p["mode"] = "string"
            return p
        return None

    if op_name == "ParseNumber":
        parse_cols = _pick_some(text_cols + categorical_cols, rng, 1, 2)
        if not parse_cols:
            return None
        p["cols"] = parse_cols
        return p

    # ------------------------------------------------------------ Integration
    if op_name == "JoinTable":
        if ctx.task_type == "rec":
            if not (ctx.has_user_df or ctx.has_item_df):
                return None
            p["user_col"] = ctx.user_col
            p["item_col"] = ctx.item_col
            p["user_df"] = "$user_df" if ctx.has_user_df else None
            p["item_df"] = "$item_df" if ctx.has_item_df else None
            p["how"] = "left"
            p["method"] = "rec"
            return p
        else:
            # tabular: pick an aux table; stochastically choose with/without prefix
            if not ctx.aux_dfs or ctx.id_col is None:
                return None
            aux = rng.choice(ctx.aux_dfs)
            p["aux_df"] = f"${aux}"
            p["key_col"] = ctx.id_col
            p["method"] = "key"
            if rng.random() < 0.5:
                p["prefix"] = aux.upper()[:8]
                p["max_cols"] = 20
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
        picked = _pick_some(cols, rng, 2, 2)
        p["cols"] = picked
        p["output_col"] = "_".join(picked) + "_combo"
        return p

    if op_name == "SplitColumn":
        if not text_cols:
            return None
        p["col"] = text_cols[0]
        p["sep"] = r"\s+"
        p["regex"] = True
        return p

    # -------------------------------------------------- Preprocessing-Schema
    if op_name == "CastType":
        if not numeric_cols:
            return None
        p["col_dtypes"] = {numeric_cols[0]: "float"}
        return p

    if op_name == "RenameColumn":
        return None

    if op_name == "CustomProcess":
        # Stochastically choose among: drop id, drop high-null, frequency encode,
        # passthrough (default empty params)
        variant = rng.choice(["drop_id", "drop_null", "freq_encode", "passthrough"])
        if variant == "drop_id":
            cols = []
            if ctx.id_col and ctx.task_type == "tabular":
                cols.append(ctx.id_col)
            if not cols:
                return None
            p["cols"] = cols
            p["mode"] = "drop_columns"
            return p
        elif variant == "drop_null":
            p["threshold"] = rng.choice([0.7, 0.8, 0.9])
            p["mode"] = "drop_high_null"
            return p
        elif variant == "freq_encode":
            if not categorical_cols:
                return None
            p["cols"] = _pick_some(categorical_cols, rng, 1, 5)
            p["mode"] = "frequency_encode"
            return p
        else:
            return p

    if op_name == "CustomTransform":
        return p

    if op_name == "AlignSchema":
        return p

    # ------------------------------------------------- Preprocessing-Scaling
    if op_name == "ScaleFeature":
        if not numeric_cols:
            return None
        p["method"] = rng.choice(["standard", "minmax", "maxabs", "robust", "l2"])
        if p["method"] == "standard" and rng.random() < 0.3:
            p["cols"] = list(numeric_cols)
            p["auto_numeric"] = False
        else:
            p["cols"] = _pick_some(numeric_cols, rng, 1, 5)
        return p

    if op_name == "TransformPower":
        if not numeric_cols:
            return None
        p["cols"] = _pick_some(numeric_cols, rng, 1, 5)
        p["method"] = rng.choice(["log", "sqrt", "quantile"])
        return p

    if op_name == "ClipOutlier":
        if not numeric_cols:
            return None
        p["cols"] = _pick_some(numeric_cols, rng, 1, 3)
        return p

    # ------------------------------------------------ Preprocessing-Encoding
    if op_name == "OneHotEncode":
        if not categorical_cols:
            return None
        p["cols"] = _pick_some(categorical_cols, rng, 1, 3)
        return p

    if op_name == "OrdinalEncode":
        return None

    if op_name == "LabelEncode":
        return p

    if op_name == "HashEncode":
        if not categorical_cols:
            return None
        p["cols"] = _pick_some(categorical_cols, rng, 1, 3)
        p["n_buckets"] = rng.choice([1024, 4096, 10000])
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
        strategy = rng.choice(["uniform", "quantile", "kmeans", "manual"])
        p["strategy"] = strategy
        if strategy == "manual":
            p["boundaries"] = {col: [10, 50, 100]}
        else:
            p["cols"] = [col]
            p["n_bins"] = rng.choice([5, 10, 20])
        return p

    # -------------------------------- Preprocessing-Imbalance / Augmentation
    if op_name == "Oversample":
        if ctx.task_type != "tabular" or not ctx.target_col:
            return None
        p["target_col"] = ctx.target_col
        p["method"] = rng.choice(["random", "smote", "adasyn"])
        return p

    if op_name == "Undersample":
        if ctx.task_type != "tabular" or not ctx.target_col:
            return None
        p["target_col"] = ctx.target_col
        p["method"] = rng.choice(["random", "tomek", "enn"])
        return p

    if op_name == "AugmentNoise":
        if not numeric_cols:
            return None
        p["cols"] = _pick_some(numeric_cols, rng, 1, 3)
        return p

    if op_name == "AugmentMixup":
        if ctx.task_type != "tabular" or not ctx.target_col or not numeric_cols:
            return None
        p["label_col"] = ctx.target_col
        p["cols"] = _pick_some(numeric_cols, rng, 1, 5)
        return p

    # -------------------------------------------------------- FE-Generation
    if op_name == "CreateFeature":
        if len(numeric_cols) < 2:
            return None
        p["source_cols"] = numeric_cols[:2]
        method = rng.choice(["mean", "sum", "std", "min", "max", "ratio", "diff"])
        p["method"] = method
        suffix = method if method != "ratio" else "ratio"
        p["output_col"] = f"{numeric_cols[0]}_{suffix}"
        return p

    if op_name == "CreatePolynomialFeature":
        if len(numeric_cols) < 2:
            return None
        p["cols"] = numeric_cols[:5]
        p["degree"] = rng.choice([2, 3])
        return p

    if op_name == "ExtractDateTimeFeature":
        if not ctx.time_col:
            return None
        p["cols"] = [ctx.time_col]
        p["features"] = ["day_of_week", "hour_of_day"]
        p["drop_original"] = rng.choice([True, False])
        return p

    if op_name == "AggregateGroupFeature":
        if not categorical_cols or not numeric_cols:
            return None
        p["group_col"] = categorical_cols[0]
        p["agg_cols"] = _pick_some(numeric_cols, rng, 1, 3)
        p["aggs"] = ["mean", "std"]
        return p

    if op_name == "ExtractTextFeature":
        if not text_cols:
            return None
        p["cols"] = _pick_some(text_cols, rng, 1, 2)
        p["method"] = "tfidf"
        p["max_features"] = 100
        return p

    if op_name == "ExtractTextEmbedding":
        if not text_cols:
            return None
        p["cols"] = _pick_some(text_cols, rng, 1, 1)
        return p

    if op_name == "ExtractGraphFeature":
        return p

    if op_name == "CustomFE":
        return p

    # -------------------------------------------------------- FE-TimeSeries
    if op_name == "CreateLagFeature":
        if not ctx.time_col or not ctx.target_col:
            return None
        p["target_col"] = ctx.target_col
        p["lags"] = rng.choice([[1], [1, 2], [1, 2, 3]])
        p["time_col"] = ctx.time_col
        return p

    if op_name == "CreateRollingFeature":
        if not ctx.time_col or not ctx.target_col:
            return None
        p["target_col"] = ctx.target_col
        p["windows"] = rng.choice([[3], [3, 7], [7, 14]])
        p["aggs"] = ["mean", "std"]
        p["time_col"] = ctx.time_col
        return p

    if op_name == "ResampleTimeSeries":
        if not ctx.time_col or not numeric_cols:
            return None
        p["time_col"] = ctx.time_col
        p["freq"] = rng.choice(["D", "W", "M"])
        p["aggs"] = {numeric_cols[0]: ["mean"]}
        return p

    # ------------------------------------------------------ FE-Selection
    if op_name == "SelectFeature":
        if not ctx.target_col:
            return None
        method = rng.choice(["variance", "univariate", "model"])
        p["target_col"] = ctx.target_col
        p["method"] = method
        if method == "variance":
            p["threshold"] = 0.0
        elif method == "univariate":
            if len(numeric_cols) < 5:
                p["k"] = max(3, len(numeric_cols))
            else:
                p["k"] = max(5, min(50, len(numeric_cols)))
        else:
            p["n_features_to_select"] = max(5, min(20, max(2, len(numeric_cols) // 2)))
        return p

    # ------------------------------------------------------ FE-Reduction
    if op_name == "ReduceDimension":
        method = rng.choice(["pca", "kernel_pca", "lda", "umap"])
        if method == "lda":
            if not ctx.target_col or len(numeric_cols) < 2:
                return None
            p["cols"] = numeric_cols
            p["target_col"] = ctx.target_col
            p["n_components"] = min(4, max(2, len(numeric_cols) // 3))
        else:
            if len(numeric_cols) < 4:
                return None
            p["cols"] = numeric_cols
            if method == "pca":
                p["n_components"] = min(8, max(2, len(numeric_cols) // 2))
            elif method == "kernel_pca":
                p["n_components"] = min(4, max(2, len(numeric_cols) // 3))
            else:
                p["n_components"] = min(4, max(2, len(numeric_cols) // 3))
        p["method"] = method
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
        p["k"] = rng.choice([3, 5, 10])
        return p

    if op_name == "CreateSequence":
        if ctx.task_type != "rec" or not ctx.time_col:
            return None
        p["user_col"] = ctx.user_col
        p["item_col"] = ctx.item_col
        p["time_col"] = ctx.time_col
        p["seq_col"] = "item_id_seq"
        p["max_len"] = rng.choice([10, 20, 50])
        return p

    if op_name == "TruncateSequence":
        p["max_len"] = rng.choice([10, 20])
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
        p["n_negatives"] = rng.choice([1, 2, 3])
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
