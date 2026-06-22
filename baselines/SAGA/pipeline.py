"""Pipeline representation, serialization, and random factory for SAGA.

A ``Pipeline`` is a list of ``PipelineStep`` objects. Each step corresponds to
one entry in the YAML format consumed by
``dppbench.dataset.{TabularData,RecData}.run_pre_process``.
"""
from __future__ import annotations

import copy
import hashlib
import random as _random
from dataclasses import dataclass, field, asdict
from typing import Any

import yaml

from .operator_catalog import CATALOG, OpCategory, OpSpec, operators_for_task


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------
@dataclass
class PipelineStep:
    op: str
    target: str = "both"
    params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {"op": self.op, "target": self.target}
        if self.params:
            d["params"] = copy.deepcopy(self.params)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineStep":
        return cls(
            op=d["op"],
            target=d.get("target", "both"),
            params=copy.deepcopy(d.get("params", {})) or {},
        )


@dataclass
class Pipeline:
    steps: list[PipelineStep] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_yaml(self) -> str:
        body = {"pipeline": [s.to_dict() for s in self.steps]}
        return yaml.safe_dump(body, sort_keys=False, default_flow_style=False)

    @classmethod
    def from_yaml(cls, text: str) -> "Pipeline":
        data = yaml.safe_load(text) or {}
        steps_raw = data.get("pipeline", []) or []
        return cls(steps=[PipelineStep.from_dict(s) for s in steps_raw])

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def copy(self) -> "Pipeline":
        return Pipeline(steps=[PipelineStep.from_dict(s.to_dict()) for s in self.steps])

    def hash(self) -> str:
        return hashlib.sha1(self.to_yaml().encode("utf-8")).hexdigest()

    def op_names(self) -> list[str]:
        return [s.op for s in self.steps]

    def __len__(self) -> int:
        return len(self.steps)


# -----------------------------------------------------------------------------
# Schema / context used to construct context-dependent operator parameters
# -----------------------------------------------------------------------------
@dataclass
class DataContext:
    task_type: str
    data_name: str
    numeric_cols: list[str] = field(default_factory=list)
    categorical_cols: list[str] = field(default_factory=list)
    list_cols: list[str] = field(default_factory=list)
    text_cols: list[str] = field(default_factory=list)
    target_col: str | None = None
    id_col: str | None = None
    time_col: str | None = None
    user_col: str = "user_id"
    item_col: str = "item_id"
    has_user_df: bool = False
    has_item_df: bool = False
    aux_dfs: list[str] = field(default_factory=list)  # names available as $name
    sentinel_rules: list[dict] = field(default_factory=list)  # for CustomClean

    @property
    def all_cols(self) -> list[str]:
        return self.numeric_cols + self.categorical_cols + self.list_cols + self.text_cols


# -----------------------------------------------------------------------------
# Default-params builder for individual operators (context aware)
# -----------------------------------------------------------------------------
def _exclude_special(cols: list[str], ctx: DataContext) -> list[str]:
    bad = {ctx.target_col, ctx.id_col, ctx.user_col, ctx.item_col, ctx.time_col}
    return [c for c in cols if c not in bad]


def _pick_some(cols: list[str], rng: _random.Random, lo: int = 1, hi: int = 3) -> list[str]:
    if not cols:
        return []
    n = min(len(cols), rng.randint(lo, hi))
    return cols[:n]


def build_default_params(op_name: str, ctx: DataContext, rng: _random.Random) -> dict | None:
    """Return a dict of default params for ``op_name``, or ``None`` if the
    operator cannot be applied given the current context (e.g. no time column
    for ``ExtractDateTimeFeature``).

    Covers all 59 dppbench operators registered in :data:`CATALOG`.
    """
    if op_name not in CATALOG:
        return None
    spec = CATALOG[op_name]
    p = copy.deepcopy(spec.default_params)

    numeric_cols = _exclude_special(ctx.numeric_cols, ctx)
    categorical_cols = _exclude_special(ctx.categorical_cols, ctx)
    text_cols = list(ctx.text_cols)

    # ---- JOIN ----
    if op_name == "JoinTable":
        if not (ctx.has_user_df or ctx.has_item_df):
            return None
        p["user_col"] = ctx.user_col
        p["item_col"] = ctx.item_col
        p["user_df"] = "$user_df" if ctx.has_user_df else None
        p["item_df"] = "$item_df" if ctx.has_item_df else None
        p["how"] = "left"
        return p
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
        cols = _exclude_special(ctx.categorical_cols, ctx)
        if len(cols) < 2:
            return None
        p["cols"] = cols[:2]
        p["output_col"] = "_".join(cols[:2]) + "_combo"
        return p

    # ---- FILTER_ROW ----
    if op_name == "FilterSample":
        if ctx.task_type == "rec" and ctx.target_col:
            p["subset"] = [ctx.target_col]
        return p
    if op_name == "Deduplicate":
        return p
    if op_name == "FilterKCore":
        p["user_col"] = ctx.user_col
        p["item_col"] = ctx.item_col
        return p

    # ---- SCHEMA / FILTER_COL ----
    if op_name == "CastType":
        if not numeric_cols:
            return None
        p["col_dtypes"] = {numeric_cols[0]: "float"}
        return p
    if op_name == "RenameColumn":
        return None
    if op_name == "CustomProcess":
        cols = []
        if ctx.id_col:
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

    # ---- CLEAN_VALUE ----
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
    if op_name == "CustomClean":
        if not text_cols:
            return None
        p["cols"] = _pick_some(text_cols, rng, 1, 1)
        p["pattern"] = r"\s+"
        p["replacement"] = " "
        p["regex"] = True
        return p

    # ---- ERROR_DETECTION ----
    if op_name == "HandleError":
        if not numeric_cols:
            return None
        p["cols"] = _pick_some(numeric_cols, rng, 1, 2)
        p["rule"] = "numeric"
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

    # ---- DATETIME_PARSE ----
    if op_name == "ParseDate":
        if not ctx.time_col:
            return None
        p["cols"] = [ctx.time_col]
        return p
    if op_name == "ParseDate":
        return None

    # ---- OUTLIER ----
    if op_name == "HandleOutlier":
        if not numeric_cols:
            return None
        p["cols"] = _pick_some(numeric_cols, rng, 1, 3)
        p["action"] = "delete"
        return p

    # ---- MISSING_VALUE ----
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

    # ---- DISCRETIZATION ----
    if op_name == "DiscretizeFeature":
        if not numeric_cols:
            return None
        col = numeric_cols[0]
        p["boundaries"] = {col: [10, 50, 100]}
        return p

    # ---- ENCODING ----
    if op_name == "OneHotEncode":
        if not categorical_cols:
            return None
        p["cols"] = _pick_some(categorical_cols, rng, 1, 3)
        return p
    if op_name == "OrdinalEncode":
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

    # ---- SCALING / DISTRIBUTION_RESHAPE ----
    if op_name in ("ScaleFeature", "ScaleFeature", "ScaleFeature"):
        # Never auto-pick target / id / user / item / time columns: scaling
        # them would corrupt downstream ops (e.g. embedding lookup on a
        # scaled user_id).
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

    # ---- IMBALANCE / AUGMENT ----
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

    # ---- NORMALIZATION ----
    if op_name == "TransformPower":
        if not numeric_cols:
            return None
        p["cols"] = _pick_some(numeric_cols, rng, 1, 3)
        return p

    # ---- FEATURE_GEN ----
    if op_name == "CreateFeature":
        if len(numeric_cols) < 2:
            return None
        p["source_cols"] = numeric_cols[:2]
        p["output_col"] = f"{numeric_cols[0]}_mean"
        p["method"] = "mean"
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

    # ---- FE-TimeSeries ----
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

    # ---- FE-Selection ----
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

    # ---- FE-Reduction ----
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

    # ---- Reshape / Sort / String ----
    if op_name == "SortRows":
        if ctx.time_col:
            p["by"] = [ctx.time_col]
            return p
        return None

    # ---- SEQUENCE ----
    if op_name == "CreateSequence":
        if ctx.task_type != "rec" or not ctx.time_col:
            return None
        p["user_col"] = ctx.user_col
        p["item_col"] = ctx.item_col
        p["time_col"] = ctx.time_col
        p["seq_col"] = "item_id_seq"
        p["max_len"] = 20
        return p

    # ---- SAMPLING ----
    if op_name == "SampleNegative":
        if ctx.task_type != "rec" or not ctx.target_col:
            return None
        p["user_col"] = ctx.user_col
        p["item_col"] = ctx.item_col
        p["target_col"] = ctx.target_col
        return p

    return p


def default_target_for(op_name: str, task_type: str) -> str:
    spec = CATALOG[op_name]
    if spec.valid_targets == ("interaction",):
        return "interaction"
    if task_type == "tabular":
        return "both"
    return "interaction"


def make_step(op_name: str, ctx: DataContext, rng: _random.Random) -> PipelineStep | None:
    """Construct a PipelineStep with sensible default params, or ``None`` if the
    operator is not applicable in the given context."""
    params = build_default_params(op_name, ctx, rng)
    if params is None:
        return None
    return PipelineStep(
        op=op_name,
        target=default_target_for(op_name, ctx.task_type),
        params=params,
    )


# -----------------------------------------------------------------------------
# Random pipeline factory
# -----------------------------------------------------------------------------
# Canonical category order per task type. Operators are picked one category at
# a time, in order.
_TABULAR_ORDER: list[OpCategory] = [
    OpCategory.SCHEMA,            # CastType, RenameColumn
    OpCategory.DATETIME_PARSE,    # ParseDate, ParseDate
    OpCategory.CLEAN_VALUE,       # CustomClean, CustomClean
    OpCategory.ERROR_DETECTION,   # HandleError, HandleNonIID, ReweightUPG
    OpCategory.DEDUPLICATE,       # Deduplicate
    OpCategory.JOIN,              # JoinTable, JoinTable, ConcatTable
    OpCategory.SORT_ORDER,        # SortRows
    OpCategory.FEATURE_GEN,       # CreateFeature, CrossFeature, ...
    OpCategory.FEATURE_TIME,      # ExtractDateTimeFeature, CreateLagFeature, CreateRollingFeature
    OpCategory.OUTLIER,           # HandleOutlier
    OpCategory.MISSING_VALUE,     # HandleMV, HandleMV, HandleMV
    OpCategory.DISCRETIZATION,    # DiscretizeFeature
    OpCategory.ENCODING,          # OneHotEncode, CustomProcess, TargetEncode, ...
    OpCategory.NORMALIZATION,     # TransformPower
    OpCategory.SCALING,           # ScaleFeature, ScaleFeature, ...
    OpCategory.DISTRIBUTION_RESHAPE,  # TransformPower, TransformPower
    OpCategory.FEATURE_REDUCTION, # ReduceDimension, ReduceDimension, ReduceDimension, ReduceDimension
    OpCategory.FEATURE_SELECTION, # SelectFeature, SelectFeature
    OpCategory.FILTER_COL,        # CustomProcess, CustomProcess, SelectFeature
    OpCategory.FILTER_ROW,        # (rare in tabular)
    OpCategory.IMBALANCE,         # Oversample, Undersample, ...
    OpCategory.AUGMENT,           # AugmentNoise
]

_REC_ORDER: list[OpCategory] = [
    OpCategory.SCHEMA,
    OpCategory.JOIN,            # JoinTable (mandatory)
    OpCategory.DATETIME_PARSE,
    OpCategory.CLEAN_VALUE,
    OpCategory.FILTER_COL,
    OpCategory.FILTER_ROW,      # FilterKCore, FilterSample
    OpCategory.OUTLIER,         # HandleOutlier
    OpCategory.NORMALIZATION,   # TransformPower
    OpCategory.SCALING,
    OpCategory.DISTRIBUTION_RESHAPE,
    OpCategory.FEATURE_GEN,
    OpCategory.FEATURE_TIME,
    OpCategory.FEATURE_REDUCTION,
    OpCategory.FEATURE_SELECTION,
    OpCategory.ENCODING,        # TargetEncode
    OpCategory.SEQUENCE,        # CreateSequence (mandatory)
    OpCategory.DISCRETIZATION,  # DiscretizeFeature (optional)
    OpCategory.MISSING_VALUE,
    OpCategory.SAMPLING,        # SampleNegative
]


def _ops_in_category(cat: OpCategory, task_type: str) -> list[str]:
    out = []
    for name in operators_for_task(task_type):
        if CATALOG[name].category == cat:
            out.append(name)
    return out


def random_pipeline(
    task_type: str,
    ctx: DataContext,
    rng: _random.Random | None = None,
    p_optional: float = 0.5,
) -> Pipeline:
    """Generate a random *legal* pipeline.

    Mandatory operators are always present. Optional operators are added with
    probability ``p_optional`` per category slot.
    """
    rng = rng or _random.Random()
    order = _REC_ORDER if task_type == "rec" else _TABULAR_ORDER

    # always include LabelEncode + HandleMV for tabular as the trailing
    # normalisation; this matches dppbench task templates.
    steps: list[PipelineStep] = []
    for cat in order:
        candidates = _ops_in_category(cat, task_type)
        if not candidates:
            continue
        mandatory = [n for n in candidates if CATALOG[n].mandatory]
        for op_name in mandatory:
            step = make_step(op_name, ctx, rng)
            if step is not None:
                steps.append(step)
        # also pick at most one optional from this category
        optional = [n for n in candidates if not CATALOG[n].mandatory]
        if optional and rng.random() < p_optional:
            chosen = rng.choice(optional)
            step = make_step(chosen, ctx, rng)
            if step is not None:
                steps.append(step)

    pipe = Pipeline(steps=steps)

    # Tabular tail: ensure LabelEncode + HandleMV are present (mandatory in
    # practice for the LightGBM training path).
    if task_type == "tabular":
        from .pipeline_constraints import ensure_tabular_tail
        ensure_tabular_tail(pipe, ctx)

    # Rec post-condition: ensure SampleNegative order is correct.
    from .pipeline_constraints import repair
    repair(pipe, task_type, ctx)
    return pipe
