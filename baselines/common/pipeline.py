"""Shared pipeline representation, serialization, and random factory.

A ``Pipeline`` is a list of ``PipelineStep`` objects. Each step corresponds to
one entry in the YAML format consumed by
``dppbench.dataset.{TabularData,RecData}.run_pre_process``.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import math
import os
import random as _random
from dataclasses import dataclass, field
from typing import Any

import yaml

from baselines.common.data_ctx import DataContext
from baselines.common.budget import Budget, get_active_budget
from baselines.common.op_spec import (
    CATALOG, OpCategory, is_mandatory, operators_for_task, valid_targets_for,
)

_DF_PARAM_KEYS = ("aux_df", "user_df", "item_df")
_DF_LIST_PARAM_KEYS = {"other_dfs"}


def _is_df_ref(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("$") and value[1:] != "target_col"


def _table_from_ref(value: str) -> str:
    name = value[1:]
    return name[:-3] if name.endswith("_df") else name



def _root_table_for_steps(steps: list["PipelineStep"]) -> str:
    rec_targets = {"interaction", "user", "item"}
    if any(step.target in rec_targets for step in steps):
        return "interaction"
    for step in steps:
        params = step.params or {}
        if params.get("method") == "rec" or params.get("user_df") or params.get("item_df"):
            return "interaction"
    return "main"


def _target_table(target: str, root_table: str) -> str:
    if target in ("both", "test", "train"):
        return root_table
    return target or root_table


def _source_entries_for_context(ctx: "DataContext", selected_aux: list[str] | None = None) -> list[dict]:
    if ctx.task_type == "rec":
        selected = list(ctx.secondary_tables)
    else:
        selected = selected_aux or []
    return ctx.source_entries(selected)


def _strip_df_params(params: dict) -> tuple[dict, list[str]]:
    params = copy.deepcopy(params or {})
    refs: list[str] = []
    for key in list(_DF_PARAM_KEYS):
        if key not in params:
            continue
        value = params.get(key)
        if _is_df_ref(value):
            refs.append(_table_from_ref(value))
        params.pop(key, None)
    values = params.get("other_dfs")
    if isinstance(values, list):
        remaining = []
        for value in values:
            if _is_df_ref(value):
                refs.append(_table_from_ref(value))
            else:
                remaining.append(value)
        if remaining:
            params["other_dfs"] = remaining
        else:
            params.pop("other_dfs", None)
    return params, refs


def _table_to_source_id(sources: list[dict]) -> dict[str, str]:
    return {source["table"]: source["id"] for source in sources}


def _steps_to_dag(
    steps: list["PipelineStep"],
    sources: list[dict] | None = None,
    train_prev: list[str] | None = None,
) -> dict:
    if sources is not None:
        source_by_table = _table_to_source_id(sources)
        ops = []
        for idx, step in enumerate(steps, start=1):
            node_id = step.id or f"o{idx}"
            params, _ = _strip_df_params(step.params)
            prev = list(step.prev or [])
            if not prev:
                root_table = _target_table(step.target, "interaction" if source_by_table.get("interaction") else "main")
                prev = [source_by_table.get(root_table) or sources[0]["id"]]
            node = {"id": node_id, "op": step.op, "prev": prev}
            if params:
                node["params"] = params
            ops.append(node)
        final_prev = train_prev or ([ops[-1]["id"]] if ops else [sources[0]["id"]])
        return {"dag": {"sources": sources, "ops": ops, "train": {"prev": final_prev}}}

    root_table = _root_table_for_steps(steps)
    tables: list[str] = []

    def add_table(name: str) -> None:
        if name and name not in tables:
            tables.append(name)

    add_table(root_table)
    if root_table == "interaction":
        for step in steps:
            if step.params.get("user_df"):
                add_table("user")
            if step.params.get("item_df"):
                add_table("item")
    for step in steps:
        add_table(_target_table(step.target, root_table))
        _, refs = _strip_df_params(step.params or {})
        for table in refs:
            add_table(table)

    inferred_sources = [
        {"id": f"s{idx}", "table": table}
        for idx, table in enumerate(tables)
    ]
    source_by_table = _table_to_source_id(inferred_sources)
    current = dict(source_by_table)
    ops = []

    for idx, step in enumerate(steps, start=1):
        target = _target_table(step.target, root_table)
        params, refs = _strip_df_params(step.params or {})
        prev = [current[target]]
        for table in refs:
            if table in current:
                prev.append(current[table])

        node_id = step.id or f"o{idx}"
        node = {"id": node_id, "op": step.op, "prev": prev}
        if params:
            node["params"] = params
        ops.append(node)
        current[target] = node_id

    return {"dag": {"sources": inferred_sources, "ops": ops, "train": {"prev": [current[root_table]]}}}


def _pipeline_parts_from_dag(data: dict) -> tuple[list["PipelineStep"], list[dict] | None, list[str] | None]:
    dag = data.get("dag") if isinstance(data, dict) else None
    if not isinstance(dag, dict):
        return [], None, None
    sources = copy.deepcopy(dag.get("sources", []) or [])
    table_by_source = {}
    for entry in sources:
        if isinstance(entry, dict):
            table_by_source[entry.get("id")] = entry.get("table")
        elif isinstance(entry, str):
            table_by_source[entry] = None
    steps = []
    for node in dag.get("ops", []) or []:
        prev = list(node.get("prev") or [])
        first = prev[0] if prev else None
        target = table_by_source.get(first, "both") or "both"
        if target in ("main", "train"):
            target = "both"
        steps.append(
            PipelineStep(
                op=node["op"],
                target=target,
                params=copy.deepcopy(node.get("params", {})) or {},
                id=node.get("id"),
                prev=prev,
            )
        )
    train_prev = copy.deepcopy((dag.get("train") or {}).get("prev") or None)
    return steps, sources, train_prev


def _steps_from_dag(data: dict) -> list["PipelineStep"]:
    return _pipeline_parts_from_dag(data)[0]



# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------
@dataclass
class PipelineStep:
    op: str
    target: str = "both"
    params: dict = field(default_factory=dict)
    id: str | None = None
    prev: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {"op": self.op, "target": self.target}
        if self.params:
            d["params"] = copy.deepcopy(self.params)
        if self.id is not None:
            d["id"] = self.id
        if self.prev:
            d["prev"] = list(self.prev)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineStep":
        return cls(
            op=d["op"],
            target=d.get("target", "both"),
            params=copy.deepcopy(d.get("params", {})) or {},
            id=d.get("id"),
            prev=list(d.get("prev", []) or []),
        )


@dataclass
class Pipeline:
    steps: list[PipelineStep] = field(default_factory=list)
    sources: list[dict] | None = None
    train_prev: list[str] | None = None

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_yaml(self) -> str:
        body = _steps_to_dag(self.steps, self.sources, self.train_prev)
        return yaml.safe_dump(body, sort_keys=False, default_flow_style=False)

    @classmethod
    def from_yaml(cls, text: str) -> "Pipeline":
        data = yaml.safe_load(text) or {}
        if "dag" in data:
            steps, sources, train_prev = _pipeline_parts_from_dag(data)
            return cls(steps=steps, sources=sources, train_prev=train_prev)
        steps_raw = data.get("pipeline", []) or []
        return cls(steps=[PipelineStep.from_dict(s) for s in steps_raw])

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def copy(self) -> "Pipeline":
        return Pipeline(
            steps=[PipelineStep.from_dict(s.to_dict()) for s in self.steps],
            sources=copy.deepcopy(self.sources),
            train_prev=copy.deepcopy(self.train_prev),
        )

    def hash(self) -> str:
        return hashlib.sha1(self.to_yaml().encode("utf-8")).hexdigest()

    def op_names(self) -> list[str]:
        return [s.op for s in self.steps]

    def __len__(self) -> int:
        return len(self.steps)


@dataclass
class PipelineCheckReport:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.valid


# -----------------------------------------------------------------------------
# Default-params builder for individual operators (context aware)
# -----------------------------------------------------------------------------
def _exclude_special(cols: list[str], ctx: DataContext) -> list[str]:
    bad = {ctx.target_col, ctx.id_col, ctx.user_col, ctx.item_col, ctx.time_col}
    return [c for c in cols if c not in bad]


def _rec_required_id_cols(ctx: DataContext) -> list[str]:
    cols = []
    for col in (ctx.user_col, ctx.item_col):
        if col and col not in cols:
            cols.append(col)
    return cols


def _rec_protected_cols(ctx: DataContext) -> list[str]:
    cols = []
    for col in (ctx.user_col, ctx.item_col, ctx.target_col, ctx.time_col):
        if col and col not in cols:
            cols.append(col)
    return cols


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


_OPERATOR_PARAM_CACHE: dict[str, tuple[set[str] | None, set[str]]] | None = None


def _operator_param_cache() -> dict[str, tuple[set[str] | None, set[str]]]:
    """Return ``op_name -> (accepted params, required params)``.

    ``accepted params`` is ``None`` when the implementation has ``**kwargs`` and
    therefore cannot be checked for unknown keyword names by signature alone.
    """
    global _OPERATOR_PARAM_CACHE
    if _OPERATOR_PARAM_CACHE is not None:
        return _OPERATOR_PARAM_CACHE
    out: dict[str, tuple[set[str] | None, set[str]]] = {}
    try:
        from baselines.common.op_registry import _load_all_operators
        operators = _load_all_operators()
    except Exception:
        operators = {}
    for name, cls in operators.items():
        try:
            sig = inspect.signature(cls.__init__)
        except (TypeError, ValueError):
            continue
        accepted: set[str] = set()
        required: set[str] = set()
        has_kwargs = False
        for pname, param in sig.parameters.items():
            if pname == "self":
                continue
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                has_kwargs = True
                continue
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                continue
            accepted.add(pname)
            if param.default is inspect.Parameter.empty:
                required.add(pname)
        out[name] = (None if has_kwargs else accepted, required)
    _OPERATOR_PARAM_CACHE = out
    return out


def get_operator_param_names(op_name: str) -> set[str] | None:
    """Return constructor keyword names for an operator, if introspectable."""
    item = _operator_param_cache().get(op_name)
    return None if item is None else item[0]


def _signature_required_params(op_name: str) -> set[str]:
    item = _operator_param_cache().get(op_name)
    return set() if item is None else set(item[1])


def _missing_required_params(op_name: str, params: dict) -> list[str]:
    spec = CATALOG.get(op_name)
    required = set(spec.required_params if spec is not None else ())
    missing = []
    for name in sorted(required):
        value = (params or {}).get(name)
        if value is None or value == [] or value == "":
            missing.append(name)
    return missing


def _unknown_params(op_name: str, params: dict) -> list[str]:
    accepted = get_operator_param_names(op_name)
    if accepted is None:
        return []
    return sorted(k for k in (params or {}) if k not in accepted)


def _param_space_value_errors(op_name: str, params: dict) -> list[str]:
    spec = CATALOG.get(op_name)
    if spec is None:
        return []
    errors: list[str] = []
    for key, choices in (spec.param_space or {}).items():
        if key not in (params or {}) or not isinstance(choices, list) or not choices:
            continue
        value = params[key]
        if isinstance(value, list) and not any(isinstance(c, list) for c in choices):
            bad = [v for v in value if v not in choices]
            if bad:
                errors.append(f"{op_name} param {key} has values {bad}, allowed {choices}")
        elif not isinstance(value, (list, dict)) and value not in choices:
            errors.append(f"{op_name} param {key}={value!r}, allowed {choices}")
    return errors


def validate_step_params(
    op_name: str,
    params: dict | None,
    ctx: DataContext | None = None,
    *,
    strict: bool = True,
) -> list[str]:
    """Validate one operator's concrete parameter dict.

    This is the common guard used by all baselines before execution.  It checks
    the shared ``OpSpec`` metadata, the actual operator ``__init__`` signature,
    declared parameter spaces when available, and recommendation ID-column
    protection rules.
    """
    params = params or {}
    if op_name not in CATALOG:
        return [f"unknown operator {op_name}"]
    errors: list[str] = []
    missing = _missing_required_params(op_name, params)
    if missing:
        errors.append(f"{op_name} missing required params: {missing}")
    unknown = _unknown_params(op_name, params)
    if strict and unknown:
        errors.append(f"{op_name} has unknown params: {unknown}")
    errors.extend(_param_space_value_errors(op_name, params))
    if ctx is not None:
        errors.extend(_target_leakage_errors(op_name, params, ctx))
        errors.extend(_rec_id_protection_errors(op_name, params, ctx))
    return errors


_FEATURE_INPUT_PARAMS = {
    "CreateFeature": ("source_cols",),
    "CreatePolynomialFeature": ("cols",),
    "CrossFeature": ("cols",),
    "AggregateGroupFeature": ("group_cols", "agg_cols"),
    "ReduceDimension": ("cols",),
}


def _target_leakage_errors(op_name: str, params: dict, ctx: DataContext) -> list[str]:
    # Derived features must never directly consume the supervised target.
    target = ctx.target_col
    if not target:
        return []
    leaked_params = [
        key
        for key in _FEATURE_INPUT_PARAMS.get(op_name, ())
        if target in set(_as_list((params or {}).get(key)))
    ]
    if not leaked_params:
        return []
    return [
        f"{op_name} cannot use target column {target!r} as a feature input "
        f"in params {leaked_params}"
    ]


def _rec_id_protection_errors(op_name: str, params: dict, ctx: DataContext) -> list[str]:
    if ctx.task_type != "rec":
        return []
    protected_ids = set(_rec_required_id_cols(ctx))
    if not protected_ids:
        return []
    if op_name == "DropColumns":
        dropped = protected_ids.intersection(_as_list((params or {}).get("cols")))
        if dropped:
            return [f"DropColumns cannot drop rec id columns: {sorted(dropped)}"]
    if op_name == "SelectFeature":
        excluded = set(_as_list((params or {}).get("exclude_cols")))
        missing = protected_ids - excluded
        if missing:
            return [f"SelectFeature exclude_cols must protect rec id columns: {sorted(missing)}"]
    return []


def _pick_some(cols: list[str], rng: _random.Random, lo: int = 1, hi: int = 3) -> list[str]:
    if not cols:
        return []
    n = min(len(cols), rng.randint(lo, hi))
    return cols[:n]


def build_default_params(op_name: str, ctx: DataContext, rng: _random.Random) -> dict | None:
    """Return a dict of default params for ``op_name``, or ``None`` if the
    operator cannot be applied given the current context (e.g. no time column
    for ``ExtractDateTimeFeature``).

    Covers all dppbench operators registered in :data:`CATALOG`.
    Stochastic variation across calls is used for search diversity.
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
            if not ctx.aux_dfs or ctx.id_col is None:
                return None
            joinable_aux = [
                name for name in ctx.aux_dfs
                if ctx.tables.get(name)
                and ctx.tables[name].join_keys_to_primary
            ]
            if not joinable_aux:
                return None
            aux = rng.choice(joinable_aux)
            p["aux_df"] = f"${aux}"
            join_keys = ctx.tables.get(aux).join_keys_to_primary if aux in ctx.tables else []
            p["key_col"] = join_keys[0]["primary"] if join_keys else ctx.id_col
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
    if op_name == "ParseNumber":
        parse_cols = _pick_some(text_cols + categorical_cols, rng, 1, 2)
        if not parse_cols:
            return None
        p["cols"] = parse_cols
        return p
    if op_name == "SplitColumn":
        if not text_cols:
            return None
        p["col"] = text_cols[0]
        p["sep"] = r"\s+"
        p["regex"] = True
        return p
    if op_name == "DropColumns":
        candidates = _exclude_special(ctx.all_cols, ctx)
        if not candidates:
            return None
        p["cols"] = _pick_some(candidates, rng, 1, 2)
        return p
    if op_name == "CustomProcess":
        variant = rng.choice(["drop_null", "passthrough"])
        if variant == "drop_null":
            p["threshold"] = 0.9
            p["mode"] = "drop_high_null"
            return p
        return p
    if op_name == "FrequencyEncode":
        if not categorical_cols:
            return None
        p["cols"] = _pick_some(categorical_cols, rng, 1, 5)
        return p
    if op_name == "CustomTransform":
        return p
    if op_name == "AlignSchema":
        return p
    if op_name == "SelectFeature":
        if not ctx.target_col:
            return None
        method = rng.choice(["variance", "univariate", "model"])
        p["target_col"] = ctx.target_col
        p["method"] = method
        p["exclude_cols"] = _rec_protected_cols(ctx) if ctx.task_type == "rec" else [ctx.target_col]
        if method == "univariate":
            p["k"] = max(5, min(50, len(numeric_cols)))
        elif method == "model":
            p["n_features_to_select"] = max(5, min(20, max(2, len(numeric_cols) // 2)))
        return p

    # ---- CLEAN_VALUE ----
    if op_name == "CustomClean":
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
        p["confidence_threshold"] = 0.9
        return p
    if op_name == "CorrectTypo":
        if not text_cols:
            return None
        p["cols"] = _pick_some(text_cols, rng, 1, 2)
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

    # ---- OUTLIER ----
    if op_name == "HandleOutlier":
        if not numeric_cols:
            return None
        p["cols"] = _pick_some(numeric_cols, rng, 1, 3)
        p["action"] = "delete"
        return p
    if op_name == "ClipOutlier":
        if not numeric_cols:
            return None
        p["cols"] = _pick_some(numeric_cols, rng, 1, 3)
        return p

    # ---- MISSING_VALUE ----
    if op_name == "HandleMV":
        variant = rng.choice(["global", "cols"])
        if variant == "global":
            return p
        else:
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
    if op_name == "ScaleFeature":
        if not numeric_cols:
            return None
        p["method"] = rng.choice(["standard", "minmax", "maxabs", "robust", "l2"])
        if p["method"] == "standard" and rng.random() < 0.3:
            p["cols"] = list(numeric_cols)
        else:
            p["cols"] = _pick_some(numeric_cols, rng, 1, 5)
        return p
    if op_name == "TransformPower":
        if not numeric_cols:
            return None
        p["cols"] = _pick_some(numeric_cols, rng, 1, 5)
        p["method"] = rng.choice(["log", "sqrt", "quantile"])
        if p["method"] in ("log", "sqrt"):
            p["offset"] = 1.0
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
    # ---- FE-Reduction ----
    if op_name == "ReduceDimension":
        if len(numeric_cols) < 4:
            return None
        n_comp = min(8, max(2, len(numeric_cols) // 2))
        p["cols"] = numeric_cols
        p["n_components"] = n_comp
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
    if op_name == "TruncateSequence":
        if ctx.task_type != "rec":
            return None
        p["cols"] = ["item_id_seq"]
        return p

    # ---- SAMPLING ----
    if op_name == "SampleNegative":
        if ctx.task_type != "rec" or not ctx.target_col:
            return None
        p["user_col"] = ctx.user_col
        p["item_col"] = ctx.item_col
        p["target_col"] = ctx.target_col
        return p

    # ---- GROUP AGG ----
    if op_name == "AggregateGroupFeature":
        group_cols = []
        if ctx.id_col:
            group_cols.append(ctx.id_col)
        if ctx.task_type == "rec" and ctx.user_col:
            group_cols.append(ctx.user_col)
        if not group_cols and categorical_cols:
            group_cols = _pick_some(categorical_cols, rng, 1, 2)
        if not group_cols:
            return None
        p["group_cols"] = group_cols
        if numeric_cols:
            p["agg_cols"] = _pick_some(numeric_cols, rng, 1, 3)
            p["agg_funcs"] = rng.choice(
                [["mean", "std"], ["mean", "count"], ["sum", "count"], ["mean", "max", "min"]]
            )
        else:
            p["agg_funcs"] = ["count"]
        return p

    # ---- TEXT FEATURES ----
    if op_name == "ExtractTextFeature":
        if not text_cols:
            return None
        p["cols"] = _pick_some(text_cols, rng, 1, 2)
        p["method"] = rng.choice(["tfidf", "bow"])
        p["max_features"] = int(rng.choice([50, 100, 200]))
        return p
    if op_name == "ExtractTextEmbedding":
        if not text_cols:
            return None
        p["cols"] = _pick_some(text_cols, rng, 1, 2)
        p["method"] = "hash"
        p["dim"] = 32
        return p

    # ---- GRAPH FEATURES ----
    if op_name == "ExtractGraphFeature":
        return p

    # ---- CUSTOM FE ----
    if op_name == "CustomFE":
        return None

    return p


def default_target_for(op_name: str, task_type: str) -> str:
    if valid_targets_for(op_name) == ("interaction",):
        return "interaction"
    if task_type in ("tabular", "graph"):
        return "both"
    return "interaction"


def make_step(op_name: str, ctx: DataContext, rng: _random.Random) -> PipelineStep | None:
    """Construct a PipelineStep with sensible default params, or ``None`` if the
    operator is not applicable in the given context."""
    params = build_default_params(op_name, ctx, rng)
    if params is None:
        return None
    if validate_step_params(op_name, params, ctx, strict=True):
        return None
    return PipelineStep(
        op=op_name,
        target=default_target_for(op_name, ctx.task_type),
        params=params,
    )


# -----------------------------------------------------------------------------
# DAG structure helpers
# -----------------------------------------------------------------------------
def _referenced_source_tables(steps: list[PipelineStep]) -> list[str]:
    tables: list[str] = []
    for step in steps:
        _, refs = _strip_df_params(step.params or {})
        for table in refs:
            if table not in tables:
                tables.append(table)
    return tables


def _selected_aux_tables(ctx: DataContext, steps: list[PipelineStep], rng: _random.Random) -> list[str]:
    selected: list[str] = []
    for table in _referenced_source_tables(steps):
        if table in ctx.aux_dfs and table not in selected:
            selected.append(table)
    return selected


def assign_dag_structure(pipeline: Pipeline, ctx: DataContext, rng: _random.Random | None = None) -> Pipeline:
    """Assign source nodes and op predecessors for the prev-only DAG YAML.

    SAGA searches this structure explicitly. Main/interaction source tables are
    mandatory; side sources are selected when an operator references them. The
    current implementation keeps a connected primary chain and expresses table
    integration operators as multi-input DAG nodes.
    """
    rng = rng or _random.Random()
    aux_tables = _selected_aux_tables(ctx, pipeline.steps, rng)
    sources = _source_entries_for_context(ctx, aux_tables)
    source_by_table = _table_to_source_id(sources)
    root_table = ctx.primary_table
    current = source_by_table[root_table]

    for idx, step in enumerate(pipeline.steps, start=1):
        step.id = f"o{idx}"
        params, refs = _strip_df_params(step.params or {})
        prev = [current]
        if step.op == "JoinTable":
            for table in refs:
                sid = source_by_table.get(table)
                if sid and sid not in prev:
                    prev.append(sid)
            if ctx.task_type == "rec" and params.get("method") == "rec":
                for table in ("user", "item"):
                    sid = source_by_table.get(table)
                    if sid and sid not in prev:
                        prev.append(sid)
        elif step.op == "ConcatTable":
            for table in refs:
                sid = source_by_table.get(table)
                if sid and sid not in prev:
                    prev.append(sid)
        step.prev = prev
        current = step.id

    pipeline.sources = sources
    pipeline.train_prev = [current] if pipeline.steps else [source_by_table[root_table]]
    return pipeline


# -----------------------------------------------------------------------------
# Pipeline legality checks, repair, and random factory
# -----------------------------------------------------------------------------
_CATEGORY_GROUPS: list[OpCategory] = list(OpCategory)


def is_legal(pipeline: "Pipeline", task_type: str | DataContext) -> bool:
    """Return operator applicability, or full legality with a data context.

    Passing a :class:`DataContext` performs the full :func:`check_pipeline`
    contract. Passing only a task-type string is the lightweight form used for
    partial search states. Categories impose no execution order.
    """
    if isinstance(task_type, DataContext):
        return check_pipeline(pipeline, task_type).valid
    return all(
        name in CATALOG and CATALOG[name].task_type in (task_type, "both")
        for name in pipeline.op_names()
    )


def truncate_optional_steps(
    pipeline: "Pipeline",
    task_type: str,
    max_len: int | None,
) -> "Pipeline":
    """Drop trailing non-mandatory operators until ``len(pipeline) <= max_len``.

    Mandatory operators are preserved; callers should still run ``repair`` and
    legality checks after truncation.
    """
    if max_len is None or max_len <= 0:
        return pipeline
    while len(pipeline.steps) > max_len:
        drop_idx = None
        for idx in range(len(pipeline.steps) - 1, -1, -1):
            if not is_mandatory(pipeline.steps[idx].op, task_type):
                drop_idx = idx
                break
        if drop_idx is None:
            break
        pipeline.steps.pop(drop_idx)
    return pipeline


def repair(
    pipeline: "Pipeline",
    task_type: str,
    ctx: DataContext,
    *,
    sort_steps: bool = True,
) -> "Pipeline":
    """In-place structural repair. Returns the same pipeline.

    ``sort_steps`` is retained for API compatibility but categories no longer
    impose an operator order.
    """
    seen = set()
    deduped: list[PipelineStep] = []
    for step in pipeline.steps:
        if step.op in seen:
            continue
        seen.add(step.op)
        deduped.append(step)
    pipeline.steps = deduped

    pipeline.steps = [
        step for step in pipeline.steps
        if CATALOG[step.op].task_type in (task_type, "both")
    ]

    if ctx.secondary_tables and "JoinTable" not in pipeline.op_names():
        step = make_step("JoinTable", ctx, _random.Random())
        if step is not None:
            pipeline.steps.insert(0, step)

    if task_type == "rec":
        rng = _random.Random()
        existing = {step.op for step in pipeline.steps}
        required = ["CreateSequence"]
        for op_name in required:
            if op_name not in existing:
                step = make_step(op_name, ctx, rng)
                if step is not None:
                    pipeline.steps.append(step)
    else:
        ensure_tabular_tail(pipeline, ctx)

    return pipeline


def ensure_tabular_tail(pipeline: "Pipeline", ctx: DataContext) -> None:
    """Ensure tabular pipelines include LabelEncode and HandleMV."""
    rng = _random.Random()
    if "LabelEncode" not in pipeline.op_names():
        step = make_step("LabelEncode", ctx, rng)
        if step is not None:
            pipeline.steps.append(step)
    if "HandleMV" not in pipeline.op_names():
        step = make_step("HandleMV", ctx, rng)
        if step is not None:
            pipeline.steps.append(step)



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
    max_len: int | None = None,
) -> Pipeline:
    """Generate a random *legal* pipeline.

    Mandatory operators are always present. Optional operators are added with
    probability ``p_optional`` per category slot.
    """
    rng = rng or _random.Random()
    categories = list(OpCategory)

    # always include LabelEncode + HandleMV for tabular as the trailing
    # normalisation; this matches dppbench task templates.
    steps: list[PipelineStep] = []
    for cat in categories:
        candidates = _ops_in_category(cat, task_type)
        if not candidates:
            continue
        mandatory = [n for n in candidates if is_mandatory(n, task_type)]
        for op_name in mandatory:
            step = make_step(op_name, ctx, rng)
            if step is not None:
                steps.append(step)
        # also pick at most one optional from this category
        optional = [n for n in candidates if not is_mandatory(n, task_type)]
        if optional and rng.random() < p_optional:
            if max_len is None or len(steps) < max_len:
                chosen = rng.choice(optional)
                step = make_step(chosen, ctx, rng)
                if step is not None:
                    steps.append(step)

    pipe = Pipeline(steps=steps)

    # Tabular tail: ensure LabelEncode + HandleMV are present (mandatory in
    # practice for the LightGBM training path).
    if task_type == "tabular":
        ensure_tabular_tail(pipe, ctx)

    # Rec post-condition: ensure SampleNegative order is correct.
    repair(pipe, task_type, ctx)
    truncate_optional_steps(pipe, task_type, max_len)
    assign_dag_structure(pipe, ctx, rng)
    return pipe


def finalize_pipeline(
    pipeline: Pipeline,
    ctx: DataContext,
    rng: _random.Random | None = None,
    *,
    max_len: int | None = None,
    sort_steps: bool = True,
) -> tuple[Pipeline, PipelineCheckReport]:
    """Apply the common execution contract to a candidate pipeline.

    The function performs structural repair, optional budget truncation, DAG
    assignment, and full legality/parameter validation. Baselines should call
    this before evaluating or persisting a searched pipeline.
    """
    rng = rng or _random.Random()
    repair(pipeline, ctx.task_type, ctx, sort_steps=sort_steps)
    truncate_optional_steps(pipeline, ctx.task_type, max_len)
    repair(pipeline, ctx.task_type, ctx, sort_steps=sort_steps)
    assign_dag_structure(pipeline, ctx, rng)
    return pipeline, check_pipeline(pipeline, ctx)


# -----------------------------------------------------------------------------
# Public DAG/YAML interfaces exposed to baseline implementations
# -----------------------------------------------------------------------------
def get_source_tables(ctx: DataContext) -> list[dict]:
    """Return source table entries for a baseline-visible DAG.

    ``s0`` is always the task primary table; ``s1...`` are secondary tables in
    the order carried by :class:`DataContext`.
    """
    return ctx.source_entries()


def init_pipeline(ctx: DataContext) -> Pipeline:
    """Create an empty DAG pipeline with sources and ``train.prev`` populated."""
    sources = get_source_tables(ctx)
    train_prev = [sources[0]["id"]] if sources else []
    return Pipeline(steps=[], sources=sources, train_prev=train_prev)


def init_piepeline(ctx: DataContext) -> Pipeline:
    """Backward-compatible alias preserving the spelling used in early notes."""
    return init_pipeline(ctx)


def _op_name(op: str | object) -> str:
    return getattr(op, "op_name", op)


def get_param_space(op, ctx: DataContext) -> dict[str, Any]:
    """Return a context-aware parameter space for one operator.

    The static ``OpSpec.param_space`` is augmented with task columns, source
    tables, and join keys so baselines can sample valid parameters without
    duplicating dataset-specific logic.
    """
    name = _op_name(op)
    spec = CATALOG.get(name)
    if spec is None:
        raise KeyError(f"Unknown operator: {name}")
    space = copy.deepcopy(spec.param_space or {})
    numeric_cols = _exclude_special(ctx.numeric_cols, ctx)
    categorical_cols = _exclude_special(ctx.categorical_cols, ctx)
    all_cols = _exclude_special(ctx.all_cols, ctx)

    if name == "JoinTable":
        space["source_tables"] = get_source_tables(ctx)
        if ctx.task_type == "rec":
            space["user_df"] = ["$user"] if ctx.has_user_df else []
            space["item_df"] = ["$item"] if ctx.has_item_df else []
            space["user_col"] = [ctx.user_col]
            space["item_col"] = [ctx.item_col]
        else:
            aux_tables = list(ctx.secondary_tables)
            space["aux_df"] = [f"${table}" for table in aux_tables]
            join_cols = []
            for table in aux_tables:
                table_ctx = ctx.tables.get(table)
                for key in (table_ctx.join_keys_to_primary if table_ctx else []):
                    primary_key = key.get("primary")
                    if primary_key and primary_key not in join_cols:
                        join_cols.append(primary_key)
            if not join_cols and ctx.id_col:
                join_cols.append(ctx.id_col)
            space["key_col"] = join_cols
        return space

    if name == "ConcatTable":
        space["other_dfs"] = [[f"${table}"] for table in ctx.secondary_tables]
        return space

    if name in {
        "HandleMV", "HandleOutlier", "ClipOutlier", "ScaleFeature",
        "TransformPower", "DiscretizeFeature", "AugmentNoise",
        "CreatePolynomialFeature", "ReduceDimension",
    }:
        space.setdefault("cols", numeric_cols)
    elif name in {"OneHotEncode", "OrdinalEncode", "HashEncode", "TargetEncode", "FrequencyEncode"}:
        space.setdefault("cols", categorical_cols)
    elif name in {"ParseDate", "ExtractDateTimeFeature"} and ctx.time_col:
        space.setdefault("cols", [ctx.time_col])
    elif name in {"ParseNumber", "CorrectTypo", "ExtractTextFeature", "ExtractTextEmbedding"}:
        space.setdefault("cols", ctx.text_cols or categorical_cols)
    elif name in {"DropColumns"}:
        space.setdefault("cols", all_cols)

    if name == "SelectFeature":
        protected = _rec_protected_cols(ctx) if ctx.task_type == "rec" else ([ctx.target_col] if ctx.target_col else [])
        if protected:
            space.setdefault("exclude_cols", protected)
    if name in {"SelectFeature", "TargetEncode", "Oversample", "Undersample", "SampleNegative"} and ctx.target_col:
        space.setdefault("target_col", [ctx.target_col])
    if name in {"CreateLagFeature", "CreateRollingFeature"}:
        if ctx.target_col:
            space.setdefault("target_col", [ctx.target_col])
        if ctx.time_col:
            space.setdefault("time_col", [ctx.time_col])
    if name == "TruncateSequence":
        space.setdefault("cols", ["item_id_seq"])
    if name in {"FilterKCore", "CreateSequence", "SampleNegative"}:
        space.setdefault("user_col", [ctx.user_col])
        space.setdefault("item_col", [ctx.item_col])
    return {k: v for k, v in space.items() if v not in (None, [])}


def get_order_constraints(task_type: str) -> list[OpCategory]:
    """Compatibility alias returning category groups, without order semantics."""
    if task_type not in ("tabular", "rec"):
        return []
    applicable = {CATALOG[name].category for name in operators_for_task(task_type)}
    return [category for category in _CATEGORY_GROUPS if category in applicable]


def get_category_order(task_type: str) -> list[OpCategory]:
    """Compatibility alias returning the available category groups."""
    return get_order_constraints(task_type)


def _pipeline_data(pipeline: Pipeline | str | dict) -> dict:
    if isinstance(pipeline, Pipeline):
        return yaml.safe_load(pipeline.to_yaml()) or {}
    if isinstance(pipeline, str):
        return yaml.safe_load(pipeline) or {}
    if isinstance(pipeline, dict):
        return copy.deepcopy(pipeline)
    raise TypeError(f"Unsupported pipeline type: {type(pipeline).__name__}")


def check_pipeline(
    pipeline: Pipeline | str | dict,
    ctx: DataContext | None = None,
    *,
    require_mandatory: bool = True,
    budget: Budget | None = None,
    execution_data: Any | None = None,
    required_columns: list[str] | None = None,
) -> PipelineCheckReport:
    """Validate the shared DAG pipeline format.

    Checks source/op id uniqueness, predecessor references, operator input
    counts, reachability from ``train.prev``, operator applicability,
    context-dependent mandatory operators, active budgets and (when supplied)
    the shape/schema/dtypes produced by execution. Categories do not constrain
    order.
    """
    errors: list[str] = []
    warnings: list[str] = []
    budget = budget or get_active_budget()
    try:
        data = _pipeline_data(pipeline)
    except Exception as exc:
        return PipelineCheckReport(False, [f"invalid yaml: {type(exc).__name__}: {exc}"], [])

    dag = data.get("dag") if isinstance(data, dict) else None
    if not isinstance(dag, dict):
        return PipelineCheckReport(False, ["pipeline must contain a dag mapping"], [])

    sources = dag.get("sources") or []
    ops = dag.get("ops") or []
    train = dag.get("train") or {}
    if not sources:
        errors.append("dag.sources must contain at least one source")
    if not isinstance(ops, list):
        errors.append("dag.ops must be a list")
        ops = []
    if budget is not None:
        if (
            budget.max_pipeline_length is not None
            and len(ops) > budget.max_pipeline_length
        ):
            errors.append(
                f"pipeline length {len(ops)} exceeds budget "
                f"max_pipeline_length={budget.max_pipeline_length}"
            )
        errors.extend(budget.violations())

    known: set[str] = set()
    source_ids: set[str] = set()
    for idx, source in enumerate(sources):
        sid = source.get("id") if isinstance(source, dict) else None
        if not sid:
            errors.append(f"source at index {idx} is missing id")
            continue
        if sid in known:
            errors.append(f"source id {sid} is duplicated")
        known.add(sid)
        source_ids.add(sid)

    if ctx is not None:
        source_tables = {
            source.get("table") for source in sources if isinstance(source, dict)
        }
        unknown_tables = sorted(
            table for table in source_tables
            if table is not None and table not in ctx.tables
        )
        if unknown_tables:
            errors.append(f"dag.sources contains unknown context tables: {unknown_tables}")
        if ctx.primary_table not in source_tables:
            errors.append(f"dag.sources must contain primary table {ctx.primary_table}")
        if (
            require_mandatory
            and ctx.secondary_tables
            and not source_tables.intersection(ctx.secondary_tables)
        ):
            errors.append("multi-table pipeline must contain at least one secondary source table")

    op_by_id: dict[str, dict] = {}
    deps: dict[str, list[str]] = {}
    for idx, node in enumerate(ops):
        if not isinstance(node, dict):
            errors.append(f"op at index {idx} must be a mapping")
            continue
        oid = node.get("id")
        op_name = node.get("op")
        if not oid:
            errors.append(f"op at index {idx} is missing id")
            continue
        if oid in known:
            errors.append(f"op id {oid} is duplicated")
        if op_name not in CATALOG:
            errors.append(f"op {oid} has unknown operator {op_name}")
        prev = node.get("prev") or []
        if not isinstance(prev, list):
            errors.append(f"op {oid} prev must be a list")
            prev = []
        for pid in prev:
            if pid not in known:
                errors.append(f"op {oid} prev points to unknown or later node {pid}")
        if op_name in CATALOG:
            params = node.get("params") or {}
            if ctx is not None:
                spec_task = CATALOG[op_name].task_type
                if spec_task not in (ctx.task_type, "both"):
                    errors.append(
                        f"{op_name} {oid} is not applicable to "
                        f"task_type={ctx.task_type}"
                    )
            for param_error in validate_step_params(op_name, params, ctx, strict=True):
                errors.append(f"{op_name} {oid}: {param_error}")
            expected = CATALOG[op_name].prev_op_num
            if expected <= 1 and len(prev) != expected:
                errors.append(f"{op_name} {oid} expects {expected} prev nodes, got {len(prev)}")
            elif expected > 1 and len(prev) < expected:
                errors.append(f"{op_name} {oid} expects at least {expected} prev nodes, got {len(prev)}")
        known.add(oid)
        op_by_id[oid] = node
        deps[oid] = list(prev)

    train_prev = train.get("prev") or []
    if not isinstance(train_prev, list) or not train_prev:
        errors.append("dag.train.prev must be a non-empty list")
        train_prev = []
    for pid in train_prev:
        if pid not in known:
            errors.append(f"train.prev points to unknown node {pid}")

    reachable: set[str] = set()
    stack = list(train_prev)
    while stack:
        nid = stack.pop()
        if nid in reachable:
            continue
        reachable.add(nid)
        stack.extend(deps.get(nid, []))
    for oid in op_by_id:
        if oid not in reachable:
            errors.append(f"op {oid} is not reachable from train.prev")
    for source in sources:
        sid = source.get("id") if isinstance(source, dict) else None
        if sid and sid not in reachable:
            errors.append(f"source {sid} is not reachable from train.prev")

    if ctx is not None and require_mandatory:
        op_names = {
            node.get("op") for node in ops
            if isinstance(node, dict) and node.get("op") in CATALOG
        }
        required = (
            {"CreateSequence"}
            if ctx.task_type == "rec"
            else {"LabelEncode", "HandleMV"}
        )
        requires_join = bool(ctx.secondary_tables or len(sources) > 1)
        if requires_join:
            required.add("JoinTable")
        missing = sorted(required - op_names)
        if missing:
            errors.append(
                f"pipeline is missing mandatory operators for current data context: {missing}"
            )
        if requires_join and "JoinTable" in op_names:
            provenance: dict[str, set[str]] = {sid: {sid} for sid in source_ids}
            valid_join = False
            for oid, node in op_by_id.items():
                ancestors: set[str] = set()
                for pid in deps.get(oid, []):
                    ancestors.update(provenance.get(pid, set()))
                provenance[oid] = ancestors
                if node.get("op") == "JoinTable" and len(ancestors) >= 2:
                    valid_join = True
            if not valid_join:
                errors.append("JoinTable must combine branches from at least two source tables")

    if execution_data is not None:
        errors.extend(_validate_execution_data(
            execution_data,
            ctx=ctx,
            budget=budget,
            required_columns=required_columns,
        ))

    return PipelineCheckReport(not errors, errors, warnings)


def _validate_execution_data(
    execution_data: Any,
    *,
    ctx: DataContext | None,
    budget: Budget | None,
    required_columns: list[str] | None,
) -> list[str]:
    """Validate the primary frame produced by pipeline execution."""
    if isinstance(execution_data, dict):
        shape = execution_data.get("shape")
        columns = list(execution_data.get("columns") or [])
        dtypes = dict(execution_data.get("dtypes") or {})
    else:
        shape = getattr(execution_data, "shape", None)
        raw_columns = getattr(execution_data, "columns", None)
        columns = list(raw_columns) if raw_columns is not None else []
        raw_dtypes = getattr(execution_data, "dtypes", None)
        dtypes = (
            {str(col): str(dtype) for col, dtype in raw_dtypes.items()}
            if raw_dtypes is not None else {}
        )
    errors: list[str] = []
    if not isinstance(shape, (list, tuple)) or len(shape) != 2:
        return ["execution output must expose a two-dimensional shape"]
    rows, cols = int(shape[0]), int(shape[1])
    if rows <= 0 or cols <= 0:
        errors.append(f"execution output must be non-empty, got shape=({rows}, {cols})")
    if len(columns) != cols:
        errors.append(f"execution output shape has {cols} columns but schema reports {len(columns)}")
    if len(set(columns)) != len(columns):
        errors.append("execution output contains duplicate column names")

    required = set(required_columns or [])
    if ctx is not None:
        if ctx.target_col:
            required.add(ctx.target_col)
        if ctx.task_type == "rec":
            required.update(col for col in (ctx.user_col, ctx.item_col) if col)
    missing = sorted(required - set(columns))
    if missing:
        errors.append(f"execution output is missing required columns: {missing}")

    missing_dtypes = sorted(set(columns) - set(dtypes))
    if missing_dtypes:
        errors.append(f"execution output is missing dtype metadata for columns: {missing_dtypes}")
    unsupported = sorted(
        col for col, dtype in dtypes.items()
        if not str(dtype).strip() or any(token in str(dtype).lower() for token in ("complex", "void"))
    )
    if unsupported:
        errors.append(f"execution output has unsupported dtypes in columns: {unsupported}")

    if budget is not None:
        if budget.max_output_rows is not None and rows > budget.max_output_rows:
            errors.append(
                f"execution output rows {rows} exceed budget max_output_rows={budget.max_output_rows}"
            )
        if budget.max_output_columns is not None and cols > budget.max_output_columns:
            errors.append(
                f"execution output columns {cols} exceed budget "
                f"max_output_columns={budget.max_output_columns}"
            )
    return errors


class Evaluator:
    """Documented YAML-facing evaluator wrapper.

    It delegates to :class:`baselines.common.evaluator.PipelineEvaluator`, so
    scoring uses the same downstream training executor as every baseline.
    """

    def __init__(
        self,
        data_name: str,
        metric_key: str | None = None,
        device: str = "cpu",
        data_dir: str | None = None,
        task_dir: str | None = None,
        model_name: str | None = None,
        verbose: bool = False,
        scoring_method: str = "full_train",
        budget: Budget | None = None,
        seed: int | None = None,
    ) -> None:
        self.data_name = data_name
        self.metric_key = metric_key
        self.device = device
        self.data_dir = data_dir
        self.model_name = model_name
        self.verbose = verbose
        self.scoring_method = scoring_method
        self.budget = budget
        self.seed = seed
        if task_dir is None:
            root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            task_dir = os.path.join(root, "dppbench", "tasks", data_name)
        self.task_dir = task_dir
        self._delegate = None

    @property
    def delegate(self):
        if self._delegate is None:
            from baselines.common.evaluator import PipelineEvaluator
            self._delegate = PipelineEvaluator(
                task_dir=self.task_dir,
                data_name=self.data_name,
                data_dir=self.data_dir,
                metric_key=self.metric_key,
                verbose=self.verbose,
                device=self.device,
                model_name=self.model_name,
                scoring_method=self.scoring_method,
                budget=self.budget,
                seed=self.seed,
            )
        return self._delegate

    def evaluate(self, pipeline: Pipeline | str, scoring_method: str | None = None):
        if isinstance(pipeline, str):
            return self.evaluate_yaml(pipeline, scoring_method=scoring_method)
        return self.delegate.evaluate(pipeline, scoring_method=scoring_method)

    def evaluate_yaml(self, pipeline_yaml: str, scoring_method: str | None = None):
        from baselines.common.evaluator import EvaluationResult
        report = check_pipeline(pipeline_yaml)
        if not report.valid:
            return EvaluationResult(
                fitness=-math.inf,
                success=False,
                metrics={},
                error="; ".join(report.errors),
                duration_seconds=0.0,
            )
        return self.delegate.evaluate(
            Pipeline.from_yaml(pipeline_yaml),
            scoring_method=scoring_method,
        )
