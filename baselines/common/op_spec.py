"""Shared operator specs for the dppbench operators.

This is the single metadata source imported by baseline-specific catalogs.
The callable operator implementations live under ``dppbench/ operators/`` and
are loaded reflectively by class name.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class OpCategory(str, Enum):
    JOIN = "JOIN"
    FILTER_ROW = "FILTER_ROW"
    FILTER_COL = "FILTER_COL"
    CLEAN_VALUE = "CLEAN_VALUE"
    OUTLIER = "OUTLIER"
    MISSING_VALUE = "MISSING_VALUE"
    DEDUPLICATE = "DEDUPLICATE"
    ERROR_DETECTION = "ERROR_DETECTION"
    DATETIME_PARSE = "DATETIME_PARSE"
    DISCRETIZATION = "DISCRETIZATION"
    ENCODING = "ENCODING"
    NORMALIZATION = "NORMALIZATION"
    SCALING = "SCALING"
    DISTRIBUTION_RESHAPE = "DISTRIBUTION_RESHAPE"
    FEATURE_GEN = "FEATURE_GEN"
    FEATURE_TIME = "FEATURE_TIME"
    FEATURE_SELECTION = "FEATURE_SELECTION"
    FEATURE_REDUCTION = "FEATURE_REDUCTION"
    RESHAPE_PIVOT = "RESHAPE_PIVOT"
    RESHAPE_LONGWIDE = "RESHAPE_LONGWIDE"
    RESHAPE_STRING = "RESHAPE_STRING"
    SORT_ORDER = "SORT_ORDER"
    GROUP_AGG = "GROUP_AGG"
    IMBALANCE = "IMBALANCE"
    AUGMENT = "AUGMENT"
    SEQUENCE = "SEQUENCE"
    SPLIT = "SPLIT"
    SAMPLING = "SAMPLING"
    SCHEMA = "SCHEMA"


@dataclass
class OpSpec:
    op_name: str
    category: OpCategory    # 算子所属阶段
    task_type: str
    prev_op_num: int = 1    # 前序算子数量；多表集成算子至少需要 2 个输入
    default_params: dict[str, Any] = field(default_factory=dict)    # 算子默认参数
    param_space: dict[str, Any] = field(default_factory=dict)    # 算子参数取值空间
    required_params: tuple[str, ...] = ()    # 算子实例化必须提供的参数
    nl_description: str = ""    # 算子的功能说明与用例，供 LLM-driven baseline 使用


CATALOG: dict[str, OpSpec] = {}


VALID_TARGETS: dict[str, tuple[str, ...]] = {}
NEEDS_CONTEXT: set[str] = set()
TABULAR_MANDATORY_OPS: set[str] = {"LabelEncode", "HandleMV"}
REC_MANDATORY_OPS: set[str] = {"JoinTable", "CreateSequence", "SampleNegative"}


_OP_DESCRIPTIONS: dict[str, str] | None = None


def _nl_description(name: str) -> str:
    global _OP_DESCRIPTIONS
    if _OP_DESCRIPTIONS is None:
        try:
            from baselines.common.op_registry import get_operator_descriptions
            _OP_DESCRIPTIONS = get_operator_descriptions()
        except Exception:
            _OP_DESCRIPTIONS = {}
    return _OP_DESCRIPTIONS.get(name, "")


def _add(name: str, category: OpCategory, task_type: str = "both",
         default_params: dict[str, Any] | None = None,
         param_space: dict[str, Any] | None = None,
         required_params: tuple[str, ...] = (),
         prev_op_num: int = 1,
         valid_targets: tuple[str, ...] = ("both",),
         needs_context: bool = False,
         mandatory: bool = False) -> None:
    CATALOG[name] = OpSpec(
        op_name=name,
        category=category,
        task_type=task_type,
        prev_op_num=prev_op_num,
        default_params=default_params or {},
        param_space=param_space or {},
        required_params=tuple(required_params),
        nl_description=_nl_description(name),
    )
    if valid_targets != ("both",):
        VALID_TARGETS[name] = valid_targets
    if needs_context:
        NEEDS_CONTEXT.add(name)
    if mandatory:
        if task_type == "rec":
            REC_MANDATORY_OPS.add(name)
        elif task_type == "tabular":
            TABULAR_MANDATORY_OPS.add(name)
        else:
            TABULAR_MANDATORY_OPS.add(name)
            REC_MANDATORY_OPS.add(name)


# S1. Data Integration
_add("JoinTable", OpCategory.JOIN, default_params={"method": "key"}, prev_op_num=2, needs_context=True)
_add("ConcatTable", OpCategory.RESHAPE_LONGWIDE, task_type="tabular", default_params={"axis": "vertical"}, prev_op_num=2, needs_context=True)
_add("AlignSchema", OpCategory.SCHEMA, needs_context=True)
_add("RenameColumn", OpCategory.SCHEMA, needs_context=True)
_add("DropColumns", OpCategory.FILTER_COL,
     required_params=("cols",), needs_context=True)
_add("CastType", OpCategory.SCHEMA, needs_context=True)
_add("ParseDate", OpCategory.DATETIME_PARSE, default_params={"mode": "string"},
     required_params=("cols",), needs_context=True)
_add("ParseNumber", OpCategory.SCHEMA,
     required_params=("cols",), needs_context=True)
_add("SortRows", OpCategory.SORT_ORDER, needs_context=True)
_add("SplitColumn", OpCategory.RESHAPE_STRING,
     required_params=("col",), needs_context=True)
_add("CustomTransform", OpCategory.SCHEMA, needs_context=True)

# S2. Data Cleaning
_add("HandleMV", OpCategory.MISSING_VALUE,
     default_params={"method": "median", "action": "impute"},
     param_space={
         "method": ["median", "mean", "mode", "constant", "knn", "iterative"],
         "action": ["delete", "impute"],
     },
     needs_context=True)
_add("HandleOutlier", OpCategory.OUTLIER,
     default_params={"method": "iqr", "action": "delete", "repair_method": "clip"},
     param_space={
         "method": ["iqr", "zscore", "isolation_forest"],
         "threshold": [2.0, 3.0, 4.0],
         "action": ["delete", "repair"],
         "repair_method": ["clip", "median", "set_missing", "winsorize"],
     },
     valid_targets=("train",), needs_context=True)
_add("HandleError", OpCategory.ERROR_DETECTION,
     default_params={"rule": "numeric", "action": "delete"},
     param_space={
         "rule": ["numeric", "positive", "in_range", "regex", "not_in"],
         "action": ["delete", "repair"],
         "repair_method": ["set_missing", "fill_constant", "clip", "median", "mode"],
     },
     valid_targets=("train",), needs_context=True)
_add("HandleNonIID", OpCategory.ERROR_DETECTION,
     default_params={"threshold": 0.95, "action": "reweight"},
     param_space={"action": ["delete", "reweight"]},
     valid_targets=("train",), needs_context=True)
_add("ReweightUPG", OpCategory.ERROR_DETECTION,
     default_params={"threshold": 0.9, "marker_weight": 2.0},
     valid_targets=("train",), needs_context=True)
_add("CorrectLabel", OpCategory.CLEAN_VALUE, needs_context=True)
_add("Deduplicate", OpCategory.DEDUPLICATE,
     default_params={"keep": "first"},
     param_space={"keep": ["first", "last", False]},
     valid_targets=("train", "both"), needs_context=True)
_add("CorrectTypo", OpCategory.CLEAN_VALUE, needs_context=True)
_add("CustomClean", OpCategory.CLEAN_VALUE, needs_context=True)

# S3. Data Preprocessing
_add("OneHotEncode", OpCategory.ENCODING,
     required_params=("cols",), needs_context=True)
_add("OrdinalEncode", OpCategory.ENCODING,
     required_params=("cols",), needs_context=True)
_add("LabelEncode", OpCategory.ENCODING, needs_context=True)
_add("HashEncode", OpCategory.ENCODING,
     required_params=("cols",), needs_context=True)
_add("TargetEncode", OpCategory.ENCODING,
     required_params=("cols", "target_col"), needs_context=True)
_add("ScaleFeature", OpCategory.SCALING,
     default_params={"method": "standard"},
     param_space={"method": ["standard", "minmax", "maxabs", "robust", "l2"]},
     needs_context=True)
_add("TransformPower", OpCategory.DISTRIBUTION_RESHAPE,
     default_params={"method": "log", "offset": 1.0},
     param_space={"method": ["log", "sqrt", "quantile"]},
     needs_context=True)
_add("DiscretizeFeature", OpCategory.DISCRETIZATION,
     default_params={"strategy": "manual"},
     param_space={"strategy": ["manual", "uniform", "quantile", "kmeans"]},
     needs_context=True)
_add("ClipOutlier", OpCategory.OUTLIER, needs_context=True)
_add("FilterSample", OpCategory.FILTER_ROW, valid_targets=("train", "interaction"), needs_context=True)
_add("SampleNegative", OpCategory.SAMPLING, task_type="rec",
     default_params={"user_col": "user_id", "item_col": "item_id", "target_col": "rating", "n_negatives": 1},
     param_space={"n_negatives": [1, 2, 3]},
     valid_targets=("interaction",), needs_context=True)
_add("FilterKCore", OpCategory.FILTER_ROW, task_type="rec",
     default_params={"k": 5}, param_space={"k": [3, 5, 10]},
     valid_targets=("interaction",), needs_context=True)
_add("Undersample", OpCategory.IMBALANCE,
     default_params={"method": "random"},
     param_space={"method": ["random", "tomek", "enn"]},
     valid_targets=("train",), needs_context=True)
_add("Oversample", OpCategory.IMBALANCE,
     default_params={"method": "random"},
     param_space={"method": ["random", "smote", "adasyn", "smote_nc"]},
     valid_targets=("train",), needs_context=True)
_add("AugmentMixup", OpCategory.AUGMENT, valid_targets=("train",), needs_context=True)
_add("AugmentNoise", OpCategory.AUGMENT,
     required_params=("cols",), valid_targets=("train",), needs_context=True)
_add("CustomProcess", OpCategory.CLEAN_VALUE, needs_context=True)

# S4. Feature Engineering
_add("FrequencyEncode", OpCategory.FEATURE_GEN,
     required_params=("cols",), needs_context=True)
_add("CreateFeature", OpCategory.FEATURE_GEN,
     required_params=("source_cols", "output_col"), needs_context=True)
_add("CreatePolynomialFeature", OpCategory.FEATURE_GEN,
     required_params=("cols",), needs_context=True)
_add("CrossFeature", OpCategory.FEATURE_GEN,
     required_params=("cols",), needs_context=True)
_add("AggregateGroupFeature", OpCategory.GROUP_AGG,
     required_params=("group_cols",), needs_context=True)
_add("ExtractDateTimeFeature", OpCategory.FEATURE_TIME,
     required_params=("cols",), needs_context=True)
_add("CreateLagFeature", OpCategory.FEATURE_TIME,
     required_params=("target_col", "lags"), needs_context=True)
_add("CreateRollingFeature", OpCategory.FEATURE_TIME,
     required_params=("target_col", "windows"), needs_context=True)
_add("CreateSequence", OpCategory.SEQUENCE, task_type="rec",
     valid_targets=("interaction",), needs_context=True)
_add("TruncateSequence", OpCategory.SEQUENCE, task_type="rec",
     required_params=("cols",),
     valid_targets=("interaction",), needs_context=True)
_add("SelectFeature", OpCategory.FEATURE_SELECTION,
     default_params={"method": "variance"},
     param_space={"method": ["variance", "univariate", "rfe", "model"]},
     needs_context=True)
_add("ReduceDimension", OpCategory.FEATURE_REDUCTION,
     default_params={"method": "pca"},
     param_space={"method": ["pca", "svd", "kernel_pca", "lda", "umap"]},
     needs_context=True)
_add("ExtractTextFeature", OpCategory.FEATURE_GEN,
     required_params=("cols",), needs_context=True)
_add("ExtractTextEmbedding", OpCategory.FEATURE_GEN,
     required_params=("cols",), needs_context=True)
_add("ExtractGraphFeature", OpCategory.FEATURE_GEN, needs_context=True)
_add("CustomFE", OpCategory.FEATURE_GEN, needs_context=True)


EXPECTED_OP_COUNT = len(CATALOG)
assert len(CATALOG) == EXPECTED_OP_COUNT, (
    f"Shared catalog should cover all {EXPECTED_OP_COUNT} dppbench operators, "
    f"got {len(CATALOG)}"
)


def valid_targets_for(op_name: str) -> tuple[str, ...]:
    return VALID_TARGETS.get(op_name, ("both",))


def needs_context(op_name: str) -> bool:
    return op_name in NEEDS_CONTEXT


def is_mandatory(op_name: str, task_type: str | None = None) -> bool:
    if task_type == "tabular":
        return op_name in TABULAR_MANDATORY_OPS
    if task_type == "rec":
        return op_name in REC_MANDATORY_OPS
    return op_name in TABULAR_MANDATORY_OPS or op_name in REC_MANDATORY_OPS


def is_custom_operator(op_name: str) -> bool:
    """Return whether an operator is reserved for LLM/custom-code baselines."""
    return str(op_name).startswith("Custom")


def operators_for_task(task_type: str, include_custom: bool = False) -> list[str]:
    """Return operator names applicable to the task type.

    By default this is the non-LLM baseline operator space, so operators whose
    names start with ``Custom`` are excluded. LLM/custom-code baselines may pass
    ``include_custom=True`` explicitly.
    """
    if task_type not in ("tabular", "rec"):
        raise ValueError(f"Unknown task_type {task_type}")
    return [
        name for name, spec in CATALOG.items()
        if (spec.task_type == task_type or spec.task_type == "both")
        and (include_custom or not is_custom_operator(name))
    ]


def get_op(op_name: str) -> OpSpec:
    """Return the OpSpec for one operator name."""
    try:
        return CATALOG[op_name]
    except KeyError as exc:
        raise KeyError(f"Unknown operator: {op_name}") from exc


def get_avilable_ops(task_type: str, include_custom: bool = False) -> list[OpSpec]:
    """Return OpSpec objects applicable to the task type.

    The misspelling is kept because it is the public name used in the docs.
    """
    return [CATALOG[name] for name in operators_for_task(task_type, include_custom=include_custom)]


def get_available_ops(task_type: str, include_custom: bool = False) -> list[OpSpec]:
    """Spelled alias for get_avilable_ops()."""
    return get_avilable_ops(task_type, include_custom=include_custom)


def operators_by_category(task_type: str, include_custom: bool = False) -> dict[OpCategory, list[str]]:
    by_cat: dict[OpCategory, list[str]] = {}
    for name in operators_for_task(task_type, include_custom=include_custom):
        spec = CATALOG[name]
        by_cat.setdefault(spec.category, []).append(name)
    return by_cat


__all__ = [
    "CATALOG",
    "EXPECTED_OP_COUNT",
    "NEEDS_CONTEXT",
    "REC_MANDATORY_OPS",
    "TABULAR_MANDATORY_OPS",
    "VALID_TARGETS",
    "OpCategory",
    "OpSpec",
    "get_available_ops",
    "get_avilable_ops",
    "get_op",
    "is_custom_operator",
    "is_mandatory",
    "needs_context",
    "operators_by_category",
    "operators_for_task",
    "valid_targets_for",
]
