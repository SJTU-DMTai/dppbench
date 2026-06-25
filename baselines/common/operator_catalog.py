"""Shared operator catalog for the 52 dppbench operators.

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
    category: OpCategory
    task_type: str  # "tabular" | "rec" | "both"
    default_params: dict[str, Any] = field(default_factory=dict)
    param_space: dict[str, Any] = field(default_factory=dict)
    valid_targets: tuple[str, ...] = ("both",)
    needs_context: bool = False
    mandatory: bool = False


CATALOG: dict[str, OpSpec] = {}


def _add(name: str, category: OpCategory, task_type: str = "both",
         default_params: dict[str, Any] | None = None,
         param_space: dict[str, Any] | None = None,
         valid_targets: tuple[str, ...] = ("both",),
         needs_context: bool = False,
         mandatory: bool = False) -> None:
    CATALOG[name] = OpSpec(
        op_name=name,
        category=category,
        task_type=task_type,
        default_params=default_params or {},
        param_space=param_space or {},
        valid_targets=valid_targets,
        needs_context=needs_context,
        mandatory=mandatory,
    )


# S1. Data Integration
_add("JoinTable", OpCategory.JOIN, default_params={"method": "key"}, needs_context=True)
_add("ConcatTable", OpCategory.RESHAPE_LONGWIDE, default_params={"axis": "vertical"}, needs_context=True)
_add("AlignSchema", OpCategory.SCHEMA, needs_context=True)
_add("RenameColumn", OpCategory.SCHEMA, needs_context=True)
_add("CastType", OpCategory.SCHEMA, needs_context=True)
_add("ParseDate", OpCategory.DATETIME_PARSE, default_params={"mode": "string"}, needs_context=True)
_add("ParseNumber", OpCategory.SCHEMA, needs_context=True)
_add("SortRows", OpCategory.SORT_ORDER, needs_context=True)
_add("SplitColumn", OpCategory.RESHAPE_STRING, needs_context=True)
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
_add("OneHotEncode", OpCategory.ENCODING, needs_context=True)
_add("OrdinalEncode", OpCategory.ENCODING, needs_context=True)
_add("LabelEncode", OpCategory.ENCODING, needs_context=True)
_add("HashEncode", OpCategory.ENCODING, needs_context=True)
_add("TargetEncode", OpCategory.ENCODING, needs_context=True)
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
_add("AugmentNoise", OpCategory.AUGMENT, valid_targets=("train",), needs_context=True)
_add("CustomProcess", OpCategory.CLEAN_VALUE, needs_context=True)

# S4. Feature Engineering
_add("CreateFeature", OpCategory.FEATURE_GEN, needs_context=True)
_add("CreatePolynomialFeature", OpCategory.FEATURE_GEN, needs_context=True)
_add("CrossFeature", OpCategory.FEATURE_GEN, needs_context=True)
_add("AggregateGroupFeature", OpCategory.GROUP_AGG, needs_context=True)
_add("ExtractDateTimeFeature", OpCategory.FEATURE_TIME, needs_context=True)
_add("CreateLagFeature", OpCategory.FEATURE_TIME, needs_context=True)
_add("CreateRollingFeature", OpCategory.FEATURE_TIME, needs_context=True)
_add("ResampleTimeSeries", OpCategory.FEATURE_TIME, needs_context=True)
_add("CreateSequence", OpCategory.SEQUENCE, task_type="rec",
     valid_targets=("interaction",), needs_context=True)
_add("TruncateSequence", OpCategory.SEQUENCE, task_type="rec",
     valid_targets=("interaction",), needs_context=True)
_add("SelectFeature", OpCategory.FEATURE_SELECTION,
     default_params={"method": "variance"},
     param_space={"method": ["variance", "univariate", "rfe", "model"]},
     needs_context=True)
_add("ReduceDimension", OpCategory.FEATURE_REDUCTION,
     default_params={"method": "pca"},
     param_space={"method": ["pca", "svd", "kernel_pca", "lda", "umap"]},
     needs_context=True)
_add("ExtractTextFeature", OpCategory.FEATURE_GEN, needs_context=True)
_add("ExtractTextEmbedding", OpCategory.FEATURE_GEN, needs_context=True)
_add("ExtractGraphFeature", OpCategory.FEATURE_GEN, needs_context=True)
_add("CustomFE", OpCategory.FEATURE_GEN, needs_context=True)


EXPECTED_OP_COUNT = len(CATALOG)
assert len(CATALOG) == EXPECTED_OP_COUNT, (
    f"Shared catalog should cover all {EXPECTED_OP_COUNT} dppbench operators, "
    f"got {len(CATALOG)}"
)


def operators_for_task(task_type: str) -> list[str]:
    """Return operator names applicable to the task type."""
    if task_type not in ("tabular", "rec"):
        raise ValueError(f"Unknown task_type {task_type}")
    return [
        name for name, spec in CATALOG.items()
        if spec.task_type == task_type or spec.task_type == "both"
    ]


def operators_by_category(task_type: str) -> dict[OpCategory, list[str]]:
    by_cat: dict[OpCategory, list[str]] = {}
    for name in operators_for_task(task_type):
        spec = CATALOG[name]
        by_cat.setdefault(spec.category, []).append(name)
    return by_cat


__all__ = [
    "CATALOG",
    "EXPECTED_OP_COUNT",
    "OpCategory",
    "OpSpec",
    "operators_by_category",
    "operators_for_task",
]
