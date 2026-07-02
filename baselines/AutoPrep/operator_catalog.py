"""Auto-Prep operator catalog adapter.

The executable operator universe is defined once in
``baselines.common.operator_catalog``. Auto-Prep adds only the metadata that is
specific to its probabilistic transformation prior: ``prior_features`` and
human-readable ``description`` strings used for cold-start scoring and display.

Each operator appears exactly once; when an operator supports multiple internal
methods (e.g. HandleMV with median/MICE/KNN), the method is chosen stochastically
by ``pipeline_factory.build_default_params`` rather than by duplicating the
operator entry.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from baselines.common.operator_catalog import (
    CATALOG as BASE_CATALOG,
    OpCategory,
    operators_by_category as _base_operators_by_category,
    operators_for_task as _base_operators_for_task,
)


@dataclass
class OpSpec:
    op_name: str
    category: OpCategory
    task_type: str  # "tabular" | "rec" | "both"
    default_params: dict = field(default_factory=dict)
    param_space: dict = field(default_factory=dict)
    valid_targets: tuple = ("both",)
    needs_context: bool = False
    mandatory: bool = False
    prior_features: dict = field(default_factory=dict)
    description: str = ""


_AUTOPREP_METADATA = {
    "JoinTable": {
        "description": "Join with aux/user/item tables (key-merge or rec join).",
        "prior_features": {"aux": 5.0, "const": 1.0},
    },
    "ConcatTable": {
        "description": "Concat tables along axis.",
        "prior_features": {"aux": 0.3},
    },
    "AlignSchema": {
        "description": "Align schemas across tables.",
        "prior_features": {"const": 0.1},
    },
    "CastType": {
        "description": "Cast columns to dtype.",
        "prior_features": {"const": 0.1},
    },
    "RenameColumn": {
        "description": "Rename columns.",
        "prior_features": {"const": 0.05},
    },
    "DropColumns": {
        "description": "Drop columns by explicit names.",
        "prior_features": {"id": 1.0, "const": 0.1},
    },
    "ParseDate": {
        "description": "Parse date/datetime columns (string or YYMMDD int).",
        "prior_features": {"time": 1.0, "int_date": 2.0},
    },
    "ParseNumber": {
        "description": "Parse numeric values from string columns.",
        "prior_features": {"text": 0.5},
    },
    "SortRows": {
        "description": "Sort rows by columns.",
        "prior_features": {"const": 0.05, "time": 0.3},
    },
    "SplitColumn": {
        "description": "Split a string column into multiple columns.",
        "prior_features": {"text": 0.5},
    },
    "CustomTransform": {
        "description": "User-defined column transform.",
        "prior_features": {"const": 0.1},
    },
    "HandleMV": {
        "description": "Missing-value handling (median/mean/mode/constant/knn/iterative).",
        "prior_features": {"missing": 5.0, "missing_max": 1.0},
    },
    "HandleOutlier": {
        "description": "Detect outliers and delete or repair them.",
        "prior_features": {"numeric": 0.6, "outlier": 0.5},
    },
    "HandleError": {
        "description": "Detect rule violations and delete or repair them.",
        "prior_features": {"numeric": 0.4, "const": 0.2},
    },
    "HandleNonIID": {
        "description": "Detect non-IID samples and delete or reweight them.",
        "prior_features": {"numeric": 0.3},
    },
    "ReweightUPG": {
        "description": "Detect underperforming groups and up-weight their loss.",
        "prior_features": {"const": 0.3},
    },
    "CorrectLabel": {
        "description": "Identify and flag/correct noisy labels.",
        "prior_features": {"target_classes": 0.3},
    },
    "Deduplicate": {
        "description": "Drop fully duplicated rows.",
        "prior_features": {"const": 0.5},
    },
    "CorrectTypo": {
        "description": "Correct typographical errors in text columns.",
        "prior_features": {"text": 0.5},
    },
    "CustomClean": {
        "description": "Clean values (sentinel replacement or regex substitution).",
        "prior_features": {"sentinel": 5.0, "text": 1.0},
    },
    "OneHotEncode": {
        "description": "Expand categorical to dummy/one-hot columns.",
        "prior_features": {"categorical": 0.5},
    },
    "OrdinalEncode": {
        "description": "Ordinal encode by user-given order.",
        "prior_features": {"categorical": 0.3},
    },
    "LabelEncode": {
        "description": "Factorize categorical columns to integer ids.",
        "prior_features": {"categorical": 1.0},
    },
    "HashEncode": {
        "description": "Hash high-cardinality categories into buckets.",
        "prior_features": {"high_card": 1.0},
    },
    "TargetEncode": {
        "description": "Smoothed target mean encoding.",
        "prior_features": {"categorical": 0.4, "target_classes": 0.5},
    },
    "ScaleFeature": {
        "description": "Scale numeric columns (standard/minmax/maxabs/robust/l2).",
        "prior_features": {"numeric": 0.5, "outlier": 0.3},
    },
    "TransformPower": {
        "description": "Power transform for numeric columns (log/sqrt/box-cox/yeo-johnson/quantile).",
        "prior_features": {"numeric": 0.3, "skew": 0.5},
    },
    "DiscretizeFeature": {
        "description": "Discretize numeric column into buckets.",
        "prior_features": {"numeric": 0.2},
    },
    "ClipOutlier": {
        "description": "Clip outliers to a threshold range.",
        "prior_features": {"numeric": 0.3, "outlier": 0.5},
    },
    "FilterSample": {
        "description": "Filter rows with NAs in specified columns.",
        "prior_features": {"const": 0.3},
    },
    "SampleNegative": {
        "description": "Sample negative instances per positive (recommendation).",
        "prior_features": {"const": 0.5},
    },
    "FilterKCore": {
        "description": "K-core filtering on interaction graph.",
        "prior_features": {"const": 0.5},
    },
    "Undersample": {
        "description": "Undersample majority class (random/tomek/enn).",
        "prior_features": {"imbalance": 0.3},
    },
    "Oversample": {
        "description": "Oversample minority class (random/smote/adasyn).",
        "prior_features": {"imbalance": 1.0},
    },
    "AugmentMixup": {
        "description": "Mixup data augmentation.",
        "prior_features": {"const": 0.1},
    },
    "AugmentNoise": {
        "description": "Add Gaussian noise to numeric columns.",
        "prior_features": {"const": 0.2},
    },
    "CustomProcess": {
        "description": "Custom processing (drop high-null, passthrough).",
        "prior_features": {"missing_max": 1.0},
    },
    "FrequencyEncode": {
        "description": "Frequency-count features for high-cardinality categorical columns.",
        "prior_features": {"categorical": 0.5},
    },
    "CreateFeature": {
        "description": "Create a new column via built-in operation (mean/sum/std/...).",
        "prior_features": {"numeric_pairs": 0.5},
    },
    "CreatePolynomialFeature": {
        "description": "Polynomial / interaction features.",
        "prior_features": {"numeric": 0.2},
    },
    "CrossFeature": {
        "description": "Concatenate categorical cols into one string column.",
        "prior_features": {"const": 0.2},
    },
    "AggregateGroupFeature": {
        "description": "Aggregation features within groups.",
        "prior_features": {"categorical": 0.3},
    },
    "ExtractDateTimeFeature": {
        "description": "Calendar/time features from datetime columns.",
        "prior_features": {"time": 1.0},
    },
    "CreateLagFeature": {
        "description": "Lag features per group.",
        "prior_features": {"time_target": 1.0},
    },
    "CreateRollingFeature": {
        "description": "Rolling-window aggregates.",
        "prior_features": {"time_target": 1.0},
    },
    "CreateSequence": {
        "description": "Build per-user history item sequence.",
        "prior_features": {"const": 1.0},
    },
    "TruncateSequence": {
        "description": "Truncate long sequences.",
        "prior_features": {"const": 0.2},
    },
    "SelectFeature": {
        "description": "Feature selection (variance/univariate/rfe/model).",
        "prior_features": {"numeric": 0.3},
    },
    "ReduceDimension": {
        "description": "Dimensionality reduction (pca/svd/kernel_pca/lda/umap).",
        "prior_features": {"many_numeric": 0.5, "target_classes": 0.4},
    },
    "ExtractTextFeature": {
        "description": "Extract traditional text features (TF-IDF etc.).",
        "prior_features": {"text": 0.8},
    },
    "ExtractTextEmbedding": {
        "description": "Extract dense text embeddings.",
        "prior_features": {"text": 0.5},
    },
    "ExtractGraphFeature": {
        "description": "Extract graph-based features.",
        "prior_features": {"const": 0.2},
    },
    "CustomFE": {
        "description": "User-provided feature engineering callback.",
        "prior_features": {"const": 0.1},
    },
}


def _adapt_spec(name: str) -> OpSpec:
    base = BASE_CATALOG[name]
    extra = _AUTOPREP_METADATA.get(name, {})
    return OpSpec(
        op_name=base.op_name,
        category=base.category,
        task_type=base.task_type,
        default_params=dict(base.default_params),
        param_space=dict(base.param_space),
        valid_targets=tuple(base.valid_targets),
        needs_context=base.needs_context,
        mandatory=base.mandatory,
        prior_features=dict(extra.get("prior_features", {})),
        description=str(extra.get("description", "")),
    )


CATALOG: dict[str, OpSpec] = {name: _adapt_spec(name) for name in BASE_CATALOG}

assert len(CATALOG) == len(BASE_CATALOG), (
    f"Auto-Prep catalog should cover all shared dppbench operators, got {len(CATALOG)}"
)


def operators_for_task(task_type: str) -> list[str]:
    return _base_operators_for_task(task_type)


def operators_by_category(task_type: str) -> dict[OpCategory, list[str]]:
    return _base_operators_by_category(task_type)


def format_op_descriptions(task_type: str) -> str:
    lines = []
    for name in operators_for_task(task_type):
        spec = CATALOG[name]
        lines.append(f"{name} — {spec.description}")
    return "\n".join(lines)


__all__ = [
    "OpCategory",
    "OpSpec",
    "CATALOG",
    "operators_for_task",
    "operators_by_category",
    "format_op_descriptions",
]
