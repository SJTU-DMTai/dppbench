"""Auto-Prep operator catalog adapter.

The executable operator universe is defined once in
``baselines.common.operator_catalog``. Auto-Prep adds only the metadata that is
specific to its probabilistic transformation prior: ``prior_features`` and
human-readable ``description`` strings used for cold-start scoring and display.
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
    searchable: bool = True
    prior_features: dict = field(default_factory=dict)
    description: str = ""


_AUTOPREP_METADATA = {'Oversample': {'description': 'ADASYN oversampling.', 'prior_features': {'imbalance': 0.5}},
 'JoinTable': {'description': 'Aggregate aux table on key then left-merge.',
              'prior_features': {'aux': 5.0}},
 'AugmentNoise': {'description': 'Add Gaussian noise to numeric cols.',
                  'prior_features': {'const': 0.2}},
 'Oversample': {'description': 'Random oversample/undersample/SMOTE.',
                   'prior_features': {'imbalance': 1.0}},
 'DiscretizeFeature': {'description': 'Discretize numeric column to buckets.',
               'prior_features': {'numeric': 0.2}},
 'CastType': {'description': 'Cast columns to dtype.', 'prior_features': {'const': 0.1}},
 'CrossFeature': {'description': 'Concatenate cols into one string column.',
                   'prior_features': {'const': 0.2}},
 'ConcatTable': {'description': 'Concat tables along axis.', 'prior_features': {'aux': 0.3}},
 'CreateFeature': {'description': 'Create one new column from source_cols via a built-in algorithm '
                                  '(mean/sum/std/min/max/median/product/diff/ratio/inc_ratio/concat/identity) '
                                  'or a user-supplied callable. Replaces the old DeriveFeatures + '
                                  'CreateFeature.',
                   'prior_features': {'numeric_pairs': 0.5}},
 'CreateSequence': {'description': 'Build per-user history sequence.',
                    'prior_features': {'const': 1.0}},
 'ExtractDateTimeFeature': {'description': 'Calendar features from datetime.',
                      'prior_features': {'time': 1.0}},
 'Deduplicate': {'description': 'Drop fully duplicated rows.', 'prior_features': {'const': 0.5}},
 'CustomProcess': {'description': 'Drop given columns.', 'prior_features': {'id': 1.0}},
 'CustomProcess': {'description': 'Drop columns with too many NaN.',
                  'prior_features': {'missing_max': 1.0}},
 'Undersample': {'description': 'Edited nearest-neighbours undersampling.',
                    'prior_features': {'imbalance': 0.3}},
 'HandleError': {'description': 'Detect rule violations and delete or repair them.',
                 'prior_features': {'numeric': 0.4, 'const': 0.2}},
 'HandleMV': {'description': 'Median/mean/mode imputation.', 'prior_features': {'missing': 5.0}},
 'FilterSample': {'description': 'Row-level NA filter.', 'prior_features': {'const': 0.3}},
 'HandleOutlier': {'description': 'Detect outliers and delete or repair them.',
                   'prior_features': {'numeric': 0.6}},
 'HandleNonIID': {'description': 'Detect non-IID samples and delete or reweight them.',
                  'prior_features': {'numeric': 0.3}},
 'ReweightUPG': {'description': 'Detect underperforming groups and up-weight their loss.',
                 'prior_features': {'const': 0.3}},
 'CustomProcess': {'description': 'Replace category with its frequency.',
                     'prior_features': {'categorical': 0.3}},
 'HashEncode': {'description': 'Hash high-cardinality categories.',
                'prior_features': {'high_card': 1.0}},
 'HandleMV': {'description': 'MICE-style iterative imputation.',
                      'prior_features': {'missing': 3.0}},
 'FilterKCore': {'description': 'K-core filtering on interaction graph.',
                 'prior_features': {'const': 0.5}},
 'ReduceDimension': {'description': 'Kernel PCA.', 'prior_features': {'many_numeric': 0.3}},
 'HandleMV': {'description': 'KNN-based imputation.', 'prior_features': {'missing': 3.0}},
 'LabelEncode': {'description': 'Factorize categorical to ints.',
                 'prior_features': {'categorical': 1.0}},
 'CreateLagFeature': {'description': 'Lag features per group.', 'prior_features': {'time_target': 1.0}},
 'ReduceDimension': {'description': 'Supervised LDA projection.',
               'prior_features': {'target_classes': 0.4}},
 'TransformPower': {'description': 'Log transform numeric cols.',
           'prior_features': {'numeric': 0.3, 'skew': 0.5}},
 'CustomClean': {'description': 'Replace sentinel values via predicates.',
               'prior_features': {'sentinel': 5.0}},
 'ScaleFeature': {'description': 'Divide each col by max(|x|).', 'prior_features': {'numeric': 0.3}},
 'JoinTable': {'description': '1:1 / 1:0 join with aux table.', 'prior_features': {'aux': 1.0}},
 'ScaleFeature': {'description': 'Min-max scale to [0,1].', 'prior_features': {'numeric': 0.4}},
 'SampleNegative': {'description': 'Sample negatives per positive.',
                      'prior_features': {'const': 0.5}},
 'CustomFE': {'description': 'User-provided feature engineering callback.',
              'prior_features': {'const': 0.1}},
 'OneHotEncode': {'description': 'Expand categorical to dummies.',
                  'prior_features': {'categorical': 0.5}},
 'OrdinalEncode': {'description': 'Ordinal encode by user-given order.',
                   'prior_features': {'categorical': 0.3}},
 'ParseDate': {'description': 'Parse YYMMDD integer dates.',
                  'prior_features': {'int_date': 2.0}},
 'ReduceDimension': {'description': 'Linear PCA reduction.', 'prior_features': {'many_numeric': 0.5}},
 'CreatePolynomialFeature': {'description': 'Polynomial / interaction features.',
                        'prior_features': {'numeric': 0.2}},
 'TransformPower': {'description': 'Power transform for skew removal.',
                    'prior_features': {'numeric': 0.3, 'skew': 0.5}},
 'TransformPower': {'description': 'Quantile transform to uniform/normal.',
                       'prior_features': {'numeric': 0.3, 'skew': 0.5}},
 'JoinTable': {'description': 'One-stop rec join with user/item side tables.',
             'prior_features': {'const': 1.0}},
 'RenameColumn': {'description': 'Rename columns.', 'prior_features': {'const': 0.05}},
 'CustomClean': {'description': 'Regex replace inside string columns.',
                 'prior_features': {'text': 1.0}},
 'ResampleTimeSeries': {'description': 'Resample to time bucket and aggregate.',
                       'prior_features': {'time': 0.5}},
 'SelectFeature': {'description': 'Recursive feature elimination.', 'prior_features': {'numeric': 0.3}},
 'ScaleFeature': {'description': 'Robust scale (median/IQR).',
                 'prior_features': {'numeric': 0.4, 'outlier': 0.5}},
 'CreateRollingFeature': {'description': 'Rolling-window aggregates.',
                      'prior_features': {'time_target': 1.0}},
 'SelectFeature': {'description': 'Univariate feature selection.',
                 'prior_features': {'numeric': 0.3}},
 'SortRows': {'description': 'Sort rows by columns.', 'prior_features': {'const': 0.05}},
 'ScaleFeature': {'description': 'Z-score standardize.', 'prior_features': {'numeric': 0.5}},
 'ParseDate': {'description': 'Cast string column to datetime64.',
                         'prior_features': {'time': 1.0}},
 'TargetEncode': {'description': 'Smoothed target mean encoding.',
                    'prior_features': {'categorical': 0.4}},
 'Undersample': {'description': 'Drop majority of Tomek pairs.',
                'prior_features': {'imbalance': 0.3}},
 'ReduceDimension': {'description': 'UMAP non-linear embedding.',
                'prior_features': {'many_numeric': 0.1}},
 'SelectFeature': {'description': 'Drop low-variance numeric cols.',
                       'prior_features': {'numeric': 0.2}}}


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
        searchable=base.searchable,
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
