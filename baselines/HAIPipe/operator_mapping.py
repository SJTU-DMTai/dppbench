"""Map HAIPipe logical components to DPPBench operator categories."""
from __future__ import annotations

from baselines.common.op_spec import OpCategory


# HAIPipe's original AI-pipeline is component based.  DPPBench keeps the
# shared operator space, so each component is represented as a category group.
COMPONENT_CATEGORY_GROUPS: list[tuple[str, tuple[OpCategory, ...]]] = [
    (
        "Integration",
        (
            OpCategory.SCHEMA,
            OpCategory.JOIN,
            OpCategory.RESHAPE_LONGWIDE,
        ),
    ),
    (
        "Cleaning",
        (
            OpCategory.DATETIME_PARSE,
            OpCategory.CLEAN_VALUE,
            OpCategory.ERROR_DETECTION,
            OpCategory.DEDUPLICATE,
            OpCategory.OUTLIER,
            OpCategory.MISSING_VALUE,
        ),
    ),
    (
        "Encoder",
        (OpCategory.ENCODING,),
    ),
    (
        "FeaturePreprocessing",
        (
            OpCategory.NORMALIZATION,
            OpCategory.SCALING,
            OpCategory.DISTRIBUTION_RESHAPE,
            OpCategory.DISCRETIZATION,
        ),
    ),
    (
        "FeatureEngine",
        (
            OpCategory.GROUP_AGG,
            OpCategory.FEATURE_GEN,
            OpCategory.FEATURE_TIME,
            OpCategory.FEATURE_REDUCTION,
        ),
    ),
    (
        "FeatureSelection",
        (
            OpCategory.FEATURE_SELECTION,
            OpCategory.FILTER_COL,
            OpCategory.FILTER_ROW,
            OpCategory.IMBALANCE,
            OpCategory.AUGMENT,
        ),
    ),
]


__all__ = ["COMPONENT_CATEGORY_GROUPS"]
