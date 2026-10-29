"""DiffPrep operator catalog.

Re-exports the shared base catalog from ``baselines.common.op_spec``
and annotates every operator with a ``slot_kind`` flag:

* ``"soft"``  -- feature-wise + tensor-friendly. Eligible for the DARTS
  ``x_i = sum_j beta_ij f_ij(x_{i-1})`` mixture in :mod:`baselines.DiffPrep.soft_ops`.

* ``"hard"``  -- structural / non-feature-wise (joins, splits, sequence build,
  outlier row removal, etc.). These have no faithful tensor forward here.
  Optional hard choices are excluded; required structural adapters are fixed
  and replayed only by the final dataframe evaluator.

The classification follows the plan in
``.trae/documents/diffprep_baseline_implementation_plan.md`` (section 3.2)
and has been extended to cover the current shared catalog.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Dict

from baselines.common.op_spec import (
    CATALOG as _BASE_CATALOG,
    OpCategory,
    OpSpec,
    is_mandatory,
    needs_context,
    valid_targets_for,
)
from baselines.common.utils import get_available_ops


# ---------------------------------------------------------------------------
# 1. Re-export the shared operator catalog (deep-copied so this adapter
#    can add DiffPrep-specific metadata without mutating the base catalog).
# ---------------------------------------------------------------------------
CATALOG: Dict[str, OpSpec] = {name: deepcopy(spec) for name, spec in _BASE_CATALOG.items()}


# ---------------------------------------------------------------------------
# 2. Slot-kind annotation. SOFT operators are feature-wise + tensor-friendly,
#    HARD operators are structural / non-feature-wise.
# ---------------------------------------------------------------------------
SLOT_KIND: Dict[str, str] = {
    # ---- soft: mostly column-wise or tensor-friendly transforms ----
    "AlignSchema": "soft",
    "RenameColumn": "soft",
    "DropColumns": "hard",
    "CastType": "soft",
    "ParseDate": "soft",
    "ParseNumber": "soft",
    "SplitColumn": "soft",
    "CustomTransform": "soft",
    "HandleMV": "soft",
    "CorrectLabel": "soft",
    "CorrectTypo": "soft",
    "OneHotEncode": "soft",
    "OrdinalEncode": "soft",
    "HashEncode": "soft",
    "LabelEncode": "soft",
    "TargetEncode": "soft",
    "ScaleFeature": "soft",
    "TransformPower": "soft",
    "DiscretizeFeature": "soft",
    "ClipOutlier": "soft",
    "CustomProcess": "soft",
    "FrequencyEncode": "soft",
    "CreatePolynomialFeature": "soft",
    "CrossFeature": "soft",
    "AggregateGroupFeature": "soft",
    "ExtractDateTimeFeature": "hard",
    "SelectFeature": "soft",
    "ReduceDimension": "soft",
    "ExtractTextFeature": "soft",
    "ExtractTextEmbedding": "soft",
    "ExtractGraphFeature": "soft",

    # ---- hard: structural, row-count-changing, or sequence/group operations ----
    "JoinTable": "hard",
    "ConcatTable": "hard",
    "SortRows": "hard",
    "HandleOutlier": "hard",
    "HandleError": "hard",
    "HandleNonIID": "hard",
    "ReweightUPG": "hard",
    "Deduplicate": "hard",
    "CustomClean": "hard",
    "FilterSample": "hard",
    "SampleNegative": "hard",
    "FilterKCore": "hard",
    "Undersample": "hard",
    "Oversample": "hard",
    "AugmentMixup": "hard",
    "AugmentNoise": "hard",
    "CreateFeature": "hard",
    "CreateLagFeature": "hard",
    "CreateRollingFeature": "hard",
    "CreateSequence": "hard",
    "TruncateSequence": "hard",
    "CustomFE": "hard",
}


def slot_kind_of(op_name: str) -> str:
    """Return ``"soft"`` or ``"hard"`` for the given operator."""
    return SLOT_KIND.get(op_name, "hard")


# ---------------------------------------------------------------------------
# 3. Helpers (re-implemented over the *DiffPrep* CATALOG, not the SAGA one).
# ---------------------------------------------------------------------------
def operators_for_task(task_type: str, include_custom: bool = False) -> list[str]:
    return [
        spec.op_name
        for spec in get_available_ops(task_type, include_custom=include_custom)
    ]


def operators_by_category(task_type: str, include_custom: bool = False) -> dict[OpCategory, list[str]]:
    out: dict[OpCategory, list[str]] = {}
    for name in operators_for_task(task_type, include_custom=include_custom):
        out.setdefault(CATALOG[name].category, []).append(name)
    return out


__all__ = [
    "CATALOG",
    "OpCategory",
    "OpSpec",
    "SLOT_KIND",
    "slot_kind_of",
    "is_mandatory",
    "needs_context",
    "operators_for_task",
    "operators_by_category",
    "valid_targets_for",
]
