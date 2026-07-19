"""Map HAIPipe logical components to DPPBench operator categories."""
from __future__ import annotations

from baselines.common.op_spec import OpCategory


# HAIPipe's original AI-pipeline is component based.  DPPBench keeps the
# shared operator space, so each component is represented as a category group.
COMPONENT_CATEGORY_GROUPS: list[tuple[str, tuple[OpCategory, ...]]] = [
    ("Integration", (OpCategory.INTEGRATION,)),
    ("Cleaning", (OpCategory.CLEANING,)),
    ("Preprocessing", (OpCategory.PREPROCESSING,)),
    ("FeatureEngineering", (OpCategory.FEATURE_ENGINEERING,)),
]


__all__ = ["COMPONENT_CATEGORY_GROUPS"]
