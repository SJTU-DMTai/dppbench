"""BAT operator catalog rendered from shared DPPBench metadata.

The original BAT action sandbox operates over a fixed transformation DSL. In
DPPBench, BAT must use the shared operator space, so this module only renders
``baselines.common`` metadata for prompts and function-family hints; it does
not maintain a private operator catalog.
"""
from __future__ import annotations

from baselines.common.op_spec import (
    CATALOG,
    OpCategory,
    OpSpec,
    is_mandatory,
    operators_by_category,
    valid_targets_for,
)
from baselines.common.pipeline import get_category_order
from baselines.common.utils import get_available_ops


def operators_for_task(task_type: str, include_custom: bool = False) -> list[str]:
    return [
        spec.op_name
        for spec in get_available_ops(task_type, include_custom=include_custom)
    ]


def _param_space_text(spec: OpSpec) -> str:
    if not spec.param_space:
        return "{}"
    return "{" + ", ".join(
        f"{key}={value}" for key, value in spec.param_space.items()
    ) + "}"


def format_op_descriptions(task_type: str) -> str:
    """Render a rich operator whitelist for BAT prompts."""
    if task_type not in ("tabular", "rec"):
        raise ValueError(f"Unknown task_type {task_type}")
    by_cat: dict[OpCategory, list[OpSpec]] = {}
    for spec in get_available_ops(task_type):
        by_cat.setdefault(spec.category, []).append(spec)

    lines: list[str] = ["# Operator categories (grouping only; no required order)"]
    lines.append(", ".join(cat.value for cat in get_category_order(task_type)))
    lines.append("")
    lines.append("# Operators")
    for cat in get_category_order(task_type):
        specs = by_cat.get(cat, [])
        if not specs:
            continue
        lines.append(f"### {cat.value}")
        for spec in sorted(specs, key=lambda s: s.op_name):
            tag = "[mandatory]" if is_mandatory(spec.op_name, task_type) else "[optional]"
            required = ", ".join(spec.required_params) if spec.required_params else "none"
            defaults = ", ".join(spec.default_params.keys()) if spec.default_params else "none"
            targets = ", ".join(valid_targets_for(spec.op_name))
            desc = (spec.nl_description or "").strip().replace("\n", " ")
            if len(desc) > 220:
                desc = desc[:217] + "..."
            lines.append(
                f"- **{spec.op_name}** {tag} task={spec.task_type} "
                f"targets=[{targets}] required=[{required}] defaults=[{defaults}] "
                f"param_space={_param_space_text(spec)}"
            )
            if desc:
                lines.append(f"  description: {desc}")
        lines.append("")
    return "\n".join(lines).strip()


# BAT paper function families remain operator-level because the shared
# categories now mirror broad implementation directories.
FUNCTION_FAMILY_MAP: dict[str, list[str]] = {
    "join": ["JoinTable", "ConcatTable"],
    "groupby_aggregate": ["AggregateGroupFeature"],
    "string_reshape": ["SplitColumn"],
    "rename_or_drop_columns": ["RenameColumn", "DropColumns"],
    "arithmetic_or_feature": ["CreateFeature", "CrossFeature", "CreatePolynomialFeature"],
    "datetime_format": ["ParseDate", "ExtractDateTimeFeature"],
    "encoding": ["OneHotEncode", "OrdinalEncode", "LabelEncode", "HashEncode", "TargetEncode"],
    "scale_or_normalize": ["ScaleFeature", "TransformPower"],
    "missing_value": ["HandleMV"],
    "outlier": ["HandleOutlier", "ClipOutlier"],
    "deduplicate": ["Deduplicate"],
    "discretize": ["DiscretizeFeature"],
    "feature_selection": ["SelectFeature", "ReduceDimension"],
    "filter_rows": ["FilterSample", "FilterKCore"],
    "imbalance_or_augment": ["Undersample", "Oversample", "AugmentMixup", "AugmentNoise"],
    "sequence_modeling": ["CreateSequence", "TruncateSequence"],
    "split_or_sample": ["SampleNegative"],
}


def function_family_descriptions(task_type: str) -> str:
    """Render populated BAT function families for the current task type."""
    if task_type not in ("tabular", "rec"):
        raise ValueError(f"Unknown task_type {task_type}")
    valid_ops = {spec.op_name for spec in get_available_ops(task_type)}
    lines: list[str] = []
    for family, op_names in FUNCTION_FAMILY_MAP.items():
        ops_in_family = [op for op in op_names if op in valid_ops]
        if not ops_in_family:
            continue
        ops_str = ", ".join(sorted(set(ops_in_family)))
        lines.append(f"- **{family}**: {ops_str}")
    return "\n".join(lines)


__all__ = [
    "CATALOG",
    "OpCategory",
    "OpSpec",
    "operators_by_category",
    "operators_for_task",
    "format_op_descriptions",
    "function_family_descriptions",
    "FUNCTION_FAMILY_MAP",
]
