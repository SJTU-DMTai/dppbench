"""DeepPrep operator catalog rendered from shared common metadata."""
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
from .operator_adapter import ORIGINAL_OPERATOR_ALIASES


def operators_for_task(task_type: str, include_custom: bool = False) -> list[str]:
    return [
        spec.op_name
        for spec in get_available_ops(task_type, include_custom=include_custom)
    ]


def _param_space_text(spec: OpSpec) -> str:
    if not spec.param_space:
        return "{}"
    parts = []
    for key, value in spec.param_space.items():
        parts.append(f"{key}={value}")
    return "{" + ", ".join(parts) + "}"


def format_op_descriptions(task_type: str) -> str:
    """Render common operator metadata for prompt injection.

    The prompt intentionally exposes required params, allowed values, category,
    targets and descriptions so the LLM is less likely to invent parameter
    names or violate the shared operator contract.
    """
    if task_type not in ("tabular", "rec"):
        raise ValueError(f"Unknown task_type {task_type}")
    by_cat: dict[OpCategory, list[OpSpec]] = {}
    for spec in get_available_ops(task_type):
        by_cat.setdefault(spec.category, []).append(spec)

    lines: list[str] = ["# Operator categories (grouping only; no required order)"]
    lines.append(", ".join(cat.value for cat in get_category_order(task_type)))
    lines.append("")
    lines.append("# Released-checkpoint aliases accepted by the adapter")
    lines.extend(
        f"- {source} -> {target}"
        for source, target in sorted(ORIGINAL_OPERATOR_ALIASES.items())
    )
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
            if len(desc) > 240:
                desc = desc[:237] + "..."
            lines.append(
                f"- **{spec.op_name}** {tag} task={spec.task_type} "
                f"targets=[{targets}] required=[{required}] defaults=[{defaults}] "
                f"param_space={_param_space_text(spec)}"
            )
            if desc:
                lines.append(f"  description: {desc}")
        lines.append("")
    return "\n".join(lines).strip()


__all__ = [
    "CATALOG",
    "OpCategory",
    "OpSpec",
    "operators_by_category",
    "operators_for_task",
    "format_op_descriptions",
]
