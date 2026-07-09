"""DeepPrep operator catalog.

Re-exports the shared operator catalog so the LLM agent has access to the
same operators living in ``dppbench/operators/`` (cleaning + encoding +
feature-gen + sequence + split + sampling + ...). This guarantees coverage
of both **tabular** and **recommendation** tasks.
"""
from __future__ import annotations

from baselines.common.op_spec import (
    CATALOG,
    OpCategory,
    OpSpec,
    is_mandatory,
    operators_by_category,
)
from baselines.common.utils import get_available_ops


def operators_for_task(task_type: str, include_custom: bool = False) -> list[str]:
    return [
        spec.op_name
        for spec in get_available_ops(task_type, include_custom=include_custom)
    ]


def format_op_descriptions(task_type: str) -> str:
    """Render operator metadata for the given task type as a markdown
    bullet list, suitable for direct prompt injection.
    """
    if task_type not in ("tabular", "rec"):
        raise ValueError(f"Unknown task_type {task_type}")
    by_cat: dict[OpCategory, list[OpSpec]] = {}
    for spec in get_available_ops(task_type):
        by_cat.setdefault(spec.category, []).append(spec)
    lines: list[str] = []
    for cat, specs in by_cat.items():
        lines.append(f"### {cat.value}")
        for spec in specs:
            tag = "[mandatory]" if is_mandatory(spec.op_name, task_type) else "[optional]"
            params = list(spec.default_params.keys())
            params_str = ", ".join(params) if params else "(no params)"
            lines.append(
                f"- **{spec.op_name}** {tag} task={spec.task_type} params=[{params_str}]"
            )
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
