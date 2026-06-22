"""DeepPrep operator catalog.

Re-exports the shared 58-operator catalog so the LLM agent has access to the
same operators living in ``dppbench/operators/`` (cleaning + encoding +
feature-gen + sequence + split + sampling + ...). This guarantees coverage
of both **tabular** and **recommendation** tasks.
"""
from __future__ import annotations

from baselines.common.operator_catalog import (
    CATALOG,
    OpCategory,
    OpSpec,
    operators_by_category,
    operators_for_task,
)


def format_op_descriptions(task_type: str) -> str:
    """Render operator metadata for the given task type as a markdown
    bullet list, suitable for direct prompt injection.
    """
    if task_type not in ("tabular", "rec"):
        raise ValueError(f"Unknown task_type {task_type}")
    by_cat = operators_by_category(task_type)
    lines: list[str] = []
    for cat, ops in by_cat.items():
        lines.append(f"### {cat.value}")
        for op_name in ops:
            spec = CATALOG[op_name]
            tag = "[mandatory]" if spec.mandatory else "[optional]"
            tt = spec.task_type
            params = list(spec.default_params.keys())
            params_str = ", ".join(params) if params else "(no params)"
            lines.append(
                f"- **{op_name}** {tag} task={tt} params=[{params_str}]"
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
