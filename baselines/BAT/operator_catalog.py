"""BAT operator catalog.

Re-exports the shared authoritative operator catalog so that BAT operates
on all dppbench operators (cleaning, encoding, feature gen, sequence,
split, sampling, rec-specific JoinTable/CreateSequence/...). This satisfies
the requirement that BAT's operator library cover the full
``dppbench/operators/`` set and support both tabular and recommendation
tasks.

In addition, this module exposes a *function-family* description used by
BAT's ``IdentifyColumnFunctionsAction`` prompt, which maps the BAT-paper
function families (join / union / groupby / pivot / unpivot / rename /
arithmetic / dateformat / add-drop columns) onto dppbench's
:class:`OpCategory` taxonomy.
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
    """Render the operator whitelist as a markdown bullet list, grouped by
    category. Intended for direct prompt injection into BAT's
    ``TransformationAction`` prompt.
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
            params = list(spec.default_params.keys())
            params_str = ", ".join(params) if params else "(no params)"
            lines.append(
                f"- **{op_name}** {tag} task={spec.task_type} "
                f"params=[{params_str}]"
            )
        lines.append("")
    return "\n".join(lines).strip()


# BAT paper function families -> dppbench OpCategory groups.
FUNCTION_FAMILY_MAP: dict[str, list[OpCategory]] = {
    "join": [OpCategory.JOIN],
    "groupby_aggregate": [OpCategory.GROUP_AGG],
    "pivot_unpivot": [OpCategory.RESHAPE_PIVOT, OpCategory.RESHAPE_LONGWIDE],
    "string_reshape": [OpCategory.RESHAPE_STRING],
    "rename_or_drop_columns": [OpCategory.SCHEMA, OpCategory.FILTER_COL],
    "arithmetic_or_feature": [OpCategory.FEATURE_GEN, OpCategory.FEATURE_TIME],
    "datetime_format": [OpCategory.DATETIME_PARSE],
    "encoding": [OpCategory.ENCODING],
    "scale_or_normalize": [OpCategory.NORMALIZATION, OpCategory.SCALING,
                            OpCategory.DISTRIBUTION_RESHAPE],
    "missing_value": [OpCategory.MISSING_VALUE],
    "outlier": [OpCategory.OUTLIER],
    "deduplicate": [OpCategory.DEDUPLICATE],
    "discretize": [OpCategory.DISCRETIZATION],
    "feature_selection": [OpCategory.FEATURE_SELECTION,
                           OpCategory.FEATURE_REDUCTION],
    "filter_rows": [OpCategory.FILTER_ROW],
    "imbalance_or_augment": [OpCategory.IMBALANCE, OpCategory.AUGMENT],
    "sequence_modeling": [OpCategory.SEQUENCE],
    "split_or_sample": [OpCategory.SAMPLING],
}


def function_family_descriptions(task_type: str) -> str:
    """Render which function families are populated for a given task type
    and which operators belong to each family. Used by the
    ``IdentifyColumnFunctionsAction`` prompt to guide the LLM's choice of
    column-level transformations.
    """
    if task_type not in ("tabular", "rec"):
        raise ValueError(f"Unknown task_type {task_type}")
    valid_ops = set(operators_for_task(task_type))
    lines: list[str] = []
    for family, cats in FUNCTION_FAMILY_MAP.items():
        ops_in_family: list[str] = []
        for cat in cats:
            ops_in_family.extend(
                op for op in operators_by_category(task_type).get(cat, [])
                if op in valid_ops
            )
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
