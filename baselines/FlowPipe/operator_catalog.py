"""FlowPipe view of the shared DPPBench operator space."""

from baselines.common.op_spec import CATALOG, OpCategory, OpSpec, operators_by_category
from baselines.common.utils import get_available_ops


def operators_for_task(task_type: str) -> list[str]:
    return [spec.op_name for spec in get_available_ops(task_type)]


__all__ = [
    "CATALOG", "OpCategory", "OpSpec", "operators_by_category",
    "operators_for_task",
]
