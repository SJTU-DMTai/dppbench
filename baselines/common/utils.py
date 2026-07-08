"""Convenience APIs exposed to baseline implementations."""
from __future__ import annotations

from typing import Callable, Optional

from baselines.common.data_ctx import DataContext, infer_data_context
from baselines.common.data_loader import DATASET_REGISTRY, TaskDataLoader
from baselines.common.op_spec import OpSpec
from baselines.common.op_spec import get_avilable_ops as _get_avilable_ops
from baselines.common.op_spec import get_available_ops as _get_available_ops
from baselines.common.op_spec import get_op as _get_op


def get_op(op_name: str) -> OpSpec:
    """Return the OpSpec for one operator."""
    return _get_op(op_name)


def get_avilable_ops(task_type: str, include_custom: bool = False) -> list[OpSpec]:
    """Return all OpSpec objects available for a task type.

    The function name keeps the spelling used by the public docs.
    """
    return _get_avilable_ops(task_type, include_custom=include_custom)


def get_available_ops(task_type: str, include_custom: bool = False) -> list[OpSpec]:
    """Spelled alias for get_avilable_ops()."""
    return _get_available_ops(task_type, include_custom=include_custom)


def get_data(
    data_name: str,
    data_dir: Optional[str] = None,
    model_config_provider: Optional[Callable[[], dict]] = None,
):
    """Load a DPPBench task data object through the standard loader."""
    loader = TaskDataLoader(
        data_name,
        data_dir=data_dir,
        model_config_provider=model_config_provider,
    )
    return loader.load_data()


def get_data_ctx(data_name: str, data, task_type: str | None = None) -> DataContext:
    """Infer the DataContext for a loaded task data object."""
    resolved_task_type = task_type
    if resolved_task_type is None and data_name in DATASET_REGISTRY:
        resolved_task_type = DATASET_REGISTRY[data_name][0]
    return infer_data_context(data_name, data, resolved_task_type)


__all__ = [
    "get_available_ops",
    "get_avilable_ops",
    "get_data",
    "get_data_ctx",
    "get_op",
]
