"""HI-pipeline construction for the DPPBench HAIPipe adapter."""
from __future__ import annotations

import os
import random as _random

from baselines.common.data_ctx import DataContext
from baselines.common.pipeline import Pipeline


SUPPORTED_HI_SOURCES = {"prepare_yaml"}


def build_hi_pipeline(
    ctx: DataContext,
    rng: _random.Random,
    *,
    task_dir: str,
    max_len: int | None = None,
    hi_source: str = "prepare_yaml",
) -> Pipeline:
    """Build the human pipeline scaffold.

    The user-confirmed HAIPipe protocol treats the task's human-authored
    ``prepare.yaml`` as the HI-pipeline.
    """
    if hi_source not in SUPPORTED_HI_SOURCES:
        raise ValueError(
            f"Unsupported hi_source={hi_source!r}; supported: {sorted(SUPPORTED_HI_SOURCES)}"
        )
    path = os.path.join(task_dir, "prepare.yaml")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"HAIPipe HI-pipeline does not exist: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        pipe = Pipeline.from_yaml(handle.read())
    if max_len is not None and len(pipe) > int(max_len):
        raise ValueError(
            f"prepare.yaml HI-pipeline length {len(pipe)} exceeds max_pipeline_len={max_len}"
        )
    from baselines.common.pipeline import check_pipeline
    report = check_pipeline(pipe, ctx)
    if not report.valid:
        raise ValueError("Invalid prepare.yaml HI-pipeline: " + "; ".join(report.errors))
    return pipe


__all__ = ["SUPPORTED_HI_SOURCES", "build_hi_pipeline"]
