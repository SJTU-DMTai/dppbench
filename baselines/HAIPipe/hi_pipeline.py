"""HI-pipeline construction for the DPPBench HAIPipe adapter."""
from __future__ import annotations

import random as _random

from baselines.common.data_ctx import DataContext
from baselines.common.pipeline import Pipeline, finalize_pipeline


SUPPORTED_HI_SOURCES = {"mandatory"}


def build_hi_pipeline(
    ctx: DataContext,
    rng: _random.Random,
    *,
    max_len: int | None = None,
    hi_source: str = "mandatory",
) -> Pipeline:
    """Build the human pipeline scaffold.

    Standard DPPBench evaluation starts every baseline from an empty pipeline,
    so the default and only benchmark-safe HI source is the mandatory scaffold
    produced by common repair/finalization.
    """
    if hi_source not in SUPPORTED_HI_SOURCES:
        raise ValueError(
            f"Unsupported hi_source={hi_source!r}; supported: {sorted(SUPPORTED_HI_SOURCES)}"
        )
    pipe, report = finalize_pipeline(Pipeline(), ctx, rng, max_len=max_len)
    if not report.valid:
        raise ValueError("Unable to build valid mandatory HI scaffold: " + "; ".join(report.errors))
    return pipe


__all__ = ["SUPPORTED_HI_SOURCES", "build_hi_pipeline"]
