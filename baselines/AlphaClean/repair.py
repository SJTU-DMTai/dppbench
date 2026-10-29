"""Repair = AlphaClean's unified intermediate representation.

In the original paper a *repair* is a row-level conditional assignment
``ca(pred, attr, v) = if pred(r): r[attr] = v``. We generalise this to "a
single :class:`PipelineStep`" so that structural operators with trivial
predicates ("apply to all rows") fit the same abstraction. This lets the
search treat ``JoinTable``, ``CreateSequence`` etc. as repairs
in the same way as ``HandleMV`` or ``Clip``.
"""
from __future__ import annotations

import random as _random
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from baselines.common.data_ctx import DataContext
from baselines.common.pipeline import (
    Pipeline,
    PipelineStep,
    assign_dag_structure,
)
from baselines.common.pipeline import repair as repair_pipeline

from baselines.common.utils import get_op

from .operator_catalog import OpCategory


_TARGET_VOCAB = ["both", "interaction", "user", "item", "train", "test"]


@dataclass
class Repair:
    """A single AlphaClean repair = one PipelineStep."""

    op_name: str
    target: str = "both"
    params: dict = field(default_factory=dict)

    def to_pipeline_step(self) -> PipelineStep:
        return PipelineStep(op=self.op_name, target=self.target, params=dict(self.params))

    @classmethod
    def from_pipeline_step(cls, step: PipelineStep) -> "Repair":
        return cls(op_name=step.op, target=step.target, params=dict(step.params))

    # ------------------------------------------------------------------
    def featurize(self, op_index: dict, ctx: DataContext) -> np.ndarray:
        """Vectorise the repair the way the AlphaClean paper §6.3 describes:

        * one-hot of op_name across the catalog
        * one-hot of category across :class:`OpCategory`
        * one-hot of target across ``_TARGET_VOCAB``
        * a small bag of numeric params (count of params, max numeric value,
          number of column-name params, etc.) so the LR can pick up
          parameter-side regularities.
        """
        n_ops = len(op_index)
        n_cats = len(OpCategory)
        n_targets = len(_TARGET_VOCAB)
        feat = np.zeros(n_ops + n_cats + n_targets + 4, dtype=np.float32)

        # op one-hot
        if self.op_name in op_index:
            feat[op_index[self.op_name]] = 1.0

        # category one-hot
        try:
            spec = get_op(self.op_name)
        except KeyError:
            spec = None
        if spec is not None:
            cat_idx = list(OpCategory).index(spec.category)
            feat[n_ops + cat_idx] = 1.0

        # target one-hot
        if self.target in _TARGET_VOCAB:
            feat[n_ops + n_cats + _TARGET_VOCAB.index(self.target)] = 1.0

        # numeric param summary
        offset = n_ops + n_cats + n_targets
        n_params = len(self.params)
        n_numeric = 0
        max_numeric = 0.0
        n_list = 0
        for v in self.params.values():
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                n_numeric += 1
                max_numeric = max(max_numeric, float(abs(v)))
            elif isinstance(v, (list, tuple)):
                n_list += 1
        feat[offset + 0] = float(n_params)
        feat[offset + 1] = float(n_numeric)
        feat[offset + 2] = float(min(max_numeric, 1e6))
        feat[offset + 3] = float(n_list)
        return feat


# ---------------------------------------------------------------------------
# Composition: convert a list of Repair into a Pipeline + repair() it.
# ---------------------------------------------------------------------------
def repairs_to_pipeline(reps: List[Repair], ctx: DataContext,
                        rng: Optional[_random.Random] = None) -> Pipeline:
    # The common contract allows one instance per operator. A newly sampled
    # repair must update that instance, otherwise parameter/join-table search
    # is silently discarded by common repair()'s first-occurrence deduplication.
    steps = {}
    for r in reps:
        steps[r.op_name] = r.to_pipeline_step()
    rng = rng if rng is not None else _random.Random(0)
    pipe = Pipeline(steps=list(steps.values()))
    repair_pipeline(pipe, ctx.task_type, ctx, rng=rng)
    if ctx.task_type in ("tabular", "graph"):
        # Newly proposed transformations must precede the model-input tail.
        tail = [s for s in pipe.steps if s.op in {"LabelEncode", "HandleMV"}]
        pipe.steps = [s for s in pipe.steps if s.op not in {"LabelEncode", "HandleMV"}] + tail
    assign_dag_structure(pipe, ctx, rng)
    return pipe


__all__ = ["Repair", "repairs_to_pipeline"]
