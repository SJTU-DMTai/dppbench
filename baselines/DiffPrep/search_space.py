"""DiffPrep continuous relaxed pipeline.

The original DiffPrep keeps architecture logits per transformation family and
per feature.  This adaptation keeps the same feature-wise relaxation for soft
operators while projecting the final choice back to DPPBench's shared YAML
pipeline representation.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .slot_planner import IDENTITY, Slot
from .soft_ops import apply_soft_op, has_soft_impl, softmix


def sinkhorn(logits: torch.Tensor, n_iter: int = 20, tau: float = 1.0) -> torch.Tensor:
    """Convert ``[..., s, s]`` logits into a doubly-stochastic matrix."""
    log_alpha = logits / max(tau, 1e-3)
    for _ in range(n_iter):
        log_alpha = log_alpha - log_alpha.logsumexp(dim=-1, keepdim=True)
        log_alpha = log_alpha - log_alpha.logsumexp(dim=-2, keepdim=True)
    return log_alpha.exp()


class ContinuousPipeline(nn.Module):
    """Feature-wise DiffPrep architecture parameters.

    ``tau`` has shape ``[slots, features, max_candidates]``.  For a soft slot,
    each feature can choose a different candidate transformation, matching the
    original DiffPrep design.  For hard/structural slots we cannot apply the
    actual dataframe operator in tensor space; we keep a straight-through gate
    so those logits remain optimisable, and the selected operator is executed by
    the final common evaluator after discretisation.
    """

    def __init__(
        self,
        slots: List[Slot],
        flex: bool = False,
        init_scale: float = 1e-3,
        n_features: int | None = None,
        feature_names: List[str] | None = None,
    ):
        super().__init__()
        self.slots = slots
        self.s = len(slots)
        self.n_features = max(int(n_features or 1), 1)
        self.feature_names = list(feature_names or [])
        self.m_max = max((slot.n_candidates for slot in slots), default=1)
        self.flex = bool(flex) and self.s > 1
        # Official Flex keeps the initial missing-value stage outside the
        # learned permutation. DPPBench can have several mandatory stages, so
        # all mandatory slots remain fixed and only optional slots are ordered.
        self.reorderable = [i for i, slot in enumerate(slots) if not slot.mandatory]

        tau = torch.randn(self.s, self.n_features, self.m_max) * init_scale
        # Official DiffPrep initializes its default transform with probability
        # 0.5 and distributes the remaining mass over other candidates.  In the
        # shared space Identity is the default for optional slots.
        for i, slot in enumerate(slots):
            if slot.n_candidates <= 1:
                continue
            default_idx = (
                slot.candidates.index(IDENTITY)
                if IDENTITY in slot.candidates
                else 0
            )
            probs = torch.full(
                (slot.n_candidates,), 0.5 / (slot.n_candidates - 1)
            )
            probs[default_idx] = 0.5
            tau[i, :, : slot.n_candidates] = probs.log()
        self.register_buffer("_cand_mask", self._build_cand_mask())
        self.tau = nn.Parameter(tau)

        if self.flex and len(self.reorderable) > 1:
            r = len(self.reorderable)
            self.theta = nn.Parameter(
                torch.randn(self.n_features, r, r) * init_scale
            )
        else:
            self.register_parameter("theta", None)

        self.register_buffer("_forced_index", self._build_forced_index())

    def _build_cand_mask(self) -> torch.Tensor:
        mask = torch.full((self.s, 1, self.m_max), float("-inf"))
        for i, slot in enumerate(self.slots):
            mask[i, :, : slot.n_candidates] = 0.0
        return mask

    def _build_forced_index(self) -> torch.Tensor:
        idx = torch.full((self.s,), -1, dtype=torch.long)
        for i, slot in enumerate(self.slots):
            if slot.mandatory and slot.forced_op is not None:
                if slot.forced_op in slot.candidates:
                    idx[i] = slot.candidates.index(slot.forced_op)
        return idx

    def beta(self) -> torch.Tensor:
        """Return ``[slots, features, max_candidates]`` softmax weights."""
        logits = self.tau + self._cand_mask
        beta = F.softmax(logits, dim=-1)
        forced = self._forced_index
        if (forced >= 0).any():
            beta = beta.clone()
            for i, j in enumerate(forced.tolist()):
                if j >= 0:
                    beta[i].zero_()
                    beta[i, :, j] = 1.0
        return beta

    def alpha(self) -> torch.Tensor:
        """Return ``[features, slots, slots]`` order weights in flex mode."""
        if not self.flex:
            eye = torch.eye(self.s, device=self.tau.device)
            return eye.unsqueeze(0).expand(self.n_features, self.s, self.s)
        full = torch.eye(self.s, device=self.tau.device).unsqueeze(0).expand(
            self.n_features, self.s, self.s
        ).clone()
        if self.theta is None:
            return full
        learned = sinkhorn(self.theta)
        idx = torch.tensor(self.reorderable, device=self.tau.device)
        full[:, idx[:, None], idx[None, :]] = learned
        return full

    def _apply_slot(
        self,
        slot_idx: int,
        slot: Slot,
        out: torch.Tensor,
        mask: torch.Tensor,
        beta: torch.Tensor,
        gumbel_tau: float,
        hard_sample: bool,
    ) -> torch.Tensor:
        beta_row = beta[slot_idx, :, : slot.n_candidates]

        if slot.kind == "soft":
            outputs: list[torch.Tensor] = []
            for op_name in slot.candidates:
                if op_name == IDENTITY or not has_soft_impl(op_name):
                    outputs.append(out)
                else:
                    # Official DiffPrep evaluates the real transformation in
                    # the forward pass and supplies dTF/dX through centered
                    # numerical differentiation. Preserve that straight-
                    # through construction for the shared tensor adapters.
                    eps = 1e-6
                    value = apply_soft_op(op_name, out, mask)
                    value_pos = apply_soft_op(op_name, out.detach() + eps, mask)
                    value_neg = apply_soft_op(op_name, out.detach() - eps, mask)
                    grad = ((value_pos - value_neg) / (2.0 * eps)).detach()
                    outputs.append(
                        value.detach() + grad * out - (grad * out).detach()
                    )
            return softmix(beta_row, outputs)

        if hard_sample and self.training:
            logits = self.tau[slot_idx, :, : slot.n_candidates]
            weights = F.gumbel_softmax(logits, tau=gumbel_tau, hard=True, dim=-1)
        else:
            weights = beta_row

        idx = torch.arange(slot.n_candidates, device=out.device, dtype=out.dtype)
        idx = idx - idx.mean()
        gate = 1.0 + 1e-3 * torch.sum(weights.to(out.dtype) * idx.view(1, -1), dim=-1)
        return out * gate.view(1, -1)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        gumbel_tau: float = 1.0,
        hard_sample: bool = True,
    ) -> torch.Tensor:
        beta = self.beta()
        out = x

        if not self.flex:
            for i, slot in enumerate(self.slots):
                out = self._apply_slot(
                    i, slot, out, mask, beta, gumbel_tau, hard_sample
                )
            return out

        alpha = self.alpha()
        for pos in range(self.s):
            slot_outputs = [
                self._apply_slot(j, slot, out, mask, beta, gumbel_tau, hard_sample)
                for j, slot in enumerate(self.slots)
            ]
            out = sum(
                alpha[:, pos, j].to(out.dtype).view(1, -1) * slot_outputs[j]
                for j in range(self.s)
            )
        return out


__all__ = ["ContinuousPipeline", "sinkhorn"]
