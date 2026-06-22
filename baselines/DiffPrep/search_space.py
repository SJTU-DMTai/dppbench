"""DiffPrep continuous (relaxed) pipeline.

This module implements the differentiable search space described in the
DiffPrep paper, adapted for the dppbench operator zoo:

* ``tau``  -- ``[s, m_max]`` underlying parameters; ``beta = softmax(tau)`` per
  slot. ``m_max`` = maximum number of candidates across slots; we mask
  out-of-range candidates with ``-inf``.
* ``theta`` -- ``[s, s]`` permutation logits used by DiffPrep-Flex (off by
  default). Sinkhorn-normalised at forward.
* ``forward(x, mask)`` -- mixes soft slots via :func:`softmix` and keeps hard
  slots as identity (their effect is captured by argmax-only discretisation).

The hard slots' contribution to the validation loss is captured at the end
through a Gumbel-Softmax + Straight-Through Estimator on the slot's tau row,
so gradients still flow back to ``tau`` (see :class:`HardSlotSampler`).
"""
from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .slot_planner import IDENTITY, Slot
from .soft_ops import apply_soft_op, has_soft_impl, softmix


# ---------------------------------------------------------------------------
# Sinkhorn normalisation (for DiffPrep-Flex order learning)
# ---------------------------------------------------------------------------
def sinkhorn(logits: torch.Tensor, n_iter: int = 20, tau: float = 1.0) -> torch.Tensor:
    """Convert ``logits`` (s x s) into a doubly-stochastic matrix."""
    log_alpha = logits / max(tau, 1e-3)
    for _ in range(n_iter):
        log_alpha = log_alpha - log_alpha.logsumexp(dim=1, keepdim=True)
        log_alpha = log_alpha - log_alpha.logsumexp(dim=0, keepdim=True)
    return log_alpha.exp()


# ---------------------------------------------------------------------------
# ContinuousPipeline
# ---------------------------------------------------------------------------
class ContinuousPipeline(nn.Module):
    """Holds the per-slot architecture parameters tau (and optionally theta)."""

    def __init__(self, slots: List[Slot], flex: bool = False, init_scale: float = 1e-3):
        super().__init__()
        self.slots = slots
        self.s = len(slots)
        self.m_max = max((slot.n_candidates for slot in slots), default=1)
        self.flex = bool(flex) and self.s > 1

        # Padded tau matrix; out-of-range entries get -inf so softmax masks them.
        tau = torch.randn(self.s, self.m_max) * init_scale
        self.register_buffer("_cand_mask", self._build_cand_mask())
        self.tau = nn.Parameter(tau)

        if self.flex:
            self.theta = nn.Parameter(torch.randn(self.s, self.s) * init_scale)
        else:
            self.register_parameter("theta", None)

        # Mandatory slots get tau-row pinned to one-hot via a mask.
        self.register_buffer("_forced_index", self._build_forced_index())

    # ------------------------------------------------------------------
    def _build_cand_mask(self) -> torch.Tensor:
        mask = torch.full((self.s, self.m_max), float("-inf"))
        for i, slot in enumerate(self.slots):
            mask[i, : slot.n_candidates] = 0.0
        return mask

    def _build_forced_index(self) -> torch.Tensor:
        idx = torch.full((self.s,), -1, dtype=torch.long)
        for i, slot in enumerate(self.slots):
            if slot.mandatory and slot.forced_op is not None:
                if slot.forced_op in slot.candidates:
                    idx[i] = slot.candidates.index(slot.forced_op)
        return idx

    # ------------------------------------------------------------------
    def beta(self) -> torch.Tensor:
        """Return ``[s, m_max]`` softmax weights with masking + forcing."""
        logits = self.tau + self._cand_mask
        beta = F.softmax(logits, dim=-1)
        # Override mandatory rows with one-hot.
        forced = self._forced_index
        if (forced >= 0).any():
            one_hot = torch.zeros_like(beta)
            for i, j in enumerate(forced.tolist()):
                if j >= 0:
                    one_hot[i, j] = 1.0
                    beta = beta.clone()
                    beta[i] = one_hot[i]
        return beta

    def alpha(self) -> torch.Tensor:
        if not self.flex:
            return torch.eye(self.s, device=self.tau.device)
        return sinkhorn(self.theta)

    # ------------------------------------------------------------------
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
        beta_row = beta[slot_idx, : slot.n_candidates]

        if slot.kind == "soft":
            outputs: list[torch.Tensor] = []
            for op_name in slot.candidates:
                if op_name == IDENTITY or not has_soft_impl(op_name):
                    outputs.append(out)
                else:
                    outputs.append(apply_soft_op(op_name, out, mask))
            return softmix(beta_row, outputs)

        if hard_sample and self.training:
            logits = self.tau[slot_idx, : slot.n_candidates]
            weights = F.gumbel_softmax(logits, tau=gumbel_tau, hard=True)
        else:
            weights = beta_row

        # Structural operators are not tensor-differentiable in the surrogate.
        # Give their logits a tiny candidate-specific gate so the architecture
        # optimiser receives a non-constant signal instead of ``sum(weights)``.
        idx = torch.arange(
            slot.n_candidates, device=out.device, dtype=out.dtype
        )
        idx = idx - idx.mean()
        gate = 1.0 + 1e-3 * torch.sum(weights.to(out.dtype) * idx)
        return out * gate

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        gumbel_tau: float = 1.0,
        hard_sample: bool = True,
    ) -> torch.Tensor:
        """Forward through every slot.

        For *soft* slots we evaluate every candidate operator on ``x`` and
        mix them with the slot's beta row. For *hard* slots we sample one
        candidate via Gumbel-Softmax + Straight-Through; the candidate
        is treated as identity (since its real effect is structural and
        cannot be captured tensor-wise) so its sole contribution is to
        provide a gradient pathway back to ``tau``.
        """
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
                self._apply_slot(
                    j, slot, out, mask, beta, gumbel_tau, hard_sample
                )
                for j, slot in enumerate(self.slots)
            ]
            out = sum(
                alpha[pos, j].to(out.dtype) * slot_outputs[j]
                for j in range(self.s)
            )

        return out


__all__ = ["ContinuousPipeline", "sinkhorn"]
