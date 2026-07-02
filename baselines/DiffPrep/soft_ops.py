"""Differentiable soft-operator forwards used during DiffPrep search.

Each function takes a tensor batch ``x`` of shape ``[N, D]`` representing
numeric features (already imputed for NaN-mask handling) and returns a
transformed tensor of the *same shape*, so multiple candidate outputs can be
mixed via :func:`softmix`.

These tensor implementations are intentionally lightweight: they are only used
to provide gradient signal to ``tau`` (the DARTS-style architecture parameters)
during the inner-loop. The final discretized pipeline runs the *real* operators
from ``dppbench/ operators/``.
"""
from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------
def soft_identity(x: torch.Tensor) -> torch.Tensor:
    return x


# ---------------------------------------------------------------------------
# Missing-value imputation. We assume NaN entries have been preprocessed to
# zero with a separate ``mask`` tensor. The "fill" thus picks where to put
# the imputed value back.
# ---------------------------------------------------------------------------
def soft_fill_missing(x: torch.Tensor, mask: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    """Impute masked positions in ``x``.

    ``mask`` is 1 for missing positions, 0 otherwise. The imputation statistic
    is computed per-feature (column) over the *non-missing* entries.
    """
    valid = 1.0 - mask  # 1 = present
    denom = valid.sum(dim=0).clamp(min=1.0)

    if mode == "mean":
        col_stat = (x * valid).sum(dim=0) / denom
    elif mode == "median":
        # Differentiable approximation: torch.quantile w/o sort is non-diff;
        # use mean as a smooth surrogate -- DARTS only needs a search-time
        # signal, not a perfect estimator.
        col_stat = (x * valid).sum(dim=0) / denom
    elif mode == "mode":
        col_stat = torch.zeros(x.shape[1], device=x.device, dtype=x.dtype)
    elif mode == "constant":
        col_stat = torch.zeros(x.shape[1], device=x.device, dtype=x.dtype)
    else:
        col_stat = torch.zeros(x.shape[1], device=x.device, dtype=x.dtype)

    return x * valid + mask * col_stat.unsqueeze(0)


# ---------------------------------------------------------------------------
# Normalization variants
# ---------------------------------------------------------------------------
def soft_minmax_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mn = x.min(dim=0, keepdim=True).values
    mx = x.max(dim=0, keepdim=True).values
    return (x - mn) / (mx - mn + eps)


def soft_standard_scale(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mu = x.mean(dim=0, keepdim=True)
    sd = x.std(dim=0, keepdim=True)
    return (x - mu) / (sd + eps)


# ---------------------------------------------------------------------------
# Outlier / shape transforms
# ---------------------------------------------------------------------------
def soft_log(x: torch.Tensor, offset: float = 1.0) -> torch.Tensor:
    # symmetric log: sign(x) * log(1 + |x + offset|)
    return torch.sign(x) * torch.log1p(torch.abs(x + offset))


def soft_clip(x: torch.Tensor, k: float = 3.0) -> torch.Tensor:
    """Clip each column to its k*std band around the mean."""
    mu = x.mean(dim=0, keepdim=True)
    sd = x.std(dim=0, keepdim=True)
    lo, hi = mu - k * sd, mu + k * sd
    return torch.minimum(torch.maximum(x, lo), hi)


# ---------------------------------------------------------------------------
# Discretization-style transforms
# ---------------------------------------------------------------------------
def soft_bucketize(x: torch.Tensor, n_buckets: int = 8) -> torch.Tensor:
    """Quantile-based smooth bucketization. Returns a value in [0, 1]."""
    # Approximate via rank-normalisation -> n_buckets equal-mass bins.
    n = x.shape[0]
    if n <= 1:
        return torch.zeros_like(x)
    sorted_idx = x.argsort(dim=0)
    ranks = torch.zeros_like(x)
    arange = torch.arange(n, device=x.device, dtype=x.dtype).unsqueeze(1).expand_as(x)
    ranks.scatter_(0, sorted_idx, arange)
    normed = ranks / max(n - 1, 1)
    buckets = (normed * n_buckets).floor() / max(n_buckets - 1, 1)
    return buckets.clamp(0.0, 1.0)


def soft_label_encode(x: torch.Tensor) -> torch.Tensor:
    return soft_bucketize(x, n_buckets=32)


def soft_frequency_encode(x: torch.Tensor) -> torch.Tensor:
    # Approximate: empirical frequency replaced by its rank-normalised value.
    return soft_bucketize(x, n_buckets=16)


def soft_target_encode(x: torch.Tensor) -> torch.Tensor:
    return soft_standard_scale(x)


def soft_datetime_features(x: torch.Tensor) -> torch.Tensor:
    # Without a real datetime column we mimic the "extract subfields" effect
    # by emitting a sin/cos pair averaged into the column.
    return 0.5 * (torch.sin(x) + torch.cos(x))


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
SOFT_OP_FNS = {
    "Identity":        soft_identity,
    "HandleMV":     None,  # special-cased: needs mask. Handled in search_space.
    "Normalize":       soft_minmax_normalize,
    "ScaleFeature":   soft_standard_scale,
    "TransformPower":           soft_log,
    "Clip":            soft_clip,
    "DiscretizeFeature":       soft_bucketize,
    "LabelEncode":     soft_label_encode,
    "CustomProcess": soft_identity,
    "FrequencyEncode": soft_frequency_encode,
    "TargetEncode":  soft_target_encode,
    "ExtractDateTimeFeature": soft_datetime_features,
}


def has_soft_impl(op_name: str) -> bool:
    return op_name in SOFT_OP_FNS


def apply_soft_op(op_name: str, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Run a single soft-op forward."""
    if op_name == "HandleMV":
        return soft_fill_missing(x, mask, mode="mean")
    fn = SOFT_OP_FNS.get(op_name)
    if fn is None:
        return x
    return fn(x)


# ---------------------------------------------------------------------------
# Mixture: x_i = sum_j beta_ij * f_ij(x_{i-1})
# ---------------------------------------------------------------------------
def softmix(beta_row: torch.Tensor, outputs: Sequence[torch.Tensor]) -> torch.Tensor:
    """Mix a list of candidate outputs with the slot's softmax weights."""
    # outputs: list of [N, D]; beta_row: [m]
    stacked = torch.stack(list(outputs), dim=0)            # [m, N, D]
    weights = beta_row.view(-1, 1, 1)                       # [m, 1, 1]
    return (stacked * weights).sum(dim=0)


__all__ = [
    "soft_identity",
    "soft_fill_missing",
    "soft_minmax_normalize",
    "soft_standard_scale",
    "soft_log",
    "soft_clip",
    "soft_bucketize",
    "soft_label_encode",
    "soft_frequency_encode",
    "soft_target_encode",
    "soft_datetime_features",
    "SOFT_OP_FNS",
    "has_soft_impl",
    "apply_soft_op",
    "softmix",
]
