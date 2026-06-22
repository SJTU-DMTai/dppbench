"""Lightweight, fully-differentiable surrogate downstream models.

The surrogates participate only in the inner loop of DiffPrep's bilevel
optimisation. They receive features that have flown through the
:class:`ContinuousPipeline` and produce a binary classification logit.

* ``TabularSurrogate`` -- LR + 1-layer MLP head over numeric feature
  vectors.
* ``RecSurrogate`` -- user/item embedding + dot-product + MLP head; the
  feature tensor is concatenated to the embedding.
"""
from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Tabular surrogate
# ---------------------------------------------------------------------------
class TabularSurrogate(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 32):
        super().__init__()
        self.in_dim = in_dim
        self.body = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x).squeeze(-1)

    def reset_parameters(self) -> None:
        for m in self.body:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------------
# Rec surrogate (user x item + side features)
# ---------------------------------------------------------------------------
class RecSurrogate(nn.Module):
    def __init__(self, n_users: int, n_items: int, feat_dim: int = 0,
                 emb_dim: int = 16, hidden: int = 32):
        super().__init__()
        self.n_users = max(int(n_users), 1)
        self.n_items = max(int(n_items), 1)
        self.feat_dim = int(feat_dim)
        self.user_emb = nn.Embedding(self.n_users, emb_dim)
        self.item_emb = nn.Embedding(self.n_items, emb_dim)
        in_dim = 2 * emb_dim + feat_dim
        self.body = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor,
                feats: torch.Tensor) -> torch.Tensor:
        u = self.user_emb(user_ids.clamp(0, self.n_users - 1))
        i = self.item_emb(item_ids.clamp(0, self.n_items - 1))
        if feats is not None and feats.shape[-1] > 0:
            x = torch.cat([u, i, feats], dim=-1)
        else:
            x = torch.cat([u, i], dim=-1)
        return self.body(x).squeeze(-1)

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        for m in self.body:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)


__all__ = ["TabularSurrogate", "RecSurrogate"]
