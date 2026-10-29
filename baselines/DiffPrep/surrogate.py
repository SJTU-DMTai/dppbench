"""Lightweight, fully-differentiable surrogate downstream models.

The surrogates participate only in the inner loop of DiffPrep's bilevel
optimisation. They receive features that have flown through the
:class:`ContinuousPipeline` and produce a binary classification logit.

* ``TabularSurrogate`` -- a single linear classification/regression head
  over numeric feature vectors.
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
    def __init__(self, in_dim: int, n_classes: int, regression: bool = False):
        super().__init__()
        self.in_dim = in_dim
        self.n_classes = 1 if regression else max(int(n_classes), 2)
        # The official implementation uses multinomial logistic regression.
        self.body = nn.Linear(in_dim, self.n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.body.weight)
        nn.init.zeros_(self.body.bias)


# ---------------------------------------------------------------------------
# Rec surrogate (user x item + side features)
# ---------------------------------------------------------------------------
class RecSurrogate(nn.Module):
    def __init__(self, n_users: int, n_items: int, feat_dim: int = 0,
                 emb_dim: int = 16, hidden: int = 32, history: bool = False,
                 list_feature_specs=None):
        super().__init__()
        self.n_users = max(int(n_users), 1)
        self.n_items = max(int(n_items), 1)
        self.feat_dim = int(feat_dim)
        self.user_emb = nn.Embedding(self.n_users, emb_dim)
        self.item_emb = nn.Embedding(self.n_items, emb_dim)
        self.use_history = history
        self.list_feature_specs = dict(list_feature_specs or {})
        self.list_embeddings = nn.ModuleDict({str(i): nn.Embedding(spec['vocabulary_size'], emb_dim)
            for i, spec in enumerate(self.list_feature_specs.values()) if spec['embedding_source'] == 'own'})
        in_dim = 2 * emb_dim + feat_dim + (emb_dim if history else 0) + len(self.list_feature_specs)*emb_dim
        self.body = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor,
                feats: torch.Tensor, history_ids=None, history_lengths=None,
                history_mask=None, fixed_lists=None) -> torch.Tensor:
        u = self.user_emb(user_ids.clamp(0, self.n_users - 1))
        i = self.item_emb(item_ids.clamp(0, self.n_items - 1))
        if feats is not None and feats.shape[-1] > 0:
            x = torch.cat([u, i, feats], dim=-1)
        else:
            x = torch.cat([u, i], dim=-1)
        if self.use_history:
            if history_ids is None or history_lengths is None:
                raise ValueError('Recommendation proxy requires item history and explicit lengths')
            pooled = self._pool(self.item_emb, history_ids, history_lengths, history_mask)
            x = torch.cat([x, pooled], dim=-1)
        for index, (name, spec) in enumerate(self.list_feature_specs.items()):
            if fixed_lists is None or name not in fixed_lists:
                raise ValueError(f'Recommendation proxy requires fixed list input {name}')
            values = fixed_lists[name]
            emb = (self.user_emb if spec['embedding_source'] == 'user' else self.item_emb
                   if spec['embedding_source'] == 'item' else self.list_embeddings[str(index)])
            pooled = self._pool(emb, values['ids'], values['lengths'], values['mask'])
            x = torch.cat([x, pooled], dim=-1)
        return self.body(x).squeeze(-1)

    @staticmethod
    def _pool(embedding, ids, lengths, mask=None):
        valid = torch.arange(ids.size(1), device=ids.device).unsqueeze(0) < lengths.unsqueeze(1)
        if mask is not None and not torch.equal(mask, valid):
            raise ValueError('List mask disagrees with explicit lengths')
        # Token 0 inside the length is an OOV event, not padding.
        embedded = embedding(ids)
        return (embedded * valid.unsqueeze(-1)).sum(1) / lengths.clamp(min=1).unsqueeze(1)

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        for emb in self.list_embeddings.values():
            nn.init.xavier_uniform_(emb.weight)
        for m in self.body:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)


__all__ = ["TabularSurrogate", "RecSurrogate"]
