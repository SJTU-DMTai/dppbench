"""Conditional GFlowNet policy from the official FlowPipe implementation."""
from __future__ import annotations

import torch
from torch import nn


class FiLMLayer(nn.Module):
    """Official affine FiLM layer with identity initialization."""

    def __init__(self, input_dim: int, hidden_dim: int, ctx_dim: int) -> None:
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)
        self.ctx_proj = nn.Linear(ctx_dim, hidden_dim * 2)
        with torch.no_grad():
            self.ctx_proj.weight.zero_()
            self.ctx_proj.bias.zero_()
            self.ctx_proj.bias[:hidden_dim].fill_(1.0)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        x = self.layer(x)
        gamma, beta = torch.chunk(self.ctx_proj(ctx), 2, dim=-1)
        return x * gamma + beta


class FlowNetwork(nn.Module):
    """State policy and context-conditioned partition-function estimator."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        ctx_dim: int = 4096,
    ) -> None:
        super().__init__()
        self.state_encoder = nn.Linear(state_dim, hidden_dim)
        self.film1 = FiLMLayer(hidden_dim, hidden_dim, ctx_dim)
        self.film2 = FiLMLayer(hidden_dim, hidden_dim, ctx_dim)
        self.activation = nn.LeakyReLU()
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.logz_mlp = nn.Sequential(
            nn.Linear(ctx_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, ctx: torch.Tensor):
        x = self.activation(self.state_encoder(state))
        x = self.activation(self.film1(x, ctx))
        x = self.activation(self.film2(x, ctx))
        return self.action_head(x), self.logz_mlp(ctx).squeeze(-1)


__all__ = ["FiLMLayer", "FlowNetwork"]
