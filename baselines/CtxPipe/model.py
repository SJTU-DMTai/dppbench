"""CtxPipe DQN architectures aligned with the upstream implementation."""
from __future__ import annotations

from enum import Enum, auto

import torch
from torch import nn

from .context import CONTEXT_DIM, MAX_COLUMNS, COLUMN_FEATURE_DIM


N_DIM_FIRST = 128
DATA_DIM = MAX_COLUMNS * COLUMN_FEATURE_DIM


class ForwardMode(Enum):
    CLOSED = auto()
    GATED = auto()
    OPEN = auto()


def _ctx_plugin(n_output: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(CONTEXT_DIM, N_DIM_FIRST),
        nn.LeakyReLU(),
        nn.Linear(N_DIM_FIRST, n_output),
        nn.Tanh(),
    )


class _ContextGateMixin:
    def _init_context(self, actions_dim: int, device: torch.device) -> None:
        mean = 3.0 / (actions_dim + N_DIM_FIRST)
        self.ctx_linear = _ctx_plugin(actions_dim)
        self.context_gate = nn.Parameter(torch.normal(
            mean=mean, std=mean / 3.0,
            size=(actions_dim + N_DIM_FIRST, actions_dim),
            device=device,
        ))
        bias_mean = 3.0 / actions_dim
        self.context_gate_bias = nn.Parameter(torch.normal(
            mean=bias_mean, std=bias_mean / 3.0,
            size=(actions_dim,), device=device,
        ))

    def _apply_network(self, x, ctx, mode):
        value = x
        for index, layer in enumerate(self.nn):
            value = layer(value)
            if index == 0:
                ctx_integration = self.ctx_linear(ctx)
                if mode == ForwardMode.CLOSED:
                    gate = torch.zeros_like(ctx_integration)
                elif mode == ForwardMode.OPEN:
                    gate = torch.ones_like(ctx_integration)
                elif mode == ForwardMode.GATED:
                    info = torch.cat([value, ctx_integration], dim=1)
                    gate = torch.sigmoid(
                        torch.nan_to_num(info @ self.context_gate + self.context_gate_bias)
                    )
                else:
                    raise ValueError(f"unknown CtxPipe forward mode: {mode}")
                ctx_integration = ctx_integration * gate
            if index == len(self.nn) - 1:
                ctx_integration = (ctx_integration - 1.0) * gate + 1.0
                value = value * ctx_integration
        return value


class DQN(nn.Module, _ContextGateMixin):
    def __init__(self, input_dim: int, actions_dim: int, device: str) -> None:
        super().__init__()
        self.nn = nn.Sequential(
            nn.Linear(input_dim, 128), nn.LeakyReLU(),
            nn.Linear(128, 128), nn.LeakyReLU(),
            nn.Linear(128, 64), nn.LeakyReLU(),
            nn.Linear(64, 32), nn.LeakyReLU(),
            nn.Linear(32, actions_dim), nn.Tanh(),
        )
        self._init_context(actions_dim, torch.device(device))

    def forward(self, x, ctx, mode: ForwardMode = ForwardMode.GATED):
        return self._apply_network(x, ctx, mode)


class RnnDQN(nn.Module, _ContextGateMixin):
    def __init__(
        self,
        actions_dim: int,
        primitive_vocab_size: int,
        sequence_length: int,
        logical_pipeline_count: int,
        device: str,
        predictor_count: int = 1,
        seq_embedding_dim: int = 96,
        seq_hidden_size: int = 96,
        seq_num_layers: int = 1,
        predictor_embedding_dim: int = 16,
        logical_embedding_dim: int = 8,
    ) -> None:
        super().__init__()
        self.sequence_length = int(sequence_length)
        self.seq_embedding = nn.Embedding(primitive_vocab_size, seq_embedding_dim)
        self.seq_lstm = nn.LSTM(
            input_size=seq_embedding_dim,
            hidden_size=seq_hidden_size,
            num_layers=seq_num_layers,
            batch_first=True,
        )
        self.predictor_embedding = nn.Embedding(predictor_count, predictor_embedding_dim)
        self.logical_embedding = nn.Embedding(logical_pipeline_count, logical_embedding_dim)
        input_dim = (
            DATA_DIM + seq_hidden_size * self.sequence_length
            + predictor_embedding_dim + logical_embedding_dim
        )
        self.nn = nn.Sequential(
            nn.Linear(input_dim, 128), nn.LeakyReLU(),
            nn.Linear(128, 128), nn.LeakyReLU(),
            nn.Linear(128, 64), nn.LeakyReLU(),
            nn.Linear(64, 32), nn.LeakyReLU(),
            nn.Linear(32, actions_dim), nn.Tanh(),
        )
        self._init_context(actions_dim, torch.device(device))

    def forward(self, x, ctx, mode: ForwardMode = ForwardMode.GATED):
        data = x[:, :DATA_DIM]
        seq_start = DATA_DIM
        seq = x[:, seq_start:seq_start + self.sequence_length].long()
        predictor = x[:, seq_start + self.sequence_length:seq_start + self.sequence_length + 1].long()
        logical = x[:, seq_start + self.sequence_length + 1:seq_start + self.sequence_length + 2].long()
        seq_hidden, _ = self.seq_lstm(self.seq_embedding(seq))
        combined = torch.cat([
            data,
            torch.flatten(seq_hidden, start_dim=1),
            torch.flatten(self.predictor_embedding(predictor), start_dim=1),
            torch.flatten(self.logical_embedding(logical), start_dim=1),
        ], dim=1)
        return self._apply_network(combined, ctx, mode)


__all__ = ["DATA_DIM", "DQN", "ForwardMode", "RnnDQN"]
