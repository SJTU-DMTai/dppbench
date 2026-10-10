"""Neural networks used by the DPPBench port of HAIPipe's AIPipe.

The topology follows the public HAIPipe implementation: a large feed-forward
network selects the logical pipeline and one LSTM-augmented Q network is used
for each physical component.  Only vocabulary sizes and the number of
components are adapted to DPPBench's shared operator space.
"""
from __future__ import annotations

import torch
from torch import nn


class LogicalDQN(nn.Module):
    def __init__(self, input_dim: int, action_dim: int) -> None:
        super().__init__()
        self.nn = nn.Sequential(
            nn.Linear(input_dim, 1907), nn.LeakyReLU(),
            nn.Linear(1907, 1907), nn.LeakyReLU(),
            nn.Linear(1907, 950), nn.LeakyReLU(),
            nn.Linear(950, 950), nn.LeakyReLU(),
            nn.Linear(950, 387), nn.LeakyReLU(),
            nn.Linear(387, 387), nn.LeakyReLU(),
            nn.Linear(387, 128), nn.LeakyReLU(),
            nn.Linear(128, 128), nn.LeakyReLU(),
            nn.Linear(128, 64), nn.LeakyReLU(),
            nn.Linear(64, 32), nn.LeakyReLU(),
            nn.Linear(32, action_dim), nn.Tanh(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.nn(state)


class PrimitiveRnnDQN(nn.Module):
    def __init__(
        self,
        *,
        data_dim: int,
        sequence_length: int,
        primitive_vocab_size: int,
        predictor_vocab_size: int,
        logical_pipeline_count: int,
        action_dim: int,
        seq_embedding_dim: int = 30,
        seq_hidden_size: int = 18,
        seq_num_layers: int = 2,
        predictor_embedding_dim: int = 10,
        logical_embedding_dim: int = 5,
    ) -> None:
        super().__init__()
        self.data_dim = int(data_dim)
        self.sequence_length = int(sequence_length)
        self.seq_embedding = nn.Embedding(primitive_vocab_size, seq_embedding_dim)
        self.seq_lstm = nn.LSTM(
            seq_embedding_dim,
            seq_hidden_size,
            num_layers=seq_num_layers,
            dropout=0.5 if seq_num_layers > 1 else 0.0,
            batch_first=False,
        )
        self.predictor_embedding = nn.Embedding(predictor_vocab_size, predictor_embedding_dim)
        self.lpipeline_embedding = nn.Embedding(logical_pipeline_count, logical_embedding_dim)
        wide = data_dim + seq_hidden_size + predictor_embedding_dim + logical_embedding_dim
        input_dim = data_dim + sequence_length * seq_hidden_size + predictor_embedding_dim + logical_embedding_dim
        self.nn = nn.Sequential(
            nn.Linear(input_dim, wide), nn.LeakyReLU(),
            nn.Linear(wide, wide), nn.LeakyReLU(),
            nn.Linear(wide, 1907), nn.LeakyReLU(),
            nn.Linear(1907, 1907), nn.LeakyReLU(),
            nn.Linear(1907, 950), nn.LeakyReLU(),
            nn.Linear(950, 950), nn.LeakyReLU(),
            nn.Linear(950, 387), nn.LeakyReLU(),
            nn.Linear(387, 387), nn.LeakyReLU(),
            nn.Linear(387, 128), nn.LeakyReLU(),
            nn.Linear(128, 128), nn.LeakyReLU(),
            nn.Linear(128, 64), nn.LeakyReLU(),
            nn.Linear(64, 32), nn.LeakyReLU(),
            nn.Linear(32, action_dim), nn.Tanh(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        data = state[:, : self.data_dim]
        start = self.data_dim
        sequence = state[:, start : start + self.sequence_length].long()
        predictor = state[:, start + self.sequence_length : start + self.sequence_length + 1].long()
        logical = state[:, start + self.sequence_length + 1 : start + self.sequence_length + 2].long()
        embedded = self.seq_embedding(sequence).permute(1, 0, 2)
        sequence_out, _ = self.seq_lstm(embedded)
        sequence_out = sequence_out.permute(1, 0, 2).flatten(start_dim=1)
        predictor_out = self.predictor_embedding(predictor).flatten(start_dim=1)
        logical_out = self.lpipeline_embedding(logical).flatten(start_dim=1)
        return self.nn(torch.cat((data, sequence_out, predictor_out, logical_out), dim=1))


__all__ = ["LogicalDQN", "PrimitiveRnnDQN"]
