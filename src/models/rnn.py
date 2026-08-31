"""Pure temporal baselines: node-independent LSTM / GRU forecasters.

Input  x   : (B, L, N, C)
Output y   : (B, H, N, 1)
`adj` / `aoi` are accepted for a uniform call signature and ignored.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _RNNForecaster(nn.Module):
    rnn_cls: type

    def __init__(self, in_dim: int, num_nodes: int, hidden: int, horizon: int,
                 dropout: float = 0.1, layers: int = 2):
        super().__init__()
        self.num_nodes = num_nodes
        self.horizon = horizon
        self.rnn = self.rnn_cls(in_dim, hidden, num_layers=layers,
                                batch_first=True,
                                dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x: torch.Tensor, adj=None, aoi=None) -> torch.Tensor:
        B, L, N, C = x.shape
        h = x.permute(0, 2, 1, 3).reshape(B * N, L, C)
        out, _ = self.rnn(h)
        last = self.drop(out[:, -1])                 # (B*N, hidden)
        y = self.head(last).view(B, N, self.horizon)
        return y.permute(0, 2, 1).unsqueeze(-1)      # (B, H, N, 1)


class LSTMForecaster(_RNNForecaster):
    rnn_cls = nn.LSTM


class GRUForecaster(_RNNForecaster):
    rnn_cls = nn.GRU
