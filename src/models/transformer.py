"""Temporal Transformer forecaster with a light spatial GCN mixer."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .layers import GCNLayer


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):                       # x: (B, L, d)
        return x + self.pe[:, : x.size(1)]


class TemporalTransformerForecaster(nn.Module):
    def __init__(self, in_dim, num_nodes, hidden, horizon, dropout=0.1,
                 nhead=4, layers=2):
        super().__init__()
        self.embed = nn.Linear(in_dim, hidden)
        self.spatial = GCNLayer(hidden, hidden)
        self.pos = _PositionalEncoding(hidden)
        enc = nn.TransformerEncoderLayer(hidden, nhead, hidden * 2, dropout,
                                         batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, horizon)
        self.horizon = horizon

    def forward(self, x, adj=None, aoi=None):
        B, L, N, C = x.shape
        h = self.embed(x)                       # (B, L, N, hidden)
        if adj is not None:
            h = torch.relu(self.spatial(h.reshape(B * L, N, -1), adj)).view(B, L, N, -1)
        h = h.permute(0, 2, 1, 3).reshape(B * N, L, -1)
        h = self.encoder(self.pos(h))
        y = self.head(self.drop(h[:, -1])).view(B, N, self.horizon)
        return y.permute(0, 2, 1).unsqueeze(-1)
