"""Spatio-temporal GNN baselines: GCN, GAT and a compact ST-GCN.

All expect
    x   : (B, L, N, C)
    adj : (N, N)  -- GCN variants use the symmetric-normalised form
and return (B, H, N, 1).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .layers import GCNLayer, GATLayer


class _SpatialTemporalHead(nn.Module):
    """Shared: GRU over time on per-node spatial embeddings + linear horizon head."""

    def __init__(self, spatial_dim: int, hidden: int, horizon: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(spatial_dim, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, horizon)
        self.horizon = horizon

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        # seq: (B, L, N, spatial_dim)
        B, L, N, D = seq.shape
        h = seq.permute(0, 2, 1, 3).reshape(B * N, L, D)
        out, _ = self.gru(h)
        y = self.head(self.drop(out[:, -1])).view(B, N, self.horizon)
        return y.permute(0, 2, 1).unsqueeze(-1)


class GCNForecaster(nn.Module):
    def __init__(self, in_dim, num_nodes, hidden, horizon, dropout=0.1):
        super().__init__()
        self.g1 = GCNLayer(in_dim, hidden)
        self.g2 = GCNLayer(hidden, hidden)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.temporal = _SpatialTemporalHead(hidden, hidden, horizon, dropout)

    def forward(self, x, adj, aoi=None):
        B, L, N, C = x.shape
        z = x.reshape(B * L, N, C)
        z = self.drop(self.act(self.g1(z, adj)))
        z = self.act(self.g2(z, adj))
        z = z.view(B, L, N, -1)
        return self.temporal(z)


class GATForecaster(nn.Module):
    def __init__(self, in_dim, num_nodes, hidden, horizon, dropout=0.1, heads=4):
        super().__init__()
        assert hidden % heads == 0, "hidden must be divisible by heads"
        self.g1 = GATLayer(in_dim, hidden // heads, heads=heads, dropout=dropout)
        self.g2 = GATLayer(hidden, hidden // heads, heads=heads, dropout=dropout)
        self.act = nn.ELU()
        self.temporal = _SpatialTemporalHead(hidden, hidden, horizon, dropout)

    def forward(self, x, adj, aoi=None):
        B, L, N, C = x.shape
        z = x.reshape(B * L, N, C)
        z = self.act(self.g1(z, adj))
        z = self.act(self.g2(z, adj))
        z = z.view(B, L, N, -1)
        return self.temporal(z)


class _GatedTemporalConv(nn.Module):
    def __init__(self, ch: int, k: int = 3):
        super().__init__()
        self.pad = (k - 1)
        self.conv = nn.Conv2d(ch, 2 * ch, (1, k))

    def forward(self, x):                      # x: (B, C, N, L)
        x = nn.functional.pad(x, (self.pad, 0))
        p, q = self.conv(x).chunk(2, dim=1)
        return torch.tanh(p) * torch.sigmoid(q)


class STGNNForecaster(nn.Module):
    """Compact ST-GCN: (temporal gated conv -> spatial GCN -> temporal gated conv)."""

    def __init__(self, in_dim, num_nodes, hidden, horizon, dropout=0.1):
        super().__init__()
        self.inp = nn.Conv2d(in_dim, hidden, (1, 1))
        self.t1 = _GatedTemporalConv(hidden)
        self.g1 = GCNLayer(hidden, hidden)
        self.t2 = _GatedTemporalConv(hidden)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, horizon)
        self.horizon = horizon

    def forward(self, x, adj, aoi=None):
        B, L, N, C = x.shape
        h = self.inp(x.permute(0, 3, 2, 1))     # (B, hidden, N, L)
        h = self.t1(h)
        # spatial mix per time step
        h = h.permute(0, 3, 2, 1).reshape(B * L, N, -1)
        h = torch.relu(self.g1(h, adj)).view(B, L, N, -1).permute(0, 3, 2, 1)
        h = self.t2(h)                          # (B, hidden, N, L)
        last = self.drop(h[..., -1]).permute(0, 2, 1)   # (B, N, hidden)
        y = self.head(last)                     # (B, N, H)
        return y.permute(0, 2, 1).unsqueeze(-1)
