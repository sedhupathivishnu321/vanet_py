"""Graph layers: dense GCN, dense multi-head GAT, and AoI-conditioned attention.

All layers take
    x   : (B, N, F)
    adj : (N, N)      -- weighted (GCN wants the normalised form) or binary
and return (B, N, F_out).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # adj: (N, N) assumed pre-normalised (D^-1/2 (A+I) D^-1/2)
        return torch.einsum("nm,bmf->bnf", adj, self.lin(x))


class GATLayer(nn.Module):
    """Dense multi-head graph attention (Velickovic et al., 2018)."""

    def __init__(self, in_dim: int, out_dim: int, heads: int = 4,
                 dropout: float = 0.1, concat: bool = True, leaky: float = 0.2):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.concat = concat
        self.W = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(1, heads, out_dim))
        self.a_dst = nn.Parameter(torch.empty(1, heads, out_dim))
        self.leaky = nn.LeakyReLU(leaky)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, x: torch.Tensor, adj: torch.Tensor,
                extra_bias: torch.Tensor | None = None) -> torch.Tensor:
        B, N, _ = x.shape
        h = self.W(x).view(B, N, self.heads, self.out_dim)      # (B,N,H,D)
        e_src = (h * self.a_src).sum(-1)                        # (B,N,H)
        e_dst = (h * self.a_dst).sum(-1)                        # (B,N,H)
        # score_{ij} = LeakyReLU(e_src_i + e_dst_j)
        scores = self.leaky(e_src.unsqueeze(2) + e_dst.unsqueeze(1))  # (B,N,N,H)
        mask = (adj > 0).unsqueeze(0).unsqueeze(-1)             # (1,N,N,1)
        if extra_bias is not None:
            # extra_bias broadcast over query i: shape (B,N,H) keyed on j
            scores = scores + extra_bias.unsqueeze(1)
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(scores, dim=2)
        attn = torch.nan_to_num(attn)          # isolated nodes -> all -inf row
        attn = self.dropout(attn)
        out = torch.einsum("bijh,bjhd->bihd", attn, h)          # (B,N,H,D)
        if self.concat:
            return out.reshape(B, N, self.heads * self.out_dim)
        return out.mean(dim=2)


class AoIGraphAttention(nn.Module):
    """Graph attention with an Age-of-Information penalty on the key node
    (spec section 24):   alpha_ij = softmax_j( score(h_i, h_j) - beta * AoI_j ).

    `aoi` is (B, N) with AoI expressed in seconds (0 for fresh / local data).
    Set ``use_aoi=False`` to recover a plain GAT layer for the ablation.
    """

    def __init__(self, in_dim: int, out_dim: int, heads: int = 4,
                 dropout: float = 0.1, beta: float = 1.0, use_aoi: bool = True,
                 concat: bool = True):
        super().__init__()
        self.gat = GATLayer(in_dim, out_dim, heads, dropout, concat=concat)
        self.use_aoi = use_aoi
        # learnable, but initialised at the configured beta and kept positive
        self.log_beta = nn.Parameter(torch.log(torch.tensor(float(beta) + 1e-3)))
        self.aoi_scale = 1.0 / 60.0   # seconds -> minutes, keeps magnitudes sane

    def forward(self, x: torch.Tensor, adj: torch.Tensor,
                aoi: torch.Tensor | None = None) -> torch.Tensor:
        extra = None
        if self.use_aoi and aoi is not None:
            beta = torch.exp(self.log_beta)
            pen = -beta * (aoi * self.aoi_scale)               # (B, N)
            extra = pen.unsqueeze(-1).expand(-1, -1, self.gat.heads)  # (B,N,H)
        return self.gat(x, adj, extra_bias=extra)
