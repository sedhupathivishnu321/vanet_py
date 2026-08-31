"""AoI-Aware Transferable Spatio-Temporal Graph Neural Network  (proposed model).

Pipeline (spec section 8):

    input features  ->  input projection
                    ->  spatial encoder        (GCN | GAT)   [TRANSFERABLE]
                    ->  AoI-aware attention     (communication-aware embedding)
                    ->  temporal encoder       (GRU | Transformer)
                    ->  transfer adapter        [target-domain adaptation]
                    ->  horizon head           -> traffic-state prediction

`latent()` exposes the pooled spatial embedding Z used by the MMD / CORAL
domain-alignment losses.  Parameter groups ``encoder_parameters`` /
``adapt_parameters`` support the freeze-encoder transfer strategy.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .layers import GCNLayer, GATLayer, AoIGraphAttention


class _TemporalEncoder(nn.Module):
    def __init__(self, mode: str, dim: int, hidden: int, dropout: float):
        super().__init__()
        self.mode = mode
        if mode == "gru":
            self.net = nn.GRU(dim, hidden, batch_first=True)
        elif mode == "transformer":
            layer = nn.TransformerEncoderLayer(dim, 4, hidden * 2, dropout,
                                               batch_first=True)
            self.net = nn.TransformerEncoder(layer, 2)
            self.proj = nn.Linear(dim, hidden)
        else:  # pragma: no cover
            raise ValueError(mode)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        # seq: (BN, L, dim) -> (BN, hidden)
        if self.mode == "gru":
            out, _ = self.net(seq)
            return out[:, -1]
        out = self.net(seq)
        return self.proj(out[:, -1])


class AoITransferableSTGNN(nn.Module):
    def __init__(self, in_dim: int, num_nodes: int, hidden: int, horizon: int,
                 dropout: float = 0.15, spatial: str = "gat",
                 temporal: str = "gru", heads: int = 4,
                 aoi_attention: bool = True, aoi_beta: float = 1.0):
        super().__init__()
        self.num_nodes = num_nodes
        self.horizon = horizon
        self.spatial_kind = spatial
        self.use_aoi = aoi_attention

        self.input_proj = nn.Linear(in_dim, hidden)

        # ---- transferable spatial encoder -------------------------------
        if spatial == "gat":
            assert hidden % heads == 0
            self.enc1 = GATLayer(hidden, hidden // heads, heads, dropout)
            self.enc2 = GATLayer(hidden, hidden // heads, heads, dropout)
        else:
            self.enc1 = GCNLayer(hidden, hidden)
            self.enc2 = GCNLayer(hidden, hidden)
        self.enc_act = nn.ELU()
        self.enc_drop = nn.Dropout(dropout)

        # ---- communication-aware (AoI) attention ----------------------
        self.aoi_attn = AoIGraphAttention(hidden, hidden // heads if spatial == "gat" else hidden,
                                          heads if spatial == "gat" else 1,
                                          dropout, beta=aoi_beta,
                                          use_aoi=aoi_attention,
                                          concat=(spatial == "gat"))
        self.comm_norm = nn.LayerNorm(hidden)

        # ---- temporal encoder ---------------------------------------------
        self.temporal = _TemporalEncoder(temporal, hidden, hidden, dropout)

        # ---- transfer adapter (fine-tuned on target) ---------------------
        self.adapter = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.da_proj = nn.Linear(hidden, hidden)   # projection for MMD/CORAL
        self.head = nn.Linear(hidden, horizon)
        self.dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------------ #
    def _prep_aoi(self, aoi, B, N, device):
        if aoi is None:
            return torch.zeros(B, N, device=device)
        if aoi.dim() == 3:          # (B, L, N) -> current step
            aoi = aoi[:, -1]
        return aoi.to(device).float()

    def encode_spatial(self, x: torch.Tensor, adj: torch.Tensor,
                       aoi: torch.Tensor | None = None) -> torch.Tensor:
        """Returns (B, L, N, hidden) -- the transferable representation."""
        B, L, N, C = x.shape
        h = self.input_proj(x).reshape(B * L, N, -1)
        h = self.enc_drop(self.enc_act(self.enc1(h, adj)))
        h = self.enc_act(self.enc2(h, adj))
        aoi_bn = self._prep_aoi(aoi, B, N, x.device)
        aoi_rep = aoi_bn.repeat_interleave(L, dim=0)         # (B*L, N)
        h = self.aoi_attn(h, adj, aoi_rep)
        h = self.comm_norm(h)
        return h.view(B, L, N, -1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor,
                aoi: torch.Tensor | None = None) -> torch.Tensor:
        B, L, N, C = x.shape
        z = self.encode_spatial(x, adj, aoi)                 # (B, L, N, H)
        z = z.permute(0, 2, 1, 3).reshape(B * N, L, -1)
        t = self.temporal(z)                                  # (B*N, H)
        t = t + self.adapter(t)                               # residual adapter
        y = self.head(self.dropout(t)).view(B, N, self.horizon)
        return y.permute(0, 2, 1).unsqueeze(-1)               # (B, H, N, 1)

    @torch.no_grad()
    def _pool(self, z):
        return z.mean(dim=(1, 2))                             # (B, H)

    def latent(self, x, adj, aoi=None) -> torch.Tensor:
        """Pooled, projected embedding for domain-alignment losses."""
        z = self.encode_spatial(x, adj, aoi)
        pooled = z.mean(dim=(1, 2))                           # (B, hidden)
        return self.da_proj(pooled)

    # ---- parameter groups for transfer strategies ---------------------
    def encoder_parameters(self):
        mods = [self.input_proj, self.enc1, self.enc2]
        for m in mods:
            yield from m.parameters()

    def adapt_parameters(self):
        mods = [self.aoi_attn, self.comm_norm, self.temporal, self.adapter,
                self.da_proj, self.head]
        for m in mods:
            yield from m.parameters()

    def freeze_encoder(self, freeze: bool = True):
        for p in self.encoder_parameters():
            p.requires_grad = not freeze
