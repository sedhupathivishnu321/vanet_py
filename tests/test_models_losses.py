import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.models import build_model, ALL_MODELS
from src.models.losses import (masked_mae, smoothness_loss, physics_loss,
                               mmd_loss, coral_loss, aoi_consistency_loss)
from src.preprocessing.graph import normalize_adjacency, knn_adjacency


@pytest.mark.parametrize("name", ["lstm", "gru", "gcn", "gat", "stgnn",
                                  "transformer", "proposed"])
def test_forward_shapes(cfg, name):
    B, L, N, C, H = 3, cfg["training"]["history_length"], 8, 8, \
        cfg["training"]["prediction_horizon"]
    model = build_model(name, C, N, H, cfg)
    x = torch.randn(B, L, N, C)
    adj = torch.from_numpy(normalize_adjacency(knn_adjacency(
        np.random.default_rng(0).random((N, 2)), k=3)))
    aoi = torch.rand(B, N) * 30
    try:
        y = model(x, adj, aoi)
    except TypeError:
        y = model(x, adj)
    assert y.shape == (B, H, N, 1)
    assert torch.isfinite(y).all()


def test_proposed_encoder_freeze_and_latent(cfg):
    m = build_model("proposed", 8, 6, 12, cfg)
    m.freeze_encoder(True)
    assert all(not p.requires_grad for p in m.encoder_parameters())
    assert any(p.requires_grad for p in m.adapt_parameters())
    x = torch.randn(2, 12, 6, 8)
    adj = torch.eye(6)
    z = m.latent(x, adj)
    assert z.shape[0] == 2 and z.ndim == 2


def test_losses_basic():
    p = torch.zeros(2, 12, 5, 1)
    t = torch.ones(2, 12, 5, 1)
    assert abs(masked_mae(p, t).item() - 1.0) < 1e-6
    m = torch.ones(2, 12, 5, 1); m[:, :, :2] = 0
    assert masked_mae(p, t, m).item() == pytest.approx(1.0)
    assert smoothness_loss(torch.randn(2, 12, 5, 1)).item() >= 0
    assert physics_loss(torch.randn(2, 12, 5, 1), dt=300.0).item() >= 0


def test_domain_losses_zero_when_identical():
    z = torch.randn(32, 16)
    assert mmd_loss(z, z.clone()).abs().item() < 1e-3
    assert coral_loss(z, z.clone()).item() < 1e-6
    # different distributions -> positive
    assert mmd_loss(z, z + 3.0).item() > 0


def test_aoi_consistency_weights_by_age():
    pred = torch.zeros(2, 6, 4, 1)
    last = torch.ones(2, 4)
    lo_aoi = torch.zeros(2, 4)
    hi_aoi = torch.full((2, 4), 300.0)
    assert (aoi_consistency_loss(pred, last, hi_aoi).item()
            > aoi_consistency_loss(pred, last, lo_aoi).item())
