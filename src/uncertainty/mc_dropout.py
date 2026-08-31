"""Predictive uncertainty via Monte-Carlo dropout (spec section 28) with an
optional deep-ensemble path.  Returns the predictive mean and standard
deviation over K stochastic forward passes.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def enable_dropout(model: nn.Module) -> None:
    """Keep BatchNorm etc. in eval mode but re-enable Dropout for MC sampling."""
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()


@torch.no_grad()
def _forward(model, x, adj_t, aoi):
    try:
        return model(x, adj_t, aoi)
    except TypeError:
        return model(x, adj_t)


@torch.no_grad()
def mc_dropout_predict(model, x: torch.Tensor, adj_t, aoi=None, K: int = 30,
                       device: str = "cpu", batch_size: int = 128):
    model.to(device).eval()
    enable_dropout(model)
    means = []
    m2 = []
    for i in range(0, x.shape[0], batch_size):
        xb = x[i:i + batch_size].to(device)
        ab = aoi[i:i + batch_size].to(device) if aoi is not None else None
        samples = torch.stack([_forward(model, xb, adj_t, ab) for _ in range(K)], 0)
        means.append(samples.mean(0).cpu().numpy())
        m2.append(samples.std(0, unbiased=True).cpu().numpy())
    model.eval()
    return np.concatenate(means, 0), np.concatenate(m2, 0)


@torch.no_grad()
def ensemble_predict(models: list, x: torch.Tensor, adj_t, aoi=None,
                     device: str = "cpu", batch_size: int = 128):
    per_model = []
    for mdl in models:
        mdl.to(device).eval()
        outs = []
        for i in range(0, x.shape[0], batch_size):
            xb = x[i:i + batch_size].to(device)
            ab = aoi[i:i + batch_size].to(device) if aoi is not None else None
            outs.append(_forward(mdl, xb, adj_t, ab).cpu().numpy())
        per_model.append(np.concatenate(outs, 0))
    stk = np.stack(per_model, 0)
    return stk.mean(0), stk.std(0, ddof=1)


def predictive_uncertainty(model_or_models, x, adj_t, aoi=None, method="mc_dropout",
                           K=30, device="cpu"):
    if method == "ensemble" and isinstance(model_or_models, (list, tuple)):
        return ensemble_predict(list(model_or_models), x, adj_t, aoi, device)
    return mc_dropout_predict(model_or_models, x, adj_t, aoi, K, device)
