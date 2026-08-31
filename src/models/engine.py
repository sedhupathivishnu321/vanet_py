"""Training / evaluation loop shared by the source-domain and target-domain
stages.  CPU-first: manual mini-batching over in-memory tensors, no DataLoader
workers (keeps Codespaces + Windows happy and fully deterministic).
"""
from __future__ import annotations

import copy
import time
from typing import Callable, Optional

import numpy as np
import torch

from .losses import (masked_mae, smoothness_loss, physics_loss,
                     aoi_consistency_loss)


def _batches(n: int, bs: int, shuffle: bool, rng: np.random.Generator):
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_torch_model(
    model: torch.nn.Module,
    train: dict,                 # {'x','y', optional 'aoi','mask','last_obs'}
    val: dict,
    cfg,
    adj_t: Optional[torch.Tensor],
    device: str = "cpu",
    *,
    full_objective: bool = False,
    epochs: Optional[int] = None,
    lr: Optional[float] = None,
    param_groups=None,
    speed_scale: float = 1.0,
    physics_dt: Optional[float] = None,
    aux_loss_fn: Optional[Callable] = None,
    seed: int = 42,
    logger=None,
) -> dict:
    model.to(device)
    tr = cfg["training"]
    lw = cfg["loss_weights"]
    phys = cfg["physics"]
    dt_phys = float(physics_dt if physics_dt is not None else phys["dt_seconds"])
    epochs = int(epochs or tr["epochs"])
    lr = float(lr or tr["learning_rate"])
    bs = int(tr["batch_size"])
    rng = np.random.default_rng(seed)

    params = param_groups if param_groups is not None else \
        [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr, weight_decay=float(tr["weight_decay"]))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5,
                                                       patience=5)

    Xtr, Ytr = train["x"].to(device), train["y"].to(device)
    Xva, Yva = val["x"].to(device), val["y"].to(device)
    aoi_tr = train.get("aoi")
    aoi_va = val.get("aoi")
    mask_tr = train.get("mask")
    last_tr = train.get("last_obs")
    if aoi_tr is not None:
        aoi_tr = aoi_tr.to(device)
    if aoi_va is not None:
        aoi_va = aoi_va.to(device)
    if mask_tr is not None:
        mask_tr = mask_tr.to(device)
    if last_tr is not None:
        last_tr = last_tr.to(device)

    def _fwd(model, x, aoi):
        try:
            return model(x, adj_t, aoi)
        except TypeError:
            return model(x, adj_t)

    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience = int(tr["early_stopping_patience"])
    bad = 0
    history = []
    t0 = time.time()

    for ep in range(epochs):
        model.train()
        ep_loss = 0.0
        nb = 0
        for bidx in _batches(Xtr.shape[0], bs, True, rng):
            bi = torch.as_tensor(bidx, device=device)
            xb, yb = Xtr[bi], Ytr[bi]
            ab = aoi_tr[bi] if aoi_tr is not None else None
            mb = mask_tr[bi] if mask_tr is not None else None
            pred = _fwd(model, xb, ab)
            loss = masked_mae(pred, yb, mb) * lw["prediction"]
            if full_objective:
                loss = loss + lw["smooth"] * smoothness_loss(pred)
                loss = loss + lw["physics"] * physics_loss(
                    pred, dt=dt_phys,
                    use_jerk=bool(phys["use_jerk"]), speed_scale=speed_scale)
                if ab is not None and last_tr is not None:
                    lo = last_tr[bi]
                    loss = loss + lw["aoi"] * aoi_consistency_loss(
                        pred, lo, ab if ab.dim() == 2 else ab[:, -1])
            if aux_loss_fn is not None:
                loss = loss + aux_loss_fn(model, xb, ab)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           float(tr["grad_clip"]))
            opt.step()
            ep_loss += float(loss.detach())
            nb += 1

        model.eval()
        with torch.no_grad():
            vpred = _fwd(model, Xva, aoi_va)
            vloss = float(masked_mae(vpred, Yva))
        sched.step(vloss)
        history.append({"epoch": ep, "train_loss": ep_loss / max(nb, 1),
                        "val_mae": vloss})
        if logger:
            logger.info(f"  epoch {ep:3d}  train={ep_loss/max(nb,1):.4f}  "
                        f"val_mae={vloss:.4f}")
        if vloss < best_val - 1e-5:
            best_val = vloss
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    return {
        "best_val_mae": best_val,
        "history": history,
        "train_time_s": round(time.time() - t0, 2),
        "epochs_run": len(history),
        "num_parameters": count_parameters(model),
        "state_dict": best_state,
    }


@torch.no_grad()
def predict_torch(model, x: torch.Tensor, adj_t, aoi=None, device="cpu",
                  batch_size: int = 128) -> np.ndarray:
    model.to(device).eval()
    outs = []
    for i in range(0, x.shape[0], batch_size):
        xb = x[i:i + batch_size].to(device)
        ab = aoi[i:i + batch_size].to(device) if aoi is not None else None
        try:
            outs.append(model(xb, adj_t, ab).cpu().numpy())
        except TypeError:
            outs.append(model(xb, adj_t).cpu().numpy())
    return np.concatenate(outs, axis=0)
