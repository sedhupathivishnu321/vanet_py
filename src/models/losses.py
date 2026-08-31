"""Loss terms for the total objective (spec section 27):

    L = L_pred + l1 * L_AoI + l2 * L_physics + l3 * L_domain + l4 * L_smooth
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Prediction (optionally masked)
# --------------------------------------------------------------------------- #
def masked_mae(pred: torch.Tensor, target: torch.Tensor,
               mask: torch.Tensor | None = None) -> torch.Tensor:
    err = torch.abs(pred - target)
    if mask is not None:
        mask = mask.expand_as(err).float()
        denom = mask.sum().clamp_min(1.0)
        return (err * mask).sum() / denom
    return err.mean()


def masked_mse(pred, target, mask=None):
    err = (pred - target) ** 2
    if mask is not None:
        mask = mask.expand_as(err).float()
        return (err * mask).sum() / mask.sum().clamp_min(1.0)
    return err.mean()


# --------------------------------------------------------------------------- #
#  Temporal smoothness  (L_smooth)
# --------------------------------------------------------------------------- #
def smoothness_loss(pred: torch.Tensor) -> torch.Tensor:
    # pred: (B, H, N, 1)
    if pred.size(1) < 2:
        return pred.new_zeros(())
    d = pred[:, 1:] - pred[:, :-1]
    return (d ** 2).mean()


# --------------------------------------------------------------------------- #
#  Physics consistency  (L_physics = L_v + L_a [+ L_jerk])   -- spec section 26
# --------------------------------------------------------------------------- #
def physics_loss(pred_speed: torch.Tensor, dt: float,
                 a_max: float = 3.0, use_jerk: bool = True,
                 speed_scale: float = 1.0) -> torch.Tensor:
    """Penalise physically implausible acceleration / jerk implied by the
    predicted speed trajectory.  `pred_speed` is (B, H, N, 1) in *model units*;
    `speed_scale` converts model units -> m/s (e.g. std of the z-scored target,
    times mph->m/s).  Bounds are soft (hinge)."""
    v = pred_speed[..., 0] * speed_scale                # (B, H, N) m/s
    if v.size(1) < 2:
        return v.new_zeros(())
    a = (v[:, 1:] - v[:, :-1]) / dt                     # m/s^2
    l_a = F.relu(a.abs() - a_max).pow(2).mean() + 1e-3 * a.pow(2).mean()
    loss = l_a
    if use_jerk and a.size(1) >= 2:
        j = (a[:, 1:] - a[:, :-1]) / dt
        loss = loss + j.pow(2).mean()
    return loss


def kinematic_consistency(pos: torch.Tensor, vel: torch.Tensor,
                          dt: float) -> torch.Tensor:
    """For the target domain where position AND speed are reconstructed:
    enforce  v_t ~= (x_{t+1} - x_t) / dt."""
    if pos.size(1) < 2:
        return pos.new_zeros(())
    v_impl = (pos[:, 1:] - pos[:, :-1]) / dt
    return F.mse_loss(v_impl, vel[:, :-1])


# --------------------------------------------------------------------------- #
#  AoI consistency  (L_AoI)  -- spec section 27
#  When an observation is stale (large AoI) the prediction should not drift
#  far from the most recent observed value: regress toward persistence with a
#  weight that grows with AoI.
# --------------------------------------------------------------------------- #
def aoi_consistency_loss(pred: torch.Tensor, last_obs: torch.Tensor,
                         aoi: torch.Tensor, tau_s: float = 30.0,
                         scale_s: float = 30.0) -> torch.Tensor:
    """pred: (B, H, N, 1); last_obs: (B, N) or (B, N, 1); aoi: (B, N) seconds."""
    if last_obs.dim() == 2:
        last_obs = last_obs.unsqueeze(-1)
    w = torch.sigmoid((aoi - tau_s) / scale_s).unsqueeze(1).unsqueeze(-1)  # (B,1,N,1)
    persistence = last_obs.unsqueeze(1)                                    # (B,1,N,1)
    return (w * (pred - persistence) ** 2).mean()


# --------------------------------------------------------------------------- #
#  Domain-alignment losses (L_domain)  -- spec section 11
# --------------------------------------------------------------------------- #
def _pairwise_sq_dists(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.unsqueeze(1) - b.unsqueeze(0)).pow(2).sum(-1)


def mmd_loss(zs: torch.Tensor, zt: torch.Tensor,
             bandwidths=(1.0, 2.0, 4.0, 8.0, 16.0)) -> torch.Tensor:
    """Multi-bandwidth Gaussian-kernel Maximum Mean Discrepancy."""
    xx = _pairwise_sq_dists(zs, zs)
    yy = _pairwise_sq_dists(zt, zt)
    xy = _pairwise_sq_dists(zs, zt)
    # median heuristic base scale
    with torch.no_grad():
        med = torch.median(xy.detach()) + 1e-8
    k_xx = k_yy = k_xy = 0.0
    for bw in bandwidths:
        g = 1.0 / (2.0 * bw * med)
        k_xx = k_xx + torch.exp(-g * xx)
        k_yy = k_yy + torch.exp(-g * yy)
        k_xy = k_xy + torch.exp(-g * xy)
    return k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()


def coral_loss(zs: torch.Tensor, zt: torch.Tensor) -> torch.Tensor:
    """CORrelation ALignment (Sun & Saenko, 2016)."""
    d = zs.size(1)

    def cov(z):
        z = z - z.mean(0, keepdim=True)
        return (z.t() @ z) / max(z.size(0) - 1, 1)

    return (cov(zs) - cov(zt)).pow(2).sum() / (4 * d * d)


def domain_loss(zs, zt, kind: str = "mmd"):
    return coral_loss(zs, zt) if kind == "coral" else mmd_loss(zs, zt)
