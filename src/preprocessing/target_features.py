"""Assemble VANET target-domain training tensors from a PartialObservation
(spec section 23).

Per (timestep, road-segment) the input feature vector is

    [ obs_speed, obs_density, obs_flow, obs_queue,     # partially-observed state
      mask,                                            # 1 = observed
      aoi_norm,                                         # AoI / aoi_cap
      pdr, latency_norm ]                               # scalar comm-quality tags

The prediction target is the ground-truth *speed* channel (evaluation only for
the density/flow/queue channels via the reconstruction head).  Splitting is
chronological -- the test window is never seen during target adaptation.
"""
from __future__ import annotations

import numpy as np

from src.data.splits import chronological_split
from src.data.windowing import make_windows_xy
from src.preprocessing.normalize import ZScoreScaler


def build_target_tensors(po, cfg, seed: int = 42) -> dict:
    obs = np.asarray(po.observed_state, np.float32)          # (T, N, 4)
    gt = np.asarray(po.ground_truth_state, np.float32)       # (T, N, 4)
    mask = np.asarray(po.mask, np.float32)                   # (T, N)
    aoi = np.asarray(po.aoi, np.float32)                     # (T, N)
    T, N, _ = obs.shape
    aoi_cap = float(cfg["vanet"].get("aoi_cap_s", 600.0))

    from src.utils.config import target_window
    L, H = target_window(cfg)
    fr = cfg["training"]["split"]
    sp = chronological_split(T, tuple(fr))

    # scaler fit on TRAIN speed only (leakage-free)
    sc = ZScoreScaler().fit(gt[sp.train[0]:sp.train[1], :, 0])
    obs_sc = obs.copy()
    obs_sc[..., 0] = sc.transform(obs[..., 0])
    gt_speed_sc = sc.transform(gt[..., 0])

    pdr = np.full((T, N, 1), po.pdr, np.float32)
    lat = np.full((T, N, 1), min(po.latency_ms / 500.0, 1.0), np.float32)
    aoi_norm = (aoi / aoi_cap)[..., None]
    feats = np.concatenate([obs_sc,                       # 4
                            mask[..., None],              # 1
                            aoi_norm,                     # 1
                            pdr, lat], axis=-1)           # 2  -> C = 8

    def _win(a, b):
        return make_windows_xy(feats, gt_speed_sc, L, H,
                               aoi_series=aoi, obs_speed_series=obs_sc[..., 0],
                               start=a, end=b)

    train = _win(*sp.train)
    val = _win(*sp.val)
    test = _win(*sp.test)

    # torch-friendly tensor dicts
    import torch

    def _to_torch(d):
        return {
            "x": torch.from_numpy(d["X"]),
            "y": torch.from_numpy(d["Y"]),
            "aoi": torch.from_numpy(d["aoi"]),
            "last_obs": torch.from_numpy(d["last_obs"]),
            "t_index": torch.from_numpy(d["t_index"]),
        }

    return {
        "train": _to_torch(train), "val": _to_torch(val), "test": _to_torch(test),
        "scaler": sc, "n_cells": N, "in_dim": feats.shape[-1],
        "split": sp.as_dict(),
        "gt_full": gt, "obs_full": obs, "mask_full": mask, "aoi_full": aoi,
        "speed_scale_mps": float(sc.std_),
    }
