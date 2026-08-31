"""Assemble source-domain tensors once, reuse across scripts / dashboard.

Chronological split, z-score on the TRAIN block only (channel 0 = speed),
sliding windows inside each block (spec sections 6, 44).
"""
from __future__ import annotations

import numpy as np

from .source_dataset import load_source_dataset
from .splits import chronological_split
from .windowing import make_windows
from src.preprocessing.normalize import ZScoreScaler
from src.preprocessing.graph import (gaussian_kernel_adjacency,
                                     normalize_adjacency, knn_adjacency)


def _build_adjacency(ds, cfg):
    if ds.adj is not None:
        A = np.asarray(ds.adj, np.float32)
        np.fill_diagonal(A, 0.0)
        return A
    if ds.coords is not None:
        return gaussian_kernel_adjacency(
            ds.coords, cfg["source_dataset"].get("gaussian_kernel_sigma"),
            float(cfg["source_dataset"]["adjacency_threshold"]))
    # last resort: identity-ish chain
    N = ds.num_nodes
    A = np.zeros((N, N), np.float32)
    for i in range(N - 1):
        A[i, i + 1] = A[i + 1, i] = 1.0
    return A


# Unified 8-channel feature schema shared by source and target domains so the
# proposed model's input projection transfers directly (spec sections 8, 23).
FEATURE_SCHEMA = ["speed", "density", "flow", "queue",
                  "mask", "aoi_norm", "pdr", "latency_norm"]


def to_feature_schema(speed_norm: np.ndarray) -> np.ndarray:
    """(T, N) normalised speed -> (T, N, 8).  Source domain has full, fresh,
    lossless observation: mask=1, aoi=0, pdr=1, latency=0; density/flow/queue
    are unavailable in METR-LA/PEMS-BAY and are left at 0."""
    T, N = speed_norm.shape
    f = np.zeros((T, N, 8), np.float32)
    f[..., 0] = speed_norm
    f[..., 4] = 1.0     # mask (fully observed)
    f[..., 6] = 1.0     # pdr = 1
    return f


def prepare_source_domain(cfg, cap_windows: bool = True) -> dict:
    ds = load_source_dataset(cfg)
    tr = cfg["training"]
    L, H = int(tr["history_length"]), int(tr["prediction_horizon"])
    stride = int(tr.get("window_stride", 1) or 1)
    sp = chronological_split(ds.data.shape[0], tuple(tr["split"]))

    data = ds.data.astype(np.float32).copy()
    sc = ZScoreScaler().fit(data[sp.train[0]:sp.train[1], :, 0],
                            mask=(ds.missing_mask[sp.train[0]:sp.train[1]]
                                  if ds.missing_mask is not None else None))
    speed_norm = sc.transform(data[..., 0])
    feat = to_feature_schema(speed_norm)          # (T, N, 8)
    # raw (unnormalised speed) 1-channel series for the classical baselines
    raw_series = data[..., :1]

    raw_w = {k: make_windows(raw_series, L, H, *getattr(sp, k), stride=stride)
             for k in ("train", "val", "test")}
    nrm_w = {k: make_windows(feat, L, H, *getattr(sp, k), stride=stride)
             for k in ("train", "val", "test")}

    cap = tr.get("max_train_windows")
    if cap_windows and cap and len(nrm_w["train"]) > int(cap):
        step = int(np.ceil(len(nrm_w["train"]) / int(cap)))
        for W in (raw_w, nrm_w):
            wt = W["train"]
            W["train"] = type(wt)(wt.X[::step], wt.Y[::step], wt.t_index[::step])

    A = _build_adjacency(ds, cfg)
    A_norm = normalize_adjacency(A)

    import torch
    torch_sets = {k: {"x": torch.from_numpy(nrm_w[k].X),
                      "y": torch.from_numpy(nrm_w[k].Y)}
                  for k in ("train", "val", "test")}

    return {
        "dataset_name": ds.name, "is_synthetic": ds.is_synthetic,
        "num_nodes": ds.num_nodes, "in_dim": 8, "feature_schema": FEATURE_SCHEMA,
        "history": L, "horizon": H, "split": sp.as_dict(),
        "scaler": sc, "adj": A, "adj_norm": A_norm,
        "coords": ds.coords,
        "raw_windows": raw_w,          # numpy, raw speed  -> classical baselines
        "norm_windows": nrm_w,         # numpy, normalised -> reference
        "torch": torch_sets,           # torch tensors, normalised
        "speed_scale_mps": float(sc.std_) * 0.44704,  # z-unit -> m/s (mph based)
    }
