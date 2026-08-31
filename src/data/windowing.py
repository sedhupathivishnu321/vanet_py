"""Sliding-window construction for the seq2seq task  X_{t-L:t} -> X_{t+1:t+H}."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WindowedTensors:
    X: np.ndarray   # (num, L, N, C)
    Y: np.ndarray   # (num, H, N, 1)  -- target = feature 0 (speed)
    t_index: np.ndarray  # (num,) absolute timestamp of the first predicted step

    def __len__(self) -> int:
        return self.X.shape[0]


def make_windows(series: np.ndarray, history: int, horizon: int,
                 start: int = 0, end: int | None = None,
                 target_feature: int = 0, stride: int = 1) -> WindowedTensors:
    """`series` is (T, N, C).  Windows are built only within [start, end)."""
    T = series.shape[0]
    end = T if end is None else end
    seg = series[start:end]
    Ts = seg.shape[0]
    xs, ys, idx = [], [], []
    last = Ts - history - horizon + 1
    for i in range(0, max(0, last), stride):
        xs.append(seg[i:i + history])
        ys.append(seg[i + history:i + history + horizon, :, target_feature:target_feature + 1])
        idx.append(start + i + history)
    if not xs:
        C = series.shape[2]
        N = series.shape[1]
        return WindowedTensors(np.empty((0, history, N, C), np.float32),
                               np.empty((0, horizon, N, 1), np.float32),
                               np.empty((0,), np.int64))
    return WindowedTensors(np.asarray(xs, np.float32),
                           np.asarray(ys, np.float32),
                           np.asarray(idx, np.int64))


def make_windows_xy(x_series: np.ndarray, y_series: np.ndarray,
                    history: int, horizon: int,
                    aoi_series: np.ndarray | None = None,
                    obs_speed_series: np.ndarray | None = None,
                    start: int = 0, end: int | None = None, stride: int = 1):
    """Separate input / target series (used for the VANET target domain).

    x_series : (T, N, Cx)   partially-observed + communication features
    y_series : (T, N)  or (T, N, 1)   ground-truth target (speed)
    Returns dict with X (num,L,N,Cx), Y (num,H,N,1), t_index (num,),
    aoi (num,N) current-step AoI, last_obs (num,N) last observed speed.
    """
    T = x_series.shape[0]
    end = T if end is None else end
    if y_series.ndim == 2:
        y_series = y_series[..., None]
    xs, ys, idx, aoi_w, last_w = [], [], [], [], []
    lo = start
    hi = end - history - horizon + 1
    for i in range(lo, max(lo, hi), stride):
        xs.append(x_series[i:i + history])
        ys.append(y_series[i + history:i + history + horizon])
        idx.append(i + history)
        if aoi_series is not None:
            aoi_w.append(aoi_series[i + history - 1])
        if obs_speed_series is not None:
            last_w.append(obs_speed_series[i + history - 1])
    N = x_series.shape[1]
    Cx = x_series.shape[2]
    if not xs:
        empty = dict(X=np.empty((0, history, N, Cx), np.float32),
                     Y=np.empty((0, horizon, N, 1), np.float32),
                     t_index=np.empty((0,), np.int64),
                     aoi=np.empty((0, N), np.float32),
                     last_obs=np.empty((0, N), np.float32))
        return empty
    out = dict(X=np.asarray(xs, np.float32), Y=np.asarray(ys, np.float32),
               t_index=np.asarray(idx, np.int64))
    out["aoi"] = (np.asarray(aoi_w, np.float32) if aoi_w
                  else np.zeros((len(xs), N), np.float32))
    out["last_obs"] = (np.asarray(last_w, np.float32) if last_w
                       else np.zeros((len(xs), N), np.float32))
    return out


class SlidingWindowDataset:
    """torch.utils.data.Dataset wrapper (torch imported lazily)."""

    def __init__(self, wt: WindowedTensors, aoi: np.ndarray | None = None,
                 mask: np.ndarray | None = None):
        import torch
        self._torch = torch
        self.X = torch.from_numpy(wt.X)
        self.Y = torch.from_numpy(wt.Y)
        self.t_index = torch.from_numpy(wt.t_index)
        self.aoi = torch.from_numpy(aoi.astype(np.float32)) if aoi is not None else None
        self.mask = torch.from_numpy(mask.astype(np.float32)) if mask is not None else None

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        item = {"x": self.X[i], "y": self.Y[i], "t": self.t_index[i]}
        if self.aoi is not None:
            item["aoi"] = self.aoi[i]
        if self.mask is not None:
            item["mask"] = self.mask[i]
        return item
