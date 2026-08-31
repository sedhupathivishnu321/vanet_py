"""Age of Information (spec section 22).

For cell (road segment) c,   AoI_c(t) = t - u_c(t)   where u_c(t) is the
generation timestamp of the most recently *received* observation mapped to c.
A cell that has never received a fresh beacon is assigned ``cap_s``.
"""
from __future__ import annotations

import numpy as np


class AoITracker:
    def __init__(self, n_cells: int, cap_s: float = 600.0):
        self.n = n_cells
        self.cap = cap_s
        self.last_gen = np.full(n_cells, -np.inf)   # u_c
        self.per_vehicle: dict[int, float] = {}     # last gen time received per veh

    def update(self, beacons, now: float) -> None:
        for b in beacons:
            if b.arrival_time_s <= now + 1e-9:
                if 0 <= b.cell < self.n:
                    self.last_gen[b.cell] = max(self.last_gen[b.cell], b.gen_time_s)
                self.per_vehicle[b.veh_id] = max(
                    self.per_vehicle.get(b.veh_id, -np.inf), b.gen_time_s)

    def aoi(self, now: float) -> np.ndarray:
        age = now - self.last_gen
        age[~np.isfinite(age)] = self.cap
        return np.clip(age, 0.0, self.cap)


def aoi_statistics(aoi_series: np.ndarray) -> dict:
    """aoi_series: (T, N).  Returns mean / median / p95 / max over all cells+time."""
    a = np.asarray(aoi_series, float)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return {"mean": None, "median": None, "p95": None, "max": None}
    return {
        "mean": round(float(np.mean(finite)), 3),
        "median": round(float(np.median(finite)), 3),
        "p95": round(float(np.percentile(finite, 95)), 3),
        "max": round(float(np.max(finite)), 3),
    }
