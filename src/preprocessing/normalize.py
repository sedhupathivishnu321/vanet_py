"""Feature scaling.  Statistics are fit on the TRAIN block only (no leakage)."""
from __future__ import annotations

import numpy as np


class ZScoreScaler:
    def __init__(self, eps: float = 1e-6):
        self.mean_ = 0.0
        self.std_ = 1.0
        self.eps = eps
        self._fitted = False

    def fit(self, x: np.ndarray, mask: np.ndarray | None = None) -> "ZScoreScaler":
        v = x if mask is None else x[~mask.astype(bool)]
        v = v[np.isfinite(v)]
        self.mean_ = float(np.mean(v)) if v.size else 0.0
        self.std_ = float(np.std(v)) if v.size else 1.0
        self.std_ = self.std_ if self.std_ > self.eps else 1.0
        self._fitted = True
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) / self.std_

    def inverse_transform(self, x):
        return x * self.std_ + self.mean_

    def fit_transform(self, x, mask=None):
        return self.fit(x, mask).transform(x)

    def state_dict(self) -> dict:
        return {"mean": self.mean_, "std": self.std_}

    def load_state_dict(self, d: dict) -> "ZScoreScaler":
        self.mean_, self.std_ = d["mean"], d["std"]
        self._fitted = True
        return self
