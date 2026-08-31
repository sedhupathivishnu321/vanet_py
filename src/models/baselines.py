"""Non-parametric / classical baselines with a numpy fit/predict API.

Each operates on windowed tensors
    X : (num, L, N, C)   Y : (num, H, N, 1)
and predicts Yhat : (num, H, N, 1).  Feature 0 of X is the traffic speed.
"""
from __future__ import annotations

import numpy as np


class HistoricalAverage:
    """Predict every future step as the mean of the history window per node."""
    name = "historical_average"

    def fit(self, X, Y):
        return self

    def predict(self, X):
        h = X[..., 0].mean(axis=1)                      # (num, N)
        H = self._horizon
        return np.repeat(h[:, None, :, None], H, axis=1)

    def __init__(self, horizon: int):
        self._horizon = horizon


class _PerNodeSklearn:
    """Fits one sklearn regressor per (node, horizon-step); flattened history
    window (L*C) is the feature vector."""

    def __init__(self, horizon: int, make_estimator):
        self.horizon = horizon
        self._make = make_estimator
        self.models: dict[tuple[int, int], object] = {}

    def fit(self, X, Y):
        num, L, N, C = X.shape
        feats = X.reshape(num, L, N, C).transpose(0, 2, 1, 3).reshape(num, N, L * C)
        for n in range(N):
            for hstep in range(self.horizon):
                est = self._make()
                est.fit(feats[:, n, :], Y[:, hstep, n, 0])
                self.models[(n, hstep)] = est
        return self

    def predict(self, X):
        num, L, N, C = X.shape
        feats = X.transpose(0, 2, 1, 3).reshape(num, N, L * C)
        out = np.zeros((num, self.horizon, N, 1), np.float32)
        for n in range(N):
            for hstep in range(self.horizon):
                out[:, hstep, n, 0] = self.models[(n, hstep)].predict(feats[:, n, :])
        return out


class LinearRegressionForecaster(_PerNodeSklearn):
    name = "linear_regression"

    def __init__(self, horizon: int):
        from sklearn.linear_model import Ridge
        super().__init__(horizon, lambda: Ridge(alpha=1.0))


class RandomForestForecaster(_PerNodeSklearn):
    name = "random_forest"

    def __init__(self, horizon: int, n_estimators: int = 120, max_depth: int = 16,
                 n_jobs: int = -1):
        from sklearn.ensemble import RandomForestRegressor
        super().__init__(
            horizon,
            lambda: RandomForestRegressor(n_estimators=n_estimators,
                                          max_depth=max_depth, n_jobs=n_jobs,
                                          random_state=0),
        )


def build_numpy_baseline(name: str, horizon: int, cfg=None):
    name = name.lower()
    if name == "historical_average":
        return HistoricalAverage(horizon)
    if name == "linear_regression":
        return LinearRegressionForecaster(horizon)
    if name == "random_forest":
        rf = (cfg["baselines"]["random_forest"] if cfg else
              {"n_estimators": 120, "max_depth": 16})
        return RandomForestForecaster(horizon, rf["n_estimators"], rf["max_depth"])
    raise KeyError(name)
