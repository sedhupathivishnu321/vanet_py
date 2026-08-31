"""Prediction, calibration and AoI-vs-error metrics (spec sections 29, 46)."""
from __future__ import annotations

import numpy as np


def _flat(pred, target, mask=None):
    p = np.asarray(pred, float).reshape(-1)
    t = np.asarray(target, float).reshape(-1)
    if mask is not None:
        m = np.asarray(mask, float).reshape(-1).astype(bool)
        # broadcast mask if shapes differ on last dims
        if m.size != p.size and m.size and p.size % m.size == 0:
            m = np.repeat(m, p.size // m.size)
        p, t = p[m], t[m]
    ok = np.isfinite(p) & np.isfinite(t)
    return p[ok], t[ok]


def regression_metrics(pred, target, mask=None, eps: float = 1.0) -> dict:
    p, t = _flat(pred, target, mask)
    if p.size == 0:
        return {"MAE": None, "RMSE": None, "MAPE": None, "R2": None, "n": 0}
    err = p - t
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mape = float(np.mean(np.abs(err) / np.clip(np.abs(t), eps, None)) * 100.0)
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((t - t.mean()) ** 2)) or 1e-9
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4),
            "MAPE": round(mape, 4), "R2": round(1 - ss_res / ss_tot, 4),
            "n": int(p.size)}


def horizon_metrics(pred, target) -> list[dict]:
    """pred/target: (num, H, N, 1). Returns one metric dict per horizon step."""
    pred = np.asarray(pred, float)
    target = np.asarray(target, float)
    H = pred.shape[1]
    out = []
    for h in range(H):
        m = regression_metrics(pred[:, h], target[:, h])
        m["horizon"] = h + 1
        out.append(m)
    return out


def calibration_metrics(mean, std, target, z_values=(1.0, 1.28, 1.64, 1.96),
                        mask=None) -> dict:
    """Prediction-interval coverage (PICP), mean interval width (MPIW),
    uncertainty-error correlation and a calibration curve."""
    m, t = _flat(mean, target, mask)
    s, _ = _flat(std, target, mask)
    if m.size == 0 or s.size != m.size:
        return {"PICP_95": None, "MPIW_95": None, "unc_err_corr": None,
                "calibration_curve": []}
    abs_err = np.abs(m - t)
    s = np.clip(s, 1e-6, None)
    curve = []
    for z in z_values:
        cov = float(np.mean(abs_err <= z * s))
        nominal = float(_std_normal_cdf(z) * 2 - 1)
        curve.append({"z": z, "nominal": round(nominal, 4),
                      "empirical": round(cov, 4)})
    corr = float(np.corrcoef(s, abs_err)[0, 1]) if s.std() > 0 else 0.0
    picp = float(np.mean(abs_err <= 1.96 * s))
    mpiw = float(np.mean(2 * 1.96 * s))
    return {"PICP_95": round(picp, 4), "MPIW_95": round(mpiw, 4),
            "unc_err_corr": round(corr, 4), "calibration_curve": curve}


def _std_normal_cdf(x):
    from math import erf, sqrt
    return 0.5 * (1 + erf(x / sqrt(2)))


def aoi_error_relationship(aoi, abs_err, n_bins: int = 8) -> dict:
    """Bin absolute error by AoI and report the trend (spec H3)."""
    a = np.asarray(aoi, float).reshape(-1)
    e = np.asarray(abs_err, float).reshape(-1)
    ok = np.isfinite(a) & np.isfinite(e)
    a, e = a[ok], e[ok]
    if a.size < n_bins:
        return {"pearson_r": None, "bins": []}
    r = float(np.corrcoef(a, e)[0, 1]) if a.std() > 0 else 0.0
    qs = np.quantile(a, np.linspace(0, 1, n_bins + 1))
    bins = []
    for i in range(n_bins):
        lo, hi = qs[i], qs[i + 1]
        m = (a >= lo) & (a <= hi) if i == n_bins - 1 else (a >= lo) & (a < hi)
        if m.sum():
            bins.append({"aoi_lo": round(float(lo), 2), "aoi_hi": round(float(hi), 2),
                         "mean_abs_err": round(float(e[m].mean()), 4),
                         "n": int(m.sum())})
    return {"pearson_r": round(r, 4), "bins": bins}
