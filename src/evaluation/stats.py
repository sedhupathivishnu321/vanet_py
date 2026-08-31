"""Multi-seed aggregation and significance testing (spec sections 45, 47)."""
from __future__ import annotations

import numpy as np


def mean_std_ci(values, confidence: float = 0.95) -> dict:
    v = np.asarray([x for x in values if x is not None], float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"mean": None, "std": None, "ci95": None, "n": 0}
    mean = float(v.mean())
    std = float(v.std(ddof=1)) if v.size > 1 else 0.0
    if v.size > 1:
        try:
            from scipy import stats
            tcrit = float(stats.t.ppf(0.5 + confidence / 2, v.size - 1))
        except Exception:
            tcrit = 1.96
        half = tcrit * std / np.sqrt(v.size)
    else:
        half = 0.0
    return {"mean": round(mean, 5), "std": round(std, 5),
            "ci95": [round(mean - half, 5), round(mean + half, 5)],
            "n": int(v.size)}


def aggregate_seeds(records: list[dict], keys: list[str] | None = None) -> dict:
    """`records` = list of metric dicts (one per seed). Returns
    {key: {mean,std,ci95,n}, ...}."""
    if not records:
        return {}
    keys = keys or sorted({k for r in records for k, v in r.items()
                           if isinstance(v, (int, float))})
    return {k: mean_std_ci([r.get(k) for r in records]) for k in keys}


def paired_t_test(a, b) -> dict:
    a = np.asarray(a, float); b = np.asarray(b, float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    try:
        from scipy import stats
        t, p = stats.ttest_rel(a, b)
        return {"statistic": float(t), "p_value": float(p), "n": n}
    except Exception:
        d = a - b
        t = d.mean() / (d.std(ddof=1) / np.sqrt(n) + 1e-12)
        return {"statistic": float(t), "p_value": None, "n": n}


def wilcoxon_test(a, b) -> dict:
    try:
        from scipy import stats
        s, p = stats.wilcoxon(np.asarray(a, float)[:len(b)],
                              np.asarray(b, float)[:len(a)])
        return {"statistic": float(s), "p_value": float(p)}
    except Exception:
        return {"statistic": None, "p_value": None}


def cohens_d(a, b) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return float("nan")
    sp = np.sqrt(((n1 - 1) * a.std(ddof=1) ** 2 + (n2 - 1) * b.std(ddof=1) ** 2)
                 / (n1 + n2 - 2))
    return float((a.mean() - b.mean()) / (sp + 1e-12))
