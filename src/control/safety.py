"""Safety metric: Time-To-Collision (spec section 34).

    TTC = d / max(v_ego - v_lead, eps)     when the follower is closing.
"""
from __future__ import annotations

import numpy as np

VEH_LEN = 5.0


def ttc(d: float, v_ego: float, v_lead: float, eps: float = 0.1) -> float:
    rel = v_ego - v_lead
    if rel <= eps:
        return np.inf
    return max(d, 0.0) / rel


def ttc_from_frame(frame: np.ndarray, eps: float = 0.1) -> list[float]:
    """frame rows = [veh_id, cell, x, speed, accel, heading, gen_time].
    Returns TTC for every closing follower-leader pair on the corridor."""
    if len(frame) < 2:
        return []
    order = np.argsort(frame[:, 2])          # by x ascending
    xs = frame[order, 2]
    vs = frame[order, 3]
    out = []
    for i in range(len(order) - 1):
        d = xs[i + 1] - xs[i] - VEH_LEN
        out.append(ttc(d, vs[i], vs[i + 1], eps))
    return [t for t in out if np.isfinite(t)]


def count_ttc_violations(ttc_values, ttc_min: float = 2.0) -> int:
    return int(np.sum(np.asarray(ttc_values, float) < ttc_min))


def summarize_ttc(ttc_values, ttc_min: float = 2.0) -> dict:
    a = np.asarray([t for t in ttc_values if np.isfinite(t)], float)
    if a.size == 0:
        return {"min_ttc": None, "mean_ttc": None, "p5_ttc": None, "violations": 0}
    return {
        "min_ttc": round(float(a.min()), 3),
        "mean_ttc": round(float(a.mean()), 3),
        "p5_ttc": round(float(np.percentile(a, 5)), 3),
        "violations": count_ttc_violations(a, ttc_min),
    }
