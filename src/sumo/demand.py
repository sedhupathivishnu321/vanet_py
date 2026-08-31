"""Traffic-demand scenarios (spec section 18): LOW / MEDIUM / HIGH / VERY_HIGH.

`demand_schedule` returns Poisson departure times for `n_vehicles` spread over
`duration` seconds (after a warm-up), plus a cross-street arrival rate scaled to
the same level.
"""
from __future__ import annotations

import numpy as np


def demand_schedule(n_vehicles: int, duration_s: float, warmup_s: float,
                    seed: int) -> dict:
    rng = np.random.default_rng(seed)
    span = max(duration_s - warmup_s, 1.0)
    rate = n_vehicles / span                      # veh / s on the main corridor
    # inter-arrival ~ Exp(rate)
    times = []
    t = warmup_s
    while len(times) < n_vehicles and t < duration_s:
        t += rng.exponential(1.0 / max(rate, 1e-6))
        times.append(t)
    times = np.array(times[:n_vehicles], dtype=float)
    cross_rate = 0.35 * rate                      # cross street ~ 35% of main
    return {"main_departures": times, "main_rate": rate,
            "cross_rate": float(cross_rate), "n_vehicles": int(len(times))}
