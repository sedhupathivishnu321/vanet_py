"""Turn a ground-truth simulation into a *partially observed* VANET dataset
(spec sections 23, 25).

Only connected vehicles whose beacons are delivered (and have already arrived,
given latency) contribute to the observation.  Per cell we get:

    observed_state : (T, N, 4)  [speed, density, flow, queue] from received data
    mask           : (T, N)     1 if the cell has a fresh observation else 0
    aoi            : (T, N)     Age of Information in seconds

Ground truth (``sim_out.state``) is returned too but is for EVALUATION ONLY.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .aoi import AoITracker, aoi_statistics


@dataclass
class PartialObservation:
    observed_state: np.ndarray     # (T, N, 4)
    mask: np.ndarray               # (T, N) float {0,1}
    aoi: np.ndarray                # (T, N) seconds
    ground_truth_state: np.ndarray  # (T, N, 4)  -- evaluation only
    times: np.ndarray
    penetration: float
    pdr: float
    latency_ms: float
    aoi_stats: dict
    delivery_stats: dict
    n_cells: int


def build_partial_observation(sim_out, channel, cfg,
                              aoi_fresh_cap_s: float | None = None) -> PartialObservation:
    if aoi_fresh_cap_s is None:
        aoi_fresh_cap_s = float(cfg["vanet"].get("aoi_fresh_cap_s", 180.0))
    frames = sim_out.vehicle_frames
    times = np.asarray(sim_out.times, float)
    N = sim_out.n_cells
    cell_len = sim_out.cell_length_m
    T = len(frames)

    tracker = AoITracker(N, cap_s=float(cfg["vanet"].get("aoi_cap_s", 600.0)))
    observed = np.zeros((T, N, 4), np.float32)
    mask = np.zeros((T, N), np.float32)
    aoi = np.zeros((T, N), np.float32)

    # rolling store of the most recent received rows per cell
    recent_by_cell: dict[int, list] = {c: [] for c in range(N)}
    pending: list = []            # transmitted beacons not yet arrived (latency)

    for t_idx, frame in enumerate(frames):
        now = float(times[t_idx]) if t_idx < len(times) else t_idx
        # every connected vehicle in this frame transmits; delivery + latency
        if len(frame):
            pending.extend(channel.transmit_frame(frame))
        # release beacons whose (generation time + latency) has now elapsed
        arrived = [b for b in pending if b.arrival_time_s <= now + 1e-9]
        pending = [b for b in pending if b.arrival_time_s > now + 1e-9]
        tracker.update(arrived, now)
        for b in arrived:
            if 0 <= b.cell < N:
                recent_by_cell[b.cell].append(b)
        cur_aoi = tracker.aoi(now)
        aoi[t_idx] = cur_aoi
        for c in range(N):
            fresh = [b for b in recent_by_cell[c]
                     if now - b.gen_time_s <= aoi_fresh_cap_s]
            recent_by_cell[c] = fresh[-32:]
            if fresh and cur_aoi[c] <= aoi_fresh_cap_s:
                sp = np.mean([b.speed_mps for b in fresh])
                cnt = len(set(b.veh_id for b in fresh))
                den = cnt / cell_len * 1000.0
                qu = float(np.sum([b.speed_mps < 0.5 for b in fresh]))
                observed[t_idx, c] = [sp, den, den * sp * 3.6, qu]
                mask[t_idx, c] = 1.0
            # else: leave zeros, mask 0

    gt = np.asarray(sim_out.state, np.float32)
    if gt.shape[0] != T:                       # align lengths defensively
        m = min(gt.shape[0], T)
        gt, observed, mask, aoi = gt[:m], observed[:m], mask[:m], aoi[:m]
        times = times[:m]

    # exclude the connectivity cold-start from the AoI summary statistics
    settle = int(float(cfg["vanet"].get("aoi_settle_s", 0.0)) /
                 max(sim_out.agg_interval_s, 1e-6))
    settle = min(settle, max(len(aoi) - 1, 0))

    return PartialObservation(
        observed_state=observed, mask=mask, aoi=aoi, ground_truth_state=gt,
        times=times, penetration=channel.penetration, pdr=channel.pdr,
        latency_ms=channel.latency_s * 1000.0,
        aoi_stats=aoi_statistics(aoi[settle:]),
        delivery_stats={"beacons_sent": channel.sent,
                        "beacons_delivered": channel.delivered,
                        "realized_pdr": round(channel.realized_pdr, 4),
                        "mean_cell_coverage": round(float(mask[settle:].mean())
                                                    if len(mask) > settle else 0.0, 4)},
        n_cells=N,
    )
