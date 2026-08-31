"""Turn a ground-truth simulation into a *partially observed* VANET dataset
(spec sections 23, 25).

Two backends produce the same :class:`PartialObservation`:

  * **analytic**  -- :func:`build_partial_observation` uses
    :class:`~src.vanet.channel.CommunicationChannel` (Bernoulli penetration/PDR
    + configurable latency).
  * **ns-3**      -- :mod:`src.vanet.ns3_channel` runs a real IEEE 802.11p BSM
    simulation and feeds the delivered beacons here.

Both call :func:`assemble_partial_obs`, which replays the delivered beacons
bin-by-bin to build:

    observed_state : (T, N, 4)  [speed, density, flow, queue] from received data
    mask           : (T, N)     1 if the cell has a fresh observation else 0
    aoi            : (T, N)     Age of Information in seconds
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .aoi import AoITracker, aoi_statistics
from .channel import Beacon


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
    backend: str = "analytic"


# --------------------------------------------------------------------------- #
def collect_beacons_analytic(sim_out, channel) -> list[Beacon]:
    """Every connected vehicle in every frame transmits; return the beacons that
    were delivered (each carrying its arrival time = gen + latency[+jitter])."""
    out: list[Beacon] = []
    for frame in sim_out.vehicle_frames:
        if len(frame):
            out.extend(channel.transmit_frame(frame))
    return out


def assemble_partial_obs(beacons: list[Beacon], sim_out, cfg, *, penetration: float,
                         pdr: float, latency_ms: float, backend: str = "analytic",
                         extra_delivery_stats: dict | None = None,
                         aoi_fresh_cap_s: float | None = None) -> PartialObservation:
    if aoi_fresh_cap_s is None:
        aoi_fresh_cap_s = float(cfg["vanet"].get("aoi_fresh_cap_s", 180.0))
    times = np.asarray(sim_out.times, float)
    N = int(sim_out.n_cells)
    cell_len = float(sim_out.cell_length_m)
    T = len(sim_out.vehicle_frames)

    beacons = sorted(beacons, key=lambda b: b.arrival_time_s)
    tracker = AoITracker(N, cap_s=float(cfg["vanet"].get("aoi_cap_s", 600.0)))
    observed = np.zeros((T, N, 4), np.float32)
    mask = np.zeros((T, N), np.float32)
    aoi = np.zeros((T, N), np.float32)
    recent_by_cell: dict[int, list] = {c: [] for c in range(N)}

    ptr, delivered = 0, 0
    for t_idx in range(T):
        now = float(times[t_idx]) if t_idx < len(times) else float(t_idx)
        while ptr < len(beacons) and beacons[ptr].arrival_time_s <= now + 1e-9:
            b = beacons[ptr]; ptr += 1
            delivered += 1
            tracker.update([b], now)
            if 0 <= b.cell < N:
                recent_by_cell[b.cell].append(b)
        cur = tracker.aoi(now)
        aoi[t_idx] = cur
        for c in range(N):
            fresh = [b for b in recent_by_cell[c]
                     if now - b.gen_time_s <= aoi_fresh_cap_s]
            recent_by_cell[c] = fresh[-48:]
            if fresh and cur[c] <= aoi_fresh_cap_s:
                sp = float(np.mean([b.speed_mps for b in fresh]))
                cnt = len({b.veh_id for b in fresh})
                den = cnt / cell_len * 1000.0
                qu = float(np.sum([b.speed_mps < 0.5 for b in fresh]))
                observed[t_idx, c] = [sp, den, den * sp * 3.6, qu]
                mask[t_idx, c] = 1.0

    gt = np.asarray(sim_out.state, np.float32)
    if gt.shape[0] != T:
        m = min(gt.shape[0], T)
        gt, observed, mask, aoi, times = gt[:m], observed[:m], mask[:m], aoi[:m], times[:m]

    settle = int(float(cfg["vanet"].get("aoi_settle_s", 0.0)) /
                 max(sim_out.agg_interval_s, 1e-6))
    settle = min(settle, max(len(aoi) - 1, 0))

    dstats = {"beacons_delivered": delivered,
              "mean_cell_coverage": round(float(mask[settle:].mean())
                                          if len(mask) > settle else 0.0, 4)}
    if extra_delivery_stats:
        dstats.update(extra_delivery_stats)

    return PartialObservation(
        observed_state=observed, mask=mask, aoi=aoi, ground_truth_state=gt,
        times=times, penetration=float(penetration), pdr=float(pdr),
        latency_ms=float(latency_ms), n_cells=N, backend=backend,
        aoi_stats=aoi_statistics(aoi[settle:]), delivery_stats=dstats,
    )


def build_partial_observation(sim_out, channel, cfg,
                              aoi_fresh_cap_s: float | None = None) -> PartialObservation:
    """Analytic backend (backwards-compatible entry point)."""
    beacons = collect_beacons_analytic(sim_out, channel)
    po = assemble_partial_obs(
        beacons, sim_out, cfg, penetration=channel.penetration, pdr=channel.pdr,
        latency_ms=channel.latency_s * 1000.0, backend="analytic",
        aoi_fresh_cap_s=aoi_fresh_cap_s,
        extra_delivery_stats={"beacons_sent": channel.sent,
                              "beacons_delivered": channel.delivered,
                              "realized_pdr": round(channel.realized_pdr, 4)})
    return po
