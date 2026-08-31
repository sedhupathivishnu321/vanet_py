"""Built-in IDM corridor micro-simulator (fallback for SUMO in Codespaces).

A single study corridor is modelled as a 1-D road of length = OSM route length,
discretised into ``N`` cells (one per OSM route segment).  Vehicles follow the
Intelligent Driver Model.  One signalised intersection sits at the corridor
mid-point with a competing cross-street queue, giving the traffic-signal
controller something real to optimise.

This produces:
  * per-aggregation-bin traffic-state tensor  (T, N, 4) = [speed, density, flow, queue]
  * per-bin vehicle snapshots for the VANET layer
  * mobility / safety metrics (travel time, delay, stops, queue, throughput, TTC)

It is deterministic given ``seed``.  Everything it outputs is *simulated
target-domain traffic* and is labelled as such.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import numpy as np

from .demand import demand_schedule

STATE_NAMES = ["mean_speed_mps", "density_veh_per_km", "flow_veh_per_h", "queue_veh"]
VEH_COLS = ["veh_id", "cell", "x_m", "speed_mps", "accel_mps2", "heading_deg", "gen_time_s"]

# IDM parameters (passenger car, urban)
IDM = dict(T=1.5, a=1.5, b=2.0, s0=2.0, delta=4.0, veh_len=5.0)


@dataclass
class SimulationOutput:
    backend: str
    corridor_id: str
    corridor_name: str
    demand_level: str
    seed: int
    n_cells: int
    cell_length_m: float
    corridor_length_m: float
    dt: float
    agg_interval_s: float
    times: np.ndarray
    state: np.ndarray               # (T, N, 4)  -- ground-truth traffic state
    state_names: list
    vehicle_frames: list            # list[T] of (n_i, 7) arrays
    heading_deg: float
    free_flow_time_s: float
    phase_log: list
    metrics: dict = field(default_factory=dict)
    is_simulated_target_traffic: bool = True

    def summary(self) -> dict:
        d = {k: v for k, v in asdict(self).items()
             if k not in ("times", "state", "vehicle_frames")}
        d["n_bins"] = int(len(self.times))
        return d


def _bearing(lat1, lon1, lat2, lon2):
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(math.radians(lat2))
    x = (math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) -
         math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(dlon))
    return (math.degrees(math.atan2(y, x)) + 360) % 360


class IDMSimulator:
    def __init__(self, cfg, corridor: dict, demand_level: str, seed: int,
                 controller=None):
        self.cfg = cfg
        self.corr = corridor
        self.demand_level = demand_level
        self.seed = int(seed)
        self.controller = controller
        self.rng = np.random.default_rng(seed)

        s = cfg["sumo"]
        self.dt = float(s["step_length_s"])
        self.warmup = float(s["warmup_s"])
        self.duration = float(s["sim_duration_s"])
        self.agg = float(s["aggregate_interval_s"])
        self.decision = float(s.get("control_decision_interval_s", 10.0))
        self.L = float(corridor.get("length_m") or 2000.0)
        self.L = max(self.L, 400.0)
        self.N = max(int(corridor.get("n_edges") or corridor.get("n_segments") or 8), 4)
        self.cell_len = self.L / self.N
        self.tl_pos = self.L * 0.5
        self.tl_cell = min(int(self.tl_pos / self.cell_len), self.N - 1)

        tl = s["traffic_light"]
        self.cycle = float(tl["cycle_s"])
        self.min_green = float(tl["min_green_s"])
        self.max_green = float(tl["max_green_s"])

        # desired speed from OSM speed limit if present, else 40 km/h
        v0 = None
        for e in corridor.get("edges", []):
            for key in ("speed_limit_kph", "speed_est_kph"):
                if e.get(key):
                    v0 = float(e[key]); break
            if v0:
                break
        self.v0 = (v0 or 40.0) / 3.6
        self.sat_flow = 0.5           # veh/s saturation flow (cross + main discharge)

        n0 = tuple(corridor.get("route_coords", [[79.83, 11.93]])[0])
        n1 = tuple(corridor.get("route_coords", [[79.84, 11.94]])[-1])
        self.heading = _bearing(n0[1], n0[0], n1[1], n1[0])

        dl = cfg["sumo"]["demand_levels"]
        n_veh = int(dl.get(demand_level, dl.get("MEDIUM", 200)))
        self.sched = demand_schedule(n_veh, self.duration, self.warmup, seed)

    # ------------------------------------------------------------------ #
    def _idm_accel(self, v, gap, dv):
        s_star = IDM["s0"] + max(0.0, v * IDM["T"] + v * dv /
                                 (2 * math.sqrt(IDM["a"] * IDM["b"])))
        gap = max(gap, 0.1)
        return IDM["a"] * (1 - (v / self.v0) ** IDM["delta"] - (s_star / gap) ** 2)

    def _phase_fixed(self, t):
        return 0 if (t % self.cycle) < 0.6 * self.cycle else 1

    def run(self) -> SimulationOutput:
        dt, L, N = self.dt, self.L, self.N
        departures = list(self.sched["main_departures"])
        dep_i = 0
        # active vehicle arrays
        vid = np.zeros(0, int)
        x = np.zeros(0); v = np.zeros(0); acc = np.zeros(0); tin = np.zeros(0)
        moving = np.zeros(0, bool); stops = np.zeros(0, int)
        next_id = 0
        q_cross = 0.0

        phase = 0
        phase_elapsed = 0.0
        phase_log = []

        state_bins = []
        veh_frames = []
        bin_times = []
        travel_times = []
        delays = []
        stops_on_exit = []
        exit_count = 0
        ttc_samples = []
        min_ttc = math.inf

        n_steps = int(self.duration / dt)
        agg_every = max(int(self.agg / dt), 1)
        decide_every = max(int(self.decision / dt), 1)
        state_hist = []
        for step in range(n_steps):
            t = step * dt

            # --- spawn --------------------------------------------------
            while dep_i < len(departures) and departures[dep_i] <= t:
                vid = np.append(vid, next_id)
                x = np.append(x, 0.0); v = np.append(v, self.v0 * 0.8)
                acc = np.append(acc, 0.0); tin = np.append(tin, t)
                moving = np.append(moving, True); stops = np.append(stops, 0)
                next_id += 1
                dep_i += 1

            # --- signal control ---------------------------------------
            if step % decide_every == 0:
                ctx = {
                    "t": t, "phase": phase, "phase_elapsed": phase_elapsed,
                    "queue_main": float(np.sum((v < 0.5) &
                                               (x < self.tl_pos) &
                                               (x > self.tl_pos - 200))),
                    "queue_cross": float(q_cross),
                    "state_history": np.array(state_hist[-64:]) if state_hist else None,
                    "n_cells": N, "tl_cell": self.tl_cell,
                }
                if self.controller is not None:
                    want = int(self.controller(ctx))
                else:
                    want = self._phase_fixed(t)
                if want != phase and phase_elapsed >= self.min_green:
                    phase, phase_elapsed = want, 0.0
                elif phase_elapsed >= self.max_green:
                    phase, phase_elapsed = 1 - phase, 0.0
            phase_elapsed += dt
            phase_log.append((round(t, 1), phase))

            # --- car following ---------------------------------------
            if len(x):
                order = np.argsort(-x)
                for rank, idx in enumerate(order):
                    if rank == 0:
                        lead_x, lead_v = x[idx] + 1e4, self.v0
                    else:
                        j = order[rank - 1]
                        lead_x, lead_v = x[j], v[j]
                    gap = lead_x - x[idx] - IDM["veh_len"]
                    dv = v[idx] - lead_v
                    # red light acts as a stopped leader at tl_pos
                    if phase == 1 and x[idx] < self.tl_pos:
                        lgap = self.tl_pos - x[idx] - 1.0
                        if lgap < gap:
                            gap, dv = lgap, v[idx]
                    a_i = self._idm_accel(max(v[idx], 0.0), gap, dv)
                    a_i = float(np.clip(a_i, -8.0, IDM["a"]))
                    acc[idx] = a_i
                    v[idx] = max(0.0, v[idx] + a_i * dt)
                    x[idx] += v[idx] * dt
                    # stop counting
                    if v[idx] < 0.3 and moving[idx]:
                        stops[idx] += 1
                        moving[idx] = False
                    elif v[idx] > 1.0:
                        moving[idx] = True
                    # TTC vs leader
                    if rank > 0:
                        rel = v[idx] - lead_v
                        if rel > 0.1:
                            gg = max(lead_x - x[idx] - IDM["veh_len"], 0.0)
                            ttc = gg / rel
                            ttc_samples.append(ttc)
                            min_ttc = min(min_ttc, ttc)

            # --- exits -----------------------------------------------
            done = x >= L
            if np.any(done):
                for k in np.where(done)[0]:
                    tt = t - tin[k]
                    travel_times.append(tt)
                    delays.append(max(0.0, tt - L / self.v0))
                    stops_on_exit.append(int(stops[k]))
                exit_count += int(done.sum())
                keep = ~done
                vid, x, v, acc, tin = vid[keep], x[keep], v[keep], acc[keep], tin[keep]
                moving, stops = moving[keep], stops[keep]

            # --- cross-street queue --------------------------------
            q_cross += self.rng.poisson(self.sched["cross_rate"] * dt)
            if phase == 1:
                q_cross = max(0.0, q_cross - self.sat_flow * dt)

            # --- aggregation --------------------------------------
            if step % agg_every == 0 and t >= self.warmup:
                cells = np.clip((x / self.cell_len).astype(int), 0, N - 1)
                sp = np.zeros(N); den = np.zeros(N); fl = np.zeros(N); qu = np.zeros(N)
                for c in range(N):
                    m = cells == c
                    cnt = int(m.sum())
                    den[c] = cnt / self.cell_len * 1000.0
                    if cnt:
                        sp[c] = float(v[m].mean())
                        qu[c] = float(np.sum(v[m] < 0.5))
                    else:
                        sp[c] = self.v0
                    fl[c] = den[c] * sp[c] * 3.6            # veh/h
                st = np.stack([sp, den, fl, qu], axis=-1)
                state_bins.append(st)
                state_hist.append(st)
                bin_times.append(t)
                # vehicle snapshot for VANET
                if len(x):
                    frame = np.stack([
                        vid.astype(float), cells.astype(float), x.copy(),
                        v.copy(), acc.copy(),
                        np.full(len(x), self.heading), np.full(len(x), t),
                    ], axis=-1)
                else:
                    frame = np.zeros((0, len(VEH_COLS)))
                veh_frames.append(frame)

        state = np.asarray(state_bins, np.float32) if state_bins else \
            np.zeros((0, N, 4), np.float32)
        times = np.asarray(bin_times, np.float32)
        avg_speed = float(np.mean(state[..., 0])) if state.size else 0.0
        metrics = {
            "n_vehicles_departed": int(next_id),
            "n_vehicles_exited": int(exit_count),
            "throughput_veh_per_h": round(exit_count / max(self.duration, 1) * 3600, 1),
            "travel_time_mean_s": round(float(np.mean(travel_times)), 2) if travel_times else None,
            "travel_time_p95_s": round(float(np.percentile(travel_times, 95)), 2) if travel_times else None,
            "delay_mean_s": round(float(np.mean(delays)), 2) if delays else None,
            "stops_mean": round(float(np.mean(stops_on_exit)), 3) if stops_on_exit else None,
            "queue_mean_veh": round(float(state[..., 3].mean()), 3) if state.size else None,
            "queue_max_veh": round(float(state[..., 3].max()), 2) if state.size else None,
            "avg_speed_mps": round(avg_speed, 3),
            "min_ttc_s": round(min_ttc, 3) if math.isfinite(min_ttc) else None,
            "mean_ttc_s": round(float(np.mean(ttc_samples)), 3) if ttc_samples else None,
            "ttc_p5_s": round(float(np.percentile(ttc_samples, 5)), 3) if ttc_samples else None,
            "ttc_violations": int(np.sum(np.asarray(ttc_samples) < self.cfg["control"]["minimum_ttc"])) if ttc_samples else 0,
        }
        return SimulationOutput(
            backend="idm_fallback", corridor_id=self.corr.get("id", "corridor"),
            corridor_name=self.corr.get("name", ""), demand_level=self.demand_level,
            seed=self.seed, n_cells=N, cell_length_m=round(self.cell_len, 2),
            corridor_length_m=round(self.L, 1), dt=self.dt, agg_interval_s=self.agg,
            times=times, state=state, state_names=STATE_NAMES,
            vehicle_frames=veh_frames, heading_deg=round(self.heading, 1),
            free_flow_time_s=round(self.L / self.v0, 1), phase_log=phase_log,
            metrics=metrics,
        )
