"""Export an IDM / SUMO :class:`SimulationOutput` to an ns-2 mobility trace
(``.tcl``) that ns-3's ``Ns2MobilityHelper`` can consume.

The corridor is laid out as a straight 1-D line: a vehicle's position ``x`` (m
along the corridor) maps to planar ``(X=x, Y=lane_offset)``.  Only the
*equipped* (connected) vehicles are exported -- they are the ns-3 nodes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class MobilityExport:
    path: Path
    n_nodes: int
    node_to_veh: dict           # ns3 node index -> vehicle id
    veh_to_node: dict
    equipped_veh_ids: list
    duration_s: float
    corridor_length_m: float


def _vehicle_tracks(sim_out):
    """veh_id -> list of (t, x_m, speed_mps) sorted by t."""
    tracks: dict[int, list] = {}
    times = np.asarray(sim_out.times, float)
    for k, frame in enumerate(sim_out.vehicle_frames):
        t = float(times[k]) if k < len(times) else float(k)
        for row in frame:
            vid = int(row[0])
            tracks.setdefault(vid, []).append((t, float(row[2]), float(row[3])))
    for v in tracks.values():
        v.sort()
    return tracks


def export_ns2_mobility(sim_out, out_path: str | Path, penetration: float,
                        seed: int, lane_offsets=(-1.0, 0.0, 1.0)) -> MobilityExport:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tracks = _vehicle_tracks(sim_out)
    veh_ids = sorted(tracks)
    rng = np.random.default_rng(seed)
    k = max(1, int(round(float(penetration) * len(veh_ids))))
    equipped = sorted(rng.choice(veh_ids, size=min(k, len(veh_ids)), replace=False).tolist())

    node_to_veh = {i: v for i, v in enumerate(equipped)}
    veh_to_node = {v: i for i, v in node_to_veh.items()}
    L = float(sim_out.corridor_length_m)
    dur = float(sim_out.times[-1]) if len(sim_out.times) else 0.0

    lines = []
    for i, vid in node_to_veh.items():
        tr = tracks[vid]
        y = lane_offsets[vid % len(lane_offsets)]
        x0 = min(max(tr[0][1], 0.0), L)
        lines.append(f'$node_({i}) set X_ {x0:.2f}')
        lines.append(f'$node_({i}) set Y_ {y:.2f}')
        lines.append(f'$node_({i}) set Z_ 0')
        for (t0, x0, _), (t1, x1, _) in zip(tr, tr[1:]):
            dt = max(t1 - t0, 1e-3)
            spd = abs(x1 - x0) / dt
            x1c = min(max(x1, 0.0), L)
            lines.append(f'$ns_ at {t0:.2f} "$node_({i}) setdest {x1c:.2f} {y:.2f} {spd:.2f}"')
        # park on exit
        tl, xl, _ = tr[-1]
        lines.append(f'$ns_ at {tl:.2f} "$node_({i}) setdest {min(max(xl,0.0),L):.2f} {y:.2f} 0.00"')

    out_path.write_text("\n".join(lines) + "\n")
    return MobilityExport(path=out_path, n_nodes=len(equipped),
                          node_to_veh=node_to_veh, veh_to_node=veh_to_node,
                          equipped_veh_ids=equipped, duration_s=dur,
                          corridor_length_m=L)


def vehicle_state_at(sim_out, veh_id: int, t: float):
    """(cell, speed_mps, x_m) of `veh_id` at the frame nearest time `t`, or None."""
    times = np.asarray(sim_out.times, float)
    if not len(times):
        return None
    k = int(np.argmin(np.abs(times - t)))
    for row in sim_out.vehicle_frames[k]:
        if int(row[0]) == int(veh_id):
            return int(row[1]), float(row[3]), float(row[2])
    # search neighbouring frames
    for kk in (k - 1, k + 1, k - 2, k + 2):
        if 0 <= kk < len(sim_out.vehicle_frames):
            for row in sim_out.vehicle_frames[kk]:
                if int(row[0]) == int(veh_id):
                    return int(row[1]), float(row[3]), float(row[2])
    return None
