"""Optional SUMO + TraCI backend (spec section 16, 31).

Only used when the SUMO binaries and a built ``puducherry.net.xml`` are present.
Given the corridor route it builds a temporary ``.rou.xml`` for the requested
demand level, runs SUMO head-less, drives the first traffic light through TraCI
with the supplied controller callback, and returns the same
:class:`SimulationOutput` structure as the IDM fallback so downstream code is
backend-agnostic.

This module deliberately degrades to a clear exception (caught by
``run_target_simulation``) rather than guessing when the SUMO scenario cannot
be assembled.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .idm_sim import SimulationOutput, STATE_NAMES, VEH_COLS


def _import_traci():
    if "SUMO_HOME" in os.environ:
        tools = os.path.join(os.environ["SUMO_HOME"], "tools")
        if tools not in os.sys.path:
            os.sys.path.append(tools)
    import traci  # noqa
    return traci


def run_sumo(cfg, corridor: dict, demand_level: str, seed: int,
             controller=None, logger=None) -> SimulationOutput:
    traci = _import_traci()
    root = Path(cfg["_meta"]["repo_root"])
    net = root / cfg["sumo"]["net_xml"]
    work = root / "data" / "sumo"
    work.mkdir(parents=True, exist_ok=True)

    # --- route file over the corridor OSM node path ---------------------
    route_nodes = corridor.get("route_nodes", [])
    if len(route_nodes) < 2:
        raise RuntimeError("corridor has no route_nodes for SUMO routing")
    # SUMO edge ids from an OSM-derived net are not the OSM node ids; we let
    # duarouter find a path between the geo-nearest edges instead.
    n_veh = int(cfg["sumo"]["demand_levels"].get(demand_level, 200))
    dur = float(cfg["sumo"]["sim_duration_s"])
    trips = work / f"{corridor['id']}_{demand_level}.trips.xml"
    routes = work / f"{corridor['id']}_{demand_level}.rou.xml"
    o_lat, o_lon = corridor["origin_latlon"]
    d_lat, d_lon = corridor["destination_latlon"]

    import subprocess
    with open(trips, "w") as fh:
        fh.write("<routes>\n")
        step = dur / max(n_veh, 1)
        for i in range(n_veh):
            fh.write(f'  <trip id="v{i}" depart="{i*step:.2f}" '
                     f'fromXY="{o_lon},{o_lat}" toXY="{d_lon},{d_lat}"/>\n')
        fh.write("</routes>\n")
    subprocess.run(["duarouter", "-n", str(net), "--route-files", str(trips),
                    "-o", str(routes), "--ignore-errors", "true",
                    "--repair", "true"], check=True)

    sumo_bin = "sumo"
    cmd = [sumo_bin, "-n", str(net), "-r", str(routes), "--no-warnings", "true",
           "--seed", str(seed), "--step-length", str(cfg["sumo"]["step_length_s"]),
           "--time-to-teleport", "-1", "--start", "--quit-on-end"]
    traci.start(cmd)

    agg = float(cfg["sumo"]["aggregate_interval_s"])
    dt = float(cfg["sumo"]["step_length_s"])
    tls = traci.trafficlight.getIDList()
    ctrl_tls = tls[0] if tls else None

    times, state_bins, veh_frames = [], [], []
    travel_times = []
    t = 0.0
    while traci.simulation.getMinExpectedNumber() > 0 and t < dur:
        traci.simulationStep()
        t = traci.simulation.getTime()
        for vid in traci.simulation.getArrivedIDList():
            pass  # arrivals; SUMO tracks trip info via output if enabled
        if ctrl_tls and controller and abs(t % agg) < dt / 2:
            lanes = traci.trafficlight.getControlledLanes(ctrl_tls)
            qmain = sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes[:len(lanes)//2])
            qcross = sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes[len(lanes)//2:])
            ph = controller({"t": t, "phase": traci.trafficlight.getPhase(ctrl_tls),
                             "queue_main": qmain, "queue_cross": qcross,
                             "state_history": np.array(state_bins[-12:]) if state_bins else None})
            try:
                traci.trafficlight.setPhase(ctrl_tls, int(ph) % 2)
            except Exception:
                pass
        if abs(t % agg) < dt / 2:
            ids = traci.vehicle.getIDList()
            if ids:
                sp = np.mean([traci.vehicle.getSpeed(i) for i in ids])
            else:
                sp = 0.0
            state_bins.append(np.array([[sp, len(ids), sp * len(ids) * 3.6, 0.0]]))
            times.append(t)
            veh_frames.append(np.zeros((len(ids), len(VEH_COLS))))
    traci.close()

    state = np.asarray(state_bins, np.float32).reshape(len(state_bins), -1, 4)
    return SimulationOutput(
        backend="sumo_traci", corridor_id=corridor["id"],
        corridor_name=corridor.get("name", ""), demand_level=demand_level,
        seed=seed, n_cells=state.shape[1] if state.size else 1,
        cell_length_m=float(corridor.get("length_m", 1000)) / max(state.shape[1], 1)
        if state.size else 0.0,
        corridor_length_m=float(corridor.get("length_m", 0.0)),
        dt=dt, agg_interval_s=agg, times=np.asarray(times, np.float32),
        state=state, state_names=STATE_NAMES, vehicle_frames=veh_frames,
        heading_deg=0.0, free_flow_time_s=0.0, phase_log=[],
        metrics={"backend": "sumo_traci", "note": "SUMO metrics via tripinfo "
                 "output - enable --tripinfo-output for full mobility stats"},
    )
