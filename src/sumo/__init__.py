"""Target-domain traffic simulation.

`run_target_simulation(...)` returns a :class:`SimulationOutput` regardless of
backend.  It uses SUMO+TraCI when the SUMO binaries and a built network are
available, otherwise it falls back to the built-in IDM corridor micro-simulator
(:mod:`src.sumo.idm_sim`).  The backend actually used is recorded in
``SimulationOutput.backend`` and stamped into every downstream artefact.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from .idm_sim import IDMSimulator, SimulationOutput
from .build_network import sumo_available, build_sumo_network
from .demand import demand_schedule


def run_target_simulation(cfg, corridor: dict, demand_level: str, seed: int,
                          controller=None, logger=None) -> SimulationOutput:
    want_sumo = bool(cfg["sumo"].get("use_sumo_if_available", True))
    net = Path(cfg["_meta"]["repo_root"]) / cfg["sumo"]["net_xml"]
    if want_sumo and sumo_available() and net.exists():
        try:
            from .traci_runner import run_sumo
            return run_sumo(cfg, corridor, demand_level, seed, controller, logger)
        except Exception as exc:  # pragma: no cover - SUMO env dependent
            if logger:
                logger.warning(f"SUMO backend failed ({exc}); using IDM fallback")
    sim = IDMSimulator(cfg, corridor, demand_level, seed, controller)
    return sim.run()


__all__ = [
    "run_target_simulation",
    "IDMSimulator",
    "SimulationOutput",
    "sumo_available",
    "build_sumo_network",
    "demand_schedule",
]
