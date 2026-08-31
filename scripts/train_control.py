#!/usr/bin/env python
"""STEP 27-31: prepare the traffic-signal controllers (spec sections 30-33).

Max-Pressure / MPC / the proposed risk-aware controller are training-free
(MPC and proposed wrap the already-trained target predictor).  This script
trains the one RL baseline (DQN) against the IDM micro-simulator and saves it.

Writes:
    models/checkpoints/dqn_controller.pt
    outputs/tables/dqn_training.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from _common import base_parser, load, env_info


def main() -> int:
    args = base_parser(__doc__).parse_args()
    cfg, log = load(args)
    from src.utils import set_seed, ExperimentLogger
    from src.experiments import load_corridor
    from src.control import train_dqn

    seed = int(cfg["project"]["seed_default"])
    corridor_id = cfg["experiments"]["corridor_for_sweeps"]
    demand = cfg["experiments"].get("demand_for_control", cfg["experiments"]["demand_for_vanet_sweep"])
    ck = Path(cfg["_meta"]["repo_root"]) / "models" / "checkpoints"
    ck.mkdir(parents=True, exist_ok=True)
    out = Path(cfg["_meta"]["repo_root"]) / cfg["project"]["outputs_dir"]

    set_seed(seed)
    corridor = load_corridor(cfg, corridor_id)
    log.info(f"training DQN controller on {corridor_id} / {demand} "
             f"({cfg['control']['dqn']['episodes']} episodes)")
    with ExperimentLogger(out / "logs", "train_control", dict(cfg), env_info(),
                          seed) as elog:
        try:
            ctrl = train_dqn(cfg, corridor, demand, seed=seed, logger=log)
            ctrl.save(ck / "dqn_controller.pt")
            elog.add_artifact(ck / "dqn_controller.pt")
            log.info(f"saved -> {ck/'dqn_controller.pt'}")
        except Exception as exc:
            log.error(f"DQN training failed ({exc}); evaluate_control.py will "
                      f"fall back to a max-pressure policy for the 'dqn' row.")
            elog.note(f"DQN training failed: {exc}")
            return 0

    pd.DataFrame([{"episodes": cfg["control"]["dqn"]["episodes"],
                   "corridor": corridor_id, "demand": demand, "seed": seed}]
                 ).to_csv(out / "tables" / "dqn_training.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
