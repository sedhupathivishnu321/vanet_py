#!/usr/bin/env python
"""STEP 31-34 / EXPERIMENT (control): closed-loop traffic-signal control
(spec sections 30-34, 46).

Runs each controller in closed loop on the IDM micro-simulator (TraCI when a
SUMO network is available) for every seed and reports mobility + safety +
computational metrics.

Writes:
    outputs/tables/control_results.csv / _raw.csv / .md
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from _common import base_parser, load, env_info


class _AoIInject:
    def __init__(self, ctrl, aoi_level):
        self.ctrl = ctrl
        self.aoi_level = float(aoi_level)

    def __call__(self, ctx):
        ctx = dict(ctx)
        ctx.setdefault("mean_aoi", self.aoi_level)
        return int(self.ctrl(ctx))

    def reset(self):
        if hasattr(self.ctrl, "reset"):
            self.ctrl.reset()


def _build_predictor(cfg, n_cells, device):
    import torch
    from src.models import build_model
    proc = Path(cfg["_meta"]["repo_root"]) / "data" / "processed" / "target_domain.npz"
    ck = Path(cfg["_meta"]["repo_root"]) / "models" / "checkpoints" / "target_proposed.pt"
    if not (proc.exists() and ck.exists()):
        return None
    d = np.load(proc, allow_pickle=True)
    if int(d["n_cells"]) != n_cells:
        return None
    from src.utils import target_window
    blob = torch.load(ck, map_location="cpu")
    model = build_model("proposed", 8, n_cells,
                        target_window(cfg)[1], cfg).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    from src.experiments import TorchPredictorAdapter
    recon = json.loads(str(d["recon_maps"])) if "recon_maps" in d else {}
    adj_t = None
    from src.experiments import chain_adjacency
    from src.preprocessing.graph import normalize_adjacency
    adj_t = torch.from_numpy(normalize_adjacency(chain_adjacency(n_cells))).to(device)
    return TorchPredictorAdapter(model, adj_t, float(d["scaler_mean"]),
                                 float(d["scaler_std"]), cfg, recon_maps=recon,
                                 mc_samples=int(cfg["uncertainty"]["mc_samples"]),
                                 device=device)


def main() -> int:
    args = base_parser(__doc__).parse_args()
    cfg, log = load(args)
    import torch
    from src.utils import set_seed, ExperimentLogger
    from src.utils.config import resolve_device
    from src.experiments import load_corridor
    from src.sumo import run_target_simulation
    from src.control import build_controller
    from src.control.rl import DQNController
    from src.evaluation import aggregate_seeds, paired_t_test, wilcoxon_test, cohens_d

    device = resolve_device(cfg)
    seeds = list(cfg["experiments"]["seeds"])
    corridor_id = cfg["experiments"]["corridor_for_sweeps"]
    demand = cfg["experiments"].get("demand_for_control", cfg["experiments"]["demand_for_vanet_sweep"])
    controllers = list(cfg["control"]["controllers"])
    out = Path(cfg["_meta"]["repo_root"]) / cfg["project"]["outputs_dir"]
    tdir = out / "tables"; tdir.mkdir(parents=True, exist_ok=True)

    corridor = load_corridor(cfg, corridor_id)
    n_cells = max(int(corridor.get("n_segments", corridor.get("n_edges", 8))), 4)

    # AoI level for the risk-aware controller (from the default VANET run)
    aoi_level = 20.0
    vd = Path(cfg["_meta"]["repo_root"]) / "data" / "processed" / "vanet_default.npz"
    if vd.exists():
        try:
            aoi_level = float(np.nanmean(np.load(vd)["aoi"]))
        except Exception:
            pass

    predictor = _build_predictor(cfg, n_cells, device)
    if predictor is None:
        log.warning("no target predictor available - MPC/proposed will use the "
                    "recent-history fallback inside the controllers.")

    dqn_ck = Path(cfg["_meta"]["repo_root"]) / "models/checkpoints/dqn_controller.pt"

    raw = []
    metric_keys = ["travel_time_mean_s", "delay_mean_s", "queue_mean_veh",
                   "queue_max_veh", "throughput_veh_per_h", "avg_speed_mps",
                   "min_ttc_s", "mean_ttc_s", "ttc_violations", "stops_mean"]

    for name in controllers:
        for seed in seeds:
            set_seed(seed)
            if name == "dqn":
                if dqn_ck.exists():
                    base = DQNController(cfg, weights_path=str(dqn_ck))
                else:
                    base = build_controller("max_pressure", cfg)
                    log.warning("  dqn checkpoint missing -> max-pressure fallback")
            else:
                base = build_controller(name, cfg, predictor=predictor)
            ctrl = _AoIInject(base, aoi_level)
            t0 = time.time()
            sim = run_target_simulation(cfg, corridor, demand, seed,
                                        controller=ctrl, logger=None)
            wall = time.time() - t0
            row = {"controller": name, "seed": seed,
                   "wall_time_s": round(wall, 2), "backend": sim.backend}
            for k in metric_keys:
                row[k] = sim.metrics.get(k)
            raw.append(row)
            log.info(f"  {name:13s} seed={seed}  "
                     f"tt={row['travel_time_mean_s']}  q={row['queue_mean_veh']}  "
                     f"minTTC={row['min_ttc_s']}  viol={row['ttc_violations']}")

    df = pd.DataFrame(raw)
    df.to_csv(tdir / "control_results_raw.csv", index=False)

    agg_rows = []
    for name, g in df.groupby("controller"):
        a = aggregate_seeds(g.to_dict("records"), metric_keys + ["wall_time_s"])
        row = {"controller": name, "n_seeds": len(g)}
        for k, v in a.items():
            row[f"{k}_mean"] = v["mean"]; row[f"{k}_std"] = v["std"]
        agg_rows.append(row)
    agg = pd.DataFrame(agg_rows)
    order = {c: i for i, c in enumerate(controllers)}
    agg = agg.sort_values("controller", key=lambda s: s.map(order))
    agg.to_csv(tdir / "control_results.csv", index=False)
    (tdir / "control_results.md").write_text(agg.to_markdown(index=False))

    # significance: proposed vs each other controller on travel time (spec 47)
    sig = []
    if "proposed" in df["controller"].values:
        base = df[df["controller"] == "proposed"]["travel_time_mean_s"].dropna().values
        for name, g in df.groupby("controller"):
            if name == "proposed":
                continue
            other = g["travel_time_mean_s"].dropna().values
            if len(base) >= 2 and len(other) >= 2:
                sig.append({"comparison": f"proposed vs {name}",
                            "metric": "travel_time_mean_s",
                            **{f"ttest_{k}": v for k, v in paired_t_test(base, other).items()},
                            **{f"wilcoxon_{k}": v for k, v in wilcoxon_test(base, other).items()},
                            "cohens_d": round(cohens_d(base, other), 4)})
    if sig:
        pd.DataFrame(sig).to_csv(tdir / "control_significance.csv", index=False)

    with ExperimentLogger(out / "logs", "evaluate_control", dict(cfg), env_info(),
                          seeds[0]) as elog:
        elog.add_metrics(controllers=controllers, backend=str(df["backend"].iloc[0]),
                         aoi_level=aoi_level)
        elog.add_artifact(tdir / "control_results.csv")

    log.info("\n" + agg.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
