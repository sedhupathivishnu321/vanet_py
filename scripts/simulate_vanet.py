#!/usr/bin/env python
"""STEP 16-20 / EXPERIMENTS 3-5,7: VANET communication sweeps
(spec sections 19-24, 37-41).

Evaluates the transferred target model under controlled:
    * connected-vehicle penetration
    * packet-delivery ratio (PDR)
    * communication latency
    * combined communication stress tests

Writes:
    outputs/tables/vanet_results.csv / _raw.csv / .md
    outputs/tables/vanet_stress_results.csv
    data/processed/vanet_default.npz     (AoI series + vehicle snapshot for figs/map)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _common import base_parser, load, env_info


def _load_target_model(cfg, n_cells, device):
    import torch
    from src.models import build_model
    from src.utils import target_window
    ck = Path(cfg["_meta"]["repo_root"]) / "models" / "checkpoints" / "target_proposed.pt"
    H = target_window(cfg)[1]
    model = build_model("proposed", 8, n_cells, H, cfg).to(device)
    used_pretrained = False
    if ck.exists():
        blob = torch.load(ck, map_location="cpu")
        if blob.get("num_nodes") == n_cells:
            model.load_state_dict(blob["state_dict"]); used_pretrained = True
    model.eval()
    return model, used_pretrained


def _evaluate(cfg, model, adj_t, tensors, device):
    import torch
    from src.models.engine import predict_torch
    from src.evaluation import (regression_metrics, calibration_metrics,
                                aoi_error_relationship)
    from src.uncertainty import mc_dropout_predict
    sc = tensors["scaler"]
    xk = tensors["test"]["x"]; yk = tensors["test"]["y"].numpy()
    aoik = tensors["test"]["aoi"]
    if len(xk) == 0:
        return None
    K = int(cfg["uncertainty"]["mc_samples"])
    mean, std = mc_dropout_predict(model, xk, adj_t, aoi=aoik, K=K, device=device)
    pred_d = sc.inverse_transform(mean); y_d = sc.inverse_transform(yk)
    m = regression_metrics(pred_d, y_d)
    cal = calibration_metrics(mean, std, yk)
    # AoI vs error (per-window current AoI broadcast over horizon/nodes)
    abs_err = np.abs(pred_d - y_d).mean(axis=(1, 3))          # (num, N)
    rel = aoi_error_relationship(aoik.numpy(), abs_err)
    m["PICP_95"] = cal["PICP_95"]; m["unc_err_corr"] = cal["unc_err_corr"]
    m["aoi_err_pearson"] = rel["pearson_r"]
    return m


def main() -> int:
    args = base_parser(__doc__).parse_args()
    cfg, log = load(args)
    import torch
    from src.utils import set_seed, ExperimentLogger
    from src.utils.config import resolve_device
    from src.experiments import run_scenario
    from src.evaluation import aggregate_seeds

    device = resolve_device(cfg)
    seeds = list(cfg["experiments"]["seeds"])
    corridor_id = cfg["experiments"]["corridor_for_sweeps"]
    v = cfg["vanet"]
    out = Path(cfg["_meta"]["repo_root"]) / cfg["project"]["outputs_dir"]
    proc = Path(cfg["_meta"]["repo_root"]) / "data" / "processed"
    tdir = out / "tables"; tdir.mkdir(parents=True, exist_ok=True)

    raw = []

    def _run_grid(sweep, values, key):
        for val in values:
            for seed in seeds:
                set_seed(seed)
                kw = {"penetration": v["default_penetration"],
                      "pdr": v["default_pdr"],
                      "latency_ms": v["default_latency_ms"]}
                kw[key] = val
                sc = run_scenario(cfg, corridor_id, seed, **kw)
                model, used = _load_target_model(cfg, sc["sim_out"].n_cells, device)
                adj_t = torch.from_numpy(sc["adj_norm"]).to(device)
                m = _evaluate(cfg, model, adj_t, sc["tensors"], device)
                if m is None:
                    continue
                m.update(sweep=sweep, penetration=kw["penetration"],
                         pdr=kw["pdr"], latency_ms=kw["latency_ms"], seed=seed,
                         mean_aoi=sc["partial_obs"].aoi_stats["mean"],
                         p95_aoi=sc["partial_obs"].aoi_stats["p95"],
                         cell_coverage=sc["partial_obs"].delivery_stats["mean_cell_coverage"],
                         used_pretrained=used, backend=sc["backend"])
                raw.append(m)
                log.info(f"  {sweep}={val} seed={seed}  RMSE={m['RMSE']} "
                         f"meanAoI={m['mean_aoi']} cov={m['cell_coverage']}")

    log.info("--- penetration sweep ---"); _run_grid("penetration", v["penetration"], "penetration")
    log.info("--- PDR sweep ---");         _run_grid("pdr", v["pdr"], "pdr")
    log.info("--- latency sweep ---");     _run_grid("latency", v["latency_ms"], "latency_ms")

    # stress tests (spec section 41)
    stress_rows = []
    for label, combo in v["stress_test"].items():
        for seed in seeds:
            set_seed(seed)
            sc = run_scenario(cfg, corridor_id, seed,
                              penetration=combo["penetration"], pdr=combo["pdr"],
                              latency_ms=combo["latency_ms"])
            model, used = _load_target_model(cfg, sc["sim_out"].n_cells, device)
            adj_t = torch.from_numpy(sc["adj_norm"]).to(device)
            m = _evaluate(cfg, model, adj_t, sc["tensors"], device)
            if m is None:
                continue
            m.update(scenario=label, **combo, seed=seed,
                     mean_aoi=sc["partial_obs"].aoi_stats["mean"])
            stress_rows.append(m)
            log.info(f"  stress[{label}] seed={seed} RMSE={m['RMSE']} "
                     f"meanAoI={m['mean_aoi']}")

    dfr = pd.DataFrame(raw)
    dfr.to_csv(tdir / "vanet_results_raw.csv", index=False)
    # aggregate per (sweep, swept-value)
    agg = []
    for sweep, key in [("penetration", "penetration"), ("pdr", "pdr"),
                       ("latency", "latency_ms")]:
        sub = dfr[dfr["sweep"] == sweep]
        for val, g in sub.groupby(key):
            a = aggregate_seeds(g.to_dict("records"),
                                ["RMSE", "MAE", "R2", "mean_aoi", "p95_aoi",
                                 "cell_coverage", "PICP_95", "unc_err_corr",
                                 "aoi_err_pearson"])
            row = {"sweep": sweep, key: val, "n_seeds": len(g)}
            for k, vv in a.items():
                row[f"{k}_mean"] = vv["mean"]; row[f"{k}_std"] = vv["std"]
            # ensure the plotting columns exist
            row.setdefault("penetration", None); row.setdefault("pdr", None)
            row.setdefault("latency_ms", None)
            row[key] = val
            agg.append(row)
    dfa = pd.DataFrame(agg)
    dfa.to_csv(tdir / "vanet_results.csv", index=False)
    (tdir / "vanet_results.md").write_text(dfa.to_markdown(index=False))
    if stress_rows:
        pd.DataFrame(stress_rows).to_csv(tdir / "vanet_stress_results.csv", index=False)

    # default scenario dump for figures / interactive map
    set_seed(seeds[0])
    sc = run_scenario(cfg, corridor_id, seeds[0])
    so = sc["sim_out"]
    frame = so.vehicle_frames[-1] if so.vehicle_frames else np.zeros((0, 7))
    chan = sc["partial_obs"]
    connected = np.array([1 if (int(r[0]) % 100) < int(sc["comm"]["penetration"] * 100)
                          else 0 for r in frame])
    np.savez(proc / "vanet_default.npz",
             times=so.times, aoi=chan.aoi,
             corridor_coords=np.array(sc["corridor"]["route_coords"], float),
             corridor_length_m=so.corridor_length_m,
             last_frame=frame, last_frame_connected=connected,
             comm_graph_x=frame[:, 2] if len(frame) else np.zeros(0),
             comm_graph_y=frame[:, 3] if len(frame) else np.zeros(0),
             comm_graph_connected=connected)

    with ExperimentLogger(out / "logs", "simulate_vanet", dict(cfg), env_info(),
                          seeds[0]) as elog:
        elog.add_metrics(n_conditions=len(dfa), backend=so.backend,
                         default_mean_aoi=chan.aoi_stats["mean"])
        elog.add_artifact(tdir / "vanet_results.csv")

    log.info("\n" + dfa.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
