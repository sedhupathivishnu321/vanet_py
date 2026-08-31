#!/usr/bin/env python
"""STEP 26 / EXPERIMENTS 6,8 + ABLATION + UNCERTAINTY
(spec sections 25-29, 40, 42, 43).

Produces:
    outputs/tables/ablation_results.csv / .md
    outputs/tables/demand_results.csv
    outputs/tables/corridor_results.csv
    outputs/tables/horizon_metrics.csv
    data/processed/uncertainty_eval.json
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _common import base_parser, load, env_info

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import target_window  # noqa: E402

_SRC_CACHE: dict = {}


def _cached_source_domain(cfg):
    if "prep" not in _SRC_CACHE:
        from src.data.source_prep import prepare_source_domain
        _SRC_CACHE["prep"] = prepare_source_domain(cfg)
    return _SRC_CACHE["prep"]


def _train_variant(cfg, scenario, opts, seed, device):
    import torch
    from src.models import build_model
    from src.models.engine import train_torch_model, predict_torch
    from src.transfer import apply_transfer_strategy
    from src.transfer.adapt import make_domain_aux_loss
    from src.experiments import load_proposed_source_state
    from src.evaluation import regression_metrics

    vcfg = copy.deepcopy(cfg)
    vcfg["proposed_model"]["aoi_attention"] = bool(opts["aoi"])
    vcfg["loss_weights"]["physics"] = float(opts["physics"])
    vcfg["loss_weights"]["domain"] = float(opts["domain"])

    tn = scenario["tensors"]
    N, H = tn["n_cells"], target_window(cfg)[1]
    adj_t = torch.from_numpy(scenario["adj_norm"]).to(device)
    model = build_model("proposed", 8, N, H, vcfg, adj=scenario["adj_norm"]).to(device)

    src_state = load_proposed_source_state(cfg)["state_dict"] if opts["transfer"] != "none" else {}
    groups, meta = apply_transfer_strategy(model, src_state, opts["transfer"])
    aux = None
    if opts["da"]:
        sp = _cached_source_domain(cfg)
        smodel = build_model("proposed", sp["in_dim"], sp["num_nodes"],
                             sp["horizon"], cfg).to(device)
        try:
            smodel.load_state_dict(load_proposed_source_state(cfg)["state_dict"])
        except Exception:
            pass
        model._da_adj_t = adj_t
        aux = make_domain_aux_loss(smodel, sp["torch"]["train"]["x"][:128].numpy(),
                                   torch.from_numpy(sp["adj_norm"]).to(device),
                                   cfg["transfer"]["adaptation_loss"],
                                   float(cfg["transfer"]["adaptation_weight"]),
                                   device, seed)
    res = train_torch_model(model, tn["train"], tn["val"], vcfg, adj_t, device,
                            full_objective=True,
                            epochs=int(cfg["transfer"]["finetune_epochs"]),
                            lr=float(cfg["transfer"]["finetune_lr"]),
                            param_groups=groups if opts["transfer"] != "none" else None,
                            speed_scale=tn["speed_scale_mps"],
                            physics_dt=float(cfg["sumo"]["aggregate_interval_s"]),
                            aux_loss_fn=aux, seed=seed)
    pred = predict_torch(model, tn["test"]["x"], adj_t, aoi=tn["test"]["aoi"],
                         device=device)
    sc = tn["scaler"]
    m = regression_metrics(sc.inverse_transform(pred),
                           sc.inverse_transform(tn["test"]["y"].numpy()))
    m["train_time_s"] = res["train_time_s"]
    return m, model, adj_t


ABLATION = {
    #                       aoi   physics domain transfer            da
    "baseline":            dict(aoi=False, physics=0.0, domain=0.0, transfer="none",             da=False),
    "baseline+AoI":        dict(aoi=True,  physics=0.0, domain=0.0, transfer="none",             da=False),
    "baseline+physics":    dict(aoi=False, physics=0.1, domain=0.0, transfer="none",             da=False),
    "baseline+transfer":   dict(aoi=False, physics=0.0, domain=0.0, transfer="frozen_encoder",   da=False),
    "baseline+DA":         dict(aoi=False, physics=0.0, domain=0.1, transfer="frozen_encoder_da", da=True),
    "full_proposed":       dict(aoi=True,  physics=0.1, domain=0.1, transfer="frozen_encoder_da", da=True),
    "full_minus_AoI":      dict(aoi=False, physics=0.1, domain=0.1, transfer="frozen_encoder_da", da=True),
    "full_minus_physics":  dict(aoi=True,  physics=0.0, domain=0.1, transfer="frozen_encoder_da", da=True),
    "full_minus_DA":       dict(aoi=True,  physics=0.1, domain=0.0, transfer="frozen_encoder",    da=False),
}


def main() -> int:
    args = base_parser(__doc__).parse_args()
    cfg, log = load(args)
    import torch
    from src.utils import set_seed, ExperimentLogger
    from src.utils.config import resolve_device
    from src.experiments import run_scenario
    from src.evaluation import (aggregate_seeds, regression_metrics,
                                horizon_metrics, calibration_metrics)
    from src.uncertainty import mc_dropout_predict

    device = resolve_device(cfg)
    seeds = list(cfg["experiments"]["seeds"])
    corridor_id = cfg["experiments"]["corridor_for_sweeps"]
    out = Path(cfg["_meta"]["repo_root"]) / cfg["project"]["outputs_dir"]
    proc = Path(cfg["_meta"]["repo_root"]) / "data" / "processed"
    tdir = out / "tables"; tdir.mkdir(parents=True, exist_ok=True)

    # ---------------- ablation ----------------
    log.info("=== ablation study ===")
    ab_raw = []
    for seed in seeds:
        set_seed(seed)
        scen = run_scenario(cfg, corridor_id, seed)
        for name, opts in ABLATION.items():
            m, _, _ = _train_variant(cfg, scen, opts, seed, device)
            m.update(variant=name, seed=seed)
            ab_raw.append(m)
            log.info(f"  [{seed}] {name:20s} RMSE={m['RMSE']} MAE={m['MAE']}")
    ab = pd.DataFrame(ab_raw)
    ab.to_csv(tdir / "ablation_results_raw.csv", index=False)
    ab_agg = []
    for name, g in ab.groupby("variant"):
        a = aggregate_seeds(g.to_dict("records"), ["RMSE", "MAE", "MAPE", "R2"])
        row = {"variant": name, "n_seeds": len(g)}
        for k, v in a.items():
            row[f"{k}_mean"] = v["mean"]; row[f"{k}_std"] = v["std"]
        ab_agg.append(row)
    ab_aggdf = pd.DataFrame(ab_agg).sort_values("RMSE_mean")
    ab_aggdf.to_csv(tdir / "ablation_results.csv", index=False)
    (tdir / "ablation_results.md").write_text(ab_aggdf.to_markdown(index=False))

    # ---------------- demand sweep (Experiment 6) ----------------
    log.info("=== traffic-demand sweep ===")
    dem_rows = []
    for level in list(cfg["sumo"]["demand_levels"].keys()):
        for seed in seeds:
            set_seed(seed)
            scen = run_scenario(cfg, corridor_id, seed, demand_level=level)
            tn = scen["tensors"]
            from src.models import build_model
            from src.models.engine import predict_torch
            model = build_model("proposed", 8, tn["n_cells"],
                                target_window(cfg)[1], cfg).to(device)
            ck = Path(cfg["_meta"]["repo_root"]) / "models/checkpoints/target_proposed.pt"
            if ck.exists() and torch.load(ck, map_location="cpu").get("num_nodes") == tn["n_cells"]:
                model.load_state_dict(torch.load(ck, map_location="cpu")["state_dict"])
            adj_t = torch.from_numpy(scen["adj_norm"]).to(device)
            if len(tn["test"]["x"]) == 0:
                continue
            pred = predict_torch(model, tn["test"]["x"], adj_t,
                                 aoi=tn["test"]["aoi"], device=device)
            sc = tn["scaler"]
            m = regression_metrics(sc.inverse_transform(pred),
                                   sc.inverse_transform(tn["test"]["y"].numpy()))
            m.update(demand=level, seed=seed,
                     travel_time_s=scen["sim_out"].metrics.get("travel_time_mean_s"),
                     mean_aoi=scen["partial_obs"].aoi_stats["mean"])
            dem_rows.append(m)
            log.info(f"  {level:10s} seed={seed} RMSE={m['RMSE']} "
                     f"tt={m['travel_time_s']}")
    pd.DataFrame(dem_rows).to_csv(tdir / "demand_results.csv", index=False)

    # ---------------- corridor comparison (Experiment 8) ----------------
    log.info("=== corridor comparison ===")
    cor_rows = []
    for cid in ["corridor_1", "corridor_2"]:
        for seed in seeds:
            set_seed(seed)
            scen = run_scenario(cfg, cid, seed)
            tn = scen["tensors"]
            from src.models import build_model
            from src.models.engine import train_torch_model, predict_torch
            model = build_model("proposed", 8, tn["n_cells"],
                                target_window(cfg)[1], cfg).to(device)
            adj_t = torch.from_numpy(scen["adj_norm"]).to(device)
            res = train_torch_model(model, tn["train"], tn["val"], cfg, adj_t,
                                    device, full_objective=True,
                                    epochs=int(cfg["transfer"]["finetune_epochs"]),
                                    lr=float(cfg["transfer"]["finetune_lr"]),
                                    speed_scale=tn["speed_scale_mps"],
                                    physics_dt=float(cfg["sumo"]["aggregate_interval_s"]),
                                    seed=seed)
            if len(tn["test"]["x"]) == 0:
                continue
            pred = predict_torch(model, tn["test"]["x"], adj_t,
                                 aoi=tn["test"]["aoi"], device=device)
            sc = tn["scaler"]
            m = regression_metrics(sc.inverse_transform(pred),
                                   sc.inverse_transform(tn["test"]["y"].numpy()))
            cm = scen["sim_out"].metrics
            m.update(corridor=cid, name=scen["corridor"]["name"], seed=seed,
                     length_m=scen["corridor"].get("length_m"),
                     n_segments=scen["corridor"].get("n_segments"),
                     travel_time_s=cm.get("travel_time_mean_s"),
                     queue_mean=cm.get("queue_mean_veh"),
                     throughput=cm.get("throughput_veh_per_h"),
                     min_ttc=cm.get("min_ttc_s"),
                     mean_aoi=scen["partial_obs"].aoi_stats["mean"])
            cor_rows.append(m)
            log.info(f"  {cid} seed={seed} RMSE={m['RMSE']} len={m['length_m']}")
    pd.DataFrame(cor_rows).to_csv(tdir / "corridor_results.csv", index=False)

    # ---------------- uncertainty calibration + per-horizon ----------------
    log.info("=== uncertainty calibration ===")
    set_seed(seeds[0])
    scen = run_scenario(cfg, corridor_id, seeds[0])
    tn = scen["tensors"]
    from src.models import build_model
    model = build_model("proposed", 8, tn["n_cells"],
                        target_window(cfg)[1], cfg).to(device)
    ck = Path(cfg["_meta"]["repo_root"]) / "models/checkpoints/target_proposed.pt"
    if ck.exists() and torch.load(ck, map_location="cpu").get("num_nodes") == tn["n_cells"]:
        model.load_state_dict(torch.load(ck, map_location="cpu")["state_dict"])
    adj_t = torch.from_numpy(scen["adj_norm"]).to(device)
    if len(tn["test"]["x"]):
        mean, std = mc_dropout_predict(model, tn["test"]["x"], adj_t,
                                       aoi=tn["test"]["aoi"],
                                       K=int(cfg["uncertainty"]["mc_samples"]),
                                       device=device)
        y = tn["test"]["y"].numpy()
        cal = calibration_metrics(mean, std, y)
        hm = horizon_metrics(tn["scaler"].inverse_transform(mean),
                             tn["scaler"].inverse_transform(y))
        pd.DataFrame(hm).to_csv(tdir / "horizon_metrics.csv", index=False)
        rng = np.random.default_rng(0)
        flat_idx = rng.choice(mean.size, min(mean.size, 4000), replace=False)
        (proc / "uncertainty_eval.json").write_text(json.dumps({
            **cal,
            "std_flat": (std.reshape(-1)[flat_idx]).tolist(),
            "abs_err_flat": (np.abs(mean - y).reshape(-1)[flat_idx]).tolist(),
        }, indent=2))
        log.info(f"  PICP95={cal['PICP_95']} unc-err corr={cal['unc_err_corr']}")

    with ExperimentLogger(out / "logs", "evaluate_prediction", dict(cfg),
                          env_info(), seeds[0]) as elog:
        elog.add_metrics(ablation_best=str(ab_aggdf.iloc[0]["variant"]),
                         ablation_best_rmse=float(ab_aggdf.iloc[0]["RMSE_mean"]))
        elog.add_artifact(tdir / "ablation_results.csv")

    log.info("\nABLATION\n" + ab_aggdf.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
