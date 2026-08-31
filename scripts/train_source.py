#!/usr/bin/env python
"""STEP 8-9: source-domain training (spec sections 6-8, 35).

Trains every enabled baseline + the proposed model on the REAL source dataset
for each random seed, evaluates on the chronological test block, aggregates
mean +/- std, and writes:

    outputs/tables/source_prediction_results.csv / .md
    models/checkpoints/source_proposed_seed<seed>.pt
    models/checkpoints/source_proposed.pt          (primary, = first seed)
    data/processed/source_domain.npz               (adj + scaler for transfer)
    data/processed/source_history_seed<seed>.json  (training curves)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from _common import base_parser, load, env_info


def _evaluate_np(model, W, scaler, denorm: bool):
    from src.evaluation import regression_metrics
    t0 = time.time()
    pred = model.predict(W.X)
    infer = time.time() - t0
    y = W.Y
    if denorm:
        pred = scaler.inverse_transform(pred)
        y = scaler.inverse_transform(y)
    m = regression_metrics(pred, y)
    m["inference_time_s"] = round(infer, 4)
    return m, pred


def _evaluate_torch(model, tset, adj_t, scaler, device):
    from src.models.engine import predict_torch, count_parameters
    from src.evaluation import regression_metrics
    t0 = time.time()
    pred = predict_torch(model, tset["x"], adj_t, device=device)
    infer = time.time() - t0
    pred_d = scaler.inverse_transform(pred)
    y_d = scaler.inverse_transform(tset["y"].numpy())
    m = regression_metrics(pred_d, y_d)
    m["inference_time_s"] = round(infer, 4)
    m["num_parameters"] = count_parameters(model)
    return m, pred_d


def main() -> int:
    args = base_parser(__doc__).parse_args()
    cfg, log = load(args)
    import torch
    from src.utils import set_seed, ExperimentLogger
    from src.utils.config import resolve_device
    from src.data.source_prep import prepare_source_domain
    from src.models import build_model
    from src.models.baselines import build_numpy_baseline
    from src.models.engine import train_torch_model
    from src.evaluation import aggregate_seeds

    device = resolve_device(cfg)
    seeds = list(cfg["experiments"]["seeds"])
    enabled = list(cfg["baselines"]["enabled"])
    out = Path(cfg["_meta"]["repo_root"]) / cfg["project"]["outputs_dir"]
    ckpt = Path(cfg["_meta"]["repo_root"]) / "models" / "checkpoints"
    proc = Path(cfg["_meta"]["repo_root"]) / "data" / "processed"
    ckpt.mkdir(parents=True, exist_ok=True)
    proc.mkdir(parents=True, exist_ok=True)

    log.info(f"device={device}  seeds={seeds}  models={enabled + ['proposed']}")
    log.info("building source-domain windows from METR-LA (this takes ~30-60s) ...")
    prep = prepare_source_domain(cfg)
    log.info(f"  windows: train={len(prep['torch']['train']['x'])} "
             f"val={len(prep['torch']['val']['x'])} test={len(prep['torch']['test']['x'])}  "
             f"nodes={prep['num_nodes']} in_dim={prep['in_dim']}")
    if prep["is_synthetic"]:
        log.warning("SOURCE DATASET IS SYNTHETIC - results are for smoke "
                    "testing only and are stamped accordingly.")
    adj_t = torch.from_numpy(prep["adj_norm"]).to(device)
    N, Cin, H = prep["num_nodes"], prep["in_dim"], prep["horizon"]

    np.savez(proc / "source_domain.npz", adj=prep["adj"], adj_norm=prep["adj_norm"],
             scaler_mean=prep["scaler"].mean_, scaler_std=prep["scaler"].std_,
             num_nodes=N, in_dim=Cin, horizon=H,
             is_synthetic=prep["is_synthetic"], dataset_name=prep["dataset_name"])

    rows = []
    per_model_seed = {}

    for seed in seeds:
        set_seed(seed)
        log.info(f"----- seed {seed} -----")

        # classical / numpy baselines (raw speed).  Random Forest is O(N*H)
        # separate fits, so cap its training windows for tractability.
        rw_tr = prep["raw_windows"]["train"]
        classic_cap = 5000
        for name in [n for n in enabled if n in
                     ("historical_average", "linear_regression", "random_forest")]:
            mdl = build_numpy_baseline(name, H, cfg)
            Xtr, Ytr = rw_tr.X, rw_tr.Y
            if name == "random_forest" and len(Xtr) > classic_cap:
                step = len(Xtr) // classic_cap
                Xtr, Ytr = Xtr[::step], Ytr[::step]
                log.info(f"  (random_forest capped to {len(Xtr)} train windows)")
            t0 = time.time()
            mdl.fit(Xtr, Ytr)
            fit_t = time.time() - t0
            m, _ = _evaluate_np(mdl, prep["raw_windows"]["test"], prep["scaler"],
                                denorm=False)
            m.update(model=name, seed=seed, train_time_s=round(fit_t, 2),
                     num_parameters=0)
            rows.append(m)
            per_model_seed.setdefault(name, []).append(m)
            log.info(f"  {name:20s} MAE={m['MAE']} RMSE={m['RMSE']}")

        # torch baselines + proposed
        torch_models = [n for n in enabled if n not in
                        ("historical_average", "linear_regression",
                         "random_forest")] + ["proposed"]
        for name in torch_models:
            set_seed(seed)
            log.info(f"  [{name}] seed={seed} training ...")
            model = build_model(name, Cin, N, H, cfg, adj=prep["adj_norm"])
            res = train_torch_model(
                model, prep["torch"]["train"], prep["torch"]["val"], cfg, adj_t,
                device, full_objective=(name == "proposed"),
                speed_scale=prep["speed_scale_mps"], seed=seed, logger=log)
            m, _ = _evaluate_torch(model, prep["torch"]["test"], adj_t,
                                   prep["scaler"], device)
            m.update(model=name, seed=seed,
                     train_time_s=res["train_time_s"],
                     epochs_run=res["epochs_run"],
                     best_val_mae=round(res["best_val_mae"], 4))
            rows.append(m)
            per_model_seed.setdefault(name, []).append(m)
            log.info(f"  {name:20s} MAE={m['MAE']} RMSE={m['RMSE']} "
                     f"params={m['num_parameters']} ({res['train_time_s']}s)")

            if name == "proposed":
                torch.save({"state_dict": model.state_dict(),
                            "in_dim": Cin, "num_nodes": N, "horizon": H,
                            "seed": seed, "config": dict(cfg["proposed_model"])},
                           ckpt / f"source_proposed_seed{seed}.pt")
                (proc / f"source_history_seed{seed}.json").write_text(
                    json.dumps(res["history"], indent=2))
                if seed == seeds[0]:
                    torch.save({"state_dict": model.state_dict(),
                                "in_dim": Cin, "num_nodes": N, "horizon": H,
                                "seed": seed}, ckpt / "source_proposed.pt")

    # aggregate ------------------------------------------------------------
    df = pd.DataFrame(rows)
    agg_rows = []
    for name, recs in per_model_seed.items():
        a = aggregate_seeds(recs, ["MAE", "RMSE", "MAPE", "R2",
                                   "inference_time_s", "train_time_s",
                                   "num_parameters"])
        flat = {"model": name, "n_seeds": len(recs),
                "is_synthetic_source": prep["is_synthetic"]}
        for k, v in a.items():
            flat[f"{k}_mean"] = v["mean"]
            flat[f"{k}_std"] = v["std"]
        agg_rows.append(flat)
    agg = pd.DataFrame(agg_rows).sort_values("RMSE_mean", na_position="last")

    tdir = out / "tables"; tdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tdir / "source_prediction_results_raw.csv", index=False)
    agg.to_csv(tdir / "source_prediction_results.csv", index=False)
    (tdir / "source_prediction_results.md").write_text(
        agg.to_markdown(index=False))

    with ExperimentLogger(out / "logs", "train_source", dict(cfg), env_info(),
                          seeds[0]) as elog:
        elog.add_metrics(models=list(per_model_seed),
                         best_model=str(agg.iloc[0]["model"]),
                         best_rmse=float(agg.iloc[0]["RMSE_mean"]))
        elog.add_artifact(tdir / "source_prediction_results.csv")

    log.info("\n" + agg.to_string(index=False))
    log.info(f"tables -> {tdir/'source_prediction_results.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
