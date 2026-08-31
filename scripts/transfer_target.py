#!/usr/bin/env python
"""STEP 21-25 / EXPERIMENT 2: transfer the pretrained source model to the
Puducherry target domain and compare transfer strategies
(spec sections 10-11, 36).

Strategies (from config.transfer.strategies):
    none | frozen_encoder | full_finetune | frozen_encoder_da

Writes:
    outputs/tables/transfer_results.csv / .md
    models/checkpoints/target_proposed.pt        (best strategy, first seed)
    data/processed/target_domain.npz             (scaler + reconstruction maps)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _common import base_parser, load, env_info


def _fit_recon_maps(gt_full, split):
    """speed -> {density(ch1), flow(ch2), queue(ch3)} linear maps on train."""
    a, b = split["train_range"]
    tr = gt_full[a:b].reshape(-1, gt_full.shape[-1])
    sp = tr[:, 0]
    maps = {}
    for ch in (1, 2, 3):
        if np.std(sp) > 1e-6:
            coef = np.polyfit(sp, tr[:, ch], 1)
            maps[ch] = [float(coef[0]), float(coef[1])]
        else:
            maps[ch] = [0.0, float(np.mean(tr[:, ch]))]
    return maps


def main() -> int:
    args = base_parser(__doc__).parse_args()
    cfg, log = load(args)
    import torch
    from src.utils import set_seed, ExperimentLogger
    from src.utils.config import resolve_device
    from src.models import build_model
    from src.models.engine import train_torch_model, predict_torch
    from src.evaluation import regression_metrics, aggregate_seeds
    from src.transfer import apply_transfer_strategy
    from src.transfer.adapt import make_domain_aux_loss
    from src.experiments import (run_scenario, load_source_context,
                                 load_proposed_source_state)

    device = resolve_device(cfg)
    seeds = list(cfg["experiments"]["seeds"])
    strategies = list(cfg["transfer"]["strategies"])
    corridor_id = cfg["experiments"]["corridor_for_sweeps"]
    da_kind = cfg["transfer"]["adaptation_loss"]
    da_w = float(cfg["transfer"]["adaptation_weight"])
    ft_epochs = int(cfg["transfer"]["finetune_epochs"])
    ft_lr = float(cfg["transfer"]["finetune_lr"])

    out = Path(cfg["_meta"]["repo_root"]) / cfg["project"]["outputs_dir"]
    proc = Path(cfg["_meta"]["repo_root"]) / "data" / "processed"
    ck = Path(cfg["_meta"]["repo_root"]) / "models" / "checkpoints"
    proc.mkdir(parents=True, exist_ok=True)

    sctx = load_source_context(cfg)
    if sctx["is_synthetic"]:
        log.warning("source model was trained on SYNTHETIC data - transfer "
                    "results are smoke-test only.")
    src_blob = load_proposed_source_state(cfg)
    src_state = src_blob["state_dict"]

    # source windows for the domain-alignment loss (built lazily, cached)
    _src_cache = {}

    def _source_pool():
        if "x" not in _src_cache:
            from src.data.source_prep import prepare_source_domain
            sp = prepare_source_domain(cfg)
            _src_cache["x"] = sp["torch"]["train"]["x"][:256].numpy()
            _src_cache["adj_t"] = torch.from_numpy(sp["adj_norm"]).to(device)
            _src_cache["model"] = build_model("proposed", sp["in_dim"],
                                              sp["num_nodes"], sp["horizon"], cfg)
            try:
                _src_cache["model"].load_state_dict(src_state)
            except Exception:
                pass
            _src_cache["model"].to(device).eval()
        return _src_cache

    rows = []
    per_strategy = {}
    primary_saved = None

    for seed in seeds:
        set_seed(seed)
        sc = run_scenario(cfg, corridor_id, seed, logger=log)
        tn = sc["tensors"]
        adj_t = torch.from_numpy(sc["adj_norm"]).to(device)
        from src.utils import target_window
        N, Cin, H = tn["n_cells"], tn["in_dim"], target_window(cfg)[1]
        log.info(f"seed {seed}: backend={sc['backend']} corridor={corridor_id} "
                 f"cells={N} train/val/test="
                 f"{len(tn['train']['x'])}/{len(tn['val']['x'])}/{len(tn['test']['x'])}")

        for strat in strategies:
            set_seed(seed)
            model = build_model("proposed", Cin, N, H, cfg).to(device)
            groups, meta = apply_transfer_strategy(model, src_state, strat)
            aux = None
            if meta["domain_adaptation"]:
                pool = _source_pool()
                model._da_adj_t = adj_t
                aux = make_domain_aux_loss(pool["model"], pool["x"],
                                           pool["adj_t"], da_kind, da_w, device,
                                           seed)
            res = train_torch_model(
                model, tn["train"], tn["val"], cfg, adj_t, device,
                full_objective=True, epochs=ft_epochs, lr=ft_lr,
                param_groups=groups if strat != "none" else None,
                speed_scale=tn["speed_scale_mps"],
                physics_dt=float(cfg["sumo"]["aggregate_interval_s"]),
                aux_loss_fn=aux, seed=seed)

            pred = predict_torch(model, tn["test"]["x"], adj_t,
                                 aoi=tn["test"]["aoi"], device=device)
            sc_obj = tn["scaler"]
            m = regression_metrics(sc_obj.inverse_transform(pred),
                                   sc_obj.inverse_transform(tn["test"]["y"].numpy()))
            m.update(strategy=strat, seed=seed,
                     transferred_params=meta["transferred_params"],
                     frozen_encoder=meta["frozen_encoder"],
                     domain_adaptation=meta["domain_adaptation"],
                     train_time_s=res["train_time_s"])
            rows.append(m)
            per_strategy.setdefault(strat, []).append(m)
            log.info(f"  {strat:20s} MAE={m['MAE']} RMSE={m['RMSE']} "
                     f"R2={m['R2']} (xfer params={meta['transferred_params']})")

            if seed == seeds[0]:
                if (primary_saved is None or
                        m["RMSE"] < primary_saved[1]):
                    primary_saved = (strat, m["RMSE"], model.state_dict(),
                                     tn["scaler"], sc["sim_out"].n_cells,
                                     _fit_recon_maps(tn["gt_full"], tn["split"]))

    df = pd.DataFrame(rows)
    agg_rows = []
    for strat, recs in per_strategy.items():
        a = aggregate_seeds(recs, ["MAE", "RMSE", "MAPE", "R2", "train_time_s"])
        flat = {"strategy": strat, "n_seeds": len(recs),
                "transferred_params": recs[0]["transferred_params"]}
        for k, v in a.items():
            flat[f"{k}_mean"] = v["mean"]; flat[f"{k}_std"] = v["std"]
        agg_rows.append(flat)
    agg = pd.DataFrame(agg_rows).sort_values("RMSE_mean", na_position="last")

    tdir = out / "tables"; tdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tdir / "transfer_results_raw.csv", index=False)
    agg.to_csv(tdir / "transfer_results.csv", index=False)
    (tdir / "transfer_results.md").write_text(agg.to_markdown(index=False))

    if primary_saved:
        strat, rmse, state, scaler, ncells, recon = primary_saved
        import torch as _t
        from src.utils import target_window
        _t.save({"state_dict": state, "strategy": strat, "in_dim": 8,
                 "num_nodes": ncells, "horizon": target_window(cfg)[1]},
                ck / "target_proposed.pt")
        np.savez(proc / "target_domain.npz",
                 scaler_mean=scaler.mean_, scaler_std=scaler.std_,
                 n_cells=ncells, best_strategy=strat,
                 recon_maps=json.dumps(recon), corridor_id=corridor_id)
        log.info(f"saved primary target model (strategy={strat}, "
                 f"RMSE={rmse}) -> {ck/'target_proposed.pt'}")

    with ExperimentLogger(out / "logs", "transfer_target", dict(cfg), env_info(),
                          seeds[0]) as elog:
        elog.add_metrics(strategies=strategies,
                         best_strategy=str(agg.iloc[0]["strategy"]),
                         best_rmse=float(agg.iloc[0]["RMSE_mean"]))
        elog.add_artifact(tdir / "transfer_results.csv")

    log.info("\n" + agg.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
