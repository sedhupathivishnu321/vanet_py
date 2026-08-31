#!/usr/bin/env python
"""STEP 36-43: statistical analysis, figures, interactive map and the final
research report -- all generated from the ACTUAL experiment outputs
(spec sections 47-51, 63).  No numbers are hand-written into the report.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from _common import base_parser, load, env_info


def _read(tables: Path, name: str):
    p = tables / name
    return pd.read_csv(p) if p.exists() else None


def _md_table(df, cols=None):
    if df is None or len(df) == 0:
        return "_(not generated - run the corresponding stage)_"
    if cols:
        cols = [c for c in cols if c in df.columns]
        df = df[cols]
    return df.round(4).to_markdown(index=False)


def _hypotheses(tables: Path, proc: Path) -> list[dict]:
    H = []
    tr = _read(tables, "transfer_results.csv")
    if tr is not None and "strategy" in tr:
        s = tr.set_index("strategy")["RMSE_mean"].to_dict()
        if "none" in s and "frozen_encoder" in s:
            H.append({"id": "H1", "claim": "transfer improves over from-scratch",
                      "evidence": f"RMSE none={s['none']:.3f} vs "
                                  f"frozen_encoder={s['frozen_encoder']:.3f}",
                      "supported": bool(s["frozen_encoder"] < s["none"])})
        if "frozen_encoder" in s and "frozen_encoder_da" in s:
            H.append({"id": "H5", "claim": "domain adaptation improves transfer",
                      "evidence": f"RMSE frozen={s['frozen_encoder']:.3f} vs "
                                  f"+DA={s['frozen_encoder_da']:.3f}",
                      "supported": bool(s["frozen_encoder_da"] < s["frozen_encoder"])})
    va = _read(tables, "vanet_results.csv")
    if va is not None:
        pen = va[va["sweep"] == "penetration"]
        if len(pen) >= 2:
            r = np.corrcoef(pen["penetration"].astype(float),
                            pen["RMSE_mean"].astype(float))[0, 1]
            H.append({"id": "H2", "claim": "higher penetration -> better reconstruction",
                      "evidence": f"corr(penetration, RMSE) = {r:.3f} (expect < 0)",
                      "supported": bool(r < 0)})
        ap = va["aoi_err_pearson_mean"].dropna() if "aoi_err_pearson_mean" in va else []
        if len(ap):
            H.append({"id": "H3", "claim": "higher AoI -> higher prediction error",
                      "evidence": f"mean pearson(AoI, |err|) = {np.mean(ap):.3f}",
                      "supported": bool(np.mean(ap) > 0)})
    ab = _read(tables, "ablation_results.csv")
    if ab is not None and "variant" in ab:
        d = ab.set_index("variant")["RMSE_mean"].to_dict()
        if "full_proposed" in d and "full_minus_AoI" in d:
            H.append({"id": "H4", "claim": "AoI-aware attention helps under stale obs",
                      "evidence": f"RMSE full={d['full_proposed']:.3f} vs "
                                  f"full-minus-AoI={d['full_minus_AoI']:.3f}",
                      "supported": bool(d["full_proposed"] < d["full_minus_AoI"])})
        if "full_proposed" in d and "full_minus_physics" in d:
            H.append({"id": "H6", "claim": "physics constraints improve consistency",
                      "evidence": f"RMSE full={d['full_proposed']:.3f} vs "
                                  f"full-minus-physics={d['full_minus_physics']:.3f}",
                      "supported": bool(d["full_proposed"] <= d["full_minus_physics"])})
    un = proc / "uncertainty_eval.json"
    if un.exists():
        c = json.loads(un.read_text()).get("unc_err_corr")
        if c is not None:
            H.append({"id": "H7", "claim": "uncertainty correlates with error",
                      "evidence": f"corr(std, |err|) = {c:.3f}",
                      "supported": bool(c > 0)})
    ct = _read(tables, "control_results.csv")
    if ct is not None and "controller" in ct:
        d = ct.set_index("controller")
        if "proposed" in d.index and "fixed" in d.index and \
                "travel_time_mean_s_mean" in d.columns:
            p = d.loc["proposed", "travel_time_mean_s_mean"]
            f = d.loc["fixed", "travel_time_mean_s_mean"]
            H.append({"id": "H9", "claim": "proposed reduces delay vs conventional",
                      "evidence": f"travel time proposed={p:.1f}s vs fixed={f:.1f}s",
                      "supported": bool(p < f)})
        if "proposed" in d.index and "max_pressure" in d.index and \
                "ttc_violations_mean" in d.columns:
            H.append({"id": "H8", "claim": "uncertainty-aware control improves safety",
                      "evidence": f"TTC violations proposed="
                                  f"{d.loc['proposed','ttc_violations_mean']:.1f} vs "
                                  f"max_pressure={d.loc['max_pressure','ttc_violations_mean']:.1f}",
                      "supported": bool(d.loc["proposed", "ttc_violations_mean"]
                                        <= d.loc["max_pressure", "ttc_violations_mean"])})
    return H


def main() -> int:
    args = base_parser(__doc__).parse_args()
    cfg, log = load(args)
    from src.visualization import generate_all_figures, build_interactive_map

    root = Path(cfg["_meta"]["repo_root"])
    out = root / cfg["project"]["outputs_dir"]
    tables = out / "tables"
    proc = root / "data" / "processed"
    reports = out / "reports"; reports.mkdir(parents=True, exist_ok=True)

    log.info("generating figures ...")
    figs = generate_all_figures(cfg, logger=log)
    log.info("building interactive map ...")
    try:
        build_interactive_map(cfg, vanet_npz=str(proc / "vanet_default.npz"),
                              logger=log)
    except Exception as exc:
        log.warning(f"interactive map failed: {exc}")

    # experiment manifest from the JSON logs
    logs = sorted((out / "logs").glob("*.json"))
    manifest = []
    for lp in logs:
        try:
            d = json.loads(lp.read_text())
            manifest.append({k: d.get(k) for k in
                             ("experiment_id", "experiment", "timestamp_utc",
                              "seed", "wall_time_s")})
        except Exception:
            pass
    (reports / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2))

    dsrep = json.loads((out / "dataset_report.json").read_text()) \
        if (out / "dataset_report.json").exists() else {}
    env = env_info()
    H = _hypotheses(tables, proc)

    synthetic_source = bool(dsrep.get("is_synthetic"))
    prof = cfg["_meta"]["profile"]

    lines = []
    A = lines.append
    A(f"# Final Research Report\n")
    A(f"**Communication-Aware Transfer Learning for Real-Time Traffic-State "
      f"Prediction and Proactive VANET Traffic Control on the Puducherry Road "
      f"Network**\n")
    A(f"_Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} · profile: "
      f"`{prof}` · git `{env.get('git_commit')}`_\n")
    if synthetic_source:
        A("> **WARNING - SYNTHETIC SOURCE DATA.** No real METR-LA/PEMS-BAY file "
          "was found, so a synthetic source dataset was used. The numbers below "
          "demonstrate the pipeline only and are **not scientifically valid**. "
          "Provide the real dataset (`scripts/download_dataset.py`) and re-run.\n")

    A("## 1. Title\nSee header.\n")
    A("## 2. Abstract\n"
      "A spatio-temporal graph neural network is pre-trained on a real, public "
      "traffic dataset (METR-LA/PEMS-BAY) and transferred to an OpenStreetMap-"
      "derived model of two Puducherry corridors (Reddiyarpalayam->Beach Road; "
      "Nainarmandapam->Lawspet). Target-domain traffic is **simulated** (SUMO or "
      "a built-in IDM micro-simulator) because no measured trajectory dataset "
      "exists for these corridors. A VANET layer imposes controlled connected-"
      "vehicle penetration, packet-delivery ratio, latency and Age of "
      "Information; the model reconstructs the partially observed traffic state "
      "and feeds conventional (fixed, Max-Pressure, MPC, DQN) and a proposed "
      "uncertainty/AoI-aware controller. All results below are produced by "
      "`run_all.py` from the stated dataset, OSM retrieval, configuration and "
      "random seeds.\n")
    A("## 3. Introduction\n"
      "Motivation, research questions RQ1-RQ10 and hypotheses H1-H9 are listed "
      "in the README. This report is regenerated from `outputs/` on every run.\n")
    A("## 4. Research gap\n"
      "Communication-aware transfer learning for traffic-state prediction, where "
      "a model pre-trained on an existing real dataset is adapted to an OSM-"
      "derived network and evaluated under controlled VANET penetration / PDR / "
      "latency / AoI, has not been systematically characterised for the "
      "Puducherry corridors. No novelty claim is made beyond a literature "
      "review the user must complete.\n")
    A("## 5. Related work\n_Populate from a literature review; not auto-"
      "generated to avoid fabricated citations._\n")

    A("## 6. Dataset\n")
    A(_md_table(pd.DataFrame([dsrep])) if dsrep else "_run scripts/inspect_dataset.py_")
    A("")
    A("## 7. OSM Puducherry network\n")
    prov = root / "data" / "osm" / "osm_provenance.json"
    if prov.exists():
        A("```json\n" + prov.read_text() + "\n```")
    cg = root / "data" / "osm" / "corridor_routes.geojson"
    A(f"\nCorridor routes: `{cg}` " + ("(present)" if cg.exists() else "(missing)"))

    A("\n## 8. Source-domain training\n")
    A(_md_table(_read(tables, "source_prediction_results.csv"),
                ["model", "MAE_mean", "MAE_std", "RMSE_mean", "RMSE_std",
                 "MAPE_mean", "R2_mean", "num_parameters_mean",
                 "inference_time_s_mean"]))
    A("\n## 9. Transfer learning\n")
    A(_md_table(_read(tables, "transfer_results.csv"),
                ["strategy", "transferred_params", "MAE_mean", "RMSE_mean",
                 "RMSE_std", "R2_mean"]))
    A("\n## 10. VANET communication model\n")
    A(_md_table(_read(tables, "vanet_results.csv"),
                ["sweep", "penetration", "pdr", "latency_ms", "RMSE_mean",
                 "RMSE_std", "mean_aoi_mean", "cell_coverage_mean"]))
    A("\n### Communication stress tests\n")
    A(_md_table(_read(tables, "vanet_stress_results.csv")))
    A("\n## 11. AoI model\n"
      "AoI_c(t) = t - u_c(t); statistics and AoI-vs-error correlation are in the "
      "VANET table (`aoi_err_pearson_mean`) and figures 09-10, 18.\n")
    A("## 12. Proposed architecture\nSee README section 'Proposed model' and "
      "figure 06. Parameter count is in the source-prediction table.\n")
    A("## 13. Physics constraints\nAcceleration/jerk hinge penalty; ablation "
      "row `full_minus_physics`.\n")
    A("## 14. Uncertainty estimation\n")
    ue = proc / "uncertainty_eval.json"
    if ue.exists():
        A("```json\n" + ue.read_text()[:1500] + "\n```")
    A("\n## 15. Traffic-control algorithms\n")
    A(_md_table(_read(tables, "control_results.csv"),
                ["controller", "travel_time_mean_s_mean", "delay_mean_s_mean",
                 "queue_mean_veh_mean", "throughput_veh_per_h_mean",
                 "min_ttc_s_mean", "ttc_violations_mean"]))
    A("\n## 16. Experimental design\n"
      f"Seeds: {cfg['experiments']['seeds']}. Profile: `{prof}`. "
      "Chronological 70/15/15 split; target test interval never used for "
      "adaptation. Full matrix in README.\n")
    A("## 17. Results\nSee sections 8-15 and `outputs/figures/`.\n")
    A("## 18. Statistical analysis\n")
    A(_md_table(_read(tables, "control_significance.csv")))
    A("\n## 19. Ablation\n")
    A(_md_table(_read(tables, "ablation_results.csv"),
                ["variant", "RMSE_mean", "RMSE_std", "MAE_mean", "R2_mean"]))
    A("\n## 20. Limitations\n"
      "- Target-domain traffic is **simulated**, not measured.\n"
      "- The IDM fallback is a single-corridor + cross-street abstraction when "
      "SUMO is unavailable.\n"
      "- Sensor coordinates may be absent for the source dataset (graph from "
      "adjacency only).\n"
      "- `--quick`/`--smoke` profiles reduce epochs, seeds and demand.\n")
    A("## 21. Threats to validity\n"
      "Sim-to-real gap; corridor abstraction; limited seeds in reduced "
      "profiles; transfer of only the hidden-space encoder when input "
      "parametrisations differ.\n")
    A("## 22. Future work\nReal probe-vehicle data; ns-3/Veins channel model; "
      "multi-intersection coordination; larger source corpora.\n")

    A("## 23. Hypothesis outcomes\n")
    if H:
        A(_md_table(pd.DataFrame(H)[["id", "claim", "evidence", "supported"]]))
    else:
        A("_insufficient outputs to evaluate hypotheses - run the full "
          "pipeline._")
    A("\n## 24. Reproducibility instructions\n"
      "```bash\n"
      "# in a GitHub Codespace opened on this repo\n"
      "python scripts/check_env.py\n"
      "python scripts/download_dataset.py         # real METR-LA/PEMS-BAY\n"
      "python run_all.py --full                   # or --quick\n"
      "```\n"
      f"Environment used for this run:\n\n```json\n{json.dumps(env, indent=2, default=str)}\n```\n")

    (reports / "final_research_report.md").write_text("\n".join(lines))
    log.info(f"report -> {reports/'final_research_report.md'}")
    log.info(f"figures -> {len(figs)} in {out/'figures'}")
    for h in H:
        log.info(f"  {h['id']}: supported={h['supported']}  ({h['evidence']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
