# Communication-Aware Transfer Learning for Real-Time Traffic-State Prediction and Proactive VANET Traffic Control on the Puducherry Road Network

**Technical model:** *AoI-Aware Spatio-Temporal Graph Neural Network with Transfer
Learning for VANET-Based Traffic-State Prediction and Intelligent Traffic Control.*

This repository pre-trains a spatio-temporal graph neural network on a **real,
public** traffic dataset, transfers the learned representation to an
**OpenStreetMap-derived** model of two **Puducherry** corridors, injects a
**simulated VANET** communication layer (connected-vehicle penetration, packet
loss, latency, Age of Information), reconstructs the partially observed traffic
state, and drives conventional and a proposed uncertainty/AoI-aware traffic
signal controller. Everything runs in a **GitHub Codespace**.

---

## ⚠️ Scientific-integrity statement (read first)

| Component | Nature |
|---|---|
| **Source dataset** (METR-LA / PEMS-BAY) | **REAL** measured loop-detector data, downloaded from its public mirror. |
| **Puducherry road network** | **REAL geographic information** — OpenStreetMap topology & geometry (OSMnx / Nominatim). |
| **VANET communication** (penetration, PDR, latency, AoI) | **SIMULATED** — a controlled stochastic model. |
| **Puducherry target-domain traffic** | **SIMULATED** — SUMO, or a built-in IDM micro-simulator when SUMO is unavailable. Used **only** because no measured trajectory dataset exists for these exact corridors. |

Simulated Puducherry traffic is **never** called "real traffic". OpenStreetMap
provides **geometry, not traffic measurements**. No novelty is claimed beyond a
literature review the user must complete. If a proposed component performs worse
than a baseline, the pipeline reports it (see `outputs/reports/final_research_report.md`).

If the real source dataset cannot be downloaded, a **synthetic** source dataset
is used **for smoke-testing only** and every artefact is stamped `SYNTHETIC`.

---

## 1. Research motivation

Vehicular ad-hoc networks (VANETs) promise real-time traffic-state awareness, but
in practice only a fraction of vehicles are connected, packets are lost, reports
arrive late, and information ages. Cities such as Puducherry have **no public
high-resolution traffic dataset**, yet do have accurate OSM road geometry. This
project asks whether a model trained on an existing real dataset can be
**transferred** to such a city and remain useful under realistic communication
impairment, and whether **communication-aware** prediction improves proactive
signal control.

## 2. Research questions

- **RQ1** Can a model trained on an existing real dataset transfer to a different road topology?
- **RQ2** How much does transfer learning improve prediction vs training from scratch?
- **RQ3** How does VANET penetration affect traffic-state prediction?
- **RQ4** How does packet-delivery ratio affect prediction?
- **RQ5** How does communication latency affect prediction?
- **RQ6** Does AoI-aware graph attention improve robustness to stale observations?
- **RQ7** Does domain adaptation improve source→Puducherry transfer?
- **RQ8** Does a physics-consistency loss improve predicted vehicle behaviour?
- **RQ9** Does prediction uncertainty carry useful information for control?
- **RQ10** Does communication-aware prediction improve traffic-control performance?

Hypotheses **H1–H9** (one per RQ area) are *tested, not assumed* — verdicts are
written into the final report from the actual numbers.

## 3. Dataset source

- **Primary:** METR-LA (207 loop detectors, Los Angeles, 5-min sampling, 4 months).
- **Alternative:** PEMS-BAY (325 sensors, Bay Area).
- Origin: Li, Yu, Shahabi, Liu, *"Diffusion Convolutional Recurrent Neural
  Network: Data-Driven Traffic Forecasting"*, ICLR 2018.
### Download links (`scripts/download_dataset.py` fetches these automatically)

Public, no-auth GitHub raw endpoints (verified 2026-09-01):

**METR-LA** → `data/source/METR-LA/`

| file | URL |
|---|---|
| `metr-la.h5` (57 MB, traffic speeds) | `https://raw.githubusercontent.com/deepkashiwa20/DL-Traff-Graph/main/METRLA/metr-la.h5` |
| `adj_mx.pkl` (sensor adjacency) | `https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/adj_mx.pkl` |
| `graph_sensor_locations.csv` (lat/lon) | `https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/graph_sensor_locations.csv` |
| `graph_sensor_ids.txt` (sensor order) | `https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/graph_sensor_ids.txt` |

**PEMS-BAY** (alternative) → `data/source/PEMS-BAY/`

| file | URL |
|---|---|
| `pems-bay.zip` (→ `pems-bay.h5`, 31 MB) | `https://raw.githubusercontent.com/deepkashiwa20/DL-Traff-Graph/main/PEMSBAY/pems-bay.zip` |
| `adj_mx_bay.pkl` | `https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/adj_mx_bay.pkl` |
| `graph_sensor_locations_bay.csv` | `https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/graph_sensor_locations_bay.csv` |

Sources: time series vendored by **deepkashiwa20/DL-Traff-Graph**; sensor graph
from the original **liyaguang/DCRNN** repo (Li et al., ICLR 2018). Legacy
single-zip mirror (`graphmining.ai`) is kept in `config.yaml` as a last resort.

**Manual download** (if the script has no network):

```bash
mkdir -p data/source/METR-LA
cd data/source/METR-LA
curl -LO https://raw.githubusercontent.com/deepkashiwa20/DL-Traff-Graph/main/METRLA/metr-la.h5
curl -LO https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/adj_mx.pkl
curl -LO https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/graph_sensor_locations.csv
curl -LO https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/graph_sensor_ids.txt
```

**If a dataset already exists under `data/source/<NAME>/`, it is used as-is and
nothing is downloaded.** `data/` is git-ignored, so the dataset stays out of the
repo and is reproducibly re-fetched by the script in a fresh Codespace.

## 4. Dataset license / usage

METR-LA / PEMS-BAY are released by their original authors for research use; comply
with the original terms. This repo redistributes **no data**, only download
scripts pointing at public mirrors. OSM data © OpenStreetMap contributors (ODbL).
Nominatim used lightly + cached. SUMO is EPL-2.0. See `LICENSE`.

## 5. OSM methodology

`scripts/download_osm.py` calls `osmnx.graph_from_place("Puducherry, India",
network_type="drive")` (fallback: a 6 km bounding circle around
11.9338 N, 79.8298 E). Edge speeds/travel-times are imputed from OSM tags where
present. `scripts/build_puducherry_routes.py` geocodes the four endpoints with
Nominatim, snaps to nearest OSM nodes, routes by shortest length, and exports:

```
data/osm/puducherry.graphml
data/osm/corridor_routes.geojson
data/osm/corridor_1_ml_graph.json   data/osm/corridor_2_ml_graph.json
data/osm/osm_provenance.json        # retrieval date, method, every fallback used
```

Missing OSM attributes are recorded as `null`, never fabricated.

## 6. Puducherry corridors

| id | corridor | endpoints |
|---|---|---|
| `corridor_1` | **Reddiyarpalayam → Beach Road** | geocoded, then nearest-node routed |
| `corridor_2` | **Nainarmandapam → Lawspet** | geocoded, then nearest-node routed |

Each corridor is converted to a node/edge ML graph with
`latitude, longitude, degree, intersection_indicator` (nodes) and
`length, road_class, speed_limit, travel_time, lanes` (edges).

## 7. VANET model

Each connected vehicle beacons `id, timestamp, position, speed, acceleration,
heading, road_segment` every `beacon_interval_s`. The channel applies:

- **Penetration** — a persistent per-vehicle Bernoulli(`penetration`) connected flag.
- **PDR** — per beacon, `M_received ~ Bernoulli(PDR)`.
- **Latency** — a beacon generated at `t` is *usable only from* `t + latency`
  (it arrives later in simulation time; it is not merely deleted).

Sweeps: penetration ∈ {0.10, 0.20, 0.30, 0.50, 0.70, 1.00}; PDR ∈ {0.70, 0.80,
0.90, 0.95, 1.00}; latency ∈ {0, 20, 50, 100, 200, 500} ms.

## 8. AoI formulation

For road segment *c*: `AoI_c(t) = t − u_c(t)`, where `u_c(t)` is the **generation**
timestamp of the most recently *received* observation mapped to *c* (a segment
that has never received a fresh beacon gets `aoi_cap_s`). Reported: mean, median,
P95, max; plotted vs time, as a distribution, and against prediction error.

## 9. Transfer learning

Pre-train the proposed ST-GNN on the source dataset, then adapt to the Puducherry
target graph. Strategies compared (`config.transfer.strategies`):

| strategy | description |
|---|---|
| `none` | target model trained from random init (control) |
| `frozen_encoder` | **A** — load source hidden-space encoder, freeze it, fine-tune AoI/temporal/adapter/head |
| `full_finetune` | **B** — load whole model, fine-tune end-to-end at low LR |
| `frozen_encoder_da` | **A + explicit domain-alignment loss** (MMD or CORAL) between source & target latents |

Source and target share an **8-channel feature schema**
(`speed, density, flow, queue, mask, aoi_norm, pdr, latency_norm`) so the input
projection transfers directly; the graph-space GCN/GAT encoder is `N`-independent
and transfers regardless of corridor size.

## 10. Domain adaptation

`L = L_pred + λ_DA · L_domain`, with `L_domain ∈ {MMD, CORAL}` on the pooled,
projected encoder latent. `λ_DA` = `transfer.adaptation_weight`. Feature
normalisation statistics are fit on the target **train** block only.

## 11. Proposed model

**AoI-Aware Transferable Spatio-Temporal GNN** (`src/models/proposed.py`):

```
input features ─▶ input projection
               ─▶ spatial encoder  (GAT | GCN)            [TRANSFERABLE]
               ─▶ AoI-aware attention   αᵢⱼ = softmaxⱼ( score(hᵢ,hⱼ) − β·AoIⱼ )
               ─▶ temporal encoder (GRU | Transformer)
               ─▶ transfer adapter  (residual MLP)        [target fine-tune]
               ─▶ horizon head  ─▶ traffic-state prediction
                                └▶ latent() ─▶ MMD / CORAL domain alignment
```

Total objective (spec §27):
`L = L_pred + λ₁L_AoI + λ₂L_physics + λ₃L_domain + λ₄L_smooth` — every term is
ablated; none is assumed to help.

- **L_physics** — hinge penalty on implausible acceleration & jerk implied by the
  predicted speed trajectory (`v_t ≈ Δx/Δt`, `a_t ≈ Δv/Δt`).
- **L_AoI** — regularises predictions toward persistence with a weight that grows
  with segment AoI (don't hallucinate when the observation is stale).
- **Uncertainty** — Monte-Carlo dropout (`K` stochastic passes → μ, σ);
  deep-ensemble path also provided. Calibration: PICP, MPIW, σ–|error|
  correlation, calibration curve.

Baselines (spec §7, all comparable, none claimed novel): Historical Average,
Ridge regression, Random Forest, LSTM, GRU, GCN, GAT, ST-GCN, Temporal
Transformer.

## 12. Control algorithm

Predicted state → decision layer. Controllers (`config.control.controllers`):

- **fixed** — benchmarked explicit fixed cycle.
- **max_pressure** — serve the movement with greatest `ΣQ_in − ΣQ_out`.
- **mpc** — roll a point-queue model over the predicted horizon for
  {hold, switch}; minimise `w₁Delay + w₂Queue + w₃Stops + w₄TravelTime + w₅Risk`.
- **dqn** — small DQN trained against the micro-simulator (the one practical RL baseline).
- **proposed** — risk-aware:
  `J = J_traffic + λ_A J_AoI + λ_U J_uncertainty + λ_S J_safety`. High
  uncertainty / AoI **widen switching hysteresis** (more conservative); the
  controller never takes an unsafe action because a prediction is uncertain.

**Safety:** `TTC = d / max(v_ego − v_lead, ε)`; min / mean / P5 TTC and violations
against `control.minimum_ttc = 2.0 s`.

## 13. Installation

### GitHub Codespaces (recommended)

1. Push this folder to a GitHub repo → **Code ▸ Codespaces ▸ Create codespace**.
2. The devcontainer (`.devcontainer/`) runs `setup.sh`: installs **SUMO**
   (`apt: sumo sumo-tools`) and all Python deps, sets `SUMO_HOME`, runs
   `scripts/check_env.py`.
3. Wait for "Bootstrap complete".

### Local (Linux/macOS/WSL, Python 3.10–3.11)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# optional, for the SUMO backend:  sudo apt-get install -y sumo sumo-tools
export PYTHONPATH=$PWD
python scripts/check_env.py
```

`conda env create -f environment.yml` also works (bundles SUMO).

> Without SUMO the project automatically uses the built-in **IDM micro-simulator**
> (backend `idm_fallback`), clearly labelled in every artefact. Nothing is skipped.

## 14. Commands (step-by-step)

Run the whole pipeline:

```bash
python run_all.py --quick     # reduced scale, ~15–25 min on a 4-core CPU
python run_all.py --full      # complete experiment matrix (hours on CPU)
python run_all.py --smoke     # CI wiring check (minutes)
```

…or stage by stage (each is standalone; append `--quick` / `--smoke` or nothing for full):

```bash
python scripts/check_env.py                    # STEP 1–2  environment / GPU / disk / SUMO
python scripts/download_dataset.py             # STEP 5    identify or download METR-LA/PEMS-BAY
python scripts/inspect_dataset.py              # STEP 6–7  validate + report + chronological split
python scripts/download_osm.py                 # STEP 10   real Puducherry OSM network
python scripts/build_puducherry_routes.py      # STEP 11–13 geocode + build both corridors
python scripts/build_sumo.py                   # STEP 14   OSM→SUMO (or record IDM fallback)
python scripts/train_source.py                 # STEP 8–9  baselines + proposed on the REAL source data
python scripts/transfer_target.py              # STEP 21–25 transfer + domain adaptation (Experiment 2)
python scripts/simulate_vanet.py               # STEP 16–20 penetration/PDR/latency/stress (Exp 3–5,7)
python scripts/evaluate_prediction.py          # STEP 26   ablation + demand + corridor + uncertainty (Exp 6,8)
python scripts/train_control.py                # STEP 29   train the DQN controller
python scripts/evaluate_control.py             # STEP 31–34 closed-loop control + safety + significance
python scripts/generate_report.py              # STEP 37–43 figures, interactive map, final report
python -m pytest -q                            # STEP 41   unit tests
```

Resume / subset:

```bash
python run_all.py --from train_source --quick
python run_all.py --only vanet,eval-control --quick
```

Dashboard (spec §51):

```bash
streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0
# Codespaces auto-forwards port 8501
```

## 15. Experiment matrix

| # | experiment | script | key output |
|---|---|---|---|
| 1 | source-domain prediction (9 baselines + proposed) | `train_source.py` | `outputs/tables/source_prediction_results.csv` |
| 2 | transfer strategies (none/frozen/full/DA) | `transfer_target.py` | `transfer_results.csv` |
| 3 | VANET penetration sweep | `simulate_vanet.py` | `vanet_results.csv` |
| 4 | PDR sweep | `simulate_vanet.py` | `vanet_results.csv` |
| 5 | latency sweep | `simulate_vanet.py` | `vanet_results.csv` |
| 6 | traffic-demand sweep (LOW…VERY_HIGH) | `evaluate_prediction.py` | `demand_results.csv` |
| 7 | communication stress tests | `simulate_vanet.py` | `vanet_stress_results.csv` |
| 8 | corridor comparison | `evaluate_prediction.py` | `corridor_results.csv` |
| – | ablation (baseline → full, each term removed) | `evaluate_prediction.py` | `ablation_results.csv` |
| – | closed-loop control + safety + significance | `evaluate_control.py` | `control_results.csv`, `control_significance.csv` |

Seeds: `42, 123, 2024, 3407, 7777` (reduced to `42, 123, 2024` in `--quick`).
Results are reported as **mean ± std**; paired *t*-test + Wilcoxon + Cohen's *d*
for the control comparison. `--quick` / `--smoke` reduce epochs, seeds, demand
and MC samples — every reduction is logged, never silent.

## 16. Results

Generated, not hand-written. After a run see:

- `outputs/tables/*.csv` and `*.md`
- `outputs/figures/01…24_*.png` (missing-data figures become labelled placeholders)
- `outputs/maps/puducherry_vanet_map.html` (interactive)
- `outputs/reports/final_research_report.md` (24 sections, with H1–H9 verdicts)
- `outputs/logs/*.json` (one provenance record per experiment)

## 17. Limitations

- **Target traffic is simulated.** No measured Puducherry trajectory dataset exists
  for these corridors; the sim-to-real gap is a threat to validity.
- The IDM fallback abstracts a corridor as a 1-D road + one signalised
  intersection with a cross-street queue.
- Source sensor coordinates may be unavailable → source graph from adjacency only.
- Only the hidden-space encoder transfers when input parametrisations differ.
- `--quick`/`--smoke` numbers are indicative, not final.
- The VANET channel is a stochastic abstraction, not a full PHY/MAC (no ns-3/Veins).

## 18. Reproducibility

Every result is traceable to: **dataset** (name + `is_synthetic` +
`dataset_report.json`) + **OSM retrieval** (`osm_provenance.json`) +
**configuration** (`config.yaml`, profile recorded) + **random seed** +
**model checkpoint** (`models/checkpoints/`) + **experiment log**
(`outputs/logs/<experiment_id>.json`: git commit, Python/PyTorch versions,
hardware, wall time, metrics). `run_all.py` writes `outputs/logs/run_all_*.log`.
`scripts/generate_report.py` embeds the environment JSON into the report.

```bash
python scripts/check_env.py
python scripts/download_dataset.py
python run_all.py --full
```

## Citation

```bibtex
@software{puducherry_vanet_transfer,
  title  = {Communication-Aware Transfer Learning for Real-Time Traffic-State
            Prediction and Proactive VANET Traffic Control on the Puducherry
            Road Network},
  year   = {2026},
  note   = {AoI-Aware Spatio-Temporal GNN with Transfer Learning; research code}
}
```

Please also cite METR-LA/PEMS-BAY (Li et al., ICLR 2018), OpenStreetMap
contributors, and SUMO (Lopez et al., ITSC 2018) as appropriate.

## Project structure

```
puducherry-vanet-transfer/
├── .devcontainer/        Codespaces: SUMO + Python bootstrap
├── config.yaml           single source of truth (+ quick/smoke profiles)
├── run_all.py            end-to-end orchestrator
├── src/
│   ├── data/  osm/  sumo/  vanet/  preprocessing/
│   ├── models/           layers, baselines, proposed, losses, training engine
│   ├── transfer/  uncertainty/  control/  evaluation/  visualization/
│   └── experiments/      target-scenario assembly + predictor adapter
├── scripts/              one file per pipeline STEP
├── dashboard/app.py      Streamlit
├── tests/                pytest
└── outputs/              tables, figures, maps, reports, logs   (git-ignored)
```
