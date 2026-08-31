"""Streamlit dashboard (spec section 51).

    streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0

Runs a Puducherry target-domain scenario live (IDM micro-simulator or SUMO),
applies the selected VANET impairments, and -- if a transferred model
checkpoint exists -- shows the traffic-state prediction, its uncertainty and
the closed-loop control result.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.utils import load_config, set_seed
from src.experiments import run_scenario
from src.control import build_controller
from src.sumo import run_target_simulation
from src.experiments import load_corridor

st.set_page_config(page_title="Puducherry VANET transfer", layout="wide")
cfg = load_config(profile="quick")

st.title("Communication-Aware Transfer Learning — Puducherry VANET")
st.info(
    "**SOURCE DATASET:** REAL PUBLIC DATA (METR-LA / PEMS-BAY)  |  "
    "**ROAD NETWORK:** OpenStreetMap (real geometry)  |  "
    "**PUDUCHERRY TARGET TRAFFIC:** SIMULATED  |  "
    "**VANET COMMUNICATION:** SIMULATED (penetration / PDR / latency / AoI)"
)

sb = st.sidebar
sb.header("Scenario controls")
corridor_id = sb.selectbox("Corridor", ["corridor_1", "corridor_2"],
                           format_func=lambda c: {"corridor_1": "Reddiyarpalayam → Beach Road",
                                                  "corridor_2": "Nainarmandapam → Lawspet"}[c])
demand = sb.selectbox("Traffic demand", list(cfg["sumo"]["demand_levels"].keys()), index=1)
pen = sb.select_slider("VANET penetration", cfg["vanet"]["penetration"], value=0.30)
pdr = sb.select_slider("Packet delivery ratio", cfg["vanet"]["pdr"], value=0.90)
lat = sb.select_slider("Latency (ms)", cfg["vanet"]["latency_ms"], value=100)
controller_name = sb.selectbox("Controller", cfg["control"]["controllers"], index=1)
seed = sb.selectbox("Random seed", cfg["experiments"]["seeds"], index=0)
go = sb.button("Run scenario", type="primary")


@st.cache_data(show_spinner=True)
def _run(corridor_id, demand, pen, pdr, lat, seed):
    set_seed(int(seed))
    sc = run_scenario(cfg, corridor_id, int(seed), demand_level=demand,
                      penetration=float(pen), pdr=float(pdr), latency_ms=float(lat))
    return sc


@st.cache_data(show_spinner=True)
def _run_control(corridor_id, demand, controller_name, seed):
    set_seed(int(seed))
    corridor = load_corridor(cfg, corridor_id)
    ctrl = build_controller(controller_name, cfg)
    out = run_target_simulation(cfg, corridor, demand, int(seed), controller=ctrl)
    return out.metrics, out.backend


if go:
    sc = _run(corridor_id, demand, pen, pdr, lat, seed)
    so, po = sc["sim_out"], sc["partial_obs"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Backend", so.backend)
    c2.metric("Mean AoI (s)", po.aoi_stats["mean"])
    c3.metric("P95 AoI (s)", po.aoi_stats["p95"])
    c4.metric("Cell coverage", f"{po.delivery_stats['mean_cell_coverage']*100:.0f}%")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Travel time (s)", so.metrics.get("travel_time_mean_s"))
    c2.metric("Mean queue (veh)", so.metrics.get("queue_mean_veh"))
    c3.metric("Throughput (veh/h)", so.metrics.get("throughput_veh_per_h"))
    c4.metric("Min TTC (s)", so.metrics.get("min_ttc_s"))

    st.subheader("Ground-truth vs observed segment speed")
    gt = so.state[..., 0]; obs = po.observed_state[..., 0]
    df = pd.DataFrame({"time_s": np.repeat(so.times, gt.shape[1]),
                       "segment": np.tile(np.arange(gt.shape[1]), len(so.times)),
                       "ground_truth": gt.reshape(-1), "observed": obs.reshape(-1)})
    a, b = st.columns(2)
    a.caption("Ground truth (simulator)")
    a.dataframe(pd.DataFrame(gt, columns=[f"seg{i}" for i in range(gt.shape[1])]).round(1),
                height=240)
    b.caption("VANET-observed (0 where unobserved)")
    b.dataframe(pd.DataFrame(obs, columns=[f"seg{i}" for i in range(obs.shape[1])]).round(1),
                height=240)

    st.subheader("Age of Information over time")
    st.line_chart(pd.DataFrame({"mean AoI": po.aoi.mean(1),
                                "p95 AoI": np.percentile(po.aoi, 95, axis=1)},
                               index=so.times))

    st.subheader(f"Closed-loop control — {controller_name}")
    metrics, backend = _run_control(corridor_id, demand, controller_name, seed)
    st.json(metrics)

    mp = REPO / "outputs" / "maps" / "puducherry_vanet_map.html"
    if mp.exists():
        st.subheader("Interactive map (pre-generated)")
        components.html(mp.read_text(), height=520, scrolling=True)
    else:
        st.caption("Run `python scripts/generate_report.py` to build the "
                   "interactive map.")
else:
    st.caption("Set the controls in the sidebar and press **Run scenario**.")
    tbl = REPO / "outputs" / "tables" / "vanet_results.csv"
    if tbl.exists():
        st.subheader("Latest VANET sweep results")
        st.dataframe(pd.read_csv(tbl))
