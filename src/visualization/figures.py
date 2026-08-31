"""Publication-quality figure generation (spec section 49).

Every figure in the required list is always produced.  When the underlying
experiment output is missing, a labelled placeholder is written instead (and a
warning logged) so the figure set is complete and nothing is silently skipped.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 160, "font.size": 10,
    "axes.grid": True, "grid.alpha": 0.3, "figure.autolayout": True,
})

REQUIRED_FIGURES = [
    "01_puducherry_osm_network", "02_corridor_reddiyarpalayam_beach",
    "03_corridor_nainarmandapam_lawspet", "04_source_dataset_graph",
    "05_source_prediction", "06_transfer_architecture", "07_vanet_architecture",
    "08_vehicle_comm_graph", "09_aoi_vs_time", "10_aoi_distribution",
    "11_pdr_vs_rmse", "12_latency_vs_rmse", "13_penetration_vs_rmse",
    "14_transfer_comparison", "15_baseline_comparison", "16_ablation_comparison",
    "17_uncertainty_calibration", "18_uncertainty_vs_error",
    "19_travel_time_comparison", "20_queue_length_comparison",
    "21_throughput_comparison", "22_ttc_comparison",
    "23_computational_complexity", "24_final_control_map",
]


def savefig(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def placeholder_figure(path, title, reason="data not yet generated - run the "
                                           "corresponding pipeline stage"):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axis("off")
    ax.text(0.5, 0.6, title, ha="center", va="center", fontsize=13, wrap=True)
    ax.text(0.5, 0.35, reason, ha="center", va="center", fontsize=9,
            color="crimson", wrap=True)
    return savefig(fig, path)


def bar_comparison(labels, values, title, ylabel, path, yerr=None, rotate=30):
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.9), 4))
    ax.bar(range(len(labels)), values, yerr=yerr, capsize=3,
           color="#3b7dd8", alpha=0.9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=rotate, ha="right")
    ax.set_ylabel(ylabel); ax.set_title(title)
    return savefig(fig, path)


def line_sweep(x, series: dict, title, xlabel, ylabel, path):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for name, ys in series.items():
        ax.plot(x, ys, marker="o", label=str(name))
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(fontsize=8)
    return savefig(fig, path)


def plot_osm_network(graphml_path, corridor_geojson, path):
    try:
        import osmnx as ox
        G = ox.load_graphml(graphml_path)
        fig, ax = ox.plot_graph(G, show=False, close=False, node_size=0,
                                edge_color="#999999", edge_linewidth=0.5,
                                bgcolor="white")
        if Path(corridor_geojson).exists():
            gj = json.loads(Path(corridor_geojson).read_text())
            colors = ["#d1495b", "#2a9d8f"]
            for i, feat in enumerate([f for f in gj["features"]
                                      if f["geometry"]["type"] == "LineString"]):
                xs = [c[0] for c in feat["geometry"]["coordinates"]]
                ys = [c[1] for c in feat["geometry"]["coordinates"]]
                ax.plot(xs, ys, color=colors[i % 2], linewidth=3,
                        label=feat["properties"]["name"])
            ax.legend(fontsize=8, loc="lower right")
        ax.set_title("Puducherry OSM driving network + study corridors")
        return savefig(fig, path)
    except Exception as exc:
        return placeholder_figure(path, "Puducherry OSM road network",
                                  f"OSM plot unavailable: {exc}")


def plot_corridor(ml_graph_json, path, color="#d1495b"):
    try:
        d = json.loads(Path(ml_graph_json).read_text())
        coords = d["route_coords"]
        xs = [c[0] for c in coords]; ys = [c[1] for c in coords]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(xs, ys, color=color, linewidth=2.5)
        ax.scatter([xs[0], xs[-1]], [ys[0], ys[-1]], c=["green", "red"], zorder=5)
        ax.set_title(f"{d['name']}  ({d['length_m']:.0f} m, "
                     f"{d['n_nodes']} nodes)")
        ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
        return savefig(fig, path)
    except Exception as exc:
        return placeholder_figure(path, Path(path).stem, f"corridor data: {exc}")


def plot_aoi_time(times, aoi_series, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    a = np.asarray(aoi_series)
    ax.plot(times, a.mean(1), label="mean AoI")
    ax.fill_between(times, np.percentile(a, 5, axis=1),
                    np.percentile(a, 95, axis=1), alpha=0.2, label="5-95%")
    ax.set_xlabel("simulation time (s)"); ax.set_ylabel("AoI (s)")
    ax.set_title("Age of Information vs time"); ax.legend(fontsize=8)
    return savefig(fig, path)


def plot_aoi_distribution(aoi_series, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(np.asarray(aoi_series).reshape(-1), bins=40, color="#8e44ad", alpha=0.85)
    ax.set_xlabel("AoI (s)"); ax.set_ylabel("count")
    ax.set_title("AoI distribution")
    return savefig(fig, path)


def plot_calibration(curve, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    if curve:
        nom = [c["nominal"] for c in curve]
        emp = [c["empirical"] for c in curve]
        ax.plot([0, 1], [0, 1], "k--", label="ideal")
        ax.plot(nom, emp, "o-", label="model")
        ax.set_xlabel("nominal coverage"); ax.set_ylabel("empirical coverage")
    ax.set_title("Uncertainty calibration"); ax.legend(fontsize=8)
    return savefig(fig, path)


def plot_uncertainty_vs_error(std, abs_err, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    s = np.asarray(std).reshape(-1); e = np.asarray(abs_err).reshape(-1)
    n = min(s.size, e.size, 5000)
    idx = np.random.default_rng(0).choice(min(s.size, e.size), n, replace=False)
    ax.scatter(s[idx], e[idx], s=6, alpha=0.3)
    if s.std() > 0:
        r = np.corrcoef(s[:n], e[:n])[0, 1]
        ax.set_title(f"Predicted uncertainty vs |error|  (r = {r:.2f})")
    ax.set_xlabel("predictive std"); ax.set_ylabel("absolute error")
    return savefig(fig, path)


def plot_training_curves(history, path, title="training curves"):
    fig, ax = plt.subplots(figsize=(6, 4))
    if history:
        ep = [h["epoch"] for h in history]
        ax.plot(ep, [h["train_loss"] for h in history], label="train loss")
        ax.plot(ep, [h["val_mae"] for h in history], label="val MAE")
        ax.legend(fontsize=8)
    ax.set_xlabel("epoch"); ax.set_title(title)
    return savefig(fig, path)


def _schematic(path, title, boxes):
    fig, ax = plt.subplots(figsize=(6, 7)); ax.axis("off")
    y = 0.95
    for b in boxes:
        ax.add_patch(plt.Rectangle((0.15, y - 0.06), 0.7, 0.06, fill=True,
                                   facecolor="#eaf2fb", edgecolor="#3b7dd8"))
        ax.text(0.5, y - 0.03, b, ha="center", va="center", fontsize=9)
        if y < 0.9:
            ax.annotate("", xy=(0.5, y + 0.005), xytext=(0.5, y + 0.03),
                        arrowprops=dict(arrowstyle="->"))
        y -= 0.11
    ax.set_title(title)
    return savefig(fig, path)


def generate_all_figures(cfg, logger=None):
    root = Path(cfg["_meta"]["repo_root"])
    figs = root / cfg["project"]["outputs_dir"] / "figures"
    tables = root / cfg["project"]["outputs_dir"] / "tables"
    figs.mkdir(parents=True, exist_ok=True)
    made = {}

    def _log(name, p):
        made[name] = str(p)
        if logger:
            logger.info(f"  figure: {Path(p).name}")

    # 1-3 OSM
    _log("01", plot_osm_network(root / cfg["osm"]["graphml"],
                                root / cfg["osm"]["corridor_geojson"],
                                figs / "01_puducherry_osm_network.png"))
    for i, cid in enumerate(["corridor_1", "corridor_2"], start=2):
        mlj = root / "data" / "osm" / f"{cid}_ml_graph.json"
        _log(f"0{i}", plot_corridor(mlj, figs / f"0{i}_corridor_{cid}.png",
                                    color=["#d1495b", "#2a9d8f"][i - 2]))
    # 6-7 schematics
    _log("06", _schematic(figs / "06_transfer_architecture.png",
                          "Transfer-learning architecture",
                          ["Source dataset (METR-LA)", "Spatial encoder (GAT) [transferable]",
                           "Temporal encoder (GRU)", "AoI-aware attention",
                           "Domain adaptation (MMD/CORAL)", "Transfer adapter",
                           "Puducherry target prediction"]))
    _log("07", _schematic(figs / "07_vanet_architecture.png", "VANET pipeline",
                          ["SUMO / IDM target traffic", "Connected-vehicle penetration",
                           "Packet delivery (PDR)", "Latency", "Age of Information",
                           "Partial observation + mask", "AoI-aware ST-GNN",
                           "Uncertainty", "Risk-aware controller"]))

    # table-driven bar/line figures
    def _csv(name):
        p = tables / name
        if p.exists():
            import pandas as pd
            return pd.read_csv(p)
        return None

    specs = [
        ("05", "05_source_prediction.png", "source_prediction_results.csv",
         "model", "RMSE_mean", "Source-domain prediction (RMSE)"),
        ("14", "14_transfer_comparison.png", "transfer_results.csv",
         "strategy", "RMSE_mean", "Transfer-learning strategies (target RMSE)"),
        ("15", "15_baseline_comparison.png", "source_prediction_results.csv",
         "model", "MAE_mean", "Baseline comparison (source MAE)"),
        ("16", "16_ablation_comparison.png", "ablation_results.csv",
         "variant", "RMSE_mean", "Ablation study (target RMSE)"),
        ("19", "19_travel_time_comparison.png", "control_results.csv",
         "controller", "travel_time_mean_s_mean", "Travel time by controller"),
        ("20", "20_queue_length_comparison.png", "control_results.csv",
         "controller", "queue_mean_veh_mean", "Queue length by controller"),
        ("21", "21_throughput_comparison.png", "control_results.csv",
         "controller", "throughput_veh_per_h_mean", "Throughput by controller"),
        ("22", "22_ttc_comparison.png", "control_results.csv",
         "controller", "min_ttc_s_mean", "Minimum TTC by controller"),
        ("23", "23_computational_complexity.png", "source_prediction_results.csv",
         "model", "num_parameters", "Model size (parameters)"),
    ]
    for tag, fname, csv, xcol, ycol, title in specs:
        df = _csv(csv)
        if df is not None and xcol in df and ycol in df:
            err_col = ycol.replace("_mean", "_std")
            yerr = df[err_col] if err_col in df else None
            _log(tag, bar_comparison(df[xcol].astype(str).tolist(),
                                     df[ycol].tolist(), title, ycol,
                                     figs / fname, yerr=yerr))
        else:
            _log(tag, placeholder_figure(figs / fname, title))

    for tag, fname, csv, xcol, title, xlabel in [
        ("11", "11_pdr_vs_rmse.png", "vanet_results.csv", "pdr",
         "PDR vs prediction RMSE", "packet-delivery ratio"),
        ("12", "12_latency_vs_rmse.png", "vanet_results.csv", "latency_ms",
         "Latency vs prediction RMSE", "latency (ms)"),
        ("13", "13_penetration_vs_rmse.png", "vanet_results.csv", "penetration",
         "VANET penetration vs prediction RMSE", "penetration"),
    ]:
        df = _csv(csv)
        if df is not None and xcol in df and "RMSE_mean" in df:
            sub = df.groupby(xcol)["RMSE_mean"].mean()
            _log(tag, line_sweep(sub.index.tolist(),
                                 {"proposed": sub.values.tolist()},
                                 title, xlabel, "RMSE", figs / fname))
        else:
            _log(tag, placeholder_figure(figs / fname, title))

    # AoI / uncertainty figures from npz dumps if present
    proc = root / "data" / "processed"
    aoi_npz = proc / "vanet_default.npz"
    if aoi_npz.exists():
        d = np.load(aoi_npz)
        _log("09", plot_aoi_time(d["times"], d["aoi"], figs / "09_aoi_vs_time.png"))
        _log("10", plot_aoi_distribution(d["aoi"], figs / "10_aoi_distribution.png"))
        if "comm_graph_x" in d:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(d["comm_graph_x"], d["comm_graph_y"], s=8,
                       c=d.get("comm_graph_connected", None), cmap="coolwarm")
            ax.set_title("Vehicle communication graph snapshot")
            _log("08", savefig(fig, figs / "08_vehicle_comm_graph.png"))
    for tag, fname, title in [("08", "08_vehicle_comm_graph.png", "Vehicle communication graph"),
                              ("09", "09_aoi_vs_time.png", "AoI vs time"),
                              ("10", "10_aoi_distribution.png", "AoI distribution")]:
        if tag not in made:
            _log(tag, placeholder_figure(figs / fname, title))

    calib = proc / "uncertainty_eval.json"
    if calib.exists():
        d = json.loads(calib.read_text())
        _log("17", plot_calibration(d.get("calibration_curve", []),
                                    figs / "17_uncertainty_calibration.png"))
        if "std_flat" in d and "abs_err_flat" in d:
            _log("18", plot_uncertainty_vs_error(d["std_flat"], d["abs_err_flat"],
                                                 figs / "18_uncertainty_vs_error.png"))
    for tag, fname, title in [("17", "17_uncertainty_calibration.png", "Uncertainty calibration"),
                              ("18", "18_uncertainty_vs_error.png", "Uncertainty vs error")]:
        if tag not in made:
            _log(tag, placeholder_figure(figs / fname, title))

    # 4 source graph, 24 final map (static)
    src_npz = proc / "source_windows_meta.json"
    _log("04", placeholder_figure(figs / "04_source_dataset_graph.png",
                                  "Source dataset sensor graph",
                                  "rendered by scripts/inspect_dataset.py when "
                                  "sensor coordinates are available")
         if not (proc / "source_adj.png").exists() else str(proc / "source_adj.png"))
    _log("24", placeholder_figure(figs / "24_final_control_map.png",
                                  "Final Puducherry traffic-control map",
                                  "see outputs/maps/puducherry_vanet_map.html "
                                  "for the interactive version"))

    (figs / "figure_index.json").write_text(json.dumps(made, indent=2))
    return made
