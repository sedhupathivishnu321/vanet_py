"""Assemble a full Puducherry target-domain scenario:

    OSM corridor -> SUMO/IDM traffic -> VANET (penetration/PDR/latency/AoI)
                 -> partial observation -> windowed tensors

and helpers to load the pretrained source encoder for transfer.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.sumo import run_target_simulation
from src.vanet import (CommunicationChannel, build_partial_observation,
                       ns3_available, build_partial_observation_ns3, Ns3Unavailable)
from src.preprocessing.target_features import build_target_tensors
from src.preprocessing.graph import normalize_adjacency


def chain_adjacency(n: int) -> np.ndarray:
    A = np.zeros((n, n), np.float32)
    for i in range(n - 1):
        A[i, i + 1] = A[i + 1, i] = 1.0
    return A


def load_corridor(cfg, corridor_id: str) -> dict:
    root = Path(cfg["_meta"]["repo_root"])
    p = root / "data" / "osm" / f"{corridor_id}_ml_graph.json"
    if p.exists():
        d = json.loads(p.read_text())
        d.setdefault("n_segments", d.get("n_edges", 8))
        return d
    # fallback: minimal corridor from config (flagged)
    spec = next(c for c in cfg["osm"]["corridors"] if c["id"] == corridor_id)
    o = spec["origin_fallback"]; de = spec["destination_fallback"]
    from math import radians, sin, cos, asin, sqrt
    dlat = radians(de[0] - o[0]); dlon = radians(de[1] - o[1])
    a = sin(dlat/2)**2 + cos(radians(o[0]))*cos(radians(de[0]))*sin(dlon/2)**2
    length = 6371000 * 2 * asin(sqrt(a))
    n_seg = max(int(length // 250), 4)
    coords = [[o[1] + (de[1]-o[1])*t/n_seg, o[0] + (de[0]-o[0])*t/n_seg]
              for t in range(n_seg + 1)]
    return {"id": corridor_id, "name": spec["name"], "length_m": round(length, 1),
            "n_edges": n_seg, "n_segments": n_seg, "edges": [], "nodes": [],
            "route_coords": coords, "route_nodes": [],
            "origin_latlon": tuple(o), "destination_latlon": tuple(de),
            "synthetic_corridor": True}


def run_scenario(cfg, corridor_id: str, seed: int, demand_level: str | None = None,
                 penetration: float | None = None, pdr: float | None = None,
                 latency_ms: float | None = None, controller=None,
                 logger=None) -> dict:
    v = cfg["vanet"]
    demand_level = demand_level or cfg["experiments"]["demand_for_vanet_sweep"]
    penetration = v["default_penetration"] if penetration is None else penetration
    pdr = v["default_pdr"] if pdr is None else pdr
    latency_ms = v["default_latency_ms"] if latency_ms is None else latency_ms

    corridor = load_corridor(cfg, corridor_id)
    sim_out = run_target_simulation(cfg, corridor, demand_level, seed,
                                    controller=controller, logger=logger)

    backend = str(v.get("backend", "analytic")).lower()
    comm_backend = "analytic"
    connected_ids: list = []
    if backend == "ns3":
        if ns3_available(cfg):
            try:
                po = build_partial_observation_ns3(sim_out, cfg, penetration, seed,
                                                   logger=logger)
                comm_backend = "ns3"
                from src.vanet.mobility_export import export_ns2_mobility
                connected_ids = export_ns2_mobility(
                    sim_out, Path(cfg["_meta"]["repo_root"]) / "data" / "vanet"
                    / "_equipped.tcl", penetration, seed).equipped_veh_ids
            except Ns3Unavailable as exc:
                if logger:
                    logger.warning(f"  ns-3 backend failed ({str(exc)[:120]}); "
                                   f"falling back to analytic channel")
        elif logger:
            logger.warning("  vanet.backend=ns3 but ns-3 not found "
                           "(run ns3/setup_ns3.sh); using analytic channel")
    if comm_backend == "analytic":
        channel = CommunicationChannel(penetration, pdr, latency_ms, seed=seed)
        po = build_partial_observation(sim_out, channel, cfg)
        connected_ids = channel.connected_ids()

    tensors = build_target_tensors(po, cfg, seed=seed)
    adj = normalize_adjacency(chain_adjacency(sim_out.n_cells))
    return {"corridor": corridor, "sim_out": sim_out, "partial_obs": po,
            "tensors": tensors, "adj_norm": adj, "connected_veh_ids": connected_ids,
            "comm": {"penetration": penetration, "pdr": pdr,
                     "latency_ms": latency_ms, "demand_level": demand_level,
                     "backend": comm_backend},
            "backend": sim_out.backend, "comm_backend": comm_backend}


# --------------------------------------------------------------------------- #
#  Source-model loading for transfer
# --------------------------------------------------------------------------- #
def load_source_context(cfg) -> dict:
    p = Path(cfg["_meta"]["repo_root"]) / "data" / "processed" / "source_domain.npz"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing - run scripts/train_source.py first")
    d = np.load(p, allow_pickle=True)
    return {"adj_norm": d["adj_norm"], "num_nodes": int(d["num_nodes"]),
            "in_dim": int(d["in_dim"]), "horizon": int(d["horizon"]),
            "scaler_mean": float(d["scaler_mean"]), "scaler_std": float(d["scaler_std"]),
            "is_synthetic": bool(d["is_synthetic"]),
            "dataset_name": str(d["dataset_name"])}


def load_proposed_source_state(cfg, seed: int | None = None) -> dict:
    ck = Path(cfg["_meta"]["repo_root"]) / "models" / "checkpoints"
    path = (ck / f"source_proposed_seed{seed}.pt") if seed is not None else \
        (ck / "source_proposed.pt")
    if not path.exists():
        path = ck / "source_proposed.pt"
    if not path.exists():
        raise FileNotFoundError("no source_proposed checkpoint - run train_source.py")
    import torch
    return torch.load(path, map_location="cpu")


def load_proposed_source_model(cfg, in_dim: int, num_nodes: int, horizon: int,
                               seed: int | None = None):
    import torch
    from src.models import build_model
    blob = load_proposed_source_state(cfg, seed)
    model = build_model("proposed", in_dim, num_nodes, horizon, cfg)
    try:
        model.load_state_dict(blob["state_dict"])
    except Exception:
        pass  # shape mismatch tolerated; encoder copied selectively downstream
    return model, blob


# --------------------------------------------------------------------------- #
#  Predictor adapter for the MPC / proposed controllers
# --------------------------------------------------------------------------- #
class TorchPredictorAdapter:
    """Wraps a trained target model so a controller can call ``predictor(ctx)``
    and get (mean_pred, std_pred) as (H, N, 4) arrays.

    It builds the 8-channel model input on the fly from ``ctx['state_history']``
    (raw per-cell [k, N, 4] traffic-state bins produced by the simulator),
    normalising the speed channel with the target scaler and tagging the
    communication features at the configured defaults (mask=1, aoi=0).
    density/flow/queue are reconstructed from the predicted speed via a
    per-channel linear map fitted on the target-train block.
    """

    def __init__(self, model, adj_t, scaler_mean: float, scaler_std: float, cfg,
                 recon_maps=None, mc_samples: int = 10, device: str = "cpu"):
        self.model = model
        self.adj_t = adj_t
        self.mean_, self.std_ = float(scaler_mean), float(scaler_std)
        self.cfg = cfg
        self.device = device
        self.mc = int(mc_samples)
        self.recon = recon_maps or {}
        from src.utils.config import target_window
        self.L = target_window(cfg)[0]
        self._pdr = float(cfg["vanet"]["default_pdr"])
        self._lat = min(float(cfg["vanet"]["default_latency_ms"]) / 500.0, 1.0)

    def _features(self, hist):
        hist = np.asarray(hist, np.float32)
        if hist.ndim != 3:
            return None
        k, N, _ = hist.shape
        if k < self.L:
            hist = np.concatenate([np.repeat(hist[:1], self.L - k, 0), hist], 0)
        hist = hist[-self.L:]
        f = np.zeros((self.L, N, 8), np.float32)
        f[..., 0] = (hist[..., 0] - self.mean_) / (self.std_ or 1.0)
        f[..., 1] = hist[..., 1]
        f[..., 2] = hist[..., 2]
        f[..., 3] = hist[..., 3]
        f[..., 4] = 1.0
        f[..., 6] = self._pdr
        f[..., 7] = self._lat
        return f

    def __call__(self, ctx):
        hist = ctx.get("state_history")
        f = self._features(hist) if hist is not None else None
        if f is None:
            return None, None
        import torch
        from src.uncertainty import mc_dropout_predict
        x = torch.from_numpy(f[None]).float()
        mean, std = mc_dropout_predict(self.model, x, self.adj_t, aoi=None,
                                       K=self.mc, device=self.device)
        mean = mean[0, ..., 0] * (self.std_ or 1.0) + self.mean_   # (H, N)
        std = std[0, ..., 0] * (self.std_ or 1.0)
        H, N = mean.shape
        out = np.zeros((H, N, 4), np.float32)
        out[..., 0] = mean
        for ch, (a, b) in self.recon.items():
            out[..., int(ch)] = a * mean + b
        s = np.zeros((H, N, 4), np.float32)
        s[..., 0] = std
        return out, s
