"""Configuration loading with dotted access and profile merging.

`load_config()` reads ``config.yaml``.  Passing ``profile="quick"`` (or
``"smoke"``) deep-merges the corresponding override block on top of the base
config so every downstream script sees a single flat view.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"

# `smoke` = even smaller than `quick`; used by CI to prove the wiring works.
_SMOKE_OVERRIDE: dict[str, Any] = {
    "training": {"epochs": 2, "batch_size": 64, "hidden_dim": 16,
                 "window_stride": 12, "max_train_windows": 1200,
                 "target_history_length": 6, "target_prediction_horizon": 3},
    "transfer": {"finetune_epochs": 2},
    "baselines": {"enabled": ["historical_average", "linear_regression", "gru", "gcn"]},
    "sumo": {"warmup_s": 60, "sim_duration_s": 600,
             "demand_levels": {"LOW": 30, "MEDIUM": 60, "HIGH": 100, "VERY_HIGH": 150}},
    "vanet": {"penetration": [0.3, 1.0], "pdr": [0.7, 1.0], "latency_ms": [0, 200]},
    "uncertainty": {"mc_samples": 5},
    "control": {"dqn": {"episodes": 4}},
    "experiments": {"seeds": [42, 123]},
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class Config(dict):
    """dict subclass with attribute + dotted-path access."""

    def __getattr__(self, item):
        try:
            val = self[item]
        except KeyError as exc:  # pragma: no cover
            raise AttributeError(item) from exc
        return Config(val) if isinstance(val, dict) else val

    __setattr__ = dict.__setitem__

    def dotted(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return Config(node) if isinstance(node, dict) else node


def load_config(path: str | os.PathLike | None = None,
                profile: str | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    cfg = dict(raw)
    if profile == "quick":
        cfg = _deep_merge(cfg, raw.get("quick", {}))
    elif profile == "smoke":
        cfg = _deep_merge(cfg, raw.get("quick", {}))
        cfg = _deep_merge(cfg, _SMOKE_OVERRIDE)
    cfg.pop("quick", None)

    cfg["_meta"] = {"config_path": str(path), "profile": profile or "full",
                    "repo_root": str(REPO_ROOT)}
    return Config(cfg)


def resolve_device(cfg: Config) -> str:
    want = cfg.get("project", {}).get("device", "auto")
    if want != "auto":
        return want
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def target_window(cfg: Config) -> tuple[int, int]:
    """(history, horizon) for the TARGET domain -- may differ from the source."""
    tr = cfg["training"]
    return (int(tr.get("target_history_length", tr["history_length"])),
            int(tr.get("target_prediction_horizon", tr["prediction_horizon"])))


def out_dir(cfg: Config, *parts: str) -> Path:
    d = REPO_ROOT / cfg["project"]["outputs_dir"]
    for p in parts:
        d = d / p
    d.mkdir(parents=True, exist_ok=True)
    return d


def data_dir(cfg: Config, *parts: str) -> Path:
    d = REPO_ROOT / cfg["project"]["data_dir"]
    for p in parts:
        d = d / p
    d.mkdir(parents=True, exist_ok=True)
    return d
