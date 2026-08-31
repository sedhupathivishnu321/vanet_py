"""Load a *real* public traffic dataset (METR-LA / PEMS-BAY) for source-domain
pre-training.

Supported on-disk layouts (auto-detected recursively under ``data/source/``):

1. DCRNN layout (what ``scripts/download_dataset.py`` fetches):
       data/source/METR-LA/metr-la.h5    (pandas DataFrame, index=timestamps)
       data/source/METR-LA/adj_mx.pkl    (pickle: [sensor_ids, id_to_ind, adj_mx])
       data/source/METR-LA/graph_sensor_locations.csv  (index,sensor_id,lat,lon)
2. PyTorch-Geometric-Temporal zip layout:
       .../node_values.npy   (T, N, C)   +   .../adj_mat.npy   (N, N)
3. Raw CSV: any ``*.csv`` (index=timestamps, cols=sensors) that is not a
   sensor-graph auxiliary file.
Sensor coordinates are aligned to the time-series column order by sensor id.

If none is present and ``allow_synthetic_fallback`` is true, a clearly-labelled
SYNTHETIC dataset is produced *for smoke testing only* -- every downstream
artefact is stamped ``SYNTHETIC`` so results can never be mistaken for real.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

# METR-LA is sampled every 5 minutes; PEMS-BAY likewise.
_SAMPLING_INTERVAL_S = {"METR-LA": 300, "PEMS-BAY": 300}


@dataclass
class DatasetReport:
    dataset_name: str
    is_synthetic: bool
    num_nodes: int
    num_timestamps: int
    sampling_interval_s: Optional[int]
    num_features: int
    feature_names: list
    missing_value_fraction: float
    missing_value_count: int
    geographic_information: bool
    adjacency_available: bool
    traffic_variables: list
    value_min: float
    value_max: float
    value_mean: float
    value_std: float
    train_samples: int = 0
    val_samples: int = 0
    test_samples: int = 0
    history_length: int = 0
    prediction_horizon: int = 0
    source_layout: str = ""
    notes: list = field(default_factory=list)

    def to_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)

    def to_csv(self, path: str | Path) -> None:
        import csv
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        d = asdict(self)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["field", "value"])
            for k, v in d.items():
                w.writerow([k, v])


@dataclass
class SourceDataset:
    name: str
    data: np.ndarray            # (T, N, C) float32, feature 0 == traffic speed
    adj: Optional[np.ndarray]   # (N, N) weighted adjacency or None
    coords: Optional[np.ndarray]  # (N, 2) lat/lon or None
    sampling_interval_s: Optional[int]
    is_synthetic: bool
    feature_names: list
    layout: str
    missing_mask: Optional[np.ndarray] = None  # (T, N) bool, True == missing

    @property
    def num_nodes(self) -> int:
        return self.data.shape[1]

    @property
    def speed(self) -> np.ndarray:
        return self.data[..., 0]

    def build_report(self) -> DatasetReport:
        vals = self.speed.astype(np.float64)
        if self.missing_mask is not None:
            miss = int(self.missing_mask.sum())
            frac = float(self.missing_mask.mean())
            finite = vals[~self.missing_mask]
        else:
            # heuristic: exact zeros are the METR-LA missing sentinel
            zeros = np.isclose(vals, 0.0)
            miss = int(zeros.sum())
            frac = float(zeros.mean())
            finite = vals[~zeros]
        finite = finite[np.isfinite(finite)]
        return DatasetReport(
            dataset_name=self.name,
            is_synthetic=self.is_synthetic,
            num_nodes=int(self.data.shape[1]),
            num_timestamps=int(self.data.shape[0]),
            sampling_interval_s=self.sampling_interval_s,
            num_features=int(self.data.shape[2]),
            feature_names=list(self.feature_names),
            missing_value_fraction=round(frac, 6),
            missing_value_count=miss,
            geographic_information=self.coords is not None,
            adjacency_available=self.adj is not None,
            traffic_variables=["speed_mph" if not self.is_synthetic else "speed_synthetic"],
            value_min=float(np.min(finite)) if finite.size else 0.0,
            value_max=float(np.max(finite)) if finite.size else 0.0,
            value_mean=float(np.mean(finite)) if finite.size else 0.0,
            value_std=float(np.std(finite)) if finite.size else 0.0,
            source_layout=self.layout,
        )


# --------------------------------------------------------------------------- #
#  Loaders per layout
# --------------------------------------------------------------------------- #
def _find(root: Path, *patterns: str) -> list[str]:
    """Recursive glob under `root` for any of the given patterns."""
    out: list[str] = []
    for p in patterns:
        out += [str(x) for x in root.rglob(p)]
    return sorted(set(out))


def _prefer(paths: list[str], name: str) -> list[str]:
    """Put files whose path mentions the configured dataset name first."""
    key = name.lower().replace("-", "")
    return sorted(paths, key=lambda p: (key not in Path(p).parent.name.lower()
                                        .replace("-", ""), p))


_AUX_CSV = ("sensor_locations", "locations", "sensor_ids", "_ids", "distances",
            "adj", "w_", "se_")


def _try_pyg_temporal(root: Path, name: str):
    nvs = _prefer(_find(root, "node_values.npy"), name)
    if not nvs:
        return None
    nv = Path(nvs[0])
    am = nv.with_name("adj_mat.npy")
    data = np.load(nv).astype(np.float32)
    if data.ndim == 2:
        data = data[:, :, None]
    adj = np.load(am).astype(np.float32) if am.exists() else None
    feats = ["speed"] + [f"cov{i}" for i in range(1, data.shape[2])]
    if data.shape[2] >= 2:
        feats[1] = "time_of_day"
    return SourceDataset(name, data, adj, None, _SAMPLING_INTERVAL_S.get(name, 300),
                         False, feats, "pyg_temporal_npy")


def _read_h5_speed(path: str):
    """Return (speed [T, N] float32, sensor_id list).  Tries pandas first, then
    a direct h5py read of the pandas 'fixed' layout (works across pandas versions
    that broke ``read_hdf`` on old files)."""
    try:
        import pandas as pd
        df = pd.read_hdf(path)
        return df.to_numpy(dtype=np.float32), [str(c) for c in df.columns]
    except Exception:
        import h5py
        with h5py.File(path, "r") as f:
            grp = f[list(f.keys())[0]]                      # e.g. 'df' or 'speed'
            vals = np.asarray(grp["block0_values"], dtype=np.float32)
            try:
                ids = [x.decode() if isinstance(x, bytes) else str(x)
                       for x in np.asarray(grp["axis0"])]
            except Exception:
                ids = [str(i) for i in range(vals.shape[1])]
        return vals, ids


def _try_dcrnn_h5(root: Path, name: str):
    h5s = _prefer(_find(root, "*.h5"), name)
    if not h5s:
        return None
    speed, cols = _read_h5_speed(h5s[0])
    arr = speed[:, :, None]
    miss = np.isclose(arr[..., 0], 0.0)
    adj = None
    for pkl in _find(root, "*adj*.pkl"):
        try:
            with open(pkl, "rb") as fh:
                obj = pickle.load(fh, encoding="latin1")
            adj = np.asarray(obj[-1], dtype=np.float32)
            break
        except Exception:
            continue
    coords = _try_coords(root, [str(c) for c in cols])
    return SourceDataset(name, arr, adj, coords, _SAMPLING_INTERVAL_S.get(name, 300),
                         False, ["speed"], "dcrnn_h5", missing_mask=miss)


def _try_csv(root: Path, name: str):
    csvs = [c for c in _prefer(_find(root, "*.csv"), name)
            if not any(a in Path(c).name.lower() for a in _AUX_CSV)]
    if not csvs:
        return None
    import pandas as pd
    df = pd.read_csv(csvs[0], index_col=0)
    arr = df.to_numpy(dtype=np.float32)[:, :, None]
    miss = ~np.isfinite(arr[..., 0]) | np.isclose(arr[..., 0], 0.0)
    coords = _try_coords(root, [str(c) for c in df.columns])
    return SourceDataset(name, np.nan_to_num(arr), None, coords,
                         _SAMPLING_INTERVAL_S.get(name, 300), False, ["speed"],
                         "raw_csv", missing_mask=miss)


def _try_coords(root: Path, sensor_ids: list) -> Optional[np.ndarray]:
    """Return (N, 2) lat/lon **aligned to `sensor_ids` order** where possible.

    Handles both the METR-LA layout (`index,sensor_id,latitude,longitude` with
    a header) and the PEMS-BAY layout (`id,lat,lon`, no header).
    """
    import pandas as pd
    for c in _find(root, "*sensor_locations*.csv", "*locations*.csv"):
        try:
            head = pd.read_csv(c, nrows=1, header=None).iloc[0].tolist()
            has_header = not _looks_numeric(head[0])
            loc = pd.read_csv(c) if has_header else pd.read_csv(
                c, header=None, names=["sensor_id", "latitude", "longitude"])
            lat_col = next(x for x in loc.columns if "lat" in str(x).lower())
            lon_col = next(x for x in loc.columns if "lon" in str(x).lower())
            id_col = next((x for x in loc.columns
                           if "id" in str(x).lower() and "index" not in str(x).lower()),
                          None)
            if id_col is not None and sensor_ids:
                m = {str(r[id_col]): (float(r[lat_col]), float(r[lon_col]))
                     for _, r in loc.iterrows()}
                if all(str(s) in m for s in sensor_ids):
                    return np.array([m[str(s)] for s in sensor_ids], dtype=np.float64)
            return loc[[lat_col, lon_col]].to_numpy(dtype=np.float64)
        except Exception:
            continue
    return None


def _looks_numeric(x) -> bool:
    try:
        float(x)
        return True
    except (TypeError, ValueError):
        return False


def _synthetic(name: str, num_nodes: int = 40, days: int = 14,
               seed: int = 0) -> SourceDataset:
    """Smoke-test-only synthetic source data. Loudly labelled SYNTHETIC."""
    rng = np.random.default_rng(seed)
    steps_per_day = 24 * 12  # 5-min bins
    T = days * steps_per_day
    t = np.arange(T)
    tod = (t % steps_per_day) / steps_per_day
    # diurnal speed profile: fast at night, slow at rush hours
    base = 62 - 22 * np.exp(-((tod - 0.34) ** 2) / 0.004) \
              - 26 * np.exp(-((tod - 0.72) ** 2) / 0.004)
    coords = np.c_[11.0 + rng.random(num_nodes) * 0.4,
                   79.0 + rng.random(num_nodes) * 0.4]
    # spatial correlation via a random geometric graph
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    sigma = np.std(d[d > 0])
    adj = np.exp(-(d ** 2) / (2 * sigma ** 2))
    adj[adj < 0.1] = 0.0
    np.fill_diagonal(adj, 1.0)
    node_phase = rng.normal(0, 0.03, num_nodes)
    speed = (base[:, None] + 40 * node_phase[None, :]
             + rng.normal(0, 2.5, (T, num_nodes)))
    # propagate a bit of spatial smoothing
    W = adj / adj.sum(1, keepdims=True)
    speed = 0.6 * speed + 0.4 * speed @ W.T
    speed = np.clip(speed, 3, 75).astype(np.float32)
    tod_feat = np.tile(tod[:, None], (1, num_nodes)).astype(np.float32)
    data = np.stack([speed, tod_feat], axis=-1)
    return SourceDataset(f"SYNTHETIC-{name}", data, adj.astype(np.float32), coords,
                         300, True, ["speed_synthetic", "time_of_day"],
                         "synthetic_fallback")


def load_source_dataset(cfg) -> SourceDataset:
    name = cfg["source_dataset"]["name"]
    root = Path(cfg["_meta"]["repo_root"]) / cfg["source_dataset"]["root"]
    root.mkdir(parents=True, exist_ok=True)

    for loader in (_try_pyg_temporal, _try_dcrnn_h5, _try_csv):
        ds = loader(root, name)
        if ds is not None:
            return ds

    if cfg["source_dataset"].get("allow_synthetic_fallback", False):
        import warnings
        warnings.warn(
            "\n" + "!" * 72 +
            "\nNo real source dataset found under data/source/. Falling back to a"
            "\nSYNTHETIC dataset. This is for SMOKE TESTING ONLY -- results are NOT"
            "\nscientifically valid. Run `python scripts/download_dataset.py` to get"
            "\nthe real METR-LA / PEMS-BAY data.\n" + "!" * 72)
        return _synthetic(name)

    raise FileNotFoundError(
        f"No source dataset found under {root}. Run "
        f"`python scripts/download_dataset.py` first, or set "
        f"source_dataset.allow_synthetic_fallback: true for a smoke test.")
