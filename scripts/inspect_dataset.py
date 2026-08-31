#!/usr/bin/env python
"""STEP 6-7: inspect + validate the source dataset and create the chronological
train/val/test split (spec sections 5, 44).

Writes:
    outputs/dataset_report.json
    outputs/dataset_report.csv
    data/processed/source_split.json
    data/processed/source_adj.png   (if sensor coordinates / adjacency exist)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from _common import base_parser, load, env_info


def main() -> int:
    args = base_parser(__doc__).parse_args()
    cfg, log = load(args)
    from src.data import load_source_dataset, chronological_split, make_windows
    from src.utils import ExperimentLogger

    out = Path(cfg["_meta"]["repo_root"]) / cfg["project"]["outputs_dir"]
    proc = Path(cfg["_meta"]["repo_root"]) / "data" / "processed"
    proc.mkdir(parents=True, exist_ok=True)

    with ExperimentLogger(out / "logs", "inspect_dataset", dict(cfg),
                          env_info(), cfg["project"]["seed_default"]) as elog:
        ds = load_source_dataset(cfg)
        rep = ds.build_report()
        L = int(cfg["training"]["history_length"])
        H = int(cfg["training"]["prediction_horizon"])
        sp = chronological_split(ds.data.shape[0], tuple(cfg["training"]["split"]))
        rep.history_length, rep.prediction_horizon = L, H
        rep.train_samples = len(make_windows(ds.data, L, H, *sp.train))
        rep.val_samples = len(make_windows(ds.data, L, H, *sp.val))
        rep.test_samples = len(make_windows(ds.data, L, H, *sp.test))
        if ds.is_synthetic:
            rep.notes.append("SYNTHETIC dataset - smoke test only, results NOT "
                             "scientifically valid.")

        rep.to_json(out / "dataset_report.json")
        rep.to_csv(out / "dataset_report.csv")
        (proc / "source_split.json").write_text(json.dumps(sp.as_dict(), indent=2))
        (proc / "source_windows_meta.json").write_text(json.dumps({
            "history": L, "horizon": H, "num_nodes": rep.num_nodes,
            "num_features": rep.num_features, "is_synthetic": ds.is_synthetic,
        }, indent=2))

        # optional adjacency / sensor-graph figure
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            A = ds.adj
            if A is None and ds.coords is not None:
                from src.preprocessing import gaussian_kernel_adjacency
                A = gaussian_kernel_adjacency(
                    ds.coords, cfg["source_dataset"].get("gaussian_kernel_sigma"),
                    float(cfg["source_dataset"]["adjacency_threshold"]))
            if A is not None:
                fig, ax = plt.subplots(1, 2 if ds.coords is not None else 1,
                                       figsize=(10, 4))
                axes = np.atleast_1d(ax)
                axes[0].imshow(A, cmap="viridis"); axes[0].set_title("adjacency")
                if ds.coords is not None:
                    axes[1].scatter(ds.coords[:, 1], ds.coords[:, 0], s=8)
                    axes[1].set_title("sensor locations")
                fig.savefig(proc / "source_adj.png", bbox_inches="tight")
                plt.close(fig)
                elog.add_artifact(proc / "source_adj.png")
        except Exception as exc:
            log.warning(f"adjacency figure skipped: {exc}")

        elog.add_metrics(**{k: v for k, v in rep.__dict__.items()
                            if isinstance(v, (int, float, bool, str))})
        elog.add_artifact(out / "dataset_report.json")

    log.info("=" * 60)
    for k, v in rep.__dict__.items():
        if k != "notes":
            log.info(f"  {k:28s}: {v}")
    for n in rep.notes:
        log.warning(f"  NOTE: {n}")
    log.info(f"reports -> {out/'dataset_report.json'} / .csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
