#!/usr/bin/env python
"""STEP 5: identify / download the source traffic dataset (spec section 4).

Priority:
  1. If a dataset already exists under data/source/  -> use it (no download).
  2. Otherwise download METR-LA (or PEMS-BAY) from the public mirror configured
     in config.yaml  (https://graphmining.ai/temporal_datasets/, the mirror used
     by PyTorch-Geometric-Temporal).
  3. If offline and source_dataset.allow_synthetic_fallback is true, a SYNTHETIC
     dataset is generated *for smoke testing only* and everything is stamped so.

Never fabricates or scrapes undocumented data.
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen, Request

from _common import base_parser, load


def _already_present(root: Path) -> bool:
    patterns = ["*.h5", "*/node_values.npy", "node_values.npy", "*.csv"]
    return any(list(root.glob(p)) for p in patterns)


def _download(url: str, dest_dir: Path, log) -> bool:
    try:
        log.info(f"  downloading {url}")
        req = Request(url, headers={"User-Agent": "puducherry-vanet-transfer/0.1"})
        with urlopen(req, timeout=120) as resp:
            blob = resp.read()
        if url.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(blob)) as zf:
                zf.extractall(dest_dir)
            log.info(f"  extracted {len(blob)/1e6:.1f} MB -> {dest_dir}")
        else:
            (dest_dir / Path(url).name).write_bytes(blob)
        return True
    except Exception as exc:  # pragma: no cover - network dependent
        log.warning(f"  download failed: {exc}")
        return False


def main() -> int:
    args = base_parser(__doc__).parse_args()
    cfg, log = load(args)
    name = cfg["source_dataset"]["name"]
    root = Path(cfg["_meta"]["repo_root"]) / cfg["source_dataset"]["root"]
    root.mkdir(parents=True, exist_ok=True)
    sub = root / name
    sub.mkdir(exist_ok=True)

    if _already_present(root):
        log.info(f"source dataset already present under {root} -- using it.")
        return _finish(cfg, log)

    ok = False
    for url in cfg["source_dataset"]["mirrors"].get(name, []):
        ok = _download(url, sub, log)
        if ok:
            break

    if not ok:
        # optional Google-Drive style fallback via gdown, if configured
        gd = cfg["source_dataset"].get("gdrive_ids", {}).get(name)
        if gd:
            try:
                import gdown
                zpath = sub / f"{name}.zip"
                gdown.download(id=gd, output=str(zpath), quiet=False)
                if zpath.exists():
                    with zipfile.ZipFile(zpath) as zf:
                        zf.extractall(sub)
            except Exception as exc:
                log.warning(f"  gdown fallback failed: {exc}")

    if not _already_present(root):
        if cfg["source_dataset"].get("allow_synthetic_fallback", False):
            log.warning("could not obtain the real dataset; a SYNTHETIC dataset "
                        "will be generated at load time (smoke test only).")
            return 0
        log.error("Failed to obtain the source dataset and synthetic fallback "
                  "is disabled. Download METR-LA manually into data/source/ "
                  "(see README section 'Dataset').")
        return 1
    return _finish(cfg, log)


def _finish(cfg, log) -> int:
    from src.data import load_source_dataset
    ds = load_source_dataset(cfg)
    log.info(f"loaded '{ds.name}'  data shape={ds.data.shape}  "
             f"adj={'yes' if ds.adj is not None else 'no'}  "
             f"coords={'yes' if ds.coords is not None else 'no'}  "
             f"synthetic={ds.is_synthetic}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
