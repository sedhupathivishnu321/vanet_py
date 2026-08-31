#!/usr/bin/env python
"""STEP 5: identify / download the source traffic dataset (spec section 4).

Priority:
  1. If the dataset already exists under data/source/<NAME>/  -> use it, no download.
  2. Otherwise fetch each file listed in config.yaml -> source_dataset.download
     (public, no-auth GitHub raw endpoints; first working URL per file).
       METR-LA  : metr-la.h5 + adj_mx.pkl + graph_sensor_locations.csv
       PEMS-BAY : pems-bay.zip (-> pems-bay.h5) + adj_mx_bay.pkl + locations csv
  3. Last resort: the single-zip mirror in source_dataset.mirrors.
  4. If everything fails and allow_synthetic_fallback is true, a SYNTHETIC
     dataset is used later (smoke test only, stamped SYNTHETIC).

Never fabricates or scrapes undocumented data. Prints SHA-256 of every file.
"""
from __future__ import annotations

import hashlib
import io
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen, Request

from _common import base_parser, load

_UA = {"User-Agent": "puducherry-vanet-transfer/0.1 (+https://openstreetmap.org)"}


def _present(root: Path) -> bool:
    """True if a usable time-series file exists anywhere under `root`."""
    if list(root.rglob("*.h5")) or list(root.rglob("node_values.npy")):
        return True
    aux = ("sensor_locations", "locations", "sensor_ids", "distances", "adj",
           "_ids", "W_", "SE_")
    for c in root.rglob("*.csv"):
        if not any(a in c.name.lower() for a in (s.lower() for s in aux)):
            return True
    return False


def _verify_checksums(root: Path, log) -> None:
    manifest = root.parents[1] / "DATASET_CHECKSUMS.txt"
    if not manifest.exists():
        return
    want = {}
    for ln in manifest.read_text().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split()
        if len(parts) >= 2:
            want[parts[1]] = parts[0]
    for rel, sha in want.items():
        f = root / rel
        if f.exists():
            got = hashlib.sha256(f.read_bytes()).hexdigest()
            tag = "ok" if got == sha else "MISMATCH"
            (log.info if tag == "ok" else log.warning)(f"    checksum {tag}: {rel}")


def _fetch(url: str, dest: Path, log) -> bool:
    try:
        log.info(f"    GET {url}")
        with urlopen(Request(url, headers=_UA), timeout=300) as resp:
            blob = resp.read()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(blob)
        log.info(f"        -> {dest.name}  {len(blob):,} bytes  "
                 f"sha256={hashlib.sha256(blob).hexdigest()[:16]}")
        return True
    except Exception as exc:  # pragma: no cover - network dependent
        log.warning(f"        failed: {exc}")
        return False


def _download_files(specs, dest_dir: Path, log) -> bool:
    got_any = False
    for spec in specs:
        fname, urls = spec[0], list(spec[1:])
        target = dest_dir / fname
        if target.exists() and target.stat().st_size > 0:
            log.info(f"    have {fname}")
            got_any = True
            continue
        for u in urls:
            if _fetch(u, target, log):
                got_any = True
                if fname.lower().endswith(".zip"):
                    try:
                        with zipfile.ZipFile(target) as zf:
                            zf.extractall(dest_dir)
                        log.info(f"        extracted {fname}")
                        target.unlink()
                    except Exception as exc:
                        log.warning(f"        unzip failed: {exc}")
                break
    return got_any


def _download_zip_mirror(urls, dest_dir: Path, log) -> bool:
    for u in urls:
        try:
            log.info(f"    GET {u}")
            with urlopen(Request(u, headers=_UA), timeout=300) as resp:
                blob = resp.read()
            with zipfile.ZipFile(io.BytesIO(blob)) as zf:
                zf.extractall(dest_dir)
            log.info(f"        extracted {len(blob):,} bytes -> {dest_dir}")
            return True
        except Exception as exc:  # pragma: no cover
            log.warning(f"        failed: {exc}")
    return False


def main() -> int:
    args = base_parser(__doc__).parse_args()
    cfg, log = load(args)
    sd = cfg["source_dataset"]
    name = sd["name"]
    root = Path(cfg["_meta"]["repo_root"]) / sd["root"]
    dest_dir = root / name
    dest_dir.mkdir(parents=True, exist_ok=True)

    if _present(root):
        log.info(f"source dataset already present under {root} -- using it.")
        _verify_checksums(root, log)
        return _finish(cfg, log)

    log.info(f"downloading '{name}' into {dest_dir}")
    specs = sd.get("download", {}).get(name, [])
    ok = _download_files(specs, dest_dir, log) if specs else False

    if not _present(root):
        log.warning("per-file download incomplete; trying single-zip mirror ...")
        _download_zip_mirror(sd.get("mirrors", {}).get(name, []), dest_dir, log)

    if not _present(root):
        gd = sd.get("gdrive_ids", {}).get(name)
        if gd:
            try:
                import gdown
                zp = dest_dir / f"{name}.zip"
                gdown.download(id=gd, output=str(zp), quiet=False)
                if zp.exists():
                    with zipfile.ZipFile(zp) as zf:
                        zf.extractall(dest_dir)
            except Exception as exc:
                log.warning(f"gdown fallback failed: {exc}")

    if not _present(root):
        if sd.get("allow_synthetic_fallback", False):
            log.warning("could not obtain the real dataset; a SYNTHETIC dataset "
                        "will be generated at load time (smoke test only). "
                        "Download manually -- see README section 'Dataset'.")
            return 0
        log.error("Failed to obtain the source dataset and synthetic fallback "
                  "is disabled.")
        return 1
    _verify_checksums(root, log)
    return _finish(cfg, log)


def _finish(cfg, log) -> int:
    from src.data import load_source_dataset
    ds = load_source_dataset(cfg)
    log.info("-" * 60)
    log.info(f"loaded '{ds.name}'  shape={ds.data.shape} (T, N, C)  "
             f"layout={ds.layout}")
    log.info(f"  adjacency : {'yes' if ds.adj is not None else 'no'}"
             + (f" {ds.adj.shape}" if ds.adj is not None else ""))
    log.info(f"  coords    : {'yes' if ds.coords is not None else 'no'}"
             + (f" {ds.coords.shape}" if ds.coords is not None else ""))
    log.info(f"  synthetic : {ds.is_synthetic}")
    if ds.is_synthetic:
        log.warning("  (this is the SYNTHETIC fallback -- results are not "
                    "scientifically valid; provide the real dataset)")
    log.info("next: python scripts/inspect_dataset.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
