#!/usr/bin/env python
"""STEP 1-2: inspect the Codespace -- Python, CUDA/GPU, disk, SUMO, deps."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from _common import REPO_ROOT, env_info, get_logger


def main() -> int:
    log = get_logger("env")
    info = env_info()
    total, used, free = shutil.disk_usage(REPO_ROOT)
    info["disk_free_gb"] = round(free / 1e9, 2)
    info["disk_total_gb"] = round(total / 1e9, 2)

    deps = {}
    for mod in ["numpy", "pandas", "scipy", "sklearn", "torch", "osmnx",
                "networkx", "geopandas", "folium", "matplotlib", "seaborn",
                "streamlit", "yaml", "h5py"]:
        try:
            m = __import__(mod)
            deps[mod] = getattr(m, "__version__", "ok")
        except Exception as exc:
            deps[mod] = f"MISSING ({exc.__class__.__name__})"
    info["dependencies"] = deps

    try:
        from src.sumo.build_network import sumo_available, sumo_version
        info["sumo_available"] = sumo_available()
        info["sumo_version"] = sumo_version()
    except Exception as exc:
        info["sumo_check_error"] = str(exc)

    out = REPO_ROOT / "outputs" / "logs" / "env_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, indent=2, default=str))

    log.info("Environment report")
    for k in ["python_version", "platform", "cpu_count", "memory_gb",
              "disk_free_gb", "torch_version", "cuda_available",
              "sumo_available", "sumo_version", "in_codespace", "git_commit"]:
        log.info(f"  {k:16s}: {info.get(k)}")
    missing = [k for k, v in deps.items()
               if isinstance(v, str) and v.startswith("MISSING")]
    if missing:
        log.warning(f"missing python deps: {missing} "
                    f"(run: pip install -r requirements.txt)")
    if not info.get("cuda_available"):
        log.info("no GPU -> pipeline will run in CPU mode "
                 "(use --quick for reduced-scale experiments)")
    log.info(f"full report -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
