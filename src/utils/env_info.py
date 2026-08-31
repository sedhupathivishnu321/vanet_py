"""Collect hardware / software / git provenance for reproducibility."""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone


def _git(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except Exception:
        return None


def _cpu_count() -> int:
    try:
        return os.cpu_count() or 1
    except Exception:
        return 1


def _mem_gb() -> float | None:
    try:
        if hasattr(os, "sysconf"):
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return round(pages * page_size / 1e9, 2)
    except Exception:
        pass
    return None


def collect_env_info() -> dict:
    info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": _cpu_count(),
        "memory_gb": _mem_gb(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "in_codespace": bool(os.environ.get("CODESPACES")),
        "sumo_home": os.environ.get("SUMO_HOME"),
        "sumo_binary": shutil.which("sumo"),
    }
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_device"] = (torch.cuda.get_device_name(0)
                               if torch.cuda.is_available() else None)
        info["torch_num_threads"] = torch.get_num_threads()
    except Exception as exc:  # pragma: no cover
        info["torch_version"] = None
        info["torch_import_error"] = str(exc)
    try:
        import numpy
        info["numpy_version"] = numpy.__version__
    except Exception:
        pass
    return info
