"""Shared CLI / bootstrap helpers for the pipeline scripts."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils import load_config, get_logger, collect_env_info, set_seed  # noqa: E402


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--quick", action="store_true",
                   help="reduced-scale profile (default for run_all.py --quick)")
    g.add_argument("--full", action="store_true", help="full experiment matrix")
    g.add_argument("--smoke", action="store_true",
                   help="minimal smoke profile (CI)")
    p.add_argument("--config", default=str(REPO_ROOT / "config.yaml"))
    p.add_argument("--seed", type=int, default=None)
    return p


def load(args):
    profile = "quick" if args.quick else "smoke" if args.smoke else None
    cfg = load_config(args.config, profile=profile)
    logger = get_logger("pvt")
    if args.seed is not None:
        cfg["project"]["seed_default"] = args.seed
    set_seed(int(cfg["project"]["seed_default"]))
    return cfg, logger


def env_info():
    return collect_env_info()
