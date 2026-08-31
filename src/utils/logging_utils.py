"""Console logging + per-experiment JSON provenance records (spec section 62)."""
from __future__ import annotations

import json
import logging
import platform
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str = "pvt", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S"))
        logger.addHandler(h)
    logger.setLevel(level)
    logger.propagate = False
    return logger


class ExperimentLogger:
    """Writes one JSON record per experiment run under outputs/logs/."""

    def __init__(self, logs_dir: str | Path, experiment: str, config: dict,
                 env: dict | None = None, seed: int | None = None):
        self.dir = Path(logs_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.experiment = experiment
        self.exp_id = f"{experiment}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"
        self._t0 = time.time()
        self.record: dict[str, Any] = {
            "experiment_id": self.exp_id,
            "experiment": experiment,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "seed": seed,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "config": config,
            "env": env or {},
            "metrics": {},
            "artifacts": [],
            "notes": [],
        }

    def add_metrics(self, **kwargs) -> None:
        self.record["metrics"].update(kwargs)

    def add_artifact(self, path: str | Path) -> None:
        self.record["artifacts"].append(str(path))

    def note(self, msg: str) -> None:
        self.record["notes"].append(msg)

    def close(self) -> Path:
        self.record["wall_time_s"] = round(time.time() - self._t0, 2)
        path = self.dir / f"{self.exp_id}.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.record, fh, indent=2, default=str)
        return path

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            self.note(f"FAILED: {exc_type.__name__}: {exc}")
        self.close()
        return False
