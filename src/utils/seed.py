"""Deterministic seeding across python / numpy / torch."""
from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int, deterministic_torch: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def seed_worker(worker_id: int) -> None:  # for DataLoader workers
    worker_seed = (int(os.environ.get("PYTHONHASHSEED", "0")) + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
