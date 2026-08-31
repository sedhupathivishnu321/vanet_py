import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.utils import load_config, set_seed  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    c = load_config(profile="smoke")
    set_seed(0)
    return c


@pytest.fixture
def tiny_corridor():
    n_seg = 6
    coords = [[79.80 + 0.002 * i, 11.93 + 0.001 * i] for i in range(n_seg + 1)]
    return {"id": "corridor_test", "name": "test corridor", "length_m": 1500.0,
            "n_edges": n_seg, "n_segments": n_seg, "edges": [], "nodes": [],
            "route_coords": coords, "route_nodes": [],
            "origin_latlon": (11.93, 79.80), "destination_latlon": (11.936, 79.812)}


@pytest.fixture
def synth_series():
    rng = np.random.default_rng(0)
    T, N, C = 240, 8, 8
    x = np.zeros((T, N, C), np.float32)
    t = np.arange(T)
    x[..., 0] = (50 + 10 * np.sin(2 * np.pi * t / 48))[:, None] + rng.normal(0, 1, (T, N))
    x[..., 4] = 1.0
    x[..., 6] = 1.0
    return x
