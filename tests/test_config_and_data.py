import numpy as np

from src.utils import load_config
from src.data import chronological_split, make_windows, make_windows_xy
from src.data.source_dataset import _synthetic


def test_profiles_merge():
    base = load_config()
    quick = load_config(profile="quick")
    smoke = load_config(profile="smoke")
    assert quick["training"]["epochs"] < base["training"]["epochs"]
    assert smoke["training"]["epochs"] <= quick["training"]["epochs"]
    assert "quick" not in quick  # profile block stripped


def test_chronological_split_no_overlap():
    sp = chronological_split(1000, (0.7, 0.15, 0.15))
    assert sp.train[1] == sp.val[0]
    assert sp.val[1] == sp.test[0]
    assert sp.test[1] == 1000
    # windows built per block never cross the boundary
    series = np.zeros((1000, 4, 2), np.float32)
    wtr = make_windows(series, 12, 12, *sp.train)
    assert wtr.t_index.max() < sp.train[1]


def test_window_shapes():
    series = np.random.rand(200, 5, 8).astype(np.float32)
    w = make_windows(series, 12, 6, 0, 200)
    assert w.X.shape[1:] == (12, 5, 8)
    assert w.Y.shape[1:] == (6, 5, 1)
    assert len(w) == 200 - 12 - 6 + 1


def test_make_windows_xy_aoi_lastobs():
    x = np.random.rand(100, 4, 8).astype(np.float32)
    y = np.random.rand(100, 4).astype(np.float32)
    aoi = np.random.rand(100, 4).astype(np.float32)
    d = make_windows_xy(x, y, 10, 5, aoi_series=aoi, obs_speed_series=x[..., 0])
    assert d["X"].shape[1:] == (10, 4, 8)
    assert d["Y"].shape[1:] == (5, 4, 1)
    assert d["aoi"].shape == (len(d["X"]), 4)
    assert d["last_obs"].shape == (len(d["X"]), 4)


def test_synthetic_source_report():
    ds = _synthetic("METR-LA", num_nodes=12, days=3)
    rep = ds.build_report()
    assert rep.is_synthetic is True
    assert rep.num_nodes == 12
    assert rep.adjacency_available is True
    assert rep.value_std > 0
