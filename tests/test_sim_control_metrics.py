import numpy as np
import pytest

from src.sumo.idm_sim import IDMSimulator
from src.control import (MaxPressureController, FixedController,
                         ProposedRiskAwareController, ttc, count_ttc_violations)
from src.control.safety import summarize_ttc
from src.evaluation import regression_metrics, calibration_metrics


def test_idm_simulator_runs(cfg, tiny_corridor):
    out = IDMSimulator(cfg, tiny_corridor, "LOW", seed=1).run()
    assert out.state.ndim == 3 and out.state.shape[2] == 4
    assert out.n_cells == tiny_corridor["n_edges"]
    assert len(out.vehicle_frames) == out.state.shape[0]
    for k in ("travel_time_mean_s", "queue_mean_veh", "min_ttc_s",
              "throughput_veh_per_h"):
        assert k in out.metrics
    assert out.backend == "idm_fallback"


def test_idm_controller_changes_behaviour(cfg, tiny_corridor):
    base = IDMSimulator(cfg, tiny_corridor, "MEDIUM", seed=2).run()
    always_main = IDMSimulator(cfg, tiny_corridor, "MEDIUM", seed=2,
                               controller=lambda ctx: 0).run()
    # forcing main-green permanently should not increase the main queue vs fixed
    assert always_main.metrics["queue_mean_veh"] <= base.metrics["queue_mean_veh"] + 5


def test_max_pressure_decision():
    mp = MaxPressureController.__new__(MaxPressureController)
    assert mp({"queue_main": 10, "queue_cross": 3}) == 0
    assert mp({"queue_main": 1, "queue_cross": 9}) == 1


def test_fixed_controller_cycles(cfg):
    fc = FixedController(cfg)
    phases = {fc({"t": t}) for t in range(0, 180, 5)}
    assert phases == {0, 1}


def test_proposed_controller_outputs_valid_phase(cfg):
    pc = ProposedRiskAwareController(cfg, predictor=None)
    hist = np.random.rand(6, 6, 4).astype(np.float32)
    a = pc({"t": 60, "phase": 0, "phase_elapsed": 15, "queue_main": 5,
            "queue_cross": 2, "state_history": hist, "tl_cell": 3,
            "mean_aoi": 40.0})
    assert a in (0, 1)


def test_ttc_formula():
    assert ttc(10.0, 20.0, 10.0) == pytest.approx(1.0)
    assert np.isinf(ttc(10.0, 5.0, 20.0))          # not closing
    assert count_ttc_violations([1.0, 3.0, 0.5], 2.0) == 2


def test_regression_metrics_known():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    p = y + 1.0
    m = regression_metrics(p, y)
    assert m["MAE"] == pytest.approx(1.0)
    assert m["RMSE"] == pytest.approx(1.0)


def test_calibration_metrics_runs():
    rng = np.random.default_rng(0)
    y = rng.normal(size=500)
    mean = y + rng.normal(scale=0.5, size=500)
    std = np.full(500, 0.5)
    c = calibration_metrics(mean, std, y)
    assert 0.0 <= c["PICP_95"] <= 1.0
    assert len(c["calibration_curve"]) == 4
