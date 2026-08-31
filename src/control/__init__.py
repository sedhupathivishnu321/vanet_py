from .safety import ttc, ttc_from_frame, count_ttc_violations
from .max_pressure import MaxPressureController, FixedController
from .mpc import MPCController
from .rl import DQNController, train_dqn
from .proposed_controller import ProposedRiskAwareController

CONTROLLERS = ["fixed", "max_pressure", "mpc", "dqn", "proposed"]

__all__ = [
    "ttc", "ttc_from_frame", "count_ttc_violations",
    "MaxPressureController", "FixedController", "MPCController",
    "DQNController", "train_dqn", "ProposedRiskAwareController",
    "CONTROLLERS", "build_controller",
]


def build_controller(name: str, cfg, predictor=None, **kw):
    name = name.lower()
    if name == "fixed":
        return FixedController(cfg)
    if name == "max_pressure":
        return MaxPressureController(cfg)
    if name == "mpc":
        return MPCController(cfg, predictor=predictor, **kw)
    if name == "dqn":
        return DQNController(cfg, **kw)
    if name == "proposed":
        return ProposedRiskAwareController(cfg, predictor=predictor, **kw)
    raise KeyError(name)
