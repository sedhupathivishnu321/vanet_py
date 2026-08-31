"""Fixed-time and Max-Pressure signal controllers (spec sections 30-31).

Controller call signature:  ``controller(ctx) -> int``  where the return is the
desired phase (0 = corridor/main green, 1 = cross-street green).  ``ctx`` is the
dict assembled by the simulator (queues, phase, recent state history, ...).
"""
from __future__ import annotations


class FixedController:
    """Reproduces the default fixed cycle explicitly so it can be benchmarked."""

    def __init__(self, cfg):
        tl = cfg["sumo"]["traffic_light"]
        self.cycle = float(tl["cycle_s"])
        self.split = 0.6

    def __call__(self, ctx) -> int:
        return 0 if (ctx["t"] % self.cycle) < self.split * self.cycle else 1

    reset = lambda self: None


class MaxPressureController:
    """Serve the movement with the greatest pressure.

        P(phase) = sum(Q_incoming) - sum(Q_outgoing)

    For the corridor+cross intersection this reduces to comparing the main and
    cross approach queues (downstream assumed uncongested)."""

    def __init__(self, cfg):
        self.min_switch_gap = 1.0

    def __call__(self, ctx) -> int:
        p_main = ctx.get("queue_main", 0.0)
        p_cross = ctx.get("queue_cross", 0.0)
        return 0 if p_main >= p_cross else 1

    def reset(self):
        pass
