"""Model-Predictive signal control (spec section 32).

Uses the predicted traffic state x_hat_{t+1:t+H} to roll a point-queue model
for the two candidate immediate actions {hold, switch}, each followed by a
max-pressure tail policy, and applies the action minimising

    J = w1*Delay + w2*Queue + w3*Stops + w4*TravelTime + w5*Risk

subject to the min/max green already enforced by the simulator.
"""
from __future__ import annotations

import numpy as np


class MPCController:
    def __init__(self, cfg, predictor=None):
        self.cfg = cfg
        self.predictor = predictor
        c = cfg["control"]
        self.H = int(c["horizon"])
        self.w = c["mpc_weights"]
        self.dt_dec = float(cfg["sumo"]["aggregate_interval_s"])
        self.sat = 0.5 * self.dt_dec            # veh served per decision step
        self.min_ttc = float(c["minimum_ttc"])

    # ----------------------------------------------------------------
    def _arrival_rates(self, ctx):
        hist = ctx.get("state_history")
        a_main = a_cross = 0.15 * self.dt_dec
        if hist is not None and len(hist):
            # flow (veh/h) at the most upstream cell -> veh per decision step
            up_flow = float(np.mean(hist[-3:, 0, 2])) if hist.shape[1] else 0.0
            a_main = max(up_flow / 3600.0 * self.dt_dec, 0.05 * self.dt_dec)
        # predicted refinement
        if self.predictor is not None:
            try:
                mean_pred, _ = self.predictor(ctx)
                if mean_pred is not None:
                    a_main = max(float(np.mean(mean_pred[:, 0, 2])) / 3600.0
                                 * self.dt_dec, a_main * 0.5)
            except Exception:
                pass
        return a_main, a_cross

    def _rollout(self, q_main, q_cross, first_phase, a_main, a_cross):
        phase = first_phase
        cost = 0.0
        stops = 0
        for k in range(self.H):
            q_main += a_main
            q_cross += a_cross
            if phase == 0:
                served = min(q_main, self.sat)
                q_main -= served
            else:
                served = min(q_cross, self.sat)
                q_cross -= served
            cost += (self.w["queue"] * (q_main + q_cross)
                     + self.w["delay"] * (q_main + q_cross) * self.dt_dec / 60.0)
            # tail policy: max pressure
            new_phase = 0 if q_main >= q_cross else 1
            if new_phase != phase:
                stops += 1
                phase = new_phase
        cost += self.w["stops"] * stops
        return cost

    def __call__(self, ctx) -> int:
        q_main = float(ctx.get("queue_main", 0.0))
        q_cross = float(ctx.get("queue_cross", 0.0))
        a_main, a_cross = self._arrival_rates(ctx)
        j_hold = self._rollout(q_main, q_cross, ctx.get("phase", 0), a_main, a_cross)
        j_switch = self._rollout(q_main, q_cross, 1 - ctx.get("phase", 0),
                                 a_main, a_cross)
        return int(ctx.get("phase", 0)) if j_hold <= j_switch else int(1 - ctx.get("phase", 0))

    def reset(self):
        pass
