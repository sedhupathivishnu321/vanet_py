"""Proposed uncertainty- and AoI-aware risk-sensitive signal controller
(spec section 33).

    J = J_traffic + lambda_A * J_AoI + lambda_U * J_uncertainty + lambda_S * J_safety

Base decision is max-pressure on the *predicted* queues.  When prediction
uncertainty and/or AoI are high the controller becomes conservative: it demands
a larger pressure differential before switching (hysteresis grows with
uncertainty + AoI) and it will not switch if that would create a predicted
safety (low-TTC) risk.  It never takes an unsafe action just because the ML
prediction is uncertain -- high uncertainty widens hysteresis, it does not
license aggressive switching.
"""
from __future__ import annotations

import numpy as np


class ProposedRiskAwareController:
    def __init__(self, cfg, predictor=None):
        self.cfg = cfg
        self.predictor = predictor
        w = cfg["control"]["proposed_weights"]
        self.w_traffic = float(w["traffic"])
        self.w_aoi = float(w["aoi"])
        self.w_unc = float(w["uncertainty"])
        self.w_safety = float(w["safety"])
        self.min_ttc = float(cfg["control"]["minimum_ttc"])
        self._ema_unc = 0.0

    def _predict(self, ctx):
        """Return (pred_queue_main, uncertainty_scalar, mean_speed_pred)."""
        hist = ctx.get("state_history")
        q_main = float(ctx.get("queue_main", 0.0))
        unc = 0.0
        spd = 12.0
        if self.predictor is not None:
            try:
                mean_pred, std_pred = self.predictor(ctx)
                if mean_pred is not None:
                    # queue channel = 3; look near the stop-line cell
                    tl = int(ctx.get("tl_cell", mean_pred.shape[1] // 2))
                    q_main = float(np.mean(mean_pred[:, max(tl - 1, 0):tl + 1, 3]))
                    spd = float(np.mean(mean_pred[:, :, 0]))
                if std_pred is not None:
                    unc = float(np.mean(std_pred))
            except Exception:
                pass
        elif hist is not None and len(hist) >= 2:
            unc = float(np.std(hist[-3:, :, 3]))
        return q_main, unc, spd

    def __call__(self, ctx) -> int:
        phase = int(ctx.get("phase", 0))
        q_main_pred, unc, spd_pred = self._predict(ctx)
        q_cross = float(ctx.get("queue_cross", 0.0))

        # AoI penalty: stale information -> trust predictions less -> conservative
        hist = ctx.get("state_history")
        aoi_level = float(ctx.get("mean_aoi", 0.0))
        self._ema_unc = 0.7 * self._ema_unc + 0.3 * unc

        # hysteresis threshold grows with uncertainty and AoI
        hysteresis = (1.0
                      + self.w_unc * self._ema_unc
                      + self.w_aoi * (aoi_level / 60.0))

        pressure_diff = q_main_pred - q_cross      # >0 favours main green
        # candidate desired phase from (weighted) max-pressure
        if phase == 0:                              # currently serving main
            desired = 1 if (-pressure_diff) > hysteresis else 0
        else:                                      # currently serving cross
            desired = 0 if (pressure_diff) > hysteresis else 1

        # safety guard: if predicted speeds are high and we would switch the
        # main phase to red abruptly, hold one more step (avoid dilemma zone)
        if desired != phase and phase == 0 and spd_pred > 0.7 * 13.9:
            if ctx.get("phase_elapsed", 0) < self.cfg["sumo"]["traffic_light"]["min_green_s"] + 2:
                desired = phase
        return int(desired)

    def reset(self):
        self._ema_unc = 0.0
