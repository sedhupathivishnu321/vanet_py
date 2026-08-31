"""VANET communication channel (spec sections 19-21).

* connected-vehicle penetration  -> which vehicles beacon at all (persistent)
* packet-delivery ratio (PDR)     -> M_received ~ Bernoulli(PDR) per beacon
* latency                         -> a beacon generated at t is *usable* only
                                     from t + latency (it arrives later in
                                     simulation time, it is not merely deleted)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Beacon:
    veh_id: int
    cell: int
    x_m: float
    speed_mps: float
    accel_mps2: float
    heading_deg: float
    gen_time_s: float
    arrival_time_s: float


class CommunicationChannel:
    def __init__(self, penetration: float, pdr: float, latency_ms: float,
                 seed: int = 0, jitter: bool = True):
        self.penetration = float(penetration)
        self.pdr = float(pdr)
        self.latency_s = float(latency_ms) / 1000.0
        # exponential jitter (mean = nominal latency) -> heavy tail, so a
        # fraction of beacons miss their update slot and inflate AoI. This is
        # what makes the latency sweep (RQ5) produce a measurable AoI trend
        # despite a 1 s aggregation bin.
        self.jitter = jitter
        self.rng = np.random.default_rng(seed)
        self._connected: dict[int, bool] = {}
        self.sent = 0
        self.delivered = 0

    def is_connected(self, veh_id: int) -> bool:
        vid = int(veh_id)
        if vid not in self._connected:
            self._connected[vid] = bool(self.rng.random() < self.penetration)
        return self._connected[vid]

    def transmit_frame(self, frame: np.ndarray) -> list[Beacon]:
        """`frame` rows = [veh_id, cell, x, speed, accel, heading, gen_time].
        Returns the beacons that were successfully delivered (with an arrival
        time = gen_time + latency)."""
        out: list[Beacon] = []
        for row in frame:
            vid = int(row[0])
            if not self.is_connected(vid):
                continue
            self.sent += 1
            if self.rng.random() > self.pdr:
                continue                         # packet lost
            self.delivered += 1
            gen = float(row[6])
            delay = self.latency_s
            if self.jitter and self.latency_s > 0:
                delay += float(self.rng.exponential(self.latency_s))
            out.append(Beacon(vid, int(row[1]), float(row[2]), float(row[3]),
                              float(row[4]), float(row[5]), gen, gen + delay))
        return out

    @property
    def realized_pdr(self) -> float:
        return self.delivered / self.sent if self.sent else 0.0
