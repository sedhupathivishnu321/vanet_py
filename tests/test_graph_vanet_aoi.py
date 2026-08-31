import numpy as np

from src.preprocessing.graph import (gaussian_kernel_adjacency,
                                     normalize_adjacency, knn_adjacency)
from src.vanet import CommunicationChannel, AoITracker
from src.vanet.aoi import aoi_statistics


def test_gaussian_adjacency_symmetric_zero_diag():
    coords = np.random.default_rng(0).random((20, 2)) * 0.1 + [11.9, 79.8]
    A = gaussian_kernel_adjacency(coords, threshold=0.05)
    assert np.allclose(A, A.T)
    assert np.allclose(np.diag(A), 0)
    assert (A >= 0).all()


def test_normalize_adjacency_spectral_bound():
    A = knn_adjacency(np.random.default_rng(1).random((15, 2)), k=3)
    An = normalize_adjacency(A)
    # symmetric-normalised adjacency has eigenvalues in [-1, 1]
    ev = np.linalg.eigvalsh(An)
    assert ev.max() <= 1.0 + 1e-5
    assert ev.min() >= -1.0 - 1e-5


def test_channel_pdr_and_latency():
    ch = CommunicationChannel(penetration=1.0, pdr=0.8, latency_ms=100, seed=0,
                              jitter=False)
    frame = np.array([[i, i % 4, i * 10.0, 12.0, 0.0, 90.0, 5.0]
                      for i in range(4000)])
    got = ch.transmit_frame(frame)
    assert 0.75 < len(got) / 4000 < 0.85            # ~ PDR
    assert all(abs(b.arrival_time_s - (b.gen_time_s + 0.1)) < 1e-6 for b in got)


def test_channel_latency_jitter_monotone_mean():
    frame = np.array([[i, 0, 0.0, 12.0, 0.0, 90.0, 0.0] for i in range(3000)])
    lo = CommunicationChannel(1.0, 1.0, 20, seed=1).transmit_frame(frame)
    hi = CommunicationChannel(1.0, 1.0, 500, seed=1).transmit_frame(frame)
    assert np.mean([b.arrival_time_s for b in hi]) > \
           np.mean([b.arrival_time_s for b in lo])


def test_channel_penetration_persistent():
    ch = CommunicationChannel(0.5, 1.0, 0, seed=3)
    first = {i: ch.is_connected(i) for i in range(200)}
    second = {i: ch.is_connected(i) for i in range(200)}
    assert first == second
    frac = np.mean(list(first.values()))
    assert 0.4 < frac < 0.6


def test_aoi_tracker_decreases_on_fresh_beacon():
    from src.vanet.channel import Beacon
    tr = AoITracker(n_cells=3, cap_s=100)
    assert tr.aoi(now=10.0)[0] == 100          # never received -> cap
    tr.update([Beacon(1, 0, 0, 10, 0, 0, gen_time_s=8.0, arrival_time_s=8.5)], now=10.0)
    assert abs(tr.aoi(now=10.0)[0] - 2.0) < 1e-6
    assert tr.aoi(now=20.0)[0] == 12.0         # ages with time


def test_aoi_statistics():
    s = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    st = aoi_statistics(s)
    assert st["max"] == 6.0 and st["mean"] == 3.5
