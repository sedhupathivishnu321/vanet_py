"""Graph construction utilities (spec section 9).

For a set of node coordinates we build a thresholded Gaussian-kernel adjacency

    A_ij = exp(-d_ij^2 / (2 sigma^2))   for d_ij below a distance cutoff,

and provide the symmetric normalisation  D^-1/2 (A+I) D^-1/2  used by GCN.
"""
from __future__ import annotations

import numpy as np


def _haversine_matrix(coords: np.ndarray) -> np.ndarray:
    """Great-circle distance (metres) between rows of `coords` = [lat, lon]."""
    lat = np.radians(coords[:, 0])[:, None]
    lon = np.radians(coords[:, 1])[:, None]
    dlat = lat - lat.T
    dlon = lon - lon.T
    a = np.sin(dlat / 2) ** 2 + np.cos(lat) * np.cos(lat.T) * np.sin(dlon / 2) ** 2
    return 6_371_000.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def gaussian_kernel_adjacency(coords: np.ndarray,
                              sigma: float | None = None,
                              threshold: float = 0.1,
                              geographic: bool = True) -> np.ndarray:
    """coords: (N, 2). Returns weighted, symmetric (N, N) adjacency with zero
    diagonal.  `threshold` sparsifies entries with weight < threshold."""
    if geographic:
        d = _haversine_matrix(coords)
    else:
        d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    finite = d[np.isfinite(d) & (d > 0)]
    if sigma is None:
        sigma = float(np.std(finite)) if finite.size else 1.0
    sigma = max(sigma, 1e-6)
    A = np.exp(-(d ** 2) / (2 * sigma ** 2))
    A[A < threshold] = 0.0
    np.fill_diagonal(A, 0.0)
    A = np.maximum(A, A.T)
    return A.astype(np.float32)


def knn_adjacency(coords: np.ndarray, k: int = 5,
                  geographic: bool = True) -> np.ndarray:
    d = (_haversine_matrix(coords) if geographic
         else np.linalg.norm(coords[:, None] - coords[None], axis=-1))
    N = d.shape[0]
    A = np.zeros((N, N), np.float32)
    k = min(k, N - 1)
    for i in range(N):
        nn = np.argsort(d[i])[1:k + 1]
        A[i, nn] = 1.0
    return np.maximum(A, A.T)


def add_self_loops(A: np.ndarray) -> np.ndarray:
    A = A.copy()
    np.fill_diagonal(A, 1.0)
    return A


def normalize_adjacency(A: np.ndarray, add_self: bool = True) -> np.ndarray:
    """Symmetric normalisation for GCN:  D^-1/2 (A+I) D^-1/2."""
    A = add_self_loops(A) if add_self else A.copy()
    deg = A.sum(axis=1)
    deg[deg == 0] = 1e-12
    dinv_sqrt = 1.0 / np.sqrt(deg)
    return (dinv_sqrt[:, None] * A * dinv_sqrt[None, :]).astype(np.float32)
