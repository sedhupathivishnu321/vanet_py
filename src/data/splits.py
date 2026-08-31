"""Chronological (leakage-free) train / val / test splitting -- spec section 44.

We split the *time axis* into three contiguous blocks.  Sliding windows are
then built inside each block only, so no window ever straddles a boundary and
no future timestamp is used to predict the past.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SplitIndices:
    train: tuple[int, int]   # [start, end) on the time axis
    val: tuple[int, int]
    test: tuple[int, int]
    fractions: tuple[float, float, float]

    def as_dict(self) -> dict:
        return {"train_range": list(self.train), "val_range": list(self.val),
                "test_range": list(self.test), "fractions": list(self.fractions)}


def chronological_split(num_timestamps: int,
                        fractions=(0.7, 0.15, 0.15)) -> SplitIndices:
    f_tr, f_va, f_te = fractions
    assert abs(f_tr + f_va + f_te - 1.0) < 1e-6, "fractions must sum to 1"
    n = int(num_timestamps)
    n_tr = int(n * f_tr)
    n_va = int(n * f_va)
    return SplitIndices(
        train=(0, n_tr),
        val=(n_tr, n_tr + n_va),
        test=(n_tr + n_va, n),
        fractions=(f_tr, f_va, f_te),
    )
