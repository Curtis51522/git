"""Utilities for keeping S2 quantile forecasts coherent."""

from __future__ import annotations

import numpy as np


def enforce_quantile_monotonicity(q10, q50, q90) -> dict[str, np.ndarray | int]:
    """Return monotonically ordered Q10/Q50/Q90 arrays.

    Independent quantile models can cross because each quantile is fitted as a
    separate estimator. Quantile rearrangement fixes that output-level issue by
    sorting the three predicted values per sample.
    """
    q10_arr = np.asarray(q10, dtype=float)
    q50_arr = np.asarray(q50, dtype=float)
    q90_arr = np.asarray(q90, dtype=float)
    if q10_arr.shape != q50_arr.shape or q50_arr.shape != q90_arr.shape:
        raise ValueError("Quantile arrays must have matching shapes")

    crossing_mask = (q10_arr > q50_arr) | (q50_arr > q90_arr)
    ordered = np.sort(np.vstack([q10_arr, q50_arr, q90_arr]), axis=0)
    return {
        "q10": ordered[0],
        "q50": ordered[1],
        "q90": ordered[2],
        "crossing_mask": crossing_mask,
        "crossing_count": int(crossing_mask.sum()),
    }
