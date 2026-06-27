"""Sample weights from LOCAL GROUP CONSENSUS + out-of-fold residuals.

A trustworthy sample is one the model predicts well out-of-fold *and* whose error
agrees with its local neighborhood (groups of k consecutive, time-ordered rows).
A sample with a large out-of-fold residual relative to its neighbors is an
outlier / mislabeled / regime-break — it drags the fit toward memorization, so we
downweight it. The neighborhood comparison (the "group of 5" lens) makes this
robust to a globally hard but locally consistent stretch.
"""

from __future__ import annotations

import numpy as np


def _local_median(v, k):
    n = len(v)
    h = k // 2
    out = np.empty(n)
    for i in range(n):
        a, b = max(0, i - h), min(n, i + h + 1)
        out[i] = np.median(v[a:b])
    return out


def sample_weights(y, oof_pred, prev_sw, *, k_window=7, lr=0.5, floor=0.05) -> np.ndarray:
    """Update sample weights: downweight rows whose out-of-fold error is large
    relative to their local neighborhood."""
    y = np.asarray(y, float).ravel()
    r = np.abs(y - np.asarray(oof_pred, float).ravel())
    local = _local_median(r, k_window) + 1e-9
    rel = r / local                      # >1 = worse than its neighbors
    inv = 1.0 / (1.0 + rel)              # high when locally consistent
    rank = inv.argsort().argsort() / max(1, len(inv) - 1)
    target = floor + (1.0 - floor) * rank
    sw = (1 - lr) * np.asarray(prev_sw, float) + lr * target
    sw = np.clip(sw, floor, None)
    return sw / sw.mean()
