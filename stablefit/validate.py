"""Held-out + null validation for the learned weighting.

The loop optimizes cross-validated agreement on the *training* rows. The only
honest test of whether that bought generalization is a split the loop never saw.
We learn weights on the early rows, refit, and score the late rows — then compare
to (a) uniform weighting and (b) a magnitude-matched null (the learned sample and
feature weights permuted across rows / features). A win counts only if it beats
both. If it doesn't, the honest verdict is 'no gain' — not something to override.
"""

from __future__ import annotations

import numpy as np

from .core import pearson, weighted_ridge
from .loop import stablefit


def held_out(X, y, *, low_frac=0.6, n_null=200, seed=0, lam=1.0, **kw) -> dict:
    X = np.asarray(X, float)
    y = np.asarray(y, float).ravel()
    n = len(y)
    cut = max(2, int(n * low_frac))
    low, high = np.arange(cut), np.arange(cut, n)

    res = stablefit(X[low], y[low], lam=lam, **kw)
    sw, fw = res["sample_weights"], res["feature_weights"]

    learned = pearson(weighted_ridge(X[low], y[low], X[high], sw=sw, fw=fw, lam=lam), y[high])
    base = pearson(weighted_ridge(X[low], y[low], X[high], lam=lam), y[high])

    rng = np.random.default_rng(seed)
    nulls = np.empty(n_null)
    for i in range(n_null):
        swn = sw[rng.permutation(len(sw))]
        fwn = fw[rng.permutation(len(fw))]
        nulls[i] = pearson(
            weighted_ridge(X[low], y[low], X[high], sw=swn, fw=fwn, lam=lam), y[high])
    p95 = float(np.quantile(nulls, 0.95))

    return {
        "learned": learned, "uniform": base, "lift": learned - base,
        "null_p95": p95, "null_mean": float(nulls.mean()),
        "null_p": float((nulls >= learned).mean()),
        "beats_null": bool(learned > p95 and learned > base),
        "sample_weights": sw, "feature_weights": fw, "history": res["history"],
    }
