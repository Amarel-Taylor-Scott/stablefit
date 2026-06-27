"""The feedback loop: iterate sample-weight and feature-weight updates, driven
by the train↔CV gap rather than raw fit.

Start every sample and every feature at weight 1. Each iteration:
  1. fit per-fold coefficients → downweight unstable (overfit-prone) features,
  2. recompute out-of-fold predictions → downweight locally inconsistent samples.
The objective is the **cross-validated** correlation (and the shrinking gap to the
in-sample number) — so the loop actively trades in-sample fit for agreement
between train and held-out folds. Everything it does is inspectable: the returned
weights say which features were trusted and which samples were down-weighted.
"""

from __future__ import annotations

import numpy as np

from .consensus import sample_weights
from .core import fit_coef, pearson, predict_coef
from .stability import feature_weights, fold_coefs, oof_predict


def _cv(X, y, sw, fw, k, lam, purge):
    oof = oof_predict(X, y, sw=sw, fw=fw, k=k, lam=lam, purge=purge)
    beta, mu, yb = fit_coef(X, y, sw=sw, fw=fw, lam=lam)
    ins = pearson(predict_coef(X, beta, mu, yb, fw), y)
    return ins, pearson(oof, y), oof


def stablefit(X, y, *, k=4, lam=1.0, iters=8, lr=0.5, k_window=7, floor=0.05,
              purge=0) -> dict:
    """Jointly learn sample + feature weights that maximize CV agreement.

    Returns ``sample_weights`` (n,), ``feature_weights`` (p,), the ``history`` of
    (insample, oof, gap) per iteration, and the best ``oof``.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float).ravel()
    n, p = X.shape
    sw = np.ones(n)
    fw = np.ones(p)

    ins, oof_c, _ = _cv(X, y, sw, fw, k, lam, purge)
    history = [{"iter": 0, "insample": ins, "oof": oof_c, "gap": ins - oof_c}]
    best = (oof_c, sw.copy(), fw.copy())

    for it in range(1, iters + 1):
        coefs = fold_coefs(X, y, sw=sw, fw=fw, k=k, lam=lam, purge=purge)
        fw = feature_weights(coefs, fw, lr=lr, floor=floor)
        oof = oof_predict(X, y, sw=sw, fw=fw, k=k, lam=lam, purge=purge)
        sw = sample_weights(y, oof, sw, k_window=k_window, lr=lr, floor=floor)
        ins, oof_c, _ = _cv(X, y, sw, fw, k, lam, purge)
        history.append({"iter": it, "insample": ins, "oof": oof_c, "gap": ins - oof_c})
        if oof_c > best[0]:
            best = (oof_c, sw.copy(), fw.copy())

    oof_c, sw, fw = best
    return {"sample_weights": sw, "feature_weights": fw,
            "history": history, "oof": oof_c}
