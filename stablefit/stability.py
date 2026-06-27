"""Feature weights from CROSS-FOLD COEFFICIENT STABILITY.

A feature whose fitted coefficient is large and consistent across time-folds is
carrying a real, transferable relationship; one whose coefficient swings sign or
magnitude fold-to-fold is overfit-prone — the model is using it to memorize a
fold, not to generalize. So we score each feature by a t-statistic-like stability
``|mean(coef)| / (std(coef) + eps)`` across folds and downweight the unstable
ones. This is the explainable anti-overfitting lever: you can read off exactly
which features were trusted and which were suppressed.
"""

from __future__ import annotations

import numpy as np

from .core import fit_coef, predict_coef, time_folds


def fold_coefs(X, y, sw=None, fw=None, k=4, lam=1.0, purge=0):
    """Coefficient vector from each fold's training portion → (K, p)."""
    betas = []
    for tr, _te in time_folds(len(y), k, purge):
        sw_tr = None if sw is None else np.asarray(sw)[tr]
        beta, _mu, _yb = fit_coef(X[tr], np.asarray(y)[tr], sw=sw_tr, fw=fw, lam=lam)
        betas.append(beta)
    return np.array(betas)


def oof_predict(X, y, sw=None, fw=None, k=4, lam=1.0, purge=0):
    """Out-of-fold predictions over all rows (each row predicted by a model that
    did not train on its fold)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float).ravel()
    oof = np.full(len(y), np.nan)
    for tr, te in time_folds(len(y), k, purge):
        sw_tr = None if sw is None else np.asarray(sw)[tr]
        beta, mu, yb = fit_coef(X[tr], y[tr], sw=sw_tr, fw=fw, lam=lam)
        oof[te] = predict_coef(X[te], beta, mu, yb, np.ones(X.shape[1]) if fw is None else fw)
    return oof


def stability_scores(coefs) -> np.ndarray:
    """Per-feature ``|mean| / (std + eps)`` across folds (high = stable)."""
    mean = np.abs(coefs.mean(axis=0))
    std = coefs.std(axis=0)
    return mean / (std + 1e-9)


def feature_weights(coefs, prev_fw, *, lr=0.5, floor=0.05) -> np.ndarray:
    """Update feature weights toward normalized stability (never fully zero)."""
    stab = stability_scores(coefs)
    rank = stab.argsort().argsort() / max(1, len(stab) - 1)   # 0..1 by stability
    target = floor + (1.0 - floor) * rank
    fw = (1 - lr) * np.asarray(prev_fw, float) + lr * target
    fw = np.clip(fw, floor, None)
    return fw / fw.mean()
