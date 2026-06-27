"""Numeric core: ridge with BOTH per-sample and per-feature weights, plus the
time-fold cross-validation primitives the stability loop is built on.

Per-feature weights scale the columns (a feature with weight ~0 contributes ~0),
so downweighting an unstable feature is a soft, continuous form of selection.
Per-sample weights are the usual WLS weights. The Gram solve makes every refit
cheap, so the iterated loop and the null permutations stay fast.
"""

from __future__ import annotations

import numpy as np


def pearson(a, b) -> float:
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b / d) if d > 0 else 0.0


def time_folds(n, k=4, purge=0):
    """K contiguous time blocks → list of (train_idx, test_idx), purged."""
    folds = np.array_split(np.arange(n), k)
    out = []
    for f in folds:
        lo, hi = f[0], f[-1]
        mask = np.ones(n, bool)
        mask[max(0, lo - purge):hi + 1 + purge] = False
        out.append((np.arange(n)[mask], f))
    return out


def fit_coef(X, y, sw=None, fw=None, lam=1.0):
    """Weighted, feature-scaled ridge. Returns (beta, mu, ybar) in the scaled,
    centered space (so betas from folds sharing the same ``fw`` are comparable)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float).ravel()
    n, p = X.shape
    fw = np.ones(p) if fw is None else np.asarray(fw, float)
    Xs = X * fw
    w = np.ones(n) if sw is None else np.asarray(sw, float)
    w = w / w.mean() if w.mean() > 0 else np.ones(n)
    sw_sum = w.sum()
    mu = (Xs * w[:, None]).sum(axis=0) / sw_sum
    ybar = float((y * w).sum() / sw_sum)
    Xc = Xs - mu
    A = Xc.T @ (Xc * w[:, None])
    b = Xc.T @ (w * (y - ybar))
    beta = np.linalg.solve(A + lam * np.eye(p), b)
    return beta, mu, ybar


def predict_coef(X, beta, mu, ybar, fw):
    X = np.asarray(X, float)
    return (X * fw - mu) @ beta + ybar


def weighted_ridge(X_tr, y_tr, X_ev, sw=None, fw=None, lam=1.0):
    fw = np.ones(np.asarray(X_tr).shape[1]) if fw is None else np.asarray(fw, float)
    beta, mu, ybar = fit_coef(X_tr, y_tr, sw=sw, fw=fw, lam=lam)
    return predict_coef(X_ev, beta, mu, ybar, fw)
