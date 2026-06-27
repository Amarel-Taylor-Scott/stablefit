"""Synthetic datasets that isolate what stablefit is meant to fix.

* ``make_overfit_trap`` — a few stable signal features plus many *trap* features
  that correlate with y with a sign that FLIPS across time-blocks. A pooled fit
  latches onto them in-sample (overfit); their coefficients are unstable across
  folds, so feature-stability should suppress them.
* ``make_outliers`` — clean signal with a fraction of rows whose target is
  corrupted; local consensus should downweight them.
* ``make_both`` — traps and outliers together (the joint case).
* ``make_no_signal_control`` — clean, low-dimensional, nothing to gain; stablefit
  must NOT beat the null here (the honest negative).
"""

from __future__ import annotations

import numpy as np


def make_overfit_trap(n=1600, p_signal=5, p_trap=30, noise=0.5, seed=0):
    rng = np.random.default_rng(seed)
    Xs = rng.standard_normal((n, p_signal))
    beta = rng.standard_normal(p_signal)
    y = Xs @ beta + noise * rng.standard_normal(n)
    t = np.arange(n)
    block_sign = np.where((t // (n // 4)) % 2 == 0, 1.0, -1.0)   # flips across blocks
    Xtrap = rng.standard_normal((n, p_trap)) + (block_sign * y)[:, None] * 0.5
    X = np.column_stack([Xs, Xtrap])
    return X, y, {"signal": list(range(p_signal)),
                  "trap": list(range(p_signal, p_signal + p_trap))}


def make_outliers(n=1600, p=8, frac=0.25, noise=0.3, seed=0, clean_high=True,
                  low_frac=0.6):
    """Clean signal with a fraction of corrupted-target rows. With ``clean_high``
    the corruption sits only in the training region and the held-out tail is
    clean — so downweighting the bad rows measurably improves held-out transfer
    (and beats a random-downweighting null)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    beta = rng.standard_normal(p)
    y = X @ beta + noise * rng.standard_normal(n)
    mask = rng.random(n) < frac
    if clean_high:
        mask[int(n * low_frac):] = False
    y[mask] += rng.standard_normal(int(mask.sum())) * 12.0          # corrupted targets
    return X, y, {"outliers": np.where(mask)[0].tolist()}


def make_both(n=1800, p_signal=5, p_trap=30, frac=0.12, noise=0.4, seed=0):
    rng = np.random.default_rng(seed)
    Xs = rng.standard_normal((n, p_signal))
    beta = rng.standard_normal(p_signal)
    y = Xs @ beta + noise * rng.standard_normal(n)
    t = np.arange(n)
    block_sign = np.where((t // (n // 4)) % 2 == 0, 1.0, -1.0)
    Xtrap = rng.standard_normal((n, p_trap)) + (block_sign * y)[:, None] * 0.5
    X = np.column_stack([Xs, Xtrap])
    mask = rng.random(n) < frac
    y[mask] += rng.standard_normal(int(mask.sum())) * 10.0
    return X, y, {"signal": list(range(p_signal)),
                  "trap": list(range(p_signal, p_signal + p_trap)),
                  "outliers": np.where(mask)[0].tolist()}


def make_no_signal_control(n=1500, p=8, noise=1.0, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    beta = rng.standard_normal(p)
    y = X @ beta + noise * rng.standard_normal(n)
    return X, y, {}
