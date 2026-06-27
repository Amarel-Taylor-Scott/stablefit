"""Tests for stablefit — the mechanism (gap shrinks, traps suppressed, outliers
downweighted, held-out beats null) and the honesty control (no-signal must NOT
beat null). Requires numpy; no network."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stablefit import (  # noqa: E402
    pearson, weighted_ridge, stablefit, held_out, stability_scores, fold_coefs,
    sample_weights, oof_predict,
)
from stablefit.synth import (  # noqa: E402
    make_overfit_trap, make_outliers, make_no_signal_control,
)


def test_weighted_ridge_ols_baseline():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((120, 4))
    beta = np.array([1.0, -2.0, 0.5, 3.0])
    y = X @ beta
    pred = weighted_ridge(X[:90], y[:90], X[90:], lam=1e-6)
    assert pearson(pred, y[90:]) > 0.999


def test_feature_scaling_shrinks_influence():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 3))
    y = X[:, 0] * 5
    full = weighted_ridge(X[:70], y[:70], X[70:], lam=0.1)
    fw = np.array([1.0, 1.0, 1.0]); fw0 = np.array([0.01, 1.0, 1.0])
    zeroed = weighted_ridge(X[:70], y[:70], X[70:], fw=fw0, lam=0.1)
    assert pearson(full, y[70:]) > pearson(zeroed, y[70:])   # killing feature 0 hurts


def test_stability_separates_stable_from_flipping():
    X, y, info = make_overfit_trap(seed=0)
    coefs = fold_coefs(X, y, k=4, lam=1.0)
    stab = stability_scores(coefs)
    assert np.mean(stab[info["signal"]]) > np.mean(stab[info["trap"]])


def test_loop_shrinks_gap_and_suppresses_traps():
    X, y, info = make_overfit_trap(seed=0)
    res = stablefit(X, y, iters=8)
    h0, h1 = res["history"][0], res["history"][-1]
    assert h1["gap"] <= h0["gap"]                       # gap does not grow; should shrink
    fw = res["feature_weights"]
    assert np.mean(fw[info["signal"]]) > np.mean(fw[info["trap"]])


def test_consensus_downweights_outliers():
    X, y, info = make_outliers(seed=1)
    oof = oof_predict(X, y, k=4, lam=1.0)
    sw = sample_weights(y, oof, np.ones(len(y)))
    outl = np.array(info["outliers"])
    assert sw[outl].mean() < np.delete(sw, outl).mean()


def test_held_out_outliers_beats_null():
    # the sample-consensus lever: downweighting genuine bad rows beats chance
    X, y, _ = make_outliers(frac=0.3, seed=1)
    r = held_out(X, y, iters=10, n_null=200, seed=0)
    assert r["lift"] > 0.01, r["lift"]
    assert r["beats_null"], r


def test_no_signal_control_does_not_beat_null():
    X, y, _ = make_no_signal_control(seed=2)
    r = held_out(X, y, iters=6, n_null=200, seed=0)
    assert r["lift"] < 0.03, r["lift"]
    assert not r["beats_null"], r


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
    print("\n%d passed" % len(fns))
