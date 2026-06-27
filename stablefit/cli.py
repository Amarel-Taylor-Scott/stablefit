"""``stablefit`` CLI — a transparent demo and fit-your-own-CSV.

    stablefit demo                       # overfit-trap, outliers, and a no-signal control
    stablefit fit data.csv --target y    # learn sample+feature weights, held-out + null verdict
"""

from __future__ import annotations

import argparse
import csv
import sys

import numpy as np

from .synth import (make_both, make_no_signal_control, make_outliers,
                    make_overfit_trap)
from .validate import held_out


def _show(res, info=None):
    h0, h1 = res["history"][0], res["history"][-1]
    print("  CV gap: %.4f -> %.4f  (in-sample %.3f->%.3f, OOF %.3f->%.3f over %d iters)"
          % (h0["gap"], h1["gap"], h0["insample"], h1["insample"],
             h0["oof"], h1["oof"], len(res["history"]) - 1))
    fw = res["feature_weights"]
    dropped = int((fw < 0.5).sum())
    print("  features: %d of %d down-weighted (<0.5)" % (dropped, len(fw)))
    sw = res["sample_weights"]
    print("  samples : %d of %d down-weighted (<0.5)" % (int((sw < 0.5).sum()), len(sw)))
    print("  held-out transfer: %.4f   (uniform %.4f, lift %+.4f)"
          % (res["learned"], res["uniform"], res["lift"]))
    print("  null p95 / p     : %.4f / %.3f" % (res["null_p95"], res["null_p"]))
    print("  VERDICT          : %s" %
          ("REAL — beats held-out null" if res["beats_null"]
           else "no gain over null (the honest answer)"))


def cmd_demo(_a) -> int:
    print("Weights start at 1 everywhere; the loop trades in-sample fit for a")
    print("smaller train<->CV gap. Every result is validated on a held-out split")
    print("against a magnitude-matched null.\n")

    print("## 1. Overfit trap (5 signal + 30 sign-flipping trap features)")
    X, y, info = make_overfit_trap(seed=0)
    r = held_out(X, y, iters=8, seed=0)
    _show(r, info)
    fw = r["feature_weights"]
    sig_w = float(np.mean(fw[info["signal"]]))
    trap_w = float(np.mean(fw[info["trap"]]))
    print("  -> mean weight: signal features %.2f vs trap features %.2f (trap suppressed)\n"
          % (sig_w, trap_w))

    print("## 2. Outlier-injected rows (~25%% corrupted targets in train, clean held-out)")
    X, y, info = make_outliers(seed=1)
    r = held_out(X, y, iters=8, seed=0)
    _show(r)
    sw = r["sample_weights"]
    outl = np.array(info["outliers"])
    outl = outl[outl < len(sw)]
    print("  -> mean weight on injected outliers %.2f vs clean %.2f\n"
          % (float(sw[outl].mean()), float(np.delete(sw, outl).mean())))

    print("## 3. No-signal control (nothing to gain — must NOT beat null)")
    X, y, _ = make_no_signal_control(seed=2)
    _show(held_out(X, y, iters=6, seed=0))
    return 0


def _load_csv(path, target):
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = next(r)
        rows = [row for row in r if row]
    if target not in header:
        raise SystemExit("stablefit: target %r not found" % target)
    ti = header.index(target)
    fi = [i for i in range(len(header)) if i != ti]
    X = np.array([[float(row[i]) for i in fi] for row in rows], float)
    y = np.array([float(row[ti]) for row in rows], float)
    return X, y, [header[i] for i in fi]


def cmd_fit(a) -> int:
    X, y, names = _load_csv(a.file, a.target)
    print("# %s: %d rows x %d feats (rows assumed time-ordered)\n" % (a.file, *X.shape))
    r = held_out(X, y, iters=a.iters, lam=a.lam, k=a.folds, seed=0)
    _show(r)
    fw = r["feature_weights"]
    order = np.argsort(fw)
    print("\n  most-suppressed features:", ", ".join(names[i] for i in order[:5]))
    print("  most-trusted features  :", ", ".join(names[i] for i in order[-5:][::-1]))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="stablefit", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo", help="trap / outliers / control walk-through").set_defaults(fn=cmd_demo)
    p = sub.add_parser("fit", help="learn sample+feature weights on your CSV")
    p.add_argument("file")
    p.add_argument("--target", required=True)
    p.add_argument("--iters", type=int, default=8)
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--lam", type=float, default=1.0)
    p.set_defaults(fn=cmd_fit)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
