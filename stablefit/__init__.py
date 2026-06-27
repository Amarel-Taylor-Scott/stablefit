"""stablefit — jointly learn sample weights AND feature weights for stability.

A transparent feedback loop for tabular ML whose objective is a low train↔CV gap
(generalization), not raw fit. Starting from weight 1 everywhere it iterates:

* **feature weights** ← cross-fold coefficient stability (suppress features whose
  coefficient swings across time-folds — the overfit-prone ones);
* **sample weights** ← local group consensus (downweight rows whose out-of-fold
  error disagrees with their neighborhood — outliers / regime breaks).

Both are explainable (read off which features were trusted, which samples
suppressed) and validated on a held-out split against a magnitude-matched null,
so a reported gain is real, not memorized. numpy-only.

    from stablefit import held_out
    from stablefit.synth import make_overfit_trap
    X, y, info = make_overfit_trap()
    r = held_out(X, y)
    print(r["uniform"], r["learned"], r["beats_null"])
"""

from __future__ import annotations

from . import synth
from .core import pearson, weighted_ridge, time_folds, fit_coef, predict_coef
from .stability import (fold_coefs, oof_predict, stability_scores,
                        feature_weights)
from .consensus import sample_weights
from .loop import stablefit
from .validate import held_out

__all__ = [
    "synth", "pearson", "weighted_ridge", "time_folds", "fit_coef", "predict_coef",
    "fold_coefs", "oof_predict", "stability_scores", "feature_weights",
    "sample_weights", "stablefit", "held_out",
]
__version__ = "0.1.0"
