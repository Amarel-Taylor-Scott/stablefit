"""UNG node adapters for stablefit — joint sample+feature weights for stability.

Pure top-level functions over the documented public API (``stablefit``,
``held_out``).  Matrices and vectors cross the boundary as nested lists; numpy
arrays never leak out.  Stochastic steps are seeded, so every node is
deterministic given ``seed``.  Each function returns a dict keyed by its
declared output port names.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from stablefit import held_out, stablefit


def _jsonable(obj: Any) -> Any:
    """Convert numpy scalars/arrays (and containers of them) to plain JSON types."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)) and not isinstance(obj, bool):
        return int(obj)
    return obj


def compute_weights(X: list[list[float]], y: list[float], k: int = 4,
                    lam: float = 1.0, iters: int = 8, lr: float = 0.5,
                    k_window: int = 7, floor: float = 0.05,
                    purge: int = 0) -> dict[str, Any]:
    """Jointly learn per-sample and per-feature weights that shrink the train-CV gap."""
    res = stablefit(np.asarray(X, float), np.asarray(y, float), k=int(k),
                    lam=float(lam), iters=int(iters), lr=float(lr),
                    k_window=int(k_window), floor=float(floor), purge=int(purge))
    return {
        "weights": _jsonable(res["sample_weights"]),
        "feature_weights": _jsonable(res["feature_weights"]),
        "report": {"history": _jsonable(res["history"]),
                   "oof": float(res["oof"])},
    }


def validate_held_out(X: list[list[float]], y: list[float], low_frac: float = 0.6,
                      n_null: int = 200, seed: int = 0, lam: float = 1.0,
                      k: int = 4, iters: int = 8) -> dict[str, Any]:
    """Learn joint weights on the early rows and validate on held-out rows vs a permutation null."""
    rep = held_out(np.asarray(X, float), np.asarray(y, float),
                   low_frac=float(low_frac), n_null=int(n_null), seed=int(seed),
                   lam=float(lam), k=int(k), iters=int(iters))
    return {"report": _jsonable(rep)}


_TAGS = ["license.mit", "runtime.python"]
_X_PORT = {"name": "X", "type_id": "amarel.types.matrix",
           "description": "Feature matrix, rows in time order (nested lists)."}
_Y_PORT = {"name": "y", "type_id": "amarel.types.vector",
           "description": "Target vector aligned with the rows of X."}

NODES = [
    {
        "fn": compute_weights,
        "id": "amarel.stablefit.compute-weights",
        "capabilities": ["weights.compute", "weights.joint-stability"],
        "summary": "Jointly learn sample weights (local consensus) and feature weights (cross-fold coefficient stability) that shrink the train-CV gap.",
        "inputs": [_X_PORT, _Y_PORT],
        "outputs": [
            {"name": "weights", "type_id": "amarel.types.weights",
             "description": "Per-sample weights (outlier/regime-break rows downweighted)."},
            {"name": "feature_weights", "type_id": "amarel.types.weights",
             "description": "Per-feature weights (fold-unstable features suppressed)."},
            {"name": "report", "type_id": "amarel.types.report",
             "description": "{'history': [{iter, insample, oof, gap}], 'oof': best out-of-fold score}."},
        ],
        "parameters": [
            {"name": "k", "value_type": "integer", "default": 4, "required": False},
            {"name": "lam", "value_type": "number", "default": 1.0, "required": False},
            {"name": "iters", "value_type": "integer", "default": 8, "required": False},
            {"name": "lr", "value_type": "number", "default": 0.5, "required": False},
            {"name": "k_window", "value_type": "integer", "default": 7, "required": False},
            {"name": "floor", "value_type": "number", "default": 0.05, "required": False},
            {"name": "purge", "value_type": "integer", "default": 0, "required": False},
        ],
        "effects": [],
        "determinism": "deterministic",
        "idempotency": "idempotent",
        "tags": _TAGS,
    },
    {
        "fn": validate_held_out,
        "id": "amarel.stablefit.validate-held-out",
        "capabilities": ["weights.validate", "validation.null-controlled"],
        "summary": "Held-out validation of the jointly learned weights against uniform and a magnitude-matched permutation null.",
        "inputs": [_X_PORT, _Y_PORT],
        "outputs": [
            {"name": "report", "type_id": "amarel.types.report",
             "description": "{learned, uniform, lift, null_p95, null_mean, null_p, beats_null, sample_weights, feature_weights, history}."},
        ],
        "parameters": [
            {"name": "low_frac", "value_type": "number", "default": 0.6, "required": False},
            {"name": "n_null", "value_type": "integer", "default": 200, "required": False},
            {"name": "seed", "value_type": "integer", "default": 0, "required": False},
            {"name": "lam", "value_type": "number", "default": 1.0, "required": False},
            {"name": "k", "value_type": "integer", "default": 4, "required": False},
            {"name": "iters", "value_type": "integer", "default": 8, "required": False},
        ],
        "effects": [],
        "determinism": "deterministic",
        "idempotency": "idempotent",
        "tags": _TAGS,
    },
]
