# stablefit

> A transparent feedback loop that jointly learns **sample weights** *and*
> **feature weights**, with the objective being a low **train↔CV gap**
> (generalization / stability) — not raw fit. Every sample and feature starts at
> weight 1; the loop downweights overfit-prone features (unstable across folds)
> and inconsistent samples (local-consensus outliers), and **validates the result
> on a held-out split against a magnitude-matched null** so a reported gain is
> real, not memorized. numpy-only.

```python
from stablefit import held_out
from stablefit.synth import make_outliers
X, y, info = make_outliers()          # rows in time order
r = held_out(X, y)
print(r["uniform"], r["learned"], r["beats_null"])
```

```
$ stablefit demo
## 2. Outlier-injected rows (~25% corrupted in train, clean held-out)
  CV gap: 0.0240 -> 0.0041   samples: 199/960 down-weighted
  held-out transfer: 0.9871   (uniform 0.9822, lift +0.0049)
  null p95 / p     : 0.9849 / 0.000
  VERDICT          : REAL — beats held-out null
  -> mean weight on injected outliers 0.39 vs clean 1.20
```

## The idea (and why the objective is the gap)

Minimizing the **train↔CV gap** is inherently anti-overfitting: it rewards the
weights that make the in-fold and out-of-fold scores *agree*, so it fights
fragility instead of feeding it. stablefit pursues that with two transparent,
explainable levers, iterated from a uniform start:

| Lever | Signal | Effect |
|---|---|---|
| **feature weights** | cross-fold **coefficient stability** (`|mean| / std` across time-folds) | suppress features whose coefficient swings fold-to-fold — the overfit-prone ones |
| **sample weights** | local **group consensus** (out-of-fold residual vs the k-neighborhood) | downweight rows whose error disagrees with their neighbors — outliers / regime breaks |

You can read both off the result: which features were trusted, which samples
suppressed, and the gap shrinking over iterations.

## Honesty is the whole point

This was built to a strict rule: **nothing counts unless it beats a held-out
null.** `held_out` learns the weights on the early rows, refits, scores the late
rows it never saw, and compares to (a) uniform weighting and (b) the learned
weights **permuted** across rows/features (same magnitude, structure destroyed).

What the demo shows — reported faithfully, not cherry-picked:

- **Sample lever → real.** Downweighting genuine corrupted rows beats the null
  (p≈0.000): placing the low weights on the *right* rows matters.
- **Feature lever → an explainable diagnostic.** It correctly identifies unstable
  features (in the demo, mean weight 1.6 on signal vs 0.9 on the sign-flipping
  traps) and shrinks the train↔CV gap — but on held-out its gain is usually
  *within the null band*, because plain ridge already regularizes. The tool
  **says so** ("no gain over null") rather than dressing generic shrinkage up as
  a discovery.
- **No-signal control → nothing.** When there's nothing to gain, no weighting
  beats the null. That's the correct answer, and seeing it is how you know the
  tool isn't fooling you.

A null result here is a *finding*. If you believe there's signal, widen the
levers (more folds, different `k_window`, your own driver) — it stays
null-controlled, so a real effect surfaces and a phantom won't.

## API

```python
from stablefit import stablefit, held_out
res = stablefit(X, y, k=4, iters=8)      # learn weights (no held-out split)
res["sample_weights"], res["feature_weights"], res["history"]   # all inspectable

r = held_out(X, y, low_frac=0.6, n_null=200)   # learn + validate vs null
r["learned"], r["uniform"], r["lift"], r["beats_null"]
```

## CLI

```bash
stablefit demo                       # trap (feature) / outliers (sample) / no-signal control
stablefit fit data.csv --target y    # learn weights + the held-out null verdict, with the
                                     # most-suppressed and most-trusted features named
```

## Pairs with

[`waveweight`](https://github.com/Amarel-Taylor-Scott/waveweight) (which *rows*
to trust, by pattern/window) and
[`shiftblend`](https://github.com/Amarel-Taylor-Scott/shiftblend) (which
*models/features* to blend under shift). stablefit does both axes jointly, driven
by stability. Same held-out + null discipline throughout.

## Install

```bash
pip install numpy && git clone https://github.com/Amarel-Taylor-Scott/stablefit.git
```

MIT. Depends only on numpy.
