"""L2-regularized linear logistic regression, fitted with the library's L-BFGS.

Used by :class:`mimcs.adaptation.ClassifierTermination` to ask whether a draw's *features* reveal
which part of warmup it came from. Same shape of use as the factory's metric regression
(:func:`mimcs.factory.regression.fit_metric_expr`): hand :func:`mimcs.optim.minimize` a scalar
JAX loss and an initial pytree.

**The ridge is not optional.** Early in warmup the two periods are often linearly separable ---
exactly when the classifier must say so and warmup must continue --- and there the unregularized
maximum-likelihood estimate does not exist: the likelihood keeps rising as the coefficients grow,
with no optimum to find.

What that looks like in practice is worse than a failure. L-BFGS does *not* diverge or complain:
the gradient of a separable fit underflows exponentially as ``|w|`` grows, so ``gtol`` is met and
``converged`` comes back true --- at whatever point the tolerance happened to bite. Tightening
``gtol`` from 1e-4 to 1e-10 walks ``|w|`` from ~2.8 to ~6.1 on the same data, each time reporting
success. The fit is a property of the stopping rule, not of the data, and nothing in the result
says so. With a ridge the optimum is real and ``|w|`` is identical across those tolerances.

The ridge also makes the fit unique when the features are rank-deficient.

**The fit is buffered.** A termination check refits on the whole feature history, which grows by a
fixed number of rows every check, so the naive arrangement hands XLA a shape it has never seen at
every single check. Two things are needed to fix that, and *neither works without the other*:

* rows are padded up to a power of two (:func:`row_buffer`), the padding carrying zero weight, so
  that shapes repeat instead of growing; and
* the fit runs under a cached :func:`jax.jit` (:func:`_fit_jit`) with the data passed as
  arguments, so a repeated shape is a cache *hit*.

The second is the counter-intuitive half. :func:`mimcs.optim.minimize` is a bare
``lax.while_loop``, and calling it outside ``jit`` binds that loop eagerly: the cond/body jaxprs
are rebuilt and the dispatch cache missed on **every call, at any shape, even an identical one**.
Buffering alone therefore saves nothing at all. Measured on Neal's funnel through the factory
default (2000 warmup iterations), the pair together took compilation events 489 -> 98, XLA
compilation 10.5 s -> 2.5 s, warmup wall 16.1 s -> 5.5 s and peak RSS 981 -> 559 MiB, with warmup
stopping at the same iteration.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .._logging import get_logger
from ..optim import minimize

log = get_logger(__name__)

DEFAULT_L2 = 1e-2
LOG2 = float(np.log(2.0))

#: smallest row buffer. Below this the padding is pure waste and the fits are trivially cheap.
MIN_ROWS_BITS = 6                      # 2**6 = 64 rows


class LogisticFit(NamedTuple):
    """A fitted classifier. ``w`` are the coefficients, ``b`` the intercept."""

    w: Array
    b: Array
    loss: float
    converged: bool


def logistic_loss(params, X: Array, y: Array, l2: float, wt: Array | None = None) -> Array:
    """Weighted mean binary cross-entropy plus ``l2/2 * ||w||^2``. The intercept is not penalized.

    ``wt`` are per-sample weights, normalized internally by their sum. With ``wt=None`` (or any
    constant vector) this is the plain mean, so the weighted path is a strict generalization ---
    nothing about the unweighted fit moves.
    """
    w, b = params
    z = X @ w + b
    # softplus(z) - y*z is -log p(y|z) written so neither branch overflows for large |z|.
    per_sample = jax.nn.softplus(z) - y * z
    nll = jnp.mean(per_sample) if wt is None else jnp.sum(wt * per_sample) / jnp.sum(wt)
    return nll + 0.5 * l2 * jnp.sum(w ** 2)


def class_weights(y, keep=None) -> np.ndarray:
    """Per-sample weights that give the two classes equal total weight; ``0`` outside ``keep``.

    The burn-in search fits a *prefix vs tail* classifier whose classes are deliberately
    unbalanced --- a candidate prefix may be a twentieth of the history. Unweighted, the fit is
    free to buy most of its likelihood by predicting the majority, and both the loss and the
    accuracy stop being about separability. Weighting restores the balanced problem without
    discarding rows, which subsampling the majority class would do.

    ``keep`` is a boolean mask of the rows that take part. Zeroing a row's weight is how the
    search excludes it, rather than slicing it out: every candidate split then presents the fit
    with the *same* ``(n, p)`` array, and :func:`mimcs.optim.minimize` is one ``lax.while_loop``
    whose XLA compilation is keyed on that shape. Slicing would recompile the whole optimizer for
    each of the eight-or-so candidates at every check.

    That is the same mechanism :func:`row_buffer` uses to hold the shape steady *across* checks,
    one layer up: excluded row and padding row are the same thing to the loss.
    """
    y = np.asarray(y, dtype=np.float64) > 0.5
    keep = np.ones(y.shape, dtype=bool) if keep is None else np.asarray(keep, dtype=bool)
    wt = np.zeros(y.shape, dtype=np.float64)
    for label in (False, True):
        rows = keep & (y == label)
        if rows.any():
            wt[rows] = 0.5 / rows.sum()      # each class sums to 1/2, whatever its size
    return wt


def row_buffer(n: int) -> int:
    """The buffered row count a fit on ``n`` rows is padded up to: the next power of two.

    The classifier's design matrix grows by a fixed number of rows at every termination check, so
    without buffering *every* check presents a shape XLA has never seen. Rounding up to a power of
    two turns that unbounded family of shapes into ``log2`` of it --- a check only pays for a
    compilation when the history crosses a power of two.
    """
    return 1 << max(MIN_ROWS_BITS, int(n - 1).bit_length())


def _buffered(X, y, wt):
    """Pad ``(X, y, wt)`` up to :func:`row_buffer` rows, the padding carrying **zero weight**.

    The padded rows are zeros, not junk: they still flow through ``X @ w``, and ``0 * nan`` is
    ``nan``, so a non-finite pad would poison the loss however little weight it carried.

    A caller who passed no weights gets ``1`` on the real rows, so
    ``sum(wt * l) / sum(wt)`` is exactly the mean over them --- see :func:`logistic_loss`.
    """
    n, p = np.shape(X)[0], np.shape(X)[1]
    size = row_buffer(n)
    Xb = np.zeros((size, p), dtype=np.float64)
    Xb[:n] = np.asarray(X, dtype=np.float64)
    yb = np.zeros(size, dtype=np.float64)
    yb[:n] = np.asarray(y, dtype=np.float64)
    wb = np.zeros(size, dtype=np.float64)
    wb[:n] = 1.0 if wt is None else np.asarray(wt, dtype=np.float64)
    return (jnp.asarray(Xb, float), jnp.asarray(yb, float), jnp.asarray(wb, float))


@partial(jax.jit, static_argnames=("l2", "max_iter", "m", "gtol", "max_ls", "c1", "shrink"))
def _fit_jit(X, y, wt, w0, b0, *, l2, max_iter, m, gtol, max_ls, c1, shrink):
    """The fit itself, under a **cached** ``jax.jit``.

    Buffering the rows is necessary but on its own does nothing, which is the counter-intuitive
    part: :func:`mimcs.optim.minimize` is a bare ``lax.while_loop``, and calling it outside ``jit``
    binds that loop *eagerly* --- the cond/body jaxprs are rebuilt and the dispatch cache missed on
    every call, at any shape, even an identical one (measured: ~0.2 s per repeat call on a
    ``(300, 6)`` fit). Only a cached ``jit`` turns a repeated shape into a cache hit, and only
    buffering makes shapes repeat. Both, or neither works.

    ``X``, ``y``, ``wt`` and the warm start are **arguments, not closures**: a closed-over array is
    a compile-time constant, which would key the cache on its contents and defeat the point (the
    same rule :mod:`mimcs.pt.tempering` documents for the ladder).
    """
    res = minimize(lambda params: logistic_loss(params, X, y, l2, wt), (w0, b0),
                   max_iter=max_iter, m=m, gtol=gtol, max_ls=max_ls, c1=c1, shrink=shrink,
                   warn_max_iter=False)
    w, b = res.x
    return w, b, res.fun, res.converged


def fit_logistic(X, y, *, l2: float = DEFAULT_L2, init=None, wt=None, max_iter: int = 1000,
                 m: int = 10, gtol: float = 1e-6, max_ls: int = 25, c1: float = 1e-4,
                 shrink: float = 0.5, **opt) -> LogisticFit:
    """Fit ``P(y=1 | X) = sigmoid(X w + b)`` by L-BFGS.

    The rows are **buffered** to a power of two and the padding given zero weight, so that a
    sequence of fits on a growing history compiles once per power of two rather than once per
    fit --- see :func:`row_buffer` and :func:`_fit_jit`.

    Args:
        X: ``(n, p)`` features. Standardize them first --- L-BFGS conditioning depends on it, and
            a single ridge strength only means the same thing across features on a common scale.
        y: ``(n,)`` labels in ``{0, 1}``.
        l2: ridge strength (see the module docstring: do not set this to 0).
        init: ``(w, b)`` to start from --- e.g. the previous check's fit, which is close enough to
            save most of the iterations. Only an initial point: :func:`mimcs.optim.minimize`
            rebuilds its curvature history from scratch either way.
        wt: ``(n,)`` per-sample weights, e.g. :func:`class_weights` for an unbalanced split.
            ``None`` weights every row equally.
        max_iter, m, gtol, max_ls, c1, shrink: :func:`mimcs.optim.minimize` hyperparameters. They
            are *static* to the cached fit, so each distinct combination is its own compilation ---
            vary them across calls only deliberately.

    Returns:
        A :class:`LogisticFit`. ``converged`` is reported rather than assumed --- a fit that ran
        out of iterations should not be read as evidence of anything.
    """
    if opt:                        # keeps a typo from being silently swallowed by **opt
        raise TypeError(f"fit_logistic got unexpected keyword argument(s) {sorted(opt)}")
    n, p = np.shape(X)[0], np.shape(X)[1]
    Xb, yb, wb = _buffered(X, y, wt)
    w0, b0 = (jnp.zeros(p, float), jnp.zeros((), float)) if init is None else init
    # The iteration cap is not an event here: this fit runs at every termination check, warm
    # started and ridged, and routinely stops on the cap with a gradient already near ``gtol``.
    # The caller is handed ``converged`` and decides what to make of it, so the fit reports the
    # cap at DEBUG rather than warning a few dozen times per run.
    w, b, fun, converged = _fit_jit(Xb, yb, wb, jnp.asarray(w0, float), jnp.asarray(b0, float),
                                    l2=float(l2), max_iter=int(max_iter), m=int(m),
                                    gtol=float(gtol), max_ls=int(max_ls), c1=float(c1),
                                    shrink=float(shrink))
    fit = LogisticFit(w=w, b=b, loss=float(fun), converged=bool(converged))
    if not fit.converged:
        log.debug("logistic fit on %d x %d features did not converge (loss %.6g); the check that "
                  "reads it should not be taken as evidence of much", n, p, fit.loss)
    return fit


def accuracy(fit: LogisticFit, X, y) -> float:
    """Fraction of ``y`` the fit predicts correctly (at the natural 1/2 cut).

    On :func:`scores`, i.e. in numpy, for the reason given there: the validation block is a new
    row count at every check, and every new shape on the JAX path is another XLA compilation.
    """
    return float(np.mean((scores(fit, X) > 0) == (np.asarray(y, dtype=np.float64) > 0.5)))


def scores(fit: LogisticFit, X) -> np.ndarray:
    """``X w + b`` as plain numpy. The scorers below are elementwise arithmetic on this, and
    doing it in numpy keeps them off the JAX dispatch path --- which matters because the burn-in
    search calls them on a different row count at every check, and each new shape is a new XLA
    compilation."""
    return np.asarray(X, dtype=np.float64) @ np.asarray(fit.w, dtype=np.float64) + float(fit.b)


def _class_means(values, y, empty: float) -> tuple[float, float]:
    """``(mean over class 0, mean over class 1)``, with ``empty`` where a class has no rows.

    A candidate prefix short enough to contribute no validation rows is a real case in the
    burn-in search, and the mean of nothing is a silent NaN.
    """
    out = []
    for c in (False, True):
        rows = values[y == c]
        out.append(float(np.mean(rows)) if rows.size else empty)
    return out[0], out[1]


def balanced_accuracy(fit: LogisticFit, X, y) -> float:
    """Mean of the two per-class accuracies --- 1/2 under chance whatever the class sizes.

    Plain :func:`accuracy` is the right statistic for the *decision*, whose two halves are equal
    by construction. It is the wrong one for an unbalanced prefix-vs-tail split, where predicting
    the majority already scores well above 1/2.
    """
    y = np.asarray(y, dtype=np.float64) > 0.5
    correct = (scores(fit, X) > 0) == y
    a0, a1 = _class_means(correct, y, empty=0.5)      # an absent class contributes chance
    return 0.5 * (a0 + a1)


def log_score(fit: LogisticFit, X, y) -> float:
    """``log 2 + `` the balanced mean held-out log-likelihood, in nats.

    The classifier two-sample statistic read as a divergence estimate (Friedman 2003; Lopez-Paz &
    Oquab 2017): for a classifier trained to tell two samples apart, ``log 2 + E[log p_hat]`` is a
    lower-bound estimate of the Jensen--Shannon divergence between them. It is ``0`` when the fit
    is uninformative and grows with separability.

    Preferred to :func:`accuracy` wherever the number only *guides* a search: it is a proper
    scoring rule, so it uses the whole predictive distribution rather than which side of 1/2 it
    fell on, which makes it smooth in the split point and far less granular on the small
    validation sets a short candidate prefix produces. The classes are weighted equally, for the
    same reason :func:`balanced_accuracy` exists.

    Clipped below at 0: an over-fitted direction can score *worse* than uninformative out of
    sample, and a negative divergence is not a meaningful degree of separation.
    """
    y = np.asarray(y, dtype=np.float64) > 0.5
    z = scores(fit, X)
    # -log(1 + exp(-z)) = log sigmoid(z) = log p_hat(class 1); the mirror for class 0.
    ll = np.where(y, -np.logaddexp(0.0, -z), -np.logaddexp(0.0, z))
    l0, l1 = _class_means(ll, y, empty=-LOG2)          # an absent class contributes 0
    return max(LOG2 + 0.5 * (l0 + l1), 0.0)
