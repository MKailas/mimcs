"""Plain (mean-field) variational inference: ADVI with Adam and a patience stop.

Automatic differentiation variational inference (Kucukelbir, Tran, Ranganath, Gelman & Blei 2017,
JMLR 18(14)) in its mean-field form. The approximation is a normal with a diagonal covariance,
``q = N(mu, diag(exp(2 omega)))``, over the unconstrained coordinates. It maximises the ELBO
``E_q[log p] + H(q)`` by stochastic gradient ascent:

* **Gradient.** At every iteration ``K`` (``elbo_draws``) reparameterised draws
  ``phi_k = mu + exp(omega) * eps_k`` estimate the gradient:
  ``d/dmu = mean_k grad log p(phi_k)`` and ``d/domega = mean_k grad log p(phi_k) * eps_k *
  exp(omega) + 1``, the ``+ 1`` being the entropy's exact gradient.
* **Optimiser.** Adam (Kingma & Ba 2015) with its standard constants: ``learning_rate`` 1e-3,
  ``b1`` 0.9, ``b2`` 0.999, ``eps`` 1e-8.
* **Start.** ``mu`` is the start point and ``omega = 0``, a unit scale (Stan's initialisation).

The defaults differ from Stan's ADVI, which uses its own adaptive step-size sequence, a single
gradient draw, and a relative-tolerance stop on a separate 100-draw ELBO every 100 iterations.

**Stopping.** The same ``K`` draws give an unbiased ELBO estimate at the current iterate for free:
``mean_k log p(phi_k) + sum(omega) + n/2 (1 + log 2 pi)``. The run stops once ``patience``
iterations pass without that estimate improving on its best, or at ``max_iters``. Unlike
Pathfinder, the stopping criterion is the optimisation objective itself. The **last** iterate is
returned, not the best-estimate one: the best estimate is a noisy pick ``patience`` iterations
back, and stochastic ascent does not move away from the optimum on average.

**Non-finite draws** (a non-finite log density or gradient) are dropped from the gradient, which
averages over an iteration's finite draws. Dropping the whole update instead strands a fit whose
unit-scale start straddles a support boundary: every iteration then has some draw outside, and
the fit never moves (measured on a half line). The ELBO estimate of an iteration with a dropped
draw is ``-inf``, since q has mass where p has none: averaging the rest is biased high while q
leaks, and patience then stops on that stale record (measured: half-line fit stopped at 1.0 of
3.0). An iteration with *no* finite draw makes no update. Every iteration counts toward
``patience`` and ``max_iters``. If no iteration had all its draws finite the run has **failed**:
the result repeats the final mean (the start, if nothing was ever finite) with ``log q = +inf``,
Pathfinder's convention.
"""

from __future__ import annotations

import math
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

__all__ = ["ADVIResult", "advi", "ADVI_DEFAULTS", "advi_options"]

#: Mean-field ADVI's defaults: Adam's standard constants, ``K = 5`` draws per iteration (as
#: Pathfinder's ELBO), ``M = 100`` draws from the result. ``patience`` 2000 and ``max_iters``
#: 10000 are provisional (ELBO traces, 2026-11-21): at learning rate 1e-3, a patience below 2000
#: stops even a 10-d Gaussian > 1 nat short (90th percentile), and the slow targets are still
#: climbing at 10000 whatever the patience.
ADVI_DEFAULTS = dict(elbo_draws=5, learning_rate=1e-3, b1=0.9, b2=0.999, eps=1e-8,
                     max_iters=10_000, patience=2000, draws=100)


class ADVIResult(NamedTuple):
    """One mean-field ADVI fit.

    ``draws`` (M, n) are from the final approximation ``N(mean, diag(exp(2 log_sd)))``, with their
    ``log_p`` and ``log_q``. ``elbo`` is the best per-iteration estimate and ``best_iter`` its
    iteration (0 = none). ``n_iter`` is the number of iterations taken, ``n_skipped`` how many
    made no update (no finite draw), and ``n_dropped`` how many draws were dropped. ``n_grad`` / ``n_value`` count the log-density evaluations with and without a
    gradient. A failed fit (no iteration with every draw finite) repeats its final mean with
    ``log_q = +inf``.
    """
    draws: Array
    log_p: Array
    log_q: Array
    mean: Array
    log_sd: Array
    elbo: Array
    best_iter: Array
    n_iter: Array
    n_skipped: Array
    n_dropped: Array
    n_grad: Array
    n_value: Array
    failed: Array


class _Carry(NamedTuple):
    mu: Array
    omega: Array
    m: Array              # Adam's first moment, (2, n): [mu, omega]
    v: Array              # Adam's second moment
    t: Array              # Adam's update count (skipped iterations make none)
    k: Array              # iterations taken
    best_elbo: Array
    best_iter: Array
    n_skipped: Array
    n_dropped: Array
    key: Array


def _entropy_const(n: int) -> float:
    return 0.5 * n * (1.0 + math.log(2.0 * math.pi))


def _init(x0: Array, key: Array) -> _Carry:
    dtype, n = x0.dtype, x0.shape[0]
    zeros = jnp.zeros((2, n), dtype)
    return _Carry(mu=x0, omega=jnp.zeros((n,), dtype), m=zeros, v=zeros, t=jnp.asarray(0),
                  k=jnp.asarray(0), best_elbo=jnp.asarray(-jnp.inf, dtype),
                  best_iter=jnp.asarray(0), n_skipped=jnp.asarray(0),
                  n_dropped=jnp.asarray(0), key=key)


def _step(logp: Callable, c: _Carry, *, elbo_draws: int, learning_rate: float, b1: float,
          b2: float, eps: float) -> tuple[_Carry, Array]:
    """One iteration: the ELBO estimates at the current iterate and one Adam update.

    Returns the new carry and the two estimates ``(mean(log p - log q), mean(log p) + H(q))``
    (``-inf`` for an iteration with a non-finite draw); the first is the one tracked for stopping.
    The gradient averages over the finite draws.
    """
    dtype, n = c.mu.dtype, c.mu.shape[0]
    key, sub = jax.random.split(c.key)
    e = jax.random.normal(sub, (elbo_draws, n), dtype)
    sd = jnp.exp(c.omega)
    lp, g = jax.vmap(jax.value_and_grad(logp))(c.mu + sd * e)
    fin = jnp.isfinite(lp) & jnp.all(jnp.isfinite(g), axis=1)
    n_fin = jnp.sum(fin)
    ok = n_fin > 0
    w = fin / jnp.maximum(n_fin, 1).astype(dtype)                 # mean over the finite draws
    lp = jnp.where(fin, lp, 0.0)
    g = jnp.where(fin[:, None], g, 0.0)
    # Two unbiased ELBO estimates from the same draws: with the exact entropy, and as
    # mean(log p - log q), whose variance vanishes as q approaches p.
    entropy = jnp.sum(c.omega) + _entropy_const(n)
    log_q = -0.5 * jnp.sum(e * e, axis=1) - jnp.sum(c.omega) - 0.5 * n * math.log(2.0 * math.pi)
    # A dropped draw leaves the estimate -inf (q has mass where p has none), not the mean of the
    # rest: that is biased high while q leaks, and patience would then stop on a stale record.
    full = n_fin == elbo_draws
    elbo_h = jnp.where(full, w @ lp + entropy, -jnp.inf)
    elbo_r = jnp.where(full, w @ (lp - log_q), -jnp.inf)
    elbo = elbo_r
    grad = jnp.stack([w @ g, (w @ (g * e)) * sd + 1.0])
    grad = jnp.where(ok, grad, 0.0)

    t = c.t + 1
    m = b1 * c.m + (1.0 - b1) * grad
    v = b2 * c.v + (1.0 - b2) * grad * grad
    m_hat = m / (1.0 - b1 ** t.astype(dtype))
    v_hat = v / (1.0 - b2 ** t.astype(dtype))
    step = learning_rate * m_hat / (jnp.sqrt(v_hat) + eps)
    k = c.k + 1
    better = elbo > c.best_elbo
    new = _Carry(
        mu=jnp.where(ok, c.mu + step[0], c.mu), omega=jnp.where(ok, c.omega + step[1], c.omega),
        m=jnp.where(ok, m, c.m), v=jnp.where(ok, v, c.v), t=jnp.where(ok, t, c.t), k=k,
        best_elbo=jnp.where(better, elbo, c.best_elbo),
        best_iter=jnp.where(better, k, c.best_iter), n_skipped=c.n_skipped + (~ok),
        n_dropped=c.n_dropped + (elbo_draws - n_fin), key=key)
    return new, (elbo_r, elbo_h)


def _draw(key: Array, count: int, mu: Array, omega: Array):
    """``count`` draws from ``N(mu, diag(exp(2 omega)))`` and their ``log q``."""
    n = mu.shape[0]
    u = jax.random.normal(key, (count, n), mu.dtype)
    log_q = -0.5 * jnp.sum(u * u, axis=1) - jnp.sum(omega) - 0.5 * n * math.log(2.0 * math.pi)
    return mu + jnp.exp(omega) * u, log_q


def _fit(logp: Callable, x0: Array, key: Array, *, elbo_draws: int, learning_rate: float,
         b1: float, b2: float, eps: float, max_iters: int, patience: int | None,
         draws: int) -> ADVIResult:
    """One fit, traceable."""
    step = dict(elbo_draws=elbo_draws, learning_rate=learning_rate, b1=b1, b2=b2, eps=eps)
    fit_key, draw_key = jax.random.split(key)

    def cond(c):
        go = c.k < max_iters
        if patience is not None:              # no ELBO improvement in `patience` iterations
            go = go & (c.k - c.best_iter < patience)
        return go

    c = jax.lax.while_loop(cond, lambda c: _step(logp, c, **step)[0], _init(x0, fit_key))
    failed = ~jnp.isfinite(c.best_elbo)
    phi, log_q = _draw(draw_key, draws, c.mu, c.omega)
    log_p = jax.vmap(logp)(phi)
    log_p = jnp.where(jnp.isfinite(log_p), log_p, -jnp.inf)
    phi = jnp.where(failed, jnp.broadcast_to(c.mu, phi.shape), phi)
    log_q = jnp.where(failed, jnp.inf, log_q)
    log_p = jnp.where(failed, logp(c.mu), log_p)
    return ADVIResult(draws=phi, log_p=log_p, log_q=log_q, mean=c.mu, log_sd=c.omega,
                      elbo=c.best_elbo, best_iter=c.best_iter, n_iter=c.k,
                      n_skipped=c.n_skipped, n_dropped=c.n_dropped, n_grad=c.k * elbo_draws,
                      n_value=draws + failed, failed=failed)


def advi_options(kwargs: dict) -> dict:
    """:data:`ADVI_DEFAULTS` updated with ``kwargs``, validated (unknown keys raise)."""
    opts = dict(ADVI_DEFAULTS)
    unknown = sorted(set(kwargs) - set(opts))
    if unknown:
        raise TypeError(f"unknown ADVI option(s) {unknown} (it takes {sorted(opts)})")
    opts.update(kwargs)
    for name in ("elbo_draws", "max_iters", "patience", "draws"):
        v = opts[name]
        if name == "patience" and v is None:
            continue
        if not (isinstance(v, (int, np.integer)) and v >= 1):
            raise ValueError(f"ADVI option {name} must be an integer >= 1"
                             f"{' or None' if name == 'patience' else ''}, got {v!r}")
        opts[name] = int(v)
    for name, need, ok in (("learning_rate", "finite and > 0", lambda v: 0 < v < math.inf),
                           ("eps", "finite and > 0", lambda v: 0 < v < math.inf),
                           ("b1", "in [0, 1)", lambda v: 0 <= v < 1),
                           ("b2", "in (0, 1)", lambda v: 0 < v < 1)):
        v = opts[name]
        if not (isinstance(v, (int, float, np.integer, np.floating))
                and not isinstance(v, bool) and ok(float(v))):
            raise ValueError(f"ADVI option {name} must be {need}, got {v!r}")
        opts[name] = float(v)
    return opts


def advi(logp: Callable[[Array], Array], x0, key, **options) -> ADVIResult:
    """Mean-field ADVI on a flat target ``logp: (n,) -> ()`` from ``x0``.

    ``options`` are those of :data:`ADVI_DEFAULTS`. ``logp`` may return ``-inf``/NaN off its
    support. Compiled once per call (``logp`` is closed over), so call it once per fit rather than
    in a loop.
    """
    opts = advi_options(options)
    x0 = jnp.asarray(x0, float)
    return jax.jit(lambda x, k: _fit(logp, x, k, **opts))(x0, key)
