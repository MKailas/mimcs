"""Fixed-draw (deterministic) mean-field variational inference, optimised by L-BFGS.

The sample-average approximation of the mean-field ELBO: draw ``K`` (``elbo_draws``) standard
normal vectors ``g_k`` once, and maximise the then deterministic objective

    F(mu, omega) = mean_k [log p(x_k) - log q(x_k)],   x_k = mu + exp(omega) * g_k,
                 = mean_k log p(x_k) + sum(omega) + mean_k |g_k|^2 / 2 + n/2 log(2 pi),

over the mean ``mu`` and log scale ``omega`` of ``q = N(mu, diag(exp(2 omega)))``, with the same
L-BFGS (and Wolfe backtracking) as Pathfinder. This is DADVI (Giordano, Ingram & Broderick 2024,
"Black box variational inference with a deterministic objective", JMLR 25(18)), here with
``K = 5`` draws (theirs: 30) and L-BFGS (theirs: trust-region Newton-CG). Unlike Pathfinder the
optimised objective is the one the approximation is judged by, and unlike ADVI it needs no
learning rate.

What the objective is (``docs/design/02``): since ``log sd_k(x_ki) = omega_i + log sd_k(g_ki)``,
``F`` is the mean log density of ``K`` points, constrained to be a coordinatewise-affine image of
the fixed pattern, plus the log of each coordinate's spread. ``K >= 2`` is needed (at ``K = 1`` a
point can sit at the mode while ``sigma -> inf``). The spread term cancels a hierarchical scale's
normaliser, so ``F`` is bounded down a funnel's neck, where ``log p`` (and so Pathfinder's path)
is not. On a Gaussian target the realised centroid ``mu + sigma * gbar`` is the exact mean, but
``sigma`` itself is the cloud's spread divided by the pattern's sample sd: a noisy, inflated
standard deviation (``E[sigma^2] / sd^2 = K / (K - 3)``). ``standardize=True`` centres and scales
the pattern per coordinate, which removes that noise (a departure from i.i.d. draws).

**Read-out** (``covariance``):

* ``"diag"``: ``N(mu, diag(exp(2 omega)))``, the fitted ``q``.
* ``"lr"`` (default): the linear-response covariance of the fit's own mean estimate ``xbar = mu +
  sigma * gbar``, ``J^T (-hess F)^-1 J`` with ``J = d xbar / d eta = [I; diag(sigma * gbar)]``,
  around ``xbar``. It is exact (full covariance, any draws) on a Gaussian target. The Hessian is
  formed exactly, from ``2n`` Hessian-vector products of ``F`` (about ``4 n K`` gradient
  evaluations), and the ``n x n`` covariance is Cholesky-factored.
* ``"lr_lbfgs"``: the same sandwich with L-BFGS's own estimate of ``(-hess F)^-1``, Pathfinder's
  diagonal-plus-low-rank ``alpha``-recovered factors over the ``2n`` coordinates. The sandwich
  keeps that form (``diag(alpha_mu + v^2 alpha_omega) + beta' gamma beta'^T``, ``beta' = beta_mu
  + v * beta_omega``, ``v = sigma * gbar``) and is drawn with Pathfinder's Algorithm 4. It costs
  nothing extra, but underestimates the variances 3-7x (writeups/dadvi_init).

A covariance that is not positive definite falls back, ``"lr"`` to ``"lr_lbfgs"`` and that to
``"diag"`` (``readout``, ``lr_failed``). The exact Hessian is formed only for a fit that converged
(stopped on ``rel_tol``); an unconverged fit falls back without forming it (``lr_skipped``), and
one whose Hessian would exceed ``lr_max_gb`` (:func:`lr_memory_gb`) uses ``"lr_lbfgs"`` as
configured (``lr_capped``).

**Run.** The start is ``(x0, omega = init_log_sd)``, 0 by default. If ``F`` is not finite there,
``omega`` is lowered by ``log 4`` up to 10 times (the points then collapse onto ``x0``). A
non-finite log density or
gradient at any point makes ``F = -inf``, which the line search backtracks from. A step taken
with no curvature pair in the history (the first, or after every pair so far was rejected) is
scaled to unit length when the gradient is longer (L-BFGS-B's rule; Pathfinder's ``lambda = 1``
cannot reach a finite trial in 30 halvings from a gradient of 1e14). It stops on L-BFGS-B's
relative-decrease test (``rel_tol``), on a failed line search, or at ``max_iters``. Ascent is
monotone, so the last iterate is the best. A fit that never took a step has **failed**: the result
repeats its centre with ``log q = +inf`` (Pathfinder's convention).
"""

from __future__ import annotations

import math
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..optim.lbfgs import _two_loop
from .pathfinder import _Hist, _factors, _lbfgs_push, _line_search, _normal, _recover

__all__ = ["DADVIResult", "dadvi", "DADVI_DEFAULTS", "dadvi_options", "STOP_REASONS",
           "READOUTS", "lr_memory_gb"]

#: Defaults: ``K = 5`` draws (as Pathfinder's ELBO), Pathfinder's L-BFGS settings (history 6,
#: Wolfe constants, line-search cap), 5000 iterations, ``M = 100`` draws from the result.
#: ``max_iters`` 5000 (Pathfinder's 1000 until the cluster study, 2026-11-24): at 1000 every fit
#: on cscm and reg_horseshoe stopped at the cap, and the exact read-out needs a converged fit;
#: at 5000 cscm_observed converges 4/4 and its exact read-out gives Pareto k 0.63 (3.2 at 1000),
#: and diamonds 8/8 (1/8 at 1000), for a fit that costs a fraction of the Hessian.
#: ``rel_tol`` 1e-9 (about L-BFGS-B's factr 1e7) from the traces (2026-11-22, writeups/dadvi_init):
#: 1e-6 stops 8-116 nats short on kilpisjarvi, irt and hmm; 1e-13 costs 1.5x the iterations for
#: <= 17 nats, and diamonds never reaches 1e-9 within 1000. ``covariance="lr"`` (the exact
#: Hessian): L-BFGS's estimate underestimates the variances (start lp percentile 0.97-1.00 in high
#: d); the exact sandwich's draws are posterior-like (0.13-0.71), Pareto k drops (bci 4.3 -> 0.6),
#: downstream is unchanged, for 2.3-3.3x the fit's gradient count (+16-38% wall time).
#: The exact read-out is formed only for a fit that converged (stopped on ``rel_tol``): elsewhere
#: its Hessian is not negative definite as a rule (cluster study: the read-out fell back in 125
#: of the 132 unconverged large-target fits, after 1-5 min and 3-6 GB each), so ``"lr_lbfgs"``
#: is drawn from without forming it (``lr_skipped``). ``lr_max_gb``: the memory budget of the
#: exact read-out, in GiB, estimated by :func:`lr_memory_gb`; above it ``"lr"`` uses
#: ``"lr_lbfgs"`` without forming the Hessian (``lr_capped``). 1 GiB is n = 2364 in float64 and 3344 in float32; time does not
#: bind (about a minute at n = 5019 on 32 CPUs). ``None``: no cap. ``init_log_sd``: the start's
#: log scale ``omega0`` (lowered further if ``F`` is not finite there).
DADVI_DEFAULTS = dict(elbo_draws=5, standardize=False, covariance="lr", history=6,
                      max_iters=5000, rel_tol=1e-9, c1=1e-4, c2=0.9, max_halvings=30, draws=100,
                      lr_max_gb=1.0, init_log_sd=0.0)

#: The exact read-out's extra peak memory, in units of its ``2n x 2n`` Hessian: measured 4.6-6.2
#: at n = 5019-8000 (bci and correlated-Gaussian targets, float32 and float64, 8 and 32 CPUs;
#: 7.0 at n = 9819 on 384 CPUs), against 1 for the matrix alone: the Hessian, its symmetrised
#: and negated copies, the Cholesky factor and the sandwich's ``2n x n`` solves are live together.
_LR_MEMORY_FACTOR = 6.0


def lr_memory_gb(n: int, dtype=float) -> float:
    """Estimated extra peak memory (GiB) of the exact ``"lr"`` read-out over ``n`` coordinates:
    :data:`_LR_MEMORY_FACTOR` dense ``2n x 2n`` matrices of ``dtype`` (``float``: the precision
    in force; an explicit dtype is taken as given, whatever the x64 setting)."""
    itemsize = (jnp.asarray(0.0).dtype if dtype is float else jnp.dtype(dtype)).itemsize
    return _LR_MEMORY_FACTOR * (2.0 * n) ** 2 * itemsize / 2.0 ** 30


#: ``DADVIResult.stop`` codes.
STOP_REASONS = ("running", "tolerance", "line search", "max_iters", "no finite start")

_SHRINKS = 10                     # start-scale reductions tried for a non-finite F
_SHRINK = math.log(4.0)


class DADVIResult(NamedTuple):
    """One fixed-draw fit.

    ``draws`` (M, n) are from the read-out normal, with their ``log_p`` and ``log_q``. ``mean`` and
    ``log_sd`` are the fitted ``q``; ``center`` is the realised centroid ``mean + sd * gbar``.
    ``elbo`` is the objective ``F`` at the fit (in-sample, so optimistic). ``stop`` indexes
    :data:`STOP_REASONS`. ``n_grad`` counts log-density gradients (``K`` per evaluation of
    ``F``, plus two per Hessian-vector product of ``F`` for the exact ``"lr"`` read-out),
    ``n_value`` value-only evaluations; ``omega0`` is the start's log scale, ``readout`` indexes
    :data:`READOUTS` with the read-out drawn from, and ``lr_failed`` whether that is a fallback.
    ``lr_capped`` is whether ``"lr"`` was replaced by ``"lr_lbfgs"`` because the Hessian's
    memory would exceed ``lr_max_gb`` (a configured choice, not a failure); ``lr_skipped`` whether
    the Hessian was not formed because the fit did not converge (a fallback, so ``lr_failed`` is
    set too). A failed fit (no step taken) repeats its centre with ``log_q = +inf``.
    """
    draws: Array
    log_p: Array
    log_q: Array
    mean: Array
    log_sd: Array
    center: Array
    elbo: Array
    n_iter: Array
    stop: Array
    n_grad: Array
    n_value: Array
    omega0: Array
    readout: Array
    lr_failed: Array
    failed: Array
    lr_capped: Array
    lr_skipped: Array


class _Carry(NamedTuple):
    eta: Array            # (2n,): [mu, omega]
    f: Array
    g: Array
    hist: _Hist
    alpha: Array          # Algorithm 3's diagonal over the 2n coordinates
    PS: Array
    PZ: Array
    pmask: Array
    k: Array              # iterations (steps) taken
    n_eval: Array         # evaluations of F
    stop: Array           # 0 while running, else an index into STOP_REASONS
    shrinks: Array        # start-scale reductions made


def _pattern(key: Array, count: int, n: int, dtype, standardize: bool) -> Array:
    """The fixed draws ``(K, n)``; standardised per coordinate (mean 0, sd 1) on request."""
    g = jax.random.normal(key, (count, n), dtype)
    if standardize:
        g = g - jnp.mean(g, axis=0)
        g = g / jnp.sqrt(jnp.mean(g * g, axis=0))
    return g


def _objective(logp: Callable, g: Array) -> Callable:
    """``F(eta)``: the fixed-draw ELBO (non-finite wherever any point is)."""
    K, n = g.shape
    const = 0.5 * jnp.mean(jnp.sum(g * g, axis=1)) + 0.5 * n * math.log(2.0 * math.pi)
    logp_many = jax.vmap(logp)

    def F(eta):
        mu, omega = eta[:n], eta[n:]
        return jnp.mean(logp_many(mu + jnp.exp(omega) * g)) + jnp.sum(omega) + const
    return F


def _init(vg: Callable, x0: Array, history: int, init_log_sd: float = 0.0) -> _Carry:
    """The start ``(x0, omega0)``, ``omega0 = init_log_sd`` lowered until ``F`` is finite there."""
    dtype, n = x0.dtype, x0.shape[0]

    def finite(f, g):
        return jnp.isfinite(f) & jnp.all(jnp.isfinite(g))

    def cond(s):
        i, _, f, g = s
        return ~finite(f, g) & (i < _SHRINKS)

    def body(s):
        i, omega, _, _ = s
        omega = omega - _SHRINK
        f, g = vg(jnp.concatenate([x0, omega]))
        return i + 1, omega, f, g

    omega = jnp.full((n,), init_log_sd, dtype)
    f, g = vg(jnp.concatenate([x0, omega]))
    i, omega, f, g = jax.lax.while_loop(cond, body, (jnp.asarray(0), omega, f, g))
    zeros = jnp.zeros((history, 2 * n), dtype)
    return _Carry(eta=jnp.concatenate([x0, omega]), f=f, g=g,
                  hist=_Hist(zeros, zeros, jnp.zeros((history,), dtype), jnp.asarray(1.0, dtype)),
                  alpha=jnp.ones((2 * n,), dtype), PS=zeros, PZ=zeros,
                  pmask=jnp.zeros((history,), dtype), k=jnp.asarray(0), n_eval=1 + i,
                  stop=jnp.where(finite(f, g), 0, 4), shrinks=i)


def _step(vg: Callable, c: _Carry, *, history: int, max_iters: int, rel_tol: float, c1: float,
          c2: float, max_halvings: int) -> _Carry:
    """One L-BFGS iteration on ``F`` (a no-op once stopped)."""
    d = _two_loop(-c.g, c.hist, history)
    d = jnp.where(jnp.dot(c.g, d) > 0, d, c.g)
    # With no curvature pair held (the first step, or every pair so far rejected) d is the raw
    # gradient, whose scale can be anything: L-BFGS-B's unit-length step. (bci: |grad F| ~ 1e11
    # after the first pair was rejected, and 30 halvings of lambda = 1 never reached a finite F.)
    norm = jnp.sqrt(jnp.dot(d, d))
    empty = jnp.all(c.hist.rho == 0)
    d = jnp.where(empty & (norm > 1.0), d / jnp.where(norm > 0, norm, 1.0), d)
    ok, eta, f, g, trials = _line_search(vg, c.eta, c.f, c.g, d, c1=c1, c2=c2,
                                         max_halvings=max_halvings)
    s, z = eta - c.eta, c.g - g
    hist = _lbfgs_push(c.hist, s, z, ok)
    alpha, PS, PZ, pmask = _recover(c.alpha, c.PS, c.PZ, c.pmask, s, z, ok)
    k = c.k + ok
    rel = jnp.abs(f - c.f) / jnp.maximum(jnp.maximum(jnp.abs(c.f), jnp.abs(f)), 1.0)
    stop = jnp.where(~ok, 2, jnp.where(rel < rel_tol, 1, jnp.where(k >= max_iters, 3, 0)))
    new = _Carry(eta=jnp.where(ok, eta, c.eta), f=jnp.where(ok, f, c.f),
                 g=jnp.where(ok, g, c.g), hist=hist, alpha=alpha, PS=PS, PZ=PZ, pmask=pmask,
                 k=k, n_eval=c.n_eval + trials, stop=stop, shrinks=c.shrinks)
    return jax.tree_util.tree_map(lambda a, b: jnp.where(c.stop == 0, a, b), new, c)


#: The read-outs, as ``DADVIResult.readout`` codes (the one actually drawn from).
READOUTS = ("diag", "lr_lbfgs", "lr")
_HVP_BATCH = 32                   # Hessian-vector products per vmapped batch (memory bound)


def _hessian(F: Callable, eta: Array) -> Array:
    """The dense Hessian of ``F`` at ``eta``: ``2n`` Hessian-vector products (forward over
    reverse), ``_HVP_BATCH`` at a time, symmetrised."""
    grad = jax.grad(F)
    basis = jnp.eye(eta.shape[0], dtype=eta.dtype)
    H = jax.lax.map(lambda e: jax.jvp(grad, (eta,), (e,))[1], basis, batch_size=_HVP_BATCH)
    return 0.5 * (H + H.T)


def _lr_exact(F: Callable, eta: Array, v: Array):
    """Cholesky factor and log-determinant of ``J^T (-hess F)^-1 J``, ``J = [I; diag(v)]`` (NaN
    if ``-hess F`` or the sandwich is not positive definite)."""
    n = v.shape[0]
    L = jnp.linalg.cholesky(-_hessian(F, eta))
    J = jnp.concatenate([jnp.eye(n, dtype=eta.dtype), jnp.diag(v)])
    X = jax.scipy.linalg.cho_solve((L, True), J)
    sigma = J.T @ X
    Ls = jnp.linalg.cholesky(0.5 * (sigma + sigma.T))
    return Ls, 2.0 * jnp.sum(jnp.log(jnp.diagonal(Ls)))


def _readout(logp: Callable, c: _Carry, g: Array, key: Array, *, covariance: str, draws: int,
             F: Callable | None = None):
    """``draws`` from the read-out normal: ``(phi, log_p, log_q, center, readout)``.

    ``covariance="lr"`` (needs the objective ``F``) falls back to ``"lr_lbfgs"``, and that to
    ``"diag"``, when its covariance is not positive definite; ``readout`` indexes
    :data:`READOUTS` with the one used. The exact Hessian is formed only if the fit converged
    (``c.stop`` is ``"tolerance"``), under a ``lax.cond``, so an unconverged fit never pays for
    it; it then falls back to ``"lr_lbfgs"`` as for a non-positive-definite one."""
    dtype, n = c.eta.dtype, g.shape[1]
    mu, omega = c.eta[:n], c.eta[n:]
    sd = jnp.exp(omega)
    v = sd * jnp.mean(g, axis=0)
    center = mu + v
    u = jax.random.normal(key, (draws, n), dtype)
    const = 0.5 * n * math.log(2.0 * math.pi)
    phi = mu + sd * u
    log_q = -0.5 * jnp.sum(u * u, axis=1) - jnp.sum(omega) - const
    readout = jnp.asarray(0)

    def take(ok, code, phi_new, log_q_new):
        return (jnp.where(ok, phi_new, phi), jnp.where(ok, log_q_new, log_q),
                jnp.where(ok, code, readout))

    if covariance in ("lr_lbfgs", "lr"):
        beta, gamma = _factors(c.alpha, c.PS, c.PZ, c.pmask)
        alpha = c.alpha[:n] + v * v * c.alpha[n:]
        beta = beta[:n] + v[:, None] * beta[n:]
        draw, logdet = _normal(center, alpha, beta, gamma)
        phi_lr = draw(u)
        ok = jnp.isfinite(logdet) & jnp.all(jnp.isfinite(phi_lr))
        phi, log_q, readout = take(ok, 1, phi_lr,
                                   -0.5 * (logdet + jnp.sum(u * u, axis=1)) - const)
    if covariance == "lr":
        Ls, logdet = jax.lax.cond(
            c.stop == STOP_REASONS.index("tolerance"), lambda e: _lr_exact(F, e, v),
            lambda e: (jnp.zeros((n, n), dtype), jnp.asarray(jnp.nan, dtype)), c.eta)
        phi_x = center + u @ Ls.T
        ok = jnp.isfinite(logdet) & jnp.all(jnp.isfinite(phi_x))
        phi, log_q, readout = take(ok, 2, phi_x, -0.5 * (logdet + jnp.sum(u * u, axis=1)) - const)
    log_p = jax.vmap(logp)(phi)
    log_p = jnp.where(jnp.isfinite(log_p), log_p, -jnp.inf)
    return phi, log_p, log_q, center, readout


def _fit(logp: Callable, x0: Array, key: Array, *, elbo_draws: int, standardize: bool,
         covariance: str, history: int, max_iters: int, rel_tol: float, c1: float, c2: float,
         max_halvings: int, draws: int, lr_max_gb: float | None = None,
         init_log_sd: float = 0.0) -> DADVIResult:
    """One fit, traceable."""
    n = x0.shape[0]
    capped = (covariance == "lr" and lr_max_gb is not None
              and lr_memory_gb(n, x0.dtype) > lr_max_gb)
    if capped:                            # static: the Hessian is never traced
        covariance = "lr_lbfgs"
    g_key, draw_key = jax.random.split(key)
    g = _pattern(g_key, elbo_draws, n, x0.dtype, standardize)
    F = _objective(logp, g)
    vg = jax.value_and_grad(F)
    step = dict(history=history, max_iters=max_iters, rel_tol=rel_tol, c1=c1, c2=c2,
                max_halvings=max_halvings)
    c = jax.lax.while_loop(lambda c: c.stop == 0, lambda c: _step(vg, c, **step),
                           _init(vg, x0, history, init_log_sd))
    phi, log_p, log_q, center, readout = _readout(logp, c, g, draw_key, covariance=covariance,
                                                  draws=draws, F=F)
    # an exact Hessian is 2n Hessian-vector products, each about two gradients of F; it is formed
    # only for a converged fit
    converged = c.stop == STOP_REASONS.index("tolerance")
    skipped = jnp.asarray(covariance == "lr") & ~converged
    n_hess = jnp.where(converged, 2 * (2 * n) * elbo_draws, 0) if covariance == "lr" else 0
    failed = c.k == 0
    phi = jnp.where(failed, jnp.broadcast_to(center, phi.shape), phi)
    log_q = jnp.where(failed, jnp.inf, log_q)
    log_p = jnp.where(failed, logp(center), log_p)
    return DADVIResult(draws=phi, log_p=log_p, log_q=log_q, mean=c.eta[:n], log_sd=c.eta[n:],
                       center=center, elbo=c.f, n_iter=c.k, stop=c.stop,
                       n_grad=c.n_eval * elbo_draws + n_hess, n_value=jnp.asarray(draws + 1),
                       omega0=init_log_sd - _SHRINK * c.shrinks, readout=readout,
                       lr_failed=readout < READOUTS.index(covariance), failed=failed,
                       lr_capped=jnp.asarray(capped), lr_skipped=skipped)


def dadvi_options(kwargs: dict) -> dict:
    """:data:`DADVI_DEFAULTS` updated with ``kwargs``, validated (unknown keys raise)."""
    opts = dict(DADVI_DEFAULTS)
    unknown = sorted(set(kwargs) - set(opts))
    if unknown:
        raise TypeError(f"unknown DADVI option(s) {unknown} (it takes {sorted(opts)})")
    opts.update(kwargs)
    for name, low in (("elbo_draws", 2), ("history", 1), ("max_iters", 1),
                      ("max_halvings", 1), ("draws", 1)):
        v = opts[name]
        if not (isinstance(v, (int, np.integer)) and not isinstance(v, bool) and v >= low):
            extra = (" (one draw makes the objective unbounded: a point at the mode with an "
                     "infinite scale)") if name == "elbo_draws" else ""
            raise ValueError(f"DADVI option {name} must be an integer >= {low}, got {v!r}{extra}")
        opts[name] = int(v)
    if not isinstance(opts["standardize"], (bool, np.bool_)):
        raise ValueError(f"DADVI option standardize must be a bool, got {opts['standardize']!r}")
    opts["standardize"] = bool(opts["standardize"])
    if opts["covariance"] not in READOUTS:
        raise ValueError(f"DADVI option covariance must be one of {list(READOUTS)}, got "
                         f"{opts['covariance']!r}")
    cap = opts["lr_max_gb"]
    if cap is not None:
        if not (isinstance(cap, (int, float, np.integer, np.floating))
                and not isinstance(cap, bool) and math.isfinite(cap) and cap > 0):
            raise ValueError(f"DADVI option lr_max_gb must be a positive number (GiB) or None, "
                             f"got {cap!r}")
        opts["lr_max_gb"] = float(cap)
    w0 = opts["init_log_sd"]
    if not (isinstance(w0, (int, float, np.integer, np.floating)) and not isinstance(w0, bool)
            and math.isfinite(w0)):
        raise ValueError(f"DADVI option init_log_sd must be a finite number, got {w0!r}")
    opts["init_log_sd"] = float(w0)
    rt = opts["rel_tol"]
    if not (isinstance(rt, (int, float, np.integer, np.floating)) and not isinstance(rt, bool)
            and math.isfinite(rt) and rt >= 0):
        raise ValueError(f"DADVI option rel_tol must be finite and >= 0, got {rt!r}")
    opts["rel_tol"] = float(rt)
    if not 0 < opts["c1"] < opts["c2"] < 1:
        raise ValueError(f"DADVI Wolfe constants need 0 < c1 < c2 < 1, got c1={opts['c1']!r}, "
                         f"c2={opts['c2']!r}")
    return opts


def dadvi(logp: Callable[[Array], Array], x0, key, **options) -> DADVIResult:
    """Fixed-draw mean-field VI (DADVI) by L-BFGS on a flat target ``logp: (n,) -> ()`` from ``x0``.

    ``options`` are those of :data:`DADVI_DEFAULTS`. ``logp`` may return ``-inf``/NaN off its
    support. Compiled once per call (``logp`` is closed over), so call it once per fit rather than
    in a loop.
    """
    opts = dadvi_options(options)
    x0 = jnp.asarray(x0, float)
    return jax.jit(lambda x, k: _fit(logp, x, k, **opts))(x0, key)
