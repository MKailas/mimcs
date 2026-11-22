"""Pathfinder: quasi-Newton variational approximations along an L-BFGS path.

Zhang, Carpenter, Gelman & Vehtari (2022), "Pathfinder: Parallel quasi-Newton variational
inference", JMLR 23(306). Starting from a draw of an initial distribution, L-BFGS climbs the log
density; at every iterate a normal approximation is built from the path's own curvature (a
diagonal-plus-low-rank inverse-Hessian estimate), its ELBO estimated by Monte Carlo, and the
best one kept. Multi-path Pathfinder runs several independent paths and importance-resamples the
pooled draws with Pareto smoothing.

Everything follows the paper, with its defaults (§3.1): ``history`` J = 6, ``max_iters`` L = 1000,
``rel_tol`` 1e-13, ``elbo_draws`` K = 5, ``draws`` M = 100, Wolfe constants (1e-4, 0.9); multi-path
I = 20 paths, R = 100 resampled draws. The departures, each deliberate:

* **Two misprints are not copied.** Algorithm 3 line 5 prints the pair filter as
  ``s.z < eps |z|^2``; the text ("filters out candidates whose gradient changes 10^12 times more
  than the position") means ``s.z >= eps |z|^2``, which is what is used. Algorithm 7 line 14
  prints the L-BFGS initial scale as ``|z|^2 / s.z``, the *Hessian* scale; the inverse-Hessian
  scale ``s.z / |z|^2`` is used (it is what ``mimcs.optim``'s L-BFGS already does).
* **The ELBO is evaluated as the path is made**, not after it, keeping only the best iterate's
  factors: the same selection in O(J n) memory instead of O(L n).
* **The convergence test is L-BFGS-B's**, ``|f_l - f_{l+1}| / max(|f_l|, |f_{l+1}|, 1) <
  rel_tol``, i.e. what the paper's own implementation (R's ``optim`` L-BFGS-B, its footnote 3)
  computes. Algorithm 7 prints the denominator as ``|log p(theta_l)|`` alone, which never
  converges when the log density at the mode is near 0 (measured: every path of a 10-d Gaussian
  with ``max log p = 0`` ran all 1000 iterations).
* **The line search is capped** at ``max_halvings`` trial steps (the paper's has no cap). If no
  trial satisfies both Wolfe conditions, the first that satisfies the sufficient-increase one is
  taken; if none does, the path ends.
* **Small dimensions are handled densely.** The thin-QR factorisation of Algorithm 4 needs
  ``n > 2J``; when ``2J >= n`` the covariance is formed and Cholesky-factored directly (it is at
  most ``2J x 2J`` then), which is the same distribution.

Sign conventions are the paper's: ``s = theta_l - theta_{l-1}``, ``z = grad log p(theta_{l-1}) -
grad log p(theta_l)`` (the change in the gradient of ``-log p``). The covariance factors were
checked against the dense recursive BFGS inverse update (1e-16) and the sampler's ``log q``
against a dense normal density (1e-14) before this module was written.
"""

from __future__ import annotations

import math
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..optim.lbfgs import _two_loop
from .psis import psis, resample

__all__ = ["PathResult", "MultiPathResult", "single_path", "multi_path", "PATHFINDER_DEFAULTS",
           "MULTIPATH_DEFAULTS", "pathfinder_options"]

#: The paper's single-path defaults (§3.1 and Algorithm 7), plus the line-search cap and
#: ``patience``: stop a path once that many iterations pass without an ELBO improvement (``None``,
#: the paper's behaviour, never stops it on the ELBO).
PATHFINDER_DEFAULTS = dict(history=6, max_iters=1000, rel_tol=1e-13, elbo_draws=5, draws=100,
                           c1=1e-4, c2=0.9, max_halvings=30, patience=None)
#: The paper's multi-path defaults: I = 20 paths, R = 100 resampled draws (M = 100 per path).
MULTIPATH_DEFAULTS = dict(num_paths=20, resample=100)

_LBFGS_EPS = 2.2e-16       # Algorithm 7: curvature threshold for the optimiser's own history
_RECOVER_EPS = 1e-12       # Algorithm 3: the (stricter) filter for the approximation's history


class PathResult(NamedTuple):
    """One Pathfinder path. Under :func:`multi_path` every field gains a leading path axis.

    ``draws`` (M, n) are from the ELBO-maximising normal approximation, with their ``log_p`` and
    ``log_q``. A failed path (no step taken, or no finite ELBO anywhere along it) instead repeats
    its last iterate with ``log_q = +inf``, so it carries zero importance weight (the paper's
    convention). ``l_star`` is the selected iteration (0 = none), ``n_iter`` the L-BFGS iterations
    taken, ``n_grad`` / ``n_value`` the log-density evaluations with and without a gradient.
    """
    draws: Array
    log_p: Array
    log_q: Array
    elbo: Array
    l_star: Array
    n_iter: Array
    n_grad: Array
    n_value: Array
    failed: Array
    last: Array


class MultiPathResult(NamedTuple):
    """Multi-path Pathfinder: the per-path results, and ``resample`` PSIS-resampled draws.

    ``index`` points into the pooled ``(num_paths * draws)`` draws (path-major); ``khat`` is the
    Pareto k of the pooled importance weights, ``distinct`` the number of distinct draws among the
    resampled ones.
    """
    paths: PathResult
    draws: Array
    log_p: Array
    index: np.ndarray
    khat: float
    distinct: int


class _Hist(NamedTuple):
    """A J-pair ring buffer in the layout ``mimcs.optim.lbfgs._two_loop`` reads."""
    S: Array
    Y: Array
    rho: Array
    gamma: Array


class _Carry(NamedTuple):
    x: Array
    f: Array
    g: Array
    hist: _Hist
    alpha: Array          # Algorithm 3's diagonal
    PS: Array             # (J, n) its pairs, oldest first
    PZ: Array
    pmask: Array          # (J,) 1 for a held pair
    best_elbo: Array
    best_x: Array
    best_g: Array
    best_alpha: Array
    best_PS: Array
    best_PZ: Array
    best_pmask: Array
    l_star: Array
    k: Array
    n_grad: Array
    n_value: Array
    done: Array
    key: Array


def _push(buf, row):
    return jnp.roll(buf, -1, axis=0).at[-1].set(row)


def _factors(alpha, PS, PZ, pmask):
    """``beta`` (n, 2J) and ``gamma`` (2J, 2J) of eq. (2) for the held pairs (others zeroed)."""
    J = PS.shape[0]
    S = PS * pmask[:, None]
    Z = PZ * pmask[:, None]
    SZ = S @ Z.T                                             # [i, j] = S_i . Z_j
    E = jnp.triu(SZ)
    diag = jnp.where(pmask > 0, jnp.diagonal(SZ), 1.0)       # padding keeps E invertible
    E = E.at[jnp.arange(J), jnp.arange(J)].set(diag)
    eta = jnp.where(pmask > 0, jnp.diagonal(SZ), 0.0)
    Einv = jax.scipy.linalg.solve_triangular(E, jnp.eye(J, dtype=E.dtype), lower=False)
    beta = jnp.concatenate([(alpha[None, :] * Z).T, S.T], axis=1)
    lower_right = Einv.T @ (jnp.diag(eta) + (Z * alpha[None, :]) @ Z.T) @ Einv
    gamma = jnp.block([[jnp.zeros((J, J), E.dtype), -Einv], [-Einv.T, lower_right]])
    return beta, gamma


def _normal(mu, alpha, beta, gamma):
    """``N(mu, diag(alpha) + beta gamma beta^T)`` (Algorithm 4): ``(draw(u) -> phi, log|Sigma|)``.

    Thin-QR factored when ``n > 2J``, dense otherwise (the covariance is then at most 2J x 2J)."""
    n, J2 = mu.shape[0], beta.shape[1]
    if J2 >= n:
        sigma = jnp.diag(alpha) + beta @ gamma @ beta.T
        L = jnp.linalg.cholesky(sigma)
        logdet = 2.0 * jnp.sum(jnp.log(jnp.diagonal(L)))
        return (lambda u: mu + u @ L.T), logdet
    sqrt_a = jnp.sqrt(alpha)
    Q, R = jnp.linalg.qr(beta / sqrt_a[:, None], mode="reduced")
    Lt = jnp.linalg.cholesky(jnp.eye(J2, dtype=mu.dtype) + R @ gamma @ R.T)
    logdet = jnp.sum(jnp.log(alpha)) + 2.0 * jnp.sum(jnp.log(jnp.abs(jnp.diagonal(Lt))))
    LtI = Lt - jnp.eye(J2, dtype=mu.dtype)
    return (lambda u: mu + sqrt_a * ((u @ Q) @ LtI.T @ Q.T + u)), logdet


def _approximation(x, g, alpha, PS, PZ, pmask):
    """The normal approximation at ``x``: ``(mu, draw(u) -> phi, log|Sigma|)`` (Algorithm 4)."""
    beta, gamma = _factors(alpha, PS, PZ, pmask)
    mu = x + alpha * g + beta @ (gamma @ (beta.T @ g))
    draw, logdet = _normal(mu, alpha, beta, gamma)
    return mu, draw, logdet


def _draw(key, count, x, g, alpha, PS, PZ, pmask):
    """``count`` draws from the approximation at ``x`` and their ``log q``."""
    n = x.shape[0]
    _, draw, logdet = _approximation(x, g, alpha, PS, PZ, pmask)
    u = jax.random.normal(key, (count, n), x.dtype)
    log_q = -0.5 * (logdet + jnp.sum(u * u, axis=1) + n * math.log(2.0 * math.pi))
    return draw(u), log_q


def _finite_or(value, fill):
    return jnp.where(jnp.isfinite(value), value, fill)


def _line_search(vg: Callable, x, f, g, d, *, c1: float, c2: float, max_halvings: int):
    """Halve lambda from 1 until both Wolfe conditions hold (ascent on ``f``); fall back to the
    first step with sufficient increase. ``vg`` returns value and gradient. Returns ``(ok, x_new,
    f_new, g_new, trials)``."""
    dtype = x.dtype
    gd = jnp.dot(g, d)

    def cond(s):
        i, wolfe = s[0], s[1]
        return (~wolfe) & (i < max_halvings)

    def body(s):
        i, _, have_a, xa, fa, ga, _, _, _ = s
        lam = jnp.asarray(0.5, dtype) ** i
        xt = x + lam * d
        ft, gt = vg(xt)
        ok = jnp.isfinite(ft) & jnp.all(jnp.isfinite(gt))
        armijo = ok & (ft >= f + c1 * lam * gd)
        wolfe = armijo & (jnp.dot(gt, d) <= c2 * gd)
        first_a = armijo & ~have_a
        xa = jnp.where(first_a, xt, xa)
        fa = jnp.where(first_a, ft, fa)
        ga = jnp.where(first_a, gt, ga)
        return (i + 1, wolfe, have_a | armijo, xa, fa, ga, xt, ft, gt)

    s0 = (jnp.asarray(0), jnp.asarray(False), jnp.asarray(False), x, f, g, x, f, g)
    i, wolfe, have_a, xa, fa, ga, xw, fw, gw = jax.lax.while_loop(cond, body, s0)
    x_new = jnp.where(wolfe, xw, xa)
    f_new = jnp.where(wolfe, fw, fa)
    g_new = jnp.where(wolfe, gw, ga)
    return wolfe | have_a, x_new, f_new, g_new, i


def _lbfgs_push(hist: _Hist, s, z, ok) -> _Hist:
    """The optimiser's own history (Algorithm 7, lines 10-14): keep ``(s, z)`` if it is a step
    with ``s.z > eps |z|^2``."""
    sz, zz = jnp.dot(s, z), jnp.dot(z, z)
    keep = ok & (sz > _LBFGS_EPS * zz)
    rho = jnp.where(keep, 1.0 / jnp.where(keep, sz, 1.0), 0.0)
    return _Hist(S=jnp.where(keep, _push(hist.S, s), hist.S),
                 Y=jnp.where(keep, _push(hist.Y, z), hist.Y),
                 rho=jnp.where(keep, _push(hist.rho, rho), hist.rho),
                 gamma=jnp.where(keep & (zz > 0), sz / jnp.where(zz > 0, zz, 1.0), hist.gamma))


def _recover(alpha, PS, PZ, pmask, s, z, ok):
    """Algorithm 3: update the diagonal ``alpha`` and the approximation's own pair history."""
    dtype = alpha.dtype
    sz, zz = jnp.dot(s, z), jnp.dot(z, z)
    a = jnp.dot(z, alpha * z)
    cc = jnp.dot(s, s / alpha)
    safe_sz = jnp.where(sz != 0, sz, 1.0)
    new_alpha = 1.0 / (a / (safe_sz * alpha) + z * z / safe_sz
                       - a * s * s / (safe_sz * cc * alpha * alpha))
    recover = (ok & (sz >= _RECOVER_EPS * zz) & (sz > 0)
               & jnp.all(jnp.isfinite(new_alpha)) & jnp.all(new_alpha > 0))
    return (jnp.where(recover, new_alpha, alpha), jnp.where(recover, _push(PS, s), PS),
            jnp.where(recover, _push(PZ, z), PZ),
            jnp.where(recover, _push(pmask, jnp.asarray(1.0, dtype)), pmask))


def _path(logp: Callable, x0: Array, key: Array, *, history: int, max_iters: int, rel_tol: float,
          elbo_draws: int, draws: int, c1: float, c2: float, max_halvings: int,
          patience: int | None = None) -> PathResult:
    """One path, traceable (``multi_path`` vmaps it)."""
    dtype = x0.dtype
    n, J = x0.shape[0], history
    vg = jax.value_and_grad(logp)
    logp_many = jax.vmap(logp)
    neg_inf = jnp.asarray(-jnp.inf, dtype)

    f0, g0 = vg(x0)
    zeros_J = jnp.zeros((J, n), dtype)
    init = _Carry(
        x=x0, f=f0, g=g0,
        hist=_Hist(zeros_J, zeros_J, jnp.zeros((J,), dtype), jnp.asarray(1.0, dtype)),
        alpha=jnp.ones((n,), dtype), PS=zeros_J, PZ=zeros_J, pmask=jnp.zeros((J,), dtype),
        best_elbo=neg_inf, best_x=x0, best_g=g0, best_alpha=jnp.ones((n,), dtype),
        best_PS=zeros_J, best_PZ=zeros_J, best_pmask=jnp.zeros((J,), dtype),
        l_star=jnp.asarray(0), k=jnp.asarray(0), n_grad=jnp.asarray(1), n_value=jnp.asarray(0),
        done=~(jnp.isfinite(f0) & jnp.all(jnp.isfinite(g0))), key=key)

    def body(c: _Carry) -> _Carry:
        # Ascent direction H grad log p, from the two-loop recursion on -log p.
        d = _two_loop(-c.g, c.hist, J)
        d = jnp.where(jnp.dot(c.g, d) > 0, d, c.g)
        ok, x_new, f_new, g_new, trials = _line_search(vg, c.x, c.f, c.g, d, c1=c1, c2=c2,
                                                       max_halvings=max_halvings)
        n_grad = c.n_grad + trials
        s = x_new - c.x
        z = c.g - g_new
        hist = _lbfgs_push(c.hist, s, z, ok)
        alpha, PS, PZ, pmask = _recover(c.alpha, c.PS, c.PZ, c.pmask, s, z, ok)

        # ELBO of the approximation at the new iterate (Algorithm 6), K Monte Carlo draws.
        key, sub = jax.random.split(c.key)
        phi, log_q = _draw(sub, elbo_draws, x_new, g_new, alpha, PS, PZ, pmask)
        log_p = _finite_or(logp_many(phi), neg_inf)
        elbo = jnp.mean(log_p - log_q)
        elbo = jnp.where(ok & jnp.isfinite(elbo), elbo, neg_inf)
        better = elbo > c.best_elbo
        k = c.k + 1
        rel = jnp.abs(f_new - c.f) / jnp.maximum(jnp.maximum(jnp.abs(c.f), jnp.abs(f_new)), 1.0)
        done = (~ok) | (rel < rel_tol) | (k >= max_iters)
        l_star = jnp.where(better, k, c.l_star)
        if patience is not None:              # no ELBO improvement in `patience` iterations
            done = done | (k - l_star >= patience)

        pick = lambda new, old: jnp.where(better, new, old)
        return _Carry(
            x=jnp.where(ok, x_new, c.x), f=jnp.where(ok, f_new, c.f),
            g=jnp.where(ok, g_new, c.g), hist=hist, alpha=alpha, PS=PS, PZ=PZ, pmask=pmask,
            best_elbo=pick(elbo, c.best_elbo), best_x=pick(x_new, c.best_x),
            best_g=pick(g_new, c.best_g), best_alpha=pick(alpha, c.best_alpha),
            best_PS=pick(PS, c.best_PS), best_PZ=pick(PZ, c.best_PZ),
            best_pmask=pick(pmask, c.best_pmask), l_star=l_star,
            k=jnp.where(ok, k, c.k), n_grad=n_grad, n_value=c.n_value + elbo_draws,
            done=done, key=key)

    c = jax.lax.while_loop(lambda c: ~c.done, body, init)

    failed = ~jnp.isfinite(c.best_elbo)
    key, sub = jax.random.split(c.key)
    phi, log_q = _draw(sub, draws, c.best_x, c.best_g, c.best_alpha, c.best_PS, c.best_PZ,
                       c.best_pmask)
    log_p = _finite_or(logp_many(phi), neg_inf)
    # Failure: the last iterate, with log q = +inf (zero importance weight).
    phi = jnp.where(failed, jnp.broadcast_to(c.x, phi.shape), phi)
    log_q = jnp.where(failed, jnp.inf, log_q)
    log_p = jnp.where(failed, c.f, log_p)
    return PathResult(draws=phi, log_p=log_p, log_q=log_q, elbo=c.best_elbo, l_star=c.l_star,
                      n_iter=c.k, n_grad=c.n_grad, n_value=c.n_value + draws, failed=failed,
                      last=c.x)


def pathfinder_options(kwargs: dict) -> dict:
    """The paper's defaults updated with ``kwargs``, validated (unknown keys raise)."""
    opts = dict(PATHFINDER_DEFAULTS)
    unknown = sorted(set(kwargs) - set(opts))
    if unknown:
        raise TypeError(f"unknown Pathfinder option(s) {unknown} (it takes {sorted(opts)})")
    opts.update(kwargs)
    for name in ("history", "max_iters", "elbo_draws", "draws", "max_halvings", "patience"):
        v = opts[name]
        if name == "patience" and v is None:
            continue
        if not (isinstance(v, (int, np.integer)) and v >= 1):
            raise ValueError(f"Pathfinder option {name} must be an integer >= 1"
                             f"{' or None' if name == 'patience' else ''}, got {v!r}")
        opts[name] = int(v)
    if not (math.isfinite(opts["rel_tol"]) and opts["rel_tol"] >= 0):
        raise ValueError(f"Pathfinder option rel_tol must be finite and >= 0, got "
                         f"{opts['rel_tol']!r}")
    if not 0 < opts["c1"] < opts["c2"] < 1:
        raise ValueError(f"Pathfinder Wolfe constants need 0 < c1 < c2 < 1, got "
                         f"c1={opts['c1']!r}, c2={opts['c2']!r}")
    return opts


def single_path(logp: Callable[[Array], Array], x0, key, **options) -> PathResult:
    """Single-path Pathfinder (Algorithm 1) on a flat target ``logp: (n,) -> ()`` from ``x0``.

    ``options`` are those of :data:`PATHFINDER_DEFAULTS` (the paper's). ``logp`` may return
    ``-inf``/NaN off its support. Compiled once per call (``logp`` is closed over), so call it
    once per fit rather than in a loop.
    """
    opts = pathfinder_options(options)
    x0 = jnp.asarray(x0, float)
    return jax.jit(lambda x, k: _path(logp, x, k, **opts))(x0, key)


def multi_path(logp: Callable[[Array], Array], x0s, key, *, resample_size: int = 100,
               **options) -> MultiPathResult:
    """Multi-path Pathfinder (Algorithm 2): independent paths from the rows of ``x0s``, pooled
    and Pareto-smoothed-importance-resampled to ``resample_size`` draws (with replacement).

    The paths run vmapped in one compilation. The importance weight of a pooled draw is
    ``log p - log q`` (the paper's ``log p~ - log q~`` with the common ``log I`` cancelled); a
    failed path's ``log q = +inf`` gives its draws zero weight. Raises ``RuntimeError`` if every
    path failed.
    """
    opts = pathfinder_options(options)
    if not (isinstance(resample_size, (int, np.integer)) and resample_size >= 1):
        raise ValueError(f"resample_size must be an integer >= 1, got {resample_size!r}")
    x0s = jnp.asarray(x0s, float)
    path_key, rs_key = jax.random.split(key)
    keys = jax.random.split(path_key, x0s.shape[0])
    paths = jax.jit(jax.vmap(lambda x, k: _path(logp, x, k, **opts)))(x0s, keys)
    draws = np.asarray(paths.draws).reshape(-1, x0s.shape[1])
    log_p = np.asarray(paths.log_p).ravel()
    log_w = log_p - np.asarray(paths.log_q).ravel()
    if not np.any(np.isfinite(log_w)):
        raise RuntimeError(f"multi-path Pathfinder: all {x0s.shape[0]} paths failed (no finite "
                           f"importance weight)")
    smoothed, khat = psis(log_w)
    seed = int(jax.random.randint(rs_key, (), 0, np.iinfo(np.int32).max))
    index = resample(np.random.default_rng(seed), smoothed, int(resample_size))
    return MultiPathResult(paths=paths, draws=jnp.asarray(draws[index]),
                           log_p=jnp.asarray(log_p[index]), index=index, khat=float(khat),
                           distinct=int(len(np.unique(index))))
