"""Fixed-draw (deterministic) mean-field variational inference, by trust-region Newton-CG.

The sample-average approximation of the mean-field ELBO: draw ``K`` (``elbo_draws``) standard
normal vectors ``g_k`` once, and maximise the then deterministic objective

    F(mu, omega) = mean_k [log p(x_k) - log q(x_k)],   x_k = mu + exp(omega) * g_k,
                 = mean_k log p(x_k) + sum(omega) + mean_k |g_k|^2 / 2 + n/2 log(2 pi),

over the mean ``mu`` and log scale ``omega`` of ``q = N(mu, diag(exp(2 omega)))``, by SciPy's
trust-region Newton-CG (``trust-ncg``, the paper's optimiser and defaults) or, with
``optimizer="lbfgs"``, the same L-BFGS (and Wolfe backtracking) as Pathfinder. This is DADVI
(Giordano, Ingram & Broderick 2024, "Black box variational inference with a deterministic
objective", JMLR 25(18)), here with ``K = 5`` draws (theirs: 30). Unlike Pathfinder the
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
* ``"lr_lbfgs"``: the same sandwich with a secant estimate of ``(-hess F)^-1`` from the last
  ``history`` steps (either optimiser's), Pathfinder's diagonal-plus-low-rank ``alpha``-recovered
  factors over the ``2n`` coordinates. The sandwich
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
non-finite log density or gradient at any point makes ``F`` non-finite.

``optimizer="trust_ncg"`` (default; :func:`_tr_step`): each step minimises the exact local
quadratic model of ``-F`` (Hessian-vector products by ``jax.linearize``) over a ball of radius
``initial_trust_radius`` (1) to ``max_trust_radius`` (1000) by Steihaug's CG
(:func:`mimcs.optim.trust_ncg.steihaug_cg`, at most ``cg_max_iters``, default ``2n``, products);
it is accepted if the actual over the predicted gain exceeds ``trust_eta`` (0.15), and the radius
adapts as in SciPy. A non-finite proposal is rejected and the radius quartered. It stops on
``"tolerance"`` (``|grad F| < gtol``, SciPy's 1e-4, or an interior Newton step predicting a gain
below ``rel_tol max(|F|, 1)``), on ``"trust region"`` (no predicted gain from a boundary step, or
a collapsed radius),
or at ``max_iters`` iterations, rejected ones included. Why not L-BFGS: on bci from the uniform
start its secant pairs, all taken while the field's scale fell and its length scale rose, turned
the field's gradient into a length-scale step against that coordinate's own gradient, and the
Armijo test, on the total ``F``, accepted it (``docs/design/02``); the trust region's model is the
exact Hessian, and its steps are checked against it.

``optimizer="lbfgs"``: a non-finite ``F`` is backtracked from by the line search. A step taken
with no curvature pair in the history (the first, or after every pair so far was rejected) is
scaled to unit length when the gradient is longer (L-BFGS-B's rule; Pathfinder's ``lambda = 1``
cannot reach a finite trial in 30 halvings from a gradient of 1e14). It stops on L-BFGS-B's
relative-decrease test (``rel_tol``), on a failed line search, or at ``max_iters`` steps. Either
way ascent is monotone, so the last iterate is the best. A fit that never took a step has
**failed**: the result repeats its centre with ``log q = +inf`` (Pathfinder's convention).
"""

from __future__ import annotations

import math
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..optim.lbfgs import _two_loop
from ..optim.trust_ncg import steihaug_cg
from .pathfinder import _Hist, _factors, _lbfgs_push, _line_search, _normal, _recover

__all__ = ["DADVIResult", "dadvi", "DADVI_DEFAULTS", "dadvi_options", "STOP_REASONS",
           "READOUTS", "OPTIMIZERS", "lr_memory_gb"]

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
#: ``"lr_lbfgs"`` without forming the Hessian (``lr_capped``). 1 GiB is n = 2364 in float64 and
#: 3344 in float32; time does not bind (about a minute at n = 5019 on 32 CPUs). ``None``: no
#: cap. ``init_log_sd``: the start's log scale ``omega0`` (lowered further if ``F`` is not finite
#: there). ``optimizer``
#: "trust_ncg" (:data:`OPTIMIZERS`) with SciPy's ``trust-ncg`` defaults, as the paper's code runs
#: it: ``initial_trust_radius`` 1, ``max_trust_radius`` 1000, ``trust_eta`` 0.15 (SciPy's ``eta``),
#: ``gtol`` 1e-4; ``cg_max_iters`` None (``2n``: SciPy's CG has no cap). ``rel_tol`` is then the
#: Newton-decrement test; ``c1``, ``c2``, ``max_halvings`` are L-BFGS's only. ``secant``: the pairs
#: of the ``"lr_lbfgs"`` read-out, ``"steps"`` (the fit's) or ``"cg"`` (``cg_pairs`` exact pairs
#: from CG at the fit, all kept, :func:`_cg_secant`); ``None``: ``"cg"`` under trust_ncg,
#: ``"steps"`` under L-BFGS. The trust region's few long steps make a poor secant read-out (draws'
#: ELBO on bci 10 m -1e6 .. -1.6e10, cscm_imputed -40k); 200 CG pairs give -6.8k .. -9.7k and
#: -31k .. -33k for 1-3 s (2026-11-26, 2 seeds; the exact read-out: bci -3.2k, Pareto k 7.4).
#: Under L-BFGS its own pairs are as good (cscm_imputed -32.6k against CG's -33.3k .. -34.5k).
DADVI_DEFAULTS = dict(elbo_draws=5, standardize=False, covariance="lr", history=6,
                      max_iters=5000, rel_tol=1e-9, c1=1e-4, c2=0.9, max_halvings=30, draws=100,
                      lr_max_gb=1.0, init_log_sd=0.0, optimizer="trust_ncg",
                      initial_trust_radius=1.0, max_trust_radius=1000.0, trust_eta=0.15,
                      gtol=1e-4, cg_max_iters=None, secant=None, cg_pairs=200)

#: ``optimizer`` values: the trust-region Newton-CG (SciPy's ``trust-ncg``, the paper's) and L-BFGS.
OPTIMIZERS = ("trust_ncg", "lbfgs")

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
STOP_REASONS = ("running", "tolerance", "line search", "max_iters", "no finite start",
                "trust region")

_SHRINKS = 10                     # start-scale reductions tried for a non-finite F
_SHRINK = math.log(4.0)


class DADVIResult(NamedTuple):
    """One fixed-draw fit.

    ``draws`` (M, n) are from the read-out normal, with their ``log_p`` and ``log_q``. ``mean`` and
    ``log_sd`` are the fitted ``q``; ``center`` is the realised centroid ``mean + sd * gbar``.
    ``elbo`` is the objective ``F`` at the fit (in-sample, so optimistic). ``stop`` indexes
    :data:`STOP_REASONS`. ``n_grad`` counts log-density gradients (``K`` per evaluation of
    ``F``, plus two per Hessian-vector product of ``F``: ``trust_ncg``'s CG, ``n_hvp``, and the
    exact ``"lr"`` read-out's ``2n``),
    ``n_value`` value-only evaluations; ``omega0`` is the start's log scale, ``readout`` indexes
    :data:`READOUTS` with the read-out drawn from, and ``lr_failed`` whether that is a fallback.
    ``lr_capped`` is whether ``"lr"`` was replaced by ``"lr_lbfgs"`` because the Hessian's
    memory would exceed ``lr_max_gb`` (a configured choice, not a failure); ``lr_skipped`` whether
    the Hessian was not formed because the fit did not converge (a fallback, so ``lr_failed`` is
    set too). ``n_iter`` counts accepted steps, ``n_hvp`` the CG's Hessian-vector products and
    ``n_cg_capped`` the CG solves cut off by ``cg_max_iters`` (both 0 under ``"lbfgs"``). A failed
    fit (no step taken) repeats its centre with ``log_q = +inf``.
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
    n_hvp: Array
    n_cg_capped: Array


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
    radius: Array         # trust_ncg: the trust radius (over the 2n coordinates)
    it: Array             # iterations, accepted or not (trust_ncg's max_iters counts these)
    n_hvp: Array          # Hessian-vector products of F (trust_ncg's CG)
    n_cg_capped: Array    # CG solves stopped by cg_max_iters


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


def _init(vg: Callable, x0: Array, history: int, init_log_sd: float = 0.0,
          radius: float = 1.0) -> _Carry:
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
                  stop=jnp.where(finite(f, g), 0, 4), shrinks=i,
                  radius=jnp.asarray(radius, dtype), it=jnp.asarray(0), n_hvp=jnp.asarray(0),
                  n_cg_capped=jnp.asarray(0))


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
    new = c._replace(eta=jnp.where(ok, eta, c.eta), f=jnp.where(ok, f, c.f),
                     g=jnp.where(ok, g, c.g), hist=hist, alpha=alpha, PS=PS, PZ=PZ, pmask=pmask,
                     k=k, n_eval=c.n_eval + trials, stop=stop, it=c.it + 1)
    return jax.tree_util.tree_map(lambda a, b: jnp.where(c.stop == 0, a, b), new, c)


def _norm(v):
    return jnp.sqrt(jnp.dot(v, v))


def _tr_step(vg: Callable, grad: Callable, c: _Carry, *, max_iters: int, rel_tol: float,
             gtol: float, trust_eta: float, max_trust_radius: float, cg_max_iters: int) -> _Carry:
    """One trust-region Newton-CG iteration (SciPy's ``trust-ncg``) on ``phi = -F``; a no-op once
    stopped.

    The step solves the exact local quadratic model (Hessian-vector products of ``F`` through one
    ``jax.linearize`` of its gradient) over the ball ``|p| <= radius`` by Steihaug's CG; it is
    accepted if ``rho``, the actual over the predicted decrease of ``phi``, exceeds ``trust_eta``,
    and the radius is quartered for ``rho < 1/4`` and doubled (to ``max_trust_radius``) for a
    boundary step with ``rho > 3/4``. Departures from SciPy: a proposal with a non-finite ``F`` or
    gradient has ``rho = -inf`` (rejected, radius quartered; SciPy's NaN ``rho`` would retry the
    same step forever); besides SciPy's ``|grad| < gtol``, an interior Newton-CG step whose
    predicted decrease is below ``max(rel_tol, eps) max(|F|, 1)`` (the Newton decrement, floored
    at what an evaluation of ``F`` can resolve), or not positive (which for CG only rounding
    produces), stops on ``"tolerance"`` (``rel_tol = 0`` turns this test off): scale-free, and
    reachable in float32, where ``|grad| < 1e-4`` often is not; SciPy's ``pred <= 0`` on a
    boundary step, or a radius collapsed below ``eps max(1, |eta|)``, stops on
    ``"trust region"``."""
    dtype = c.eta.dtype
    _, lin = jax.linearize(grad, c.eta)
    sub = steihaug_cg(-c.g, lambda v: -lin(v), c.radius, cg_max_iters)
    p = sub.p
    pred = jnp.dot(c.g, p) - 0.5 * jnp.dot(p, sub.Hp)        # phi - m(p): the predicted decrease
    f, g = vg(c.eta + p)
    finite = jnp.isfinite(f) & jnp.all(jnp.isfinite(g))
    good = finite & (pred > 0)
    rho = jnp.where(good, (f - c.f) / jnp.where(good, pred, 1.0), -jnp.inf)
    radius = jnp.where(rho < 0.25, 0.25 * c.radius,
                       jnp.where((rho > 0.75) & sub.hits_boundary,
                                 jnp.minimum(2.0 * c.radius, max_trust_radius), c.radius))
    ok = rho > trust_eta
    s = jnp.where(ok, p, 0.0)
    z = jnp.where(ok, c.g - g, 0.0)
    hist = _lbfgs_push(c.hist, s, z, ok)
    alpha, PS, PZ, pmask = _recover(c.alpha, c.PS, c.PZ, c.pmask, s, z, ok)
    eta = jnp.where(ok, c.eta + p, c.eta)
    f_new, g_new = jnp.where(ok, f, c.f), jnp.where(ok, g, c.g)
    it = c.it + 1
    # the Newton decrement: a full (interior) Newton-CG step predicting a gain below rel_tol |F|,
    # or below F's own resolution (eps |F|: a gain no evaluation of F can confirm, so rho is
    # rounding and the radius would only collapse). That includes none at all: CG strictly lowers
    # the model, so an interior pred <= 0 is rounding too. A solve cut short by a cg_max_iters
    # below 2n does not count (its gain can be far below the Newton step's); one at 2n does.
    # rel_tol = 0 turns the test off (SciPy's stops only).
    truncated = sub.capped & (cg_max_iters < c.eta.shape[0])
    resolution = max(rel_tol, float(jnp.finfo(dtype).eps)) if rel_tol > 0 else 0.0
    newton = (~sub.hits_boundary & ~truncated
              & (pred < resolution * jnp.maximum(jnp.abs(c.f), 1.0)))
    small = ok & (_norm(g_new) < gtol)
    floor = radius < jnp.finfo(dtype).eps * jnp.maximum(1.0, _norm(eta))
    stop = jnp.where(small | newton, 1, jnp.where((pred <= 0) | floor, 5,
                                                  jnp.where(it >= max_iters, 3, 0)))
    new = c._replace(eta=eta, f=f_new, g=g_new, hist=hist, alpha=alpha, PS=PS, PZ=PZ,
                     pmask=pmask, k=c.k + ok, n_eval=c.n_eval + 2, stop=stop, radius=radius,
                     it=it, n_hvp=c.n_hvp + sub.n_hvp, n_cg_capped=c.n_cg_capped + sub.capped)
    # SciPy does not step from a point whose gradient is already below gtol
    new = jax.tree_util.tree_map(lambda a, b: jnp.where(_norm(c.g) < gtol, b, a), new,
                                 c._replace(stop=jnp.asarray(1, c.stop.dtype)))
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


def _cg_secant(F: Callable, eta: Array, key: Array, n_pairs: int, history: int):
    """Algorithm 3's factors ``(alpha, PS, PZ, pmask)`` from exact curvature pairs at ``eta``, and
    the Hessian-vector products spent.

    Plain CG on ``(-hess F) x = b`` from ``x = 0``, ``b ~ N(0, I)`` (``key``), for at most
    ``n_pairs`` iterations: each direction ``d`` with its product, ``(d, -hess F d)``, is a pair
    ``(s, z)`` exactly as a step's (``z = grad F(eta) - grad F(eta + s)`` for a quadratic), fed
    through Algorithm 3 from ``alpha = 1`` and an empty ``history``-pair buffer. CG stops at
    non-positive curvature (the pair is then rejected) or a residual below ``1e-10 |b|``."""
    dtype, m = eta.dtype, eta.shape[0]
    _, lin = jax.linearize(jax.grad(F), eta)
    b = jax.random.normal(key, (m,), dtype)
    tol = 1e-10 * _norm(b)
    zeros = jnp.zeros((history, m), dtype)

    def body(st):
        alpha, PS, PZ, pmask, r, d, j, _ = st
        Ad = -lin(d)
        dAd = jnp.dot(d, Ad)
        ok = dAd > 0
        alpha, PS, PZ, pmask = _recover(alpha, PS, PZ, pmask, d, Ad, ok)
        rr = jnp.dot(r, r)
        r_new = r - (rr / jnp.where(ok, dAd, 1.0)) * Ad
        rr_new = jnp.dot(r_new, r_new)
        d_new = r_new + (rr_new / jnp.where(rr > 0, rr, 1.0)) * d
        return alpha, PS, PZ, pmask, r_new, d_new, j + 1, ~ok | (jnp.sqrt(rr_new) <= tol)

    init = (jnp.ones((m,), dtype), zeros, zeros, jnp.zeros((history,), dtype), b, b,
            jnp.asarray(0), jnp.asarray(False))
    alpha, PS, PZ, pmask, *_, j, _ = jax.lax.while_loop(
        lambda st: ~st[-1] & (st[-2] < n_pairs), body, init)
    return (alpha, PS, PZ, pmask), j


def _readout(logp: Callable, c: _Carry, g: Array, key: Array, *, covariance: str, draws: int,
             F: Callable | None = None, secant_factors=None):
    """``draws`` from the read-out normal: ``(phi, log_p, log_q, center, readout)``.

    ``covariance="lr"`` (needs the objective ``F``) falls back to ``"lr_lbfgs"``, and that to
    ``"diag"``, when its covariance is not positive definite; ``readout`` indexes
    :data:`READOUTS` with the one used. The exact Hessian is formed only if the fit converged
    (``c.stop`` is ``"tolerance"``), under a ``lax.cond``, so an unconverged fit never pays for
    it; it then falls back to ``"lr_lbfgs"`` as for a non-positive-definite one.
    ``secant_factors`` replaces the fit's own ``(alpha, PS, PZ, pmask)`` in the ``"lr_lbfgs"``
    read-out (``secant="cg"``: :func:`_cg_secant`)."""
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
        fa, fPS, fPZ, fmask = (secant_factors if secant_factors is not None
                               else (c.alpha, c.PS, c.PZ, c.pmask))
        beta, gamma = _factors(fa, fPS, fPZ, fmask)
        alpha = fa[:n] + v * v * fa[n:]
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
         init_log_sd: float = 0.0, optimizer: str = "trust_ncg", initial_trust_radius: float = 1.0,
         max_trust_radius: float = 1000.0, trust_eta: float = 0.15, gtol: float = 1e-4,
         cg_max_iters: int | None = None, secant: str | None = None,
         cg_pairs: int = 200) -> DADVIResult:
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
    if optimizer == "trust_ncg":
        grad = jax.grad(F)
        opts = dict(max_iters=max_iters, rel_tol=rel_tol, gtol=gtol, trust_eta=trust_eta,
                    max_trust_radius=max_trust_radius,
                    cg_max_iters=2 * n if cg_max_iters is None else cg_max_iters)

        def body(c):
            return _tr_step(vg, grad, c, **opts)
    else:
        opts = dict(history=history, max_iters=max_iters, rel_tol=rel_tol, c1=c1, c2=c2,
                    max_halvings=max_halvings)

        def body(c):
            return _step(vg, c, **opts)
    c = jax.lax.while_loop(lambda c: c.stop == 0, body,
                           _init(vg, x0, history, init_log_sd, initial_trust_radius))
    factors, n_pair_hvp = None, 0
    if secant is None:
        secant = "cg" if optimizer == "trust_ncg" else "steps"
    if secant == "cg" and covariance in ("lr_lbfgs", "lr"):
        # every pair is kept (history cg_pairs, not the fit's): the low-rank part is then CG's
        # exact inverse on its Krylov space (exact on a Gaussian at cg_pairs >= 2n)
        factors, n_pair_hvp = _cg_secant(F, c.eta, jax.random.fold_in(draw_key, 1), cg_pairs,
                                         cg_pairs)
    phi, log_p, log_q, center, readout = _readout(logp, c, g, draw_key, covariance=covariance,
                                                  draws=draws, F=F, secant_factors=factors)
    # a Hessian-vector product counts as two gradients of F (trust_ncg's CG; the exact Hessian is
    # 2n of them, formed only for a converged fit)
    converged = c.stop == STOP_REASONS.index("tolerance")
    skipped = jnp.asarray(covariance == "lr") & ~converged
    n_hess = jnp.where(converged, 2 * (2 * n) * elbo_draws, 0) if covariance == "lr" else 0
    failed = c.k == 0
    phi = jnp.where(failed, jnp.broadcast_to(center, phi.shape), phi)
    log_q = jnp.where(failed, jnp.inf, log_q)
    log_p = jnp.where(failed, logp(center), log_p)
    return DADVIResult(draws=phi, log_p=log_p, log_q=log_q, mean=c.eta[:n], log_sd=c.eta[n:],
                       center=center, elbo=c.f, n_iter=c.k, stop=c.stop,
                       n_grad=(c.n_eval * elbo_draws + 2 * elbo_draws * (c.n_hvp + n_pair_hvp)
                               + n_hess),
                       n_value=jnp.asarray(draws + 1),
                       omega0=init_log_sd - _SHRINK * c.shrinks, readout=readout,
                       lr_failed=readout < READOUTS.index(covariance), failed=failed,
                       lr_capped=jnp.asarray(capped), lr_skipped=skipped,
                       n_hvp=c.n_hvp + n_pair_hvp,
                       n_cg_capped=c.n_cg_capped)


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
    if opts["optimizer"] not in OPTIMIZERS:
        raise ValueError(f"DADVI option optimizer must be one of {list(OPTIMIZERS)}, got "
                         f"{opts['optimizer']!r}")
    for name in ("initial_trust_radius", "max_trust_radius", "trust_eta", "gtol"):
        v = opts[name]
        if not (isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)
                and math.isfinite(v)):
            raise ValueError(f"DADVI option {name} must be a finite number, got {v!r}")
        opts[name] = float(v)
    if not 0 < opts["initial_trust_radius"] < opts["max_trust_radius"]:
        raise ValueError("DADVI trust radii need 0 < initial_trust_radius < max_trust_radius, got "
                         f"{opts['initial_trust_radius']!r} and {opts['max_trust_radius']!r}")
    if not 0 <= opts["trust_eta"] < 0.25:
        raise ValueError(f"DADVI option trust_eta must be in [0, 0.25), got {opts['trust_eta']!r}")
    if opts["gtol"] < 0:
        raise ValueError(f"DADVI option gtol must be >= 0, got {opts['gtol']!r}")
    if opts["secant"] not in (None, "steps", "cg"):
        raise ValueError(f"DADVI option secant must be None, 'steps' or 'cg', got "
                         f"{opts['secant']!r}")
    v = opts["cg_pairs"]
    if not (isinstance(v, (int, np.integer)) and not isinstance(v, bool) and v >= 1):
        raise ValueError(f"DADVI option cg_pairs must be an integer >= 1, got {v!r}")
    opts["cg_pairs"] = int(v)
    cap = opts["cg_max_iters"]
    if cap is not None:
        if not (isinstance(cap, (int, np.integer)) and not isinstance(cap, bool) and cap >= 1):
            raise ValueError(f"DADVI option cg_max_iters must be an integer >= 1 or None, got "
                             f"{cap!r}")
        opts["cg_max_iters"] = int(cap)
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
    """Fixed-draw mean-field VI (DADVI) on a flat target ``logp: (n,) -> ()`` from ``x0``.

    ``options`` are those of :data:`DADVI_DEFAULTS`. ``logp`` may return ``-inf``/NaN off its
    support. Compiled once per call (``logp`` is closed over), so call it once per fit rather than
    in a loop.
    """
    opts = dadvi_options(options)
    x0 = jnp.asarray(x0, float)
    return jax.jit(lambda x, k: _fit(logp, x, k, **opts))(x0, key)
