"""Fixed-point solvers for the implicit RMHMC integrator.

The Riemannian kinetic's implicit flow (:class:`~mimcs.hmc.riemannian.RiemannianKinetic`) solves an
implicit equation ``x = g(x)`` twice per step. The solver for that fixed point is a swappable
strategy --- the flow's structure is identical regardless of which solver is used.

* :class:`PicardSolver` -- naive Picard iteration ``x <- g(x)`` (what the classical algorithm uses).
* :class:`AndersonSolver` -- Anderson acceleration (the default), which extrapolates from the last
  few residuals to converge faster and more stably on stiff problems.
* :class:`NewtonSolver` -- Anderson to a coarse residual, then Newton's method on ``g(x) - x`` with
  a dense forward-mode Jacobian. Each Newton step costs ``d`` Jacobian-vector products, so it is
  meant for the small blocks the implicit kinetic is used on, where Anderson's last digits come
  slowly (a spiralling fixed point) and Newton's come quadratically.

All iterate **to a tolerance**: a ``lax.while_loop`` stops once ``norm(g(x) - x) <= tol`` or after
``max_iter`` evaluations of ``g``, and :meth:`FixedPointSolver.solve` reports whether it converged.
That report matters for correctness, not just diagnostics: the generalized leapfrog is reversible
and volume-preserving only at the exact fixed point, so a step whose solve did not converge is not
a valid proposal, and the kinetic turns it into a divergence rather than integrate on from it.

``norm`` is supplied by the caller because only the caller knows the natural scale: the kinetic
measures a momentum residual in ``G^{-1/2}``-whitened units and a position residual in
``G^{1/2}``-whitened ones, which is what makes one absolute ``tol`` meaningful on any problem. The
default tolerance is ``sqrt(eps)`` of the working float type (``~3.5e-4`` in float32, ``~1.5e-8``
with x64) --- tighter than the energy error of any usable step, and loose enough that the residual
of a converged solve is not rounding noise.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

FixedPointMap = Callable[[Array], Array]

#: default cap on evaluations of ``g`` per solve.
DEFAULT_MAX_ITER = 30


def default_tol() -> float:
    """``sqrt(eps)`` of the working float type --- read at call time, so it follows x64."""
    return float(jnp.finfo(jnp.asarray(0.0, float).dtype).eps) ** 0.5


def max_abs(r: Array) -> Array:
    """The default residual norm: the largest absolute component."""
    return jnp.max(jnp.abs(r))


class SolveResult(NamedTuple):
    """The outcome of one solve. ``x`` is the last iterate ``g(x_{n-1})``; ``converged`` says
    whether its residual reached ``tol``; ``n_iter`` counts evaluations of ``g``."""

    x: Array
    converged: Array
    n_iter: Array
    residual: Array


class FixedPointSolver:
    """Solves ``x = g(x)`` from an initial guess ``x0``, to ``tol`` within ``max_iter`` steps."""

    def __init__(self, tol: float | None = None, max_iter: int = DEFAULT_MAX_ITER):
        if int(max_iter) < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter}")
        self.tol = None if tol is None else float(tol)
        self.max_iter = int(max_iter)

    def _tol(self) -> float:
        return default_tol() if self.tol is None else self.tol

    def solve(self, g: FixedPointMap, x0: Array, norm: Callable = max_abs) -> SolveResult:
        raise NotImplementedError


class PicardSolver(FixedPointSolver):
    """Naive Picard iteration ``x_{k+1} = g(x_k)``, stopped at ``tol``."""

    def solve(self, g, x0, norm=max_abs):
        tol = self._tol()
        gx = g(x0)
        res = norm(gx - x0)

        def cond(c):
            i, _, res = c
            return (res > tol) & (i < self.max_iter)

        def body(c):
            i, x, _ = c
            gx = g(x)
            return i + 1, gx, norm(gx - x)

        n, x, res = jax.lax.while_loop(cond, body, (jnp.int32(1), gx, res))
        return SolveResult(x, res <= tol, n, res)


class AndersonSolver(FixedPointSolver):
    """Anderson acceleration (type II), stopped at ``tol``.

    Keeps the last ``depth`` pairs ``(x_i, g(x_i))`` in a ring buffer. Each step solves the
    constrained least squares ``min_alpha ||sum_i alpha_i r_i||`` subject to ``sum_i alpha_i = 1``
    over the valid history (``r_i = g(x_i) - x_i``, via the bordered linear system) and moves to
    ``x = beta * sum_i alpha_i g(x_i) + (1 - beta) * sum_i alpha_i x_i``. Unfilled slots are masked
    out of the system (their row and column replaced by the identity, so their ``alpha`` is 0),
    which keeps the loop shape-stable. ``depth = 1`` is Picard.

    Two details decide whether this beats Picard near convergence rather than only far from it:

    * the ridge on the normal equations is **relative** --- ``reg`` times the mean diagonal of
      ``R R^T``. An absolute ridge swamps ``R R^T`` once the residuals are small, and the mixture
      degrades to a plain average of the history, which converges *slower* than Picard;
    * the step is **safeguarded**: when a residual grows by more than ``safeguard`` (10x), the
      history is reset to the newest pair, so the next step is a plain Picard step from there
      instead of an extrapolation built on the stale directions that just failed. The factor is
      loose on purpose --- Anderson residuals are not monotone, and resetting on *any* growth
      (``safeguard=1``) was measured to cost iterations and even convergence (40 random stiff
      6-d contractions: one failure at 300 iterations, none at 10x or with no safeguard).

    Args:
        depth: memory window ``m`` (number of past residuals to mix).
        tol, max_iter: stopping rule (see :class:`FixedPointSolver`).
        mixing: ``beta`` (1.0 = pure ``g`` extrapolation).
        regularization: relative ridge on the normal equations.
        safeguard: residual growth factor that resets the history (``None``: never reset).
    """

    def __init__(self, depth: int = 3, tol: float | None = None,
                 max_iter: int = DEFAULT_MAX_ITER, mixing: float = 1.0,
                 regularization: float = 1e-8, safeguard: float | None = 10.0):
        super().__init__(tol=tol, max_iter=max_iter)
        self.safeguard = None if safeguard is None else float(safeguard)
        if int(depth) < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        self.depth = int(depth)
        self.mixing = float(mixing)
        self.reg = float(regularization)

    def _mix(self, X, Gx, valid):
        """The Anderson iterate from the history ``(X, Gx)`` restricted to ``valid`` slots."""
        m = self.depth
        R = jnp.where(valid[:, None], Gx - X, 0.0)
        A = R @ R.T
        n_valid = jnp.maximum(jnp.sum(valid), 1)
        scale = jnp.trace(A) / n_valid
        tiny = jnp.finfo(A.dtype).tiny
        eye = jnp.eye(m, dtype=A.dtype)
        pair = valid[:, None] & valid[None, :]
        A = jnp.where(pair, A + self.reg * (scale + tiny) * eye, eye)
        ones = valid.astype(A.dtype)
        top = jnp.concatenate([A, ones[:, None]], axis=1)
        bot = jnp.concatenate([ones[None, :], jnp.zeros((1, 1), A.dtype)], axis=1)
        M = jnp.concatenate([top, bot], axis=0)
        rhs = jnp.concatenate([jnp.zeros((m,), A.dtype), jnp.ones((1,), A.dtype)])
        alpha = jnp.linalg.solve(M, rhs)[:m]
        alpha = jnp.where(valid, alpha, 0.0)
        return self.mixing * (alpha @ Gx) + (1.0 - self.mixing) * (alpha @ X)

    def solve(self, g, x0, norm=max_abs):
        tol = self._tol()
        m = self.depth
        d = x0.shape[0]
        gx0 = g(x0)
        res0 = norm(gx0 - x0)
        X = jnp.zeros((m, d), x0.dtype).at[0].set(x0)
        Gx = jnp.zeros((m, d), x0.dtype).at[0].set(gx0)
        valid = jnp.zeros((m,), bool).at[0].set(True)

        def cond(c):
            i, *_, res = c
            return (res > tol) & (i < self.max_iter)

        def body(c):
            i, slot, X, Gx, valid, last_g, res = c
            x = self._mix(X, Gx, valid)
            # A singular bordered system (residuals collinear to machine precision) is the one way
            # the mixture goes non-finite; the newest ``g`` is the right step then.
            x = jnp.where(jnp.all(jnp.isfinite(x)), x, last_g)
            gx = g(x)
            new_res = norm(gx - x)
            grew = (new_res > self.safeguard * res) if self.safeguard is not None else False
            slot = (slot + 1) % m
            # Safeguard: a grown residual discards the history, keeping only the newest pair.
            valid = jnp.where(grew, jnp.zeros((m,), bool), valid)
            X, Gx = X.at[slot].set(x), Gx.at[slot].set(gx)
            valid = valid.at[slot].set(True)
            return i + 1, slot, X, Gx, valid, gx, new_res

        init = (jnp.int32(1), jnp.int32(0), X, Gx, valid, gx0, res0)
        n, _, _, _, _, x, res = jax.lax.while_loop(cond, body, init)
        return SolveResult(x, res <= tol, n, res)


class NewtonSolver(FixedPointSolver):
    """Newton's method on ``F(x) = g(x) - x``, warm-started by Anderson, stopped at ``tol``.

    Two phases share one evaluation budget:

    1. **Anderson** (:class:`AndersonSolver`, depth ``depth``) until the residual is at most
       ``warm_start``. Newton's basin is local: between the explicit guess and the root of the
       implicit drift, ``J_g`` can have eigenvalues above 1 (a fold in ``F``), and Newton started
       from the guess was measured to fail more often than Anderson alone.
    2. **Newton**: full steps ``dx = -(J_g - I)^{-1} F(x)``, with the dense Jacobian from ``d``
       forward-mode products (``jax.jacfwd``). Near the root it converges quadratically where
       Anderson spirals slowly (``J_g`` with complex eigenvalues of modulus 0.5-0.8 in a funnel's
       neck). A singular or non-finite step falls back to the Picard direction ``dx = F(x)``.

    There is deliberately no line search: one on the residual norm was measured to *raise* the
    failure rate, because the Newton path often has to pass a larger residual to reach the root.

    Measured on centered eight schools' ``(mu, log tau)`` drift (4000 exact draws, eps 0.4):
    drift failures 4.1% / 2.2% / 0.36% for Anderson at a budget of 30 / 60 / 300 evaluations,
    1.8% / 0.85% / 0.23% for this solver, and 5.9% for Newton from the guess
    (``warm_start=None``) at 30. No solver helps where *no* fixed point exists --- the implicit
    kick past its critical step, a quadratic with no real root --- and this one reports
    non-convergence there like the others.

    ``n_iter`` and ``max_iter`` count **evaluations of g**: one per iterate plus ``d`` per Jacobian
    (a forward-mode product costs about one evaluation), so the budget, the ``fp_iters``
    diagnostic and the kinetic's cost model mean the same thing for every solver. The Jacobian is
    dense, so this is for the small blocks the implicit kinetic is used on.

    Args:
        tol, max_iter: stopping rule (see :class:`FixedPointSolver`), shared by both phases.
        warm_start: residual at which Anderson hands over to Newton (``None``: Newton throughout).
        depth: the Anderson phase's memory window.
    """

    def __init__(self, tol: float | None = None, max_iter: int = DEFAULT_MAX_ITER,
                 warm_start: float | None = 1e-2, depth: int = 3):
        super().__init__(tol=tol, max_iter=max_iter)
        if warm_start is not None and not float(warm_start) > 0.0:
            raise ValueError(f"warm_start must be positive or None, got {warm_start}")
        self.warm_start = None if warm_start is None else float(warm_start)
        self.depth = int(depth)

    def solve(self, g, x0, norm=max_abs):
        tol = self._tol()
        d = x0.shape[0]
        eye = jnp.eye(d, dtype=x0.dtype)
        if self.warm_start is None:
            n0, gx = jnp.int32(1), g(x0)
            x = x0
        else:
            # Anderson's result is g at its last iterate; continue from there.
            warm = AndersonSolver(depth=self.depth, tol=max(self.warm_start, tol),
                                  max_iter=self.max_iter).solve(g, x0, norm)
            x, n0 = warm.x, warm.n_iter + 1
            gx = g(x)

        def cond(c):
            i, _, _, res = c
            return (res > tol) & (i < self.max_iter)

        def body(c):
            i, x, gx, _ = c
            r = gx - x
            dx = -jnp.linalg.solve(jax.jacfwd(g)(x) - eye, r)
            dx = jnp.where(jnp.all(jnp.isfinite(dx)), dx, r)
            x = x + dx
            gx = g(x)
            return i + d + 1, x, gx, norm(gx - x)

        n, x, gx, res = jax.lax.while_loop(cond, body, (n0, x, gx, norm(gx - x)))
        # Same contract as the other solvers: return g at the last iterate, whose residual is res.
        return SolveResult(gx, res <= tol, n, res)


#: the solver names a spec may give, and the options each accepts.
SOLVERS = {"picard": PicardSolver, "anderson": AndersonSolver, "newton": NewtonSolver}
SOLVER_PARAMS = {"picard": frozenset({"tol", "max_iter"}),
                 "newton": frozenset({"tol", "max_iter", "warm_start", "depth"}),
                 "anderson": frozenset({"tol", "max_iter", "depth", "mixing", "regularization",
                                     "safeguard"})}


def resolve_solver(solver=None, **params) -> FixedPointSolver:
    """Map a user-facing ``solver`` to a :class:`FixedPointSolver`.

    ``None`` -> Anderson (the default); ``"picard"`` / ``"anderson"`` -> that kind built from
    ``params`` (unknown keys raise, so a typo cannot pass silently); an object is returned unchanged
    (and ``params`` must then be empty).
    """
    if isinstance(solver, FixedPointSolver):
        if params:
            raise ValueError(f"solver options {sorted(params)} given alongside a solver object; "
                             f"configure the object instead")
        return solver
    name = "anderson" if solver is None else solver
    if name not in SOLVERS:
        raise ValueError(f"unknown solver {solver!r} (expected one of {sorted(SOLVERS)}, or a "
                         f"FixedPointSolver)")
    unknown = sorted(set(params) - SOLVER_PARAMS[name])
    if unknown:
        raise ValueError(f"unknown solver option(s) {unknown} for {name!r} "
                         f"(it takes {sorted(SOLVER_PARAMS[name])})")
    return SOLVERS[name](**params)
