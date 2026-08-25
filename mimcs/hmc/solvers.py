"""Fixed-point solvers for the implicit RMHMC integrator.

**[experimental]**, with :mod:`mimcs.hmc.riemannian` --- these exist only for the implicit variant.
The explicit block-Riemannian path needs no solver at all.

The Riemannian kinetic's implicit flow (``RiemannianKinetic.flow`` in
``mimcs/hmc/riemannian.py``) solves an implicit equation ``x = g(x)`` twice per step. The
solver for that fixed point is a swappable strategy -- the flow's structure is identical
regardless of which solver is used. This mirrors the :class:`~mimcs.hmc.Metric` pattern: a
small object with subclasses, injected into the host (here ``RiemannianKinetic``).

* :class:`PicardSolver` -- naive Picard iteration ``x <- g(x)`` (the default; what the
  classical algorithm uses).
* :class:`AndersonSolver` -- Anderson acceleration, which extrapolates from the last few
  residuals to converge faster and more stably on stiff problems.

Both run a fixed number of iterations (JIT-friendly, shape-stable). ``AndersonSolver``
unrolls its loop in Python (the iteration count is small and static), so the changing
history-window size is handled with ordinary slicing rather than masking.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array

FixedPointMap = Callable[[Array], Array]


class FixedPointSolver:
    """Solves ``x = g(x)`` from an initial guess ``x0``."""

    def solve(self, g: FixedPointMap, x0: Array) -> Array:
        raise NotImplementedError


class PicardSolver(FixedPointSolver):
    """Naive Picard iteration: ``x_{k+1} = g(x_k)`` for a fixed number of sweeps."""

    def __init__(self, n_iterations: int = 8):
        self.n_iterations = int(n_iterations)

    def solve(self, g, x0):
        return jax.lax.fori_loop(0, self.n_iterations, lambda _, x: g(x), x0)


class AndersonSolver(FixedPointSolver):
    """Anderson acceleration (type-II) with a fixed memory depth and iteration count.

    At each step it forms the last ``min(k, depth)`` residuals ``r_i = g(x_i) - x_i`` and
    solves the constrained least squares ``min_alpha ||sum_i alpha_i r_i||`` subject to
    ``sum_i alpha_i = 1`` (via the bordered linear system), then mixes
    ``x_{k+1} = beta * sum_i alpha_i g(x_i) + (1-beta) * sum_i alpha_i x_i``. With
    ``depth = 1`` this reduces to Picard.

    Args:
        depth: memory window ``m`` (number of past residuals to mix).
        n_iterations: total iterations (same budget interpretation as Picard).
        beta: mixing parameter (1.0 = pure g-extrapolation).
        regularization: ridge added to the normal equations for conditioning.
    """

    def __init__(self, depth: int = 3, n_iterations: int = 8, beta: float = 1.0,
                 regularization: float = 1e-6):
        self.depth = int(depth)
        self.n_iterations = int(n_iterations)
        self.beta = float(beta)
        self.reg = float(regularization)

    def solve(self, g, x0):
        m = self.depth
        xs = [x0]
        gs = [g(x0)]
        x = gs[0]                       # first iterate is a Picard step
        for _ in range(1, self.n_iterations):
            xs.append(x)
            gs.append(g(x))
            w = min(len(xs), m)         # static (Python int): window size
            Xw = jnp.stack(xs[-w:])     # (w, d)
            Gw = jnp.stack(gs[-w:])
            R = Gw - Xw                 # residuals (w, d)
            A = R @ R.T + self.reg * jnp.eye(w)
            # bordered system enforcing sum(alpha) = 1
            top = jnp.concatenate([A, jnp.ones((w, 1))], axis=1)
            bot = jnp.concatenate([jnp.ones((1, w)), jnp.zeros((1, 1))], axis=1)
            M = jnp.concatenate([top, bot], axis=0)
            rhs = jnp.concatenate([jnp.zeros((w,)), jnp.ones((1,))])
            alpha = jnp.linalg.solve(M, rhs)[:w]
            x = self.beta * (alpha @ Gw) + (1.0 - self.beta) * (alpha @ Xw)
        return x


def resolve_solver(solver, n_fixed_point: int) -> FixedPointSolver | None:
    """Map a user-facing ``solver`` (None / string / object) to a FixedPointSolver.

    ``None`` -> use the integrator's default (Picard with ``n_fixed_point`` sweeps);
    ``"picard"`` / ``"anderson"`` -> a default solver of that kind; an object is returned
    unchanged.
    """
    if solver is None or isinstance(solver, FixedPointSolver):
        return solver
    if solver == "picard":
        return PicardSolver(n_fixed_point)
    if solver == "anderson":
        return AndersonSolver(depth=3, n_iterations=n_fixed_point)
    raise ValueError(f"unknown solver {solver!r} (expected 'picard', 'anderson', or a "
                     f"FixedPointSolver)")
