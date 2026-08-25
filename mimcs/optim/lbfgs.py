"""Limited-memory BFGS (L-BFGS) minimisation, in pure JAX.

A general ``minimize(fun, x0)`` for a smooth scalar objective over any pytree of parameters
(ravelled to a flat vector internally). The search direction is the two-loop recursion over
the last ``m`` correction pairs ``(s_k, y_k)`` (``s = x_{k+1} - x_k``, ``y = g_{k+1} - g_k``),
scaled by the Barzilai--Borwein factor ``gamma = (s.y)/(y.y)``; the step length comes from a
backtracking Armijo line search. A correction pair is only accepted when the curvature
condition ``s.y > eps |y|^2`` holds, so the implicit inverse-Hessian stays positive definite.

The whole routine is a single ``lax.while_loop`` (fixed-shape history buffers), so it is
jittable and differentiable-through where needed. Written for offline use --- fitting a
handful-to-hundreds of parameters against a batch of evidence --- not as an inner-loop kernel.
"""

from __future__ import annotations

import logging
import math
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.flatten_util import ravel_pytree

from .._logging import get_logger

log = get_logger(__name__)


class OptimizeResult(NamedTuple):
    """Outcome of :func:`minimize`.

    Attributes:
        x: the minimiser, in the same pytree structure as ``x0``.
        fun: objective value at ``x``.
        grad_norm: max-norm of the gradient at ``x``.
        n_iter: number of outer iterations taken.
        converged: whether ``grad_norm`` fell below the tolerance.
    """

    x: object
    fun: Array
    grad_norm: Array
    n_iter: Array
    converged: Array


class _State(NamedTuple):
    x: Array
    f: Array
    g: Array
    S: Array          # (m, n) correction steps, newest at row m-1
    Y: Array          # (m, n) gradient differences, newest at row m-1
    rho: Array        # (m,) 1/(s.y) per pair, 0 for empty/rejected slots
    gamma: Array      # scalar initial-Hessian scale
    k: Array          # iteration counter
    gnorm: Array      # max-norm of g


def _two_loop(g: Array, st: _State, m: int) -> Array:
    """L-BFGS two-loop recursion: the search direction ``-H g`` from the history in ``st``.

    Rows are ordered oldest (0) to newest (m-1); empty/rejected slots have ``rho = 0`` and are
    therefore inert. With no history this returns ``-gamma * g`` (scaled steepest descent).
    """
    S, Y, rho = st.S, st.Y, st.rho

    def loop1(i, carry):
        q, alphas = carry
        idx = m - 1 - i                       # newest first
        a = rho[idx] * jnp.dot(S[idx], q)
        q = q - a * Y[idx]
        return q, alphas.at[idx].set(a)

    q, alphas = jax.lax.fori_loop(0, m, loop1, (g, jnp.zeros(m, g.dtype)))
    r = st.gamma * q

    def loop2(idx, r):
        b = rho[idx] * jnp.dot(Y[idx], r)     # oldest first
        return r + S[idx] * (alphas[idx] - b)

    r = jax.lax.fori_loop(0, m, loop2, r)
    return -r


def _line_search(f_flat: Callable[[Array], Array], x: Array, f: Array, g: Array, p: Array,
                 *, c1: float, shrink: float, max_ls: int) -> Array:
    """Backtracking Armijo line search: a step ``t`` with ``f(x+t p) <= f + c1 t g.p``."""
    gp = jnp.dot(g, p)

    def cond(state):
        t, ft, i = state
        armijo = ft <= f + c1 * t * gp
        return (~armijo) & (i < max_ls) & jnp.isfinite(gp)

    def body(state):
        t, _, i = state
        t = t * shrink
        return t, f_flat(x + t * p), i + 1

    t0 = jnp.asarray(1.0, x.dtype)
    t, _, _ = jax.lax.while_loop(cond, body, (t0, f_flat(x + t0 * p), jnp.asarray(0)))
    return t


def _host_float(x):
    """``float(x)``, or ``None`` when ``x`` is a tracer (``minimize`` called under ``jit``).

    The whole routine is a ``lax.while_loop``, so its outcome is only inspectable on the host
    when the caller is not itself tracing; there is nothing to report in the traced case.
    """
    try:
        return float(x)
    except Exception:                       # TracerArrayConversionError / ConcretizationTypeError
        return None


def _log_outcome(res: "OptimizeResult", max_iter: int, gtol: float, warn: bool) -> None:
    """Report how the minimisation ended: DEBUG always, WARNING when it ran out of iterations."""
    n_iter, gnorm, f = (_host_float(res.n_iter), _host_float(res.grad_norm),
                        _host_float(res.fun))
    if n_iter is None:                      # traced: no concrete outcome to report
        log.debug("L-BFGS traced under jit; termination not reported")
        return
    converged = gnorm < gtol
    if not math.isfinite(f):
        log.warning("L-BFGS stopped on a non-finite objective (f=%g) after %d iteration(s); "
                    "the returned point is not a minimiser", f, int(n_iter))
    elif int(n_iter) >= max_iter and not converged:
        log.log(logging.WARNING if warn else logging.DEBUG,
                "L-BFGS hit max_iter=%d without converging: gradient max-norm %.3g still "
                "above gtol=%.3g (f=%.6g). The fit is the last iterate, not a minimiser.",
                max_iter, gnorm, gtol, f)
    log.debug("L-BFGS terminated after %d/%d iteration(s): f=%.6g, grad max-norm=%.3g, "
              "converged=%s", int(n_iter), max_iter, f, gnorm, converged)


def minimize(fun: Callable, x0, *, max_iter: int = 1000, m: int = 10, gtol: float = 1e-6,
             max_ls: int = 25, c1: float = 1e-4, shrink: float = 0.5,
             warn_max_iter: bool = True) -> OptimizeResult:
    """Minimise ``fun`` from ``x0`` by L-BFGS.

    Args:
        fun: scalar objective ``fun(x) -> ()``; ``x`` has the pytree structure of ``x0``.
        x0: initial parameters (any pytree of arrays).
        max_iter: maximum outer iterations.
        m: correction-pair history length.
        gtol: stop when the gradient max-norm is below this.
        max_ls: maximum backtracking steps per line search.
        c1: Armijo sufficient-decrease constant.
        shrink: line-search step shrink factor.
        warn_max_iter: log a WARNING when the iteration cap is reached without convergence
            (the default). A caller for which stopping on the cap is routine, and which reads
            ``converged`` itself, can set this ``False`` to log that at DEBUG instead.

    Returns:
        An :class:`OptimizeResult` (``x`` in the structure of ``x0``).
    """
    x0_flat, unravel = ravel_pytree(x0)
    n = x0_flat.shape[0]
    log.debug("L-BFGS on %d parameter(s): max_iter=%d, m=%d, gtol=%.3g", n, max_iter, m, gtol)
    value_and_grad = jax.value_and_grad(lambda z: fun(unravel(z)))
    f_flat = lambda z: fun(unravel(z))
    eps = jnp.asarray(1e-10, x0_flat.dtype)

    def cond(st: _State):
        return (st.k < max_iter) & (st.gnorm >= gtol) & jnp.isfinite(st.f)

    def body(st: _State) -> _State:
        p = _two_loop(st.g, st, m)
        gp = jnp.dot(st.g, p)
        p = jnp.where(gp < 0, p, -st.g)                 # guard a non-descent direction
        t = _line_search(f_flat, st.x, st.f, st.g, p,
                         c1=c1, shrink=shrink, max_ls=max_ls)
        s = t * p
        x_new = st.x + s
        f_new, g_new = value_and_grad(x_new)
        y = g_new - st.g

        sy = jnp.dot(s, y)
        yy = jnp.dot(y, y)
        accept = sy > eps * yy                          # curvature: keep H positive definite
        rho_new = jnp.where(accept, 1.0 / jnp.where(accept, sy, 1.0), 0.0)

        def push(a, row):
            return jnp.roll(a, -1, axis=0).at[m - 1].set(row)

        S = jnp.where(accept, push(st.S, s), st.S)
        Y = jnp.where(accept, push(st.Y, y), st.Y)
        rho = jnp.where(accept, push(st.rho, rho_new), st.rho)
        gamma = jnp.where(accept & (yy > 0), sy / yy, st.gamma)

        return _State(x=x_new, f=f_new, g=g_new, S=S, Y=Y, rho=rho, gamma=gamma,
                      k=st.k + 1, gnorm=jnp.max(jnp.abs(g_new)))

    f0, g0 = value_and_grad(x0_flat)
    init = _State(
        x=x0_flat, f=f0, g=g0,
        S=jnp.zeros((m, n), x0_flat.dtype), Y=jnp.zeros((m, n), x0_flat.dtype),
        rho=jnp.zeros((m,), x0_flat.dtype), gamma=jnp.asarray(1.0, x0_flat.dtype),
        k=jnp.asarray(0), gnorm=jnp.max(jnp.abs(g0)))
    st = jax.lax.while_loop(cond, body, init)

    result = OptimizeResult(
        x=unravel(st.x), fun=st.f, grad_norm=st.gnorm, n_iter=st.k,
        converged=st.gnorm < gtol)
    _log_outcome(result, max_iter, gtol, warn_max_iter)
    return result
