"""L-BFGS on a shared block, Newton on the lanes: :func:`lbfgs_newton` (in :mod:`mimcs.optim`).

:func:`mimcs.optim.separable_newton` solves an arrow-structured problem --- many small per-lane
blocks plus ``s`` parameters shared by every lane --- with a full Newton step, whose shared part is
a Schur complement: ``s`` Hessian-vector probes and an ``s x s`` eigendecomposition per iteration.
That is cheap for a handful of shared slots and prohibitive for thousands (the metric regression's
dense shared-weight atoms: up to ``min(dependency width, evidence rows)`` slots once fitted on the
row span, ``regression.SPAN_REDUCE``). This solver keeps the per-lane Newton and replaces the
shared Newton step by L-BFGS:

* each outer iteration takes an L-BFGS direction ``d`` for the shared block from its gradient;
* every trial point of the line search along ``d`` re-solves the lanes at the trial shared value,
  with up to ``inner_max_iter`` separable Newton iterations (:func:`separable_newton` with no shared
  leaves: per-lane steps, step lengths and convergence), so the line search is on the objective
  with the lanes (partly or fully) re-equilibrated;
* the shared gradient is then read at the re-solved lanes. With the lanes solved to convergence
  this is the gradient of the lane-profiled objective ``phi(x_s) = min_{x_l} F(x_l, x_s)`` by the
  envelope theorem, and the method is L-BFGS on ``phi``, i.e. Newton with the explicit Schur
  complement replaced by its secant estimate. With ``inner_max_iter = 1`` it interleaves one
  shared L-BFGS step with one lane Newton step.

Options, each one arm of the solver study (``tests/experiments/writeups/shared_solver.md``):

* ``predictor``: start each trial's lane solve from the lanes' first-order response to the shared
  move, ``x_l - t A^-1 C^T d`` (one HVP plus the lane Hessians, ``p`` probes), instead of from the
  current lanes;
* ``curvature = (c, b)``: a diagonal model ``D = a c + b`` of the shared block's curvature, ``c``
  and ``b`` given per shared slot by the caller and the scalar ``a`` fitted to one Rayleigh quotient
  of the Schur complement at the start; L-BFGS's initial inverse Hessian is then ``theta D^-1``
  (``theta = s.y / y.D^-1 y``) instead of a scalar. Without it the start is the scalar Rayleigh
  step ``g.g / g.Sg``;
* ``shared_newton``: take the shared direction from a Newton step on the shared block **with the
  lanes held fixed** (``s`` probes, as expensive as the Schur complement): alternating Newton
  half-steps, the block-coordinate scheme that zigzagged on cscm. A control, not a candidate.

A line search that finds no decrease does not end the solve: the history is dropped and the
direction falls back to the scaled gradient; only a second failure in a row stops it.
Convergence: the shared gradient's max-norm below ``gtol`` with every lane converged, or the
objective unchanged at the dtype's resolution (``ftol``) for ``ftol_patience`` accepted steps --- the
tests :func:`separable_newton` uses.

One ``lax.while_loop`` over fixed shapes, so jittable, like the other solvers here.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from .._logging import get_logger
from ._common import OptimizeResult, log_outcome
from . import newton as _nt
from .lbfgs import _two_loop

log = get_logger(__name__)

_TRACE_KEYS = ("f", "t", "gnorm", "n_inner", "n_eval", "reset")


class _State(NamedTuple):
    xl: Array          # (K, p)
    xs: Array          # (s,)
    f: Array           # () summed objective
    gs: Array          # (s,) shared gradient at the current lanes
    gl: Array          # (K, p) lane gradient
    lconv: Array       # (K,) lanes converged in the last lane solve
    S: Array           # (m, s) L-BFGS steps
    Y: Array           # (m, s) gradient differences
    rho: Array         # (m,)
    gamma: Array       # (s,) initial inverse Hessian, theta * D^-1
    theta: Array       # ()
    k: Array
    flat: Array        # consecutive accepted steps that left the objective unchanged
    done: Array        # converged
    stop: Array        # two failed line searches in a row
    reset: Array       # the last line search failed: next step from the scaled gradient
    n_inner: Array     # lane Newton iterations so far
    n_eval: Array      # lane solves (objective evaluations) so far
    hist: dict = None


def lbfgs_newton(loss_vec: Callable, x0, *, inner_max_iter: int = 1, predictor: bool = False,
                 curvature=None, shared_newton: bool = False, m: int = 10, max_iter: int = 500,
                 gtol: float | None = None, ftol: float | None = None, ftol_patience: int = 30,
                 max_ls: int = 25, c1: float = 1e-4, shrink: float = 0.5,
                 eig_floor: float = 1e-8, warn_max_iter: bool = True,
                 trace: bool = False) -> OptimizeResult:
    """Minimise ``sum(loss_vec(x))``: L-BFGS on the shared leaves, Newton on the per-lane ones.

    Args:
        loss_vec, x0: as :func:`separable_newton` --- the objective per lane, and a pytree whose
            leaves have axis 0 of length ``K`` (per lane) or 1 (shared). With no shared leaf this
            *is* :func:`separable_newton`, and the call is handed to it.
        inner_max_iter: lane Newton iterations per lane solve (1: interleaved; ~20: profiled).
        predictor: warm-start each lane solve from the lanes' linear response to the shared move.
        curvature: ``(c, b)`` --- ``c`` a pytree like ``x0`` whose shared leaves hold the per-slot
            curvature shape (lane leaves are ignored), ``b`` a scalar or per-slot floor; the start is
            ``D = a c + b`` with ``a`` fitted. ``None``: the scalar Rayleigh start.
        shared_newton: the shared direction from a Newton step with the lanes fixed (the
            alternating-Newton control; ``s`` probes per iteration).
        m: L-BFGS history length.
        max_iter: outer (shared) iterations.
        gtol, ftol, ftol_patience, max_ls, c1, shrink, eig_floor: as :func:`separable_newton`
            (``gtol``/``ftol`` default to ``sqrt(eps)`` / ``eps / 2`` of ``x0``'s dtype).
        warn_max_iter: WARNING when the cap is reached unconverged.
        trace: also return per-iteration arrays in the result's ``trace``: ``f`` (objective after
            the iteration), ``t`` (step length), ``gnorm`` (shared gradient max-norm),
            ``n_inner`` and ``n_eval`` (cumulative lane Newton iterations and lane solves: the
            work, since each is a pass over the data), ``reset`` (a failed line search).

    Returns:
        An :class:`~mimcs.optim.OptimizeResult`; ``lane_converged`` is the global verdict
        broadcast over the lanes (a shared solve converges as one).
    """
    out = jax.eval_shape(loss_vec, x0)
    if len(out.shape) != 1:
        raise ValueError(f"lbfgs_newton: loss_vec must return a 1-D per-lane vector, got "
                         f"shape {out.shape}")
    K = int(out.shape[0])
    layout = _nt.lane_layout(x0, K)
    p, s = layout.n_slots, layout.n_shared
    gtol, ftol = _nt.default_tolerances(x0, gtol, ftol)
    if s == 0:
        return _nt.separable_newton(loss_vec, x0, max_iter=max_iter, gtol=gtol, ftol=ftol,
                                    ftol_patience=ftol_patience, max_ls=max_ls, c1=c1,
                                    shrink=shrink, eig_floor=eig_floor,
                                    warn_max_iter=warn_max_iter, trace=trace)
    log.debug("L-BFGS/Newton on %d lane(s) x %d slot(s) + %d shared: inner %d, predictor %s, "
              "curvature %s, shared Newton %s", K, p, s, inner_max_iter, predictor,
              curvature is not None, shared_newton)

    def lossv(zl, zs):
        return jnp.asarray(loss_vec(_nt.lane_unravel(zl, zs, layout)))

    grad_fn = jax.grad(lambda zl, zs: jnp.sum(lossv(zl, zs)), argnums=(0, 1))

    def value_and_grads(zl, zs):
        fv, vjp = jax.vjp(lossv, zl, zs)
        gl, gs = vjp(jnp.ones_like(fv))
        return jnp.sum(fv), gl, gs

    def lane_solve(xl_start, xs):
        """The lanes re-solved at fixed ``xs``: ``(xl, f, gl, gs, lanes converged, iterations)``."""
        if p == 0:
            xl, lc, n = xl_start, jnp.ones((K,), bool), jnp.asarray(0)
        else:
            res = _nt.separable_newton(lambda zl: lossv(zl, xs), xl_start,
                                       max_iter=inner_max_iter, gtol=gtol, ftol=ftol,
                                       max_ls=max_ls, c1=c1, shrink=shrink, eig_floor=eig_floor,
                                       warn_max_iter=False)
            xl, lc, n = res.x, res.lane_converged, jnp.asarray(res.n_iter)
        f, gl, gs = value_and_grads(xl, xs)
        return xl, f, gl, gs, lc, n

    def lane_factor(xl, xs):
        """The floored lane Hessians, ``(V, 1/lambda)``, for the lanes' response to a move."""
        H, _, _ = _nt._hessian_probe(grad_fn, xl, xs, p, 0)
        good = jnp.all(jnp.isfinite(H), axis=(-2, -1))
        H = jnp.where(good[:, None, None], H, jnp.eye(p, dtype=H.dtype))
        return _nt._modified_eigh(H, eig_floor)

    def lane_response(xl, xs, factor, d):
        """``-A^-1 C^T d``: the lanes' first-order response to the shared move ``d``."""
        ctd = jax.jvp(grad_fn, (xl, xs), (jnp.zeros_like(xl), d))[1][0]
        V, inv = factor
        return -_nt._apply_inv(V, inv, ctd)

    def schur_times(xl, xs, g):
        """``S g`` for the Schur complement of the lane blocks, matrix-free (two HVPs)."""
        dl = (lane_response(xl, xs, lane_factor(xl, xs), g) if p
              else jnp.zeros_like(xl))
        return jax.jvp(grad_fn, (xl, xs), (dl, g))[1][1]

    xl0, xs0 = _nt.lane_ravel(x0, layout)
    dtype = xs0.dtype
    tiny = jnp.asarray(jnp.finfo(dtype).tiny, dtype)
    xl, f, gl, gs, lc, n0 = lane_solve(xl0, xs0)

    # The initial inverse Hessian: D = a c + b with a from one Rayleigh quotient of S.
    if curvature is not None:
        c = jnp.asarray(_nt.lane_ravel(curvature[0], layout)[1], dtype)
        b = jnp.broadcast_to(jnp.asarray(curvature[1], dtype), (s,))
    else:
        c, b = jnp.ones((s,), dtype), jnp.zeros((s,), dtype)
    gSg = jnp.dot(gs, schur_times(xl, xs0, gs))
    a = (gSg - jnp.dot(b * gs, gs)) / jnp.maximum(jnp.dot(c * gs, gs), tiny)
    a = jnp.where(jnp.isfinite(a) & (a > 0), a, 1.0)
    dinv = 1.0 / jnp.maximum(a * c + b, tiny)

    def stop_test(gs, lc, flat):
        return ((jnp.max(jnp.abs(gs)) < gtol) & jnp.all(lc)) | (flat >= ftol_patience)

    init = _State(xl=xl, xs=xs0, f=f, gs=gs, gl=gl, lconv=lc,
                  S=jnp.zeros((m, s), dtype), Y=jnp.zeros((m, s), dtype),
                  rho=jnp.zeros((m,), dtype), gamma=dinv, theta=jnp.asarray(1.0, dtype),
                  k=jnp.asarray(0), flat=jnp.asarray(0), done=stop_test(gs, lc, 0),
                  stop=~jnp.isfinite(f), reset=jnp.asarray(False),
                  n_inner=jnp.asarray(n0), n_eval=jnp.asarray(1),
                  hist=({k_: jnp.full((max_iter,), jnp.nan, dtype) for k_ in _TRACE_KEYS}
                        if trace else None))

    def cond(st: _State):
        return (~st.done) & (~st.stop) & (st.k < max_iter)

    def body(st: _State) -> _State:
        if shared_newton:
            _, _, Hss = _nt._hessian_probe(grad_fn, st.xl, st.xs, p, s)
            V, inv = _nt._modified_eigh(0.5 * (Hss + Hss.T), eig_floor)
            d = -_nt._apply_inv(V, inv, st.gs)
        else:
            d = _two_loop(st.gs, st, m)
        gp = jnp.dot(st.gs, d)
        steepest = -st.gamma * st.gs
        d = jnp.where((gp < 0) & ~st.reset, d, steepest)
        gp = jnp.dot(st.gs, d)
        dl = (lane_response(st.xl, st.xs, lane_factor(st.xl, st.xs), d)
              if (predictor and p) else jnp.zeros_like(st.xl))

        def trial(t):
            return lane_solve(st.xl + t * dl, st.xs + t * d)

        def ls_cond(c_):
            t, ft = c_[0], c_[1]
            return ~(ft <= st.f + c1 * t * gp) & (c_[-1] < max_ls)

        def ls_body(c_):
            t = c_[0] * shrink
            xl_t, ft, gl_t, gs_t, lc_t, n = trial(t)
            return (t, ft, xl_t, gl_t, gs_t, lc_t, c_[6] + n, c_[7] + 1)

        t0 = jnp.asarray(1.0, dtype)
        xl_t, ft, gl_t, gs_t, lc_t, n = trial(t0)
        t, ft, xl_t, gl_t, gs_t, lc_t, ni, ne = jax.lax.while_loop(
            ls_cond, ls_body, (t0, ft, xl_t, gl_t, gs_t, lc_t, n, jnp.asarray(0)))
        # Accepted on non-increase, as `separable_newton` does: a step that leaves the objective
        # unchanged at the dtype's resolution is what the plateau test counts.
        accept = jnp.isfinite(ft) & (ft <= st.f)
        sv, y = t * d, gs_t - st.gs
        sy, yy = jnp.dot(sv, y), jnp.dot(y, y)
        good = accept & (sy > 1e-10 * yy)

        def push(arr, row):
            return jnp.roll(arr, -1, axis=0).at[m - 1].set(row)

        # A curvature pair is pushed when it keeps the inverse positive definite; a failed line
        # search drops the whole history (and the next step is the scaled gradient).
        S = jnp.where(good, push(st.S, sv), jnp.where(accept, st.S, 0.0))
        Y = jnp.where(good, push(st.Y, y), jnp.where(accept, st.Y, 0.0))
        rho = jnp.where(good, push(st.rho, 1.0 / jnp.where(good, sy, 1.0)),
                        jnp.where(accept, st.rho, 0.0))
        yDy = jnp.dot(y, dinv * y)
        theta = jnp.where(good & (yDy > 0), sy / jnp.where(yDy > 0, yDy, 1.0),
                          jnp.where(accept, st.theta, 1.0))
        xl = jnp.where(accept, xl_t, st.xl)
        xs = jnp.where(accept, st.xs + sv, st.xs)
        f = jnp.where(accept, ft, st.f)
        gs = jnp.where(accept, gs_t, st.gs)
        gl = jnp.where(accept, gl_t, st.gl)
        lc = jnp.where(accept, lc_t, st.lconv)
        unchanged = (st.f - f) <= ftol * jnp.maximum(jnp.maximum(jnp.abs(st.f), jnp.abs(f)), 1.0)
        flat = jnp.where(accept & unchanged, st.flat + 1, jnp.where(accept, 0, st.flat))
        hist = st.hist
        n_inner, n_eval = st.n_inner + ni, st.n_eval + ne + 1
        if trace:
            rec = dict(f=f, t=t, gnorm=jnp.max(jnp.abs(gs)), n_inner=n_inner, n_eval=n_eval,
                       reset=~accept)
            hist = {k_: hist[k_].at[st.k].set(jnp.asarray(v, dtype)) for k_, v in rec.items()}
        return _State(xl=xl, xs=xs, f=f, gs=gs, gl=gl, lconv=lc, S=S, Y=Y, rho=rho,
                      gamma=theta * dinv, theta=theta, k=st.k + 1, flat=flat,
                      done=stop_test(gs, lc, flat), stop=~accept & st.reset, reset=~accept,
                      n_inner=n_inner, n_eval=n_eval, hist=hist)

    st = jax.lax.while_loop(cond, body, init)
    gnorm = jnp.maximum(jnp.max(jnp.abs(st.gs)), jnp.max(jnp.abs(st.gl), initial=0.0))
    result = OptimizeResult(
        x=_nt.lane_unravel(st.xl, st.xs, layout), fun=st.f, grad_norm=gnorm, n_iter=st.k,
        converged=st.done, lane_converged=jnp.broadcast_to(st.done, (K,)),
        trace=(dict(st.hist, n_inner_total=st.n_inner, n_eval_total=st.n_eval)
               if trace else None))
    log_outcome(result, max_iter, gtol, warn_max_iter, solver="L-BFGS/Newton", logger=log,
                ftol=ftol)
    return result
