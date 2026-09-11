"""Newton minimisation for *lane-separable* objectives, in pure JAX.

The objective this is written for is a sum of many small, independent problems that share one
array of data --- the sampler factory's metric regression being the motivating case. There a
block's KL loss is

    L(theta) = sum_d L_d(theta_d),

one term per coordinate ``d`` of the block, each over the ``p <~ 21`` parameters of that
coordinate's row of the mini-language expression (``mimcs/hmc/metric_expr.py``: every atom is
``link(W[d,:] . f + b_d)`` and ``Sum``/``Product`` are elementwise, so ``M_d`` depends on row
``d`` alone). Fitting that with a general L-BFGS over the whole ``block_dim * p`` vector makes
``block_dim`` independent problems share one step length and one correction history, so the fit
proceeds at the pace of its worst coordinate. Here each **lane** gets its own Newton step, its own
step length, and its own convergence test.

**How the per-lane Hessians are obtained.** Write the parameters as ``(K, p)`` --- ``K`` lanes of
``p`` slots. ``L``'s Hessian is then block diagonal with ``K`` blocks ``H_d``, and a
Hessian-vector product with a probe that is **1 in slot ``a`` for every lane** returns, at lane
``d``, column ``a`` of ``H_d``. So ``p`` HVPs of the ordinary whole-array objective give *every*
``H_d`` --- no per-lane loss function, no re-materialising the shared data per lane, and nothing
for the caller to restructure beyond returning its loss per lane instead of summed.

**Shared parameters.** A leaf may instead have length 1 on its lane axis, meaning one value
serving every lane (a weight shared across a block's coordinates). The reduced Hessian is then
*arrow*-structured --- block-diagonal in the per-lane slots, dense in the ``s`` shared ones, with
a coupling block --- and is solved by a Schur complement on the shared corner. The probes for the
coupling must be the **shared** slots: probing a lane slot returns only ``sum_d C[:, (d, a)]``,
which has lost the per-lane resolution, whereas probing shared slot ``i`` returns
``C[i, (d, a)]`` in full in its lane part and ``H_ss[:, i]`` in its shared part. So ``p + s``
probes give the whole arrow matrix.

With shared parameters present a lane can no longer take its own step (a shared step is one
step), so the line search and the convergence test become global. The fully separable case
(``s = 0``) is the fast one and the one the metric regression uses today.

**Globalisation** is a modified Newton: each ``H_d`` is eigendecomposed and its eigenvalues
replaced by ``max(|lam|, eig_floor * max|lam|)``, which is positive definite by construction --- so
the direction is always a descent direction, on an indefinite Hessian as much as a convex one, and
no damping-parameter state machine is needed --- followed by the same backtracking Armijo line
search :mod:`mimcs.optim.lbfgs` uses, vectorised over lanes.

Like the L-BFGS, the whole routine is one ``lax.while_loop`` over fixed shapes, so it is jittable;
it is written for offline use against a batch of evidence, not as an inner-loop kernel.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import Array

from .._logging import get_logger
from ._common import OptimizeResult, log_outcome

log = get_logger(__name__)


class _LaneLayout(NamedTuple):
    """How a parameter pytree maps onto ``(K, p)`` lane slots plus ``(s,)`` shared slots."""

    n_lanes: int
    n_slots: int                 # p, per lane
    n_shared: int                # s
    treedef: object
    specs: tuple                 # per leaf: (is_lane, shape, width, offset)


def lane_layout(params, n_lanes: int) -> _LaneLayout:
    """Describe ``params`` as lane-major slots; raises if a leaf does not fit the contract.

    Every leaf's axis 0 must be the lane axis --- length ``n_lanes`` (a per-lane parameter) or 1
    (a parameter shared by every lane). This is the invariant the metric mini-language already
    satisfies by construction (see :func:`mimcs.adaptation.metric._per_coord_size`, which relies
    on the same fact); stating it as a checked precondition here is what makes the block-diagonal
    Hessian a theorem rather than a hope. A length-1 leaf is read as *lane* when ``n_lanes == 1``,
    since there is then nothing to share it with.
    """
    leaves, treedef = jax.tree_util.tree_flatten(params)
    specs, lane_off, shared_off = [], 0, 0
    for leaf in leaves:
        a = jnp.asarray(leaf)
        if a.ndim == 0 or a.shape[0] not in (n_lanes, 1):
            raise ValueError(
                f"separable_newton: every parameter leaf's axis 0 must be the lane axis, of "
                f"length {n_lanes} (per lane) or 1 (shared); got shape {a.shape}")
        is_lane = a.shape[0] == n_lanes
        width = int(np.prod(a.shape[1:], dtype=np.int64)) if is_lane else int(a.size)
        if is_lane:
            specs.append((True, a.shape, width, lane_off))
            lane_off += width
        else:
            specs.append((False, a.shape, width, shared_off))
            shared_off += width
    return _LaneLayout(n_lanes, lane_off, shared_off, treedef, tuple(specs))


def lane_ravel(params, layout: _LaneLayout):
    """``params`` -> ``(x_lane (K, p), x_shared (s,))``."""
    leaves = jax.tree_util.tree_leaves(params)
    lane, shared = [], []
    for leaf, (is_lane, _shape, _w, _off) in zip(leaves, layout.specs):
        a = jnp.asarray(leaf)
        (lane if is_lane else shared).append(
            a.reshape(layout.n_lanes, -1) if is_lane else a.reshape(-1))
    dtype = jnp.result_type(*(lane + shared)) if (lane or shared) else jnp.result_type(float)
    x_lane = (jnp.concatenate(lane, axis=1) if lane
              else jnp.zeros((layout.n_lanes, 0), dtype))
    x_shared = jnp.concatenate(shared) if shared else jnp.zeros((0,), dtype)
    return x_lane, x_shared


def lane_unravel(x_lane: Array, x_shared: Array, layout: _LaneLayout):
    """The inverse of :func:`lane_ravel`: flat lane/shared arrays -> the original pytree."""
    leaves = []
    for is_lane, shape, width, off in layout.specs:
        if is_lane:
            leaves.append(x_lane[:, off:off + width].reshape(shape))
        else:
            leaves.append(x_shared[off:off + width].reshape(shape))
    return jax.tree_util.tree_unflatten(layout.treedef, leaves)


def _modified_eigh(H: Array, eig_floor: float):
    """Eigendecompose ``H`` and floor ``|eigenvalue|``; returns ``(V, 1 / floored)``.

    Applying ``V diag(1/floored) V^T`` is the inverse of a positive-definite *modification* of
    ``H`` --- the classic modified Newton. Flooring the **absolute** eigenvalue (rather than
    clipping negatives to a constant) keeps the step scale-aware in a direction of strong negative
    curvature instead of launching it to the floor's reciprocal, and the floor is relative to the
    spectrum's own scale so it means the same thing whatever the objective's units.
    """
    lam, V = jnp.linalg.eigh(H)
    # ``initial=0.0``: an expression whose parameters are *all* shared leaves the per-lane block
    # empty (``p == 0``), and a reduction over a zero-size axis has no identity to fall back on.
    scale = jnp.max(jnp.abs(lam), axis=-1, keepdims=True, initial=0.0)
    tiny = jnp.asarray(jnp.finfo(lam.dtype).tiny, lam.dtype)
    floored = jnp.maximum(jnp.abs(lam), jnp.maximum(eig_floor * scale, tiny))
    return V, 1.0 / floored


def _apply_inv(V: Array, inv_lam: Array, y: Array) -> Array:
    """``V diag(inv_lam) V^T y``, batched over every leading axis of ``y``."""
    return jnp.einsum("...ij,...j->...i", V, inv_lam * jnp.einsum("...ji,...j->...i", V, y))


def _hessian_probe(grad_fn, x_lane: Array, x_shared: Array, p: int, s: int):
    """The arrow Hessian by probe HVPs: ``(H (K, p, p), C (s, K, p), H_ss (s, s))``.

    ``p`` lane probes and ``s`` shared probes, each a forward-over-reverse pass. Run as a
    ``lax.scan`` over slots rather than a ``vmap`` over all probes at once. The reason to expect
    that to be *slower* is real --- ``vmap`` computes the primal once and batches the tangents,
    where the scan recomputes it per probe --- and it was chosen anyway for memory, since the
    batched intermediates are ``p`` times a gradient's and this fit is the memory high-water mark
    of a second-round ``analyze`` (``mimcs/_chunked.py``).

    **Measured, and the prediction was wrong in the half that matters.** ``vmap`` is faster only
    at tiny ``p``, and loses at the ``p`` where the cost is actually paid (float32, one Hessian
    assembly, scan/vmap):

        N=2000  K= 30  p= 2:   1.60 /   1.07 ms   vmap 1.49x faster
        N=2000  K=200  p= 2:   8.32 /   4.95 ms   vmap 1.68x faster
        N=4000  K=100  p=11:  19.02 /  28.17 ms   scan 1.47x faster
        N=2000  K=400  p=20:  76.77 / 121.86 ms   scan 1.59x faster

    Same mechanism ``_chunked`` found: past a point the batched tangent working set stops fitting
    and the memory traffic dominates the arithmetic the sharing was meant to save. So the scan is
    not a memory-for-speed trade here --- recorded so the next reader does not "fix" it.
    """
    K = x_lane.shape[0]
    dtype = x_lane.dtype
    zeros_lane, zeros_shared = jnp.zeros_like(x_lane), jnp.zeros_like(x_shared)

    def hvp(v_lane, v_shared):
        return jax.jvp(grad_fn, (x_lane, x_shared), (v_lane, v_shared))[1]

    def lane_probe(_, e):                       # e: (p,) one-hot, the SAME slot in every lane
        gl, _gs = hvp(jnp.broadcast_to(e, (K, p)), zeros_shared)
        return None, gl                         # (K, p): column `a` of every H_d
    _, cols = jax.lax.scan(lane_probe, None, jnp.eye(p, dtype=dtype))
    H = jnp.moveaxis(cols, 0, -1)               # (K, p, p)
    H = 0.5 * (H + jnp.swapaxes(H, -1, -2))     # symmetrise away the AD round-off

    if s == 0:
        return H, jnp.zeros((0, K, p), dtype), jnp.zeros((0, 0), dtype)

    def shared_probe(_, e):                     # e: (s,) one-hot
        return None, hvp(zeros_lane, e)
    _, (C, Hss_cols) = jax.lax.scan(shared_probe, None, jnp.eye(s, dtype=dtype))
    Hss = 0.5 * (Hss_cols + Hss_cols.T)
    return H, C, Hss


def _backtrack(f_at: Callable, f0: Array, gp: Array, *, c1: float, shrink: float, max_ls: int):
    """Vectorised backtracking Armijo: a step ``t`` with ``f(x + t p) <= f + c1 t g.p``.

    ``f0``, ``gp`` and the return share a shape --- ``(K,)`` for a per-lane search, ``()`` for a
    global one --- and only the entries that still fail the condition are shrunk. A non-finite
    ``gp`` (a lane whose objective blew up) is left alone; a non-finite trial value fails the
    comparison and therefore keeps shrinking, which is the wanted behaviour.
    """
    def ok(t, ft):
        return (ft <= f0 + c1 * t * gp) | ~jnp.isfinite(gp)

    def cond(state):
        t, ft, i = state
        return (~jnp.all(ok(t, ft))) & (i < max_ls)

    def body(state):
        t, ft, i = state
        t = jnp.where(ok(t, ft), t, t * shrink)
        return t, f_at(t), i + 1

    t0 = jnp.ones_like(f0)
    t, _, _ = jax.lax.while_loop(cond, body, (t0, f_at(t0), jnp.asarray(0)))
    return t


class _State(NamedTuple):
    xl: Array          # (K, p) lane parameters
    xs: Array          # (s,)   shared parameters
    f: Array           # (K,)   per-lane objective
    gl: Array          # (K, p)
    gs: Array          # (s,)
    active: Array      # (K,) bool --- lanes still being worked on
    k: Array


def separable_newton(loss_vec: Callable, x0, *, max_iter: int = 100, gtol: float | None = None,
                     max_ls: int = 25, c1: float = 1e-4, shrink: float = 0.5,
                     eig_floor: float = 1e-8,
                     warn_max_iter: bool = True) -> OptimizeResult:
    """Minimise ``sum(loss_vec(x))`` by a per-lane modified Newton.

    Args:
        loss_vec: the objective **per lane**, ``loss_vec(x) -> (K,)``. The contract that licenses
            the block-diagonal Hessian: entry ``k`` must depend only on lane ``k`` of every
            per-lane leaf of ``x``, plus the shared leaves.
        x0: initial parameters, any pytree whose leaves all have axis 0 of length ``K`` (per lane)
            or 1 (shared by every lane); ``K`` is read from ``loss_vec(x0)``.
        max_iter: maximum outer iterations.
        gtol: a lane stops when its own gradient max-norm falls below this (with shared
            parameters present, when the whole gradient does). ``None`` (the default) reads it
            from the parameters' dtype as ``sqrt(eps)`` --- ~3.5e-4 in float32, ~1.5e-8 under x64.
            A **fixed** tolerance cannot serve both: this library is float32 by default and x64 on
            request (``docs/design/15``), and L-BFGS's fixed 1e-6 is unreachable in float32, so
            every lane would run its tail out against a threshold it cannot meet. Newton converges
            quadratically, so the last iteration typically takes the gradient from ~1e-3 to below
            whichever of these it is aiming at --- the threshold costs iterations, not accuracy.
        max_ls: maximum backtracking steps per line search.
        c1: Armijo sufficient-decrease constant.
        shrink: line-search step shrink factor.
        eig_floor: eigenvalue floor for the Hessian modification, relative to each lane's own
            spectral radius.
        warn_max_iter: log a WARNING when the iteration cap is reached with lanes unconverged.

    Returns:
        An :class:`~mimcs.optim.OptimizeResult` whose ``x`` has the structure of ``x0``, whose
        ``fun`` is the summed objective, and whose ``lane_converged`` is the per-lane flag vector.
    """
    # ``eval_shape``, not a real call: the lane count is a *shape* question, and the objective is
    # a pass over the whole evidence. Asking for it costs nothing this way.
    out = jax.eval_shape(loss_vec, x0)
    if len(out.shape) != 1:
        raise ValueError(f"separable_newton: loss_vec must return a 1-D per-lane vector, got "
                         f"shape {out.shape}")
    K = int(out.shape[0])
    layout = lane_layout(x0, K)
    p, s = layout.n_slots, layout.n_shared
    if gtol is None:
        dtype = jnp.result_type(*(jax.tree_util.tree_leaves(x0) or [float]))
        gtol = float(np.sqrt(np.finfo(dtype).eps))
    log.debug("Newton on %d lane(s) x %d slot(s) + %d shared: max_iter=%d, gtol=%.3g",
              K, p, s, max_iter, gtol)

    def lossv(zl, zs):
        return jnp.asarray(loss_vec(lane_unravel(zl, zs, layout)))

    def value_and_grad(zl, zs):
        """The per-lane value vector and the gradient of its sum, in one reverse pass."""
        fv, vjp = jax.vjp(lossv, zl, zs)
        gl, gs = vjp(jnp.ones_like(fv))
        return fv, gl, gs

    grad_fn = jax.grad(lambda zl, zs: jnp.sum(lossv(zl, zs)), argnums=(0, 1))

    def lane_gnorm(gl, gs):
        """Each lane's own gradient max-norm --- or, with shared parameters, one global norm
        broadcast over the lanes (a shared slot belongs to no single lane)."""
        n = jnp.max(jnp.abs(gl), axis=1, initial=0.0)
        if s == 0:
            return n
        return jnp.full((K,), jnp.maximum(jnp.max(n, initial=0.0),
                                          jnp.max(jnp.abs(gs), initial=0.0)))

    def direction(st: _State):
        """The modified-Newton direction, zeroed on inactive lanes so they stay frozen."""
        H, C, Hss = _hessian_probe(grad_fn, st.xl, st.xs, p, s)
        eye = jnp.eye(p, dtype=H.dtype)
        good = jnp.all(jnp.isfinite(H), axis=(-2, -1))            # (K,)
        H = jnp.where(good[:, None, None], H, eye)
        V, inv_lam = _modified_eigh(H, eig_floor)

        if s == 0:
            dl = -_apply_inv(V, inv_lam, st.gl)
            ds = st.gs
        else:
            ainv_g = _apply_inv(V, inv_lam, st.gl)                # (K, p)
            ainv_C = _apply_inv(V[None], inv_lam[None], C)        # (s, K, p)
            S = Hss - jnp.einsum("ikp,jkp->ij", C, ainv_C)
            rhs = -st.gs + jnp.einsum("ikp,kp->i", C, ainv_g)
            Vs, inv_s = _modified_eigh(0.5 * (S + S.T), eig_floor)
            ds = _apply_inv(Vs, inv_s, rhs)
            dl = -ainv_g - _apply_inv(V, inv_lam, jnp.einsum("ikp,i->kp", C, ds))

        # Guard a non-descent direction exactly as `mimcs.optim.lbfgs` does: fall back to
        # steepest descent rather than trusting a curvature model that points uphill.
        gp_lane = jnp.sum(st.gl * dl, axis=1)                # (K,)
        if s == 0:
            uphill = ~(gp_lane < 0) | ~good                  # each lane judged on its own step
        else:
            # One step, one test --- and the shared term is added **once**, not once per lane.
            gp_total = jnp.sum(gp_lane) + jnp.sum(st.gs * ds)
            uphill = jnp.full((K,), ~(gp_total < 0) | ~jnp.all(good))
            ds = jnp.where(uphill[0], -st.gs, ds)
        dl = jnp.where(uphill[:, None], -st.gl, dl)
        dl = jnp.where(st.active[:, None], dl, 0.0)
        return dl, (ds if s else st.xs * 0.0)

    def cond(st: _State):
        return jnp.any(st.active) & (st.k < max_iter)

    def body(st: _State) -> _State:
        dl, ds = direction(st)
        gp_lane = jnp.sum(st.gl * dl, axis=1)

        if s == 0:
            t = _backtrack(lambda tt: lossv(st.xl + tt[:, None] * dl, st.xs),
                           st.f, gp_lane, c1=c1, shrink=shrink, max_ls=max_ls)
            xl_try, xs_try = st.xl + t[:, None] * dl, st.xs
        else:
            gp = jnp.sum(gp_lane) + jnp.sum(st.gs * ds)
            t = _backtrack(lambda tt: jnp.sum(lossv(st.xl + tt * dl, st.xs + tt * ds)),
                           jnp.sum(st.f), gp, c1=c1, shrink=shrink, max_ls=max_ls)
            xl_try, xs_try = st.xl + t * dl, st.xs + t * ds

        f_try, gl_try, gs_try = value_and_grad(xl_try, xs_try)

        # A lane is only moved if the trial actually improved it. A lane that did not is not
        # merely left where it is but **retired**: nothing about it will change on the next
        # iteration, so re-deriving the same rejected step would just burn the iteration budget.
        if s:
            # The line search minimised the **total**, so the total is what has to have improved:
            # a global step routinely lifts an individual lane while lowering the sum, and testing
            # every lane would reject the step the search just certified. (It did, on the first
            # float32 run: one lane up by a rounding step retired the whole solve at iteration 1.)
            total_try, total_old = jnp.sum(f_try), jnp.sum(st.f)
            accept = jnp.full((K,), jnp.isfinite(total_try) & (total_try <= total_old))
        else:
            accept = jnp.isfinite(f_try) & (f_try <= st.f)
        accept = accept & st.active
        # The rejected lanes keep their previous f and gradient exactly --- they did not move --
        # so no re-evaluation is needed to restore them.
        xl = jnp.where(accept[:, None], xl_try, st.xl)
        f = jnp.where(accept, f_try, st.f)
        gl = jnp.where(accept[:, None], gl_try, st.gl)
        if s:
            xs = jnp.where(accept[0], xs_try, st.xs)
            gs = jnp.where(accept[0], gs_try, st.gs)
        else:
            xs, gs = st.xs, st.gs

        converged = lane_gnorm(gl, gs) < gtol
        active = st.active & accept & ~converged
        return _State(xl=xl, xs=xs, f=f, gl=gl, gs=gs, active=active, k=st.k + 1)

    xl0, xs0 = lane_ravel(x0, layout)
    f0, gl0, gs0 = value_and_grad(xl0, xs0)
    init = _State(xl=xl0, xs=xs0, f=f0, gl=gl0, gs=gs0,
                  active=jnp.isfinite(f0) & ~(lane_gnorm(gl0, gs0) < gtol),
                  k=jnp.asarray(0))
    st = jax.lax.while_loop(cond, body, init)

    lane_conv = lane_gnorm(st.gl, st.gs) < gtol
    result = OptimizeResult(
        x=lane_unravel(st.xl, st.xs, layout), fun=jnp.sum(st.f),
        grad_norm=jnp.max(lane_gnorm(st.gl, st.gs), initial=0.0), n_iter=st.k,
        converged=jnp.all(lane_conv), lane_converged=lane_conv)
    n_bad = _unconverged_count(lane_conv)
    log_outcome(result, max_iter, gtol, warn_max_iter, solver="Newton", logger=log,
                detail=("" if n_bad is None else f"{n_bad}/{K} lane(s) unconverged."))
    return result


def _unconverged_count(lane_conv):
    """How many lanes are unconverged, or ``None`` under a tracer (nothing to report)."""
    try:
        return int(np.size(np.asarray(lane_conv)) - np.count_nonzero(np.asarray(lane_conv)))
    except Exception:
        return None


def newton_minimize(fun: Callable, x0, **kwargs) -> OptimizeResult:
    """Minimise a scalar ``fun`` over any pytree by a modified Newton --- the one-lane case.

    A thin wrapper on :func:`separable_newton`: give every leaf a leading axis of length 1 and the
    lane machinery degenerates to a single full ``n x n`` modified-Newton solve, with the same
    eigenvalue flooring and Armijo line search. A drop-in alternative to
    :func:`mimcs.optim.minimize` when ``n`` is small enough for an ``n x n`` Hessian (the cost is
    ``n`` forward-over-reverse passes per iteration).
    """
    lifted = jax.tree_util.tree_map(lambda a: jnp.asarray(a)[None], x0)

    def loss_vec(p):
        return jnp.reshape(fun(jax.tree_util.tree_map(lambda a: a[0], p)), (1,))

    res = separable_newton(loss_vec, lifted, **kwargs)
    return res._replace(x=jax.tree_util.tree_map(lambda a: a[0], res.x))
