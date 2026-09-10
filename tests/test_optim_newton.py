"""Tests for the Newton minimisers (``mimcs.optim.separable_newton`` / ``newton_minimize``).

These run at the library's default float32 (``docs/design``: x64 is opt-in), so the tolerances are
float32 tolerances and ``gtol`` is left at its default, which the solver reads from the dtype.
Asking for 1e-10 here would not be a stricter test, it would be an unreachable one.

Three things need pinning, in rising order of how easy they would be to get silently wrong:

1. that a lane-separable problem really is solved lane by lane (a quadratic in one step, whatever
   the spread of curvature across lanes, where the joint L-BFGS control visibly is not);
2. that the eigenvalue flooring turns a *negative*-curvature step around instead of climbing to
   the nearest stationary point, which plain Newton would do;
3. that the **arrow** path with parameters shared across lanes reproduces the true Hessian and the
   closed-form minimiser --- with the control that makes that non-vacuous, since the natural wrong
   way to probe the coupling block returns something that looks perfectly plausible.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from mimcs.optim import minimize, separable_newton, newton_minimize
from mimcs.optim.newton import _hessian_probe
from mimcs.hmc.metric_expr import Exp


# --- the separable case ------------------------------------------------------- #

def _lane_quadratics(K, p, seed=0, spread=6):
    """``K`` independent quadratics whose curvatures span ``10**spread``."""
    rng = np.random.default_rng(seed)
    A = np.stack([np.eye(p) * (10.0 ** (spread * k / max(1, K - 1))) for k in range(K)])
    xstar = rng.normal(size=(K, p))
    return A, xstar


def test_solves_independent_quadratics_in_one_step():
    """Each lane is exactly quadratic, so one Newton step is the answer --- and the lanes'
    curvatures spanning a millionfold does not slow any of them down."""
    K, p = 8, 3
    A, xstar = _lane_quadratics(K, p)
    Aj, xj = jnp.asarray(A), jnp.asarray(xstar)
    lv = lambda x: 0.5 * jnp.einsum("kij,ki,kj->k", Aj, x - xj, x - xj)

    res = separable_newton(lv, jnp.zeros((K, p)))
    # One step is the mathematics; the second is float32 arithmetic checking it. At a millionfold
    # curvature spread the exact step still leaves a residual gradient of order
    # ``|A| * eps * |x*|`` in the stiffest lane, which is above ``sqrt(eps)`` --- so the solver
    # takes one more (no-op) iteration to see the gradient fall. Under x64 this is exactly 1.
    assert int(res.n_iter) <= 2
    assert bool(res.converged) and np.all(np.asarray(res.lane_converged))
    assert np.allclose(np.asarray(res.x), xstar, atol=1e-5)


def test_joint_lbfgs_is_the_control_that_does_not_keep_up():
    """The same problem, same iteration budget, one *joint* L-BFGS: the shared step length and
    shared history are exactly what the per-lane solve removes, so this must fall well short.
    Without this control the test above only says "Newton solves quadratics"."""
    K, p = 8, 3
    A, xstar = _lane_quadratics(K, p)
    Aj, xj = jnp.asarray(A), jnp.asarray(xstar)
    lv = lambda x: 0.5 * jnp.einsum("kij,ki,kj->k", Aj, x - xj, x - xj)

    newton = separable_newton(lv, jnp.zeros((K, p)), max_iter=5)
    lbfgs = minimize(lambda x: jnp.sum(lv(x)), jnp.zeros((K, p)),
                     max_iter=5, gtol=1e-6, warn_max_iter=False)
    assert bool(newton.converged) and not bool(lbfgs.converged)
    assert float(newton.grad_norm) < float(lbfgs.grad_norm)


def test_flooring_turns_a_negative_curvature_step_around():
    """``f(x) = x^4/4 - x^2/2`` started just off its local *maximum*. The Hessian is negative
    there, so an unmodified Newton step points back at the maximum; flooring ``|eigenvalue|``
    flips it and every lane walks out to one of the two minima at ``+-1``."""
    x0 = jnp.asarray([[0.1], [-0.1], [0.05], [-0.3]])
    lv = lambda x: jnp.sum(0.25 * x ** 4 - 0.5 * x ** 2, axis=1)

    res = separable_newton(lv, x0)
    got = np.asarray(res.x).ravel()
    assert bool(res.converged)
    assert np.allclose(np.abs(got), 1.0, atol=1e-4), got
    assert np.all(np.sign(got) == np.sign(np.asarray(x0).ravel()))   # each stayed on its side


def test_a_lane_that_starts_converged_never_moves():
    """A lane already at its optimum is inactive from the first iteration, and freezing means
    frozen --- bit-for-bit, not merely 'close'.

    The minimiser is chosen to be exactly representable in float32. That is not tidiness: with a
    minimiser that is only *nearly* representable the lane is genuinely not at its optimum, so it
    would move, and the test would be asserting the wrong thing about a correct solver."""
    K, p = 4, 2
    A, _ = _lane_quadratics(K, p, seed=1)
    xstar = np.array([[0.5, -0.25], [1.0, 2.0], [-0.75, 0.125], [3.0, -1.5]], dtype=np.float32)
    Aj, xj = jnp.asarray(A), jnp.asarray(xstar)
    lv = lambda x: 0.5 * jnp.einsum("kij,ki,kj->k", Aj, x - xj, x - xj)

    x0 = np.zeros((K, p), dtype=np.float32)
    x0[0] = xstar[0]                                   # lane 0 starts at its minimiser
    res = separable_newton(lv, jnp.asarray(x0))
    assert np.array_equal(np.asarray(res.x)[0], x0[0])
    assert np.allclose(np.asarray(res.x)[1:], xstar[1:], atol=1e-4)


def test_accepts_a_pytree_of_lanes():
    K = 5
    lv = lambda p: 0.5 * jnp.sum((p["W"] - 2.0) ** 2, axis=1) + 0.5 * (p["b"] + 1.0) ** 2
    res = separable_newton(lv, {"W": jnp.zeros((K, 3)), "b": jnp.zeros((K,))})
    assert set(res.x) == {"W", "b"}
    assert np.allclose(np.asarray(res.x["W"]), 2.0, atol=1e-6)
    assert np.allclose(np.asarray(res.x["b"]), -1.0, atol=1e-6)


def test_is_jittable():
    lv = lambda x: jnp.sum((x - 1.0) ** 2, axis=1)
    res = jax.jit(lambda x0: separable_newton(lv, x0))(jnp.zeros((4, 2)))
    assert bool(res.converged)
    assert np.allclose(np.asarray(res.x), 1.0, atol=1e-6)


def test_rejects_a_leaf_whose_axis_0_is_not_the_lane_axis():
    """The block-diagonal Hessian is a theorem *given* the layout, so the layout is checked
    rather than assumed --- silently mis-slicing it would give a plausible wrong answer."""
    bad = {"a": jnp.zeros((5, 2)), "oops": jnp.zeros((3,))}
    with pytest.raises(ValueError, match="lane axis"):
        separable_newton(lambda p: jnp.sum(p["a"] ** 2, axis=1), bad)


# --- the arrow (shared-parameter) case ---------------------------------------- #

def _arrow_quadratic(K, p, s, seed=3):
    """An SPD ``Q`` with every lane-lane cross block zeroed: the arrow structure by construction."""
    rng = np.random.default_rng(seed)
    n = K * p + s
    B = rng.normal(size=(n, n))
    Q = B @ B.T + n * np.eye(n)
    for d in range(K):
        for e in range(K):
            if d != e:
                Q[d * p:(d + 1) * p, e * p:(e + 1) * p] = 0.0
    return Q, rng.normal(size=n)


def test_probes_reconstruct_the_arrow_hessian_exactly():
    """``p`` lane probes + ``s`` shared probes give every block of the arrow matrix."""
    K, p, s = 4, 3, 2
    Q, _c = _arrow_quadratic(K, p, s)
    Qj = jnp.asarray(Q)
    total = lambda zl, zs: 0.5 * (lambda z: z @ Qj @ z)(jnp.concatenate([zl.reshape(-1), zs]))
    grad_fn = jax.grad(total, argnums=(0, 1))

    H, C, Hss = (np.asarray(a) for a in
                 _hessian_probe(grad_fn, jnp.zeros((K, p)), jnp.zeros((s,)), p, s))
    R = np.zeros_like(Q)
    for d in range(K):
        R[d * p:(d + 1) * p, d * p:(d + 1) * p] = H[d]
        R[K * p:, d * p:(d + 1) * p] = C[:, d, :]
        R[d * p:(d + 1) * p, K * p:] = C[:, d, :].T
    R[K * p:, K * p:] = Hss
    assert np.allclose(R, Q, rtol=1e-4, atol=1e-4)


def test_a_lane_probe_cannot_give_the_coupling_block():
    """The control for the test above. Probing a *lane* slot and reading the shared part of the
    HVP looks like the obvious way to get the coupling ``C``, and returns a perfectly plausible
    ``(s, p)`` array --- but it is ``sum_d C[:, d, :]``, having lost the per-lane resolution
    entirely. That is why the shared slots are the ones probed."""
    K, p, s = 4, 3, 2
    Q, _c = _arrow_quadratic(K, p, s)
    Qj = jnp.asarray(Q)
    total = lambda zl, zs: 0.5 * (lambda z: z @ Qj @ z)(jnp.concatenate([zl.reshape(-1), zs]))
    grad_fn = jax.grad(total, argnums=(0, 1))
    _H, C, _Hss = _hessian_probe(grad_fn, jnp.zeros((K, p)), jnp.zeros((s,)), p, s)

    zeros_s = jnp.zeros((s,))
    lane_probed = np.stack(
        [np.asarray(jax.jvp(grad_fn, (jnp.zeros((K, p)), zeros_s),
                            (jnp.broadcast_to(jnp.eye(p)[a], (K, p)), zeros_s))[1][1])
         for a in range(p)], axis=-1)                       # (s, p)
    assert np.allclose(lane_probed, np.asarray(C).sum(axis=1), rtol=1e-4, atol=1e-4)  # what it is
    assert not np.allclose(lane_probed, np.asarray(C)[:, 0, :])             # not any one lane's


def test_shared_parameters_reach_the_closed_form_minimiser():
    """A quadratic over ``K`` lanes plus ``s`` parameters shared by all of them: the Schur solve
    must land on ``Q^-1 c`` exactly, in one step."""
    K, p, s = 5, 3, 2
    Q, c = _arrow_quadratic(K, p, s, seed=7)
    Qj, cj = jnp.asarray(Q), jnp.asarray(c)

    def lv(pr):
        th, ph = pr["theta"], pr["phi"][0]
        shared_part = (0.5 * ph @ Qj[K * p:, K * p:] @ ph - cj[K * p:] @ ph) / K
        def lane(d, zd):
            return (0.5 * zd @ Qj[d * p:(d + 1) * p, d * p:(d + 1) * p] @ zd
                    + zd @ Qj[d * p:(d + 1) * p, K * p:] @ ph
                    - cj[d * p:(d + 1) * p] @ zd + shared_part)
        return jnp.stack([lane(d, th[d]) for d in range(K)])

    res = separable_newton(lv, {"theta": jnp.zeros((K, p)), "phi": jnp.zeros((1, s))},
                           max_iter=50)
    got = np.concatenate([np.asarray(res.x["theta"]).reshape(-1), np.asarray(res.x["phi"])[0]])
    assert int(res.n_iter) == 1 and bool(res.converged)
    assert np.allclose(got, np.linalg.solve(Q, c), atol=1e-4)


# --- the one-lane wrapper ----------------------------------------------------- #

def test_newton_minimize_solves_a_quadratic_in_one_step():
    A = jnp.diag(jnp.array([1.0, 10.0, 100.0, 1000.0]))
    xstar = jnp.array([1.0, -2.0, 3.0, 0.5])
    res = newton_minimize(lambda x: 0.5 * (x - xstar) @ A @ (x - xstar), jnp.zeros(4))
    assert int(res.n_iter) == 1 and bool(res.converged)
    assert np.allclose(np.asarray(res.x), np.asarray(xstar), atol=1e-5)


def test_newton_minimize_solves_rosenbrock():
    """The two-variable banana, whose only stationary point is the minimum at ``(1, 1)``.

    Deliberately *not* the four-variable chain ``tests/test_optim.py`` gives L-BFGS: that variant
    has other local minima (verified --- the Hessian there is positive definite), and a Newton
    method converges to whichever stationary point it is led to, so it would be testing the
    starting point rather than the solver."""
    rosen = lambda x: (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2
    res = newton_minimize(rosen, jnp.array([-1.2, 1.0]), max_iter=200)
    assert bool(res.converged)
    assert np.allclose(np.asarray(res.x), 1.0, atol=1e-3), res.x


def test_newton_minimize_accepts_a_pytree():
    target = {"a": jnp.array([1.0, 2.0]), "b": jnp.array(-3.0)}
    fun = lambda p: jnp.sum((p["a"] - target["a"]) ** 2) + (p["b"] - target["b"]) ** 2
    res = newton_minimize(fun, {"a": jnp.zeros(2), "b": jnp.array(0.0)})
    assert np.allclose(np.asarray(res.x["a"]), [1.0, 2.0], atol=1e-6)
    assert np.isclose(float(res.x["b"]), -3.0, atol=1e-6)
    assert np.shape(res.x["b"]) == ()          # the lifted axis is stripped back off


# --- the objective this exists for -------------------------------------------- #

def test_fits_the_metric_kl_loss_lane_by_lane():
    """The factory use case, several coordinates at once: fit ``M_d(v) = exp(W_d v + b_d)`` to
    data whose conditional score second moment is ``e^{-v}``. The optimum is ``W -> -1, b -> 0``
    for every coordinate, and it should take a handful of iterations, not hundreds."""
    rng = np.random.default_rng(0)
    N, K = 4000, 6
    v = rng.normal(0.0, 1.5, size=N)
    g = rng.normal(0.0, 1.0, size=(N, K)) * np.exp(-0.5 * v)[:, None]
    gj, vj = jnp.asarray(g), jnp.asarray(v)
    expr = Exp("v")

    def loss_vec(params):
        def row(g_row, vv):
            M = expr.evaluate(params, {"v": vv.reshape(1)})
            return 0.5 * (jnp.log(M) + g_row ** 2 / M)
        return jnp.mean(jax.vmap(row)(gj, vj), axis=0)

    p0 = expr.init_params(K, {"v": 1}, target=jnp.mean(gj ** 2, axis=0))
    res = separable_newton(loss_vec, p0)
    assert bool(res.converged) and int(res.n_iter) < 25
    assert np.allclose(np.asarray(res.x["W"][0]).ravel(), -1.0, atol=0.1)
    assert np.allclose(np.asarray(res.x["b"]), 0.0, atol=0.1)
