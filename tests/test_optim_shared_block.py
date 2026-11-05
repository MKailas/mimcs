"""Tests for ``lbfgs_newton`` (L-BFGS on a shared block, Newton on the lanes) and its routing.

The solver is a study arm, not a default (``regression.SHARED_SOLVER = None``), so what is pinned
here is that every option reaches the optimum ``separable_newton`` reaches on convex problems,
that the routing engages only above its threshold and leaves every other fit bit-identical, and
the diagnostics both solvers' ``trace=True`` record without changing the solve.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from mimcs import config
from mimcs.optim import separable_newton, lbfgs_newton
from mimcs.factory import regression
from mimcs.hmc.metric_expr import Exp, Sigmoid, SpExp

ARMS = {
    "alt": dict(inner_max_iter=1),
    "alt_pc": dict(inner_max_iter=1, curvature=True),
    "pred_pc": dict(inner_max_iter=20, predictor=True, curvature=True),
    "altN": dict(inner_max_iter=1, shared_newton=True),
    "m300p": dict(inner_max_iter=1, curvature=True, m=300, precise=True),
    "m300p_n5": dict(inner_max_iter=1, curvature=True, m=300, precise=True, polish=5),
}


@pytest.fixture
def x64():
    was = config.x64_enabled()
    config.enable_x64(True)
    try:
        yield
    finally:
        config.enable_x64(was)


@pytest.fixture(autouse=True)
def _no_kick_guard(monkeypatch):
    """Synthetic dependency columns carry no score of their own (see test_metric_kick_guard.py)."""
    monkeypatch.setattr(regression, "KICK_GUARD", None)


def _arrow(K=40, p=2, s=30, N=200, seed=0):
    """A convex arrow problem: per-lane log-metric ``X w + b0 + Z b1``, ``w`` shared."""
    rng = np.random.default_rng(seed)
    X = jnp.asarray(rng.normal(size=(N, s)))
    Z = jnp.asarray(rng.normal(size=(N, K)))
    Y = jnp.asarray(rng.normal(size=(N, K)) * np.exp(0.3 * rng.normal(size=(N, K))))

    def loss_vec(prm):
        u = (X @ prm["w"][0])[:, None] + prm["b"][:, 0] + Z * prm["b"][:, 1]
        return (jnp.mean(0.5 * (u + Y ** 2 * jnp.exp(-u)), axis=0)
                + 1e-3 * (jnp.sum(prm["w"] ** 2) / K + jnp.sum(prm["b"] ** 2, 1)))
    x0 = {"w": jnp.zeros((1, s)), "b": jnp.zeros((K, p))}
    curv = {"w": jnp.ones((1, s)), "b": jnp.zeros((K, p))}
    return loss_vec, x0, curv


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_every_arm_reaches_the_newton_optimum_on_a_convex_arrow_problem(x64, arm):
    loss_vec, x0, curv = _arrow()
    ref = separable_newton(loss_vec, x0, max_iter=500)
    kw = dict(ARMS[arm])
    if kw.pop("curvature", False):
        kw["curvature"] = (curv, 2e-3)
    res = lbfgs_newton(loss_vec, x0, max_iter=2000, **kw)
    assert bool(res.converged)
    assert float(res.fun) == pytest.approx(float(ref.fun), rel=1e-12, abs=1e-12)
    np.testing.assert_allclose(res.x["w"], ref.x["w"], atol=1e-6)


def test_without_shared_leaves_it_is_separable_newton():
    loss_vec, x0, _ = _arrow()
    lanes_only = {"b": x0["b"]}

    def lv(prm):
        return loss_vec({"w": jnp.zeros_like(x0["w"]), "b": prm["b"]})
    a = separable_newton(lv, lanes_only)
    b = lbfgs_newton(lv, lanes_only)
    np.testing.assert_array_equal(a.x["b"], b.x["b"])
    assert float(a.fun) == float(b.fun)


def test_every_parameter_shared_works(x64):
    """No lane leaves at all (the fully pooled rung): the lane solve is empty."""
    loss_vec, x0, _ = _arrow()

    def lv(prm):
        return loss_vec({"w": prm["w"], "b": jnp.broadcast_to(prm["b"], x0["b"].shape)})
    pooled = {"w": x0["w"], "b": jnp.zeros((1, 2))}
    ref = separable_newton(lv, pooled, max_iter=500)
    res = lbfgs_newton(lv, pooled, max_iter=2000)
    assert float(res.fun) == pytest.approx(float(ref.fun), rel=1e-12)


def test_it_is_jittable_and_traces_its_work(x64):
    loss_vec, x0, _ = _arrow()
    res = jax.jit(lambda x: lbfgs_newton(loss_vec, x, trace=True))(x0)
    n = int(res.n_iter)
    tr = {k: np.asarray(v) for k, v in res.trace.items()}
    assert set(tr) >= {"f", "t", "gnorm", "n_inner", "n_eval", "reset"}
    assert np.all(np.isfinite(tr["f"][:n])) and np.all(np.isnan(tr["f"][n:]))
    assert np.all(np.diff(tr["f"][:n]) <= 1e-12)                  # monotone: steps are accepted
    assert tr["n_eval"][n - 1] == tr["n_eval_total"]
    assert float(tr["f"][n - 1]) == pytest.approx(float(res.fun))


@pytest.mark.parametrize("ulps", [2.0, 10.0])
def test_a_stop_at_the_dtype_floor_is_convergence(x64, ulps):
    """At the optimum the summed objective's last digits are rounding noise, so every trial of a
    line search can come out higher and it fails twice in a row --- which ended irt_2pl's fits
    'unconverged' at the optimum. The noise is emulated by a pseudo-random offset below ``ulps``
    half-ulps on the value only (the gradient stays exact), with a gtol nothing can reach; the
    solver before the floor rule reported ``converged=False`` here (checked), now ``True``."""
    loss_vec, x0, _ = _arrow()
    ref = separable_newton(loss_vec, x0, max_iter=500)
    amp = ulps * float(np.finfo(np.float64).eps) / 2 * abs(float(ref.fun)) / x0["b"].shape[0]

    def noisy(prm):
        z = 1e9 * (jnp.sum(prm["w"]) + jnp.sum(prm["b"], 1))
        return loss_vec(prm) + jax.lax.stop_gradient(amp * (z - jnp.floor(z)))
    res = lbfgs_newton(noisy, x0, gtol=1e-30, max_iter=2000, warn_max_iter=False)
    assert bool(res.converged) and int(res.n_iter) < 2000
    assert float(jnp.sum(loss_vec(res.x))) == pytest.approx(float(ref.fun), rel=1e-12)


def test_the_polish_finishes_with_newton(x64):
    """A capped L-BFGS run is short of the optimum; the Newton polish from it reaches it, never
    raises the objective, and the trace records the L-BFGS stage and the polish's iterations."""
    loss_vec, x0, _ = _arrow()
    ref = separable_newton(loss_vec, x0, max_iter=500)
    short = lbfgs_newton(loss_vec, x0, max_iter=3, warn_max_iter=False)
    assert not bool(short.converged) and float(short.fun) > float(ref.fun) + 1e-8
    res = lbfgs_newton(loss_vec, x0, max_iter=3, polish=20, warn_max_iter=False, trace=True)
    assert bool(res.converged)
    assert float(res.fun) == pytest.approx(float(ref.fun), rel=1e-12)
    assert float(res.trace["f_lbfgs"]) == float(short.fun)
    assert int(res.n_iter) == 3 + int(res.trace["polish_iter"])


def test_the_polish_steps_from_a_stop_below_gtol(x64):
    """An L-BFGS stop below ``gtol`` can leave objective in directions of small curvature, and the
    polish exists for those: its first Newton iteration is taken whatever the gradient. Control:
    a gradient-gated Newton from the same point (the polish as first built) takes no step."""
    loss_vec, x0, _ = _arrow()
    ref = separable_newton(loss_vec, x0, max_iter=500)
    short = lbfgs_newton(loss_vec, x0, gtol=1e-3)
    assert bool(short.converged) and float(short.grad_norm) < 1e-3
    gated = separable_newton(loss_vec, short.x, gtol=1e-3, max_iter=5)
    assert int(gated.n_iter) == 0 and float(gated.fun) == float(short.fun)
    res = lbfgs_newton(loss_vec, x0, gtol=1e-3, polish=5, trace=True)
    assert int(res.trace["polish_iter"]) >= 1
    assert float(res.fun) < float(short.fun) - 1e-9
    assert float(res.fun) - float(ref.fun) < 1e-3 * (float(short.fun) - float(ref.fun))


def test_min_iter_steps_from_a_converged_start(x64):
    """``separable_newton(min_iter=1)`` takes a step from a point already below ``gtol``, never
    raising the objective; the default (0) takes none, and is unchanged when spelled out."""
    loss_vec, x0, _ = _arrow()
    ref = separable_newton(loss_vec, x0, max_iter=500)
    again = separable_newton(loss_vec, ref.x)
    assert int(again.n_iter) == 0
    forced = separable_newton(loss_vec, ref.x, min_iter=1)
    assert int(forced.n_iter) >= 1 and float(forced.fun) <= float(ref.fun)
    a = separable_newton(loss_vec, x0)
    b = separable_newton(loss_vec, x0, min_iter=0)
    np.testing.assert_array_equal(a.x["w"], b.x["w"])
    np.testing.assert_array_equal(a.x["b"], b.x["b"])


def test_the_options_off_leave_the_solve_unchanged():
    """`precise=False, polish=0` (the defaults) are the solver as it was: spelled out, bit-equal."""
    loss_vec, x0, _ = _arrow()
    a = lbfgs_newton(loss_vec, x0)
    b = lbfgs_newton(loss_vec, x0, precise=False, polish=0)
    np.testing.assert_array_equal(a.x["w"], b.x["w"])
    assert int(a.n_iter) == int(b.n_iter)


def test_the_newton_trace_only_records():
    """``separable_newton(trace=True)`` returns the same solve bit for bit, plus its records."""
    loss_vec, x0, _ = _arrow()
    a = separable_newton(loss_vec, x0)
    b = separable_newton(loss_vec, x0, trace=True)
    np.testing.assert_array_equal(a.x["w"], b.x["w"])
    np.testing.assert_array_equal(a.x["b"], b.x["b"])
    assert a.trace is None and int(b.n_iter) == int(a.n_iter)
    f = np.asarray(b.trace["f"])
    assert float(f[int(b.n_iter) - 1]) == float(b.fun)


# --- routing in the metric regression ------------------------------------------------------- #

def _funnel(N=120, K=60, n=300, seed=1):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(N, n))
    var = np.exp(v @ (rng.normal(size=n) * 0.05))[:, None]
    x = rng.normal(size=(N, K)) * np.sqrt(var)
    return (np.concatenate([x, v], 1), np.concatenate([-x / var, np.zeros((N, n))], 1),
            list(range(K)), {"v": list(range(K, K + n))})


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_a_routed_fit_reaches_the_newton_fit(x64, monkeypatch, arm):
    """Rank-120 span of a 300-d dependency: 120 shared slots, above the threshold, so routed."""
    C, G, bc, dc = _funnel()
    e = Exp("v", shared_weights=(0,)) + Exp()
    ref = regression._fit(e, bc, dc, C, G, max_iter=500)
    monkeypatch.setattr(regression, "SHARED_SOLVER", dict(ARMS[arm], max_iter=3000))
    got = regression._fit(e, bc, dc, C, G)
    assert got.loss == pytest.approx(ref.loss, abs=1e-6)


def test_a_fit_below_the_threshold_is_not_routed(monkeypatch):
    """A shared slope (one shared slot) stays on separable_newton: bit-identical."""
    C, G, bc, dc = _funnel(N=200, K=30, n=30)
    e = SpExp("v", shared_weights=(0,)) + Exp()
    a = regression._fit(e, bc, dc, C, G)
    monkeypatch.setattr(regression, "SHARED_SOLVER", dict(ARMS["alt_pc"]))
    b = regression._fit(e, bc, dc, C, G)
    assert a.loss == b.loss
    np.testing.assert_array_equal(a.params[0]["W"][0], b.params[0]["W"][0])


def test_the_curvature_shape(monkeypatch):
    """With centring on: centred mean squares beside a per-lane bias, raw ones beside a shared bias,
    1 for a shared bias, zero (ignored) for lane leaves."""
    rng = np.random.default_rng(0)
    v = rng.normal(loc=3.0, size=(500, 4))
    e = Exp("v", shared_weights=(0,)) + Exp("v", shared_weights=(0,), shared_bias=(0,))
    params = e.init_params(10, {"v": 4})
    monkeypatch.setattr(regression, "SHARED_CURVATURE_CENTRED", True)
    c = regression.shared_curvature(e, params, {"v": jnp.asarray(v)})
    np.testing.assert_allclose(c[0]["W"][0][0], v.var(0), rtol=1e-5)
    np.testing.assert_allclose(c[1]["W"][0][0], (v ** 2).mean(0), rtol=1e-5)
    np.testing.assert_array_equal(c[0]["b"], np.zeros(10))
    np.testing.assert_array_equal(c[1]["b"], np.ones(1))


def test_the_default_curvature_shape_is_the_raw_second_moment():
    """On the row span the basis comes from the uncentred rows, so the raw mean square is the
    column's exact sigma_i^2 / N; it is the default (Stage A)."""
    rng = np.random.default_rng(0)
    v = rng.normal(loc=3.0, size=(500, 4))
    e = Exp("v", shared_weights=(0,)) + Exp()
    c = regression.shared_curvature(e, e.init_params(10, {"v": 4}), {"v": jnp.asarray(v)})
    np.testing.assert_allclose(c[0]["W"][0][0], (v ** 2).mean(0), rtol=1e-5)
