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


def test_the_curvature_shape():
    """Centred mean squares for a weight beside a per-lane bias, raw ones beside a shared bias,
    1 for a shared bias, zero (ignored) for lane leaves."""
    rng = np.random.default_rng(0)
    v = rng.normal(loc=3.0, size=(500, 4))
    e = Exp("v", shared_weights=(0,)) + Exp("v", shared_weights=(0,), shared_bias=(0,))
    params = e.init_params(10, {"v": 4})
    c = regression.shared_curvature(e, params, {"v": jnp.asarray(v)})
    np.testing.assert_allclose(c[0]["W"][0][0], v.var(0), rtol=1e-5)
    np.testing.assert_allclose(c[1]["W"][0][0], (v ** 2).mean(0), rtol=1e-5)
    np.testing.assert_array_equal(c[0]["b"], np.zeros(10))
    np.testing.assert_array_equal(c[1]["b"], np.ones(1))
