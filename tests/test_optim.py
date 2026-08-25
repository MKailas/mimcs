"""Tests for the L-BFGS minimiser (``mimcs.optim.minimize``).

Standard smooth objectives (ill-conditioned quadratic, Rosenbrock) and the objective the
sampler factory actually uses it for --- fitting a mass-matrix mini-language expression to a
batch by the KL loss --- plus a pytree-structured argument and jittability.
"""

import numpy as np
import jax
import jax.numpy as jnp

from mimcs.optim import minimize
from mimcs.hmc.metric_expr import Exp


def test_minimizes_ill_conditioned_quadratic():
    """min 1/2 (x-x*)^T A (x-x*): converges to x* for a stiff A."""
    A = jnp.diag(jnp.array([1.0, 10.0, 100.0, 1000.0]))
    xstar = jnp.array([1.0, -2.0, 3.0, 0.5])
    fun = lambda x: 0.5 * (x - xstar) @ A @ (x - xstar)
    res = minimize(fun, jnp.zeros(4))
    assert res.converged
    assert np.allclose(np.asarray(res.x), np.asarray(xstar), atol=1e-4)
    assert float(res.fun) < 1e-8


def test_minimizes_rosenbrock():
    """The banana valley: min sum 100 (x_{i+1}-x_i^2)^2 + (1-x_i)^2, optimum all-ones."""
    def rosen(x):
        return jnp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)
    res = minimize(rosen, jnp.array([-1.2, 1.0, -1.0, 1.2]), max_iter=500, gtol=1e-8)
    assert np.allclose(np.asarray(res.x), 1.0, atol=1e-3), res.x
    assert float(res.fun) < 1e-6


def test_accepts_pytree_argument():
    """x0 may be any pytree; the result comes back in the same structure."""
    target = {"a": jnp.array([1.0, 2.0]), "b": jnp.array(-3.0)}
    fun = lambda p: jnp.sum((p["a"] - target["a"]) ** 2) + (p["b"] - target["b"]) ** 2
    res = minimize(fun, {"a": jnp.zeros(2), "b": jnp.array(0.0)})
    assert set(res.x) == {"a", "b"}
    assert np.allclose(np.asarray(res.x["a"]), [1.0, 2.0], atol=1e-4)
    assert np.isclose(float(res.x["b"]), -3.0, atol=1e-4)


def test_is_jittable():
    fun = lambda x: jnp.sum((x - 1.0) ** 2)
    res = jax.jit(lambda x0: minimize(fun, x0))(jnp.zeros(5))
    assert bool(res.converged)
    assert np.allclose(np.asarray(res.x), 1.0, atol=1e-4)


def test_fits_metric_expression_kl_loss():
    """The factory use case: fit ``M(v) = exp(W v + b)`` to data whose conditional score
    second moment is ``e^{-v}`` (a funnel-like block) by minimising the batch KL loss
    ``mean_n 1/2 (log M + g^2 / M)``. The optimum is ``W -> -1``, ``b -> 0``."""
    rng = np.random.default_rng(0)
    N = 4000
    v = rng.normal(0.0, 1.5, size=N)                       # dependency coordinate
    # g | v ~ N(0, e^{-v}) so E[g^2 | v] = e^{-v}: the ideal metric M(v) = e^{-v}
    g = rng.normal(0.0, 1.0, size=N) * np.exp(-0.5 * v)
    v_arr, g_arr = jnp.asarray(v), jnp.asarray(g)

    expr = Exp("v")

    def batch_kl(params):
        def one(vv, gg):
            M = expr.evaluate(params, {"v": vv.reshape(1)})[0]   # block_dim 1 -> scalar
            return 0.5 * (jnp.log(M) + gg ** 2 / M)
        return jnp.mean(jax.vmap(one)(v_arr, g_arr))

    p0 = expr.init_params(1, {"v": 1})
    res = minimize(batch_kl, p0, max_iter=200)
    W = float(np.asarray(res.x["W"][0]).ravel()[0])
    b = float(np.asarray(res.x["b"]).ravel()[0])
    assert abs(W - (-1.0)) < 0.1, f"W={W}"
    assert abs(b) < 0.1, f"b={b}"
