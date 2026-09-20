"""The learned metrics compute in log space, so nothing differentiated forms ``M^{-2}``.

PT on Neal's funnel froze a chain because a hot rung's learned ``D(x)`` reached 2.2e-20: autodiff of
``p^2 / D`` (the metric-derivative kick) and of ``g^2 / M`` (the metric SGD's KL loss) forms
``D^{-2}`` = 2e39, above float32's max, so the kick went infinite at any step size
(``tests/experiments/writeups/collapse_traces.md``). Now ``log M`` comes from
:meth:`~mimcs.hmc.metric_expr.MetricExpr.log_evaluate`, and the energy and loss are written with the
whitened quantities ``p exp(-log M / 2)`` and ``g exp(-log M / 2)``.

What must hold: ``log_evaluate`` is ``log(evaluate)`` for every node; the energy and loss have the
same *values* as before at a benign point; and at the failing scale their gradients are finite ---
where the old formulas' are not (the control that keeps this from passing vacuously).
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from mimcs.hmc import LearnedDiagonalBlock, ShapedLearnedBlock
from mimcs.hmc.metric_expr import Exp, Sigmoid, SpExp, SpSigmoid


def _random_params(expr, block_dim, dep_dims, seed):
    """The expression's init pytree with every leaf replaced by small random values."""
    params = expr.init_params(block_dim, dep_dims)
    leaves, tree = jax.tree_util.tree_flatten(params)
    rng = np.random.default_rng(seed)
    return jax.tree_util.tree_unflatten(
        tree, [jnp.asarray(rng.normal(0.0, 0.7, np.shape(l))) for l in leaves])


EXPRS = [
    Exp("v"),
    Sigmoid("v"),
    Exp("v", features="quadratic") + Exp(),
    Exp() * Sigmoid("v", "u") + Exp(),
    Exp("v", shared_weights=(0,)) + Exp("u", shared_weights=(0,), shared_bias=(0,)),
    SpExp("s"),
    SpSigmoid("s") * Exp() + Exp("v"),
]


@pytest.mark.parametrize("expr", EXPRS, ids=repr)
def test_log_evaluate_is_log_of_evaluate(expr):
    d = 4
    dep_dims = {"v": 2, "u": 3, "s": d}
    rng = np.random.default_rng(1)
    deps = {k: jnp.asarray(rng.normal(size=n)) for k, n in dep_dims.items()}
    for seed in range(3):
        params = _random_params(expr, d, dep_dims, seed)
        want = np.log(np.asarray(expr.evaluate(params, deps)))
        got = np.asarray(expr.log_evaluate(params, deps))
        np.testing.assert_allclose(np.broadcast_to(got, want.shape), want, rtol=1e-5, atol=1e-6)


def test_log_evaluate_stays_finite_where_evaluate_underflows():
    """An ``Exp`` far below float range still has a finite log --- ``log(evaluate)`` is ``-inf``."""
    expr = Exp()
    params = {"W": [], "b": jnp.full((3,), -200.0, dtype=jnp.float32)}
    assert np.all(np.isfinite(np.asarray(expr.log_evaluate(params, {}))))
    assert np.all(np.asarray(expr.evaluate(params, {})) == 0.0)          # control: underflows


def _funnel_block():
    """x (3 coords, at 0:3) with a learned metric exp(W v + b) on v (at 3:4)."""
    return LearnedDiagonalBlock("x", (0, 3), Exp("v"), {"v": [(3, 4)]})


def _old_energy(block, q, p, params):
    M = block._mass(q, None, params)
    return 0.5 * jnp.sum(p ** 2 / M) + 0.5 * jnp.sum(jnp.log(M))


def _old_loss(block, params, q, score):
    M = block._mass(q, None, params)
    g = score[block.s:block.e]
    return 0.5 * jnp.sum(jnp.log(M) + g ** 2 / M)


def test_energy_and_loss_values_are_unchanged_at_a_benign_point():
    blk = _funnel_block()
    params = {"W": [jnp.asarray([[-0.9], [-1.1], [-1.0]])], "b": jnp.asarray([0.1, -0.2, 0.0])}
    q = jnp.asarray([0.3, -1.2, 2.0, 0.7])
    p = jnp.asarray([0.4, -0.3, 1.1])
    score = jnp.asarray([0.5, -1.5, 0.2, 0.0])
    np.testing.assert_allclose(float(blk._energy(q, None, p, params)),
                               float(_old_energy(blk, q, p, params)), rtol=1e-6)
    np.testing.assert_allclose(float(blk.metric_loss(params, q, None, score)),
                               float(_old_loss(blk, params, q, score)), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(blk._velocity(q, None, p, params)),
                               np.asarray(p / blk._mass(q, None, params)), rtol=1e-6)


def test_gradients_are_finite_where_the_metric_is_below_float32_range_squared():
    """The PT funnel's failing state, in float32: v = 25.1 with W = -1.77, b = -0.73, so
    log M = -45.2 (M = 2.2e-20) and M^-2 = 2e39 overflows float32."""
    blk = _funnel_block()
    params = {"W": [jnp.full((3, 1), -1.77, dtype=jnp.float32)],
              "b": jnp.full((3,), -0.73, dtype=jnp.float32)}
    q = jnp.asarray([2.9e5, 2.5e3, 3.2e5, 25.1], dtype=jnp.float32)
    M = np.asarray(blk._mass(q, None, params))
    assert np.all(M < 5e-20) and np.all(M > 0)                     # the failing scale, representable
    p = jnp.sqrt(jnp.asarray(M)) * jnp.asarray([0.8, -1.3, 0.4], dtype=jnp.float32)   # p ~ N(0, M)
    score = jnp.asarray([1.8e-6, 1.6e-8, 2.0e-6, 0.0], dtype=jnp.float32)

    kick = jax.grad(lambda qq: blk._energy(qq, None, p, params))(q)        # the metric-derivative kick
    dloss = jax.grad(lambda pp: blk.metric_loss(pp, q, None, score))(params)
    assert np.all(np.isfinite(np.asarray(kick)))
    assert all(np.all(np.isfinite(np.asarray(l))) for l in jax.tree_util.tree_leaves(dloss))

    # Control: the old formulas at the same point are not finite.
    old_kick = jax.grad(lambda qq: _old_energy(blk, qq, p, params))(q)
    old_dloss = jax.grad(lambda pp: _old_loss(blk, pp, q, score))(params)
    assert not np.all(np.isfinite(np.asarray(old_kick)))
    assert not all(np.all(np.isfinite(np.asarray(l))) for l in jax.tree_util.tree_leaves(old_dloss))


def test_shaped_metric_loss_is_the_same_log_space_loss():
    shp = ShapedLearnedBlock("x", (0, 3), Exp("v"), {"v": [(3, 4)]}, ("lowrank", 1))
    plain = _funnel_block()
    dp = {"W": [jnp.asarray([[-0.9], [-1.1], [-1.0]])], "b": jnp.asarray([0.1, -0.2, 0.0])}
    q = jnp.asarray([0.3, -1.2, 2.0, 0.7])
    score = jnp.asarray([0.5, -1.5, 0.2, 0.0])
    got = float(shp.metric_loss({"diag": dp, "shape": shp.init_params()["shape"]}, q, None, score))
    assert np.isclose(got, float(plain.metric_loss(dp, q, None, score)), rtol=1e-6)


@pytest.mark.parametrize("shape", ["dense", ("lowrank", 2)])
def test_shaped_kinetic_gradients_are_finite_at_the_failing_scale(shape):
    """The shaped block's kinetic is whitened too: ``D`` enters only as ``exp(+-log D / 2)``."""
    blk = ShapedLearnedBlock("x", (0, 3), Exp("v"), {"v": [(3, 4)]}, shape)
    params = blk.init_params()
    params["diag"] = {"W": [jnp.full((3, 1), -1.77, dtype=jnp.float32)],
                      "b": jnp.full((3,), -0.73, dtype=jnp.float32)}
    if shape == "dense":
        params["shape"] = jnp.asarray([[1.2, 0, 0], [0.3, 0.9, 0], [-0.2, 0.1, 1.1]], jnp.float32)
    else:
        W = jnp.asarray(np.linalg.qr(np.random.default_rng(0).standard_normal((3, 2)))[0], jnp.float32)
        params["shape"] = (W, jnp.asarray([3.0, 0.5], jnp.float32))
    q = jnp.asarray([2.9e5, 2.5e3, 3.2e5, 25.1], dtype=jnp.float32)
    D = np.asarray(blk._D(q, None, params["diag"]))
    assert np.all(D < 5e-20) and np.all(D > 0)
    p = jnp.sqrt(jnp.asarray(D)) * jnp.asarray([0.8, -1.3, 0.4], dtype=jnp.float32)

    kick = jax.grad(lambda qq: blk._energy(qq, None, p, params))(q)
    assert np.all(np.isfinite(np.asarray(kick)))
    assert np.all(np.isfinite(np.asarray(blk._velocity(q, None, p, params))))
    assert np.all(np.isfinite(np.asarray(blk._sample_factor(q, None, p / jnp.sqrt(jnp.asarray(D)), params))))
