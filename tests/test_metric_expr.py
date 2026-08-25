"""Unit tests for the mass-matrix mini-language (``mimcs.hmc.metric_expr``).

Cover the expression interface directly (no sampler): dependency collection, identity-ish
initialisation, evaluation against hand-computed values, structural positivity, parameter
counts, and that the parameter pytree is differentiable (as ``MetricAdaptation`` requires).
"""

import numpy as np
import jax
import jax.numpy as jnp

from mimcs.hmc.metric_expr import MetricExpr, Exp, Sigmoid, SpExp, SpSigmoid, Sum, Product


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def test_operators_build_sum_and_product():
    assert isinstance(Exp("v") + Exp(), Sum)
    assert isinstance(Exp() * Sigmoid("v"), Product)
    e = Exp() * Sigmoid("v", "x") + Exp()
    assert isinstance(e, Sum) and isinstance(e.a, Product)
    # non-expression operands are rejected (so typos fail loudly, not silently coerce)
    for bad in (2.0, "v", None):
        try:
            _ = Exp("v") + bad
            assert False, "expected TypeError"
        except TypeError:
            pass


def test_deps_collected_across_tree():
    assert Exp().deps() == set()
    assert Exp("v").deps() == {"v"}
    assert (Exp() * Sigmoid("v", "x") + Exp("y")).deps() == {"v", "x", "y"}


def test_n_add_counts_additive_terms():
    assert Exp("v")._n_add() == 1
    assert (Exp("v") * Sigmoid("x"))._n_add() == 1          # a product is one additive term
    assert (Exp("v") + Exp("x") + Exp())._n_add() == 3


def test_init_is_near_identity():
    """Every seed expression initialises to ~ I at zero weights."""
    dep_dims = {"v": 2, "x": 3}
    zero = {"v": jnp.zeros(2), "x": jnp.zeros(3)}
    for expr in [Exp(), Exp("v"), Exp("v") + Exp(),
                 Exp("v") + Exp("x") + Exp(),
                 Exp() * Sigmoid("v", "x") + Exp(),
                 Exp("v", features="quadratic") + Exp()]:
        bd = 4
        M = np.asarray(expr.evaluate(expr.init_params(bd, dep_dims), zero))
        assert M.shape == (bd,)
        assert np.allclose(M, 1.0, atol=0.02), f"{expr!r} init M={M}"


def test_evaluate_matches_hand_computation():
    # M[d] = exp(W_d . v + b_d) with a chosen non-trivial W, b
    W = jnp.array([[2.0, -1.0], [0.5, 0.0]])   # (block_dim=2, feat_dim=2)
    b = jnp.array([0.3, -0.7])
    v = jnp.array([1.5, -2.0])
    got = np.asarray(Exp("v").evaluate({"W": [W], "b": b}, {"v": v}))
    want = np.exp(np.asarray(W) @ np.asarray(v) + np.asarray(b))
    assert np.allclose(got, want)

    # a product of a bias-Exp scale and a Sigmoid gate
    e = Exp() * Sigmoid("v")
    params = [{"W": [], "b": jnp.array([0.4, -0.2])},
              {"W": [jnp.array([[1.0], [-3.0]])], "b": jnp.array([0.0, 0.5])}]
    got = np.asarray(e.evaluate(params, {"v": jnp.array([0.8])}))
    want = np.exp([0.4, -0.2]) * _sig(np.array([1.0 * 0.8 + 0.0, -3.0 * 0.8 + 0.5]))
    assert np.allclose(got, want)


def test_quadratic_features_expand_weight_dim():
    e = Exp("v", features="quadratic")
    p = e.init_params(3, {"v": 2})
    assert p["W"][0].shape == (3, 4)             # feat = [v, v^2] -> 2*2
    # evaluate uses [v, v^2]
    W = jnp.arange(12.0).reshape(3, 4)
    v = jnp.array([2.0, -1.0])
    got = np.asarray(e.evaluate({"W": [W], "b": jnp.zeros(3)}, {"v": v}))
    feat = np.array([2.0, -1.0, 4.0, 1.0])
    assert np.allclose(got, np.exp(np.asarray(W) @ feat))


def test_positivity_for_random_params():
    """Structural positivity: any parameters give a strictly positive mass."""
    e = Exp() * Sigmoid("v", "x") + Exp("v") + Exp()
    dep_dims = {"v": 2, "x": 2}
    key = jax.random.PRNGKey(0)
    p = e.init_params(3, dep_dims)
    for i in range(20):
        key, k1, k2 = jax.random.split(key, 3)
        # perturb every leaf with large random noise
        leaves, tree = jax.tree_util.tree_flatten(p)
        noisy = [leaf + 5.0 * jax.random.normal(jax.random.fold_in(k1, j), leaf.shape)
                 for j, leaf in enumerate(leaves)]
        pp = jax.tree_util.tree_unflatten(tree, noisy)
        dc = {"v": jax.random.normal(k1, (2,)) * 3, "x": jax.random.normal(k2, (2,)) * 3}
        M = np.asarray(e.evaluate(pp, dc))
        # structural guarantee: strictly positive and never NaN (float32 exp may overflow
        # to +inf under these deliberately extreme params -- still positive, not a violation)
        assert np.all(M > 0.0) and not np.any(np.isnan(M))


def test_n_params_counts():
    dep_dims = {"v": 2, "x": 3}
    bd = 4
    assert Exp().n_params(bd, dep_dims) == bd                       # bias only
    assert Exp("v").n_params(bd, dep_dims) == bd * 2 + bd           # W(4x2) + b(4)
    # Exp()*Sigmoid("v","x") + Exp(): bias(4) + [W_v(4x2)+W_x(4x3)+b(4)] + bias(4)
    e = Exp() * Sigmoid("v", "x") + Exp()
    assert e.n_params(bd, dep_dims) == bd + (bd * 2 + bd * 3 + bd) + bd


def test_param_pytree_is_differentiable():
    """MetricAdaptation differentiates the whole pytree; check grad flows to every leaf."""
    e = Exp() * Sigmoid("v", "x") + Exp("v") + Exp()
    dep_dims = {"v": 2, "x": 2}
    p = e.init_params(3, dep_dims)
    dc = {"v": jnp.array([0.5, -1.0]), "x": jnp.array([1.0, 2.0])}
    g = jax.grad(lambda pp: jnp.sum(e.evaluate(pp, dc) ** 2))(p)
    leaves = [np.asarray(l) for l in jax.tree_util.tree_leaves(g)]
    assert leaves and all(np.all(np.isfinite(l)) for l in leaves)
    # at least one non-trivial gradient (weights that the input actually reaches)
    assert any(np.any(np.abs(l) > 1e-6) for l in leaves)


# --- sparse (elementwise) atoms --------------------------------------------- #


def test_sparse_mro_is_sparse_numerics_plus_dense_link():
    """SpExp = sparse numerics (from _SparseAtom) + Exp's link, via a valid diamond MRO."""
    assert [c.__name__ for c in SpExp.__mro__][:4] == ["SpExp", "_SparseAtom", "Exp", "_Atom"]
    assert repr(SpExp("lambda")) == "SpExp('lambda')"
    assert SpExp("lambda").deps() == {"lambda"} and SpExp("lambda")._n_add() == 1


def test_sparse_n_params_is_cheap():
    # identity: block_dim (weights, one per coordinate) + block_dim (bias) = 2*block_dim
    assert SpExp("s").n_params(30, {"s": 30}) == 60
    # quadratic: 2*block_dim (weights) + block_dim (bias) = 3*block_dim
    assert SpExp("s", features="quadratic").n_params(30, {"s": 30}) == 90


def test_sparse_evaluate_is_elementwise():
    """M_j = exp(W_j s_j + b_j): no coupling across coordinates (contrast the dense matmul)."""
    W = jnp.array([[2.0], [-1.0], [0.5]])       # (block_dim=3, per_feat=1)
    b = jnp.array([0.1, -0.2, 0.3])
    s = jnp.array([1.0, 2.0, -1.0])
    got = np.asarray(SpExp("s").evaluate({"W": [W], "b": b}, {"s": s}))
    want = np.exp(np.array([2.0, -1.0, 0.5]) * np.asarray(s) + np.asarray(b))
    assert np.allclose(got, want)
    # quadratic per-coordinate feature [s_j, s_j^2]
    Wq = jnp.array([[1.0, -0.5], [0.0, 0.3], [2.0, 0.0]])
    gq = np.asarray(SpExp("s", features="quadratic").evaluate({"W": [Wq], "b": b}, {"s": s}))
    wq = np.exp(np.sum(np.asarray(Wq) * np.stack([s, s ** 2], -1), -1) + np.asarray(b))
    assert np.allclose(gq, wq)


def test_sparse_init_near_identity_and_gate_range():
    zero = {"s": jnp.zeros(5)}
    for expr in [SpExp("s") + Exp(), Exp() * SpSigmoid("s") + Exp()]:
        M = np.asarray(expr.evaluate(expr.init_params(5, {"s": 5}), zero))
        assert M.shape == (5,) and np.allclose(M, 1.0, atol=0.02), f"{expr!r}: {M}"
    # SpSigmoid stays in (0, 1) (float32 saturates to exactly 1.0 for very large inputs, so use
    # a moderate range and assert the closed bound)
    p = SpSigmoid("s").init_params(5, {"s": 5})
    val = np.asarray(SpSigmoid("s").evaluate({"W": [jnp.full((5, 1), 2.0)], "b": p["b"]},
                                             {"s": jnp.array([3.0, -3.0, 0.0, 1.5, -1.5])}))
    assert np.all((val > 0.0) & (val < 1.0))


def test_sparse_dimension_mismatch_raises():
    """A sparse dependency must match the block dimension (bijective row correspondence)."""
    try:
        SpExp("s").init_params(4, {"s": 3})
        assert False, "expected ValueError"
    except ValueError as ex:
        assert "match the block dimension" in str(ex)


def test_sparse_composition_and_differentiability():
    e = Exp() * SpSigmoid("s") + SpExp("s")
    p = e.init_params(3, {"s": 3})
    dc = {"s": jnp.array([0.5, -1.0, 2.0])}
    g = jax.grad(lambda pp: jnp.sum(e.evaluate(pp, dc) ** 2))(p)
    leaves = [np.asarray(l) for l in jax.tree_util.tree_leaves(g)]
    assert leaves and all(np.all(np.isfinite(l)) for l in leaves)
    assert any(np.any(np.abs(l) > 1e-6) for l in leaves)
