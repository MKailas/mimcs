"""The metric regression's compiled-fit cache: one compilation per candidate *structure*.

Before this, every candidate fit re-traced its whole Newton program (closures over the evidence,
bound to an eager ``lax.while_loop``), which on `irt_2pl`'s 306-candidate pool cost ~0.5 s and
~13 MB per fit and OOM-killed a 6.4 GB box. These tests pin the three things the change must
deliver: dependency names do not split the cache, a hit does not re-trace, and the fitted answer
is the one the old eager path produced.
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp

from mimcs.factory import regression
from mimcs.factory.regression import (
    fit_metric_expr, fit_is_usable, ridge_penalty_vec, structure_key)
from mimcs.hmc.metric_expr import Exp, Sigmoid, SpExp
from mimcs.optim import minimize, separable_newton

X64 = jax.config.jax_enable_x64
#: the compiled program lets XLA fuse what the eager path ran as separate dispatches, so parameters
#: may move in the last bits (measured 4e-14 under x64); float32 compounds that over iterations.
RTOL = 1e-9 if X64 else 2e-4


def _evidence(seed=0, n=400, k=5):
    """Scores whose conditional variance really depends on `v` (col 0) --- a funnel-like target."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=n)
    u = rng.normal(size=n)
    x = rng.normal(size=(n, k)) * np.exp(0.5 * v)[:, None]
    coords = np.column_stack([v, u, x])
    grads = np.column_stack([rng.normal(size=n), rng.normal(size=n),
                             -x / np.exp(v)[:, None]])
    return coords, grads, list(range(2, 2 + k))


def _reference_fit(expr, block_cols, dep_cols, coords, grads, optimizer="newton",
                   ridge_sigma=regression.RIDGE_SIGMA, **opt):
    """The pre-cache eager fit, verbatim in its numerics (whole-array path)."""
    block_cols = jnp.asarray(np.asarray(block_cols, dtype=int))
    block_dim = int(block_cols.shape[0])
    dep_dims = {d: len(c) for d, c in dep_cols.items()}
    g = jnp.asarray(grads, float)[:, block_cols]
    dep_data = {d: jnp.asarray(coords, float)[:, jnp.asarray(c)] for d, c in dep_cols.items()}
    n_rows = int(g.shape[0])

    def row(params, g_row, dep_row):
        M = expr.evaluate(params, dep_row)
        return 0.5 * jnp.sum(jnp.log(M) + g_row ** 2 / M)

    def row_vec(params, g_row, dep_row):
        M = expr.evaluate(params, dep_row)
        return 0.5 * (jnp.log(M) + g_row ** 2 / M)

    def mean_loss(params):
        return jnp.mean(jax.vmap(lambda a, b: row(params, a, b))(g, dep_data))

    def loss_vec(params):
        return jnp.mean(jax.vmap(lambda a, b: row_vec(params, a, b))(g, dep_data), axis=0)

    scale = jnp.maximum(jnp.mean(g ** 2, axis=0), regression.INIT_SCALE_FLOOR)
    anchor = expr.init_params(block_dim, dep_dims, target=scale)

    def pen(p):
        return ridge_penalty_vec(p, anchor, block_dim, ridge_sigma, n_rows)

    if optimizer == "newton":
        res = separable_newton(lambda p: loss_vec(p) + pen(p), anchor, **opt)
    else:
        res = minimize(lambda p: mean_loss(p) + jnp.sum(pen(p)), anchor, **opt)
    return float(mean_loss(res.x)), res.x


def _assert_same_fit(a, b):
    assert a[0] == pytest.approx(b[0], rel=RTOL, abs=RTOL)
    for x, y in zip(jax.tree_util.tree_leaves(a[1]), jax.tree_util.tree_leaves(b[1])):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=RTOL, atol=RTOL)


# --- the canonical form ------------------------------------------------------------------------ #

def test_dependency_names_do_not_change_the_structure_key():
    assert structure_key(Exp("a") + Exp()) == structure_key(Exp("b") + Exp())
    assert structure_key(Exp("a") + Exp("b")) == structure_key(Exp("x") + Exp("y"))


@pytest.mark.parametrize("other", [
    Exp("a", shared_weights=(0,)) + Exp(),          # sharing
    SpExp("a") + Exp(),                             # sparse vs dense
    Exp("a", features="quadratic") + Exp(),         # feature map
    Exp() * Sigmoid("a") + Exp(),                   # link / product
    Exp("a") + Exp("a"),                            # a repeated dependency is one slot, not two
    Exp(ordinal="a") + Exp(),                       # coding
])
def test_what_does_change_the_structure_key(other):
    assert structure_key(other) != structure_key(Exp("b") + Exp())


def test_categorical_and_ordinal_codings_are_distinct_structures():
    assert structure_key(Exp(categorical="z")) != structure_key(Exp(ordinal="z"))


def test_relabel_keeps_the_parameter_layout():
    expr = Exp("b", "a") * Sigmoid("a") + Exp()
    assert expr.dep_order() == ["b", "a"]
    canon, mapping = expr.canonical()
    assert mapping == {"b": "_d0", "a": "_d1"}
    params = expr.init_params(3, {"a": 2, "b": 4})
    params = jax.tree_util.tree_map(lambda l: l + 0.1 * jnp.arange(l.size).reshape(l.shape),
                                    params)
    rng = np.random.default_rng(1)
    deps = {"a": jnp.asarray(rng.normal(size=2)), "b": jnp.asarray(rng.normal(size=4))}
    same = canon.evaluate(params, {mapping[d]: v for d, v in deps.items()})
    assert np.array_equal(np.asarray(expr.evaluate(params, deps)), np.asarray(same))


# --- one trace per structure ------------------------------------------------------------------ #

def test_same_structure_on_different_dependencies_traces_once():
    coords, grads, bcols = _evidence()
    regression.clear_fit_cache()
    t0 = regression.fit_cache_info()["traces"]
    fit_metric_expr(Exp("v") + Exp(), bcols, {"v": [0]}, coords, grads)
    t1 = regression.fit_cache_info()["traces"]
    fit_metric_expr(Exp("u") + Exp(), bcols, {"u": [1]}, coords, grads)
    t2 = regression.fit_cache_info()["traces"]
    assert t1 - t0 == 1
    assert t2 == t1                                  # a hit: no new trace
    # control: a different structure must trace, or the counter proves nothing
    fit_metric_expr(Exp() * Sigmoid("v") + Exp(), bcols, {"v": [0]}, coords, grads)
    assert regression.fit_cache_info()["traces"] == t2 + 1


def test_ridge_strength_is_data_not_structure():
    coords, grads, bcols = _evidence()
    regression.clear_fit_cache()
    t0 = regression.fit_cache_info()["traces"]
    loose = fit_metric_expr(Exp("v"), bcols, {"v": [0]}, coords, grads, ridge_sigma=5.0)
    tight = fit_metric_expr(Exp("v"), bcols, {"v": [0]}, coords, grads, ridge_sigma=0.1)
    assert regression.fit_cache_info()["traces"] == t0 + 1
    # ... and the strength still acts: the tight ridge pulls the slope toward its zero anchor
    assert (np.abs(np.asarray(tight[1]["W"][0])).max()
            < np.abs(np.asarray(loose[1]["W"][0])).max())


def test_the_cache_is_bounded(monkeypatch):
    coords, grads, bcols = _evidence()
    regression.clear_fit_cache()
    monkeypatch.setattr(regression, "FIT_CACHE_SIZE", 2)
    for expr in (Exp("v"), Exp("v") + Exp(), Exp() * Sigmoid("v")):
        fit_metric_expr(expr, bcols, {"v": [0]}, coords, grads, max_iter=3)
    assert regression.fit_cache_info()["programs"] == 2


# --- the same answer as the eager path -------------------------------------------------------- #

@pytest.mark.parametrize("expr", [
    Exp(),
    Exp("v") + Exp(),
    SpExp("x", shared_weights=(0,)) + Exp(),
    Exp() * Sigmoid("v") + Exp(),
    Exp("v", shared_weights=(0,), shared_bias=(0,)),
])
def test_cached_fit_matches_the_eager_reference(expr):
    coords, grads, bcols = _evidence(k=5)
    dep_cols = {"v": [0], "x": [2, 3, 4, 5, 6]}
    used = {d: dep_cols[d] for d in expr.deps()}
    _assert_same_fit(fit_metric_expr(expr, bcols, used, coords, grads),
                     _reference_fit(expr, bcols, used, coords, grads))


def test_lbfgs_control_arm_matches_the_eager_reference():
    coords, grads, bcols = _evidence()
    args = (Exp("v") + Exp(), bcols, {"v": [0]}, coords, grads)
    _assert_same_fit(fit_metric_expr(*args, optimizer="lbfgs", max_iter=200),
                     _reference_fit(*args, optimizer="lbfgs", max_iter=200))


def test_equivalence_is_not_vacuous():
    """Control: the comparison above must be able to fail --- different evidence, different fit."""
    coords, grads, bcols = _evidence()
    ref = _reference_fit(Exp("v") + Exp(), bcols, {"v": [0]}, coords, grads)
    moved = fit_metric_expr(Exp("v") + Exp(), bcols, {"v": [0]}, coords, 1.1 * grads)
    with pytest.raises(AssertionError):
        _assert_same_fit(moved, ref)


def test_unused_dependency_data_does_not_split_the_cache():
    coords, grads, bcols = _evidence()
    regression.clear_fit_cache()
    a = fit_metric_expr(Exp("v"), bcols, {"v": [0]}, coords, grads)
    t = regression.fit_cache_info()["traces"]
    b = fit_metric_expr(Exp("v"), bcols, {"v": [0], "u": [1]}, coords, grads)
    assert regression.fit_cache_info()["traces"] == t
    _assert_same_fit(a, b)


def test_chunked_path_runs_through_the_program(monkeypatch):
    coords, grads, bcols = _evidence(n=300)
    args = (Exp("v") + Exp(), bcols, {"v": [0]}, coords, grads)
    whole = fit_metric_expr(*args, max_iter=30)
    monkeypatch.setattr(regression, "CHUNK_LOSS_BYTES", 1)
    from mimcs import config
    monkeypatch.setattr(config, "chunk_bytes", lambda: 64 * 8 * 5)    # ~64-row chunks
    chunked = fit_metric_expr(*args, max_iter=30)
    assert chunked[0] == pytest.approx(whole[0], rel=1e-6 if not X64 else 1e-10)


# --- usability verdict, from the same program ---------------------------------------------------- #

def test_usable_verdict_matches_fit_is_usable_on_a_good_fit():
    coords, grads, bcols = _evidence()
    fit = regression._fit(Exp("v") + Exp(), bcols, {"v": [0]}, coords, grads)
    assert fit.usable is True
    assert fit.usable == fit_is_usable(Exp("v") + Exp(), fit.params, {"v": [0]}, coords, fit.loss)


def test_usable_verdict_rejects_a_metric_that_overflows():
    """A warm start at an overflowing bias and no iterations: finite parameters, infinite ``M``."""
    coords, grads, bcols = _evidence()
    expr = Exp("v")
    init = expr.init_params(len(bcols), {"v": 1})
    init = {"W": init["W"], "b": init["b"] + (1e4 if X64 else 1e3)}
    fit = regression._fit(expr, bcols, {"v": [0]}, coords, grads, init=init, max_iter=0)
    assert fit.usable is False
    assert fit.usable == fit_is_usable(expr, fit.params, {"v": [0]}, coords, 1.0)
