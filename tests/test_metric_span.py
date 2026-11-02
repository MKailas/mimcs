"""Tests for fitting dense metric weights on the span of the evidence rows (``SPAN_REDUCE``).

A dense identity-feature atom reads a dependency only through ``W . v_n``, so where the evidence
rows span less than the dependency's width the part of ``W`` orthogonal to them changes no row's
metric and the ridge (anchored at ``W = 0``) pins it at zero. Fitting on the span is then the
*same* penalised optimum with fewer weights. What is pinned here: that it is the same optimum (the
reason this may be on by default), that it is inert where it should be (full rank, sparse atoms,
quadratic features), and that the parameters come back in the raw, full-width layout.
"""

import numpy as np
import pytest
import jax.numpy as jnp

from mimcs import config
from mimcs.hmc.metric_expr import Exp, Sigmoid, SpExp
from mimcs.factory import regression
from mimcs.factory.regression import project_params, row_span, span_reducible


@pytest.fixture(autouse=True)
def _no_kick_guard(monkeypatch):
    """The kick guard off: the synthetic dependency columns below carry no score of their own, so
    every kick ratio would divide by the score floor (see ``test_metric_kick_guard.py``)."""
    monkeypatch.setattr(regression, "KICK_GUARD", None)


@pytest.fixture
def x64():
    was = config.x64_enabled()
    config.enable_x64(True)
    try:
        yield
    finally:
        config.enable_x64(was)


def _funnel_evidence(N, K, n, seed=1, duplicate=True):
    """``x_k | v ~ N(0, exp(v . w))`` for a K-lane block over an n-d dependency; scores of ``x``."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(N, n))
    if duplicate:
        v[10] = v[3]
    w = rng.normal(size=n) * 0.05
    var = np.exp(v @ w)[:, None]
    x = rng.normal(size=(N, K)) * np.sqrt(var)
    coords = np.concatenate([x, v], 1)
    grads = np.concatenate([-x / var, np.zeros((N, n))], 1)
    return coords, grads, list(range(K)), {"v": list(range(K, K + n))}


def test_row_span_is_the_rank_and_spans_the_rows():
    """Fewer rows than columns, plus a repeated row: rank N - 1, and every row is reproduced by its
    projection (an orthonormal basis of exactly their span)."""
    rng = np.random.default_rng(0)
    v = rng.normal(size=(40, 90))
    v[7] = v[2]
    U = row_span(v)
    assert U.shape == (90, 39)
    np.testing.assert_allclose(U.T @ U, np.eye(39), atol=1e-12)
    np.testing.assert_allclose((v @ U) @ U.T, v, atol=1e-12)


def test_row_span_is_none_at_full_rank():
    rng = np.random.default_rng(0)
    assert row_span(rng.normal(size=(300, 20))) is None
    assert row_span(rng.normal(size=(300, 1))) is None


def test_only_dense_identity_dependencies_are_reducible():
    """A sparse atom reads its dependency elementwise and a quadratic feature map is not linear,
    so either one reading a dependency keeps it full width --- even alongside a dense reader."""
    sw = dict(shared_weights=(0,))
    assert span_reducible(Exp("v", **sw) + Exp()) == {"v"}
    assert span_reducible(Exp() * Sigmoid("v", "u") + Exp()) == {"v", "u"}
    assert span_reducible(SpExp("v", **sw) + Exp()) == set()
    assert span_reducible(Exp("v", features="quadratic") + Exp()) == set()
    assert span_reducible(Exp("v") + SpExp("v")) == set()


def test_the_projection_reproduces_the_metric_at_every_evidence_row():
    """``W = alpha U^T`` on the raw data gives the metric ``alpha`` gives on the projected data,
    and a component of ``W`` orthogonal to the rows changes nothing."""
    rng = np.random.default_rng(3)
    N, K, n = 30, 5, 70
    v = rng.normal(size=(N, n))
    U = row_span(v)
    e = Exp("v", shared_weights=(0,)) + Exp()
    red = e.init_params(K, {"v": U.shape[1]}, target=1.0)
    red[0]["W"][0] = jnp.asarray(rng.normal(size=(1, U.shape[1])) * 0.1)
    raw = project_params(e, red, {"v": U.T})
    assert raw[0]["W"][0].shape == (1, n)
    perp = rng.normal(size=(1, n))
    perp = perp - (perp @ U) @ U.T
    off = project_params(e, raw, {})
    off[0]["W"][0] = raw[0]["W"][0] + jnp.asarray(perp)
    vu = jnp.asarray(v @ U)
    for i in range(N):
        m_red = e.evaluate(red, {"v": vu[i]})
        np.testing.assert_allclose(e.evaluate(raw, {"v": jnp.asarray(v[i])}), m_red, rtol=1e-5)
        np.testing.assert_allclose(e.evaluate(off, {"v": jnp.asarray(v[i])}), m_red, rtol=1e-5)


@pytest.mark.parametrize("expr", [
    Exp("v", shared_weights=(0,)) + Exp(),
    Exp("v", shared_weights=(0,), shared_bias=(0,)),
    Exp() * Sigmoid("v", shared_weights=(0,)) + Exp(),
])
def test_the_reduced_fit_is_the_same_optimum(x64, expr):
    """The claim that licenses default-on: with fewer rows than dependency coordinates, the fit on
    the span reaches the full-width fit's penalised optimum --- loss and the raw weights both."""
    coords, grads, bc, dc = _funnel_evidence(60, 30, 100)
    a = regression._fit(expr, bc, dc, coords, grads, span_reduce=False, max_iter=500)
    b = regression._fit(expr, bc, dc, coords, grads, span_reduce=True, max_iter=500)
    assert b.loss == pytest.approx(a.loss, rel=1e-9, abs=1e-9)

    def weights(params, e):
        if isinstance(e, (regression.metric_expr.Sum, regression.metric_expr.Product)):
            return weights(params[0], e.a) + weights(params[1], e.b)
        return list(params["W"])
    for x, y in zip(weights(a.params, expr), weights(b.params, expr)):
        assert y.shape == x.shape                      # back in the raw, full-width layout
        np.testing.assert_allclose(y, x, atol=1e-7)


def test_a_warm_start_is_projected_and_reaches_the_same_optimum(x64):
    """A raw full-width warm start goes in through the projection: same optimum as cold."""
    coords, grads, bc, dc = _funnel_evidence(60, 30, 100)
    e = Exp("v", shared_weights=(0,)) + Exp()
    cold = regression._fit(e, bc, dc, coords, grads, span_reduce=True, max_iter=500)
    rng = np.random.default_rng(5)
    init = project_params(e, cold.params, {})
    init[0]["W"][0] = cold.params[0]["W"][0] + jnp.asarray(rng.normal(size=(1, 100)) * 0.01)
    warm = regression._fit(e, bc, dc, coords, grads, span_reduce=True, init=init, max_iter=500)
    assert warm.loss == pytest.approx(cold.loss, rel=1e-9)


def test_full_rank_evidence_is_left_exactly_alone():
    """More rows than coordinates and no repeats: no basis, the same program, the same bits."""
    coords, grads, bc, dc = _funnel_evidence(300, 10, 20, duplicate=False)
    e = Exp("v", shared_weights=(0,)) + Exp()
    a = regression._fit(e, bc, dc, coords, grads, span_reduce=False)
    b = regression._fit(e, bc, dc, coords, grads, span_reduce=True)
    assert a.loss == b.loss
    np.testing.assert_array_equal(a.params[0]["W"][0], b.params[0]["W"][0])


def test_a_sparse_atom_is_not_reduced():
    """``SpExp`` over a rank-deficient equal-width dependency: unaffected by the flag."""
    coords, grads, bc, dc = _funnel_evidence(20, 30, 30)
    e = SpExp("v", shared_weights=(0,)) + Exp()
    a = regression._fit(e, bc, dc, coords, grads, span_reduce=False)
    b = regression._fit(e, bc, dc, coords, grads, span_reduce=True)
    assert a.loss == b.loss
