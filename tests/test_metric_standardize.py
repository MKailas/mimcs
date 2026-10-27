"""Tests for the metric regression's covariate standardisation (``regression.STANDARDIZE``).

A fit on ``z = (x - centre) / scale`` with its parameters mapped back to raw coordinates is the
same function class with the same unpenalised loss; what moves is the ridge, which becomes a prior
per *scale* of the covariate instead of per unit. So the properties pinned here are: the map-back
is exact (or the whole scheme silently changes the metric), the fit becomes unit-free, nothing
moves with the ridge off, and the ``"stein"`` centre and scale are the target's rather than the
pilot's.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from mimcs.hmc.metric_expr import Exp, Sigmoid, SpExp, SpSigmoid
from mimcs.factory import regression
from mimcs.factory.regression import (
    SHARING_LADDER, covariate_stats, fit_metric_expr, pooled_deps, reparametrize_params, _pool)


# --- the map-back ------------------------------------------------------------------------------ #

_B, _K = 4, 3


def _stats(rng):
    return {"v": (np.array([3.0, -1.0, 0.5]), np.array([0.5, 2.0, 1.5])),
            "u": (rng.normal(size=_B), np.abs(rng.normal(size=_B)) + 0.5)}


@pytest.mark.parametrize("form", [
    Exp("v") + Exp(), Exp() * Sigmoid("v") + Exp(), Exp("v", "u"),
    Exp("v", features="quadratic"), SpExp("u") + Exp(), SpExp("u", features="quadratic"),
    Exp() * SpSigmoid("u")], ids=repr)
def test_the_map_back_reproduces_the_metric_for_every_sharing(form):
    """``M`` of the mapped-back parameters on raw data equals ``M`` of the fitted ones on the
    standardised data, with every leaf keeping its shape, and mapping in again returns the fit."""
    rng = np.random.default_rng(0)
    st = _stats(rng)
    x = {"v": st["v"][0] + st["v"][1] * rng.normal(size=(40, _K)),
         "u": st["u"][0] + st["u"][1] * rng.normal(size=(40, _B))}
    for sw, sb in SHARING_LADDER:
        expr = form.with_sharing(sw, sb)
        p = jax.tree_util.tree_map(lambda a: a + 0.3 * rng.normal(size=np.shape(a)),
                                   expr.init_params(_B, {"v": _K, "u": _B}))
        s = {d: (_pool(*st[d]) if d in pooled_deps(expr) else st[d]) for d in expr.deps()}
        raw = reparametrize_params(expr, p, {d: (1 / sd, -m / sd) for d, (m, sd) in s.items()})
        assert jax.tree_util.tree_structure(raw) == jax.tree_util.tree_structure(p)
        assert all(np.shape(a) == np.shape(b) for a, b in
                   zip(jax.tree_util.tree_leaves(raw), jax.tree_util.tree_leaves(p)))
        z = {d: (x[d] - m) / sd for d, (m, sd) in s.items()}
        M_std = jax.vmap(lambda r: expr.evaluate(p, r))(z)
        M_raw = jax.vmap(lambda r: expr.evaluate(raw, r))({d: x[d] for d in s})
        np.testing.assert_allclose(np.log(M_raw), np.log(M_std), atol=1e-4)
        back = reparametrize_params(expr, raw, {d: (sd, m) for d, (m, sd) in s.items()})
        for a, b in zip(jax.tree_util.tree_leaves(back), jax.tree_util.tree_leaves(p)):
            np.testing.assert_allclose(a, b, atol=1e-4)


def test_a_shared_sparse_weight_is_standardised_by_one_pooled_scale():
    """One weight over every coordinate of the dependency cannot absorb per-coordinate scales, so
    such a dependency is pooled --- and a per-coordinate scale for it is refused, not broadcast."""
    shared = SpExp("u").with_sharing((0,), (0,))
    assert pooled_deps(shared) == {"u"}
    assert pooled_deps(SpExp("u")) == set() and pooled_deps(Exp("u")) == set()
    p = shared.init_params(_B, {"u": _B})
    with pytest.raises(ValueError, match="pooled"):
        reparametrize_params(shared, p, {"u": (np.ones(_B) * 2, np.zeros(_B))})


# --- the estimators ---------------------------------------------------------------------------- #

def test_stein_reads_the_targets_scale_from_an_underdispersed_shifted_pilot():
    """A pilot 2 sd off and 6x too narrow: the scores' regression line is the target's whatever
    the design, so ``"stein"`` recovers ``N(2, 3^2)``; the empirical estimate is the control."""
    rng = np.random.default_rng(1)
    x = rng.normal(2.0 + 6.0, 0.5, size=(4000, 3))
    g = -(x - 2.0) / 9.0
    m, s = covariate_stats(x, g, "stein")
    np.testing.assert_allclose(s, 3.0, rtol=0.02)
    np.testing.assert_allclose(m, 2.0, atol=0.06)
    m_e, s_e = covariate_stats(x, g, "empirical")
    assert np.all(s_e < 0.6) and np.all(m_e > 7.5)


def test_a_frozen_column_reads_its_scale_from_the_scores():
    """A coordinate stuck in its pilot (AR(1) jitter of sd 0.02, ESS of a handful) whose score keeps
    moving with a coupled coordinate: the slope is resolved and reads the target's scale. Gating on the ESS of ``x`` itself would call it unresolved and keep the pilot's 0.02."""
    from mimcs.diagnostics import ess
    rng = np.random.default_rng(2)
    n, P = 2000, np.linalg.inv(np.array([[1.0, 0.5], [0.5, 1.0]]))
    e = rng.normal(size=n)
    x1 = np.empty(n); x1[0] = 1.5
    for t in range(1, n):
        x1[t] = 1.5 + 0.999 * (x1[t - 1] - 1.5) + 0.02 * np.sqrt(1 - 0.999 ** 2) * e[t]
    x2 = 0.5 * x1 + np.sqrt(0.75) * rng.normal(size=n)        # x2 | x1 under the target
    x = np.c_[x1, x2]
    g = -x @ P
    assert ess(x1[:, None])[0] < 20
    _, s = covariate_stats(x, g, "stein")
    # x2 follows x1 as under the target, so the slope is the *marginal* curvature: sd 1.
    assert 0.85 < s[0] < 1.15
    assert covariate_stats(x, g, "empirical")[1][0] < 0.1       # the control


def test_an_unresolved_slope_never_reads_below_the_pilot_spread():
    """Pure score noise around a 100x-too-narrow pilot: where the slope is not resolved the scale
    is the larger of the pilot's spread and ``1 / sd(g)`` --- never the noisy Stein value, which is
    biased low, the dangerous direction."""
    rng = np.random.default_rng(3)
    x = rng.normal(0.0, 0.01, size=(1000, 6))
    g = -x + 30.0 * rng.normal(size=(1000, 6))
    m_s, s_s, beta, se1, n_eff = regression._stein_moments(x, g)
    unresolved = ~(np.isfinite(s_s) & (-beta > se1 / np.sqrt(n_eff)))
    assert unresolved.sum() >= 3
    m, s = covariate_stats(x, g, "stein")
    np.testing.assert_allclose(s[unresolved], np.maximum(x.std(0), 1 / g.std(0))[unresolved])
    np.testing.assert_allclose(m[unresolved], x.mean(0)[unresolved])
    assert np.all(s >= x.std(0))


def test_a_degenerate_column_is_left_raw():
    x = np.c_[np.full(50, 4.0), np.arange(50.0)]
    m, s = covariate_stats(x, np.zeros_like(x), "empirical")
    assert (m[0], s[0]) == (0.0, 1.0) and s[1] > 0


# --- the fit ----------------------------------------------------------------------------------- #

def _evidence(unit=1.0, n=2000, seed=4):
    """Two block coordinates whose score scale is ``exp(u)``, with ``u`` the standardised covariate
    ``v = (5 + 0.01 u) * unit``: one e-fold per covariate sd, i.e. a raw slope of 100 / unit."""
    rng = np.random.default_rng(seed)
    u = rng.normal(size=n)
    v = (5.0 + 0.01 * u) * unit
    g_block = np.exp(0.5 * u)[:, None] * rng.normal(size=(n, 2))
    g_v = -(v - 5.0 * unit) / (0.01 * unit) ** 2
    coords = np.c_[rng.normal(size=(n, 2)), v]
    return coords, np.c_[g_block, g_v]


def _metric(expr, params, coords):
    return np.asarray(jax.vmap(lambda r: expr.evaluate(params, {"v": r}))(
        jnp.asarray(coords[:, 2:3], float)))


@pytest.mark.parametrize("method", ["empirical", "stein"])
def test_a_standardised_fit_does_not_depend_on_the_covariates_units(method):
    """Rescaling the covariate by 1000 leaves the fitted metric where it was, and the slope is
    found; on raw coordinates the ridge pins it (raw slope 100 costs 100^2 / (2 sigma^2) nats)."""
    expr = Exp("v") + Exp()
    fits = {}
    for unit in (1.0, 1000.0):
        coords, grads = _evidence(unit)
        _, p = fit_metric_expr(expr, [0, 1], {"v": [2]}, coords, grads, standardize=method)
        fits[unit] = (_metric(expr, p, coords), p)
    np.testing.assert_allclose(np.log(fits[1.0][0]), np.log(fits[1000.0][0]), atol=2e-3)
    slope = np.asarray(fits[1.0][1][0]["W"][0])[:, 0]
    np.testing.assert_allclose(slope, 100.0, rtol=0.15)
    coords, grads = _evidence(1.0)
    _, p_raw = fit_metric_expr(expr, [0, 1], {"v": [2]}, coords, grads, standardize=None)
    assert np.all(np.abs(np.asarray(p_raw[0]["W"][0])) < 30.0)


def test_without_the_ridge_standardising_changes_nothing_but_the_path():
    """Same function class, same objective: raw and standardised fits reach the same loss."""
    expr = Exp() * Sigmoid("v") + Exp()
    coords, grads = _evidence(1.0, n=1000, seed=5)
    loss_raw, _ = fit_metric_expr(expr, [0, 1], {"v": [2]}, coords, grads, ridge_sigma=None,
                                  standardize=None)
    loss_std, _ = fit_metric_expr(expr, [0, 1], {"v": [2]}, coords, grads, ridge_sigma=None,
                                  standardize="empirical")
    assert loss_std == pytest.approx(loss_raw, rel=1e-4, abs=1e-4)


def test_the_default_is_raw_coordinates():
    """Off by default, and off is bit-identical to not passing it."""
    assert regression.STANDARDIZE is None
    coords, grads = _evidence(1.0, n=500, seed=6)
    a = fit_metric_expr(Exp("v") + Exp(), [0, 1], {"v": [2]}, coords, grads)
    b = fit_metric_expr(Exp("v") + Exp(), [0, 1], {"v": [2]}, coords, grads, standardize=None)
    assert a[0] == b[0]
    for x, y in zip(jax.tree_util.tree_leaves(a[1]), jax.tree_util.tree_leaves(b[1])):
        np.testing.assert_array_equal(x, y)
