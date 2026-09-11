"""Tests for metric weights **shared** across a block's coordinates.

A weight of shape ``(block_dim, feat)`` has a broadcastable sibling of shape ``(1, feat)`` --- one
value serving every coordinate, declared as ``Exp(d, shared_weights=(0,))``. That is the right
model whenever the geometry has a single cause: on a horseshoe the funnel comes from the positivity
and ``log`` of ``lambda``, the same relation for every coordinate, so one slope should serve all of
them.

Every assertion here is about something that **broadcasts rather than raising** when it is wrong,
which is why they are shape assertions and paired controls rather than value checks:

* a shared leaf that quietly becomes per-coordinate again during warmup;
* a fully-shared expression evaluating to ``(1,)`` and shortening a log-determinant;
* an AIC comparison that would prefer sharing whether or not the truth is shared.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from mimcs.hmc.metric_expr import Exp, Sigmoid, SpExp, check_params
from mimcs.hmc.block_riemannian import build_block
from mimcs.factory.regression import (
    enumerate_candidates, fit_metric_expr, select_metric, sharing_variants)
from mimcs.testing.problems import neal_funnel_blocks
from mimcs.testing.runner import explicit_rmhmc


# --- the spec ----------------------------------------------------------------- #

def test_shared_axes_change_the_emitted_shapes_and_the_count():
    e = SpExp("l")
    assert e.param_shapes(4, {"l": 4}) == {"W": [(4, 1)], "b": (4,)}
    assert e.n_params(4, {"l": 4}) == 8
    w = SpExp("l", shared_weights=(0,))
    assert w.param_shapes(4, {"l": 4}) == {"W": [(1, 1)], "b": (4,)}
    assert w.n_params(4, {"l": 4}) == 5
    both = SpExp("l", shared_weights=(0,), shared_bias=(0,))
    assert both.param_shapes(4, {"l": 4}) == {"W": [(1, 1)], "b": (1,)}
    assert both.n_params(4, {"l": 4}) == 2


def test_n_params_and_init_params_cannot_drift():
    """Both are driven by ``param_shapes``; AIC charging a different count than the fit actually
    carries is the failure that makes a shared candidate unselectable while looking fine."""
    for e in [Exp("v"), Exp("v", shared_weights=(0,)), SpExp("l", shared_bias=(0,)),
              Exp() * Sigmoid("v", shared_weights=(0,)) + Exp()]:
        dims = {"v": 3, "l": 4}
        got = sum(int(np.size(leaf))
                  for leaf in jax.tree_util.tree_leaves(e.init_params(4, dims)))
        assert got == e.n_params(4, dims), repr(e)


def test_only_the_coordinate_axis_is_shareable():
    with pytest.raises(ValueError, match="cannot be shared"):
        Exp("v", shared_weights=(1,))
    with pytest.raises(ValueError, match="cannot be shared"):
        Exp("v", shared_bias=(0, 1))


def test_a_dep_less_atom_carries_no_weight_sharing():
    """``Exp()`` has no weights, so ``with_sharing`` must not stamp a meaningless flag on it ---
    two identical expressions would otherwise print and compare differently."""
    assert repr(Exp().with_sharing((0,), ())) == "Exp()"
    assert repr((SpExp("l") + Exp()).with_sharing((0,), ())) == \
        "SpExp('l', shared_weights=(0,)) + Exp()"


def test_a_shared_bias_starts_at_the_geometric_mean_of_the_target():
    """The regression passes a ``(block_dim,)`` per-coordinate target. ``jnp.zeros((1,)) +
    log(target)`` broadcasts straight back to ``(block_dim,)``, so a bias declared shared would
    come back per-coordinate -- fitted as a dense one while charged the shared count."""
    target = jnp.asarray([1.0, 4.0, 9.0, 16.0])
    e = SpExp("l", shared_weights=(0,), shared_bias=(0,))
    params = e.init_params(4, {"l": 4}, target=target)
    assert np.shape(params["b"]) == (1,)
    M = np.asarray(e.evaluate(params, {"l": jnp.zeros(4)}))
    assert np.allclose(M, np.exp(np.mean(np.log(np.asarray(target)))), rtol=1e-5)
    # The per-coordinate bias is untouched by the reduction.
    plain = SpExp("l", shared_weights=(0,)).init_params(4, {"l": 4}, target=target)
    assert np.allclose(np.asarray(plain["b"]), np.log(np.asarray(target)), rtol=1e-5)


# --- the runtime -------------------------------------------------------------- #

def _block(expr, dim=4):
    """The block and its problem. ``neal_funnel_blocks(dim=n)`` puts ``v`` and ``x`` in the same
    ``n`` coordinates, so ``x`` has ``n - 1`` of them --- read the size off the block."""
    prob = neal_funnel_blocks(dim=dim, scale=3.0)
    return build_block(prob.model, "x", expr), prob


def test_a_shared_metric_is_the_same_function_as_its_tiled_unshared_twin():
    """A shared weight and the equivalent per-coordinate weight with identical rows must give the
    same mass, the same energy (log-determinant included) and the same KL loss. ``_energy`` is the
    one that does not broadcast on its own: ``jnp.sum(jnp.log(M))`` over a ``(1,)`` mass sums one
    element instead of ``size``, and ``flow`` differentiates exactly that."""
    shared, prob = _block(Exp("v", shared_weights=(0,), shared_bias=(0,)))
    plain, _ = _block(Exp("v"))
    size = shared.size
    ps = {"W": [jnp.asarray([[-0.7]])], "b": jnp.asarray([0.3])}
    pp = {"W": [jnp.full((size, 1), -0.7)], "b": jnp.full((size,), 0.3)}
    q = jnp.asarray(np.linspace(-1.0, 1.0, prob.model.coord_dim))
    p_i = jnp.asarray(np.linspace(0.5, -0.5, size))

    assert np.allclose(np.asarray(shared._mass(q, None, ps)),
                       np.asarray(plain._mass(q, None, pp)))
    assert np.shape(shared._mass(q, None, ps)) == (size,)      # broadcast, not (1,)
    assert np.isclose(float(shared._energy(q, None, p_i, ps)),
                      float(plain._energy(q, None, p_i, pp)))
    score = jnp.asarray(np.linspace(-2.0, 2.0, prob.model.coord_dim))
    assert np.isclose(float(shared.metric_loss(ps, q, None, score)),
                      float(plain.metric_loss(pp, q, None, score)))


def test_a_fully_shared_expression_would_shorten_the_log_determinant():
    """The control for the test above: without the block's broadcast, a ``(1,)`` mass makes the
    energy's log-det term ``size`` times too small. Pinning the size of the error keeps the
    previous test from passing for the wrong reason."""
    size, M = 5, 2.0
    p_i = jnp.ones(size)
    correct = 0.5 * jnp.sum(p_i ** 2 / jnp.full((size,), M)) + \
        0.5 * jnp.sum(jnp.log(jnp.full((size,), M)))
    unbroadcast = 0.5 * jnp.sum(p_i ** 2 / jnp.asarray([M])) + \
        0.5 * jnp.sum(jnp.log(jnp.asarray([M])))
    assert not np.isclose(float(correct), float(unbroadcast))
    assert np.isclose(float(correct - unbroadcast), 0.5 * (size - 1) * np.log(M))


@pytest.mark.parametrize("expr,shapes", [
    (Exp("v"), [(3, 1), (3,)]),
    (Exp("v", shared_weights=(0,)), [(1, 1), (3,)]),
    (Exp("v", shared_weights=(0,), shared_bias=(0,)), [(1, 1), (1,)]),
])
def test_sharing_survives_adaptation(expr, shapes):
    """The regression test for the silent un-sharing: the old update keyed its scale off the
    ``(block_dim,)`` vector, so ``(block_dim, 1) * (1, feat)`` promoted a shared leaf to
    ``(block_dim, feat)`` on the first warmup step -- no exception, the sharing just evaporated."""
    prob = neal_funnel_blocks(dim=4, scale=3.0)
    builder = explicit_rmhmc(metrics={"x": expr}, n_leapfrog=15, step_size=0.25,
                             target_accept=0.9)
    s = builder(prob.model, seed=0)
    assert [np.shape(a) for a in jax.tree_util.tree_leaves(s.state.ham_params["x"])] == shapes
    s.warmup(50)
    assert [np.shape(a) for a in jax.tree_util.tree_leaves(s.state.ham_params["x"])] == shapes


def test_the_old_scale_keying_is_what_un_shared_it():
    """Makes the test above non-vacuous by showing the discarded arithmetic really did promote."""
    w, gw = jnp.zeros((1, 1)), jnp.ones((1, 1))
    per_coord_scale = jnp.full((3,), 0.1)
    old = per_coord_scale.reshape((per_coord_scale.shape[0],) + (1,) * (gw.ndim - 1))
    new = jnp.asarray([0.1]).reshape((gw.shape[0],) + (1,) * (gw.ndim - 1))
    assert np.shape(w - old * gw) == (3, 1)          # what it used to do
    assert np.shape(w - new * gw) == (1, 1)          # what it does now


def test_a_shared_weight_learns_the_funnel_slope_with_one_parameter():
    """On the funnel every coordinate has the same ideal ``M_x(v) = e^{-v}``, so one pooled slope
    must reach the same answer the per-coordinate fit does -- and stay stable at the same learning
    rate, which is what dividing a shared unit's gradient by the coordinates it serves is for."""
    prob = neal_funnel_blocks(dim=4, scale=3.0)
    builder = explicit_rmhmc(metrics={"x": Exp("v", shared_weights=(0,))},
                             n_leapfrog=25, step_size=0.25, target_accept=0.9)
    s = builder(prob.model, seed=0)
    s.warmup(8000)
    W = np.asarray(s.state.ham_params["x"]["W"][0]).ravel()
    assert W.shape == (1,)
    assert abs(float(W[0]) + 1.0) < 0.2, W


# --- the structural guard ------------------------------------------------------ #

def test_check_params_catches_a_mismatched_init_in_both_directions():
    """Nothing else validates a metric parameter tree, and a wrong one broadcasts rather than
    raising -- a metric fitted under one sharing pattern and paired with another expression would
    sample happily from the wrong Hamiltonian."""
    plain, shared = Exp("v"), Exp("v", shared_weights=(0,))
    dims = {"v": 2}
    check_params(shared, shared.init_params(4, dims), 4, dims)      # matching: no raise
    with pytest.raises(ValueError, match="shape"):
        check_params(plain, shared.init_params(4, dims), 4, dims)
    with pytest.raises(ValueError, match="shape"):
        check_params(shared, plain.init_params(4, dims), 4, dims)


def test_a_block_rejects_a_metric_init_of_the_wrong_sharing():
    prob = neal_funnel_blocks(dim=4, scale=3.0)
    shared = Exp("v", shared_weights=(0,))
    bad = Exp("v").init_params(4, {"v": 1})
    blk = build_block(prob.model, "x", shared, init=bad)
    with pytest.raises(ValueError, match="metric_init"):
        blk.initial_mass_params(prob.model.coord_dim)


# --- enumeration and selection -------------------------------------------------- #

def test_the_ladder_is_cheapest_first_and_de_duplicated():
    v = sharing_variants(SpExp("l") + Exp(), include_shared=True)
    counts = [e.n_params(100, {"l": 100}) for e in v]
    assert counts == sorted(counts), counts
    assert repr(v[-1]) == "SpExp('l') + Exp()"                 # the historical pool is last
    # `Exp()` has no weights, so two of the three rungs coincide: it must not be fitted twice.
    assert len(sharing_variants(Exp(), include_shared=True)) == 2


def test_enumeration_offers_shared_forms_and_the_flag_removes_them():
    kw = dict(param_budget=40000, max_candidates=60)
    with_shared = enumerate_candidates(2000, {"lam": 2000}, include_shared=True, **kw)
    without = enumerate_candidates(2000, {"lam": 2000}, include_shared=False, **kw)
    assert any("shared_weights" in repr(e) for e in with_shared)
    assert not any("shared_weights" in repr(e) for e in without)
    assert len(with_shared) > len(without)


def _sloped_evidence(slopes, N=400, seed=0):
    """Evidence whose block score has conditional variance ``exp(-slope_j * lam_j)``."""
    K = len(slopes)
    rng = np.random.default_rng(seed)
    lam = rng.normal(0.0, 1.0, size=(N, K))
    g = rng.normal(size=(N, K)) * np.exp(-0.5 * np.asarray(slopes)[None, :] * lam)
    coords = np.concatenate([lam, np.zeros((N, K))], axis=1)
    grads = np.concatenate([np.zeros((N, K)), g], axis=1)
    return coords, grads, list(range(K, 2 * K)), {"lam": list(range(K))}


def test_aic_picks_a_shared_slope_when_the_truth_is_shared():
    coords, grads, bcols, dep = _sloped_evidence(np.full(60, 1.0))
    best = select_metric(bcols, dep, coords, grads)[0]
    assert "shared_weights" in repr(best.expr), repr(best.expr)
    assert best.n_params <= 4, best.n_params        # a handful, not 60 per leaf


def test_aic_refuses_to_share_when_the_slopes_genuinely_differ():
    """The control. Without it the test above passes for a selector that always prefers the
    cheapest candidate, which would be a worse metric on every problem whose coordinates differ."""
    coords, grads, bcols, dep = _sloped_evidence(np.linspace(0.2, 2.0, 60))
    best = select_metric(bcols, dep, coords, grads)[0]
    assert "shared_weights" not in repr(best.expr), repr(best.expr)
    assert best.n_params >= 60, best.n_params


def test_a_warm_start_reaches_the_same_fit_as_a_cold_one():
    """The ladder warm-starts each rung from its pooled parent. That must change the cost, not the
    answer -- a warm start that moved the minimiser would silently bias the AIC comparison."""
    coords, grads, bcols, dep = _sloped_evidence(np.linspace(0.2, 2.0, 20))
    expr = SpExp("lam")
    cold_loss, cold = fit_metric_expr(expr, bcols, dep, coords, grads)
    parent = SpExp("lam", shared_weights=(0,), shared_bias=(0,))
    _, pp = fit_metric_expr(parent, bcols, dep, coords, grads)
    warm = jax.tree_util.tree_map(lambda a, b: jnp.broadcast_to(a, jnp.shape(b)),
                                  pp, expr.init_params(20, {"lam": 20}))
    warm_loss, _ = fit_metric_expr(expr, bcols, dep, coords, grads, init=warm)
    assert abs(warm_loss - cold_loss) < 1e-4, (warm_loss, cold_loss)


def test_an_all_shared_expression_solves_through_the_arrow_path():
    """Every leaf shared leaves the per-lane block empty (``p == 0``); the Schur solve then
    degenerates to a plain ``s x s`` Newton, which used to be a zero-size reduction error."""
    coords, grads, bcols, dep = _sloped_evidence(np.full(12, 1.0))
    expr = SpExp("lam", shared_weights=(0,), shared_bias=(0,))
    loss, params = fit_metric_expr(expr, bcols, dep, coords, grads)
    assert np.isfinite(loss)
    assert np.shape(params["W"][0]) == (1, 1) and np.shape(params["b"]) == (1,)
    assert abs(float(np.asarray(params["W"][0]).ravel()[0]) + 1.0) < 0.2
