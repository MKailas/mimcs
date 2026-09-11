"""Tests for the metric regression's ridge toward its scale-aware init.

The fit is anchored at ``expr.init_params(..., target=scale)`` — zero weights, biases at the
empirical log second moment — and penalised for leaving it::

    objective(theta) = mean_loss(theta) + sum_leaves ||theta - theta_init||^2 / (2 sigma^2 N)

Three things here are easy to get wrong in ways nothing else would catch: the ``1/N`` that makes
``sigma`` a prior standard deviation rather than a tuning constant, the ``/K`` that charges a
*shared* leaf once rather than once per lane, and the distinction between the penalised objective
the optimiser minimises and the **data** loss AIC must rank on.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from mimcs.hmc.metric_expr import Exp, Sigmoid, SpExp
from mimcs.factory.regression import (
    RIDGE_SIGMA, fit_metric_expr, ridge_penalty_vec, select_metric, aic)


# --- the penalty itself --------------------------------------------------------- #

def _tree(W, b):
    return {"W": [jnp.asarray(W)], "b": jnp.asarray(b)}


def test_the_penalty_is_the_prior_it_claims_to_be():
    """One weight a known distance from the anchor costs exactly ``d^2 / (2 sigma^2 N)``."""
    K, N, sigma, d = 1, 100, 5.0, 0.5
    pen = ridge_penalty_vec(_tree([[d]], [0.0]), _tree([[0.0]], [0.0]), K, sigma, N)
    assert float(jnp.sum(pen)) == pytest.approx(d ** 2 / (2 * sigma ** 2 * N), rel=1e-6)


def test_doubling_the_rows_halves_the_penalty():
    """The ``1/N`` pinned directly. Without it ``sigma`` would stop being a prior standard
    deviation and become a constant whose meaning shifts with the dataset size — at N=4000 the
    penalty would be ~4000x too strong, an effective sigma of 0.08 rather than 5."""
    args = (_tree([[0.5]], [0.0]), _tree([[0.0]], [0.0]), 1, 5.0)
    assert (float(jnp.sum(ridge_penalty_vec(*args, 200)))
            == pytest.approx(0.5 * float(jnp.sum(ridge_penalty_vec(*args, 100))), rel=1e-6))


def test_a_shared_leaf_is_charged_once_not_once_per_lane():
    """A ``(1, f)`` leaf lives in the solver's shared block, so every lane may depend on it; handing
    each lane the undivided scalar would inflate the shared Hessian by ``K lambda``."""
    K, f, N, sigma, w = 4, 2, 100, 5.0, 0.5
    shared = ridge_penalty_vec(_tree(np.full((1, f), w), np.zeros(1)),
                               _tree(np.zeros((1, f)), np.zeros(1)), K, sigma, N)
    per_lane = ridge_penalty_vec(_tree(np.full((K, f), w), np.zeros(K)),
                                 _tree(np.zeros((K, f)), np.zeros(K)), K, sigma, N)
    assert float(jnp.sum(shared)) == pytest.approx(f * w ** 2 / (2 * sigma ** 2 * N), rel=1e-6)
    # ... which is exactly the per-lane total divided by K -- the control that makes this
    # non-vacuous, since the undivided (wrong) accounting would instead make the two equal.
    assert float(jnp.sum(shared)) == pytest.approx(float(jnp.sum(per_lane)) / K, rel=1e-6)
    assert not np.isclose(float(jnp.sum(shared)), float(jnp.sum(per_lane)))
    # Spread evenly, so no lane depends on another lane's parameters.
    assert np.allclose(np.asarray(shared), float(jnp.sum(shared)) / K)


def test_the_anchor_is_the_init_not_the_origin():
    """A bias sitting at its scale-aware value is unpenalised. Anchoring biases at *zero* would
    pull ``M`` toward 1, which is the badly-scaled-target failure the scale-aware init exists to
    prevent."""
    K, N = 3, 100
    b_init = jnp.log(jnp.asarray([1.0, 1e4, 1e-4]))          # wildly non-unit scales
    at_anchor = _tree(np.zeros((K, 1)), b_init)
    assert float(jnp.sum(ridge_penalty_vec(at_anchor, at_anchor, K, 5.0, N))) == 0.0
    moved = _tree(np.zeros((K, 1)), b_init + 1.0)
    assert float(jnp.sum(ridge_penalty_vec(moved, at_anchor, K, 5.0, N))) > 0.0


def test_disabling_it_is_exactly_zero():
    t = _tree([[3.0]], [2.0])
    for off in (None, float("inf")):
        assert float(jnp.sum(ridge_penalty_vec(t, _tree([[0.0]], [0.0]), 1, off, 100))) == 0.0


# --- the fit -------------------------------------------------------------------- #

def _funnel_evidence(N=4000, seed=0):
    """Scores whose conditional variance is ``e^{-v}``: the ideal metric is ``W=-1, b=0``."""
    rng = np.random.default_rng(seed)
    v = rng.normal(0.0, 1.5, size=N)
    g = rng.normal(size=N) * np.exp(-0.5 * v)
    return np.column_stack([v, np.zeros(N)]), np.column_stack([np.zeros(N), g])


def test_sigma_off_reproduces_the_unregularised_fit_exactly():
    coords, grads = _funnel_evidence()
    a = fit_metric_expr(Exp("v"), [1], {"v": [0]}, coords, grads, ridge_sigma=None)
    b = fit_metric_expr(Exp("v"), [1], {"v": [0]}, coords, grads, ridge_sigma=None)
    assert a[0] == b[0]                                        # deterministic
    assert np.array_equal(np.asarray(a[1]["W"][0]), np.asarray(b[1]["W"][0]))


def test_the_default_sigma_leaves_an_identified_fit_alone():
    """At sigma=5 the penalty is ~0.00% of an identified fit's loss, so recovery is untouched."""
    coords, grads = _funnel_evidence()
    off, _ = fit_metric_expr(Exp("v"), [1], {"v": [0]}, coords, grads, ridge_sigma=None)
    on, params = fit_metric_expr(Exp("v"), [1], {"v": [0]}, coords, grads,
                                 ridge_sigma=RIDGE_SIGMA)
    W = float(np.asarray(params["W"][0]).ravel()[0])
    assert abs(W + 1.0) < 0.05, W
    assert on == pytest.approx(off, rel=1e-5)


def test_a_tight_sigma_does_bite():
    """The control for the test above: the ridge is weak by choice, not inert by construction."""
    coords, grads = _funnel_evidence()
    _, loose = fit_metric_expr(Exp("v"), [1], {"v": [0]}, coords, grads, ridge_sigma=5.0)
    _, tight = fit_metric_expr(Exp("v"), [1], {"v": [0]}, coords, grads, ridge_sigma=0.1)
    w = lambda p: abs(float(np.asarray(p["W"][0]).ravel()[0]))
    assert w(tight) < w(loose) - 0.01, (w(tight), w(loose))    # shrunk toward zero


def test_the_reported_loss_is_the_data_term_not_the_penalised_objective():
    """AIC's ``2 N loss`` is the data term and ``2 k`` is the complexity term; folding the penalty
    in would double-charge complexity, and would make these numbers incomparable with an
    unregularised run. A regularised fit must therefore report a loss no *lower* than the
    unregularised optimum of the same objective."""
    coords, grads = _funnel_evidence()
    free, _ = fit_metric_expr(Exp("v"), [1], {"v": [0]}, coords, grads, ridge_sigma=None)
    tight, params = fit_metric_expr(Exp("v"), [1], {"v": [0]}, coords, grads, ridge_sigma=0.1)
    assert tight >= free                                        # constrained: worse data fit
    pen = float(jnp.sum(ridge_penalty_vec(
        params, Exp("v").init_params(1, {"v": 1}), 1, 0.1, len(coords))))
    assert pen > 0.0 and tight < free + pen                     # and the penalty is not in it


def test_both_optimiser_arms_fit_the_same_penalised_objective():
    """The L-BFGS control arm must see the ridge too, or it is a control of a different objective."""
    coords, grads = _funnel_evidence()
    args = (Exp("v"), [1], {"v": [0]}, coords, grads)
    ln, _ = fit_metric_expr(*args, optimizer="newton", ridge_sigma=0.5)
    ll, _ = fit_metric_expr(*args, optimizer="lbfgs", max_iter=500, ridge_sigma=0.5)
    assert abs(ln - ll) < 1e-4, (ln, ll)


def test_it_pins_a_direction_the_data_does_not_identify():
    """The case this was built for. On a flat target a sigmoid gate can sit anywhere outside the
    data range for the same loss, so a warm start — which, unlike the scale-aware init, does not
    start the weights at zero — lets it drift: ``max|theta| = 565`` against the cold fit's ~11.
    The unregularised arm in the same test shows the drift, so the assertion discriminates."""
    from test_factory_learned_metric import _flat_scores
    coords, grads = _flat_scores(1e5)
    bcols, dep = list(range(8)), {"v": [8]}

    def worst(sigma):
        ranked = select_metric(bcols, dep, coords, grads, ridge_sigma=sigma, warm_start=True)
        assert ranked[0].expr.deps() == set()                   # baseline still wins either way
        return max(max(float(np.max(np.abs(np.asarray(l))))
                       for l in jax.tree_util.tree_leaves(c.params)) for c in ranked)

    assert worst(None) > 100.0                                  # the control: it really drifts
    assert worst(RIDGE_SIGMA) < 50.0                            # and the ridge really pins it


def test_selection_still_recovers_a_sparse_metric_under_the_ridge():
    """A weak prior must not change what gets selected on evidence that identifies the answer."""
    rng = np.random.default_rng(0)
    N, K = 500, 20
    s = rng.normal(0.0, 1.0, size=(N, K))
    g = rng.normal(size=(N, K)) * np.exp(-0.5 * s)
    coords = np.concatenate([s, np.zeros((N, K))], axis=1)
    grads = np.concatenate([np.zeros((N, K)), g], axis=1)
    best = select_metric(list(range(K, 2 * K)), {"s": list(range(K))}, coords, grads,
                         ridge_sigma=RIDGE_SIGMA)[0]
    assert "SpExp('s'" in repr(best.expr), repr(best.expr)
    W = np.asarray(best.params["W"][0]).ravel()
    assert np.allclose(W, -1.0, atol=0.2), W
