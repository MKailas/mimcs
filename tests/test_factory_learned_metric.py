"""The factory's learned-metric rule end to end.

On a blocked funnel (``v`` scalar, ``x`` its own 30-d block, ``x_i | v ~ N(0, e^v)``) the
conditional score covariance of ``x`` is ``e^{-v}``, so from evidence the metric-regression
rule should turn the ``x`` block into a ``learned_metric`` carrying ``Exp("v") + Exp()`` and
build a sampler that adapts it. Also covers the manual override path and that evidence
without gradients leaves the factory unchanged.
"""

import numpy as np
import jax
import jax.numpy as jnp

from mimcs.testing import neal_funnel_blocks, neal_funnel_vector
from mimcs.factory import analyze
from mimcs.adaptation import MetricAdaptation
from mimcs.hmc import LearnedDiagonalBlock, Exp


def _evidence(model, n=3000, seed=0):
    """Row-aligned coordinates + coordinate-space scores from the exact funnel reference."""
    prob = neal_funnel_blocks(dim=31, scale=3.0)
    coords = prob.exact_sample(n, seed=seed)                 # identity charts: coords == samples
    chp, ci = model.init_chart_hyperparams(), model.init_chart_indices()
    score_fn = jax.grad(lambda c: model.log_prob_at_coordinate(c, chp, ci))
    grads = jax.vmap(score_fn)(jnp.asarray(coords, float))
    return {"coordinates": np.asarray(coords), "gradients": np.asarray(grads)}


def _x_block(spec):
    return next(b for b in spec.blocks if b.names == ["x"])


def test_regresses_learned_metric_from_evidence():
    prob = neal_funnel_blocks(dim=31, scale=3.0)
    model = prob.model
    spec = analyze(model, _evidence(model))

    xb = _x_block(spec)
    assert xb.kind == "learned_metric"
    assert "v" in xb.params["metric"].deps()
    # the fitted metric is the ideal e^{-v}: weight ~ -1 on v across all 30 coordinates
    W0 = np.asarray(xb.params["metric_init"][0]["W"][0]).ravel()   # Exp("v") term's weight
    assert np.allclose(W0, -1.0, atol=0.1), W0
    # the scalar v block, whose only candidate dependency (30-d x) blows the param budget, stays put
    assert next(b for b in spec.blocks if b.names == ["v"]).kind == "diagonal"
    assert any("learned_metric" in r for r in spec.rationale)


def test_build_adapts_and_samples():
    prob = neal_funnel_blocks(dim=31, scale=3.0)
    model = prob.model
    s = analyze(model, _evidence(model)).build(seed=0)

    assert MetricAdaptation in type(s).__mro__
    assert any(isinstance(k, LearnedDiagonalBlock) and k.id == "x" for k in s.kinetics)
    s.warmup(800)
    s.sample(1500)
    draws = s.get_samples_flat()
    assert draws.shape == (1500, 31) and np.all(np.isfinite(draws))
    # MetricAdaptation keeps the learned weight near the ideal -1 through warmup
    W0 = np.asarray(s.state.ham_params["x"][0]["W"][0]).ravel()
    assert abs(W0.mean() - (-1.0)) < 0.2, W0.mean()


def test_manual_metric_override():
    """A user can set a block's metric by hand on the spec and build it (no fitted init)."""
    prob = neal_funnel_blocks(dim=31, scale=3.0)
    model = prob.model
    spec = analyze(model)                              # no evidence -> plain partition
    xb = _x_block(spec)
    xb.kind = "learned_metric"
    xb.params = {"metric": Exp("v") + Exp()}           # no metric_init: uses init_params()
    s = spec.build(seed=0)
    assert any(isinstance(k, LearnedDiagonalBlock) and k.id == "x" for k in s.kinetics)
    s.warmup(300)
    s.sample(500)
    assert np.all(np.isfinite(s.get_samples_flat()))


def test_no_learned_metric_without_gradients():
    """Evidence without gradients (or none) leaves the factory output unchanged."""
    prob = neal_funnel_blocks(dim=31, scale=3.0)
    model = prob.model
    coords = prob.exact_sample(3000, seed=0)
    spec = analyze(model, {"coordinates": np.asarray(coords)})     # no gradients
    assert all(b.kind != "learned_metric" for b in spec.blocks)
    assert all(b.kind != "learned_metric" for b in analyze(model).blocks)  # no evidence at all


# --- sparse (elementwise) learned metric ------------------------------------ #


def _vector_evidence(model, n=3000, seed=0):
    prob = neal_funnel_vector(n=30, scale=3.0)
    coords = prob.exact_sample(n, seed=seed)
    chp, ci = model.init_chart_hyperparams(), model.init_chart_indices()
    score_fn = jax.grad(lambda c: model.log_prob_at_coordinate(c, chp, ci))
    grads = jax.vmap(score_fn)(jnp.asarray(coords, float))
    return {"coordinates": np.asarray(coords), "gradients": np.asarray(grads)}


def test_regresses_sparse_metric_on_vector_funnel():
    """On the vector funnel (``s``, ``x`` both 30-d, ``x_j | s_j ~ N(0, e^{s_j})``) the ``x``
    block's conditional score covariance is elementwise ``e^{-s_j}``, so the regression adopts a
    *sparse* ``SpExp("s")`` metric with per-coordinate weight ~ -1 -- the dense form cannot even
    be enumerated at this dimension (30*30 params > 20*30 budget)."""
    prob = neal_funnel_vector(n=30, scale=3.0)
    model = prob.model
    spec = analyze(model, _vector_evidence(model))

    xb = next(b for b in spec.blocks if b.names == ["x"])
    assert xb.kind == "learned_metric"
    assert "SpExp" in repr(xb.params["metric"]) and xb.params["metric"].deps() == {"s"}
    W = np.asarray(xb.params["metric_init"][0]["W"][0]).ravel()   # SpExp("s") per-coord weights
    assert np.allclose(W, -1.0, atol=0.15), W


def test_build_and_sample_sparse_vector_funnel():
    """Both blocks become learned (mutual elementwise dependence); ScoreMassAdaptation no-ops
    with no quadratic kinetic and MetricAdaptation drives them; the sampler produces finite
    draws with ``s`` near its N(0, 3^2) marginal."""
    prob = neal_funnel_vector(n=30, scale=3.0)
    model = prob.model
    s = analyze(model, _vector_evidence(model)).build(seed=0)

    assert MetricAdaptation in type(s).__mro__
    assert all(isinstance(k, LearnedDiagonalBlock) for k in s.kinetics)   # s and x both learned
    s.warmup(800)
    s.sample(2000)
    draws = s.get_samples_flat()
    assert draws.shape == (2000, 60) and np.all(np.isfinite(draws))
    assert abs(draws[:, 0].mean()) < 1.0 and abs(draws[:, 0].std() - 3.0) < 1.0   # s_0 ~ N(0, 9)


# --- scale-aware initialisation + the non-finite guard ------------------------ #
#
# Regression tests for the `diamonds` failure (docs/design/09): the KL loss
# f(b) = 1/2 (b + g^2 e^{-b}) is exponentially steep below its optimum and almost exactly linear
# with slope 1/2 above it. Started at M = I on a target whose scores are ~1e5, L-BFGS takes one
# enormous (correctly Armijo-accepted) step into the flat region and cannot crawl back within
# max_iter -- leaving |b| ~ 1e4 instead of ~11, which both produces an unusable metric and, by
# crippling the *constant baseline*, makes every position-dependent candidate look far better
# than it is.

def _flat_scores(scale, d=8, n=600, seed=0):
    """Evidence for a single ``d``-dim block whose scores have variance ``scale`` and NO
    position dependence: coordinates are pure noise, so the ideal metric is the constant
    ``M = scale``."""
    rng = np.random.default_rng(seed)
    g = np.sqrt(scale) * rng.standard_normal((n, d))
    coords = rng.standard_normal((n, d + 1))            # column d is an unrelated 1-d "dep"
    grads = np.column_stack([g, np.zeros(n)])
    return coords, grads


def test_init_params_accepts_a_per_coordinate_target():
    """``init_params(target=array)`` puts the expression exactly at that per-coordinate scale."""
    from mimcs.hmc.metric_expr import Sigmoid
    target = jnp.asarray([1.0, 1e2, 1e4, 1e6])
    for expr in (Exp("v") + Exp(), Exp() * Sigmoid("v") + Exp()):
        p = expr.init_params(4, {"v": 2}, target=target)
        M = np.asarray(expr.evaluate(p, {"v": jnp.zeros(2)}))
        assert np.allclose(M, np.asarray(target), rtol=2e-3), (expr, M)


def test_constant_fit_recovers_a_large_scale():
    """The dependency-free baseline on scores of variance 1e5 must land at ``b ~ log 1e5``,
    not run away to 1e4 (the diamonds failure, which had nothing to depend on either)."""
    from mimcs.factory.regression import fit_metric_expr
    scale = 1e5
    coords, grads = _flat_scores(scale)
    loss, params = fit_metric_expr(Exp(), list(range(8)), {}, coords, grads)
    b = np.asarray(params["b"])
    assert np.allclose(b, np.log(scale), atol=0.3), b
    # and it is at the true optimum: loss == 1/2 sum(b + g^2 e^-b) at b = log mean g^2
    g2 = (grads[:, :8] ** 2).mean(axis=0)
    assert loss < float(0.5 * np.sum(np.log(g2) + 1.0)) + 1e-3


def test_badly_scaled_flat_target_keeps_the_constant_baseline():
    """AIC must still prefer the constant metric when the target has no position dependence,
    even at a score scale of 1e5. Before the scale-aware init the crippled baseline lost to
    every position-dependent candidate by orders of magnitude."""
    from mimcs.factory.regression import select_metric
    coords, grads = _flat_scores(1e5)
    ranked = select_metric(list(range(8)), {"v": [8]}, coords, grads)
    assert ranked[0].expr.deps() == set(), [(r.expr, r.aic) for r in ranked]
    for r in ranked:                       # every fit stays in a sane range
        for leaf in jax.tree_util.tree_leaves(r.params):
            assert np.all(np.abs(np.asarray(leaf)) < 100.0), (r.expr, leaf)


def test_non_finite_metric_is_rejected():
    """``fit_is_usable`` rejects a candidate whose parameters are finite but whose metric
    overflows to ``inf`` -- the shape the diamonds fit had (``exp(1e4)``)."""
    from mimcs.factory.regression import fit_is_usable
    coords, grads = _flat_scores(1e5)
    good = {"W": [], "b": jnp.full((8,), np.log(1e5))}
    bad = {"W": [], "b": jnp.full((8,), 1e4)}            # finite params, exp -> inf
    assert fit_is_usable(Exp(), good, {}, coords, 1.0)
    assert not fit_is_usable(Exp(), bad, {}, coords, 1.0)
    assert not fit_is_usable(Exp(), good, {}, coords, float("inf"))   # non-finite loss


def test_metric_adaptation_survives_a_non_finite_initial_metric():
    """A learned block handed a pathological metric (``exp(1e4) = inf``) must not poison the
    parameters or crash: the guard skips those steps, counts them, and the run completes."""
    from mimcs.factory.spec import BlockSpec
    prob = neal_funnel_blocks(dim=31, scale=3.0)
    model = prob.model
    spec = analyze(model)
    vs, ve = model.coord_block("v")
    xs, xe = model.coord_block("x")
    init = (Exp("v") + Exp()).init_params(xe - xs, {"v": ve - vs})
    init[1]["b"] = jnp.full_like(init[1]["b"], 1e4)      # the baseline term overflows
    spec.blocks = [
        BlockSpec(["v"], [(vs, ve)], "diagonal"),
        BlockSpec(["x"], [(xs, xe)], "learned_metric",
                  params={"metric": Exp("v") + Exp(), "metric_init": init}),
    ]
    spec.terminate = None
    s = spec.build(seed=0)
    s.warmup(50)
    s.sample(50)
    draws = s.get_samples_flat()
    assert draws.shape == (50, 31)
    assert s.metric_nonfinite_count() > 0                # the guard fired ...
    for leaf in jax.tree_util.tree_leaves(s.state.ham_params):
        assert np.all(np.isfinite(np.asarray(leaf)))     # ... and kept the params finite
