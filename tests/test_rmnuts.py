"""Tests for Riemannian Manifold NUTS (NUTS over a position-dependent metric).

RM-NUTS is a drop-in combination enabled by the modular design: NUTS with the
Riemannian kinetic (whose ``velocity = G(q)^{-1} p`` drives the generalized U-turn) and
the generalized (implicit) leapfrog as the leaf integrator -- no NUTS code changes.

A constant metric reduces it to ordinary NUTS (sanity), and with a good
position-dependent metric it samples Neal's funnel correctly *without any
trajectory-length tuning* -- the geometry global-metric NUTS could not handle.

(The conformal metric ``exp(-v) I`` is a known stress case: it makes the generalized
leapfrog diverge on most trajectories -- which NUTS's divergence diagnostic exposes --
so it is not tested for distributional correctness here. See doc 07.)

Seeds are fixed, so pass/fail is deterministic.
"""

import jax.numpy as jnp

from mimcs.testing import correlated_gaussian, neal_funnel, evaluate, rmnuts


def test_rmnuts_gaussian_constant_metric(artifacts_dir):
    problem = correlated_gaussian(mean=[1.0, -2.0], cov=[[2.0, 1.4], [1.4, 1.5]])
    report = evaluate(
        problem, {"rmnuts": rmnuts(metric=lambda q: jnp.eye(2), max_tree_depth=8, step_size=0.4)},
        n_warmup=1000, n_samples=6000, seed=0, out_dir=str(artifacts_dir / "rmnuts_gaussian"))
    print("\n" + report.summary())
    report.assert_correct()


def test_rmnuts_funnel_natural_metric(artifacts_dir):
    """RM-NUTS with a good position-dependent metric (constant v-block, e^{-v} x-block)
    samples the funnel correctly with no trajectory-length tuning."""
    problem = neal_funnel(dim=2, scale=3.0)
    problem.hard = False
    report = evaluate(
        problem,
        {"rmnuts": rmnuts(metric=lambda q: jnp.diag(jnp.array([1.0, jnp.exp(-q[0])])),
                          max_tree_depth=9, step_size=0.4, target_accept=0.8)},
        n_warmup=1500, n_samples=6000, seed=2,
        out_dir=str(artifacts_dir / "rmnuts_funnel_natural"))
    print("\n" + report.summary())
    report.assert_correct()
