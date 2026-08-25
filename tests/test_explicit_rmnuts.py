"""Tests for explicit (block) Riemannian NUTS.

Explicit RM-NUTS is a drop-in composition enabled by the modular design: NUTS with the
block-diagonal kinetic (whose ``velocity = M_i(q_{-i})^{-1} p_i`` drives the generalized
U-turn) and the closed-form block leapfrog as the leaf integrator -- no NUTS code changes,
and no implicit solve. It combines NUTS's automatic trajectory length with the explicit
block metric, given *or* learned.

A constant metric reduces it to ordinary NUTS (sanity); with a good given metric, and with
a fully learned metric, it samples Neal's funnel correctly -- no trajectory-length tuning
and, for the learned case, no metric supplied at all.

Seeds are fixed, so pass/fail is deterministic.
"""

import jax.numpy as jnp

from mimcs.testing import neal_funnel_blocks, correlated_gaussian, evaluate, explicit_rmnuts
from mimcs.hmc import BlockMetric, Exp


def test_explicit_rmnuts_gaussian_constant_metric(artifacts_dir):
    """All-constant (identity) block metric reduces explicit RM-NUTS to ordinary NUTS."""
    problem = correlated_gaussian(mean=[1.0, -2.0], cov=[[2.0, 1.4], [1.4, 1.5]])
    report = evaluate(
        problem, {"ermnuts": explicit_rmnuts(max_tree_depth=8, step_size=0.4)},
        n_warmup=1000, n_samples=6000, seed=0,
        out_dir=str(artifacts_dir / "explicit_rmnuts_gaussian"))
    print("\n" + report.summary())
    report.assert_correct()


def test_explicit_rmnuts_funnel_given_metric(artifacts_dir):
    """RM-NUTS with the given natural metric (``M_x = e^{-v}``) samples the funnel
    correctly with no trajectory-length tuning."""
    problem = neal_funnel_blocks(dim=2, scale=3.0)
    problem.hard = False
    report = evaluate(
        problem,
        {"ermnuts": explicit_rmnuts(
            metrics={"x": BlockMetric(depends_on=("v",), fn=lambda d: jnp.exp(-d["v"]))},
            step_size=0.5, target_accept=0.8)},
        n_warmup=1000, n_samples=10000, seed=0,
        out_dir=str(artifacts_dir / "explicit_rmnuts_funnel_given"))
    print("\n" + report.summary())
    report.assert_correct()


def test_explicit_rmnuts_funnel_learned_metric(artifacts_dir):
    """Fully automatic: RM-NUTS with a *learned* metric (only the expression ``Exp("v")``,
    no metric given, no trajectory tuning) samples the funnel correctly."""
    problem = neal_funnel_blocks(dim=2, scale=3.0)
    problem.hard = False
    report = evaluate(
        problem,
        {"ermnuts": explicit_rmnuts(
            metrics={"x": Exp("v")},
            step_size=0.5, target_accept=0.8)},
        n_warmup=4000, n_samples=10000, seed=0,
        out_dir=str(artifacts_dir / "explicit_rmnuts_funnel_learned"))
    print("\n" + report.summary())
    report.assert_correct()
