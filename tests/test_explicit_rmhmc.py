"""Tests for explicit (block) Riemannian Manifold HMC with given diagonal metrics.

The explicit variant integrates a block-diagonal position-dependent metric with a
*closed-form* (no implicit solve) leapfrog, built from the ordinary `SplittingIntegrator`.
Validated: a constant metric reduces to standard leapfrog (matched to the analytic
Gaussian map); explicit RMHMC samples a Gaussian (constant metric) and Neal's funnel
correctly -- with the metric on the `x` block depending on the `v` block, the geometry
global-metric NUTS could not handle.

Seeds are fixed, so pass/fail is deterministic.
"""

import numpy as np
import jax.numpy as jnp

from mimcs.testing import correlated_gaussian, neal_funnel_blocks, evaluate, explicit_rmhmc
from mimcs.hmc import (
    BlockMetric, ModelPotential, DiagonalBlock, leapfrog,
    init_integrator_state, HamiltonianContext)


def _analytic_leapfrog(precision, inv_mass, eps, n_steps, y0, p0):
    n = precision.shape[0]
    I, Z, Minv = np.eye(n), np.zeros((n, n)), np.diag(inv_mass)
    K = np.block([[I, Z], [-(eps / 2) * precision, I]])
    D = np.block([[I, eps * Minv], [Z, I]])
    z = np.linalg.matrix_power(K @ D @ K, n_steps) @ np.concatenate([y0, p0])
    return z[:n], z[n:]


def test_explicit_block_leapfrog_constant_metric_matches_leapfrog():
    """A constant metric makes each block's flow an ordinary drift, so the unified leapfrog
    over the block list reduces to standard leapfrog; on a Gaussian it matches the analytic map."""
    mean = np.array([1.0, -2.0])
    cov = np.array([[2.0, 1.4], [1.4, 1.5]])
    precision = np.linalg.inv(cov)

    model = correlated_gaussian(mean=mean, cov=cov).model
    potentials = [ModelPotential(model, "log_post")]
    block = DiagonalBlock("x", (0, 2), [], None)   # single block, constant identity metric
    ctx = HamiltonianContext(model.init_chart_hyperparams(), model.init_chart_indices(),
                             {"x": None})
    integ = leapfrog(potentials, [block])

    q0, p0, eps = np.array([0.5, 0.3]), np.array([-0.4, 0.9]), 0.1
    for n_steps in (1, 5, 20):
        st = init_integrator_state(
            potentials, jnp.asarray(q0, jnp.float32), jnp.asarray(p0, jnp.float32), ctx)
        out = integ.integrate(st, eps, n_steps, ctx)
        y_ref, p_ref = _analytic_leapfrog(precision, np.ones(2), eps, n_steps, q0 - mean, p0)
        assert np.allclose(np.asarray(out.q), y_ref + mean, atol=1e-4)
        assert np.allclose(np.asarray(out.p), p_ref, atol=1e-4)


def test_explicit_rmhmc_gaussian_constant_metric(artifacts_dir):
    problem = correlated_gaussian(mean=[1.0, -2.0], cov=[[2.0, 1.4], [1.4, 1.5]])
    report = evaluate(
        problem, {"ermhmc": explicit_rmhmc(n_leapfrog=20, step_size=0.2)},
        n_warmup=1500, n_samples=8000, seed=0,
        out_dir=str(artifacts_dir / "explicit_rmhmc_gaussian"))
    print("\n" + report.summary())
    report.assert_correct()


def test_explicit_rmhmc_funnel_blocks(artifacts_dir):
    """Explicit block RMHMC on the funnel: a diagonal metric on the `x` block depending on
    `v` (`M_x(v) = e^{-v}`), integrated explicitly (no implicit solve). Samples the funnel
    correctly -- the geometry global-metric NUTS could not handle -- reaching neck and
    mouth."""
    problem = neal_funnel_blocks(dim=2, scale=3.0)
    problem.hard = False
    metrics = {"x": BlockMetric(depends_on=("v",), fn=lambda d: jnp.exp(-d["v"]))}
    report = evaluate(
        problem,
        {"ermhmc": explicit_rmhmc(metrics=metrics, n_leapfrog=25, step_size=0.25,
                                  target_accept=0.9)},
        n_warmup=2000, n_samples=15000, seed=0,
        out_dir=str(artifacts_dir / "explicit_rmhmc_funnel"))
    print("\n" + report.summary())
    report.assert_correct()
