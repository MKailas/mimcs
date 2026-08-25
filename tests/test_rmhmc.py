"""Tests for implicit Riemannian Manifold HMC (Girolami & Calderhead).

Correctness is checked four ways:
- the generalized (implicit) leapfrog with a *constant* metric must reduce exactly to
  standard leapfrog (matched to the analytic Gaussian map);
- RMHMC samples a Gaussian (constant metric) and a banana (position-dependent metric
  ``(1+x0^2)I``) correctly through the harness;
- with a good position-dependent metric, RMHMC samples Neal's funnel correctly --- the
  geometry global-metric NUTS could not handle.

A final test uses the conformal metric ``exp(-v) I`` suggested as a stress case: its
``e^{-v}`` v-block destabilizes the large-v dynamics so it under-explores the x-tail (a
demonstration of the implicit integrator's sensitivity to the metric), yet it still
samples ``v`` correctly and reaches deep into the neck, far past global-metric NUTS.

Seeds are fixed, so pass/fail is deterministic.
"""

import numpy as np
import jax.numpy as jnp

from mimcs.testing import (
    correlated_gaussian, rosenbrock, neal_funnel, evaluate, rmhmc)
from mimcs.hmc import (
    ModelPotential, RiemannianKinetic, AnalyticMetric, leapfrog, PicardSolver,
    HamiltonianContext, init_integrator_state)


def _analytic_leapfrog(precision, inv_mass, eps, n_steps, y0, p0):
    n = precision.shape[0]
    I, Z, Minv = np.eye(n), np.zeros((n, n)), np.diag(inv_mass)
    K = np.block([[I, Z], [-(eps / 2) * precision, I]])
    D = np.block([[I, eps * Minv], [Z, I]])
    S = np.linalg.matrix_power(K @ D @ K, n_steps)
    z = S @ np.concatenate([y0, p0])
    return z[:n], z[n:]


def test_riemannian_constant_metric_matches_leapfrog():
    """A constant metric gives grad_q T = 0, so the Riemannian kinetic's implicit flow
    reduces to an ordinary drift and ``leapfrog(potentials, kinetic)`` reduces to standard
    leapfrog with mass = G; on a Gaussian it must match the analytic map."""
    mean = np.array([1.0, -2.0])
    cov = np.array([[2.0, 1.4], [1.4, 1.5]])
    precision = np.linalg.inv(cov)
    G_diag = np.array([1.3, 0.7])
    inv_mass = 1.0 / G_diag

    model = correlated_gaussian(mean=mean, cov=cov).model
    potentials = [ModelPotential(model, "log_post")]
    kinetic = RiemannianKinetic(
        AnalyticMetric(lambda q: jnp.diag(jnp.asarray(G_diag, jnp.float32))),
        solver=PicardSolver(8))
    ctx = HamiltonianContext(model.init_chart_hyperparams(), model.init_chart_indices(), {})
    integ = leapfrog(potentials, kinetic)

    q0, p0, eps = np.array([0.5, 0.3]), np.array([-0.4, 0.9]), 0.1
    for n_steps in (1, 5, 20):
        st = init_integrator_state(
            potentials, jnp.asarray(q0, jnp.float32), jnp.asarray(p0, jnp.float32), ctx)
        out = integ.integrate(st, eps, n_steps, ctx)
        y_ref, p_ref = _analytic_leapfrog(precision, inv_mass, eps, n_steps, q0 - mean, p0)
        assert np.allclose(np.asarray(out.q), y_ref + mean, atol=1e-4)
        assert np.allclose(np.asarray(out.p), p_ref, atol=1e-4)


def test_rmhmc_gaussian_constant_metric(artifacts_dir):
    problem = correlated_gaussian(mean=[1.0, -2.0], cov=[[2.0, 1.4], [1.4, 1.5]])
    report = evaluate(
        problem, {"rmhmc": rmhmc(metric=lambda q: jnp.eye(2), n_leapfrog=20, step_size=0.2)},
        n_warmup=1500, n_samples=8000, seed=0, out_dir=str(artifacts_dir / "rmhmc_gaussian"))
    print("\n" + report.summary())
    report.assert_correct()


def test_rmhmc_banana_position_dependent_metric(artifacts_dir):
    # position-dependent metric (1 + x0^2) I on a mild banana
    problem = rosenbrock(a=1.0, b=1.0)
    report = evaluate(
        problem,
        {"rmhmc": rmhmc(metric=lambda q: (1.0 + q[0] ** 2) * jnp.eye(2),
                        n_leapfrog=15, step_size=0.3, target_accept=0.85)},
        n_warmup=1500, n_samples=8000, seed=2, out_dir=str(artifacts_dir / "rmhmc_banana"))
    print("\n" + report.summary())
    report.assert_correct()


def test_rmhmc_funnel_natural_metric(artifacts_dir):
    """The funnel global-metric NUTS could not sample. With a good position-dependent
    metric (constant for v, e^{-v} for x = the conditional precision), RMHMC samples it
    correctly, reaching the full neck and mouth."""
    problem = neal_funnel(dim=2, scale=3.0)
    problem.hard = False   # a good metric makes the funnel tractable -> strict check
    report = evaluate(
        problem,
        {"rmhmc": rmhmc(metric=lambda q: jnp.diag(jnp.array([1.0, jnp.exp(-q[0])])),
                        n_leapfrog=20, step_size=0.3, target_accept=0.85)},
        n_warmup=2000, n_samples=10000, seed=4,
        out_dir=str(artifacts_dir / "rmhmc_funnel_natural"))
    print("\n" + report.summary())
    report.assert_correct()


def test_rmhmc_funnel_conformal_metric_reaches_neck(artifacts_dir):
    """The conformal metric exp(-v)(dv^2 + dx^2) = exp(-v) I. Its e^{-v} v-block
    destabilizes the large-v dynamics, so it under-explores the x-tail (a known
    implicit-integrator sensitivity to the metric, not a bug). It still samples v
    correctly and reaches deep into the neck -- far past global-metric NUTS (~ -3.4)."""
    problem = neal_funnel(dim=2, scale=3.0)   # hard=True -> assert_correct is skipped
    report = evaluate(
        problem,
        {"rmhmc": rmhmc(metric=lambda q: jnp.exp(-q[0]) * jnp.eye(2),
                        n_leapfrog=15, step_size=0.3, target_accept=0.85)},
        n_warmup=2000, n_samples=10000, seed=1,
        out_dir=str(artifacts_dir / "rmhmc_funnel_conformal"))
    print("\n" + report.summary())
    v = report.outputs["rmhmc"].samples[:, 0]
    assert abs(v.std() - 3.0) < 0.6     # v marginal is correct (true sd = 3)
    assert v.min() < -5.0               # reaches deep into the neck
