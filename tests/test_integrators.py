"""Tests for the leapfrog integrator against exact formulas on a Gaussian target.

For a Gaussian target the Hamiltonian is quadratic, so both the true Hamiltonian flow
and the leapfrog map are *linear* maps of phase space. We test the integrator three
ways:

1. **Exact match to the analytic leapfrog map.** Each leapfrog step on a quadratic
   Hamiltonian is the linear map ``K(eps/2) D(eps) K(eps/2)`` (half-kick, drift,
   half-kick); ``L`` steps is its ``L``-th power. The integrator must reproduce this to
   floating-point precision --- a direct check of the update equations, ordering, and
   gradient caching.
2. **Reversibility.** Integrating forward, negating momentum, and integrating again
   returns to the start (with momentum negated): leapfrog is time-reversible.
3. **Second-order accuracy.** Versus the exact flow ``exp(t L)``, the global error is
   ``O(eps^2)``: halving ``eps`` should cut the error by ~4.
"""

import numpy as np
import jax.numpy as jnp
from scipy.linalg import expm

from mimcs.testing import correlated_gaussian
from mimcs.hmc import (
    ModelPotential, DiagonalQuadraticKinetic, leapfrog, init_integrator_state,
    HamiltonianContext, total_energy)


# --- analytic references ---------------------------------------------------- #

def _leapfrog_step_matrix(precision, inv_mass_diag, eps):
    """Phase-space matrix of one leapfrog step on a Gaussian (shifted to mean 0)."""
    n = precision.shape[0]
    I, Z = np.eye(n), np.zeros((n, n))
    Minv = np.diag(inv_mass_diag)
    half_kick = np.block([[I, Z], [-(eps / 2) * precision, I]])   # p -= (eps/2) A y
    drift = np.block([[I, eps * Minv], [Z, I]])                   # y += eps Minv p
    return half_kick @ drift @ half_kick


def _analytic_leapfrog(precision, inv_mass_diag, eps, n_steps, y0, p0):
    step = _leapfrog_step_matrix(precision, inv_mass_diag, eps)
    full = np.linalg.matrix_power(step, n_steps)
    z = full @ np.concatenate([y0, p0])
    n = len(y0)
    return z[:n], z[n:]


def _true_flow(precision, inv_mass_diag, t, y0, p0):
    """Exact Hamiltonian flow exp(t L) for the quadratic system (reference for order)."""
    n = precision.shape[0]
    Z = np.zeros((n, n))
    L = np.block([[Z, np.diag(inv_mass_diag)], [-precision, Z]])
    z = expm(t * L) @ np.concatenate([y0, p0])
    return z[:n], z[n:]


# --- fixtures --------------------------------------------------------------- #

def _setup(mean, cov, inv_mass_diag):
    problem = correlated_gaussian(mean=mean, cov=cov)
    model = problem.model
    potentials = [ModelPotential(model, "log_post")]
    kinetic = DiagonalQuadraticKinetic("T")
    ctx = HamiltonianContext(
        chart_hyperparams=model.init_chart_hyperparams(),
        chart_indices=model.init_chart_indices(),
        ham_params={"T": jnp.asarray(inv_mass_diag, jnp.float32)},
    )
    precision = np.linalg.inv(np.asarray(cov, float))
    return model, potentials, kinetic, ctx, precision


# --- tests ------------------------------------------------------------------ #

def test_leapfrog_matches_analytic_linear_map():
    mean = np.array([1.0, -2.0])
    cov = np.array([[2.0, 1.4], [1.4, 1.5]])
    inv_mass = np.array([1.3, 0.7])
    model, potentials, kinetic, ctx, precision = _setup(mean, cov, inv_mass)
    integ = leapfrog(potentials, kinetic)

    q0 = np.array([0.5, 0.3])
    p0 = np.array([-0.4, 0.9])
    eps = 0.1

    for n_steps in (1, 5, 20, 73):
        state = init_integrator_state(
            potentials, jnp.asarray(q0, jnp.float32), jnp.asarray(p0, jnp.float32), ctx)
        out = integ.integrate(state, eps, n_steps, ctx)

        y_ref, p_ref = _analytic_leapfrog(precision, inv_mass, eps, n_steps,
                                          q0 - mean, p0)
        q_ref = y_ref + mean

        assert np.allclose(np.asarray(out.q), q_ref, atol=1e-4, rtol=1e-4), \
            f"q mismatch at {n_steps} steps: {np.asarray(out.q)} vs {q_ref}"
        assert np.allclose(np.asarray(out.p), p_ref, atol=1e-4, rtol=1e-4), \
            f"p mismatch at {n_steps} steps: {np.asarray(out.p)} vs {p_ref}"


def test_leapfrog_is_reversible():
    mean = np.array([0.3, -1.0])
    cov = np.array([[1.0, -0.5], [-0.5, 2.0]])
    inv_mass = np.array([1.0, 0.5])
    model, potentials, kinetic, ctx, _ = _setup(mean, cov, inv_mass)
    integ = leapfrog(potentials, kinetic)

    q0 = jnp.asarray([0.7, -0.2], jnp.float32)
    p0 = jnp.asarray([0.1, 0.5], jnp.float32)
    eps, n_steps = 0.15, 30

    fwd = integ.integrate(init_integrator_state(potentials, q0, p0, ctx),
                          eps, n_steps, ctx)
    # negate momentum and integrate again -> should return to (q0, -p0)
    back = integ.integrate(init_integrator_state(potentials, fwd.q, -fwd.p, ctx),
                           eps, n_steps, ctx)

    assert np.allclose(np.asarray(back.q), np.asarray(q0), atol=1e-4)
    assert np.allclose(np.asarray(back.p), -np.asarray(p0), atol=1e-4)


def test_leapfrog_energy_is_conserved_without_drift():
    mean = np.zeros(2)
    cov = np.array([[1.0, 0.3], [0.3, 1.0]])
    inv_mass = np.array([1.0, 1.0])
    model, potentials, kinetic, ctx, _ = _setup(mean, cov, inv_mass)
    integ = leapfrog(potentials, kinetic)

    q0 = jnp.asarray([1.5, -0.5], jnp.float32)
    p0 = jnp.asarray([0.2, 0.8], jnp.float32)
    state = init_integrator_state(potentials, q0, p0, ctx)
    H0 = float(total_energy(state, potentials, kinetic, ctx))

    # over a long trajectory the energy oscillates but must not drift
    energies = []
    for _ in range(200):
        state = integ.step(state, 0.1, ctx)
        energies.append(float(total_energy(state, potentials, kinetic, ctx)))
    energies = np.array(energies)

    assert np.max(np.abs(energies - H0)) < 0.05 * abs(H0) + 0.05


def test_leapfrog_leaves_log_weight_zero():
    """Standard leapfrog is deterministic and volume-preserving, so it must not touch
    the base log_weight field (it rides through _replace untouched)."""
    mean = np.zeros(2)
    cov = np.array([[1.0, 0.2], [0.2, 1.0]])
    inv_mass = np.array([1.0, 1.0])
    model, potentials, kinetic, ctx, _ = _setup(mean, cov, inv_mass)
    integ = leapfrog(potentials, kinetic)

    state = init_integrator_state(
        potentials, jnp.asarray([0.5, -0.5], jnp.float32),
        jnp.asarray([0.3, 0.1], jnp.float32), ctx)
    assert float(state.log_weight) == 0.0

    out = integ.integrate(state, 0.1, 40, ctx)
    assert float(out.log_weight) == 0.0


def test_leapfrog_is_second_order_accurate():
    mean = np.zeros(2)
    cov = np.array([[1.0, 0.2], [0.2, 1.5]])
    inv_mass = np.array([1.2, 0.8])
    model, potentials, kinetic, ctx, precision = _setup(mean, cov, inv_mass)
    integ = leapfrog(potentials, kinetic)

    q0 = np.array([0.6, -0.3])
    p0 = np.array([0.4, 0.5])
    total_time = 1.0

    def error_at(eps):
        n_steps = int(round(total_time / eps))
        state = init_integrator_state(
            potentials, jnp.asarray(q0, jnp.float32), jnp.asarray(p0, jnp.float32), ctx)
        out = integ.integrate(state, eps, n_steps, ctx)
        y_true, p_true = _true_flow(precision, inv_mass, total_time, q0, p0)
        q_true = y_true + mean
        return np.linalg.norm(np.concatenate(
            [np.asarray(out.q) - q_true, np.asarray(out.p) - p_true]))

    err_coarse = error_at(0.05)
    err_fine = error_at(0.025)
    ratio = err_coarse / err_fine

    # halving the step size should reduce the global error by ~4 (second order)
    assert 3.0 < ratio < 5.0, f"order ratio {ratio:.2f} not ~4 (errs {err_coarse:.2e}, {err_fine:.2e})"
