"""Tests for relativistic HMC / NUTS.

The relativistic kinetic ``T = sum_i sqrt(m^2 c^4 + c^2 |p_i|^2)`` has a velocity
``c^2 p / T`` bounded by the light speed ``c``, so the integrator cannot shoot off in
light tails / funnel-like geometry. The "particle" structure (what the inner sum spans) is
set by reshaping the flat momentum to ``shape`` and choosing ``inner_axes``.

Relativistic HMC/NUTS is a drop-in: the kinetic is separable, so it uses the ordinary
leapfrog and the existing HMC / NUTS samplers. The showcase is the stiff Rosenbrock banana
(``b=5``), whose tail standard global-metric HMC under-explores.

Seeds are fixed, so pass/fail is deterministic.
"""

import types

import numpy as np
import jax
import jax.numpy as jnp

from mimcs.model import Model, EuclideanParameter
from mimcs.hmc import RelativisticKinetic
from mimcs.testing import (
    TargetProblem, correlated_gaussian, rosenbrock, neal_funnel, evaluate,
    relativistic_hmc, relativistic_nuts)


def _centered_gaussian(mean, cov):
    mean = np.asarray(mean, float)
    cov = np.asarray(cov, float)
    precision = jnp.asarray(np.linalg.inv(cov))
    jmean = jnp.asarray(mean)
    model = Model([EuclideanParameter("x", (mean.shape[0],), centered=True)],
                  {"lp": lambda p: -0.5 * (p["x"] - jmean) @ precision @ (p["x"] - jmean)})
    chol = np.linalg.cholesky(cov)
    return TargetProblem(
        name="centered_gaussian", model=model, dim=mean.shape[0],
        labels=[f"x{i}" for i in range(mean.shape[0])],
        exact_sampler=lambda n, rng: mean + rng.standard_normal((n, mean.shape[0])) @ chol.T,
        mean=mean, cov=cov)


def _state(p):
    return types.SimpleNamespace(p=jnp.asarray(p, jnp.float32))


def test_relativistic_velocity_matches_autodiff():
    """``velocity`` must equal ``d kinetic / dp`` for every particle structure."""
    rng = np.random.default_rng(0)
    for shape, inner in [((4,), ()), ((4,), (0,)), ((2, 3), (1,)), ((2, 3), (0, 1))]:
        k = RelativisticKinetic(shape, inner_axes=inner)
        p = jnp.asarray(rng.standard_normal(k.dim), jnp.float32)
        grad = jax.grad(lambda pp: k.energy(_state(pp), None))(p)
        vel = k.velocity_into(jnp.zeros(k.dim), _state(p), None)
        assert np.allclose(np.asarray(vel), np.asarray(grad), atol=1e-5)


def test_relativistic_velocity_bounded_by_c():
    """The velocity magnitude is capped by the light speed, however large the momentum."""
    k = RelativisticKinetic((3,), inner_axes=(0,), light_speed=1.5)
    v = k.velocity_into(jnp.zeros(3), _state(np.array([1e6, -2e6, 5e5])), None)
    assert np.linalg.norm(np.asarray(v)) < 1.5 + 1e-3


def test_relativistic_momentum_matches_target():
    """The sampled momentum reproduces ``exp(-T)``: for a 1-D particle, its CDF must match
    that of ``exp(-sqrt(1 + p^2))`` (a manual Kolmogorov--Smirnov check)."""
    k = RelativisticKinetic((1,), inner_axes=())
    key_d, key_r = jax.random.split(jax.random.PRNGKey(0))
    n = 60000
    dirs = jax.random.normal(key_d, (n, 1))
    rads = jax.random.uniform(key_r, (n, 1))
    samp = jax.vmap(lambda d, r: k.sample_into(jnp.zeros(1),
        types.SimpleNamespace(T_mom_direction=d, T_mom_radius=r), None, None))(dirs, rads)
    samp = np.sort(np.asarray(samp).ravel())

    grid = np.linspace(-40, 40, 20001)
    dens = np.exp(-np.sqrt(1.0 + grid**2))
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(grid))])
    cdf /= cdf[-1]
    target = np.interp(samp, grid, cdf)
    empirical = np.arange(1, n + 1) / n
    assert np.abs(target - empirical).max() < 0.02


def test_relativistic_hmc_gaussian():
    """Relativistic HMC samples a Gaussian for both extreme particle structures."""
    problem = correlated_gaussian(mean=[1.0, -2.0], cov=[[2.0, 1.4], [1.4, 1.5]])
    for inner in [(), (0,)]:
        report = evaluate(
            problem, {"rel": relativistic_hmc(shape=(2,), inner_axes=inner,
                                              n_leapfrog=20, step_size=0.4)},
            n_warmup=2000, n_samples=8000, seed=0)
        print("\n" + report.summary())
        report.assert_correct()


def test_relativistic_hmc_stiff_banana(artifacts_dir):
    """Relativistic HMC samples the stiff Rosenbrock banana (b=5), whose funnel-like tail
    standard global-metric HMC under-explores (the bounded velocity helps)."""
    problem = rosenbrock(a=1.0, b=5.0)
    report = evaluate(
        problem, {"rel": relativistic_hmc(shape=(2,), n_leapfrog=25, step_size=0.25)},
        n_warmup=2000, n_samples=10000, seed=0,
        out_dir=str(artifacts_dir / "relativistic_hmc_banana"))
    print("\n" + report.summary())
    report.assert_correct()


def test_relativistic_nuts_stiff_banana(artifacts_dir):
    """Relativistic NUTS samples the stiff banana (b=5), with no trajectory-length tuning."""
    problem = rosenbrock(a=1.0, b=5.0)
    report = evaluate(
        problem, {"rel": relativistic_nuts(shape=(2,), step_size=0.3)},
        n_warmup=2000, n_samples=10000, seed=0,
        out_dir=str(artifacts_dir / "relativistic_nuts_banana"))
    print("\n" + report.summary())
    report.assert_correct()


def test_relativistic_explores_funnel_neck(artifacts_dir):
    """Diagnostic: on Neal's funnel (too hard for an exact global-metric fit, like standard
    HMC) relativistic NUTS still reaches deep into the neck. Distributional checks are
    skipped (``hard``); the run and its plots document the partial capability."""
    problem = neal_funnel(dim=2, scale=3.0)
    report = evaluate(
        problem, {"rel": relativistic_nuts(shape=(2,), step_size=0.2)},
        n_warmup=3000, n_samples=10000, seed=0,
        out_dir=str(artifacts_dir / "relativistic_funnel"))
    print("\n" + report.summary())
    v = report.outputs["rel"].samples[:, 0]
    assert v.min() < -4.0 and v.max() > 5.0, "did not explore the funnel neck/mouth"


# --- per-particle mass adaptation + centering --------------------------------- #


def test_relativistic_momentum_tracks_adapting_mass():
    """The (u, m) inverse-CDF momentum sampler reproduces ``exp(-T)`` across masses (a 1-D
    particle's momentum has CDF that of ``exp(-sqrt(m^2 + p^2))``)."""
    k = RelativisticKinetic((1,), inner_axes=())
    for m in (0.2, 1.0, 5.0):
        u = jax.random.uniform(jax.random.PRNGKey(0), (40000,))
        z = jax.random.normal(jax.random.PRNGKey(1), (40000,))
        samp = np.sort(np.asarray(k._sample_radius(u, jnp.full((40000,), m)))
                       * np.sign(np.asarray(z)))
        grid = np.linspace(-80, 80, 40001)
        dens = np.exp(-np.sqrt(m**2 + grid**2))
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(grid))])
        cdf /= cdf[-1]
        ks = np.abs(np.interp(samp, grid, cdf) - np.arange(1, len(samp) + 1) / len(samp)).max()
        assert ks < 0.02, f"momentum CDF off at m={m}: KS {ks}"


def test_relativistic_mass_adapts_to_score_covariance():
    """The score-covariance mass adaptation drives ``m_i -> E[|g_i|^2] / d_i``. On a Gaussian
    with 1-D particles that is ``diag(precision)``; for one d-D particle it is
    ``E[|g|^2] / d = tr(precision) / d``."""
    cov = np.array([[2.0, 1.4], [1.4, 1.5]])
    precision = np.linalg.inv(cov)
    prob = correlated_gaussian(mean=[1.0, -2.0], cov=cov)

    s = relativistic_hmc(shape=(2,), inner_axes=(), mass_adapt=True,
                         n_leapfrog=20, step_size=0.4)(prob.model, seed=0)
    s.warmup(4000)
    ratio = np.asarray(s.state.ham_params["T"]) / np.diag(precision)
    assert np.all((ratio > 0.6) & (ratio < 1.5)), \
        f"1-D particle masses {np.asarray(s.state.ham_params['T'])} off"

    s2 = relativistic_hmc(shape=(2,), inner_axes=(0,), mass_adapt=True,
                          n_leapfrog=20, step_size=0.4)(prob.model, seed=0)
    s2.warmup(4000)
    target = np.trace(precision) / 2.0
    assert 0.6 < float(np.asarray(s2.state.ham_params["T"])[0]) / target < 1.5


def test_relativistic_mass_adapt_samples_gaussian():
    """Relativistic HMC with adapted masses samples a Gaussian (both particle structures)."""
    problem = correlated_gaussian(mean=[1.0, -2.0], cov=[[2.0, 1.4], [1.4, 1.5]])
    for inner in [(), (0,)]:
        report = evaluate(
            problem, {"rel": relativistic_hmc(shape=(2,), inner_axes=inner, mass_adapt=True,
                                              n_leapfrog=20, step_size=0.4)},
            n_warmup=4000, n_samples=8000, seed=0)
        print("\n" + report.summary())
        report.assert_correct()


def test_relativistic_centering_mass_strategy(artifacts_dir):
    """The full recommended strategy: centering reparametrization + fixed light speed +
    score-adapted per-particle mass. Samples a large-mean/scale Gaussian with HMC and NUTS."""
    prob = _centered_gaussian([8.0, -5.0], [[4.0, 1.5], [1.5, 2.0]])
    for name, builder in [
        ("rel_hmc", relativistic_hmc(shape=(2,), inner_axes=(), center=True, mass_adapt=True,
                                     n_leapfrog=20, step_size=0.4, init=[8.0, -5.0])),
        ("rel_nuts", relativistic_nuts(shape=(2,), inner_axes=(), center=True, mass_adapt=True,
                                       step_size=0.4, init=[8.0, -5.0])),
    ]:
        report = evaluate(prob, {name: builder}, n_warmup=4000, n_samples=8000, seed=0,
                          out_dir=str(artifacts_dir / f"relativistic_{name}_centered"))
        print("\n" + report.summary())
        report.assert_correct()
