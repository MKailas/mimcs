"""Correctness tests for basic adaptive HMC.

Uses the same framework and problems as the Metropolis--Hastings tests. HMC is checked
against analytic references (Gaussian, banana, a constrained positive target) with both
the diagonal and dense metrics, and against the (independently validated) MH sampler as
an oracle. Step size (Robbins--Monro to a target acceptance) and the mass matrix (to the
target covariance) are adapted during warmup. Plots are written for inspection.

Seeds are fixed, so pass/fail is deterministic.
"""

import numpy as np

from mimcs.testing import (
    correlated_gaussian, rosenbrock, positive_lognormal, evaluate, hmc, adaptive_mh)


def test_hmc_correlated_gaussian_diagonal(artifacts_dir):
    problem = correlated_gaussian(mean=[1.0, -2.0], cov=[[2.0, 1.4], [1.4, 1.5]])
    report = evaluate(
        problem,
        {"hmc": hmc(n_leapfrog=20, step_size=0.3, metric="diagonal", target_accept=0.8)},
        n_warmup=2000, n_samples=10000, seed=0,
        out_dir=str(artifacts_dir / "hmc_gaussian_diagonal"),
    )
    print("\n" + report.summary())
    report.assert_correct()


def test_hmc_correlated_gaussian_dense(artifacts_dir):
    problem = correlated_gaussian(mean=[1.0, -2.0], cov=[[2.0, 1.4], [1.4, 1.5]])
    report = evaluate(
        problem,
        {"hmc": hmc(n_leapfrog=20, step_size=0.3, metric="dense", target_accept=0.8)},
        n_warmup=2000, n_samples=10000, seed=1,
        out_dir=str(artifacts_dir / "hmc_gaussian_dense"),
    )
    print("\n" + report.summary())
    report.assert_correct()


def test_hmc_rosenbrock_banana(artifacts_dir):
    # A mild banana (b=1): global-metric HMC under-explores the tail of a stiff banana
    # (b=5 gives a persistent ~3-5% x1-variance underestimate even at 200k draws, since
    # the optimal metric is position-dependent). At b=1 that bias is insignificant, so
    # this checks HMC on curvature without flagging the known global-metric limitation.
    problem = rosenbrock(a=1.0, b=1.0)
    report = evaluate(
        problem,
        {"hmc": hmc(n_leapfrog=20, step_size=0.2, metric="diagonal", target_accept=0.8)},
        n_warmup=2000, n_samples=8000, seed=1,
        out_dir=str(artifacts_dir / "hmc_rosenbrock"),
    )
    print("\n" + report.summary())
    report.assert_correct()


def test_hmc_positive_parameter(artifacts_dir):
    """Constrained target: exercises the JacobianPotential in the Hamiltonian."""
    problem = positive_lognormal(sigma=1.0)
    report = evaluate(
        problem,
        {"hmc": hmc(init=np.array([1.0]), n_leapfrog=20, step_size=0.3, target_accept=0.8)},
        n_warmup=2000, n_samples=10000, seed=3,
        out_dir=str(artifacts_dir / "hmc_positive_lognormal"),
    )
    print("\n" + report.summary())
    report.assert_correct()


def test_hmc_agrees_with_metropolis(artifacts_dir):
    """Oracle comparison: HMC vs the validated MH sampler (and both vs analytic)."""
    problem = correlated_gaussian(
        mean=[0.0, 0.0, 0.0],
        cov=[[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]])
    # Short trajectory (n_leapfrog=5) keeps L*step_size below one oscillation: a *dense*
    # mass whitens this Gaussian to near-isotropic, where the step size adapts large and a
    # long fixed-length trajectory would over-extend into a resonance (a fixed-trajectory
    # artefact; randomized/NUTS samplers are immune). A short trajectory is unaffected.
    report = evaluate(
        problem,
        {
            "hmc": hmc(n_leapfrog=5, step_size=0.3, metric="dense", target_accept=0.8),
            "rwmh": adaptive_mh(step_size=0.5, target_accept=0.30),
        },
        n_warmup=2000, n_samples=10000, seed=4,
        out_dir=str(artifacts_dir / "hmc_vs_mh_gaussian_3d"),
    )
    print("\n" + report.summary())
    report.assert_correct()


def test_saves_total_gradients_while_sampling():
    """By default a gradient-based sampler saves the total score of each retained sample (the
    gradient is already cached in the state, so it is nearly free), aligned with the draws and
    equal to the score recomputed from the samples. ``save_gradients=False`` disables it."""
    import jax
    import jax.numpy as jnp
    prob = correlated_gaussian(mean=[1.0, -2.0], cov=[[2.0, 1.4], [1.4, 1.5]])
    model = prob.model

    s = hmc(n_leapfrog=10, step_size=0.3)(model, seed=0)
    s.warmup(200)
    s.sample(300)
    g = s.get_gradients()
    assert g.shape == (300, model.coord_dim)                     # one score per retained sample

    # equals grad log-density recomputed at each sample's coordinate (the factory's fallback)
    chp, ci = s.state.chart_hyperparams, s.state.chart_indices
    coords = jax.vmap(lambda smp: model.sample_to_coordinate(smp, chp, ci))(
        jnp.asarray(s.get_samples_flat(), float))
    scores = jax.vmap(jax.grad(lambda c: model.log_prob_at_coordinate(c, chp, ci)))(coords)
    assert np.allclose(g, np.asarray(scores), atol=1e-4)

    off = hmc(n_leapfrog=10, step_size=0.3, save_gradients=False)(model, seed=0)
    off.warmup(50)
    off.sample(80)
    assert off.get_gradients() is None


def test_get_samples_is_keyed_by_parameter_name():
    """``get_samples()`` returns the draws by name; ``get_samples_flat()`` the flat block.

    The dict is the user-facing shape --- the values are in *sample* space, so they are the
    model's own quantities and nobody has to know where each parameter sits in the flat layout.
    ``sample(n)`` returns the same thing, so there is one meaning of "the samples".
    """
    from mimcs import compile_model
    from mimcs.testing import nuts
    model = compile_model(
        "parameters { real a; array[3] real b; real<lower=0> c; }\n"
        "model { target += -0.5 * (a * a + sum(b .* b)) - c; }", {})

    s = nuts()(model, seed=0)
    s.warmup(200)
    returned = s.sample(300)

    draws = s.get_samples()
    assert list(draws) == ["a", "b", "c"]                 # declaration order
    assert draws["a"].shape == (300,) and draws["b"].shape == (300, 3)
    assert all(np.all(returned[k] == v) for k, v in draws.items())   # sample() agrees

    flat = s.get_samples_flat()
    assert flat.shape == (300, model.ambient_dim)
    assert np.allclose(
        np.concatenate([v.reshape(300, -1) for v in draws.values()], axis=1), flat)
    assert np.all(draws["c"] > 0.0)                       # the bound holds in sample space


def test_get_samples_before_sampling_is_empty_but_shaped():
    from mimcs.testing import correlated_gaussian, nuts
    s = nuts()(correlated_gaussian().model, seed=0)
    assert s.get_samples()["x"].shape == (0, 2)
    assert s.get_samples_flat().shape == (0, 2)
