"""Correctness tests for HMC with randomized integration time.

Each iteration draws the number of leapfrog steps uniformly from {T, ..., 2T}. This is
checked for correctness on the usual problems (against analytic references and against
fixed-length HMC as an oracle), and is shown to fix the periodic-orbit resonance that
cripples fixed-length HMC on a well-conditioned (dense-metric) target.

Seeds are fixed, so pass/fail is deterministic.
"""

import numpy as np

from mimcs.testing import (
    correlated_gaussian, rosenbrock, positive_lognormal,
    evaluate, hmc, randomized_hmc, draw_samples, ess)


def test_randomized_hmc_gaussian_dense(artifacts_dir):
    problem = correlated_gaussian(mean=[1.0, -2.0], cov=[[2.0, 1.4], [1.4, 1.5]])
    report = evaluate(
        problem,
        {"rhmc": randomized_hmc(n_leapfrog=20, step_size=0.3, metric="dense",
                                target_accept=0.8)},
        n_warmup=2000, n_samples=10000, seed=0,
        out_dir=str(artifacts_dir / "randomized_hmc_gaussian_dense"),
    )
    print("\n" + report.summary())
    report.assert_correct()


def test_randomized_hmc_banana(artifacts_dir):
    # A mild banana (b=1). HMC with a *global* metric systematically under-explores the
    # tail of a stiff banana (e.g. b=5: x1 std ~3-5% low even at 200k draws, because the
    # optimal metric is position-dependent -- a motivation for Riemannian HMC). At b=1
    # the bias is small enough to be insignificant here, so this tests HMC on curvature
    # without flagging that known limitation.
    problem = rosenbrock(a=1.0, b=1.0)
    report = evaluate(
        problem,
        {"rhmc": randomized_hmc(n_leapfrog=20, step_size=0.2, target_accept=0.8)},
        n_warmup=2000, n_samples=8000, seed=3,
        out_dir=str(artifacts_dir / "randomized_hmc_rosenbrock"),
    )
    print("\n" + report.summary())
    report.assert_correct()


def test_randomized_hmc_positive_parameter(artifacts_dir):
    problem = positive_lognormal(sigma=1.0)
    report = evaluate(
        problem,
        {"rhmc": randomized_hmc(init=np.array([1.0]), n_leapfrog=20, step_size=0.3,
                                target_accept=0.8)},
        n_warmup=2000, n_samples=10000, seed=2,
        out_dir=str(artifacts_dir / "randomized_hmc_positive"),
    )
    print("\n" + report.summary())
    report.assert_correct()


def test_randomized_hmc_agrees_with_fixed_hmc(artifacts_dir):
    """Oracle: randomized and fixed HMC must agree (and both match analytic).

    Short trajectory (n_leapfrog=5) keeps L*step_size below one oscillation: with the mass
    whitening this Gaussian the step size adapts large, and a long *fixed-length* trajectory
    over-extends into a resonance (the artefact that randomized integration time fixes, in
    test_randomized_integration_time_fixes_resonance). A short trajectory avoids it so the
    two samplers can be compared cleanly for correctness."""
    problem = correlated_gaussian(mean=[0.0, 1.0], cov=[[1.0, -0.6], [-0.6, 2.0]])
    report = evaluate(
        problem,
        {
            "rhmc": randomized_hmc(n_leapfrog=5, step_size=0.3, metric="diagonal"),
            "hmc": hmc(n_leapfrog=5, step_size=0.3, metric="diagonal"),
        },
        n_warmup=2000, n_samples=10000, seed=3,
        out_dir=str(artifacts_dir / "randomized_vs_fixed_hmc"),
    )
    print("\n" + report.summary())
    report.assert_correct()


def test_randomized_integration_time_fixes_resonance():
    """On a well-conditioned (dense-metric) Gaussian, fixed-length HMC resonates and
    mixes poorly; randomizing the trajectory length restores good mixing.

    The mass matrix adapts to whiten the target, so every direction has ~unit frequency
    (period 2*pi); the step size is *frozen* (``step_size_adapt_rate=0``) at the resonant
    value ``eps ~ 2*pi / n_leapfrog`` where the fixed-length trajectory nearly closes on
    itself. This isolates the integration-time effect from step-size adaptation, so the
    resonance is demonstrated deterministically rather than relying on the adapted step
    happening to land on it."""
    problem = correlated_gaussian(mean=[1.0, -2.0], cov=[[2.0, 1.4], [1.4, 1.5]])
    # n_leapfrog * step_size ~ 2*pi: one full oscillation per trajectory (resonance).
    resonant = dict(n_leapfrog=20, step_size=0.314, metric="dense",
                    step_size_adapt_rate=0.0)

    fixed = hmc(**resonant)(problem.model, seed=0)
    fixed_samples = draw_samples(fixed, 2000, 10000)
    fixed_ess = ess(fixed_samples)

    rand = randomized_hmc(**resonant)(problem.model, seed=0)
    rand_samples = draw_samples(rand, 2000, 10000)
    rand_ess = ess(rand_samples)

    n = len(rand_samples)
    # randomized should mix well in absolute terms, and far better than fixed here
    assert rand_ess.min() / n > 0.3, f"randomized ESS/n too low: {rand_ess / n}"
    assert rand_ess.min() > 3.0 * fixed_ess.min(), (
        f"randomized ESS {rand_ess} not >> fixed ESS {fixed_ess}")
