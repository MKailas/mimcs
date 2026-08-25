"""Correctness tests for the No-U-Turn Sampler.

NUTS is validated through the same framework as HMC: against analytic references
(Gaussian diagonal+dense, mild banana, constrained positive) and against fixed-length
HMC as an oracle. Two NUTS-specific checks confirm the tree machinery: zero divergences
and naturally-terminating (sub-max) tree depth on an easy target.

Neal's funnel is handled honestly. Basic NUTS with a *global* metric does not sample the
deep funnel cleanly -- it misses the neck and diverges there (the classic funnel
pathology, the motivation for Riemannian / reparameterized methods). So we test a mild
funnel (scale=1) strictly, and use the deep funnel (scale=3) to confirm the divergence
diagnostic correctly flags the pathological geometry rather than asserting correctness.

Seeds are fixed, so pass/fail is deterministic.
"""

import numpy as np

from mimcs.testing import (
    correlated_gaussian, rosenbrock, positive_lognormal, neal_funnel,
    evaluate, nuts, simple_nuts, randomized_hmc, draw_samples)


def test_nuts_matches_simple_nuts():
    """The memory-efficient NUTS must trace the IDENTICAL chain as the reference
    full-memory SimpleNUTS given the same RNG -- they make the same U-turn and
    multinomial decisions, so the checkpoint bookkeeping is verified bit-for-bit."""
    problem = correlated_gaussian(mean=[1.0, -2.0], cov=[[2.0, 1.4], [1.4, 1.5]])
    cfg = dict(max_tree_depth=10, step_size=0.3, metric="dense")
    Xn = draw_samples(nuts(**cfg)(problem.model, seed=0), 500, 2000)
    Xs = draw_samples(simple_nuts(**cfg)(problem.model, seed=0), 500, 2000)
    assert np.array_equal(Xn, Xs)


def test_nuts_correlated_gaussian_diagonal(artifacts_dir):
    problem = correlated_gaussian(mean=[1.0, -2.0], cov=[[2.0, 1.4], [1.4, 1.5]])
    report = evaluate(
        problem, {"nuts": nuts(max_tree_depth=10, step_size=0.3, metric="diagonal")},
        n_warmup=2000, n_samples=8000, seed=0,
        out_dir=str(artifacts_dir / "nuts_gaussian_diagonal"))
    print("\n" + report.summary())
    report.assert_correct()


def test_nuts_correlated_gaussian_dense(artifacts_dir):
    problem = correlated_gaussian(mean=[1.0, -2.0], cov=[[2.0, 1.4], [1.4, 1.5]])
    report = evaluate(
        problem, {"nuts": nuts(max_tree_depth=10, step_size=0.3, metric="dense")},
        n_warmup=2000, n_samples=8000, seed=1,
        out_dir=str(artifacts_dir / "nuts_gaussian_dense"))
    print("\n" + report.summary())
    report.assert_correct()


def test_nuts_rosenbrock_banana(artifacts_dir):
    # Mild banana (b=1); a stiff banana is a known global-metric limitation (see HMC tests).
    problem = rosenbrock(a=1.0, b=1.0)
    report = evaluate(
        problem, {"nuts": nuts(max_tree_depth=10, step_size=0.2, target_accept=0.9)},
        n_warmup=2000, n_samples=8000, seed=2,
        out_dir=str(artifacts_dir / "nuts_rosenbrock"))
    print("\n" + report.summary())
    report.assert_correct()


def test_nuts_positive_parameter(artifacts_dir):
    problem = positive_lognormal(sigma=1.0)
    report = evaluate(
        problem, {"nuts": nuts(init=np.array([1.0]), max_tree_depth=10, step_size=0.3)},
        n_warmup=2000, n_samples=8000, seed=3,
        out_dir=str(artifacts_dir / "nuts_positive"))
    print("\n" + report.summary())
    report.assert_correct()


def test_nuts_agrees_with_randomized_hmc(artifacts_dir):
    """Oracle: NUTS (no trajectory-length tuning) and randomized HMC must agree, and both
    match analytic. (Randomized HMC is used rather than fixed-length HMC because the
    latter is resonance-prone with a dense metric on a well-conditioned Gaussian.)"""
    problem = correlated_gaussian(
        mean=[0.0, 0.0, 0.0],
        cov=[[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]])
    report = evaluate(
        problem,
        {
            "nuts": nuts(max_tree_depth=10, step_size=0.3, metric="dense"),
            "rhmc": randomized_hmc(n_leapfrog=20, step_size=0.3, metric="dense"),
        },
        n_warmup=2000, n_samples=8000, seed=4,
        out_dir=str(artifacts_dir / "nuts_vs_randomized_hmc"))
    print("\n" + report.summary())
    report.assert_correct()


def test_nuts_no_divergences_and_terminates_on_gaussian():
    """Tree machinery sanity on an easy target: no divergences, and the U-turn fires
    (mean depth modest, max depth below the cap -- not just hitting max_tree_depth)."""
    problem = correlated_gaussian(mean=[1.0, -2.0], cov=[[2.0, 1.4], [1.4, 1.5]])
    sampler = nuts(max_tree_depth=10, step_size=0.3, metric="dense")(problem.model, seed=0)
    draw_samples(sampler, 1500, 5000)
    assert sampler.divergence_count() == 0
    assert 1.0 <= sampler.mean_tree_depth() <= 7.0
    assert sampler.diagnostics("all")["tree_depth"].max() < 10   # natural termination, not the cap


def test_nuts_counts_warmup_and_sampling_divergences_separately():
    """Warmup and sampling divergences are bucketed apart: the default counts sampling-phase only,
    ``include_warmup=True`` adds the warmup bucket, and the phases partition the total. The deep
    funnel diverges in both phases, so it exercises both buckets."""
    problem = neal_funnel(dim=2, scale=3.0)
    sampler = nuts(max_tree_depth=10, step_size=0.3, target_accept=0.9)(problem.model, seed=5)
    draw_samples(sampler, 2000, 4000)
    sampling = sampler.divergence_count()                              # default: sampling only
    warmup = sampler.divergence_count(include_warmup=True, include_sampling=False)
    both = sampler.divergence_count(include_warmup=True, include_sampling=True)
    # cross-check against the uniform diagnostics store
    assert sampling == int(sampler.diagnostics("sampling")["diverging"].sum())
    assert warmup == int(sampler.diagnostics("warmup")["diverging"].sum())
    assert both == warmup + sampling
    assert warmup > 0 and sampling > 0                                # the funnel diverges in both
    # the rate is over the selected phase(s), matching the count
    assert np.isclose(sampler.divergence_rate(), sampling / 4000)


def test_nuts_mild_funnel(artifacts_dir):
    """A mild funnel (scale=1) is within global-metric NUTS's reach: strict check."""
    problem = neal_funnel(dim=2, scale=1.0)
    problem.hard = False   # mild enough for a real correctness assertion
    report = evaluate(
        problem, {"nuts": nuts(max_tree_depth=10, step_size=0.3, target_accept=0.9)},
        n_warmup=2000, n_samples=8000, seed=3,
        out_dir=str(artifacts_dir / "nuts_mild_funnel"))
    print("\n" + report.summary())
    report.assert_correct()


def test_nuts_divergence_diagnostic_on_deep_funnel(artifacts_dir):
    """Deep funnel (scale=3): basic NUTS with a global metric cannot sample the neck and
    diverges there. We do not assert distributional correctness (known limitation); we
    confirm the divergence diagnostic fires, alerting the user to the pathology. Plots
    are written for inspection."""
    problem = neal_funnel(dim=2, scale=3.0)
    # plots for inspection (the deep funnel is hard=True, so no distributional assertion)
    report = evaluate(
        problem, {"nuts": nuts(max_tree_depth=10, step_size=0.3, target_accept=0.9)},
        n_warmup=2000, n_samples=8000, seed=5,
        out_dir=str(artifacts_dir / "nuts_deep_funnel"))
    report.assert_correct()            # skipped (problem.hard is True)

    # read the divergence diagnostic directly from a sampler instance
    sampler = nuts(max_tree_depth=10, step_size=0.3, target_accept=0.9)(problem.model, seed=5)
    draw_samples(sampler, 2000, 8000)
    print(f"deep funnel divergences: {sampler.divergence_count()} / 8000")
    assert sampler.divergence_count() > 10   # the diagnostic detects the pathological neck
