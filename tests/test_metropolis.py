"""Correctness tests for adaptive random-walk Metropolis--Hastings.

Each test runs the sampler on a problem, compares against the analytic reference
(and, where relevant, against a second sampler configuration), writes trace and pair
plots to ``tests/artifacts/`` for manual inspection, and asserts correctness through
the framework's moment and shape checks.

Seeds are fixed, so pass/fail is deterministic.
"""

from mimcs.testing import (
    correlated_gaussian, rosenbrock, neal_funnel, evaluate, adaptive_mh)


def test_mh_correlated_gaussian(artifacts_dir):
    problem = correlated_gaussian(mean=[1.0, -2.0], cov=[[2.0, 1.4], [1.4, 1.5]])
    report = evaluate(
        problem,
        {"rwmh": adaptive_mh(step_size=0.5, target_accept=0.30, cov_min_samples=50)},
        n_warmup=4000, n_samples=20000, seed=0,
        out_dir=str(artifacts_dir / "correlated_gaussian"),
    )
    print("\n" + report.summary())
    report.assert_correct()


def test_mh_rosenbrock_banana(artifacts_dir):
    problem = rosenbrock(a=1.0, b=5.0)
    report = evaluate(
        problem,
        {"rwmh": adaptive_mh(step_size=0.3, target_accept=0.30, cov_min_samples=50)},
        n_warmup=8000, n_samples=40000, seed=1,
        out_dir=str(artifacts_dir / "rosenbrock"),
    )
    print("\n" + report.summary())
    report.assert_correct()


def test_mh_vs_mh_gaussian(artifacts_dir):
    """Algorithm-vs-algorithm: two independent MH configurations must agree."""
    problem = correlated_gaussian(mean=[0.0, 0.0, 0.0],
                                  cov=[[1.0, 0.5, 0.2],
                                       [0.5, 1.0, 0.3],
                                       [0.2, 0.3, 1.0]])
    report = evaluate(
        problem,
        {
            "mh_a": adaptive_mh(step_size=0.5, target_accept=0.30),
            "mh_b": adaptive_mh(step_size=1.0, target_accept=0.40),
        },
        n_warmup=5000, n_samples=25000, seed=2,
        out_dir=str(artifacts_dir / "gaussian_3d"),
    )
    print("\n" + report.summary())
    report.assert_correct()


def test_mh_funnel_graphical_only(artifacts_dir):
    """Neal's funnel is hard for RWMH: distributional checks are skipped, but the run
    and its plots are still produced for inspection (and to compare with HMC later)."""
    problem = neal_funnel(dim=2, scale=3.0)
    report = evaluate(
        problem,
        {"rwmh": adaptive_mh(step_size=0.5, target_accept=0.30)},
        n_warmup=8000, n_samples=40000, seed=3,
        out_dir=str(artifacts_dir / "neal_funnel"),
    )
    print("\n" + report.summary())
    report.assert_correct()  # skipped because problem.hard is True
