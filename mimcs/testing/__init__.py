"""Testing framework for MCMC samplers.

Compare a sampler's output against an analytic solution (when an exact reference
sampler exists) or against another sampler, with statistically valid moment checks,
a shape-sensitive energy-distance test, and graphical output for manual inspection.

Typical use::

    from mimcs.testing import correlated_gaussian, evaluate, adaptive_mh

    problem = correlated_gaussian(mean=[1, -2], cov=[[2, 1.4], [1.4, 1.5]])
    report = evaluate(problem, {"rwmh": adaptive_mh(step_size=0.5)},
                      n_warmup=4000, n_samples=20000, seed=0,
                      out_dir="artifacts/correlated_gaussian")
    print(report.summary())
    report.assert_correct()
"""

from .problems import (
    TargetProblem, correlated_gaussian, rosenbrock, neal_funnel, neal_funnel_blocks,
    neal_funnel_vector, positive_lognormal, uniform_interval, nested_uniform, block_gaussian,
    von_mises_fisher, uniform_sphere, unit_vector_array,
    dirichlet_simplex,
    wishart_cov, ordered_normal, ordered_uniform, gaussian_mixture,
    spike_and_slab)
from ..diagnostics import autocorrelation, ess, ess_1d, mcse_mean
from .comparison import (
    Thresholds, Check, ComparisonResult, compare, energy_distance_test)
from .runner import (
    Report, SamplerOutput, evaluate, draw_samples,
    adaptive_mh, hmc, randomized_hmc, nuts, simple_nuts, rmhmc, rmnuts, explicit_rmhmc,
    explicit_rmnuts, relativistic_hmc, relativistic_nuts, wal_hmc, wal_nuts, mwal_nuts,
    multirate_nuts, block_hmc, block_nuts, nuts_gibbs)
from . import plots

__all__ = [
    "TargetProblem", "correlated_gaussian", "rosenbrock", "neal_funnel",
    "neal_funnel_blocks", "neal_funnel_vector",
    "positive_lognormal", "uniform_interval", "nested_uniform", "block_gaussian",
    "von_mises_fisher", "uniform_sphere", "unit_vector_array",
    "dirichlet_simplex", "ordered_normal", "ordered_uniform", "wishart_cov",
    "gaussian_mixture", "spike_and_slab",
    "autocorrelation", "ess", "ess_1d", "mcse_mean",
    "Thresholds", "Check", "ComparisonResult", "compare", "energy_distance_test",
    "Report", "SamplerOutput", "evaluate", "draw_samples", "adaptive_mh", "hmc",
    "randomized_hmc", "nuts", "simple_nuts", "rmhmc", "rmnuts", "explicit_rmhmc",
    "explicit_rmnuts", "relativistic_hmc", "relativistic_nuts", "wal_hmc", "wal_nuts",
    "mwal_nuts", "multirate_nuts", "block_hmc", "block_nuts", "nuts_gibbs",
    "plots",
]
