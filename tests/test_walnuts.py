"""Tests for WALNUTS: the within-orbit adaptive ``LineSearchIntegrator``.

The within-orbit step-size adaptivity lives entirely in the integrator, so it composes with
the existing samplers: ``wal_hmc`` = HMC + LineSearchIntegrator, ``wal_nuts`` = NUTS + the
same. Each macro step refines the leapfrog discretization until the energy error is within a
per-level budget, using a base integrator for the forward and reversibility-preserving
backward sub-steps. Reversibility uses the finer of the forward / backward required levels.

Seeds are fixed, so pass/fail is deterministic.
"""

import numpy as np
import jax.numpy as jnp

from mimcs.hmc import (
    LineSearchIntegrator, leapfrog, default_potentials, make_kinetic,
    init_integrator_state, HamiltonianContext, doubling_schedule)
from mimcs.hmc.hamiltonians import total_energy
from mimcs.testing import (
    correlated_gaussian, rosenbrock, neal_funnel, evaluate, wal_hmc, wal_nuts)


def _setup(model, schedule, thresholds):
    pot = default_potentials(model)
    kin = make_kinetic("diagonal")
    base = leapfrog(pot, kin)
    lsi = LineSearchIntegrator(base, pot, kin, schedule=schedule,
                               error_thresholds=thresholds)
    ctx = HamiltonianContext(model.init_chart_hyperparams(), model.init_chart_indices(),
                             {"T": jnp.ones(model.coord_dim)})
    return pot, kin, base, lsi, ctx


def test_line_search_reduces_to_base_step():
    """A single-level schedule with a loose budget never refines, so a macro step is exactly
    one base (leapfrog) step."""
    model = correlated_gaussian(mean=[1.0, -2.0], cov=[[2.0, 1.4], [1.4, 1.5]]).model
    pot, kin, base, lsi, ctx = _setup(model, [(1.0, 1)], 1e9)
    s0 = init_integrator_state(pot, jnp.array([0.3, -0.5]), jnp.array([1.0, 0.7]), ctx)
    out, ref = lsi.step(s0, 0.3, ctx), base.step(s0, 0.3, ctx)
    assert np.allclose(np.asarray(out.q), np.asarray(ref.q), atol=1e-5)
    assert np.allclose(np.asarray(out.p), np.asarray(ref.p), atol=1e-5)
    assert float(out.log_weight) == 0.0


def test_line_search_refines_with_stiffness():
    """Within-orbit adaptivity: the chosen refinement level grows as the geometry stiffens
    (deeper in Neal's funnel neck), without spurious divergence."""
    model = neal_funnel(dim=2, scale=3.0).model
    pot, kin, base, lsi, ctx = _setup(model, doubling_schedule(8), 0.8)
    levels = []
    for v0 in (-2.0, -4.0, -6.0):
        s0 = init_integrator_state(pot, jnp.array([v0, 0.1]), jnp.array([1.0, 1.0]), ctx)
        level, _, diverged = lsi._line_search(s0, 0.6, ctx)
        assert not bool(diverged)
        levels.append(int(level))
    assert levels[0] <= levels[1] <= levels[2] and levels[2] > levels[0]


def test_wal_hmc_samples_gaussian():
    problem = correlated_gaussian(mean=[1.0, -2.0], cov=[[2.0, 1.4], [1.4, 1.5]])
    report = evaluate(problem, {"wal": wal_hmc(n_macro=12, step_size=0.5)},
                      n_warmup=2000, n_samples=10000, seed=0)
    print("\n" + report.summary())
    report.assert_correct()


def test_wal_nuts_samples_gaussian():
    problem = correlated_gaussian(mean=[1.0, -2.0], cov=[[2.0, 1.4], [1.4, 1.5]])
    report = evaluate(problem, {"wal": wal_nuts(step_size=0.5)},
                      n_warmup=2000, n_samples=8000, seed=0)
    print("\n" + report.summary())
    report.assert_correct()


def test_wal_nuts_samples_banana(artifacts_dir):
    """A mild Rosenbrock banana (b=1): correctness on curved geometry."""
    problem = rosenbrock(a=1.0, b=1.0)
    report = evaluate(problem, {"wal": wal_nuts(step_size=0.4)},
                      n_warmup=2000, n_samples=8000, seed=1,
                      out_dir=str(artifacts_dir / "wal_nuts_banana"))
    print("\n" + report.summary())
    report.assert_correct()


def test_wal_nuts_custom_schedule_samples_gaussian():
    """A non-default refinement schedule (tripling) still samples correctly."""
    problem = correlated_gaussian(mean=[1.0, -2.0], cov=[[2.0, 1.4], [1.4, 1.5]])
    schedule = [(3.0 ** -j, 3 ** j) for j in range(4)]
    report = evaluate(problem, {"wal": wal_nuts(step_size=0.5, schedule=schedule)},
                      n_warmup=2000, n_samples=8000, seed=0)
    print("\n" + report.summary())
    report.assert_correct()


def test_wal_nuts_beats_nuts_divergences_on_funnel():
    """The headline: within-orbit refinement lets a single orbit traverse the funnel neck
    (tiny steps) and mouth (large steps), so WAL-NUTS diverges far less than fixed-step NUTS
    and reaches deeper into the neck."""
    from mimcs.testing import nuts
    problem = neal_funnel(dim=2, scale=3.0)
    std = nuts(step_size=0.5, max_tree_depth=10)(problem.model, seed=0)
    std.warmup(1500)
    std.sample(4000)
    wal = wal_nuts(step_size=0.5, max_tree_depth=8, schedule=doubling_schedule(8),
                   error_thresholds=0.8)(problem.model, seed=0)
    wal.warmup(1500)
    wal_v = wal.sample(4000)["x"][:, 0]
    assert wal.divergence_count() < 0.25 * std.divergence_count()
    assert wal_v.min() < -7.0          # reaches deep into the neck


def test_wal_nuts_explores_funnel_neck(artifacts_dir):
    """WAL-NUTS samples Neal's funnel, reaching deep into the neck and out to the mouth (a
    single orbit refines its step within the neck). Distributional checks skipped (hard)."""
    problem = neal_funnel(dim=2, scale=3.0)
    report = evaluate(
        problem, {"wal": wal_nuts(step_size=0.6, max_tree_depth=7,
                                  schedule=doubling_schedule(8), error_thresholds=0.8)},
        n_warmup=2000, n_samples=8000, seed=0,
        out_dir=str(artifacts_dir / "wal_nuts_funnel"))
    print("\n" + report.summary())
    v = report.outputs["wal"].samples[:, 0]
    assert v.min() < -4.0 and v.max() > 5.0
