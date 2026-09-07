"""Tests for WALNUTS: the within-orbit adaptive ``LineSearchIntegrator``.

The within-orbit step-size adaptivity lives entirely in the integrator, so it composes with
the existing samplers: ``wal_hmc`` = HMC + LineSearchIntegrator, ``wal_nuts`` = NUTS + the
same. Each macro step refines the leapfrog discretization until the energy error is within a
per-level budget, using a base integrator for the forward and reversibility-preserving
backward sub-steps. The error measure is the energy *range* over the macro step, which is
direction-symmetric; a step whose backward search would pick a coarser level is invalidated.

Seeds are fixed, so pass/fail is deterministic.
"""

import numpy as np
import jax
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


def test_line_search_macro_step_is_an_involution():
    """**Reversibility.** A valid macro step must satisfy ``step(step(z, +eps), -eps) == z``.

    The forward/backward required levels disagree often in the funnel neck, and the temptation
    is to reconcile the disagreement (integrate at ``max(L_f, L_b)``) rather than invalidate it.
    That is not reversible: ``L_b`` is measured at ``Phi_{L_f}(z)`` but the step then lands on
    ``Phi_{max}(z)``, so the reverse search starts somewhere else and picks a third level. The
    energy-error criterion is not direction-symmetric (the two directions use different
    reference energies), which is precisely why ``L_f == L_b`` must be *checked*.

    The disagreement assertion is the **control**: without it the round-trip check could pass
    vacuously on a probe where every step agrees.
    """
    model = neal_funnel(dim=21, scale=3.0).model
    pot, kin, base, lsi, ctx = _setup(model, doubling_schedule(5), 1.0)

    def roundtrip(q, p, eps):
        z = init_integrator_state(pot, q, p, ctx)
        level_fwd, cand, _ = lsi._line_search(z, eps, ctx)
        level_bwd, _, _ = lsi._line_search(cand, -eps, ctx)
        z2 = lsi.step(lsi.step(z, eps, ctx), -eps, ctx)
        err = jnp.max(jnp.abs(z2.q - q)) + jnp.max(jnp.abs(z2.p - p))
        return level_fwd, level_bwd, err, jnp.isfinite(lsi.step(z, eps, ctx).log_weight)

    rng = np.random.default_rng(0)
    n, eps = 300, 0.39
    v = 3.0 * rng.standard_normal(n)
    x = rng.standard_normal((n, 20)) * np.exp(v / 2.0)[:, None]
    q = jnp.asarray(np.column_stack([v, x]), float)
    p = jnp.asarray(rng.standard_normal((n, 21)), float)
    l_f, l_b, err, finite = jax.vmap(roundtrip, in_axes=(0, 0, None))(q, p, eps)
    l_f, l_b, err, finite = (np.asarray(a) for a in (l_f, l_b, err, finite))

    assert (l_f != l_b).sum() > 10, "control: the probe must exercise level disagreement"
    assert not finite[l_f != l_b].any(), "a level disagreement must be invalidated (-inf)"
    scale = np.maximum(1.0, np.abs(np.asarray(q)).max(axis=1))
    assert (err[finite] / scale[finite] < 1e-3).all(), "valid macro steps must round-trip"


def test_wal_nuts_funnel_v_marginal_is_unbiased():
    """The ``v`` marginal of Neal's funnel is exactly ``N(0, scale^2)``, and a broken
    reversibility rule biases it without producing a single divergence. A non-reversible
    ``max(L_f, L_b)`` rule gave mean +2.6 and sd 2.1 here (8/8 seeds); the tolerances below
    pass comfortably for the correct rule and fail decisively for that one."""
    problem = neal_funnel(dim=21, scale=3.0)
    sampler = wal_nuts(step_size=0.5, max_tree_depth=8, schedule=doubling_schedule(5),
                       error_thresholds=1.0)(problem.model, seed=0)
    sampler.warmup(2000)
    v = np.asarray(sampler.sample(20000)["x"][:, 0])
    assert abs(v.mean()) < 1.0, f"v marginal is biased: mean {v.mean():.2f}, expected 0"
    assert 2.2 < v.std() < 3.8, f"v marginal is mis-scaled: sd {v.std():.2f}, expected 3"


def test_error_measure_is_direction_symmetric():
    """The energy **range** ``max_k H - min_k H`` over a macro step is the same measured forward
    from ``z`` and backward from ``Phi_j(z)`` --- the two traverse the same set of states.

    A start-relative deviation ``max_k |H - H(start)|`` is *not*: it measures against ``H(z)`` one
    way and ``H(z')`` the other. That asymmetry is computed inline here as the **control**, so the
    test cannot pass vacuously on a probe where the two happen to agree anyway.
    """
    model = neal_funnel(dim=21, scale=3.0).model
    pot, kin, base, lsi, ctx = _setup(model, doubling_schedule(5), 1.0)

    def deviation(start, level, eps):
        """The old measure, for the control: max |H - H(start)| over the same sub-steps."""
        sub_eps = eps * lsi._h[level]
        h0 = lsi._energy(start, ctx)

        def body(_, carry):
            s, m = carry
            s = base.step(s, sub_eps, ctx)
            return s, jnp.maximum(m, jnp.abs(lsi._energy(s, ctx) - h0))

        return jax.lax.fori_loop(0, lsi._T[level], body, (start, jnp.zeros(())))[1]

    def probe(q, p, eps):
        z = init_integrator_state(pot, q, p, ctx)
        out = []
        for j in range(lsi.n_levels):
            end, rng_f = lsi._integrate_level(z, jnp.int32(j), eps, ctx)
            _, rng_b = lsi._integrate_level(end, jnp.int32(j), -eps, ctx)
            dev_f = deviation(z, jnp.int32(j), eps)
            dev_b = deviation(end, jnp.int32(j), -eps)
            out.append(jnp.stack([rng_f, rng_b, dev_f, dev_b]))
        return jnp.stack(out)

    rng = np.random.default_rng(0)
    n, eps = 200, 0.39
    v = 3.0 * rng.standard_normal(n)
    x = rng.standard_normal((n, 20)) * np.exp(v / 2.0)[:, None]
    q = jnp.asarray(np.column_stack([v, x]), float)
    p = jnp.asarray(rng.standard_normal((n, 21)), float)
    e = np.asarray(jax.vmap(probe, in_axes=(0, 0, None))(q, p, eps))
    thr = np.asarray(lsi._thresholds)[None, :]

    # What the algorithm consumes is the *within-budget decision*, not the raw value, so that is
    # what must agree. (Comparing raw values instead would trip over blown-up coarse levels deep
    # in the neck, where both directions are astronomically over budget and the decision is
    # nonetheless identical.)
    within = lambda a: np.isfinite(a) & (a <= thr)
    range_disagree = (within(e[..., 0]) != within(e[..., 1])).sum()
    dev_disagree = (within(e[..., 2]) != within(e[..., 3])).sum()
    assert range_disagree == 0, f"the range measure disagreed on {range_disagree} decision(s)"
    # control: the old deviation measure genuinely disagrees on these same states
    assert dev_disagree > 0, "control: the deviation measure must disagree somewhere"

    finite = np.isfinite(e[..., 0]) & np.isfinite(e[..., 1])
    gap = np.abs(e[..., 0] - e[..., 1])
    assert np.median(gap[finite]) == 0.0, "the range measure must agree exactly in the median"
    # near the threshold --- the only place a float32 tie could flip a decision --- the two
    # directions agree to ~1e-5, five orders below the margin they are being compared against
    near = finite & (np.abs(e[..., 0] - thr) < 0.5 * thr)
    assert near.sum() > 50 and gap[near].max() < 1e-4


def test_backward_level_never_exceeds_forward():
    """``L_bwd <= L_fwd`` always, because the forward-chosen level is valid backward by symmetry.

    This is the theorem that lets ``step`` check only the *coarser* levels
    (``_coarser_level_valid``) instead of running a full backward line search."""
    model = neal_funnel(dim=21, scale=3.0).model
    pot, kin, base, lsi, ctx = _setup(model, doubling_schedule(5), 1.0)

    def levels(q, p, eps):
        z = init_integrator_state(pot, q, p, ctx)
        l_f, end, diverged = lsi._line_search(z, eps, ctx)
        l_b, _, _ = lsi._line_search(end, -eps, ctx)
        return l_f, l_b, diverged

    rng = np.random.default_rng(0)
    n, eps = 400, 0.39
    v = 3.0 * rng.standard_normal(n)
    x = rng.standard_normal((n, 20)) * np.exp(v / 2.0)[:, None]
    q = jnp.asarray(np.column_stack([v, x]), float)
    p = jnp.asarray(rng.standard_normal((n, 21)), float)
    l_f, l_b, div = (np.asarray(a) for a in jax.vmap(levels, in_axes=(0, 0, None))(q, p, eps))
    live = ~div
    assert (l_b[live] <= l_f[live]).all(), "the chosen level must be valid backward"
    assert (l_b[live] < l_f[live]).sum() > 10, "control: minimality must actually fail sometimes"
