"""Tests for the Markovian WALNUTS variant: ``MarkovianLineSearchIntegrator``.

Unlike the deterministic WALNUTS-D :class:`LineSearchIntegrator` (which concentrates the
micro-step length on the coarsest valid level and invalidates steps that disagree forward vs
backward), this variant chooses the refinement level by a coarse-to-fine Markov chain: below the
error budget it advances with probability ``p`` and stops with ``1 - p``; above it, it is forced
finer; the finest level always stops. It is reversible *without* invalidation, at the cost of a
more spread-out micro-step-length distribution.

The reversibility weight for a macro step ``z -> z'`` at level ``J`` is
``log P_rev(J|z') - log P_fwd(J|z)``: each unforced forward move contributes ``-log p`` and each
backward coarser level within budget contributes ``+log p`` (the ``(1-p)`` stop factors cancel).
A clean consequence is the antisymmetry ``C(z->z') == -C(z'->z)``, tested directly below.

Seeds are fixed, so pass/fail is deterministic.
"""

import numpy as np
import jax
import jax.numpy as jnp

from mimcs.samplers import make_sampler_class
from mimcs.hmc import (
    MarkovianLineSearchIntegrator, LineSearchIntegrator, leapfrog, default_potentials,
    make_kinetic, init_integrator_state, HamiltonianContext, doubling_schedule,
    NUTS, SimpleNUTS)
from mimcs.testing import (
    correlated_gaussian, rosenbrock, neal_funnel, evaluate, draw_samples, mwal_nuts)


def _setup(model, schedule, thresholds, p=0.5):
    pot = default_potentials(model)
    kin = make_kinetic("diagonal")
    base = leapfrog(pot, kin)
    lsi = MarkovianLineSearchIntegrator(base, pot, kin, schedule=schedule,
                                        error_thresholds=thresholds, p=p)
    ctx = HamiltonianContext(model.init_chart_hyperparams(), model.init_chart_indices(),
                             {"T": jnp.ones(model.coord_dim)})
    return pot, kin, base, lsi, ctx


def _direct_sampler(Cls, model, seed, *, schedule, thresholds, p):
    """Build a NUTS/SimpleNUTS sampler sharing a MarkovianLineSearchIntegrator (no builder
    exists for SimpleNUTS + the Markovian integrator)."""
    kin = make_kinetic("diagonal")
    pot = default_potentials(model)
    base = leapfrog(pot, kin)
    lsi = MarkovianLineSearchIntegrator(base, pot, kin, schedule=schedule,
                                        error_thresholds=thresholds, p=p)
    return Cls(model, init_position=model.default_sample(), seed=seed, kinetic=kin,
               potentials=pot, integrator=lsi, step_size=0.4, max_tree_depth=8)


# --- unit tests ---------------------------------------------------------------- #

def test_reduces_to_base_step():
    """A single-level schedule always stops at the finest level immediately, so a macro step is
    exactly one base (leapfrog) step and carries no reversibility weight."""
    model = correlated_gaussian(mean=[1.0, -2.0], cov=[[2.0, 1.4], [1.4, 1.5]]).model
    pot, kin, base, lsi, ctx = _setup(model, [(1.0, 1)], 1e9)
    s0 = init_integrator_state(pot, jnp.array([0.3, -0.5]), jnp.array([1.0, 0.7]), ctx)
    out = lsi.step(s0, 0.3, ctx, jnp.zeros(1))
    ref = base.step(s0, 0.3, ctx)
    assert np.allclose(np.asarray(out.q), np.asarray(ref.q), atol=1e-5)
    assert np.allclose(np.asarray(out.p), np.asarray(ref.p), atol=1e-5)
    assert float(out.log_weight) == 0.0


def test_reversibility_antisymmetry():
    """The reversibility weight is antisymmetric: for a macro step ``z -> z'`` at level ``J``,
    a matched reverse macro step ``z' -> z`` (retracing to ``J``) carries the opposite weight.
    This is the exact reversibility statement, checked without any sampling."""
    model = neal_funnel(dim=3, scale=3.0).model
    p = 0.5
    pot, kin, base, lsi, ctx = _setup(model, doubling_schedule(6), 0.8, p=p)
    eps = 0.7
    tested = 0
    for s in range(40):
        kq, kp, kr = jax.random.split(jax.random.PRNGKey(s), 3)
        scale = jnp.array([0.5] * (model.coord_dim - 1) + [2.0])
        q0 = jax.random.normal(kq, (model.coord_dim,)) * scale
        p0 = jax.random.normal(kp, (model.coord_dim,))
        z0 = init_integrator_state(pot, q0, p0, ctx)
        rng_f = jax.random.uniform(kr, (lsi.n_levels,))

        J, zp, lwf, nonfinite, _, _ = lsi._markov_forward(z0, eps, ctx, rng_f)
        if bool(nonfinite) or int(J) == 0:      # skip a blow-up / a trivial (no-refinement) step
            continue
        C01 = float(lwf + lsi._backward_logp(zp, eps, ctx, J))

        # Force the reverse chain to retrace: advance (coin fires, u < p) at every level below J,
        # stop at J (u >= p). p = 0.5 makes 0.0 an advance and 0.99 a stop.
        rng_r = jnp.where(jnp.arange(lsi.n_levels) < J, 0.0, 0.99)
        Jr, zr, lwf_r, _, _, _ = lsi._markov_forward(zp, -eps, ctx, rng_r)
        C10 = float(lwf_r + lsi._backward_logp(zr, -eps, ctx, Jr))

        assert int(Jr) == int(J)                                  # retraced to the same level
        assert np.allclose(np.asarray(zr.q), np.asarray(z0.q), atol=1e-4)   # ... and to z
        assert abs(C01 + C10) < 1e-4                              # antisymmetric weight
        # each nonzero contribution is a multiple of log p
        assert abs((C01 / np.log(p)) - round(C01 / np.log(p))) < 1e-4
        tested += 1
    assert tested >= 8       # the neck points must actually exercise refinement


def test_no_forced_invalidation_reaching_finest():
    """Reaching the finest level is *accepted* (finite weight), never invalidated -- the whole
    point of the Markovian variant. Only a non-finite energy yields ``-inf``."""
    model = neal_funnel(dim=2, scale=3.0).model
    # tight budget + shallow schedule => the neck forces the chain to the finest level often
    pot, kin, base, lsi, ctx = _setup(model, doubling_schedule(4), 0.2)
    reached_finest = False
    for s in range(60):
        kq, kp, kr = jax.random.split(jax.random.PRNGKey(s), 3)
        q0 = jnp.array([-5.0, 0.1]) + 0.2 * jax.random.normal(kq, (2,))
        p0 = jax.random.normal(kp, (2,))
        z0 = init_integrator_state(pot, q0, p0, ctx)
        rng = jax.random.uniform(kr, (lsi.n_levels,))
        J, _, _, nonfinite, _, _ = lsi._markov_forward(z0, 0.7, ctx, rng)
        out = lsi.step(z0, 0.7, ctx, rng)
        if not bool(nonfinite):
            assert bool(jnp.isfinite(out.log_weight))    # accepted, not invalidated
        reached_finest |= (int(J) == lsi.n_levels - 1)
    assert reached_finest


def test_spreads_microsteps_vs_walnuts_d():
    """At a fixed stiff (neck) point, WALNUTS-D concentrates on a single coarsest-valid level,
    while the Markovian variant spreads the chosen level across several -- the design trade-off."""
    model = neal_funnel(dim=2, scale=3.0).model
    pot, kin, base, lsi, ctx = _setup(model, doubling_schedule(8), 0.8)
    det = LineSearchIntegrator(leapfrog(pot, kin), pot, kin, schedule=doubling_schedule(8),
                               error_thresholds=0.8)
    z0 = init_integrator_state(pot, jnp.array([-4.0, 0.1]), jnp.array([1.0, 1.0]), ctx)

    det_level = int(det._line_search(z0, 0.6, ctx)[0])
    rngs = jax.random.uniform(jax.random.PRNGKey(0), (3000, lsi.n_levels))
    levels = np.asarray(jax.vmap(lambda r: lsi._markov_forward(z0, 0.6, ctx, r)[0])(rngs))

    assert len(np.unique(levels)) >= 3        # spread, not a point mass
    assert levels.min() >= det_level          # never coarser than the coarsest valid level
    assert float(levels.var()) > 0.25


def test_mnuts_matches_simple_mnuts():
    """Memory-efficient NUTS and reference SimpleNUTS trace the identical chain with the
    randomized integrator too -- the per-leaf line-search draws are threaded identically."""
    model = correlated_gaussian(mean=[1.0, -2.0], cov=[[2.0, 1.4], [1.4, 1.5]]).model
    cfg = dict(schedule=doubling_schedule(6), thresholds=0.8, p=0.5)
    Xn = draw_samples(_direct_sampler(make_sampler_class(NUTS), model, 0, **cfg), 200, 500)
    Xs = draw_samples(_direct_sampler(make_sampler_class(SimpleNUTS), model, 0, **cfg), 200, 500)
    assert np.array_equal(Xn, Xs)


# --- sampling correctness ------------------------------------------------------ #

def test_mwal_nuts_samples_gaussian():
    problem = correlated_gaussian(mean=[1.0, -2.0], cov=[[2.0, 1.4], [1.4, 1.5]])
    report = evaluate(problem, {"mwal": mwal_nuts(step_size=0.5, p=0.5)},
                      n_warmup=2000, n_samples=8000, seed=0)
    print("\n" + report.summary())
    report.assert_correct()


def test_mwal_nuts_samples_banana(artifacts_dir):
    """A mild Rosenbrock banana (b=1): correctness on curved geometry."""
    problem = rosenbrock(a=1.0, b=1.0)
    report = evaluate(problem, {"mwal": mwal_nuts(step_size=0.4, p=0.5)},
                      n_warmup=2000, n_samples=8000, seed=1,
                      out_dir=str(artifacts_dir / "mwal_nuts_banana"))
    print("\n" + report.summary())
    report.assert_correct()


def test_mwal_nuts_beats_nuts_divergences_on_funnel():
    """The headline, mirroring the WALNUTS-D test: within-orbit refinement lets one orbit
    traverse the neck (tiny steps) and mouth (large steps), so the Markovian variant diverges
    far less than fixed-step NUTS and reaches deeper into the neck. (Its divergences are the
    residual NUTS ``h_max-h_min`` / genuine finest-level blow-ups, not line-search invalidation,
    which the variant never produces -- see ``test_no_forced_invalidation_reaching_finest``.)"""
    from mimcs.testing import nuts
    problem = neal_funnel(dim=2, scale=3.0)
    std = nuts(step_size=0.5, max_tree_depth=10)(problem.model, seed=0)
    std.warmup(1500)
    std.sample(4000)
    wal = mwal_nuts(step_size=0.5, max_tree_depth=8, schedule=doubling_schedule(8),
                    error_thresholds=0.8, p=0.5)(problem.model, seed=0)
    wal.warmup(1500)
    wal_v = wal.sample(4000)["x"][:, 0]
    assert wal.divergence_count() < 0.25 * std.divergence_count()
    assert wal_v.min() < -6.0                 # reaches deep into the neck
    assert wal_v.max() > 5.0                  # ... and out to the mouth
