"""Tests for proxy-energy step-size adaptation of the WALNUTS line-search integrators.

Ordinary acceptance-driven step-size adaptation fails for ``LineSearchIntegrator``: it refines
until the energy error is within budget, so the real acceptance is ~1 regardless of the macro
step size and adaptation runs the step size away upward. Instead the integrator emits a *proxy
energy* whose per-macro-step increment is the actual energy change normalized to a
coarse-level-equivalent scale, ``ΔH / (T_j · h_j³)`` (see ``docs/design/06``). Step-size
adaptation (``LineSearchStepSizeAdaptation``) is driven by the proxy acceptance
``min(1, exp(-proxy_energy))``, while the real energy still governs the actual accept.

This is carried through the new ``IntegratorState.integrator_data`` contract field. Seeds fixed.
"""

import numpy as np
import jax
import jax.numpy as jnp

from mimcs.samplers import make_sampler_class
from mimcs.adaptation import RobbinsMonroStepSize, MassMatrixAdaptation, LineSearchStepSizeAdaptation
from mimcs.hmc import (
    LineSearchIntegrator, leapfrog, default_potentials, make_kinetic,
    init_integrator_state, HamiltonianContext, doubling_schedule, NUTS)
from mimcs.testing import (
    correlated_gaussian, neal_funnel, evaluate, draw_samples, wal_nuts, wal_hmc, mwal_nuts)


def _integrator(model, schedule=None, thresholds=0.8):
    pot = default_potentials(model)
    kin = make_kinetic("diagonal")
    lsi = LineSearchIntegrator(leapfrog(pot, kin), pot, kin,
                               schedule=schedule or doubling_schedule(6),
                               error_thresholds=thresholds)
    ctx = HamiltonianContext(model.init_chart_hyperparams(), model.init_chart_indices(),
                             {"T": jnp.ones(model.coord_dim)})
    return pot, kin, lsi, ctx


# --- contract + emission --------------------------------------------------------- #

def test_integrator_data_schema_and_emission():
    """Deterministic integrators carry an empty ``integrator_data``; the line-search integrator
    seeds a proxy schema and advances it each macro step."""
    model = correlated_gaussian(mean=[1.0, -2.0], cov=[[2.0, 1.4], [1.4, 1.5]]).model
    assert set(leapfrog(default_potentials(model),
                        make_kinetic("diagonal")).init_integrator_data()) == {"grad_evals"}
    pot, kin, lsi, ctx = _integrator(model)
    assert lsi.emits_step_size_proxy and set(lsi.init_integrator_data()) == {
        "proxy_energy", "refine_sum", "n_steps", "grad_evals"}
    s0 = init_integrator_state(pot, jnp.array([0.3, -0.5]), jnp.array([1.0, 0.7]), ctx)._replace(
        integrator_data=lsi.init_integrator_data())
    out = lsi.step(s0, 0.4, ctx)
    d = out.integrator_data
    assert float(d["n_steps"]) == 1.0 and jnp.isfinite(d["proxy_energy"])


def test_integrate_from_unseeded_state():
    """``integrate`` (used by fixed-length HMC and the factory init's step-size line-search probe)
    must work from an unseeded ``integrator_data={}`` state for both line-search integrators --- it
    seeds the schema so the while_loop carry structure stays stable; the Markovian one runs
    deterministically (no per-step rng in integrate)."""
    from mimcs.hmc import init_integrator_state, MarkovianLineSearchIntegrator
    model = correlated_gaussian(mean=[1.0, -2.0], cov=[[2.0, 1.4], [1.4, 1.5]]).model
    pot, kin, lsi, ctx = _integrator(model)
    s0 = init_integrator_state(pot, jnp.array([0.3, -0.5]), jnp.array([1.0, 0.7]), ctx)  # data == {}
    out = lsi.integrate(s0, 0.4, 3, ctx)
    assert jnp.isfinite(out.q).all() and float(out.integrator_data["grad_evals"]) > 0
    mk = MarkovianLineSearchIntegrator(leapfrog(pot, kin), pot, kin,
                                       schedule=doubling_schedule(6), error_thresholds=0.8, p=0.5)
    out_m = mk.integrate(s0, 0.4, 3, ctx)
    assert jnp.isfinite(out_m.q).all() and float(out_m.integrator_data["grad_evals"]) > 0


def test_proxy_is_a_restoring_signal_in_h():
    """The proxy acceptance decreases as the macro step size grows (a restoring signal that can cap
    h), where the raw within-budget acceptance would stay saturated near 1."""
    model = correlated_gaussian(mean=[1.0, -2.0], cov=[[2.0, 1.4], [1.4, 1.5]]).model
    pot, kin, lsi, ctx = _integrator(model)
    seed0 = lsi.init_integrator_data()

    def mean_proxy_accept(eps):
        def one(key):
            p = jax.random.normal(key, (model.coord_dim,))
            s = init_integrator_state(pot, jnp.array([0.2, -0.3]), p, ctx)._replace(
                integrator_data=seed0)
            e = lsi.step(s, eps, ctx).integrator_data["proxy_energy"]
            return jnp.minimum(1.0, jnp.exp(-e))
        keys = jax.random.split(jax.random.PRNGKey(0), 400)
        return float(jnp.mean(jax.vmap(one)(keys)))

    lo, hi = mean_proxy_accept(0.3), mean_proxy_accept(1.2)
    assert hi < lo - 0.02        # bigger step -> lower proxy acceptance (restoring)


# --- the core claim: h stabilizes instead of running away ------------------------ #

def _linesearch_nuts(mixins, model, seed, *, step_size, thresholds=0.8, **kw):
    pot = default_potentials(model)
    kin = make_kinetic("diagonal")
    lsi = LineSearchIntegrator(leapfrog(pot, kin), pot, kin, schedule=doubling_schedule(6),
                               error_thresholds=thresholds)
    Cls = make_sampler_class(*mixins, NUTS)
    return Cls(model, init_position=model.default_sample(), seed=seed, kinetic=kin,
               potentials=pot, integrator=lsi, step_size=step_size, max_tree_depth=8, **kw)


def test_proxy_adaptation_stabilizes_step_size_vs_naive():
    """Proxy-driven adaptation converges the step size to a finite plateau; naive adaptation on the
    real acceptance runs it away upward and forces deep refinement. Contrast on a Gaussian."""
    model = correlated_gaussian(mean=[1.0, -2.0], cov=[[2.0, 1.4], [1.4, 1.5]]).model

    def run(mixins, n=500):
        s = _linesearch_nuts(mixins, model, 0, step_size=0.5, target_accept=0.8)
        sizes = [float((s.step(), s.state.step_size)[1]) for _ in range(n)]
        return np.array(sizes), s.mean_refinements()

    proxy_sizes, proxy_refine = run([LineSearchStepSizeAdaptation, MassMatrixAdaptation])
    naive_sizes, naive_refine = run([RobbinsMonroStepSize, MassMatrixAdaptation])

    # proxy: bounded plateau, shallow refinement on a benign target
    assert proxy_sizes[-50:].mean() < 4.0 * proxy_sizes[0]
    assert proxy_refine < 1.5
    # naive: runs the step up far higher and forces much deeper refinement
    assert naive_sizes[-50:].mean() > 2.0 * proxy_sizes[-50:].mean()
    assert naive_refine > proxy_refine + 1.0


def test_markovian_proxy_not_inflated_vs_deterministic():
    """The coarsest-valid-level proxy normalization keeps the Markovian variant's adapted step size
    comparable to the deterministic one. Its extra *unforced* refinement must not make the proxy
    optimistic and inflate the step (pre-fix the ratio was ~1.9x). Averaged over seeds."""
    model = correlated_gaussian(mean=[1.0, -2.0], cov=[[2.0, 1.4], [1.4, 1.5]]).model
    cfg = dict(step_size=0.5, adapt_step_size=True, target_accept=0.8,
               schedule=doubling_schedule(6), error_thresholds=0.8, max_tree_depth=8)

    def frozen(builder, seed):
        s = builder(model, seed=seed)
        s.warmup(800)
        return float(s.state.step_size)

    wal = np.array([frozen(wal_nuts(**cfg), sd) for sd in range(5)])
    mwal = np.array([frozen(mwal_nuts(p=0.5, **cfg), sd) for sd in range(5)])
    assert mwal.mean() < 1.4 * wal.mean()      # not inflated (pre-fix ~1.9x)


def test_mean_refinements_ordered_gaussian_vs_funnel():
    """The refinement diagnostic is finite and larger where the geometry is stiffer (funnel neck)
    than on a benign Gaussian."""
    g = wal_nuts(step_size=0.6, adapt_step_size=True, schedule=doubling_schedule(8),
                 error_thresholds=0.8)(correlated_gaussian().model, seed=0)
    g.warmup(400); g.sample(400)
    f = wal_nuts(step_size=0.6, adapt_step_size=True, schedule=doubling_schedule(8),
                 error_thresholds=0.8)(neal_funnel(dim=2, scale=3.0).model, seed=0)
    f.warmup(400); f.sample(400)
    assert np.isfinite(g.mean_refinements()) and np.isfinite(f.mean_refinements())
    assert f.mean_refinements() > g.mean_refinements()


# --- sampling correctness with adaptation on ------------------------------------- #

def test_wal_nuts_adapt_samples_gaussian():
    problem = correlated_gaussian(mean=[1.0, -2.0], cov=[[2.0, 1.4], [1.4, 1.5]])
    report = evaluate(problem, {"wal": wal_nuts(step_size=0.5, adapt_step_size=True)},
                      n_warmup=2000, n_samples=8000, seed=0)
    print("\n" + report.summary())
    report.assert_correct()


def test_wal_hmc_adapt_samples_gaussian():
    problem = correlated_gaussian(mean=[1.0, -2.0], cov=[[2.0, 1.4], [1.4, 1.5]])
    report = evaluate(problem, {"wal": wal_hmc(n_macro=12, step_size=0.5, adapt_step_size=True)},
                      n_warmup=2000, n_samples=10000, seed=0)
    print("\n" + report.summary())
    report.assert_correct()


def test_mwal_nuts_adapt_samples_gaussian():
    problem = correlated_gaussian(mean=[1.0, -2.0], cov=[[2.0, 1.4], [1.4, 1.5]])
    report = evaluate(problem, {"mwal": mwal_nuts(step_size=0.5, p=0.5, adapt_step_size=True)},
                      n_warmup=2000, n_samples=8000, seed=0)
    print("\n" + report.summary())
    report.assert_correct()


# --- the harness builders adapt by default ------------------------------------- #

def test_the_walnuts_builders_adapt_the_step_size_by_default():
    """The harness should build what the factory ships.

    ``mimcs.make_sampler`` swaps in ``LineSearchStepSizeAdaptation`` whenever the integrator emits
    a step-size proxy (``mimcs/factory/build.py``), so a WALNUTS sampler from the factory always
    adapts. The test builders used to default the other way, which left every sampling test in
    ``test_walnuts.py`` / ``test_mwalnuts.py`` exercising a configuration nobody runs.
    """
    model = correlated_gaussian().model
    for build in (wal_nuts(), wal_hmc(), mwal_nuts()):
        assert LineSearchStepSizeAdaptation in type(build(model, seed=0)).__mro__
    # ... and the opt-out still composes the fixed-step class.
    off = wal_nuts(adapt_step_size=False)(model, seed=0)
    assert LineSearchStepSizeAdaptation not in type(off).__mro__
    assert MassMatrixAdaptation in type(off).__mro__


def test_the_default_adaptation_actually_moves_the_step_size():
    """Composing the mixin is not the same as it doing anything --- so check the step size moves.

    This is the assertion that would catch the adaptation going quietly inert: a sampler can carry
    the mixin and still hold its step size fixed if the proxy never reaches it. It also pins the
    *direction*, which is the substance: the step is pushed **up** on a benign Gaussian and **down**
    on the funnel's stiff neck. Measured over four seeds while writing this, the Gaussian settles
    at 0.70--0.82 and the funnel at 0.17--0.44 from the same 0.5 start, never overlapping; the
    margin below is loose enough for the worst of those.
    """
    gauss = wal_nuts(step_size=0.5)(correlated_gaussian().model, seed=0)
    gauss.warmup(400)
    g_steps = np.asarray(gauss.warmup_step_sizes(), float)

    funnel = wal_nuts(step_size=0.5)(neal_funnel(dim=2, scale=3.0).model, seed=0)
    funnel.warmup(400)
    f_steps = np.asarray(funnel.warmup_step_sizes(), float)

    assert len(np.unique(g_steps)) > 20, "the step size never moved"
    assert g_steps.max() / g_steps.min() > 1.5
    assert g_steps[-1] > 0.5, f"a benign target should admit a bigger step, got {g_steps[-1]}"
    assert f_steps[-1] < 0.5, f"a stiff neck should force a smaller step, got {f_steps[-1]}"
    assert f_steps[-1] < 0.8 * g_steps[-1]

    # the opt-out holds its step size exactly, which is what makes the contrast meaningful
    fixed = wal_nuts(step_size=0.5, adapt_step_size=False)(correlated_gaussian().model, seed=0)
    fixed.warmup(400)
    assert len(np.unique(np.asarray(fixed.warmup_step_sizes(), float))) == 1
