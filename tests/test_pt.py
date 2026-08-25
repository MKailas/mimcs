"""Tests for parallel tempering (``docs/design/13_parallel_tempering.md``).

The failure this design can produce quietly is a **biased beta=1 marginal**, and no convergence
diagnostic would reveal it: a biased PT chain can have excellent R-hat and ESS. So the
load-bearing tests compare the cold chain against distributions whose answer is known, through
the same `evaluate` harness the rest of the suite uses.

The second thing worth testing is that PT does what it exists for — crossing between modes — and
that is measured over many seeds, never one.

Seeds are fixed, so pass/fail is deterministic.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from mimcs.model import Model, EuclideanParameter
from mimcs.hmc import NUTS, make_kinetic
from mimcs.adaptation import (RobbinsMonroStepSize, LineSearchStepSizeAdaptation,
                             MassMatrixAdaptation)
from mimcs.hmc.line_search import LineSearchIntegrator, doubling_schedule
from mimcs.hmc.nuts import DEFAULT_DIVERGENCE_THRESHOLD
from mimcs.pt import (parallel_tempering, geometric_ladder, swap_log_ratios, apply_swaps,
                     ProductModel, build_tempered_potentials, product_line_search)
from mimcs.testing import (correlated_gaussian, positive_lognormal, evaluate, nuts,
                          block_gaussian, neal_funnel_blocks)


def _pt(**kw):
    """A PT builder for `evaluate`: global step size, per-temperature mass (doc 13)."""
    def build(model, seed):
        return parallel_tempering(
            model, seed=seed, extra_mixins=(RobbinsMonroStepSize,),
            adapt_mixins=(MassMatrixAdaptation,), **kw)
    return build


# --- the ladder -------------------------------------------------------------- #

def test_geometric_ladder_starts_at_one_and_decreases():
    b = np.asarray(geometric_ladder(5, 0.01))
    assert b[0] == 1.0 and np.all(np.diff(b) < 0)
    assert np.isclose(b[-1], 0.01)
    assert np.asarray(geometric_ladder(1)).tolist() == [1.0]


def test_a_ladder_reaching_zero_places_the_last_rung_there():
    b = np.asarray(geometric_ladder(4, 0.0))
    assert b[0] == 1.0 and b[-1] == 0.0 and np.all(np.diff(b) < 0)


@pytest.mark.parametrize("betas, match", [
    ([0.5, 0.1], "must be beta = 1"),
    ([1.0, 0.5, 0.7], "strictly decreasing"),
])
def test_a_bad_ladder_is_rejected(betas, match):
    with pytest.raises(ValueError, match=match):
        parallel_tempering(correlated_gaussian().model, betas=betas)


# --- the load-bearing exactness checks ---------------------------------------- #

def test_the_cold_chain_samples_the_target_gaussian(artifacts_dir):
    """The one test that would catch a silently biased cold marginal."""
    problem = correlated_gaussian(mean=[1.0, -2.0], cov=[[2.0, 1.4], [1.4, 1.5]])
    report = evaluate(problem, {"pt": _pt(n_temperatures=4, beta_min=0.05)},
                      n_warmup=2000, n_samples=20000, seed=0,
                      out_dir=artifacts_dir / "pt_gaussian")
    print("\n" + report.summary())
    report.assert_correct()


def test_the_cold_chain_samples_a_constrained_target(artifacts_dir):
    """A bounded parameter, so the chart Jacobian is in play — and must not be tempered."""
    problem = positive_lognormal(sigma=1.0)
    report = evaluate(problem, {"pt": _pt(n_temperatures=4, beta_min=0.05)},
                      n_warmup=2000, n_samples=20000, seed=0,
                      out_dir=artifacts_dir / "pt_lognormal")
    print("\n" + report.summary())
    report.assert_correct()


def test_one_temperature_reduces_to_the_base_sampler():
    """K = 1 must be the base sampler: the product machinery adds nothing at a single rung."""
    model = correlated_gaussian().model
    pt = parallel_tempering(model, n_temperatures=1, seed=0)
    pt.warmup(400)
    pt_draws = pt.sample(1500)["x"]

    plain = NUTS(model, np.asarray(model.default_sample()), seed=0)
    plain.warmup(400)
    plain_draws = plain.sample(1500)["x"]

    assert pt_draws.shape == plain_draws.shape
    assert np.allclose(pt_draws, plain_draws, atol=1e-4), (
        "K=1 diverged from plain NUTS; the product path changed something it should not have")


# --- tempering ---------------------------------------------------------------- #

def _flat_model(sd=1.0, dim=2):
    return Model([EuclideanParameter("x", (dim,))],
                 {"lp": lambda p: -0.5 * jnp.sum((p["x"] / sd) ** 2)})


def test_tempering_flattens_the_hot_chains():
    """beta scales the log-density, so a hot chain's target is wider by 1/sqrt(beta)."""
    model = _flat_model()
    betas = jnp.asarray([1.0, 0.25])
    pots = build_tempered_potentials(model, betas)
    pmodel = ProductModel(model, 2)
    ctx = type("C", (), {"chart_hyperparams": model.init_chart_hyperparams(),
                         "chart_indices": model.init_chart_indices(), "ham_params": {}})()
    q = jnp.asarray([1.0, 0.0, 1.0, 0.0])              # the same point at both temperatures
    v = sum(p.untempered_values(q, ctx) for p in pots)
    assert np.allclose(np.asarray(v), np.asarray(v)[0])            # untempered: identical
    total = sum(float(p.potential(q, ctx)) for p in pots)
    assert np.isclose(total, float(v[0]) * (1.0 + 0.25), rtol=1e-5)  # tempered: beta-weighted


def test_the_jacobian_is_never_tempered():
    """A change of variables is not part of the target; scaling it samples something else."""
    model = positive_lognormal(sigma=1.0).model
    pots = build_tempered_potentials(model, jnp.asarray([1.0, 0.3]))
    by_id = {p.id: p for p in pots}
    assert "V_jacobian" in by_id, sorted(by_id)
    assert by_id["V_jacobian"].tempered is False
    assert any(p.tempered for p in pots)                # ... while the model components are


def test_naming_components_gives_a_power_posterior():
    model = Model([EuclideanParameter("x", (1,))],
                  {"prior": lambda p: -0.5 * jnp.sum(p["x"] ** 2),
                   "lik": lambda p: -0.5 * jnp.sum((p["x"] - 3.0) ** 2)})
    pots = {p.component: p.tempered for p in build_tempered_potentials(model, jnp.asarray([1.0]),
                                                                      tempered=["lik"])}
    assert pots == {"prior": False, "lik": True}

    with pytest.raises(ValueError, match="not log-density components"):
        build_tempered_potentials(model, jnp.asarray([1.0]), tempered=["nope"])


# --- swaps -------------------------------------------------------------------- #

def test_equal_temperatures_always_swap():
    """With no temperature gap the ratio is 1, whatever the states — a useful sanity anchor."""
    betas = jnp.ones((4,))
    logr = np.asarray(swap_log_ratios(jnp.asarray([1.0, -5.0, 3.0, 0.0]), betas, jnp.int32(0)))
    attempted = np.isfinite(logr)
    assert attempted.tolist() == [True, False, True, False]        # even pairs (0,1), (2,3)
    assert np.allclose(logr[attempted], 0.0)                       # log alpha = 0 -> accept w.p. 1


def test_swap_ratio_matches_the_analytic_expression():
    betas = jnp.asarray([1.0, 0.5, 0.25])
    L = jnp.asarray([-2.0, -1.0, -0.5])
    logr = np.asarray(swap_log_ratios(L, betas, jnp.int32(0)))
    assert np.isclose(logr[0], (1.0 - 0.5) * (-1.0 - -2.0))        # pair (0,1)
    assert not np.isfinite(logr[1])                                # odd pair, not attempted
    odd = np.asarray(swap_log_ratios(L, betas, jnp.int32(1)))
    assert np.isclose(odd[1], (0.5 - 0.25) * (-0.5 - -1.0))        # pair (1,2)


def test_a_swap_sweep_permutes_disjoint_pairs():
    rows = jnp.asarray([[0.0], [1.0], [2.0], [3.0]])
    out = np.asarray(apply_swaps(rows, jnp.asarray([True, False, True, False])))
    assert out.ravel().tolist() == [1.0, 0.0, 3.0, 2.0]
    out = np.asarray(apply_swaps(rows, jnp.asarray([False, True, False, False])))
    assert out.ravel().tolist() == [0.0, 2.0, 1.0, 3.0]


def test_swaps_actually_happen_and_are_reported():
    model = correlated_gaussian().model
    s = parallel_tempering(model, n_temperatures=4, beta_min=0.05, seed=0,
                           extra_mixins=(RobbinsMonroStepSize,),
                           adapt_mixins=(MassMatrixAdaptation,))
    s.warmup(600); s.sample(600)
    rates = s.swap_rates()
    assert rates.shape == (3,)
    assert np.all(rates > 0.05), f"the ladder is disconnected: {rates}"


# --- per-temperature mass ------------------------------------------------------ #

def test_each_temperature_adapts_its_own_mass():
    """A hot chain is wider, and the mass is what absorbs that — the reason the step size can
    stay global (doc 13). The learned inverse-mass should track 1/beta."""
    problem = correlated_gaussian(mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]])
    betas = [1.0, 0.25, 0.0625]
    # `adapt_ladder=False`: this isolates the *mass*, so the ladder must stay where it is put.
    # With adaptation on (the default) the gaps widen toward the 0.234 swap target and the
    # masses track the moved ladder instead, which is correct but not what is being tested.
    s = parallel_tempering(problem.model, betas=betas, seed=0, adapt_ladder=False,
                           extra_mixins=(RobbinsMonroStepSize,),
                           adapt_mixins=(MassMatrixAdaptation,))
    s.warmup(2000); s.sample(500)
    inv_mass = np.asarray(s.state.ham_params["T"])       # (K, dim), = the target variance
    assert inv_mass.shape == (3, 2)
    ratio = inv_mass.mean(axis=1) / inv_mass.mean(axis=1)[0]
    assert np.all(np.diff(ratio) > 0), f"a hotter chain should learn a wider mass: {ratio}"
    # variance ~ 1/beta: ratios should be near [1, 4, 16]
    assert np.allclose(ratio, [1.0, 4.0, 16.0], rtol=0.45), ratio


# --- the point of the exercise ------------------------------------------------- #

def _bimodal(sep=8.0, sd=0.7):
    """Two well-separated Gaussians: NUTS started in one mode essentially never finds the other."""
    def log_post(p):
        x = p["x"]
        a = -0.5 * jnp.sum(((x - sep / 2) / sd) ** 2)
        b = -0.5 * jnp.sum(((x + sep / 2) / sd) ** 2)
        return jnp.logaddexp(a, b)
    return Model([EuclideanParameter("x", (1,))], {"lp": log_post})


def _visits_both_modes(draws, sep=8.0):
    x = np.asarray(draws).ravel()
    return bool((x > sep / 4).any() and (x < -sep / 4).any())


def test_parallel_tempering_crosses_between_modes():
    """What PT is for. Measured over 8 seeds, because one seed says nothing about mode finding.

    Prediction before running: plain NUTS started in one mode should essentially never cross a
    4-sigma-deep barrier (0/8 seeds), while PT with a ladder reaching beta = 0.02 — where the
    barrier is worth ~0.4 nats instead of ~20 — should cross in most.
    """
    model = _bimodal()
    init = np.array([4.0])                      # start in the right-hand mode, every seed

    plain = 0
    for seed in range(8):
        s = NUTS(model, init, seed=seed, step_size=0.5)
        s.warmup(500)
        plain += _visits_both_modes(s.sample(2000)["x"])

    tempered = 0
    for seed in range(8):
        s = parallel_tempering(model, init_position=init, n_temperatures=6, beta_min=0.02,
                               seed=seed, extra_mixins=(RobbinsMonroStepSize,),
                               adapt_mixins=(MassMatrixAdaptation,))
        s.warmup(1000)
        tempered += _visits_both_modes(s.sample(2000)["x"])

    print(f"\nseeds visiting both modes: plain NUTS {plain}/8, parallel tempering {tempered}/8")
    assert plain <= 2, f"the barrier is not actually trapping plain NUTS ({plain}/8 crossed)"
    assert tempered >= 6, f"PT failed to cross the barrier ({tempered}/8)"
    assert tempered > plain


# --- independent acceptance (the fixed-trajectory samplers) --------------------- #

def test_hmc_accepts_independently_at_each_temperature():
    """RWMH and HMC need not share an accept/reject decision, so they do not (doc 13).

    ``target_accept`` is the library's RWMH default of 0.234 unless asked otherwise, which is far
    too low for HMC --- hence the explicit 0.8 here, exactly as for a non-tempered HMC.
    """
    from mimcs.hmc import HMC
    problem = correlated_gaussian(mean=[1.0, -2.0], cov=[[2.0, 1.4], [1.4, 1.5]])
    s = parallel_tempering(problem.model, base=HMC, n_leapfrog=12, n_temperatures=4,
                           beta_min=0.05, seed=0, target_accept=0.8,
                           extra_mixins=(RobbinsMonroStepSize,),
                           adapt_mixins=(MassMatrixAdaptation,))
    s.warmup(2000)
    draws = s.sample(6000)["x"]

    accept = np.stack(s._diag["accept_prob"])[-3000:]
    assert accept.shape[1] == 4, "acceptance should be per temperature, not shared"
    assert np.all(accept.mean(axis=0) > 0.4), accept.mean(axis=0)
    # ... and each temperature adapts its own step size, which independent acceptance makes
    # meaningful (unlike the joint-selection NUTS path).
    steps = np.asarray(s.state.step_size)
    assert steps.shape == (4,) and np.all(steps > 0)

    assert np.all(np.abs(draws.mean(axis=0) - np.array([1.0, -2.0])) < 0.25)
    assert np.allclose(draws.std(axis=0), np.sqrt([2.0, 1.5]), rtol=0.12)



# --- ladder adaptation (Miasojedow-Moulines-Vihola) ---------------------------- #

def test_rho_parametrization_round_trips_and_keeps_the_ladder_valid():
    """Every rho gives an ordered ladder starting at beta = 1 --- no constraint to enforce."""
    from mimcs.pt import betas_from_rho, rho_from_betas
    b = jnp.asarray([1.0, 0.5, 0.2, 0.05])
    for t_max in (None, 1.0 / 0.05):                       # free gaps, and gaps as fixed shares
        got = np.asarray(betas_from_rho(rho_from_betas(b, t_max), t_max))
        assert np.allclose(got, np.asarray(b), rtol=1e-5), (t_max, got)

    rng = np.random.default_rng(0)
    for _ in range(20):                                   # arbitrary rho -> a valid ladder
        rho = jnp.asarray(rng.normal(size=4) * 3)
        for t_max in (None, 100.0):
            out = np.asarray(betas_from_rho(rho, t_max))
            assert out[0] == 1.0 and np.all(np.diff(out) < 0) and np.all(out > 0)
            if t_max is not None:                          # the held endpoint is exactly held
                assert np.isclose(out[-1], 1.0 / t_max, rtol=1e-5), out


def test_the_ladder_adapts_toward_the_target_swap_rate():
    """A deliberately lopsided ladder should have its interior rungs redistributed.

    The start is [1, 0.9, 0.8, 0.01]: the first two gaps are so narrow that those pairs swap
    almost always, while the last spans two orders of magnitude and almost never does. Since the
    endpoints are held (see `ladder.py`), the only thing adaptation *can* do is move the interior
    rungs down --- and the measure of success is that the swap rates become more equal. A fixed
    ladder from the same start is the control, so this cannot pass on a run where nothing moved.
    """
    problem = correlated_gaussian(mean=[1.0, -2.0], cov=[[2.0, 1.4], [1.4, 1.5]])
    start = [1.0, 0.9, 0.8, 0.01]

    def run(adapt):
        s = parallel_tempering(problem.model, betas=start, seed=0, adapt_ladder=adapt,
                               extra_mixins=(RobbinsMonroStepSize,),
                               adapt_mixins=(MassMatrixAdaptation,))
        s.warmup(2000)
        s.sample(1000)
        return np.asarray(s.betas), np.asarray(s.swap_rates())

    fixed_betas, fixed_rates = run(False)
    end, rates = run(True)

    assert np.allclose(fixed_betas, start)                      # the control really is fixed
    assert end[0] == 1.0 and np.all(np.diff(end) < 0)           # still a valid ladder
    assert np.isclose(end[-1], start[-1])                       # beta_min is held, by construction
    assert np.all(end[1:-1] < np.asarray(start)[1:-1]), (start, end)   # interior moved down
    assert np.nanstd(rates) < np.nanstd(fixed_rates), (rates, fixed_rates)


def test_a_free_ladder_can_leave_the_starting_range():
    """`adapt_beta_min=True` restores the unbounded MMV form, which is not the default.

    Kept as a guard on the option, not as a recommendation: on a target whose hot end goes flat
    this form runs away (see `ladder.py`), which is exactly why the endpoints are held.
    """
    s = parallel_tempering(correlated_gaussian().model, n_temperatures=4, beta_min=0.1, seed=0,
                           adapt_beta_min=True, extra_mixins=(RobbinsMonroStepSize,))
    s.warmup(500)
    assert float(np.asarray(s.betas)[-1]) < 0.1, s.betas


def _offset_model(offset: float = 1e5, dim: int = 2):
    """A standard Gaussian whose log-density carries a large additive constant.

    The constant stands in for what a large-data model has and a toy one does not: `hmm_gaussian`'s
    500 observations put ``V`` at order 1e5. It cancels in the posterior *and* in every swap ratio
    (both are differences of two replicas' ``V``), so the sampling problem is unchanged and the
    rungs still exchange --- but it does **not** cancel in ``beta * V``, which is precisely what a
    stale ladder cache gets wrong. Scaling ``V`` up any other way fails to reproduce the bug: a
    d-dimensional Gaussian only reaches ``V ~ d/2beta`` (~240 at d = 200), three orders short.
    """
    return Model([EuclideanParameter("x", (dim,))],
                 {"lp": lambda p: -0.5 * jnp.sum(p["x"] ** 2) - offset})


def test_the_ladder_update_refreshes_the_cached_potentials():
    """Moving a rung's beta invalidates that rung's cached values and gradients.

    It is the same invalidation a swap causes --- moving a replica to another beta and moving a
    beta under a replica are one event seen from two sides --- and the swap has always re-seeded
    after one. The ladder update did not, so the cache described a Hamiltonian nobody was
    integrating: the next trajectory's leading half-kick reads that gradient straight back out
    (`cached_gradient=True`) and `log_prob` is the acceptance baseline.

    Checked as an exact invariant rather than through its symptom, so it cannot pass vacuously.
    """
    model = _offset_model()
    s = parallel_tempering(model, n_temperatures=3, beta_min=0.1, seed=0, adapt_ladder=True,
                           extra_mixins=(RobbinsMonroStepSize,))
    s.warmup(40)
    st = s.state
    ctx = s.context(st)
    assert not np.allclose(np.asarray(s.betas), np.asarray(geometric_ladder(3, 0.1))), (
        "the ladder never moved, so this would hold trivially")
    for pot in s.potentials:
        assert np.allclose(np.asarray(st.potential_values[pot.id]),
                           np.asarray(pot.potential(st.coordinate, ctx)), rtol=1e-5), pot.id
    assert np.allclose(float(st.log_prob),
                       -float(sum(np.asarray(pot.potential(st.coordinate, ctx))
                                  for pot in s.potentials)), rtol=1e-5)


def test_ladder_adaptation_survives_a_large_log_density_scale():
    """The stale cache cost a step ``delta_beta * V``, so the damage is the product of the two.

    The trajectory's energy baseline ``H0`` is seeded from the *cached* potential values, so a
    stale one makes every leaf look like it carries that error: acceptance goes to zero and
    Robbins--Monro drives the step size down. Both factors are needed, which is why this went
    unseen --- every ladder test used an O(1) ``V`` (error ~1e-3 nats), and a held ``beta_min``
    barely moves the ladder at all when no pair swaps, because a uniform ``alpha - alpha*`` is a
    no-op on softmax shares. `hmm_gaussian` had both: ``V ~ 1e5`` and a ladder moving ~3e-3 per
    step, so ~300 nats per step, a step size at 1e-13 within 25 iterations, and NUTS then building
    a maximum-depth tree every step without ever U-turning.

    Measured here: 0.0024 broken vs 2.04 fixed. The smaller offsets are the control that shows it
    is the scale of ``V`` doing the damage and not the free ladder on its own.
    """
    def step_after_warmup(offset):
        s = parallel_tempering(_offset_model(offset), n_temperatures=3, beta_min=0.1, seed=0,
                               adapt_ladder=True, adapt_beta_min=True,   # a ladder that moves
                               extra_mixins=(RobbinsMonroStepSize,))
        s.warmup(200)
        return float(np.max(np.asarray(s.state.step_size)))

    small = step_after_warmup(1e3)
    assert small > 0.5, f"the control itself collapsed ({small:.3g}) --- the test says nothing"
    large = step_after_warmup(1e5)
    assert large > 0.5, (
        f"step size collapsed to {large:.3g} at V ~ 1e5 (control at 1e3: {small:.3g}): the "
        "ladder update is leaving the cached potentials at the previous beta")


def test_a_fixed_ladder_stays_put():
    s = parallel_tempering(correlated_gaussian().model, betas=[1.0, 0.3, 0.1], seed=0,
                           adapt_ladder=False, extra_mixins=(RobbinsMonroStepSize,))
    s.warmup(300)
    assert np.allclose(np.asarray(s.betas), [1.0, 0.3, 0.1])


# --- WALNUTS over the product space -------------------------------------------- #

def test_line_search_runs_over_the_product_space_unchanged():
    """WALNUTS needs no product variant: it refines against `total_energy`, which over the
    product space is already the sum over temperatures. So the integrator built here is the
    ordinary `LineSearchIntegrator`, not a subclass of anything PT-specific."""
    s = parallel_tempering(correlated_gaussian().model, n_temperatures=3, beta_min=0.1, seed=0,
                           integrator=product_line_search(error_thresholds=0.6),
                           extra_mixins=(LineSearchStepSizeAdaptation,))
    assert type(s.integrator) is LineSearchIntegrator
    s.warmup(200)
    draws = s.sample(200)
    assert np.all(np.isfinite(np.asarray(draws["x"])))


def test_the_line_search_budget_scales_with_the_number_of_temperatures():
    """The product Hamiltonian is a sum of K, so a per-temperature budget must be multiplied by
    K --- otherwise each rung is silently held to K times the intended accuracy."""
    model = correlated_gaussian().model
    built = {}
    for K in (1, 5):
        s = parallel_tempering(model, n_temperatures=K, beta_min=0.1, seed=0,
                               integrator=product_line_search(error_thresholds=0.7),
                               extra_mixins=(LineSearchStepSizeAdaptation,))
        built[K] = np.asarray(s.integrator._thresholds)
    assert np.allclose(built[1], 0.7)
    assert np.allclose(built[5], 0.7 * 5)


def test_the_divergence_threshold_scales_with_the_number_of_temperatures():
    """`max(H) - min(H)` is over the summed energy, so the K=1 budget would flag ordinary
    product-space energy ranges. Measured on the funnel: scaling cut median divergences 488 ->
    304 with the neck depth unchanged, i.e. those were false positives."""
    model = correlated_gaussian().model
    s = parallel_tempering(model, n_temperatures=6, beta_min=0.1, seed=0)
    assert s.divergence_threshold == DEFAULT_DIVERGENCE_THRESHOLD * 6
    explicit = parallel_tempering(model, n_temperatures=6, beta_min=0.1, seed=0,
                                  divergence_threshold=42.0)
    assert explicit.divergence_threshold == 42.0     # an explicit value still wins


def test_parallel_tempered_walnuts_samples_the_target_gaussian(artifacts_dir):
    """The load-bearing check for PT-WALNUTS: the cold chain must still be the target.

    Within-orbit refinement changes the trajectory, so this is not implied by the leapfrog
    exactness test --- a wrong reversibility condition over the product space would bias the
    beta=1 marginal while leaving R-hat and ESS looking healthy.
    """
    problem = correlated_gaussian(mean=[1.0, -2.0], cov=[[2.0, 1.4], [1.4, 1.5]])

    def build(model, seed):
        return parallel_tempering(
            model, n_temperatures=4, beta_min=0.05, seed=seed, step_size=0.5,
            integrator=product_line_search(schedule=doubling_schedule(6), error_thresholds=0.8),
            extra_mixins=(LineSearchStepSizeAdaptation,),
            adapt_mixins=(MassMatrixAdaptation,))

    report = evaluate(problem, {"pt_wal": build}, n_warmup=1500, n_samples=6000, seed=0,
                      out_dir=str(artifacts_dir / "pt_walnuts_gaussian"))
    print("\n" + report.summary())
    report.assert_correct()


def test_parallel_tempered_walnuts_reaches_deeper_into_the_funnel_neck(artifacts_dir):
    """What tempering buys on the funnel, which is not multimodal but *is* a barrier in v.

    A hot rung's v-marginal has sd ``scale/sqrt(beta)``, so it visits neck depths the cold chain
    would take a very long time to reach on its own, and swaps carry them down. Measured over 8
    seeds under x64 (the funnel needs it --- see the note below): PT-WALNUTS reached a median
    v.min of -8.8 against untempered WALNUTS' -3.3, for ~1.6x the gradients and an *identical*
    median divergence count (152 both). This pins seed 0 only, so it is a regression guard on
    that result rather than the measurement itself.

    Note this runs in float32, where 2 of those 8 seeds froze outright (every trajectory
    non-finite); seed 0 is one of the healthy ones. That fragility is the reason `docs/design/13`
    tells you to enable x64 for tempered funnels.
    """
    from mimcs.testing import neal_funnel

    model = neal_funnel(dim=2, scale=3.0).model
    s = parallel_tempering(
        model, n_temperatures=4, beta_min=0.05, seed=0, step_size=0.6, max_tree_depth=7,
        integrator=product_line_search(schedule=doubling_schedule(8), error_thresholds=0.8),
        extra_mixins=(LineSearchStepSizeAdaptation,),
        adapt_mixins=(MassMatrixAdaptation,))
    s.warmup(1000)
    v = np.asarray(s.sample(2000)["x"])[:, 0]
    print(f"\nPT-WALNUTS funnel: v in [{v.min():.2f}, {v.max():.2f}], "
          f"{s.divergence_count()} divergence(s)")
    assert v.min() < -6.0, f"neck depth {v.min():.2f}: tempering is not reaching the neck"
    assert v.max() > 4.0, f"mouth {v.max():.2f}: the chain is stuck in the neck"


# --- learned (position-dependent) metrics over the product space --------------- #

def _learned_product_kinetic(K=3, dim=4):
    """A `LearnedDiagonalBlock` on `x` depending on `v`, wrapped for K temperatures."""
    from mimcs.hmc.block_riemannian import build_block
    from mimcs.hmc.metric_expr import Exp
    from mimcs.pt.kinetics import ProductKinetic

    model = neal_funnel_blocks(dim=dim, scale=3.0).model
    inner = build_block(model, "x", Exp("v"))
    assert not inner.separable and inner.depends      # else these tests prove nothing
    return model, inner, ProductKinetic(inner, K, model.coord_dim)


def _product_state(q, p):
    from mimcs.hmc.state import IntegratorState
    return IntegratorState(q=q, p=p, potential_values={}, potential_grads={},
                           log_weight=jnp.zeros(()), integrator_data={})


def _distinct_params_per_rung(inner, model, K):
    """Per-rung metric parameters that differ, so a lane reading the wrong row would show."""
    one = inner.initial_mass_params(model.coord_dim)
    return jax.tree.map(lambda x: jnp.stack([jnp.asarray(x) + 0.3 * k for k in range(K)]), one)


@pytest.mark.parametrize("eps", [0.37, "per_temperature"])
def test_a_learned_metric_flows_exactly_as_it_would_per_temperature(eps):
    """The load-bearing check on the product flow: no coupling between rungs.

    A position-dependent metric has no explicit drift, so `ProductKinetic` runs the block's own
    flow per lane. That is exact rather than approximate --- the product Hamiltonian is a sum of K
    uncoupled terms over disjoint coordinates --- so the result must equal applying the inner flow
    to each temperature separately, with that rung's own parameters. Any cross-temperature leakage
    (a lane seeing another rung's coordinates or params) breaks this identity.
    """
    from mimcs.hmc.state import HamiltonianContext

    K = 3
    model, inner, pk = _learned_product_kinetic(K=K)
    n = model.coord_dim
    rng = np.random.default_rng(0)
    q, p = jnp.asarray(rng.normal(size=K * n)), jnp.asarray(rng.normal(size=K * n))
    params = _distinct_params_per_rung(inner, model, K)
    ctx = HamiltonianContext(model.init_chart_hyperparams(), model.init_chart_indices(),
                             {inner.id: params})

    per_rung_eps = [0.1, 0.2, 0.3] if eps == "per_temperature" else [eps] * K
    use = jnp.repeat(jnp.asarray(per_rung_eps), n) if eps == "per_temperature" else eps
    out = pk.flow(_product_state(q, p), use, ctx)

    ref_q, ref_p = [], []
    for k, e in enumerate(per_rung_eps):
        ctx_k = HamiltonianContext(ctx.chart_hyperparams, ctx.chart_indices,
                                   {inner.id: jax.tree.map(lambda x: x[k], params)})
        o = inner.flow(_product_state(q.reshape(K, n)[k], p.reshape(K, n)[k]), e, ctx_k)
        ref_q.append(np.asarray(o.q))
        ref_p.append(np.asarray(o.p))

    assert np.allclose(np.asarray(out.q), np.concatenate(ref_q), atol=1e-6)
    assert np.allclose(np.asarray(out.p), np.concatenate(ref_p), atol=1e-6)
    # ... and the flow must actually move both, or the comparison above is vacuous: this block
    # drifts its own coordinates *and* kicks the dependency momenta.
    assert np.abs(np.asarray(out.q) - np.asarray(q)).max() > 1e-3
    assert np.abs(np.asarray(out.p) - np.asarray(p)).max() > 1e-3


def test_the_learned_metric_product_flow_is_reversible():
    """Reversibility holds lane-wise, hence overall; the momentum kick makes this a real check."""
    from mimcs.hmc.state import HamiltonianContext

    K = 3
    model, inner, pk = _learned_product_kinetic(K=K)
    rng = np.random.default_rng(1)
    q = jnp.asarray(rng.normal(size=K * model.coord_dim))
    p = jnp.asarray(rng.normal(size=K * model.coord_dim))
    ctx = HamiltonianContext(model.init_chart_hyperparams(), model.init_chart_indices(),
                             {inner.id: _distinct_params_per_rung(inner, model, K)})

    back = pk.flow(pk.flow(_product_state(q, p), 0.3, ctx), -0.3, ctx)
    assert np.allclose(np.asarray(back.q), np.asarray(q), atol=1e-5)
    assert np.allclose(np.asarray(back.p), np.asarray(p), atol=1e-5)


def test_a_learned_metric_block_still_samples_the_target_under_tempering(artifacts_dir):
    """Exactness with a position-dependent metric in the product space.

    The blocks of `block_gaussian` are independent, so the true metric on `b` is constant and the
    analytic mean/covariance are known. A wrong product flow would bias the beta=1 marginal while
    leaving R-hat and ESS looking healthy, which is exactly what this harness catches.
    """
    from mimcs.factory import analyze
    from mimcs.hmc.metric_expr import Exp

    def build(model, seed):
        spec = analyze(model, blocks=["a", "b"])
        spec.base = "pt_nuts"
        for blk in spec.blocks:
            if blk.names == ["b"]:
                blk.kind, blk.params = "learned_metric", {"metric": Exp("a")}
        assert any(b.kind == "learned_metric" for b in spec.blocks)   # else vacuous
        spec.tempering_params = {"n_temperatures": 3, "beta_min": 0.1}
        return spec.build(seed=seed)

    report = evaluate(block_gaussian(), {"pt_lm": build}, n_warmup=1500, n_samples=8000, seed=0,
                      out_dir=str(artifacts_dir / "pt_learned_metric"))
    print("\n" + report.summary())
    report.assert_correct()


def test_a_learned_metric_is_what_lets_tempering_handle_the_funnel_neck():
    """The payoff, and the reason this flow was worth building.

    A single shared mass cannot serve a funnel whose scale varies by orders of magnitude, so PT
    with a constant diagonal mass stalls short of the neck and diverges; the learned metric
    absorbs that variation. Measured over 4 seeds at this configuration: median v.min -9.55 with
    the metric against -6.60 without, every draw distinct (800/800 against 772), and **zero**
    divergences on all four seeds against a median of 20. Two seeds here, as a regression guard.

    (Deeper ladders on this problem need x64 --- see `docs/design/13`. `beta_min = 0.5` keeps it
    inside float32, which is what the suite runs in.)
    """
    from mimcs.factory import analyze
    from mimcs.hmc.metric_expr import Exp

    def run(learned, seed):
        spec = analyze(neal_funnel_blocks(dim=4, scale=3.0).model, blocks=["v", "x"])
        if learned:
            for b in spec.blocks:
                if b.names == ["x"]:
                    b.kind, b.params = "learned_metric", {"metric": Exp("v")}
            assert any(b.kind == "learned_metric" for b in spec.blocks)
        spec.base = "pt_nuts"
        spec.tempering_params = {"n_temperatures": 3, "beta_min": 0.5}
        s = spec.build(seed=seed)
        s.warmup(500)
        v = np.asarray(s.sample(800)["v"]).ravel()
        return v, s.divergence_count()

    for seed in (0, 1):
        v_lm, div_lm = run(True, seed)
        v_const, div_const = run(False, seed)
        print(f"\nseed {seed}: learned v.min {v_lm.min():.2f} ({div_lm} div), "
              f"constant v.min {v_const.min():.2f} ({div_const} div)")
        assert v_lm.min() < -8.0, f"the learned metric did not reach the neck ({v_lm.min():.2f})"
        assert v_lm.min() < v_const.min(), (v_lm.min(), v_const.min())
        assert div_lm == 0, f"{div_lm} divergence(s) with the learned metric"


def test_the_product_model_reports_the_cold_chain_features():
    """A warmup-termination criterion buffers `state.sample`, which here spans every temperature.

    The cold chain is the one whose draws are kept, so it is the one whose mixing should decide
    when warmup ends --- a hot rung mixes more easily and would stop warmup early. This matches
    every other user-facing narrowing.
    """
    model = correlated_gaussian(mean=[1.0, -2.0], cov=[[2.0, 1.4], [1.4, 1.5]]).model
    pmodel = ProductModel(model, 3)
    cold = jnp.asarray([0.5, -0.25])
    product = jnp.concatenate([cold, jnp.asarray([9.0, 9.0]), jnp.asarray([-9.0, -9.0])])
    assert np.allclose(np.asarray(pmodel.features(product)),
                       np.asarray(model.features(cold)))


def test_a_tempered_sampler_survives_a_warmup_long_enough_to_check_termination():
    """`min_warmup` is 500, so a shorter warmup never reaches the criterion's first check --- and
    a tempered run would then fail only on longer warmups. Cross it explicitly."""
    from mimcs.adaptation import ClassifierTermination

    model = correlated_gaussian().model
    s = parallel_tempering(model, n_temperatures=3, beta_min=0.1, seed=0,
                           extra_mixins=(ClassifierTermination, RobbinsMonroStepSize),
                           adapt_mixins=(MassMatrixAdaptation,))
    s.warmup(700)
    draws = np.asarray(s.sample(200)["x"])
    assert draws.shape == (200, 2) and np.all(np.isfinite(draws))


def test_a_tempered_sampler_initializes_against_its_own_tempered_target():
    """`initialize()` must test the density the *sampler* samples, not the model's.

    `UniformInit` used to ask `model.log_prob_at_coordinate`, which the product model has no
    honest answer for --- the sampled target is beta-weighted and the ladder lives in the
    context, not on the model (see `ProductModel`). It now tests the candidate state's own
    `log_prob`, so the check below is that the accepted point's density really is the tempered
    one: equal to the potentials' cached sum, and *different* from the unweighted product sum
    that a model-level answer would have given.
    """
    from mimcs.adaptation import UniformInit

    model = correlated_gaussian().model
    s = parallel_tempering(model, n_temperatures=3, beta_min=0.1, seed=0,
                           extra_mixins=(UniformInit, RobbinsMonroStepSize))
    s.initialize()
    st = s.state
    assert np.isfinite(float(st.log_prob))
    assert np.isclose(float(st.log_prob), -float(sum(st.potential_values.values())))

    ctx = s.context(st)
    untempered = -float(sum(np.sum(np.asarray(p.untempered_values(st.coordinate, ctx)))
                            for p in s.potentials))
    assert not np.isclose(float(st.log_prob), untempered), (
        "the initializer's target is not beta-weighted --- it would be the untempered product")

    # every rung starts somewhere the target is finite, and they start apart (K independent draws)
    per_rung = np.asarray(st.coordinate).reshape(3, model.coord_dim)
    assert np.all(np.isfinite(per_rung))
    assert not np.allclose(per_rung[0], per_rung[1])


def test_a_tempered_run_summarizes_the_cold_chain():
    """`summary()` evaluates the retained draws, and PT retains the cold chain --- so the model
    it is evaluated against must be the *base* model. Handing it the product model asks for
    `ambient_score` / `stein_terms` that the product view does not have, and would be K times
    too wide even if it did."""
    model = correlated_gaussian(mean=[1.0, -2.0], cov=[[2.0, 1.4], [1.4, 1.5]]).model
    s = parallel_tempering(model, n_temperatures=3, beta_min=0.1, seed=0,
                           extra_mixins=(RobbinsMonroStepSize,),
                           adapt_mixins=(MassMatrixAdaptation,))
    s.warmup(300)
    s.sample(300)
    summary = s.summary()
    assert s.summary_model is model
    assert len(summary.mean) == model.ambient_dim          # the cold chain's width, not 3x it
    assert list(summary.feature_names) == list(model.feature_names)
    assert np.all(np.isfinite(summary.mean)) and np.all(np.isfinite(summary.ess))
