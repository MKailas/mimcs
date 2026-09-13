"""Tests for exact conditional Gibbs, and for the per-parameter update dispatch it arrived with.

Three things here are easy to get wrong in ways nothing else would catch, and each has a control
that must fail loudly:

* a Gibbs kernel satisfies detailed balance **trivially** — it draws from the exact conditional —
  so the check with teeth is *stationarity of the enumerated kernel*, against a target far from
  uniform, with kernels built the three plausible wrong ways;
* the conditional is assembled from *differences* against the current value, so the current value's
  own entry is an exact zero and the candidate list must **include** it. Dropping it (reusing the
  Metropolis candidate list) leaves a chain that can never stay put, which still samples something
  plausible-looking;
* a frozen label reads as perfectly converged — zero variance means a *perfect* ESS and R-hat
  1.000 — and an exact-Gibbs coordinate reports ``discrete_accept_prob`` 1.00 by construction, so
  that column can no longer catch it. Hence the explicit move and distinct-value assertions.

The bit-identity guard is the other half: a model whose every parameter is Metropolis-updated must
be unchanged, and the golden draws in ``data/golden_discrete.npz`` were captured before the old
code path was deleted (see ``_golden_discrete_capture.py``).
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

import mimcs
from mimcs.adaptation import DiscreteMarginalAdaptation, RobbinsMonroStepSize
from mimcs.hmc import NUTS
from mimcs.model import IntegerParameter, Model
from mimcs.samplers import (DiscreteMetropolisWithinGibbs, StaticContinuous,
                            make_sampler_class)
from mimcs.samplers.discrete_updates import (EXACT_MAX_VALUES, EXACT_MAX_VALUES_ELEMENTWISE,
                                             EXACT_MIN_VALUES, ExactGibbsUpdate,
                                             MetropolisUpdate, build_discrete_updaters)
from mimcs.samplers.gibbs import only_in_scan_components, restriction_plan

from test_discrete import _binary_model, _exact_pmf
from test_discrete_adaptation import _categorical_model
from test_discrete_restricted import LOOP, SCAN, _data

GIBBS_ONLY = make_sampler_class(DiscreteMetropolisWithinGibbs, StaticContinuous)
NUTS_GIBBS = make_sampler_class(RobbinsMonroStepSize, DiscreteMetropolisWithinGibbs, NUTS)
ADAPT_GIBBS = make_sampler_class(DiscreteMarginalAdaptation, DiscreteMetropolisWithinGibbs,
                                 StaticContinuous)


# --------------------------------------------------------------------------- #
# 1. the kernel, by enumeration                                               #
# --------------------------------------------------------------------------- #

def _gibbs_kernel(logpi, variant="correct"):
    """The single-coordinate exact-Gibbs kernel, built by enumeration rather than sampled.

    The correct kernel is ``K[a, b] = pi_b``, independent of ``a``. Each control is a way the
    implementation could plausibly go wrong while still producing a valid-looking stochastic
    matrix.
    """
    n = len(logpi)
    K = np.zeros((n, n))
    for a in range(n):
        d = logpi - logpi[a]                       # what _discrete_delta returns; d[a] == 0
        if variant == "excluded-current":
            # the Metropolis candidate list reused: `cur` is not a candidate, so the chain can
            # never stay put
            w = np.exp(d)
            w[a] = 0.0
        elif variant == "inverted":
            w = np.exp(-d)
        elif variant == "off-by-one":
            # clipped to ni - 2 as the Metropolis arm does, making the last candidate unreachable
            w = np.exp(d)
            last = (a + n - 1) % n
            w[last] = 0.0
        else:
            w = np.exp(d)
        K[a] = w / w.sum()
    return K


@pytest.mark.parametrize("variant, invariant", [
    ("correct", True),
    ("excluded-current", False),
    ("inverted", False),
    ("off-by-one", False),
])
def test_the_exact_gibbs_kernel_is_stationary(variant, invariant):
    """Detailed balance is trivial for a Gibbs kernel, so stationarity is the check that bites."""
    rng = np.random.default_rng(0)
    n = 5
    logpi = rng.normal(size=n) * 3.0               # strongly non-uniform, or this is vacuous
    pi = np.exp(logpi - logpi.max())
    pi /= pi.sum()
    assert pi.max() / pi.min() > 20

    K = _gibbs_kernel(logpi, variant)
    assert np.allclose(K.sum(axis=1), 1.0)         # a valid kernel either way
    stat = np.max(np.abs(pi @ K - pi))
    flux = pi[:, None] * K
    bal = np.max(np.abs(flux - flux.T))
    if invariant:
        assert stat < 1e-12 and bal < 1e-12, (stat, bal)
    else:
        assert stat > 1e-2, stat


def test_the_current_value_must_be_a_candidate():
    """Spelled out on its own because it is the one difference from the Metropolis candidate list,
    and getting it wrong costs a chain that can never stay put -- which still mixes, and still
    produces a plausible marginal."""
    logpi = np.array([0.0, -3.0, -5.0])            # state 0 should be held ~94% of the time
    K = _gibbs_kernel(logpi, "correct")
    assert K[0, 0] == pytest.approx(np.exp(0) / np.sum(np.exp(logpi)), rel=1e-12)
    assert _gibbs_kernel(logpi, "excluded-current")[0, 0] == 0.0


# --------------------------------------------------------------------------- #
# 2. the conditional the sampler actually builds                              #
# --------------------------------------------------------------------------- #

def _sampler(src, method="exact", seed=0, **kw):
    model = mimcs.compile_model(src, data=_data())
    return model, NUTS_GIBBS(model, model.default_sample(), seed=seed,
                             discrete_update={"z": method}, **kw)


def _delta_conditional(s, st, z, i, plan):
    """softmax of ``_discrete_delta`` at every candidate -- what ExactGibbsUpdate draws from."""
    cur = z[:, i]
    d = np.array([float(s._discrete_delta(st, s._sweep_context(st), z, "z", i, cur,
                                          jnp.asarray([v]), plan)[0]) for v in (1, 2, 3)])
    w = np.exp(d - d.max())
    return w / w.sum()


def _full_conditional(s, st, z, i):
    lp = np.array([float(s._discrete_log_prob(st, z.at[:, i].set(v).reshape(-1))[0])
                   for v in (1, 2, 3)])
    w = np.exp(lp - lp.max())
    return w / w.sum()


@pytest.mark.parametrize("src, name", [(SCAN, "scan"), (LOOP, "loop")])
def test_the_delta_built_conditional_equals_the_full_conditional(src, name):
    """The identity the whole method rests on: a softmax is shift-invariant, so differences
    against the current value are sufficient. Checked on both the elementwise (`fast`) plan and
    the whole-density (`slow`) one."""
    _, s = _sampler(src)
    st = s.state
    plan = s._restricted(force=True)["z"]
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(3):
        z = jnp.asarray(rng.integers(1, 4, size=(1, 20)), dtype=jnp.int32)
        for i in range(20):
            worst = max(worst, np.max(np.abs(_delta_conditional(s, st, z, i, plan)
                                             - _full_conditional(s, st, z, i))))
    assert worst < 1e-4, (name, worst)


def test_dropping_the_reading_component_breaks_the_conditional():
    """The control for the test above: it is not agreeing trivially."""
    _, s = _sampler(SCAN)
    st = s.state
    z = jnp.asarray(np.full((1, 20), 2), dtype=jnp.int32)
    broken = ([], ["prior"])                       # `prior` does not read z; `lik` is dropped
    worst = max(np.max(np.abs(_delta_conditional(s, st, z, i, broken)
                              - _full_conditional(s, st, z, i))) for i in range(5))
    assert worst > 1e-3, worst


def test_the_delta_indexes_within_the_parameter_not_the_flat_block():
    """A latent bug this feature exposed. ``_discrete_delta``'s ``index`` is the coordinate's
    position within *its own* parameter, but it coincides with the model's flat index for the
    **first** discrete parameter --- which is every model that had a restriction plan before this.
    A second parameter then indexes past the end of its own array, and ``.at[i].set`` *clamps*
    rather than raising, so the wrong element moves and nothing anywhere reports it.

    Checked directly rather than only through the mixed-model posterior, so a failure says which
    thing is wrong.
    """
    model, lp = _two_parameter_model()
    s = GIBBS_ONLY(model, model.default_sample(), seed=0,
                   discrete_update={"z": "exact", "w": "exact"})
    st = s.state
    z = jnp.asarray([[1, 2, 3, 4]], dtype=jnp.int32)      # z = (1,2), w = (3,4)
    plans = s._restricted(force=True)
    # Move w's *second* coordinate: flat index 3, within-parameter index 1.
    cur, prop = jnp.asarray([4]), jnp.asarray([1])
    got = float(s._discrete_delta(st, s._sweep_context(st), z, "w", 1, cur, prop, plans["w"])[0])
    want = float(s._discrete_log_prob(st, jnp.asarray([1, 2, 3, 1]))[0]
                 - s._discrete_log_prob(st, jnp.asarray([1, 2, 3, 4]))[0])
    assert got == pytest.approx(want, abs=1e-4), (got, want)
    # The control: the flat index 3 would clamp into w's length-2 array and move w[0] instead.
    wrong = float(s._discrete_delta(st, s._sweep_context(st), z, "w", 3, cur, prop, plans["w"])[0])
    assert abs(wrong - want) > 1e-3, (wrong, want)


@pytest.mark.parametrize("src", [SCAN, LOOP])
def test_the_delta_at_the_current_value_is_exactly_zero(src):
    """``ExactGibbsUpdate`` hardcodes the current value's logit as an exact zero. That is only
    right while the delta hook returns exactly zero for a no-op move -- a future Hastings-like or
    Jacobian term in that hook would tilt the conditional silently."""
    _, s = _sampler(src)
    st = s.state
    z = st.discrete.reshape(1, -1)
    plan = s._restricted(force=True)["z"]
    for i in (0, 5, 19):
        cur = z[:, i]
        d = s._discrete_delta(st, s._sweep_context(st), z, "z", i, cur, cur, plan)
        assert float(d[0]) == 0.0


# --------------------------------------------------------------------------- #
# 3. the sampler samples the right thing                                      #
# --------------------------------------------------------------------------- #

def _run(model, n=30000, seed=0, **kw):
    s = GIBBS_ONLY(model, model.default_sample(), seed=seed, **kw)
    s.initialize()
    s.warmup(200)
    s.sample(n)
    return s


def test_exact_gibbs_recovers_a_known_per_coordinate_marginal():
    model, exact = _categorical_model()            # 4 independent 4-valued coordinates
    s = _run(model, discrete_update={"z": "exact"})
    z = np.asarray(s.get_discrete_flat())
    emp = np.stack([[np.mean(z[:, j] == v) for v in (1, 2, 3, 4)] for j in range(z.shape[1])])
    assert exact.max() / exact.min() > 10          # non-vacuity: far from uniform
    assert np.max(np.abs(emp - exact)) < 0.012, np.max(np.abs(emp - exact))
    # ... and the labels must actually move. A frozen coordinate has zero variance and so a
    # *perfect* ESS -- and an exact-Gibbs coordinate reports acceptance 1.00 by construction, so
    # `discrete_moves` is the only column left that can catch it.
    assert int(np.sum(s.diagnostics()["discrete_moves"])) > 0.1 * len(z)
    assert all(len(np.unique(z[:, j])) == 4 for j in range(z.shape[1]))


def test_a_perturbed_density_fails_the_same_comparison():
    """The control: the empirical marginal tracks *this* density, not merely some plausible
    distribution over the same support."""
    model, exact = _categorical_model()
    scaled = Model([], {"p": lambda v: 0.5 * model.log_prob_fns["p"](v)},
                   discrete_parameters=model.discrete_parameters)
    s = _run(scaled, discrete_update={"z": "exact"})
    z = np.asarray(s.get_discrete_flat())
    emp = np.stack([[np.mean(z[:, j] == v) for v in (1, 2, 3, 4)] for j in range(z.shape[1])])
    assert np.max(np.abs(emp - exact)) > 0.05


def test_exact_gibbs_recovers_an_exactly_enumerable_joint():
    """Coupled coordinates, so the conditional genuinely depends on the other labels."""
    m, lp = _binary_model()
    s = _run(m, discrete_update={"z": "exact"})
    states, exact = _exact_pmf(lp, 3)
    draws = np.asarray(s.get_discrete_flat())
    key = draws @ np.array([4, 2, 1])
    emp = np.bincount(key, minlength=8) / len(key)
    assert exact.max() / exact.min() > 20
    assert np.max(np.abs(emp - exact)) < 0.01, np.max(np.abs(emp - exact))
    assert len(np.unique(key)) == 8


def test_a_degenerate_support_is_a_harmless_no_op():
    model = Model([], {"p": lambda v: jnp.sum(v["z"].astype(float))},
                  discrete_parameters=[IntegerParameter("z", (2,), lower=3, upper=3)])
    s = _run(model, n=200, discrete_update={"z": "exact"})
    assert int(np.sum(s.diagnostics()["discrete_moves"])) == 0
    assert np.all(np.asarray(s.get_discrete_flat()) == 3)


# --------------------------------------------------------------------------- #
# 4. mixing the two methods in one model                                      #
# --------------------------------------------------------------------------- #

def _two_parameter_model(seed=0):
    """One 3-valued and one 4-valued parameter, coupled, so neither can be sampled in isolation."""
    rng = np.random.default_rng(seed)
    a = jnp.asarray(rng.normal(size=(2, 3)) * 1.5, float)
    b = jnp.asarray(rng.normal(size=(2, 4)) * 1.5, float)
    c = jnp.asarray(rng.normal(size=(3, 4)) * 1.2, float)

    def lp(v):
        zi, wi = v["z"] - 1, v["w"] - 1
        return (jnp.sum(jnp.take_along_axis(a, zi[:, None], axis=1))
                + jnp.sum(jnp.take_along_axis(b, wi[:, None], axis=1))
                + jnp.sum(c[zi, wi]))

    model = Model([], {"p": lp},
                  discrete_parameters=[IntegerParameter("z", (2,), lower=1, upper=3),
                                       IntegerParameter("w", (2,), lower=1, upper=4)])
    return model, lp


def _enumerate_joint(lp):
    states, logp = [], []
    for z0 in (1, 2, 3):
        for z1 in (1, 2, 3):
            for w0 in (1, 2, 3, 4):
                for w1 in (1, 2, 3, 4):
                    states.append((z0, z1, w0, w1))
                    logp.append(float(lp({"z": jnp.asarray([z0, z1]),
                                          "w": jnp.asarray([w0, w1])})))
    logp = np.array(logp)
    w = np.exp(logp - logp.max())
    return np.array(states), w / w.sum()


@pytest.mark.parametrize("methods", [
    {"z": "exact", "w": "metropolis"},
    {"z": "metropolis", "w": "exact"},
    {"z": "exact", "w": "exact"},
    {},
])
def test_a_mixed_model_samples_the_enumerable_joint(methods):
    """The test that the forced delta path, the shared RNG index and the two updaters interleaving
    are right *together*: one parameter on each method, over a coupled target."""
    model, lp = _two_parameter_model()
    s = _run(model, n=40000, discrete_update=methods)
    draws = np.asarray(s.get_discrete_flat())
    states, exact = _enumerate_joint(lp)
    idx = {tuple(st): i for i, st in enumerate(states)}
    emp = np.bincount([idx[tuple(d)] for d in draws], minlength=len(states)) / len(draws)
    assert exact.max() / exact.min() > 20
    assert np.max(np.abs(emp - exact)) < 0.01, np.max(np.abs(emp - exact))


def test_a_mixed_model_allocates_a_table_only_for_the_metropolis_parameter():
    model, _ = _two_parameter_model()
    s = ADAPT_GIBBS(model, model.default_sample(), seed=0,
                    discrete_update={"z": "exact", "w": "metropolis"})
    s.initialize()
    s.warmup(60)
    assert set(s._dm_hat) == {"w"}
    # z keeps its uniform entry forever, which an ExactGibbsUpdate never reads
    assert np.allclose(np.asarray(s.state.discrete_proposal_params["z"]), 1.0 / 3.0)
    assert not np.allclose(np.asarray(s.state.discrete_proposal_params["w"]), 1.0 / 4.0)


def test_the_ownership_filter_is_not_a_no_op():
    """The control for the test above: built all-Metropolis, the same model learns both tables."""
    model, _ = _two_parameter_model()
    s = ADAPT_GIBBS(model, model.default_sample(), seed=0)
    s.initialize()
    s.warmup(60)
    assert set(s._dm_hat) == {"z", "w"}


# --------------------------------------------------------------------------- #
# 4b. under tempering, with no override of its own                            #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("method", ["metropolis", "exact"])
def test_exact_gibbs_samples_the_enumerable_target_under_tempering(method):
    """The claim that makes the delta-only construction worth it: because the conditional is
    assembled from ``_discrete_delta``, and the tempered sampler already overrides *that* for
    per-rung betas, exact Gibbs needs **no parallel-tempering code of its own**. So this must hold
    without anything in ``mimcs/pt`` having been told the method exists.

    The Metropolis arm is run beside it as the reference, and the target is far from uniform so a
    density-blind sweep could not pass.
    """
    from mimcs.model import EuclideanParameter
    from mimcs.pt import parallel_tempering
    rng = np.random.default_rng(0)
    a = jnp.asarray(rng.normal(size=(2, 3)) * 1.5, float)
    c = jnp.asarray(rng.normal(size=(3, 3)) * 1.2, float)

    def lp(v):
        zi = v["z"] - 1
        return (jnp.sum(jnp.take_along_axis(a, zi[:, None], axis=1)) + c[zi[0], zi[1]]
                - 0.5 * jnp.sum(v["x"] ** 2))

    model = Model([EuclideanParameter("x")], {"p": lp},
                  discrete_parameters=[IntegerParameter("z", (2,), lower=1, upper=3)])
    states = [(i, j) for i in (1, 2, 3) for j in (1, 2, 3)]
    logw = np.array([float(lp({"z": jnp.asarray(s), "x": jnp.zeros(1)})) for s in states])
    exact = np.exp(logw - logw.max())
    exact /= exact.sum()
    assert exact.max() / exact.min() > 20                   # non-vacuity

    s = parallel_tempering(model, n_temperatures=3, seed=0, discrete_update={"z": method})
    s.initialize()
    s.warmup(500)
    s.sample(30000)
    idx = {t: i for i, t in enumerate(states)}
    draws = np.asarray(s.get_discrete_flat())               # the cold chain
    emp = np.bincount([idx[tuple(r)] for r in draws], minlength=9) / len(draws)
    assert np.max(np.abs(emp - exact)) < 0.012, np.max(np.abs(emp - exact))


# --------------------------------------------------------------------------- #
# 5. the RNG stream                                                           #
# --------------------------------------------------------------------------- #

def test_exact_gibbs_reads_the_proposal_stream_and_not_the_acceptance_one():
    """The strongest available statement that the layout is preserved and the method consumes only
    what it claims. Corrupting the acceptance draws must be invisible to an all-exact model and
    must move an all-Metropolis one."""
    model, _ = _two_parameter_model()

    def draws(methods, corrupt):
        s = GIBBS_ONLY(model, model.default_sample(), seed=0, discrete_update=methods)
        s.initialize()
        if corrupt:
            orig = s._rng_buffer.next

            def patched():
                raw = dict(orig())
                # zeros make log(u) = -inf, so a Metropolis step rejects everything
                raw["discrete_accept"] = jnp.zeros_like(raw["discrete_accept"])
                return raw
            s._rng_buffer.next = patched
        s.warmup(50)
        s.sample(300)
        return np.asarray(s.get_discrete_flat())

    ex = {"z": "exact", "w": "exact"}
    assert np.array_equal(draws(ex, False), draws(ex, True))            # never read
    mh = {"z": "metropolis", "w": "metropolis"}
    assert not np.array_equal(draws(mh, False), draws(mh, True))        # the control


def test_the_update_method_does_not_change_the_draw_components():
    """Adding or dropping a draw component renumbers every other component's stream in the whole
    library, so the two methods must request exactly the same ones."""
    model, _ = _two_parameter_model()

    def comps(methods):
        s = GIBBS_ONLY(model, model.default_sample(), seed=0, discrete_update=methods)
        return [(c.name, c.shape) for c in s._draw_components]

    assert comps({}) == comps({"z": "exact", "w": "exact"})


# --------------------------------------------------------------------------- #
# 6. compile time stays flat in the support width                             #
# --------------------------------------------------------------------------- #

def test_the_candidate_axis_does_not_unroll():
    """``ExactGibbsUpdate`` vmaps over candidates rather than looping in Python. A Python loop
    would put ``n_i - 1`` copies of the density into the ``fori_loop`` body, so the jaxpr would
    grow with the support -- which is exactly what makes the wide elementwise case affordable."""
    def eqns(k):
        logits = jnp.asarray(np.random.default_rng(0).normal(size=(2, k)) * 1.5, float)
        model = Model([], {"p": lambda v: jnp.sum(
            jnp.take_along_axis(logits, (v["z"] - 1)[:, None], axis=1))},
            discrete_parameters=[IntegerParameter("z", (2,), lower=1, upper=k)])
        s = GIBBS_ONLY(model, model.default_sample(), seed=0, discrete_update={"z": "exact"})
        return len(jax.make_jaxpr(s.kernel)(s.state).jaxpr.eqns)

    small, large = eqns(4), eqns(16)
    assert large < 1.3 * small, (small, large)


# --------------------------------------------------------------------------- #
# 7. nothing else moved                                                       #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", ["binary", "loop", "scan"])
def test_a_metropolis_only_model_is_bit_for_bit_unchanged(name):
    """The regression guard. Captured on ``dev`` *before* the per-parameter refactor deleted the
    code path they came from, which is why the seed-pinned suites cannot make this claim on their
    own -- there is no longer an old path to compare against.

    Seed and ``buffer_size`` are both pinned: ``buffer_size`` is a memory knob but is **not**
    stream-neutral, so draws agree only up to the first refill.
    """
    import pathlib
    golden = np.load(pathlib.Path(__file__).parent / "data" / "golden_discrete.npz")
    if name == "binary":
        m, _ = _binary_model()
        s = GIBBS_ONLY(m, m.default_sample(), seed=0, buffer_size=1024)
    else:
        m = mimcs.compile_model(SCAN if name == "scan" else LOOP, data=_data())
        s = NUTS_GIBBS(m, m.default_sample(), seed=0, buffer_size=1024)
    s.initialize()
    s.warmup(200)
    s.sample(500)
    assert np.array_equal(np.asarray(s.get_discrete_flat())[:20], golden[f"{name}_z"])
    assert np.array_equal(np.asarray(s.get_samples_flat())[-1], golden[f"{name}_x"])


# --------------------------------------------------------------------------- #
# 8. the units themselves                                                     #
# --------------------------------------------------------------------------- #

def test_the_default_is_metropolis_for_every_parameter():
    model, _ = _two_parameter_model()
    for methods in (None, {}, {"z": "metropolis"}):
        us = build_discrete_updaters(model, methods)
        assert [type(u) for u in us] == [MetropolisUpdate, MetropolisUpdate]
        assert [u.name for u in us] == ["z", "w"]          # declaration order


def test_the_updaters_carry_their_own_slice_and_support():
    model, _ = _two_parameter_model()
    z, w = build_discrete_updaters(model, {"z": "exact"})
    assert isinstance(z, ExactGibbsUpdate) and isinstance(w, MetropolisUpdate)
    assert (z.start, z.size, z.n_values, z.lower) == (0, 2, 3, 1)
    assert (w.start, w.size, w.n_values, w.lower) == (2, 2, 4, 1)
    assert z.uses_proposal_table is False and w.uses_proposal_table is True
    assert z.forms_running_total is False and w.forms_running_total is True


@pytest.mark.parametrize("methods, message", [
    ({"nope": "exact"}, "not discrete parameters"),
    ({"z": "gibbs"}, "unknown discrete update method"),
])
def test_a_bad_method_map_says_which_mistake_it_is(methods, message):
    model, _ = _two_parameter_model()
    with pytest.raises(ValueError, match=message):
        build_discrete_updaters(model, methods)


def test_exact_gibbs_on_a_binary_parameter_warns_about_peskun(caplog):
    """Not refused -- it is worse, not wrong -- but it must say so, because a kernel that is merely
    less efficient is invisible in every diagnostic the library prints."""
    m, _ = _binary_model()
    with caplog.at_level("WARNING"):
        build_discrete_updaters(m, {"z": "exact"})
    assert "Peskun" in caplog.text


# --------------------------------------------------------------------------- #
# 9. the predicate the factory rule turns on                                  #
# --------------------------------------------------------------------------- #

def test_only_in_scan_components_distinguishes_the_two_spellings():
    scan = mimcs.compile_model(SCAN, data=_data())
    loop = mimcs.compile_model(LOOP, data=_data())
    assert only_in_scan_components(scan, "z") is True
    assert only_in_scan_components(loop, "z") is False


def test_a_hand_written_model_is_never_read_as_elementwise():
    """The trap: ``restriction_plan`` returns ``None`` for a model with no recorded reads and no
    scan component, and ``None`` must not be read as "nothing slow"."""
    m, _ = _binary_model()
    assert restriction_plan(m, "z") is None
    assert only_in_scan_components(m, "z") is False


def test_the_thresholds_are_independent_constants():
    """They coincide in value with ``WIDE_SUPPORT`` but price unrelated trades, so nothing should
    have defined one in terms of another. The *floor* is the one worth pinning: it exists because
    narrow supports mix better under Metropolis, which is the opposite of a cost argument."""
    assert 2 < EXACT_MIN_VALUES <= EXACT_MAX_VALUES <= EXACT_MAX_VALUES_ELEMENTWISE
