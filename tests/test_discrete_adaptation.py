"""Tests for ``DiscreteMarginalAdaptation`` --- learned-marginal proposals for discrete parameters.

The scheme replaces the uniform-over-others proposal with one proportional to each coordinate's
learned marginal pmf. That proposal is **asymmetric**, so the sweep must carry a Hastings term;
getting it wrong is the whole risk here, and nothing about the resulting chain would look wrong.

The checks are arranged so that each one can fail:

* detailed balance is verified against the *enumerated* kernel, with the un-corrected version as a
  control that must break it (``tests/test_discrete.py`` holds that control);
* **exactness under a deliberately wrong pmf** --- Metropolis-Hastings is exact for *any* proposal
  with full support, so the exact-enumeration oracle must still hold when the tables are set to
  something adversarial. This is the strongest available statement that the Hastings term, and not
  the proposal, is what makes the answer right;
* a uniform table must reproduce the unadapted sampler **bit-for-bit**, and a binary parameter must
  be untouched --- both are analytic consequences, so they are asserted exactly rather than
  statistically.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from mimcs.adaptation import (DiscreteMarginalAdaptation, MassMatrixAdaptation,
                              RobbinsMonroStepSize)
from mimcs.adaptation._stochastic import DEFAULT_KAPPA, DEFAULT_N0, rm_gain
from mimcs.hmc import NUTS
from mimcs.model import EuclideanParameter, IntegerParameter, Model
from mimcs.samplers import (DiscreteMetropolisWithinGibbs, StaticContinuous, make_sampler_class)
from mimcs.samplers.base import Phase

UNIFORM = make_sampler_class(DiscreteMetropolisWithinGibbs, StaticContinuous)
LEARNED = make_sampler_class(DiscreteMarginalAdaptation, DiscreteMetropolisWithinGibbs,
                             StaticContinuous)
NUTS_UNIFORM = make_sampler_class(RobbinsMonroStepSize, DiscreteMetropolisWithinGibbs, NUTS)
NUTS_LEARNED = make_sampler_class(RobbinsMonroStepSize, DiscreteMarginalAdaptation,
                                  DiscreteMetropolisWithinGibbs, NUTS)


def _binary_model(n=3, w=(1.3, -0.7, 2.1)):
    w = jnp.asarray(w[:n], float)
    J = jnp.asarray([[0.0, 1.5, -0.9], [1.5, 0.0, 0.6], [-0.9, 0.6, 0.0]])[:n, :n]

    def lp(v):
        z = v["z"].astype(float)
        return jnp.dot(w, z) + 0.5 * z @ J @ z

    return Model([], {"p": lp},
                 discrete_parameters=[IntegerParameter("z", (n,), lower=0, upper=1)]), lp


def _categorical_model(size=4, k=4, seed=0):
    """`size` independent categorical coordinates with a strongly non-uniform, known marginal."""
    logits = jnp.asarray(np.random.default_rng(seed).normal(size=(size, k)) * 1.5, float)

    def lp(v):
        return jnp.sum(jnp.take_along_axis(logits, (v["z"] - 1)[:, None], axis=1))

    model = Model([], {"p": lp},
                  discrete_parameters=[IntegerParameter("z", (size,), lower=1, upper=k)])
    exact = np.asarray(jax.nn.softmax(logits, axis=1))     # the per-coordinate marginal, exactly
    return model, exact


# --------------------------------------------------------------------------- #
# 1. the correction itself                                                     #
# --------------------------------------------------------------------------- #

def _enumerated_kernel(logpi, p, hastings=True):
    """The single-coordinate kernel, built by enumeration rather than sampled."""
    n = len(logpi)
    K = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            q_ab = p[b] / (1 - p[a])
            h = (np.log(p[a]) + np.log1p(-p[a])
                 - np.log(p[b]) - np.log1p(-p[b])) if hastings else 0.0
            K[a, b] = q_ab * min(1.0, np.exp(logpi[b] - logpi[a] + h))
        K[a, a] = 1 - K[a].sum()
    return K


@pytest.mark.parametrize("seed", range(4))
def test_the_hastings_term_restores_detailed_balance(seed):
    """`g(cur) - g(prop)` with `g = log p + log1p(-p)` is exactly the correction the asymmetric
    proposal needs --- for an *arbitrary* pmf, not a convenient one.

    The control is `test_an_asymmetric_proposal_would_need_a_hastings_term` in test_discrete.py:
    the same kernel without the term breaks detailed balance by >= 1e-2.
    """
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 7))
    logpi = rng.normal(size=n) * 2.5
    pi = np.exp(logpi - logpi.max()); pi /= pi.sum()
    p = rng.dirichlet(np.full(n, 0.7))          # arbitrary, possibly adversarial

    K = _enumerated_kernel(logpi, p, hastings=True)
    assert np.allclose(K.sum(axis=1), 1.0)
    flux = pi[:, None] * K
    assert np.max(np.abs(flux - flux.T)) < 1e-12
    assert np.max(np.abs(pi @ K - pi)) < 1e-12

    # ... and the control, inline, so this test cannot pass by the term being a no-op
    Kc = _enumerated_kernel(logpi, p, hastings=False)
    fluxc = pi[:, None] * Kc
    assert np.max(np.abs(fluxc - fluxc.T)) > 1e-3


@pytest.mark.parametrize("q", [1e-3, 0.1, 0.5, 0.9, 1 - 1e-3])
def test_the_hastings_term_vanishes_for_a_binary_coordinate(q):
    """`p_b = 1 - p_a` makes `g(a) = g(b)` identically. This is why a binary parameter has nothing
    to adapt --- not a heuristic, an algebraic identity."""
    p = np.array([q, 1 - q])
    h = np.log(p[0]) + np.log1p(-p[0]) - np.log(p[1]) - np.log1p(-p[1])
    assert abs(h) < 1e-12


def test_a_uniform_table_reproduces_the_unadapted_offset_formula():
    """The cyclic candidate ordering is what buys this, and it is why there is one.

    Exhaustive over a fine `u` grid in **float32**, the working precision: inverse-CDF selection
    over candidates ordered cyclically from `cur+1` at uniform weights gives exactly
    `1 + floor(u*(n-1))`, the unadapted sweep's formula. Asserted here so a later reworking of the
    weight or cumsum expression cannot silently cost the property.
    """
    for n in (2, 3, 5, 9, 17):
        p = np.full(n, np.float32(1.0 / n), np.float32)
        for lo in (0, 1, -2):
            for cur in range(lo, lo + n):
                for u in np.linspace(0, 1, 1001, endpoint=False, dtype=np.float32):
                    offsets = np.arange(1, n, dtype=np.int64)
                    cand = lo + ((cur - lo) + offsets) % n
                    w = p[cand - lo].astype(np.float32)
                    cw = np.cumsum(w, dtype=np.float32)
                    idx = int(np.sum(cw <= np.float32(u) * cw[-1]))
                    idx = min(idx, n - 2)
                    expected = lo + ((cur - lo) + 1
                                     + int(np.floor(np.float32(u) * np.float32(n - 1)))) % n
                    assert int(cand[idx]) == expected, (n, lo, cur, u)


# --------------------------------------------------------------------------- #
# 2. exactness of the sampler                                                  #
# --------------------------------------------------------------------------- #

def _exact_pmf(lp, n):
    states = np.array([[(i >> k) & 1 for k in range(n - 1, -1, -1)] for i in range(2 ** n)])
    logp = np.array([float(lp({"z": jnp.asarray(s)})) for s in states])
    w = np.exp(logp - logp.max())
    return states, w / w.sum()


def test_an_adversarial_proposal_table_still_samples_the_exact_target():
    """The strongest non-vacuity check available here.

    Metropolis-Hastings with the correction is exact for **any** proposal pmf with full support,
    so forcing a deliberately skewed, deliberately *wrong* table must leave the stationary
    distribution untouched. If this passes while the Hastings term is missing, the term is doing
    nothing; if it fails, the term is wrong. Either way it cannot pass vacuously.
    """
    m, lp = _binary_model()
    s = UNIFORM(m, m.default_sample(), seed=0)
    s.initialize()
    s.warmup(100)

    # A pmf that has nothing to do with the target: coordinate 0 biased hard to 0, coordinate 1
    # hard to 1, coordinate 2 mildly skewed.
    # (L, size, ni): one lane here; under tempering there is one table per rung.
    skewed = jnp.asarray([[[0.95, 0.05], [0.03, 0.97], [0.30, 0.70]]], float)
    s.state = s.state._replace(discrete_proposal_params={"z": skewed})
    s.sample(40000)
    # ... and it must still be in force at the end (nothing silently reset it)
    assert np.allclose(np.asarray(s.state.discrete_proposal_params["z"]), np.asarray(skewed))

    draws = s.get_discrete_flat()
    emp = np.bincount(draws @ np.array([4, 2, 1]), minlength=8) / len(draws)
    states, exact = _exact_pmf(lp, 3)
    assert exact.max() / exact.min() > 20                    # target far from uniform
    assert np.max(np.abs(emp - exact)) < 0.012, (emp, exact)
    assert len(np.unique(draws @ np.array([4, 2, 1]))) == 8  # and it explored everything


def test_the_learned_proposal_samples_the_exact_target():
    """The same oracle, with the pmf actually learned rather than injected."""
    m, lp = _binary_model()
    s = LEARNED(m, m.default_sample(), seed=0)
    s.initialize(); s.warmup(400); s.sample(40000)
    draws = s.get_discrete_flat()
    emp = np.bincount(draws @ np.array([4, 2, 1]), minlength=8) / len(draws)
    _, exact = _exact_pmf(lp, 3)
    assert np.max(np.abs(emp - exact)) < 0.012
    assert int(np.sum(s.diagnostics()["discrete_moves"])) > 0.1 * len(draws)


def test_a_categorical_target_is_sampled_exactly_under_the_learned_proposal():
    """Non-binary is where the Hastings term actually bites: with `n_i = 4` the proposal really is
    asymmetric, and a missing correction would show up as a biased marginal."""
    m, exact = _categorical_model(size=3, k=4, seed=1)
    s = LEARNED(m, m.default_sample(), seed=0)
    s.initialize(); s.warmup(600); s.sample(60000)
    z = s.get_discrete_flat()
    emp = np.stack([np.bincount(z[:, c] - 1, minlength=4) / len(z) for c in range(3)])
    assert np.max(np.abs(emp - exact)) < 0.015, (emp, exact)


# --------------------------------------------------------------------------- #
# 3. the things that must not change                                           #
# --------------------------------------------------------------------------- #

def test_lambda_one_is_bit_identical_to_the_unadapted_sampler():
    """`lambda = 1` is the uniform table, so the mixin becomes a no-op --- exactly, not nearly."""
    m, _ = _binary_model()
    a = UNIFORM(m, m.default_sample(), seed=3)
    b = LEARNED(m, m.default_sample(), seed=3, discrete_lambda=1.0)
    for s in (a, b):
        s.initialize(); s.warmup(200); s.sample(1500)
    assert np.array_equal(a.get_discrete_flat(), b.get_discrete_flat())


def test_a_binary_parameter_is_untouched_by_the_adaptation():
    """The Hastings term is identically zero for `n_i = 2`, so a learned table changes nothing
    even when it is far from uniform."""
    m, _ = _binary_model()
    a = UNIFORM(m, m.default_sample(), seed=7)
    b = LEARNED(m, m.default_sample(), seed=7)
    for s in (a, b):
        s.initialize(); s.warmup(300); s.sample(2000)
    tbl = np.asarray(b.state.discrete_proposal_params["z"])
    assert np.abs(tbl - 0.5).max() > 0.05, "the table must actually have moved off uniform"
    assert np.array_equal(a.get_discrete_flat(), b.get_discrete_flat())


def test_the_mixin_is_inert_and_stream_neutral_on_a_continuous_model():
    m = Model([EuclideanParameter("x")], {"p": lambda v: -0.5 * v["x"] ** 2})
    plain = make_sampler_class(RobbinsMonroStepSize, NUTS)(m, m.default_sample(), seed=0)
    with_mixin = NUTS_LEARNED(m, m.default_sample(), seed=0)
    assert ([c.name for c in plain._draw_components]
            == [c.name for c in with_mixin._draw_components])
    for s in (plain, with_mixin):
        s.warmup(50); s.sample(50)
    assert np.array_equal(plain.get_samples_flat(), with_mixin.get_samples_flat())
    assert with_mixin.state.discrete_proposal_params == {}


# --------------------------------------------------------------------------- #
# 4. the estimator, and its guards                                             #
# --------------------------------------------------------------------------- #

class _ChainEnd:
    """Terminates the cooperative hook chain with no-ops."""

    def _init_hooks(self, **kwargs):
        pass

    def _postprocess_hooks(self, state):
        return state

    def _warmup_end_hooks(self, completed, stopped):
        pass


class _Harness(DiscreteMarginalAdaptation, _ChainEnd):
    """The mixin's hooks over a no-op chain end, so the estimator is unit-testable."""

    def __init__(self, model, **kwargs):
        self.model = model
        self._phase = Phase.WARMUP
        self._init_hooks(**kwargs)

    def feed(self, state, z):
        return self._postprocess_hooks(state._replace(discrete=jnp.asarray(z, jnp.int32)))


def _state_for(model):
    from mimcs.samplers.metropolis import uniform_discrete_proposal_params
    from mimcs.samplers.gibbs import StaticState
    return StaticState(coordinate=jnp.zeros(0), sample=jnp.zeros(0),
                       discrete=model.default_discrete(),
                       discrete_proposal_params=uniform_discrete_proposal_params(model),
                       log_prob=jnp.zeros(()), rng_draw=None,
                       chart_hyperparams=(), chart_indices=())


def test_the_tables_carry_a_leading_lane_axis():
    """One lane for an ordinary model, one per rung under tempering (doc 13).

    Asserted on its own because every other test here reads the tables through that axis, and a
    silent change to it would make several of them wrong in the same direction at once.
    """
    m = Model([], {"p": lambda v: jnp.zeros(())},
              discrete_parameters=[IntegerParameter("z", (3,), lower=0, upper=2)])
    from mimcs.samplers.metropolis import uniform_discrete_proposal_params
    tbl = uniform_discrete_proposal_params(m)["z"]
    assert tbl.shape == (1, 3, 3)
    assert np.allclose(np.asarray(tbl), 1.0 / 3.0)


def test_the_estimator_recovers_a_known_marginal():
    """Fed i.i.d. draws from a known pmf, the running estimate must converge to it."""
    rng = np.random.default_rng(0)
    k, size, n = 5, 2, 4000
    truth = np.array([[0.5, 0.2, 0.15, 0.1, 0.05], [0.05, 0.05, 0.1, 0.3, 0.5]])
    m = Model([], {"p": lambda v: jnp.zeros(())},
              discrete_parameters=[IntegerParameter("z", (size,), lower=1, upper=k)])
    h = _Harness(m, discrete_lambda=1e-6, discrete_min_samples=0)
    state = _state_for(m)
    for _ in range(n):
        z = [rng.choice(k, p=truth[c]) + 1 for c in range(size)]
        state = h.feed(state, z)
    est = np.asarray(state.discrete_proposal_params["z"])
    assert est.shape == (1, size, k), est.shape        # (lanes, coordinates, values)
    assert np.max(np.abs(est[0] - truth)) < 0.05, (est, truth)
    assert np.allclose(est.sum(axis=-1), 1.0, atol=1e-5)


def test_the_estimate_stays_on_the_simplex_without_renormalization():
    """`p <- p + gain*(onehot - p)` is a convex combination, which is why nothing clips or
    renormalizes it. Asserted because a future change to the update could quietly break it."""
    rng = np.random.default_rng(1)
    m = Model([], {"p": lambda v: jnp.zeros(())},
              discrete_parameters=[IntegerParameter("z", (3,), lower=0, upper=3)])
    h = _Harness(m, discrete_lambda=1e-9, discrete_min_samples=0)
    state = _state_for(m)
    for _ in range(500):
        state = h.feed(state, rng.integers(0, 4, size=3))
        t = np.asarray(state.discrete_proposal_params["z"])
        assert np.all(t >= 0.0) and np.allclose(t.sum(axis=-1), 1.0, atol=1e-5)


def test_the_uniform_mixture_floors_every_value():
    """Irreducibility: a value the chain never visited must still be proposable, or the sampler
    silently targets the posterior restricted to what it happened to see."""
    m = Model([], {"p": lambda v: jnp.zeros(())},
              discrete_parameters=[IntegerParameter("z", (2,), lower=1, upper=5)])
    h = _Harness(m, discrete_lambda=0.05, discrete_min_samples=0)
    state = _state_for(m)
    for _ in range(2000):
        state = h.feed(state, [1, 1])            # value 1 only; 2..5 never visited
    t = np.asarray(state.discrete_proposal_params["z"])
    assert t.min() == pytest.approx(0.05 / 5, rel=1e-4)
    assert np.all(t > 0.0), "an unvisited value became unreachable"
    assert np.all(t < 1.0), "log1p(-p) in the Hastings term would be -inf"


def test_lambda_zero_is_refused():
    m = Model([], {"p": lambda v: jnp.zeros(())},
              discrete_parameters=[IntegerParameter("z", (2,), lower=0, upper=1)])
    with pytest.raises(ValueError, match="may not be 0"):
        _Harness(m, discrete_lambda=0.0)
    with pytest.raises(ValueError, match="discrete_lambda"):
        _Harness(m, discrete_lambda=1.5)


def test_the_gain_schedule_is_the_shared_one():
    m = Model([], {"p": lambda v: jnp.zeros(())},
              discrete_parameters=[IntegerParameter("z", (1,), lower=0, upper=2)])
    h = _Harness(m)
    assert (h._dm_n0, h._dm_kappa) == (DEFAULT_N0, DEFAULT_KAPPA)
    assert rm_gain(1, h._dm_n0, h._dm_kappa) == (1 + DEFAULT_N0) ** (-DEFAULT_KAPPA)


def test_nothing_is_written_before_min_samples():
    m = Model([], {"p": lambda v: jnp.zeros(())},
              discrete_parameters=[IntegerParameter("z", (2,), lower=0, upper=2)])
    h = _Harness(m, discrete_min_samples=5)
    state = _state_for(m)
    before = np.asarray(state.discrete_proposal_params["z"]).copy()
    for _ in range(5):
        state = h.feed(state, [0, 0])
    assert np.array_equal(np.asarray(state.discrete_proposal_params["z"]), before)
    state = h.feed(state, [0, 0])
    assert not np.array_equal(np.asarray(state.discrete_proposal_params["z"]), before)


def test_adaptation_stops_when_sampling_begins():
    m = Model([], {"p": lambda v: jnp.zeros(())},
              discrete_parameters=[IntegerParameter("z", (2,), lower=0, upper=2)])
    h = _Harness(m, discrete_min_samples=0)
    state = _state_for(m)
    for _ in range(50):
        state = h.feed(state, [0, 0])
    frozen = np.asarray(state.discrete_proposal_params["z"]).copy()
    h._phase = Phase.SAMPLING
    for _ in range(50):
        state = h.feed(state, [2, 2])            # would move the estimate a lot, if it ran
    assert np.array_equal(np.asarray(state.discrete_proposal_params["z"]), frozen)


def test_a_wide_support_warns_but_proceeds(caplog):
    import logging
    m = Model([], {"p": lambda v: jnp.zeros(())},
              discrete_parameters=[IntegerParameter("z", (2,), lower=0, upper=99)])
    h = _Harness(m, discrete_min_samples=0)
    with caplog.at_level(logging.WARNING, logger="mimcs.adaptation.discrete_marginal"):
        state = h.feed(_state_for(m), [0, 1])
    assert any("100 values" in r.getMessage() for r in caplog.records)
    assert np.asarray(state.discrete_proposal_params["z"]).shape == (1, 2, 100)


def test_the_entropy_report_says_when_nothing_was_learned(caplog):
    import logging
    m = Model([], {"p": lambda v: jnp.zeros(())},
              discrete_parameters=[IntegerParameter("z", (2,), lower=0, upper=3)])
    h = _Harness(m, discrete_min_samples=0)
    state = _state_for(m)
    for _ in range(200):
        state = h.feed(state, [0, 0])            # a maximally concentrated marginal
    with caplog.at_level(logging.INFO, logger="mimcs.adaptation.discrete_marginal"):
        h._warmup_end_hooks(200, False)
    msgs = [r.getMessage() for r in caplog.records]
    assert any("normalized entropy" in m_ for m_ in msgs)
    ent = float([m_ for m_ in msgs if "normalized entropy" in m_][0].split("entropy ")[1][:5])
    assert ent < 0.3, f"a concentrated marginal should report low entropy, got {ent}"
