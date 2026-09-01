"""Component- and coordinate-restricted recomputation in the Metropolis-within-Gibbs sweep.

The sweep used to re-evaluate the **whole** log density once per discrete coordinate. With a
`scan` component (see `tests/test_dsl_scan_component.py`) it evaluates only the components that
read the swept parameter, and for the elementwise ones only that coordinate's term.

Everything here is arranged so it can fail:

* the restricted delta is checked **against the full delta** it replaces, coordinate by
  coordinate, with a deliberately broken plan as the control;
* the same is done per rung under tempering, where the control is the tempting wrong spelling ---
  scaling one unweighted sum by a single global beta, which is right only if every component
  happens to be tempered;
* the posterior is checked against an **exactly enumerable** target, not against another sampler;
* and every model *without* a scan component must take the verbatim fallback and produce
  bit-identical draws, which is what makes this change safe for everything that came before.
"""

import itertools

import numpy as np
import jax.numpy as jnp
import pytest

from mimcs import compile_model
from mimcs.adaptation import DiscreteMarginalAdaptation, RobbinsMonroStepSize
from mimcs.hmc import NUTS
from mimcs.model import IntegerParameter, Model
from mimcs.pt import parallel_tempering
from mimcs.pt.lanes import per_temperature_potential
from mimcs.samplers import (DiscreteMetropolisWithinGibbs, StaticContinuous, make_sampler_class)

HEAD = """
data { int n; int k; array[n] real y; array[k] real w; }
parameters { ordered[k] mu; array[n] int<lower=1, upper=k> z; real<lower=0> sigma; }
"""
SCAN = HEAD + """
model prior { mu ~ normal(0, 10); sigma ~ lognormal(0, 1); }
model lik scan(z, y) { z ~ categorical(w); y ~ normal(mu[z], sigma); }
"""
LOOP = HEAD + """
model {
  mu ~ normal(0, 10); sigma ~ lognormal(0, 1);
  for (i in 1:n) { z[i] ~ categorical(w); y[i] ~ normal(mu[z[i]], sigma); }
}
"""
NUTS_GIBBS = make_sampler_class(RobbinsMonroStepSize, DiscreteMetropolisWithinGibbs, NUTS)


def _data(n=20, k=3, seed=0):
    rng = np.random.default_rng(seed)
    true_mu = 4.0 * (np.arange(k) - (k - 1) / 2.0)
    y = true_mu[rng.integers(0, k, n)] + rng.standard_normal(n)
    return {"n": n, "k": k, "y": y, "w": np.full(k, 1.0 / k)}


# --------------------------------------------------------------------------- #
# 1. the plan                                                                  #
# --------------------------------------------------------------------------- #

def test_the_plan_skips_a_component_that_does_not_read_the_parameter():
    """`prior` reads only `mu` and `sigma`, so its contribution to every acceptance ratio cancels
    exactly and it is never evaluated. `lik` is elementwise, so it is evaluated at one coordinate."""
    m = compile_model(SCAN, data=_data())
    s = NUTS_GIBBS(m, m.default_sample(), seed=0)
    fast, slow = s._restricted()["z"]
    assert list(fast) == ["lik"]
    assert list(slow) == []                            # `prior` is skipped, not merely not-fast


def test_a_model_without_a_scan_component_takes_the_verbatim_path():
    """`_restricted()` empty is the signal that nothing is gained --- and therefore that the sweep
    runs exactly the code it ran before, which is what keeps existing models bit-identical."""
    m = compile_model(LOOP, data=_data())
    s = NUTS_GIBBS(m, m.default_sample(), seed=0)
    assert s._restricted() == {}


def test_a_hand_written_model_gains_nothing_and_loses_nothing():
    """No `component_reads` means every component is assumed to read everything --- the
    conservative reading, and the reason 19 hand-written test problems needed no change."""
    m = Model([], {"p": lambda v: jnp.sum(v["z"].astype(float))},
              discrete_parameters=[IntegerParameter("z", (3,), lower=0, upper=1)])
    s = make_sampler_class(DiscreteMetropolisWithinGibbs, StaticContinuous)(
        m, m.default_sample(), seed=0)
    assert s._restricted() == {}


# --------------------------------------------------------------------------- #
# 2. the restricted delta equals the full delta                                #
# --------------------------------------------------------------------------- #

def test_the_restricted_delta_equals_the_full_delta():
    """The heart of it. Compared against `log_prob_at_coordinate` differences at every coordinate
    of several random label vectors --- the quantity the sweep would have computed."""
    data = _data()
    m = compile_model(SCAN, data=data)
    s = NUTS_GIBBS(m, m.default_sample(), seed=0)
    s.initialize()
    s.warmup(20)
    st, plan = s.state, s._restricted()["z"]
    ctx = s._sweep_context(st)
    rng = np.random.default_rng(1)

    def full(z, i, cur, prop):
        f = lambda zz: m.log_prob_at_coordinate(st.coordinate, st.chart_hyperparams,
                                                st.chart_indices, zz)
        return float(f(z.at[i].set(prop)) - f(z.at[i].set(cur)))

    worst = 0.0
    for _ in range(3):
        z = jnp.asarray(rng.integers(1, data["k"] + 1, data["n"]), jnp.int32)
        for i in range(data["n"]):
            cur = int(z[i]); prop = 1 + (cur % data["k"])
            got = float(s._discrete_delta(st, ctx, z.reshape(1, -1), "z", i,
                                          jnp.asarray([cur], jnp.int32),
                                          jnp.asarray([prop], jnp.int32), plan)[0])
            worst = max(worst, abs(got - full(z, i, cur, prop)))
    assert worst < 1e-3, worst


def test_dropping_an_affected_component_breaks_it():
    """The control for the test above: if a plan that omits a component the labels *do* reach still
    agreed, the comparison would be proving nothing."""
    data = _data()
    m = compile_model(SCAN, data=data)
    s = NUTS_GIBBS(m, m.default_sample(), seed=0)
    st, ctx = s.state, s._sweep_context(s.state)
    z = jnp.asarray(np.random.default_rng(2).integers(1, 4, data["n"]), jnp.int32)
    i, cur = 3, int(z[3])
    prop = 1 + (cur % data["k"])
    args = (st, ctx, z.reshape(1, -1), "z", i,
            jnp.asarray([cur], jnp.int32), jnp.asarray([prop], jnp.int32))
    good = float(s._discrete_delta(*args, s._restricted()["z"])[0])
    bad = float(s._discrete_delta(*args, ([], ["prior"]))[0])       # `lik` dropped
    assert abs(good - bad) > 1e-3, (good, bad)


def test_the_slow_path_is_exercised_and_agrees():
    """A component that reads the labels but is *not* elementwise in them falls back to a full pair
    of evaluations. Pinned because it is the branch a scan-component model normally never takes,
    so nothing else here would notice it rotting."""
    src = HEAD + """
    model coupling { target += -0.01 * sum(z) * sum(z); }
    model lik scan(z, y) { z ~ categorical(w); y ~ normal(mu[z], sigma); }
    """
    data = _data()
    m = compile_model(src, data=data)
    s = NUTS_GIBBS(m, m.default_sample(), seed=0)
    fast, slow = s._restricted()["z"]
    assert list(fast) == ["lik"] and list(slow) == ["coupling"]

    st, ctx = s.state, s._sweep_context(s.state)
    z = jnp.asarray(np.random.default_rng(3).integers(1, 4, data["n"]), jnp.int32)
    f = lambda zz: m.log_prob_at_coordinate(st.coordinate, st.chart_hyperparams,
                                            st.chart_indices, zz)
    for i in (0, 9, 19):
        cur = int(z[i]); prop = 1 + (cur % data["k"])
        got = float(s._discrete_delta(st, ctx, z.reshape(1, -1), "z", i,
                                      jnp.asarray([cur], jnp.int32),
                                      jnp.asarray([prop], jnp.int32),
                                      s._restricted()["z"])[0])
        want = float(f(z.at[i].set(prop)) - f(z.at[i].set(cur)))
        assert abs(got - want) < 1e-3, (i, got, want)


# --------------------------------------------------------------------------- #
# 3. the sampler still samples the right thing                                 #
# --------------------------------------------------------------------------- #

def _exact_pmf(lp, n):
    states = np.array(list(itertools.product((1, 2), repeat=n)))
    logp = np.array([float(lp(jnp.asarray(s, jnp.int32))) for s in states])
    w = np.exp(logp - logp.max())
    return states, w / w.sum()


def test_a_restricted_sweep_samples_the_exactly_enumerable_target():
    """Against enumeration, not against another sampler. The target is deliberately far from
    uniform, so a density-blind sweep could not pass."""
    src = """
    data { int m; array[m] real a; }
    parameters { array[m] int<lower=1, upper=2> z; }
    model lik scan(z, a) { target += a * (z - 1); }
    """
    a = np.array([1.3, -0.7, 2.1])
    model = compile_model(src, data={"m": 3, "a": a})
    s = make_sampler_class(DiscreteMetropolisWithinGibbs, StaticContinuous)(
        model, model.default_sample(), seed=0)
    assert s._restricted()                              # the fast path really is in use
    s.initialize()
    s.warmup(200)
    s.sample(40000)

    draws = np.asarray(s.get_discrete_flat())
    idx = (draws - 1) @ np.array([4, 2, 1])
    emp = np.bincount(idx, minlength=8) / len(draws)
    _, exact = _exact_pmf(lambda z: jnp.sum(jnp.asarray(a) * (z - 1)), 3)
    assert exact.max() / exact.min() > 20               # non-vacuous: far from uniform
    assert np.max(np.abs(emp - exact)) < 0.012, (emp, exact)
    assert len(np.unique(idx)) == 8                     # and it explored everything


def test_the_restricted_and_full_spellings_agree_on_the_posterior():
    """The same model written with a `for` loop and with a scan component must give the same
    answer. Not bit-identical --- the arithmetic genuinely differs --- so this is statistical, and
    the tolerance is set by the label mean's own spread."""
    data = _data(n=30, seed=5)
    out = {}
    for label, src in (("loop", LOOP), ("scan", SCAN)):
        m = compile_model(src, data=data)
        s = NUTS_GIBBS(m, m.default_sample(), seed=0)
        s.initialize()
        s.warmup(400)
        s.sample(1500)
        out[label] = (np.asarray(s.get_samples()["mu"]).mean(axis=0),
                      np.asarray(s.get_samples()["z"]).mean(axis=0))
    assert np.allclose(out["loop"][0], out["scan"][0], atol=0.25), out
    assert np.max(np.abs(out["loop"][1] - out["scan"][1])) < 0.25, out


# --------------------------------------------------------------------------- #
# 4. tempering                                                                 #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("tempered", [None, ("lik",)])
def test_the_restricted_per_rung_delta_equals_the_full_one(tempered):
    """Per rung, with the ladder **traced** (it is adapted, so it travels in the context).

    Run twice, and the second run is the one that matters: a power posterior tempers only `lik`, so
    a restricted sum scaled by one global beta would be wrong there while looking right when every
    component is tempered.
    """
    data = _data(n=10)
    m = compile_model(SCAN, data=data)
    kw = {} if tempered is None else {"tempered": tempered}
    K = 4
    s = parallel_tempering(m, n_temperatures=K, seed=0, **kw)
    s.warmup(15)
    st, plan = s.state, s._restricted()["z"]
    ctx = s._sweep_context(st)
    z = np.asarray(st.discrete).reshape(K, -1)

    def full(zz):
        c = s.context(st, kinetic_cache=False)._replace(discrete=jnp.asarray(zz).reshape(-1))
        return -np.asarray(per_temperature_potential(s.potentials, st.coordinate, c, K))

    worst = 0.0
    for i in range(data["n"]):
        cur = jnp.asarray(z[:, i], jnp.int32)
        prop = jnp.asarray(1 + (z[:, i] % data["k"]), jnp.int32)
        got = np.asarray(s._discrete_delta(st, ctx, jnp.asarray(z), "z", i, cur, prop, plan))
        zp = z.copy(); zp[:, i] = np.asarray(prop)
        worst = max(worst, float(np.max(np.abs(got - (full(zp) - full(z))))))
    assert worst < 1e-3, worst


def test_a_single_global_beta_would_be_wrong():
    """The control for the tempered check: the per-component weights are load-bearing, not
    bookkeeping. Uses a power posterior, where `prior` is untempered and `lik` is not."""
    m = compile_model(SCAN, data=_data(n=10))
    K = 4
    s = parallel_tempering(m, n_temperatures=K, seed=0, tempered=("lik",))
    s.warmup(15)
    st = s.state
    ctx = s._sweep_context(st)
    z = np.asarray(st.discrete).reshape(K, -1)
    i = 2
    cur = jnp.asarray(z[:, i], jnp.int32)
    prop = jnp.asarray(1 + (z[:, i] % 3), jnp.int32)
    correct = np.asarray(s._discrete_delta(st, ctx, jnp.asarray(z), "z", i, cur, prop,
                                           s._restricted()["z"]))
    betas = np.asarray(st.ham_params["__betas"])
    naive = correct / betas * betas[0]                  # "one global beta" spelling
    assert np.max(np.abs(naive - correct)) > 1e-2, (naive, correct)


def test_tempering_still_samples_the_enumerable_target_with_the_fast_path():
    src = """
    data { int m; array[m] real a; }
    parameters { array[m] int<lower=1, upper=2> z; }
    model lik scan(z, a) { target += a * (z - 1); }
    """
    a = np.array([1.3, -0.7, 2.1])
    model = compile_model(src, data={"m": 3, "a": a})
    s = parallel_tempering(model, n_temperatures=3, seed=0,
                           extra_mixins=(DiscreteMarginalAdaptation,))
    assert s._restricted()
    s.warmup(300)
    s.sample(30000)
    draws = np.asarray(s.get_discrete_flat())
    idx = (draws - 1) @ np.array([4, 2, 1])
    emp = np.bincount(idx, minlength=8) / len(draws)
    _, exact = _exact_pmf(lambda z: jnp.sum(jnp.asarray(a) * (z - 1)), 3)
    assert np.max(np.abs(emp - exact)) < 0.015, (emp, exact)
