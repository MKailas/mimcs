"""Tests for discrete (integer) parameters and the Metropolis-within-Gibbs sweep.

Four layers: (1) the ``IntegerParameter`` type and its bound validation; (2) the ``Model``
discrete block --- layout, the loud-``None`` policy, the no-discrete-parent rule; (3) the sweep
itself, against an **exactly enumerable** target and against detailed balance, each with a control
that must fail; and (4) composition with HMC --- above all that the potential caches are refreshed
after the labels move, which is the one defect here that no diagnostic would reveal.

Several checks would pass vacuously if written carelessly, and each carries its antidote:

* a *uniform* discrete target is matched by a sweep that ignores the density entirely, so the
  targets here are strongly non-uniform and a perturbed density must fail the same assertion;
* a *frozen* coordinate has zero variance, so ``ess_1d`` returns ``n`` and ``split_rhat`` returns
  1.0 --- a stuck label reads as perfectly converged. Hence the explicit distinct-value and
  ``discrete_moves`` assertions;
* the gradient-cache refresh compares against a fresh recomputation, which agrees trivially unless
  the *other* labels are shown to give a different answer.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from mimcs.model import (BaseDiscreteParameter, EuclideanParameter, IntegerParameter, Model,
                         PARAMETER_KINDS)
from mimcs.hmc import NUTS
from mimcs.adaptation import RobbinsMonroStepSize
from mimcs.samplers import (DiscreteMetropolisWithinGibbs, StaticContinuous, RandomWalkMH,
                            make_sampler_class)


GIBBS_ONLY = make_sampler_class(DiscreteMetropolisWithinGibbs, StaticContinuous)
NUTS_GIBBS = make_sampler_class(RobbinsMonroStepSize, DiscreteMetropolisWithinGibbs, NUTS)


# --------------------------------------------------------------------------- #
# 1. the parameter type                                                        #
# --------------------------------------------------------------------------- #

def test_an_integer_parameter_reports_its_support_and_layout():
    p = IntegerParameter("z", (3,), lower=1, upper=4)
    assert p.size == 3
    assert np.array_equal(np.asarray(p.lower), [1, 1, 1])
    assert np.array_equal(np.asarray(p.upper), [4, 4, 4])
    assert np.array_equal(np.asarray(p.n_values), [4, 4, 4])
    assert np.array_equal(np.asarray(p.default_value()), [1, 1, 1])
    assert isinstance(p, BaseDiscreteParameter)


def test_an_integer_parameter_carries_only_its_bare_value_as_a_feature():
    """Not the continuous default ``[x, x^2]``: for a binary z, ``z^2 == z`` exactly, so the
    second block would be a duplicate column inflating the multiplicity correction."""
    p = IntegerParameter("z", (2,), lower=0, upper=1)
    assert p.n_features == 2
    assert p.feature_names() == ["z[1]", "z[2]"]
    assert np.array_equal(np.asarray(p.features(jnp.asarray([1, 0]))), [1.0, 0.0])
    assert not np.asarray(p.stein_defined()).any()


@pytest.mark.parametrize("kwargs, fragment", [
    (dict(lower=None, upper=3), "needs an explicit lower bound"),
    (dict(lower=0, upper=None), "needs an explicit upper bound"),
    (dict(lower=0.5, upper=3), "non-integer lower bound"),
    (dict(lower=0, upper=float("inf")), "non-finite"),
    (dict(lower="mu", upper=3), "parameter-dependent"),
    (dict(lower=3, upper=1), "upper < lower"),
])
def test_an_integer_parameter_rejects_a_support_it_cannot_enumerate(kwargs, fragment):
    with pytest.raises(ValueError, match=fragment):
        IntegerParameter("z", (), **kwargs)


def test_a_float_bound_from_the_dsl_is_rounded_back_to_an_exact_int():
    """`dsl.semantics._resolve_bound` floats every constant bound, so `upper=4` arrives as 4.0."""
    p = IntegerParameter("z", (), lower=0.0, upper=4.0)
    assert (p.lower_value, p.upper_value) == (0, 4)


def test_validate_rejects_an_init_value_outside_the_support():
    p = IntegerParameter("z", (2,), lower=0, upper=1)
    p.validate([0, 1])
    with pytest.raises(ValueError, match="outside its support"):
        p.validate([0, 2])
    with pytest.raises(ValueError, match="non-integer"):
        p.validate([0.5, 1])


def test_the_registry_builds_a_discrete_parameter_for_int():
    """`int` used to be an alias for `real`, silently making a continuous BoundedParameter."""
    built = PARAMETER_KINDS["int"].build("z", (), base_sizes=(), lower=0.0, upper=1.0)
    assert isinstance(built, IntegerParameter)
    # Still declarable outside `parameters` -- `data { int n; }` must keep working.
    assert not PARAMETER_KINDS["int"].parameter_only


# --------------------------------------------------------------------------- #
# 2. the model's discrete block                                                #
# --------------------------------------------------------------------------- #

def _binary_model(n=3, w=(1.3, -0.7, 2.1), coupling=True):
    """`n` coupled binary coordinates: a strongly non-uniform, exactly enumerable target."""
    w = jnp.asarray(w[:n], float)
    J = jnp.asarray([[0.0, 1.5, -0.9], [1.5, 0.0, 0.6], [-0.9, 0.6, 0.0]])[:n, :n]
    if not coupling:
        J = jnp.zeros_like(J)

    def lp(v):
        z = v["z"].astype(float)
        return jnp.dot(w, z) + 0.5 * z @ J @ z

    model = Model([], {"p": lp},
                  discrete_parameters=[IntegerParameter("z", (n,), lower=0, upper=1)])
    return model, lp


def _exact_pmf(lp, n):
    states = np.array([[(i >> k) & 1 for k in range(n - 1, -1, -1)] for i in range(2 ** n)])
    logp = np.array([float(lp({"z": jnp.asarray(s)})) for s in states])
    w = np.exp(logp - logp.max())
    return states, w / w.sum()


def test_the_model_lays_out_a_separate_discrete_block():
    m, _ = _binary_model()
    assert m.discrete_dim == 3
    assert (m.coord_dim, m.ambient_dim) == (0, 0)      # discrete params add to neither
    assert m.discrete_block("z") == (0, 3)
    assert np.array_equal(np.asarray(m.default_discrete()), [0, 0, 0])


def test_features_and_names_put_the_discrete_block_last():
    m = Model([EuclideanParameter("x")], {"p": lambda v: -0.5 * v["x"] ** 2},
              discrete_parameters=[IntegerParameter("z", (2,), lower=0, upper=1)])
    assert m.feature_names == ["x", "x^2", "z[1]", "z[2]"]
    assert m.ambient_names == ["x", "z[1]", "z[2]"]
    assert list(m.stein_defined) == [True, True, False, False]
    f = m.features(jnp.asarray([2.0]), jnp.asarray([1, 0]))
    assert np.allclose(np.asarray(f), [2.0, 4.0, 1.0, 0.0])


def test_the_density_refuses_to_default_the_discrete_block_away():
    """A defaulted-away block would evaluate at stale labels: right shapes, wrong answer."""
    m, _ = _binary_model()
    with pytest.raises(ValueError, match="needs this model's discrete parameter"):
        m.log_prob_at_coordinate(jnp.zeros(0), m.init_chart_hyperparams(),
                                 m.init_chart_indices())
    with pytest.raises(ValueError, match="needs this model's discrete parameter"):
        m.features(jnp.zeros(0))


def test_a_purely_continuous_model_is_untouched_by_any_of_this():
    m = Model([EuclideanParameter("x")], {"p": lambda v: -0.5 * v["x"] ** 2})
    assert m.discrete_dim == 0 and m.discrete_parameters == []
    assert m.stein_defined.all()
    # the pre-existing three-argument call still works, unchanged
    assert float(m.log_prob_at_coordinate(jnp.zeros(1), m.init_chart_hyperparams(),
                                          m.init_chart_indices())) == 0.0


def test_a_discrete_parameter_may_not_be_a_charts_parent():
    """Stage 1's one real restriction, and what keeps the sweep from moving `sample`."""
    from mimcs.model import BoundedParameter
    x = BoundedParameter("x", (), lower=0.0, upper="z")
    with pytest.raises(NotImplementedError, match="discrete parent"):
        Model([x], {"p": lambda v: 0.0},
              discrete_parameters=[IntegerParameter("z", (), lower=1, upper=2)])


def test_a_name_cannot_be_both_continuous_and_discrete():
    with pytest.raises(ValueError, match="both continuous and discrete"):
        Model([EuclideanParameter("z")], {"p": lambda v: 0.0},
              discrete_parameters=[IntegerParameter("z", (), lower=0, upper=1)])


# --------------------------------------------------------------------------- #
# 3. the sweep: detailed balance and an exact stationary distribution          #
# --------------------------------------------------------------------------- #

def _proposal_matrix(n, lower, n_u=100001):
    """The sweep's own proposal formula, swept over a fine grid of u, as a matrix q[i, j]."""
    u = np.linspace(0.0, 1.0, n_u, endpoint=False)
    q = np.zeros((n, n))
    for i in range(n):
        cur = lower + i
        offset = np.floor(u.astype(np.float32) * np.float32(n - 1)).astype(np.int64) + 1
        prop = lower + ((cur - lower) + offset) % n
        q[i] = np.bincount(prop - lower, minlength=n) / n_u
    return q


@pytest.mark.parametrize("n", [2, 3, 5, 17])
@pytest.mark.parametrize("lower", [0, 1, -3])
def test_the_proposal_is_uniform_over_the_other_values_and_symmetric(n, lower):
    q = _proposal_matrix(n, lower)
    off = q[~np.eye(n, dtype=bool)]
    assert np.allclose(off, 1.0 / (n - 1), atol=1e-4)
    assert np.max(np.diag(q)) == 0.0                  # never proposes the current value
    assert np.allclose(q, q.T, atol=1e-4)             # symmetric => no Hastings term


def test_the_proposal_never_collapses_to_the_current_value_by_rounding():
    """`floor(u*(n-1))` reaching `n-1` would silently propose `cur`. Exhaustive over n, at the
    worst case u (the largest representable value below 1), in both precisions --- this is why
    there is no defensive clamp in the sweep."""
    n = np.arange(2, 200001, dtype=np.int64)
    for ftype in (np.float32, np.float64):
        for u in (np.nextafter(ftype(1.0), ftype(0.0)),
                  ftype(1.0) - ftype(2.0) ** -(24 if ftype is np.float32 else 53)):
            off = np.floor(ftype(u) * (n - 1).astype(ftype)).astype(np.int64)
            assert not (off >= n - 1).any()


@pytest.mark.parametrize("ratio, reversible", [
    ("correct", True),
    ("always", False),        # control: never compares the densities
    ("inverted", False),      # control: ratio upside down
])
def test_one_coordinate_update_satisfies_detailed_balance(ratio, reversible):
    """Built by enumeration, so this is the kernel itself rather than a sample of it.

    The two controls are what make the first case mean something: an assertion that only ever
    passes is not evidence.
    """
    rng = np.random.default_rng(0)
    n = 5
    logpi = rng.normal(size=n) * 3.0            # strongly non-uniform on purpose
    pi = np.exp(logpi - logpi.max()); pi /= pi.sum()
    q = _proposal_matrix(n, 0)

    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = logpi[j] - logpi[i]
            a = {"correct": min(1.0, np.exp(d)),
                 "always": 1.0,
                 "inverted": min(1.0, np.exp(-d))}[ratio]
            K[i, j] = q[i, j] * a
        K[i, i] = 1.0 - K[i].sum()

    assert np.allclose(K.sum(axis=1), 1.0)
    flux = pi[:, None] * K
    err = np.max(np.abs(flux - flux.T))
    stat = np.max(np.abs(pi @ K - pi))
    if reversible:
        assert err < 1e-4 and stat < 1e-4
    else:
        assert err > 1e-2 and stat > 1e-2


def test_an_asymmetric_proposal_would_need_a_hastings_term():
    """The third control: the acceptance ratio used here omits `q(b->a)/q(a->b)` because the
    proposal is symmetric. Swap in an asymmetric proposal and the same ratio stops being
    reversible --- which is what shows the symmetry is doing work rather than being decorative."""
    rng = np.random.default_rng(0)
    n = 5
    logpi = rng.normal(size=n) * 3.0
    pi = np.exp(logpi - logpi.max()); pi /= pi.sum()

    q = np.zeros((n, n))                      # biased toward larger indices
    for i in range(n):
        w = np.array([0.0 if j == i else (j + 1.0) for j in range(n)])
        q[i] = w / w.sum()

    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                K[i, j] = q[i, j] * min(1.0, np.exp(logpi[j] - logpi[i]))
        K[i, i] = 1.0 - K[i].sum()

    flux = pi[:, None] * K
    assert np.max(np.abs(flux - flux.T)) > 1e-2
    assert np.max(np.abs(pi @ K - pi)) > 1e-2


def _run_gibbs(model, n_samples=40000, seed=0, **kw):
    s = GIBBS_ONLY(model, model.default_sample(), seed=seed, **kw)
    s.initialize()
    s.warmup(200)
    s.sample(n_samples)
    return s


def test_the_sweep_recovers_an_exactly_enumerable_distribution():
    m, lp = _binary_model()
    s = _run_gibbs(m)
    states, exact = _exact_pmf(lp, 3)

    draws = s.get_discrete_flat()
    assert draws.dtype.kind == "i"                       # labels stay integers
    key = draws @ np.array([4, 2, 1])
    emp = np.bincount(key, minlength=8) / len(key)

    # Non-vacuity: the target must be far from uniform, or a density-blind sweep would pass.
    assert exact.max() / exact.min() > 20
    assert np.max(np.abs(emp - exact)) < 0.01

    # ... and the coordinates must actually move. A frozen one has zero variance, hence a
    # *perfect* ESS and R-hat 1.000 -- the failure this assertion exists to catch.
    assert int(np.sum(s.diagnostics()["discrete_moves"])) > 0.1 * len(draws)
    assert len(np.unique(key)) == 8


def test_a_perturbed_density_fails_the_same_comparison():
    """The control for the test above: the empirical pmf tracks *this* density, not just some
    plausible-looking distribution over the same support."""
    m, lp = _binary_model()
    s = _run_gibbs(m)
    draws = s.get_discrete_flat()
    emp = np.bincount(draws @ np.array([4, 2, 1]), minlength=8) / len(draws)

    _, wrong = _exact_pmf(lambda v: lp(v) * 0.5, 3)      # a different, equally valid pmf
    assert np.max(np.abs(emp - wrong)) > 0.05


def test_a_binary_coordinate_always_proposes_the_flip():
    """With n_i = 2 there is exactly one other value, so the acceptance probability *is* the
    sweep's mean alpha -- and the sweep should never waste a proposal on the current value."""
    m, _ = _binary_model(n=3, coupling=False)
    s = _run_gibbs(m, n_samples=4000)
    alpha = np.mean(s.diagnostics()["discrete_accept_prob"])
    w = np.array([1.3, -0.7, 2.1])
    p1 = 1.0 / (1.0 + np.exp(-w))                        # independent coordinates
    expected = np.mean(p1 * np.minimum(1, np.exp(-w)) + (1 - p1) * np.minimum(1, np.exp(w)))
    assert abs(alpha - expected) < 0.02


def test_a_degenerate_support_is_a_harmless_no_op():
    """lower == upper: the formula proposes the current value. Accepted, but not a *move*."""
    m = Model([], {"p": lambda v: 0.0 * v["z"].sum()},
              discrete_parameters=[IntegerParameter("z", (2,), lower=2, upper=2)])
    s = _run_gibbs(m, n_samples=200)
    assert np.array_equal(np.unique(s.get_discrete_flat()), [2])
    assert int(np.sum(s.diagnostics()["discrete_moves"])) == 0


def test_more_sweeps_per_iteration_move_more():
    m, _ = _binary_model()
    one = _run_gibbs(m, n_samples=2000, discrete_sweeps=1)
    three = _run_gibbs(m, n_samples=2000, discrete_sweeps=3)
    assert (np.mean(three.diagnostics()["discrete_moves"])
            > 2 * np.mean(one.diagnostics()["discrete_moves"]))


# --------------------------------------------------------------------------- #
# 4. composition with a continuous sampler                                     #
# --------------------------------------------------------------------------- #

def _mixed_model():
    """One binary z and a continuous mu, with P(z = 1 | mu) available in closed form."""
    def lp(v):
        z = v["z"].astype(float)
        return -0.5 * (1.0 - 2.0 * z) ** 2 - 0.5 * v["mu"] ** 2 + 0.4 * z + 0.3 * v["mu"] * z
    return Model([EuclideanParameter("mu")], {"p": lp},
                 discrete_parameters=[IntegerParameter("z", (), lower=0, upper=1)]), lp


def test_the_gradient_cache_is_refreshed_after_the_labels_move():
    """The one defect here that nothing else would reveal.

    `potential_grads` is cached at the current coordinate and read back verbatim by the next
    trajectory's leading half-kick. Left stale after a flip it is the gradient of the *previous*
    labels' density -- right shapes, right dtypes, an ordinary acceptance rate, wrong target.
    """
    m, _ = _mixed_model()
    s = NUTS_GIBBS(m, m.default_sample(), seed=0)
    s.initialize(); s.warmup(300); s.sample(300)
    st = s.state

    fresh_v, fresh_g = s._reseed_caches(st.coordinate, s.context(st, kinetic_cache=False))
    for k in fresh_v:
        assert np.allclose(np.asarray(fresh_g[k]), np.asarray(st.potential_grads[k]), atol=0)
        assert np.allclose(np.asarray(fresh_v[k]), np.asarray(st.potential_values[k]), atol=0)
    assert np.isclose(float(st.log_prob), float(-sum(st.potential_values.values())), atol=0)

    # CONTROL: the *other* labels must give a materially different gradient, or the agreement
    # above would hold no matter which labels the cache was built from.
    other = st._replace(discrete=1 - st.discrete)
    _, og = s._reseed_caches(other.coordinate, s.context(other, kinetic_cache=False))
    assert max(float(np.max(np.abs(np.asarray(og[k]) - np.asarray(fresh_g[k])))) for k in og) > 0.1


def test_nuts_plus_gibbs_recovers_the_marginal_label_probability():
    """The composition test: neither kernel alone covers this."""
    m, lp = _mixed_model()
    s = NUTS_GIBBS(m, m.default_sample(), seed=0)
    s.initialize(); s.warmup(1000); s.sample(12000)

    grid = np.linspace(-8, 8, 4001)
    dens = {z: np.exp([float(lp({"mu": jnp.asarray(g), "z": jnp.asarray(z)})) for g in grid])
            for z in (0, 1)}
    mass = {z: np.trapezoid(dens[z], grid) for z in (0, 1)}
    exact = mass[1] / (mass[0] + mass[1])

    z = np.asarray(s.get_samples()["z"]).ravel()
    assert len(np.unique(z)) == 2                        # it moved at all
    assert abs(z.mean() - exact) < 0.03


def test_a_sampler_that_cannot_move_labels_refuses_the_model():
    """Frozen labels are invisible in every diagnostic the library prints, so this raises."""
    m, _ = _mixed_model()
    with pytest.raises(TypeError, match="cannot move this model's discrete parameter"):
        make_sampler_class(RandomWalkMH)(m, m.default_sample(), seed=0)


def test_the_mixin_is_inert_and_stream_neutral_on_a_continuous_model():
    """It must add no RNG draw components: `RNGBuffer` splits its key into one subkey *per*
    component, so an extra one renumbers every other component's stream."""
    m = Model([EuclideanParameter("x")], {"p": lambda v: -0.5 * v["x"] ** 2})
    plain = make_sampler_class(RobbinsMonroStepSize, NUTS)(m, m.default_sample(), seed=0)
    with_mixin = NUTS_GIBBS(m, m.default_sample(), seed=0)
    assert ([c.name for c in plain._draw_components]
            == [c.name for c in with_mixin._draw_components])
    for s in (plain, with_mixin):
        s.warmup(50); s.sample(50)
    assert np.array_equal(plain.get_samples_flat(), with_mixin.get_samples_flat())


# --------------------------------------------------------------------------- #
# 5. summary, factory and tempering                                            #
# --------------------------------------------------------------------------- #

def test_the_summary_reports_a_discrete_parameter_but_gives_it_no_stein_z():
    m, _ = _mixed_model()
    s = NUTS_GIBBS(m, m.default_sample(), seed=0)
    s.initialize(); s.warmup(400); s.sample(2000)
    summary = s.summary()

    assert summary.coord_names == ["mu", "z"]
    assert summary.feature_names == ["mu", "mu^2", "z"]
    assert list(summary.stein_available) == [True, True, False]
    assert np.isfinite(summary.stein_z[:2]).all()
    assert not summary.stein_flagged[-1]                 # never flagged: there is no test
    assert np.isfinite(summary.mean).all()               # the label still gets a posterior row
    text = str(summary)
    assert "discrete" in text and "of 2 features flagged" in text


def test_the_factory_refuses_a_discrete_model():
    """No rule proposes the Gibbs sweep yet, and discrete parameters are kept out of
    `model.parameters` -- so the factory would otherwise partition the continuous half perfectly
    well and return a sampler that never moves a label."""
    from mimcs.factory import analyze, make_sampler
    m, _ = _mixed_model()
    for fn in (lambda: analyze(m), lambda: make_sampler(m)):
        with pytest.raises(NotImplementedError, match="discrete parameter"):
            fn()


def test_tempering_accepts_a_discrete_model():
    """Parallel tempering used to refuse one; it no longer does (doc 13, doc 14).

    Asserted here, in the stage-1 file, because this is where the refusal was recorded --- a
    silently-still-refusing `parallel_tempering` would otherwise look like the feature working.
    """
    from mimcs.pt import parallel_tempering
    m, _ = _mixed_model()
    s = parallel_tempering(m, n_temperatures=3, seed=0)
    assert s.model.discrete_dim == 3 * m.discrete_dim
    assert s.state.discrete.shape == (3 * m.discrete_dim,)
    assert s.state.discrete_proposal_params["z"].shape[0] == 3      # one table per rung


# --------------------------------------------------------------------------- #
# 6. the DSL, end to end                                                       #
# --------------------------------------------------------------------------- #

MIXTURE_SRC = """
data { int n; int k; array[n] real y; array[k] real w; }
parameters {
  ordered[k] mu;
  array[n] int<lower=1, upper=k> z;
  real<lower=0> sigma;
}
model {
  mu ~ normal(0, 10);
  sigma ~ lognormal(0, 1);
  for (i in 1:n) {
    z[i] ~ categorical(w);
    y[i] ~ normal(mu[z[i]], sigma);
  }
}
"""


def _mixture_data(n=90, k=3, sep=4.0, seed=0):
    """The data, the generating means, the generating labels, and the *estimand*.

    The estimand is not `true_mu`. With `m` points in a cluster the posterior concentrates on
    their sample mean, which differs from the generating value by O(sigma/sqrt(m)) --- 0.38 at
    the default n here, and still 0.23 at n = 300. Asserting that a sampler recovers `true_mu`
    is a statement about the *data*, not about the sampler, and it fails for a correct sampler
    whenever a cluster draws an unlucky sample.

    So the oracle is the conditional posterior mean given the true labels, which for a
    `normal(0, tau)` prior and unit-variance observations is `sum(y_j) / (m_j + 1/tau^2)`. Label
    uncertainty moves the true posterior a little off even that, which the tolerance allows for.
    """
    rng = np.random.default_rng(seed)
    true_mu = sep * (np.arange(k) - (k - 1) / 2.0)
    true_z = rng.integers(0, k, size=n)
    y = true_mu[true_z] + rng.standard_normal(n)
    mu_post = np.array([y[true_z == j].sum() / ((true_z == j).sum() + 1e-2) for j in range(k)])
    return ({"n": n, "k": k, "y": y, "w": np.full(k, 1.0 / k)}, true_mu, true_z + 1, mu_post)


def test_the_dsl_compiles_an_int_parameter_into_the_discrete_block():
    from mimcs import compile_model
    data, _, _, _ = _mixture_data()
    m = compile_model(MIXTURE_SRC, data=data)
    assert [p.name for p in m.discrete_parameters] == ["z"]
    assert m.discrete_dim == data["n"]
    assert [p.name for p in m.parameters] == ["mu", "sigma"]   # z is NOT among these
    assert m.coord_dim == data["k"] + 1


def test_an_int_parameter_needs_both_bounds_in_the_dsl():
    from mimcs import compile_model, DslError
    with pytest.raises(DslError, match="needs an explicit upper bound"):
        compile_model("parameters { int<lower=0> z; } model { }", data={})
    with pytest.raises(DslError, match="needs an explicit lower bound"):
        compile_model("parameters { int<lower=0> z; } model { }".replace("lower", "upper"),
                      data={})


def test_int_in_a_data_block_is_untouched():
    """`data { int n; }` sizes an array; it does not build a parameter and must keep working."""
    from mimcs import compile_model
    m = compile_model("data { int n; array[n] real y; } parameters { real x; }"
                      " model { y ~ normal(x, 1); }",
                      data={"n": 3, "y": np.zeros(3)})
    assert m.discrete_dim == 0


def test_the_dsl_mixture_recovers_its_generating_labels_and_means():
    """The flagship model: `categorical` as a prior, and a traced `mu[z[i]]` gather."""
    from mimcs import compile_model
    from mimcs.adaptation import MassMatrixAdaptation
    data, true_mu, true_z, mu_post = _mixture_data()
    m = compile_model(MIXTURE_SRC, data=data)

    cls = make_sampler_class(RobbinsMonroStepSize, MassMatrixAdaptation,
                             DiscreteMetropolisWithinGibbs, NUTS)
    s = cls(m, m.default_sample(), seed=0, target_accept=0.9)
    s.initialize(); s.warmup(1500); s.sample(2000)

    draws = s.get_samples()
    mu, z = np.asarray(draws["mu"]), np.asarray(draws["z"])
    assert z.dtype.kind == "i"
    # Against the conditional posterior mean, not the generating value -- see `_mixture_data`.
    assert np.allclose(mu.mean(axis=0), mu_post, atol=0.3), (mu.mean(axis=0), mu_post)
    # ... and the components are still resolved and correctly ordered, which is the thing the
    # `ordered[k]` chart plus a working sweep buys. (A far looser bound than the one above: the
    # point is that the three clusters are separated, not where exactly each sits.)
    assert np.allclose(mu.mean(axis=0), true_mu, atol=1.0)

    modal = np.round(np.median(z, axis=0)).astype(int)
    assert np.mean(modal == true_z) > 0.85
    # and the labels moved -- a stuck sweep would still "recover" them from a lucky init
    assert np.mean(s.diagnostics()["discrete_moves"]) > 0.5
