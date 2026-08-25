"""Tests for sample evaluation: the Stein diagnostic, the score pullback, and `sampler.summary()`.

The three things that matter, hardest first:

* **The Stein operator is correct.** On i.i.d. draws from a known target each feature's Langevin
  term averages to zero within its MCSE --- for the flat charts and for the sphere's closed-form
  generator, and robust to a spurious normal component in the score.
* **The score pullback equals the recompute.** Reusing the sampler's saved coordinate gradients
  (pulled back through the chart) gives the same ambient score as re-differentiating the model,
  across every chart type. This is what lets the summary avoid recomputing what sampling already
  spent.
* **Target-awareness.** A biased sample with healthy ESS and R-hat is caught by the Stein z where
  the mixing diagnostics pass it.

Seeds are fixed, so pass/fail is deterministic.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from mimcs.model import (Model, EuclideanParameter, PositiveParameter, IntervalParameter,
                        BoundedParameter, UnitVectorParameter)
from mimcs.summary import summarize, Summary
from mimcs.testing import correlated_gaussian, positive_lognormal, von_mises_fisher, nuts
from mimcs.testing.problems import _vmf_draw


# --- the Stein operator is mean-zero under the target ------------------------- #

def _stein_z_iid(param, samples, scores):
    """Per-feature z-score of the Stein terms on i.i.d. draws (SE = sd/sqrt(N))."""
    st = np.asarray(jax.vmap(param.stein_terms)(jnp.asarray(samples), jnp.asarray(scores)))
    return st.mean(0) / (st.std(0) / np.sqrt(len(st)))


def test_stein_terms_flat_known_value():
    """`1 + x s` and `2x + x^2 s` on one fixed point, by hand."""
    p = EuclideanParameter("x", ())
    got = np.asarray(p.stein_terms(jnp.array([2.0]), jnp.array([-3.0])))   # score -3
    assert np.allclose(got, [1 + 2 * -3, 2 * 2 + 4 * -3])                  # [-5, -8]


def test_stein_identity_gaussian():
    """N(0,1): score = -x, so E[1-x^2]=0 and E[2x-x^3]=0."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((500000, 1))
    z = _stein_z_iid(EuclideanParameter("x", ()), x, -x)
    assert np.all(np.abs(z) < 4.0), z


def test_stein_identity_lognormal():
    """PositiveParameter, log s ~ N(0,1): the first Stein term is -log s, mean 0."""
    rng = np.random.default_rng(1)
    s = np.exp(rng.standard_normal((500000, 1)))
    score = -1.0 / s - np.log(s) / s                         # d/ds log p, p = LogNormal(0,1)
    z = _stein_z_iid(PositiveParameter("s"), s, score)
    assert np.all(np.abs(z) < 4.0), z


def test_stein_identity_sphere_vmf():
    """The spherical Langevin generator averages to zero on vMF draws."""
    rng = np.random.default_rng(2)
    mu = np.array([0.3, -0.5, 0.81]); mu /= np.linalg.norm(mu)
    x = _vmf_draw(500000, rng, mu, 5.0)
    g = np.tile(5.0 * mu, (len(x), 1))                       # ambient score of exp(5<x,mu>)
    z = _stein_z_iid(UnitVectorParameter("x", 3), x, g)
    assert np.all(np.abs(z) < 4.0), z


def test_sphere_stein_ignores_the_normal_component_of_the_score():
    """Only the tangential score matters: adding a large normal part must not change the terms."""
    rng = np.random.default_rng(3)
    x = _vmf_draw(2000, rng, np.array([0.0, 0.0, 1.0]), 5.0)
    p = UnitVectorParameter("x", 3)
    g = np.tile([0.0, 0.0, 5.0], (len(x), 1))
    g_noisy = g + 100.0 * x                                  # 100 * x is purely normal at x
    a = jax.vmap(p.stein_terms)(jnp.asarray(x), jnp.asarray(g))
    b = jax.vmap(p.stein_terms)(jnp.asarray(x), jnp.asarray(g_noisy))
    # 100x amplifies float32's ~1e-7 unit-norm error to ~1e-5; the projection is exact in x64.
    assert np.allclose(np.asarray(a), np.asarray(b), atol=1e-3)


# --- the score pullback equals the recompute -------------------------------- #

def _saved_coord_score(model, x):
    """What the sampler stores: grad_q [log pi(x(q)) + log|J|] at the initial chart."""
    h, c = model.init_chart_hyperparams(), model.init_chart_indices()
    q = model.sample_to_coordinate(x, h, c)
    return np.asarray(jax.grad(lambda qq: model.log_prob_at_coordinate(qq, h, c))(q)), h, c


@pytest.mark.parametrize("name, model, x", [
    ("euclidean", Model([EuclideanParameter("x", (3,))],
                        {"lp": lambda d: -0.5 * jnp.sum((d["x"] - 1.0) ** 2)}),
     jnp.array([0.4, -1.1, 2.0])),
    ("positive-log", Model([PositiveParameter("s")],
                           {"lp": lambda d: -jnp.log(d["s"]) - 0.5 * jnp.log(d["s"]) ** 2}),
     jnp.array([1.7])),
    ("interval-logit", Model([IntervalParameter("t", -1.0, 3.0)],
                             {"lp": lambda d: -0.5 * d["t"] ** 2}), jnp.array([0.6])),
    ("parent-bound", Model([IntervalParameter("a", 0.0, 1.0),
                            BoundedParameter("b", lower=0.0, upper="a")],
                           {"lp": lambda d: jnp.zeros(())}), jnp.array([0.6, 0.3])),
    ("unit-vector", Model([UnitVectorParameter("x", 3)],
                          {"lp": lambda d: 5.0 * d["x"][2]}),
     jnp.array([0.0, 0.6, 0.8])),
])
def test_pullback_matches_recompute(name, model, x):
    """The chart pullback of the saved gradient equals the direct ambient recompute."""
    coord_score, h, c = _saved_coord_score(model, x)
    pulled = model.ambient_score(x, jnp.asarray(coord_score), h, c)
    recomputed = model.ambient_score(x)
    # Compare through stein_terms, which projects (the sphere pullback recovers only the
    # tangential score; the flat charts recover it exactly).
    st_pull = model.stein_terms(x, pulled)
    st_recomp = model.stein_terms(x, recomputed)
    assert np.allclose(np.asarray(st_pull), np.asarray(st_recomp), atol=1e-5)


def test_pullback_is_the_identity_for_a_euclidean_chart():
    """The free case: for an identity chart the saved gradient *is* the ambient score."""
    model = Model([EuclideanParameter("x", (3,))],
                  {"lp": lambda d: -0.5 * jnp.sum(d["x"] ** 2)})
    x = jnp.array([0.4, -1.1, 2.0])
    coord_score, h, c = _saved_coord_score(model, x)
    pulled = np.asarray(model.ambient_score(x, jnp.asarray(coord_score), h, c))
    assert np.allclose(pulled, coord_score, atol=1e-6)


# --- target-awareness: the point of the whole diagnostic --------------------- #

def test_stein_flags_a_biased_sample_that_ess_and_rhat_pass():
    """A biased-but-well-mixed sample: ESS and R-hat healthy, Stein z large."""
    rng = np.random.default_rng(0)
    n = 12000
    biased = rng.standard_normal((n, 1)) + 0.3            # i.i.d. from N(0.3,1), target N(0,1)
    model = Model([EuclideanParameter("x", ())], {"lp": lambda d: -0.5 * d["x"] ** 2})
    summ = summarize(model, biased)

    assert np.all(summ.ess > 0.8 * n), "i.i.d. draws should have near-perfect ESS"
    assert np.all(summ.rhat < 1.01), "i.i.d. draws should have R-hat ~1"
    assert np.any(np.abs(summ.stein_z) > 4.0), "the bias must show in a Stein z"


def test_stein_does_not_flag_an_honest_sample():
    """Draws from the actual target: nothing flagged beyond the multiplicity rate."""
    rng = np.random.default_rng(1)
    honest = rng.standard_normal((12000, 3))
    model = Model([EuclideanParameter("x", (3,))], {"lp": lambda d: -0.5 * jnp.sum(d["x"] ** 2)})
    summ = summarize(model, honest)
    assert summ.stein_flagged.sum() <= 1                 # 6 features, ~0.3 expected


def test_uniform_boundary_is_reported_not_exploded():
    """Uniform(0,1): score 0, so the Stein series is constant --- flag boundary, not inf z."""
    rng = np.random.default_rng(2)
    u = rng.random((5000, 1))
    model = Model([IntervalParameter("t", 0.0, 1.0)], {"lp": lambda d: jnp.zeros(())})
    summ = summarize(model, u)
    assert summ.stein_boundary[0]                        # the linear feature's term is const 1
    assert np.all(np.isfinite(summ.stein_z))


# --- descriptive stats and the object ---------------------------------------- #

def test_posterior_table_matches_numpy():
    rng = np.random.default_rng(0)
    draws = rng.standard_normal((5000, 2)) * [2.0, 0.5] + [1.0, -3.0]
    model = Model([EuclideanParameter("x", (2,))], {"lp": lambda d: jnp.zeros(())})
    summ = summarize(model, draws)
    assert np.allclose(summ.mean, draws.mean(0))
    assert np.allclose(summ.sd, draws.std(0, ddof=1))
    assert np.allclose(summ.quantiles, np.quantile(draws, [0.05, 0.5, 0.95], axis=0))
    assert summ.coord_names == ["x[0]", "x[1]"]
    assert len(summ.feature_names) == model.n_features == 4


def test_summary_names_and_lengths_for_a_mixed_model():
    model = Model([EuclideanParameter("a", (2,)), PositiveParameter("s"),
                   UnitVectorParameter("x", 3)], {"lp": lambda d: jnp.zeros(())})
    rng = np.random.default_rng(0)
    draws = np.tile(model.default_sample(), (200, 1)) + 0.01 * rng.standard_normal((200, 6))
    summ = summarize(model, draws)
    assert len(summ.coord_names) == model.ambient_dim == 6
    assert summ.coord_names[:3] == ["a[0]", "a[1]", "s"]
    assert len(summ.feature_names) == model.n_features
    assert len(summ.ess) == len(summ.stein_z) == model.n_features


def test_summary_renders_and_is_stable():
    rng = np.random.default_rng(0)
    draws = rng.standard_normal((2000, 2))
    model = Model([EuclideanParameter("x", (2,))], {"lp": lambda d: -0.5 * jnp.sum(d["x"] ** 2)})
    text = str(summarize(model, draws, accept_rate=0.9))
    assert "Posterior summary" in text and "Diagnostics (per feature)" in text
    assert "stein-z" in text and "Stein:" in text


def test_summarize_rejects_empty():
    model = Model([EuclideanParameter("x", ())], {"lp": lambda d: jnp.zeros(())})
    with pytest.raises(ValueError, match="non-empty"):
        summarize(model, np.empty((0, 1)))


# --- end to end through the sampler ------------------------------------------ #

def test_sampler_summary_caches_and_reuses_saved_gradients():
    problem = correlated_gaussian()
    sampler = nuts()(problem.model, 0)
    sampler.warmup(1000)
    sampler.sample(4000)
    summ = sampler.summary()
    assert isinstance(summ, Summary)
    assert sampler._summary is summ                       # cached
    assert summ.n_samples == 4000
    # honest NUTS draws: mixing healthy, target-fit not flagged beyond multiplicity
    assert np.all(summ.rhat < 1.05)
    assert summ.stein_flagged.sum() <= 1


def test_sampler_summary_before_sampling_errors():
    sampler = nuts()(correlated_gaussian().model, 0)
    with pytest.raises(RuntimeError, match="call sample"):
        sampler.summary()
