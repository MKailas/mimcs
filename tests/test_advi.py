"""Mean-field ADVI (``mimcs.vi.advi``) and its position initializer (``spec.position_init =
"advi"``).

The estimator is checked against autodiff and scipy, with controls so the checks cannot pass
vacuously. Seeds are fixed, so pass/fail is deterministic.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mimcs.factory import analyze
from mimcs.model import EuclideanParameter, IntegerParameter, Model, PositiveParameter
from mimcs.vi import advi
from mimcs.vi.meanfield import _draw, _entropy_const, _init, _step


def _diag_gaussian(d=6):
    mean = jnp.linspace(-1.0, 3.0, d)
    sd = jnp.exp(jnp.linspace(-1.5, 1.5, d))
    logz = float(jnp.sum(jnp.log(sd)) + 0.5 * d * math.log(2 * math.pi))
    return (lambda x: -0.5 * jnp.sum(((x - mean) / sd) ** 2)), np.asarray(mean), np.asarray(sd), logz


# --------------------------------------------------------------------------- #
# the estimator                                                               #
# --------------------------------------------------------------------------- #

def test_the_gradient_is_the_reparameterised_elbo_gradient():
    """One step's Adam input is the autodiff gradient of the K-draw reparameterised objective
    ``mean_k log p(mu + exp(omega) eps_k) + H(q)``. CONTROL: without the entropy's ``+ 1`` the
    omega gradient is off by exactly 1."""
    d = 7
    m_t, s_t = jnp.linspace(-1.0, 3.0, d), jnp.exp(jnp.linspace(-2.0, 2.0, d))

    def logp(x):
        return -0.5 * jnp.sum(((x - m_t) / s_t) ** 2) + 0.1 * jnp.sum(jnp.sin(x))
    c = _init(jnp.linspace(0.0, 1.0, d), jax.random.PRNGKey(0))
    c = c._replace(omega=jnp.linspace(-0.5, 0.5, d))
    _, sub = jax.random.split(c.key)
    e = jax.random.normal(sub, (5, d), c.mu.dtype)

    def objective(mu, om):
        return (jnp.mean(jax.vmap(logp)(mu + jnp.exp(om) * e)) + jnp.sum(om)
                + _entropy_const(d))
    g_mu, g_om = jax.grad(objective, (0, 1))(c.mu, c.omega)
    # The first Adam step is lr * sign(grad) (m_hat / sqrt(v_hat) = g / |g|), so read the
    # gradient off the first moment: m = (1 - b1) * grad.
    new, (elbo_r, elbo_h) = _step(logp, c, elbo_draws=5, learning_rate=1e-3, b1=0.9, b2=0.999,
                                  eps=1e-8)
    grad = np.asarray(new.m) / 0.1
    assert np.max(np.abs(grad[0] - np.asarray(g_mu))) < 1e-4
    assert np.max(np.abs(grad[1] - np.asarray(g_om))) < 1e-4
    assert np.min(np.abs(grad[1] - 1.0 - np.asarray(g_om))) > 0.99      # control
    # both ELBO estimates are the objective's value, two ways
    assert abs(float(elbo_h) - float(objective(c.mu, c.omega))) < 1e-4
    lq = -0.5 * jnp.sum(e * e, 1) - jnp.sum(c.omega) - 0.5 * d * math.log(2 * math.pi)
    assert abs(float(elbo_r) - float(jnp.mean(jax.vmap(logp)(c.mu + jnp.exp(c.omega) * e) - lq))) \
        < 1e-4


def test_draws_and_log_q_match_a_diagonal_normal():
    from scipy.stats import multivariate_normal
    mu, om = jnp.linspace(0.0, 1.0, 5), jnp.linspace(-0.5, 0.5, 5)
    phi, log_q = _draw(jax.random.PRNGKey(1), 20000, mu, om)
    ref = multivariate_normal(np.asarray(mu), np.diag(np.exp(2 * np.asarray(om))))
    assert np.max(np.abs(np.asarray(log_q[:100]) - ref.logpdf(np.asarray(phi[:100])))) < 1e-3
    z = (np.asarray(phi).std(0) / np.exp(np.asarray(om)) - 1) * math.sqrt(2 * 20000)
    assert np.max(np.abs(z)) < 5


# --------------------------------------------------------------------------- #
# fits                                                                        #
# --------------------------------------------------------------------------- #

def test_a_diagonal_gaussian_is_recovered():
    """The family contains the target, so the uncapped fit is exact up to Adam's jitter: over four
    seeds the mean is within 0.3 sd, the sd within 15%, and a fresh ELBO estimate within 0.1 nat of
    log Z. (The default patience stops it earlier, with the narrowest sd ~16% too wide: log sd
    moves at most ~learning_rate per iteration.)"""
    logp, mean, sd, logz = _diag_gaussian()
    for seed in range(4):
        r = advi(logp, np.random.default_rng(seed).uniform(-2, 2, 6), jax.random.PRNGKey(seed),
                 draws=2000, patience=None)
        assert not bool(r.failed) and int(r.n_skipped) == 0
        assert np.max(np.abs(np.asarray(r.mean) - mean) / sd) < 0.3, seed
        assert np.max(np.abs(np.exp(np.asarray(r.log_sd)) / sd - 1)) < 0.15, seed
        fresh = float(np.mean(np.asarray(r.log_p) - np.asarray(r.log_q)))
        assert abs(fresh - logz) < 0.1, (seed, fresh, logz)
        assert int(r.n_grad) == 5 * int(r.n_iter) and int(r.n_value) == 2000


def test_patience_stops_once_the_estimate_stops_improving():
    """``patience=None`` runs to ``max_iters``; with a patience the run stops exactly
    ``patience`` iterations after its best estimate, before ``max_iters``, on the same path."""
    from mimcs.vi import ADVI_DEFAULTS
    logp, *_ = _diag_gaussian()
    x0, key = np.full(6, 1.5), jax.random.PRNGKey(3)
    full = advi(logp, x0, key, max_iters=3000, patience=None)
    assert int(full.n_iter) == 3000 and ADVI_DEFAULTS["patience"] == 2000
    short = advi(logp, x0, key, max_iters=1000, patience=None)    # a prefix of the same path
    assert int(short.best_iter) == int(full.best_iter) or int(full.best_iter) > 1000
    capped = advi(logp, x0, key, max_iters=3000, patience=50)
    assert int(capped.n_iter) == int(capped.best_iter) + 50 < 3000
    assert int(capped.best_iter) <= int(full.best_iter)


def test_non_finite_draws_skip_the_update_and_a_hopeless_fit_fails():
    """A half-line support: early unit-scale draws cross into ``x < 0`` and are dropped; the fit
    still reaches the mode. (Skipping any iteration with a bad draw instead left it at its start.)
    With no finite density anywhere the fit fails: the start, repeated, with ``log q = +inf``."""
    def half(x):                                       # x > 0, mode at 3, sd 0.5
        return jnp.where(x[0] > 0, -0.5 * ((x[0] - 3.0) / 0.5) ** 2 + jnp.log(x[0]), -jnp.inf)
    # uncapped: at learning rate 1e-3 the climb to 3 takes ~8000 iterations, and the default
    # patience stops it near 1 (the slow-climb failure of the ELBO traces)
    r = advi(half, np.asarray([0.3]), jax.random.PRNGKey(0), max_iters=10000, patience=None)
    assert int(r.n_dropped) > 0 and not bool(r.failed)
    assert abs(float(r.mean[0]) - 3.0) < 0.5
    bad = advi(lambda x: -jnp.inf * jnp.sum(x ** 2), np.asarray([1.0, 2.0]),
               jax.random.PRNGKey(0), max_iters=20, draws=4)
    assert bool(bad.failed) and int(bad.n_skipped) == 20
    assert np.all(np.isposinf(np.asarray(bad.log_q)))
    assert np.allclose(np.asarray(bad.draws), [[1.0, 2.0]] * 4)


def test_options_are_validated():
    f, x0, k = (lambda x: -jnp.sum(x ** 2)), np.zeros(2), jax.random.PRNGKey(0)
    with pytest.raises(TypeError, match="unknown ADVI option"):
        advi(f, x0, k, learning_rat=0.1)
    for name, value in (("patience", 0), ("max_iters", 2.5), ("elbo_draws", "5")):
        with pytest.raises(ValueError, match=f"{name} must be an integer >= 1"):
            advi(f, x0, k, **{name: value})
    for name, value in (("learning_rate", 0.0), ("learning_rate", math.inf), ("b1", 1.0),
                        ("b2", 0.0), ("eps", -1e-8), ("b1", True)):
        with pytest.raises(ValueError, match=f"{name} must be"):
            advi(f, x0, k, **{name: value})


# --------------------------------------------------------------------------- #
# the initializer: spec.position_init = "advi"                                #
# --------------------------------------------------------------------------- #

def _narrow():
    """Far from the U(-2, 2) box: x ~ N(3, 0.1^2) per coordinate, and a log-normal scale."""
    return Model([EuclideanParameter("x", (3,)), PositiveParameter("s", ())],
                 {"lp": lambda p: (-0.5 * jnp.sum(((p["x"] - 3.0) / 0.1) ** 2)
                                   - 0.5 * jnp.log(p["s"]) ** 2)})


def test_the_slot_value_starts_the_chain_in_the_posterior():
    spec = analyze(_narrow())
    spec.position_init = "advi"
    assert "initialize()   advi + line_search" in str(spec)
    s = spec.build(seed=0)
    assert "ADVIInit" in [c.__name__ for c in type(s).__mro__]
    s.initialize()
    x = np.asarray(s.state.sample)[:3]
    assert np.max(np.abs(x - 3.0)) < 0.5                         # within 5 sd; U(-2,2) is 10+ off
    rep = s.init_report
    assert rep["method"] == "advi" and rep["failed"] == 0 and np.isfinite(rep["khat"])
    assert rep["grad_evals"] >= 5 * rep["n_iter"]
    s.warmup(200)
    s.sample(50)


def test_slot_options_reach_the_mixin_and_are_validated():
    spec = analyze(_narrow())
    spec.position_init, spec.position_init_params = "advi", {"learning_rate": 0.01, "patience": 7}
    s = spec.build(seed=1)
    assert (s._advi_options["learning_rate"], s._advi_options["patience"]) == (0.01, 7)
    s.initialize()
    assert s.init_report["n_iter"] == s.init_report["best_iter"] + 7
    spec.position_init_params = {"history": 3}
    with pytest.raises(ValueError, match="unknown position_init_params"):
        spec.build(seed=1)                          # a Pathfinder option
    spec.position_init_params = {"b2": 1.0}
    with pytest.raises(ValueError, match="b2 must be in"):
        spec.build(seed=1)
    spec.position_init_params, spec.algo_kwargs = {}, {"init_advi_patience": 3}
    with pytest.raises(ValueError, match="which the factory sets itself"):
        spec.build(seed=1)


def test_the_start_is_pathfinders_and_fixed_parameters_are_kept():
    """ADVI and Pathfinder draw their pi_0 start from the same substream, so a seed starts both
    from the same point; a supplied parameter is kept bit-exactly."""
    import mimcs.vi.meanfield as mf
    import mimcs.vi.pathfinder as pf
    seen = {}

    def spy(module, name, tag):
        real = getattr(module, name)

        def wrapped(logp, x0, key, **kw):
            seen[tag] = np.asarray(x0)
            return real(logp, x0, key, **kw)
        return real, wrapped

    for module, name, tag in ((mf, "advi", "advi"), (pf, "single_path", "pathfinder")):
        real, wrapped = spy(module, name, tag)
        setattr(module, name, wrapped)
        try:
            spec = analyze(_narrow())
            spec.position_init = tag
            s = spec.build(seed=4, init={"s": np.float32(1.7)}).initialize()
        finally:
            setattr(module, name, real)
        assert np.asarray(s.state.sample)[3] == np.float32(1.7)
    assert seen["advi"].shape == (3,) and np.array_equal(seen["advi"], seen["pathfinder"])


def test_under_tempering_each_rung_starts_at_its_own_draw():
    spec = analyze(_narrow())
    spec.base, spec.tempering_params = "pt_nuts", {"n_temperatures": 3}
    spec.position_init = "advi"
    s = spec.build(seed=0).initialize()
    rows = np.asarray(s.state.sample).reshape(3, -1)
    assert not np.allclose(rows[0], rows[1]) and not np.allclose(rows[1], rows[2])
    assert np.max(np.abs(rows[:, :3] - 3.0)) < 0.5
    assert np.all(np.isfinite(np.asarray(s.state.log_prob)))


def test_a_static_base_refuses_advi():
    m = Model([], {"p": lambda v: -jnp.sum(v["z"])},
              discrete_parameters=[IntegerParameter("z", (4,), lower=0, upper=2)])
    spec = analyze(m)
    spec.position_init = "advi"
    with pytest.raises(ValueError, match="no continuous coordinate"):
        spec.build(seed=0)
