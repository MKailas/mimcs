"""Fixed-draw mean-field VI (``mimcs.vi.dadvi``, DADVI) and its position initializer
(``spec.position_init = "dadvi"``).

The identities the design rests on (``mimcs/vi/deterministic.py``) are checked in x64 against
direct computation, each with a control so it cannot pass vacuously: on a Gaussian target the fit's
centroid is the mean and its cloud's spread the sd (while the fitted sigma is not), and the
linear-response sandwich is the exact covariance (while the plain mean block of the inverse
Hessian is not). Seeds are fixed, so pass/fail is deterministic.
"""


import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.stats as ss

from mimcs import config
from mimcs.factory import analyze
from mimcs.model import EuclideanParameter, IntegerParameter, Model, PositiveParameter
from mimcs.vi import dadvi, single_path
from mimcs.vi.deterministic import (READOUTS, STOP_REASONS, _factors, _init, _objective,
                                    _pattern, _readout, _step, dadvi_options)


@pytest.fixture
def x64():
    was = config.x64_enabled()
    config.enable_x64(True)
    try:
        yield
    finally:
        config.enable_x64(was)


def _pattern_of(key, K, n, standardize=False):
    """The fixed draws a fit with ``key`` uses (``_fit`` splits off the first key for them)."""
    return np.asarray(_pattern(jax.random.split(key)[0], K, n, jnp.asarray(0.0).dtype,
                               standardize))


def _correlated(n=3, seed=0):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    C = Q @ np.diag(np.exp(np.linspace(-2.0, 1.5, n))) @ Q.T
    m = rng.normal(size=n)
    P = jnp.asarray(np.linalg.inv(C))
    return (lambda x: -0.5 * (x - m) @ P @ (x - m)), m, C


def _eight_schools(x):
    """Centred eight schools on (mu, log tau, theta), Jacobian included: log p -> +inf down the
    neck."""
    y = jnp.array([28., 8., -3., 7., -1., 1., 18., 12.])
    s = jnp.array([15., 10., 16., 11., 9., 11., 10., 18.])
    mu, t, th = x[0], x[1], x[2:]
    tau = jnp.exp(t)
    return (-0.5 * (mu / 5.0) ** 2 - jnp.log1p((tau / 5.0) ** 2)
            - 0.5 * jnp.sum(((th - mu) / tau) ** 2) - 8 * t - 0.5 * jnp.sum(((y - th) / s) ** 2)
            + t)


# --------------------------------------------------------------------------- #
# the objective and the fit                                                   #
# --------------------------------------------------------------------------- #


def test_the_objective_is_the_fixed_draw_elbo():
    lp, m, C = _correlated(4)
    g = jnp.asarray(np.random.default_rng(1).normal(size=(5, 4)))
    mu, omega = jnp.asarray([0.3, -1.0, 2.0, 0.5]), jnp.asarray([0.2, -0.4, 0.0, 1.1])
    x = mu + jnp.exp(omega) * g
    log_q = ss.norm(np.asarray(mu), np.exp(np.asarray(omega))).logpdf(np.asarray(x)).sum(1)
    want = np.mean(np.asarray(jax.vmap(lp)(x)) - log_q)
    F = _objective(lp, g)
    assert abs(float(F(jnp.concatenate([mu, omega]))) - want) < 1e-4 * abs(want)
    bad = _objective(lambda x: jnp.where(x[0] > 0, lp(x), -jnp.inf), g)
    assert not np.isfinite(float(bad(jnp.concatenate([mu, omega]))))   # one point off support


def test_a_diagonal_gaussian_fit_has_the_exact_centroid_and_spread(x64):
    """M2: centroid mu + sigma * gbar = mean and sigma * shat = sd exactly; the fitted sigma (and
    mu) are off by the pattern's sampling noise --- unless the pattern is standardised."""
    mean, sd = np.linspace(-1.0, 3.0, 6), np.exp(np.linspace(-1.5, 1.5, 6))
    lp = lambda x: -0.5 * jnp.sum(((x - mean) / sd) ** 2)
    key = jax.random.PRNGKey(3)
    res = dadvi(lp, jnp.zeros(6), key, covariance="diag", rel_tol=1e-15, max_iters=5000)
    assert STOP_REASONS[int(res.stop)] in ("tolerance", "line search")
    g = _pattern_of(key, 5, 6)
    sigma = np.exp(np.asarray(res.log_sd))
    np.testing.assert_allclose(np.asarray(res.center), mean, atol=1e-6)
    np.testing.assert_allclose(sigma * g.std(0), sd, rtol=1e-6)
    assert np.max(np.abs(sigma / sd - 1)) > 0.1                       # control: sigma is not sd
    assert np.max(np.abs(np.asarray(res.mean) - mean) / sd) > 0.1     # nor mu the mean

    std = dadvi(lp, jnp.zeros(6), key, covariance="diag", standardize=True, rel_tol=1e-15,
                max_iters=5000)
    gs = _pattern_of(key, 5, 6, standardize=True)
    np.testing.assert_allclose(gs.mean(0), 0.0, atol=1e-12)
    np.testing.assert_allclose(gs.std(0), 1.0, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(std.mean), mean, atol=1e-6)
    np.testing.assert_allclose(np.exp(np.asarray(std.log_sd)), sd, rtol=1e-6)


def test_the_linear_response_sandwich_is_the_exact_covariance(x64):
    """M4: J^T (-hess F)^-1 J = C on a correlated Gaussian, for i.i.d. draws, with J = d xbar /
    d eta; the naive mean block of (-hess F)^-1 is not C. The lr_lbfgs read-out (L-BFGS's own
    estimate of the inverse Hessian, full history here) is close to it."""
    n = 3
    lp, m, C = _correlated(n)
    key = jax.random.PRNGKey(0)
    res = dadvi(lp, jnp.zeros(n), key, history=2 * n, rel_tol=1e-15, max_iters=5000,
                draws=40000, covariance="lr_lbfgs")
    g = jnp.asarray(_pattern_of(key, 5, n))
    eta = jnp.concatenate([res.mean, res.log_sd])
    Hinv = np.linalg.inv(-np.asarray(jax.hessian(_objective(lp, g))(eta)))
    v = np.exp(np.asarray(res.log_sd)) * np.asarray(g).mean(0)
    J = np.vstack([np.eye(n), np.diag(v)])
    np.testing.assert_allclose(J.T @ Hinv @ J, C, atol=1e-6 * np.abs(C).max())
    assert np.max(np.abs(Hinv[:n, :n] - C)) > 1e-2 * np.abs(C).max()  # control: the naive block
    np.testing.assert_allclose(np.asarray(res.center), m, atol=1e-6)
    assert not bool(res.lr_failed)
    emp = np.cov(np.asarray(res.draws).T)
    assert np.max(np.abs(emp - C)) < 0.1 * np.abs(C).max()
    np.testing.assert_allclose(np.asarray(res.draws).mean(0), m, atol=0.05 * np.sqrt(C.max()))


def test_the_readout_log_q_is_the_sandwiched_normal(x64):
    """The lr_lbfgs read-out's log q is the density of N(xbar, J^T H~ J) built densely from the
    fit's L-BFGS factors; the diag read-out's that of N(mu, diag sigma^2)."""
    n, K = 7, 5
    lp, m, C = _correlated(n, seed=2)
    opts = dadvi_options({"max_iters": 40})
    g = _pattern(jax.random.PRNGKey(5), K, n, jnp.float64, False)
    vg = jax.value_and_grad(_objective(lp, g))
    step = {k: opts[k] for k in ("history", "max_iters", "rel_tol", "c1", "c2", "max_halvings")}
    c = jax.lax.while_loop(lambda c: c.stop == 0, lambda c: _step(vg, c, **step),
                           _init(vg, jnp.zeros(n), opts["history"]))
    assert int(c.stop) != 0 and int(c.k) > 5
    phi, log_p, log_q, center, readout = _readout(lp, c, g, jax.random.PRNGKey(6),
                                                  covariance="lr_lbfgs", draws=50)
    beta, gamma = (np.asarray(a) for a in _factors(c.alpha, c.PS, c.PZ, c.pmask))
    sd = np.exp(np.asarray(c.eta[n:]))
    v = sd * np.asarray(g).mean(0)
    J = np.vstack([np.eye(n), np.diag(v)])
    sigma = J.T @ (np.diag(np.asarray(c.alpha)) + beta @ gamma @ beta.T) @ J
    want = ss.multivariate_normal(np.asarray(center), sigma).logpdf(np.asarray(phi))
    assert READOUTS[int(readout)] == "lr_lbfgs" and 2 * opts["history"] > n   # thin-QR branch
    np.testing.assert_allclose(np.asarray(log_q), want, atol=1e-8)
    np.testing.assert_allclose(np.asarray(log_p), np.asarray(jax.vmap(lp)(phi)))
    phi, _, log_q, _, _ = _readout(lp, c, g, jax.random.PRNGKey(6), covariance="diag", draws=50)
    want = ss.norm(np.asarray(c.eta[:n]), sd).logpdf(np.asarray(phi)).sum(1)
    np.testing.assert_allclose(np.asarray(log_q), want, atol=1e-8)


def test_the_exact_lr_readout_is_the_posterior_covariance(x64):
    """covariance="lr" forms the Hessian exactly: on a correlated 5-d Gaussian with a short
    history (J = 6 < 2n), its draws have the exact covariance and log q the exact density (q = p
    up to log Z), where L-BFGS's estimate (lr_lbfgs, the control) is visibly off."""
    n = 5
    lp, m, C = _correlated(n, seed=1)
    key = jax.random.PRNGKey(2)
    tight = dict(rel_tol=1e-15, max_iters=5000, draws=40000)       # converge fully
    res = dadvi(lp, jnp.zeros(n), key, covariance="lr", **tight)
    assert READOUTS[int(res.readout)] == "lr" and not bool(res.lr_failed)
    emp = np.cov(np.asarray(res.draws).T)
    assert np.max(np.abs(emp - C)) < 0.03 * np.abs(C).max()
    np.testing.assert_allclose(np.asarray(res.log_q),
                               ss.multivariate_normal(m, C).logpdf(np.asarray(res.draws)),
                               atol=1e-4)
    w = np.asarray(res.log_p) - np.asarray(res.log_q)      # constant = log Z when q = p
    assert np.std(w) < 1e-4
    ctl = dadvi(lp, jnp.zeros(n), key, covariance="lr_lbfgs", **tight)
    assert np.max(np.abs(np.cov(np.asarray(ctl.draws).T) - C)) > 0.1 * np.abs(C).max()
    assert np.std(np.asarray(ctl.log_p) - np.asarray(ctl.log_q)) > 0.1
    # the Hessian's cost is counted: 2n Hessian-vector products at ~2 K gradients each
    assert int(res.n_grad) == int(ctl.n_grad) + 2 * 2 * n * 5


def test_a_non_positive_definite_hessian_falls_back():
    """Where -hess F is not positive definite (an improper, convex log density) the exact read-out
    falls back to L-BFGS's estimate, which is positive definite by construction."""
    lp = lambda x: 0.5 * jnp.sum(x * x)
    g = _pattern(jax.random.PRNGKey(0), 5, 3, jnp.float32, False)
    F = _objective(lp, g)
    vg = jax.value_and_grad(F)
    c = _init(vg, jnp.zeros(3), 6)
    *_, readout = _readout(lp, c, g, jax.random.PRNGKey(1), covariance="lr", draws=10, F=F)
    assert READOUTS[int(readout)] == "lr_lbfgs"


def test_ascent_is_monotone_and_counted():
    lp, m, C = _correlated(5, seed=4)
    g = _pattern(jax.random.split(jax.random.PRNGKey(0))[0], 5, 5, jnp.float32, False)
    vg = jax.jit(jax.value_and_grad(_objective(lp, g)))
    opts = dadvi_options({})
    step = jax.jit(lambda c: _step(vg, c, **{k: opts[k] for k in ("history", "max_iters",
                                                                  "rel_tol", "c1", "c2",
                                                                  "max_halvings")}))
    c = _init(vg, jnp.full(5, 1.5), opts["history"])
    fs = [float(c.f)]
    while int(c.stop) == 0:
        c = step(c)
        fs.append(float(c.f))
    assert len(fs) > 5 and np.all(np.diff(fs) >= 0)
    res = dadvi(lp, jnp.full(5, 1.5), jax.random.PRNGKey(0), covariance="lr_lbfgs")  # no Hessian
    # K gradients per evaluation of F, at least one evaluation per step plus the start (the fused
    # loop rounds differently from the step-by-step one in f32, so its count may differ by a step)
    assert int(res.n_grad) % 5 == 0 and int(res.n_grad) >= 5 * (int(res.n_iter) + 1)
    assert abs(int(res.n_grad) - 5 * int(c.n_eval)) <= 0.1 * int(res.n_grad)


def test_a_raw_gradient_step_with_an_empty_history_is_scaled():
    """With no curvature pair held after the first iteration (every pair so far rejected), the
    direction is the raw gradient: it is scaled to unit length, as on the first step. Without
    that, a 1e12 gradient overshoots every one of the 30 halvings and the fit stalls (bci,
    seeds 4/10/11). The control is a carry that holds a pair, which takes the unscaled step."""
    lp = lambda x: -1e12 * jnp.sum(x * x)
    g = _pattern(jax.random.PRNGKey(0), 5, 2, jnp.float32, False)
    vg = jax.value_and_grad(_objective(lp, g))
    opts = dadvi_options({})
    step = {k: opts[k] for k in ("history", "max_iters", "rel_tol", "c1", "c2", "max_halvings")}
    c = _init(vg, jnp.full(2, 1e-3), opts["history"])._replace(k=jnp.asarray(3))
    assert float(jnp.max(jnp.abs(c.g))) > 1e6
    c1 = _step(vg, c, **step)
    assert int(c1.stop) == 0 and int(c1.k) == 4 and float(c1.f) > float(c.f)
    assert float(jnp.linalg.norm(c1.eta - c.eta)) <= 1.0 + 1e-6
    held = c._replace(hist=c.hist._replace(rho=c.hist.rho.at[-1].set(1.0)))   # control
    c2 = _step(vg, held, **step)
    assert int(c2.stop) == STOP_REASONS.index("line search")


def test_a_start_whose_unit_cloud_leaves_the_support_is_shrunk():
    """At x0 = 0.3 on a half line, unit-scale points fall outside: omega0 is lowered until F is
    finite, and the fit then reaches Gamma(3, 1)'s bulk."""
    lp = lambda x: (jnp.where(x[0] > 0, 2 * jnp.log(jnp.where(x[0] > 0, x[0], 1.0)) - x[0],
                              -jnp.inf) - 0.5 * jnp.sum(x[1:] ** 2))
    res = dadvi(lp, jnp.array([0.3, 0.0, 0.0]), jax.random.PRNGKey(1))
    assert float(res.omega0) < 0 and not bool(res.failed)
    assert STOP_REASONS[int(res.stop)] == "tolerance"
    assert 2.0 < float(res.center[0]) < 4.0
    assert np.all(np.asarray(res.draws)[:, 0][np.isfinite(np.asarray(res.log_p))] > 0)
    hopeless = dadvi(lambda x: -jnp.inf * jnp.sum(x), jnp.zeros(2), jax.random.PRNGKey(0))
    assert bool(hopeless.failed) and STOP_REASONS[int(hopeless.stop)] == "no finite start"
    assert np.all(np.isinf(np.asarray(hopeless.log_q)))


def test_the_funnel_neck_is_not_a_maximum():
    """M3: on centred eight schools the fixed-draw ELBO converges at a finite scale, while L-BFGS
    on log p itself (Pathfinder's path) runs down the neck."""
    x0 = jax.random.uniform(jax.random.PRNGKey(0), (10,), minval=-2, maxval=2)
    res = dadvi(_eight_schools, x0, jax.random.PRNGKey(0))
    assert STOP_REASONS[int(res.stop)] == "tolerance" and int(res.n_iter) < 500
    assert float(res.center[1]) > 0.0                               # log tau, posterior median ~1
    path = single_path(_eight_schools, x0, jax.random.PRNGKey(0), max_iters=300)
    assert float(path.last[1]) < -5.0                               # control: down the neck


def test_options_are_validated():
    assert dadvi_options({})["elbo_draws"] == 5
    assert dadvi_options({"covariance": "lr"})["covariance"] == "lr"
    with pytest.raises(ValueError, match="unbounded"):
        dadvi_options({"elbo_draws": 1})
    for bad, match in (({"covariance": "full"}, "covariance must be"),
                       ({"standardize": 1}, "standardize must be a bool"),
                       ({"rel_tol": -1.0}, "rel_tol must be"), ({"history": 0}, "history must"),
                       ({"c1": 0.9, "c2": 0.5}, "Wolfe"), ({"draws": 2.5}, "draws must")):
        with pytest.raises(ValueError, match=match):
            dadvi_options(bad)
    with pytest.raises(TypeError, match="unknown DADVI option"):
        dadvi_options({"learning_rate": 1e-3})


# --------------------------------------------------------------------------- #
# the position initializer                                                    #
# --------------------------------------------------------------------------- #


def _narrow():
    """Far from the U(-2, 2) box: x ~ N(3, 0.1^2) per coordinate, and a log-normal scale."""
    return Model([EuclideanParameter("x", (3,)), PositiveParameter("s", ())],
                 {"lp": lambda p: (-0.5 * jnp.sum(((p["x"] - 3.0) / 0.1) ** 2)
                                   - 0.5 * jnp.log(p["s"]) ** 2)})


def test_the_slot_value_starts_the_chain_in_the_posterior():
    spec = analyze(_narrow())
    spec.position_init = "dadvi"
    assert "initialize()   dadvi + line_search" in str(spec)
    s = spec.build(seed=0)
    assert "DADVIInit" in [c.__name__ for c in type(s).__mro__]
    s.initialize()
    x = np.asarray(s.state.sample)[:3]
    assert np.max(np.abs(x - 3.0)) < 0.5                         # within 5 sd; U(-2,2) is 10+ off
    rep = s.init_report
    assert rep["method"] == "dadvi" and rep["failed"] == 0 and np.isfinite(rep["khat"])
    assert rep["stop"] == "tolerance" and rep["lr_failed"] == 0
    assert rep["readout"] == s._dadvi_options["covariance"]
    assert rep["grad_evals"] >= 5 * rep["n_iter"] and np.isfinite(rep["elbo_out"])
    s.warmup(200)
    s.sample(50)


def test_slot_options_reach_the_mixin_and_are_validated():
    spec = analyze(_narrow())
    spec.position_init, spec.position_init_params = "dadvi", {"covariance": "diag",
                                                              "max_iters": 3}
    s = spec.build(seed=1)
    assert (s._dadvi_options["covariance"], s._dadvi_options["max_iters"]) == ("diag", 3)
    s.initialize()
    assert s.init_report["n_iter"] == 3 and s.init_report["stop"] == "max_iters"
    spec.position_init_params = {"learning_rate": 0.01}
    with pytest.raises(ValueError, match="unknown position_init_params"):
        spec.build(seed=1)                          # an ADVI option
    spec.position_init_params = {"elbo_draws": 1}
    with pytest.raises(ValueError, match="unbounded"):
        spec.build(seed=1)
    spec.position_init_params, spec.algo_kwargs = {}, {"init_dadvi_draws": 3}
    with pytest.raises(ValueError, match="which the factory sets itself"):
        spec.build(seed=1)


def test_the_start_is_pathfinders_and_fixed_parameters_are_kept():
    """DADVI and Pathfinder draw their pi_0 start from the same substream; a supplied parameter is
    kept bit-exactly."""
    import mimcs.vi.deterministic as dt
    import mimcs.vi.pathfinder as pf
    seen = {}

    for module, name, tag in ((dt, "dadvi", "dadvi"), (pf, "single_path", "pathfinder")):
        real = getattr(module, name)

        def wrapped(logp, x0, key, _real=real, _tag=tag, **kw):
            seen[_tag] = np.asarray(x0)
            return _real(logp, x0, key, **kw)
        setattr(module, name, wrapped)
        try:
            spec = analyze(_narrow())
            spec.position_init = tag
            s = spec.build(seed=4, init={"s": np.float32(1.7)}).initialize()
        finally:
            setattr(module, name, real)
        assert np.asarray(s.state.sample)[3] == np.float32(1.7)
    assert seen["dadvi"].shape == (3,) and np.array_equal(seen["dadvi"], seen["pathfinder"])


def test_under_tempering_each_rung_starts_at_its_own_draw():
    spec = analyze(_narrow())
    spec.base, spec.tempering_params = "pt_nuts", {"n_temperatures": 3}
    spec.position_init = "dadvi"
    s = spec.build(seed=0).initialize()
    rows = np.asarray(s.state.sample).reshape(3, -1)
    assert not np.allclose(rows[0], rows[1]) and not np.allclose(rows[1], rows[2])
    assert np.max(np.abs(rows[:, :3] - 3.0)) < 0.5
    assert np.all(np.isfinite(np.asarray(s.state.log_prob)))


def test_a_static_base_refuses_dadvi():
    m = Model([], {"p": lambda v: -jnp.sum(v["z"])},
              discrete_parameters=[IntegerParameter("z", (4,), lower=0, upper=2)])
    spec = analyze(m)
    spec.position_init = "dadvi"
    with pytest.raises(ValueError, match="no continuous coordinate"):
        spec.build(seed=0)
