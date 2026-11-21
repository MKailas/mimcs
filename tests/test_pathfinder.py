"""Pathfinder (``mimcs.vi``) and its position initializer (``spec.position_init = "pathfinder" /
"multipath"``).

The algebra is checked against slow, obviously-correct references (dense BFGS, a dense normal
density, ``arviz_stats``' PSIS), each with a control so the check cannot pass vacuously. Seeds are
fixed, so pass/fail is deterministic.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mimcs.factory import analyze
from mimcs.model import BoundedParameter, EuclideanParameter, IntegerParameter, Model, \
    PositiveParameter
from mimcs.vi import multi_path, psis, resample, single_path
from mimcs.vi.pathfinder import _approximation, _draw, _factors


# --------------------------------------------------------------------------- #
# the covariance factors and the sampler (Algorithm 4)                        #
# --------------------------------------------------------------------------- #

def _pairs(n, J, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    hess = A @ A.T + n * np.eye(n)
    S = rng.normal(size=(J, n))
    return S, S @ hess, rng.uniform(0.05, 0.5, size=n)              # rows z = H s


def _dense_bfgs(alpha, S, Z, mask):
    H = np.diag(alpha)
    for s, y, keep in zip(S, Z, mask):
        if keep:
            rho = 1.0 / (s @ y)
            V = np.eye(len(alpha)) - rho * np.outer(y, s)
            H = V.T @ H @ V + rho * np.outer(s, s)
    return H


@pytest.mark.parametrize("mask", [[1, 1, 1, 1], [0, 0, 1, 1], [0, 1, 0, 1], [0, 0, 0, 0]])
def test_the_compact_covariance_is_the_bfgs_inverse_hessian(mask):
    """Sigma = diag(alpha) + beta gamma beta^T (eq. 1-2) is the recursive BFGS inverse update from
    diag(alpha) over the held pairs, padding included (a padded slot must drop out)."""
    S, Z, alpha = _pairs(9, 4)
    mask = np.asarray(mask, float)
    beta, gamma = _factors(jnp.asarray(alpha, jnp.float32), jnp.asarray(S, jnp.float32),
                           jnp.asarray(Z, jnp.float32), jnp.asarray(mask, jnp.float32))
    sigma = np.diag(alpha) + np.asarray(beta) @ np.asarray(gamma) @ np.asarray(beta).T
    ref = _dense_bfgs(alpha, S, Z, mask)
    assert np.max(np.abs(sigma - ref)) / np.max(np.abs(ref)) < 1e-4


@pytest.mark.parametrize("n", [11, 5])                       # factored (n > 2J) and dense
def test_draws_and_log_q_match_a_dense_normal(n):
    """``log q`` of the factored sampler equals the dense normal log density under Sigma, and the
    draws have covariance Sigma. CONTROL: the diagonal alone (dropping the low-rank part) gives a
    visibly different density."""
    from jax.scipy.stats import multivariate_normal as mvn
    J = 4
    S, Z, alpha = _pairs(n, J, seed=1)
    mask = np.asarray([0, 1, 1, 1], float)
    a, s, z, m = (jnp.asarray(v, jnp.float32) for v in (alpha, S, Z, mask))
    x = jnp.asarray(np.random.default_rng(2).normal(size=n), jnp.float32)
    g = jnp.asarray(np.random.default_rng(3).normal(size=n), jnp.float32)
    phi, log_q = _draw(jax.random.PRNGKey(0), 40000, x, g, a, s, z, m)
    beta, gamma = _factors(a, s, z, m)
    sigma = jnp.diag(a) + beta @ gamma @ beta.T
    mu, _, _ = _approximation(x, g, a, s, z, m)
    ref = mvn.logpdf(phi[:200], mu, sigma)
    assert float(jnp.max(jnp.abs(log_q[:200] - ref))) < 2e-3
    cov = np.cov(np.asarray(phi).T)
    sig = np.asarray(sigma)
    zscore = (cov - sig) / np.sqrt((sig ** 2 + np.outer(np.diag(sig), np.diag(sig))) / 40000)
    assert np.max(np.abs(zscore)) < 5
    bad = mvn.logpdf(phi[:200], mu, jnp.diag(a))                  # control
    assert float(jnp.max(jnp.abs(log_q[:200] - bad))) > 0.1


# --------------------------------------------------------------------------- #
# PSIS                                                                        #
# --------------------------------------------------------------------------- #

def test_psis_matches_the_reference_implementation():
    """Smoothed weights (normalised) and k-hat equal ``arviz_stats``' PSIS on heavy-tailed weights.
    The reference is a test-only oracle, not a dependency."""
    arviz_stats = pytest.importorskip("arviz_stats.base")
    from scipy.special import logsumexp
    rng = np.random.default_rng(1)
    for size in (60, 300, 2000):
        for _ in range(5):
            lw = rng.standard_t(df=rng.uniform(1.5, 8), size=size) * rng.uniform(0.3, 3)
            ref, k_ref = arviz_stats.array_stats.psislw(-lw)        # loo convention: -log w
            got, k = psis(lw)
            assert abs(float(k_ref) - k) < 1e-10
            assert np.max(np.abs((ref - logsumexp(ref)) - (got - logsumexp(got)))) < 1e-10


def test_psis_recovers_a_known_tail_and_leaves_zero_weights_alone():
    rng = np.random.default_rng(2)
    for k_true in (0.3, 0.7):
        ks = [psis(np.log(rng.pareto(1.0 / k_true, 4000) + 1.0))[1] for _ in range(9)]
        assert abs(np.median(ks) - k_true) < 0.06, (k_true, ks)
    lw = np.r_[rng.normal(size=500), [-np.inf] * 10, [np.nan] * 3]
    smoothed, k = psis(lw)
    assert np.all(np.isneginf(smoothed[500:])) and np.isfinite(k)
    idx = resample(np.random.default_rng(0), smoothed, 5000)
    assert idx.max() < 500                                         # a zero weight is never drawn


# --------------------------------------------------------------------------- #
# single- and multi-path Pathfinder on flat targets                           #
# --------------------------------------------------------------------------- #

def _gaussian(d, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(d, d))
    A = A @ A.T / d + np.diag(np.exp(rng.uniform(-1.5, 1.5, d)))
    b = rng.normal(size=d) * 3
    logz = (0.5 * b @ np.linalg.solve(A, b) + 0.5 * d * math.log(2 * math.pi)
            - 0.5 * np.linalg.slogdet(A)[1])
    Aj, bj = jnp.asarray(A, jnp.float32), jnp.asarray(b, jnp.float32)
    return (lambda x: -0.5 * x @ Aj @ x + bj @ x), np.linalg.solve(A, b), np.linalg.inv(A), logz


def test_single_path_approximates_a_gaussian():
    """On a Gaussian the selected approximation is close but, as the paper says, not exact: the
    K = 5 ELBO estimate is noisy, so the argmax often lands a few iterations before the end, with
    variances measured 0.5--1.9x the truth (f32 and x64 alike, six seeds). So, over four seeds:
    the mean is right, the variances within a factor of 2.5, and a fresh ELBO estimate is below
    log Z (it is a bound) and within 1 nat of it."""
    d = 6
    logp, mean, cov, logz = _gaussian(d)
    for seed in range(4):
        r = single_path(logp, np.random.default_rng(seed).uniform(-2, 2, d),
                        jax.random.PRNGKey(seed), history=d, draws=4000)
        draws = np.asarray(r.draws)
        assert not bool(r.failed) and int(r.l_star) >= 1
        assert np.max(np.abs(draws.mean(0) - mean) / np.sqrt(np.diag(cov))) < 0.5, seed
        ratio = np.var(draws, axis=0) / np.diag(cov)
        assert np.all((ratio > 0.4) & (ratio < 2.5)), (seed, ratio)
        fresh = float(np.mean(np.asarray(r.log_p) - np.asarray(r.log_q)))
        assert logz - 1.0 < fresh < logz + 0.02, (seed, fresh, logz)
        # the evaluation accounting: K per iteration plus the M final draws; >= 1 gradient each
        assert int(r.n_value) == 5 * int(r.n_iter) + 4000
        assert int(r.n_grad) >= int(r.n_iter) + 1


def test_on_a_funnel_the_selected_approximation_comes_before_the_pole():
    """The paper's Fig. 2: L-BFGS heads for the funnel's pole, the ELBO peaks on the way."""
    def funnel(x):
        v, z = x[0], x[1:]
        return -v ** 2 / 18 - 0.5 * jnp.sum(z ** 2) * jnp.exp(-v) - 0.5 * z.shape[0] * v
    r = single_path(funnel, np.random.default_rng(0).uniform(-2, 2, 10), jax.random.PRNGKey(2))
    assert not bool(r.failed)
    assert int(r.l_star) < int(r.n_iter), (int(r.l_star), int(r.n_iter))


def _funnel(x):
    v, z = x[0], x[1:]
    return -v ** 2 / 18 - 0.5 * jnp.sum(z ** 2) * jnp.exp(-v) - 0.5 * z.shape[0] * v


def test_patience_stops_a_path_that_has_stopped_improving():
    """On the funnel the uncapped path runs far past its ELBO peak (in float32 to the line search's
    failure near the pole, at 175 iterations; to max_iters in x64). With patience P it stops exactly
    P iterations after its best so far; the path is the same prefix, so its pick is at most as good
    as the uncapped one. ``patience=None`` is the paper's run, bit for bit."""
    x0, key = np.random.default_rng(0).uniform(-2, 2, 10), jax.random.PRNGKey(2)
    full = single_path(_funnel, x0, key)
    same = single_path(_funnel, x0, key, patience=None)
    assert all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(full, same))
    assert int(full.n_iter) > int(full.l_star) + 50         # the control: runs long past its peak
    for P in (5, 20):
        r = single_path(_funnel, x0, key, patience=P)
        assert int(r.n_iter) == int(r.l_star) + P
        assert int(r.l_star) <= int(full.l_star) and float(r.elbo) <= float(full.elbo)
        assert int(r.n_iter) < int(full.n_iter) // 4        # the point: a fraction of the cost


def test_a_path_that_cannot_start_fails_with_zero_weight():
    """No finite gradient at the start: the result is the start itself with ``log q = +inf``."""
    r = single_path(lambda x: jnp.sum(jnp.sqrt(x)), np.asarray([-1.0, 1.0]),
                    jax.random.PRNGKey(0), draws=7)
    assert bool(r.failed) and int(r.n_iter) == 0
    assert np.all(np.isposinf(np.asarray(r.log_q)))
    assert np.allclose(np.asarray(r.draws), [[-1.0, 1.0]] * 7)


def test_multi_path_resamples_by_importance_weight():
    """Two well-separated modes of mass 0.8 / 0.2. Each path finds one mode; the resampled share of
    the major mode must follow the importance weights --- 0.8 per path there against 0.2 --- not
    the share of paths that happened to land on it."""
    def bimodal(x):
        return jnp.logaddexp(math.log(0.8) - 0.5 * jnp.sum((x - 3) ** 2),
                             math.log(0.2) - 0.5 * jnp.sum((x + 3) ** 2))
    rng = np.random.default_rng(0)
    x0s = np.r_[rng.uniform(1, 4, (4, 2)), rng.uniform(-4, -1, (12, 2))]
    m = multi_path(bimodal, x0s, jax.random.PRNGKey(3), resample_size=4000)
    on_major = np.asarray(m.paths.draws)[:, 0, 0] > 0
    expected = 0.8 * on_major.sum() / (0.8 * on_major.sum() + 0.2 * (~on_major).sum())
    share = float(np.mean(np.asarray(m.draws)[:, 0] > 0))
    assert on_major.sum() == 4                       # each path stayed in its basin
    assert abs(share - expected) < 0.04, (share, expected)
    assert abs(share - on_major.mean()) > 0.2        # control: not the share of paths
    assert m.distinct > 100


def test_options_are_validated():
    with pytest.raises(TypeError, match="unknown Pathfinder option"):
        single_path(lambda x: -jnp.sum(x ** 2), np.zeros(2), jax.random.PRNGKey(0), histroy=3)
    with pytest.raises(ValueError, match="c1 < c2"):
        single_path(lambda x: -jnp.sum(x ** 2), np.zeros(2), jax.random.PRNGKey(0), c1=0.9, c2=0.5)
    for bad in (0, 2.5, "10"):
        with pytest.raises(ValueError, match="patience must be an integer >= 1 or None"):
            single_path(lambda x: -jnp.sum(x ** 2), np.zeros(2), jax.random.PRNGKey(0),
                        patience=bad)


# --------------------------------------------------------------------------- #
# the initializer: spec.position_init = "pathfinder" / "multipath"            #
# --------------------------------------------------------------------------- #

def _narrow():
    """Far from the U(-2, 2) box: x ~ N(5, 0.1^2) per coordinate, and a log-normal scale."""
    return Model([EuclideanParameter("x", (3,)), PositiveParameter("s", ())],
                 {"lp": lambda p: (-0.5 * jnp.sum(((p["x"] - 5.0) / 0.1) ** 2)
                                   - 0.5 * jnp.log(p["s"]) ** 2)})


@pytest.mark.parametrize("kind", ["pathfinder", "multipath"])
def test_the_slot_values_start_the_chain_in_the_posterior(kind):
    spec = analyze(_narrow())
    spec.position_init = kind
    assert f"initialize()   {kind} + line_search" in str(spec)
    s = spec.build(seed=0)
    assert "PathfinderInit" in [c.__name__ for c in type(s).__mro__]
    s.initialize()
    x = np.asarray(s.state.sample)[:3]
    assert np.max(np.abs(x - 5.0)) < 0.5                         # within 5 sd; U(-2,2) is 30 sd off
    rep = s.init_report
    assert rep["method"] == kind and rep["paths"] == (20 if kind == "multipath" else 1)
    assert rep["grad_evals"] > 0 and rep["failed"] == 0 and np.isfinite(rep["khat"])
    s.warmup(200)
    s.sample(50)


def test_slot_options_reach_the_mixin_and_are_validated():
    spec = analyze(_narrow())
    spec.position_init, spec.position_init_params = "multipath", {"num_paths": 3, "history": 4}
    assert "multipath(num_paths=3, history=4)" in str(spec)
    s = spec.build(seed=1)
    assert (s._pf_paths, s._pf_options["history"]) == (3, 4)
    s.initialize()
    assert len(s.init_report["n_iter"]) == 3
    spec.position_init, spec.position_init_params = "pathfinder", {"num_paths": 3}
    with pytest.raises(ValueError, match="unknown position_init_params"):
        spec.build(seed=1)                          # single-path takes no num_paths
    spec.position_init_params = {"history": 0}
    with pytest.raises(ValueError, match="history must be an integer >= 1"):
        spec.build(seed=1)
    spec.position_init_params, spec.algo_kwargs = {}, {"init_pathfinder_history": 3}
    with pytest.raises(ValueError, match="which the factory sets itself"):
        spec.build(seed=1)


def test_a_fixed_child_of_a_drawn_parent_is_kept_and_the_target_is_its_conditional():
    """``x in (0, sigma)`` with ``x`` supplied: Pathfinder fits sigma (and m) given x held in
    *sample* space. The objective must be the sample-space conditional --- the coordinate density
    minus x's own chart Jacobian, which moves with sigma. CONTROL: without the subtraction the
    two differ by a sigma-dependent amount.

    That conditional is *truncated* (sigma > max x), so every normal approximation puts mass where
    the density is zero and every ELBO estimate is -inf: the path "fails" in the paper's sense and
    the start is the last L-BFGS iterate --- the conditional mode, a fine start, inside the
    support."""
    def lp(p):
        return (-0.5 * jnp.log(p["sigma"]) ** 2 - 3.0 * jnp.log(p["sigma"])
                - 0.5 * jnp.sum(p["m"] ** 2))
    m = Model([PositiveParameter("sigma"), BoundedParameter("x", (3,), lower=0.0, upper="sigma"),
               EuclideanParameter("m", (2,))], {"lp": lp})
    x0 = np.asarray([0.9, 0.6, 0.4], np.float32)
    spec = analyze(m)
    spec.position_init = "pathfinder"
    s = spec.build(seed=3, init={"x": x0}).initialize()
    sample = np.asarray(s.state.sample)
    assert np.array_equal(sample[1:4], x0)                       # bit-exact
    assert sample[0] > x0.max()                                  # sigma in support
    assert s.init_report["failed"] == 1                          # truncated target: see above

    h, idx = m.init_chart_hyperparams(), m.init_chart_indices()
    base = jnp.asarray(s.state.coordinate)
    gaps = []
    for c_sigma in (0.0, 0.5, 1.2):                              # sigma > max(x) = 0.9
        c, smp = m.pin_fixed(base.at[0].set(c_sigma), jnp.asarray(sample), {"x"}, h, idx)
        ours = m.log_prob_at_coordinate(c, h, idx) - m.log_jacobian(c, h, idx, {"x"})
        sig = math.exp(c_sigma)                                  # log chart: d sigma / dc = sigma
        direct = (float(lp({"sigma": sig, "m": np.asarray(smp)[4:]})) + c_sigma)
        assert abs(float(ours) - direct) < 1e-4, (c_sigma, float(ours), direct)
        gaps.append(float(m.log_prob_at_coordinate(c, h, idx)) - direct)
    assert np.ptp(gaps) > 0.5                                    # control: the Jacobian moves


def test_under_tempering_each_rung_starts_at_its_own_draw():
    spec = analyze(_narrow())
    spec.base, spec.tempering_params = "pt_nuts", {"n_temperatures": 3}
    spec.position_init = "pathfinder"
    s = spec.build(seed=0).initialize()
    rows = np.asarray(s.state.sample).reshape(3, -1)
    assert not np.allclose(rows[0], rows[1]) and not np.allclose(rows[1], rows[2])
    assert np.max(np.abs(rows[:, :3] - 5.0)) < 0.5
    assert np.all(np.isfinite(np.asarray(s.state.log_prob)))


def test_labels_are_drawn_first_and_conditioned_on():
    """A discrete mixture: the sweep's label draw runs before Pathfinder, which fits ``mu`` given
    those labels; supplied labels are kept."""
    rng = np.random.default_rng(0)
    y = np.concatenate([rng.normal(3.0 * (j - 1), 1.0, 6) for j in range(3)])

    def lp(v):
        return -0.5 * jnp.sum((y - v["mu"][v["z"]]) ** 2) - 0.5 * jnp.sum(v["mu"] ** 2)
    m = Model([EuclideanParameter("mu", (3,))], {"p": lp},
              discrete_parameters=[IntegerParameter("z", (18,), lower=0, upper=2)])
    z0 = np.repeat([0, 1, 2], 6).astype(np.int32)
    spec = analyze(m)
    spec.position_init = "pathfinder"
    s = spec.build(seed=1, init={"z": z0}).initialize()
    assert np.array_equal(np.asarray(s.state.discrete), z0)
    mu = np.asarray(s.state.sample)
    post_mean = np.asarray([y[z0 == j].sum() / 7 for j in range(3)])  # conjugate, prior N(0,1)
    assert np.max(np.abs(mu - post_mean)) < 1.5                  # posterior sd 1/sqrt(7) ~ 0.38


def test_a_static_base_refuses_pathfinder():
    m = Model([], {"p": lambda v: -jnp.sum(v["z"])},
              discrete_parameters=[IntegerParameter("z", (4,), lower=0, upper=2)])
    spec = analyze(m)
    spec.position_init = "pathfinder"
    with pytest.raises(ValueError, match="no continuous coordinate"):
        spec.build(seed=0)
