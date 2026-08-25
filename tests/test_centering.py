"""Tests for the centering (standardizing) reparametrization of EuclideanParameter.

``EuclideanParameter(centered=True)`` uses the chart ``q = (x - mu) / sigma`` with adaptive
``(mu, sigma)`` fitted by :class:`mimcs.adaptation.CenteringAdaptation` to the chain's running
mean / standard deviation (same diagonal SA estimator and schedule as the mass adaptation).
It standardizes the sample and pairs with the score-covariance mass adaptation.

Seeds are fixed, so pass/fail is deterministic.
"""

import numpy as np
import jax
import jax.numpy as jnp

from mimcs.model import Model, EuclideanParameter, PositiveParameter, IntervalParameter
from mimcs.testing import TargetProblem, evaluate, hmc, nuts


def _centered_gaussian(mean, cov):
    mean = np.asarray(mean, float)
    cov = np.asarray(cov, float)
    precision = jnp.asarray(np.linalg.inv(cov))
    jmean = jnp.asarray(mean)

    def log_post(params):
        d = params["x"] - jmean
        return -0.5 * d @ precision @ d

    model = Model([EuclideanParameter("x", (mean.shape[0],), centered=True)],
                  {"lp": log_post})
    chol = np.linalg.cholesky(cov)
    return TargetProblem(
        name="centered_gaussian", model=model, dim=mean.shape[0],
        labels=[f"x{i}" for i in range(mean.shape[0])],
        exact_sampler=lambda n, rng: mean + rng.standard_normal((n, mean.shape[0])) @ chol.T,
        mean=mean, cov=cov)


def test_centering_chart_jacobian_matches_autodiff():
    """``log_jacobian_det`` must equal ``log|det d(from_coordinate)/dq|`` = ``sum log sigma``."""
    p = EuclideanParameter("x", (3,), centered=True)
    hp = (jnp.array([5.0, -3.0, 0.0]), jnp.array([2.0, 1.4, 0.5]))
    q = jnp.array([0.3, -0.2, 1.1])
    J = jax.jacobian(lambda qq: p.from_coordinate(qq, hp))(q)
    ad = np.log(np.abs(np.linalg.det(np.asarray(J))))
    assert np.isclose(ad, float(p.log_jacobian_det(q, hp)), atol=1e-5)
    assert np.isclose(ad, float(jnp.sum(jnp.log(hp[1]))), atol=1e-5)
    # round trip: to_coordinate o from_coordinate = identity
    x = p.from_coordinate(q, hp)
    assert np.allclose(np.asarray(p.to_coordinate(x, hp)), np.asarray(q), atol=1e-5)


def test_centering_uncentered_is_identity():
    """Without ``centered`` the chart and Jacobian are unchanged (identity)."""
    p = EuclideanParameter("x", (2,))
    assert p.init_hyperparams() is None
    q = jnp.array([1.5, -2.0])
    assert np.allclose(np.asarray(p.to_coordinate(q)), np.asarray(q))
    assert np.allclose(np.asarray(p.from_coordinate(q)), np.asarray(q))
    assert float(p.log_jacobian_det(q)) == 0.0


def test_centering_learns_mean_and_std():
    """``mu -> E[x]`` and ``sigma -> std[x]`` on a Gaussian with large mean and scale."""
    mean = np.array([5.0, -3.0])
    cov = np.array([[4.0, 1.5], [1.5, 2.0]])
    prob = _centered_gaussian(mean, cov)
    s = hmc(n_leapfrog=20, step_size=0.5, mass_adapt="score", center=True,
            init=mean)(prob.model, seed=0)
    s.warmup(4000)
    mu, sigma = (np.asarray(a) for a in s.state.chart_hyperparams[0])
    assert np.all(np.abs(mu - mean) < 0.5), f"mu {mu} far from {mean}"
    assert np.all(np.abs(sigma - np.sqrt(np.diag(cov))) < 0.4), \
        f"sigma {sigma} far from {np.sqrt(np.diag(cov))}"


def test_centering_state_consistent_after_recharting():
    """Recharting keeps the state self-consistent: the stored coordinate maps back to the
    stored sample under the current chart, and ``log_prob`` matches the coordinate target."""
    prob = _centered_gaussian([5.0, -3.0], [[4.0, 1.5], [1.5, 2.0]])
    s = hmc(n_leapfrog=20, step_size=0.5, mass_adapt="score", center=True,
            init=[5.0, -3.0])(prob.model, seed=0)
    s.warmup(1000)
    st = s.state
    x = prob.model.coordinate_to_sample(st.coordinate, st.chart_hyperparams, st.chart_indices)
    assert np.allclose(np.asarray(x), np.asarray(st.sample), atol=1e-4)
    lp = prob.model.log_prob_at_coordinate(st.coordinate, st.chart_hyperparams, st.chart_indices)
    assert np.isclose(float(lp), float(st.log_prob), atol=1e-3)


def test_centering_samples_gaussian(artifacts_dir):
    """HMC with centering + score-mass adaptation samples a large-mean/scale Gaussian."""
    prob = _centered_gaussian([5.0, -3.0], [[4.0, 1.5], [1.5, 2.0]])
    report = evaluate(
        prob,
        {"centered": hmc(n_leapfrog=20, step_size=0.5, mass_adapt="score", center=True,
                         init=[5.0, -3.0])},
        n_warmup=4000, n_samples=8000, seed=0,
        out_dir=str(artifacts_dir / "centering_gaussian"))
    print("\n" + report.summary())
    report.assert_correct()


# --- bounded parameters: centering standardizes the link value ----------------- #


def test_centering_positive_jacobian_matches_autodiff():
    """For a centered positive parameter ``log|dx/dq| = log|dx/dz| + log sigma``."""
    p = PositiveParameter("s", centered=True)
    hp = (jnp.array([2.0]), jnp.array([0.5]))
    q = jnp.array([0.3])
    J = jax.jacobian(lambda qq: p.from_coordinate(qq, hp).reshape(()))(q)[0]
    assert np.isclose(float(jnp.log(jnp.abs(J))), float(p.log_jacobian_det(q, hp)), atol=1e-5)
    x = p.from_coordinate(q, hp)
    assert np.allclose(np.asarray(p.to_coordinate(x, hp)), np.asarray(q), atol=1e-5)


def test_centering_positive_learns_log_moments_and_samples(artifacts_dir):
    """A centered positive parameter standardizes ``log x``: ``mu -> mean(log x)``,
    ``sigma -> std(log x)``; it samples a log-normal (``log s ~ N(2, 0.5)``) correctly."""
    m, sigma = 2.0, 0.5

    def log_post(params):
        log_s = jnp.log(params["s"])
        return -0.5 * ((log_s - m) / sigma) ** 2 - log_s

    model = Model([PositiveParameter("s", centered=True)], {"lp": log_post})
    prob = TargetProblem(
        name="lognormal", model=model, dim=1, labels=["s"],
        exact_sampler=lambda n, rng: np.exp(m + sigma * rng.standard_normal((n, 1))))

    builder = hmc(n_leapfrog=20, step_size=0.5, mass_adapt="score", center=True,
                  init=[float(np.exp(m))])
    s = builder(model, seed=0)
    s.warmup(4000)
    mu, sig = (np.asarray(a) for a in s.state.chart_hyperparams[0])
    assert abs(mu[0] - m) < 0.2 and abs(sig[0] - sigma) < 0.15

    report = evaluate(prob, {"pos": builder}, n_warmup=4000, n_samples=8000, seed=0,
                      out_dir=str(artifacts_dir / "centering_positive"))
    print("\n" + report.summary())
    report.assert_correct()


def test_centering_interval_samples_uniform(artifacts_dir):
    """A centered interval parameter standardizes the logit; it samples a uniform on
    ``(-2, 3)`` correctly (the logit of a uniform is standard logistic, std ~1.81)."""
    model = Model([IntervalParameter("x", -2.0, 3.0, centered=True)],
                  {"lp": lambda params: jnp.zeros(())})
    prob = TargetProblem(
        name="uniform_interval", model=model, dim=1, labels=["x"],
        exact_sampler=lambda n, rng: rng.uniform(-2.0, 3.0, (n, 1)))

    builder = hmc(n_leapfrog=20, step_size=0.5, mass_adapt="score", center=True, init=[0.5])
    s = builder(model, seed=0)
    s.warmup(4000)
    sig = float(np.asarray(s.state.chart_hyperparams[0][1])[0])
    assert 1.4 < sig < 2.2, f"logit std {sig} not near logistic 1.81"

    report = evaluate(prob, {"interval": builder}, n_warmup=4000, n_samples=8000, seed=0,
                      out_dir=str(artifacts_dir / "centering_interval"))
    print("\n" + report.summary())
    report.assert_correct()


# --- robust (median / MAD) centering ------------------------------------------- #


def test_robust_centering_learns_median_and_mad():
    """On a Gaussian the median -> mean and MAD * 1.4826 -> std, so RobustCenteringAdaptation
    standardizes it just like the mean/std version."""
    mean = np.array([5.0, -3.0])
    cov = np.array([[4.0, 1.5], [1.5, 2.0]])
    prob = _centered_gaussian(mean, cov)
    s = hmc(n_leapfrog=20, step_size=0.5, mass_adapt="score", center="robust",
            init=mean)(prob.model, seed=0)
    s.warmup(4000)
    mu, sigma = (np.asarray(a) for a in s.state.chart_hyperparams[0])
    assert np.all(np.abs(mu - mean) < 0.5), f"median {mu} far from {mean}"
    assert np.all(np.abs(sigma - np.sqrt(np.diag(cov))) < 0.5), \
        f"MAD-scale {sigma} far from std {np.sqrt(np.diag(cov))}"


def test_robust_centering_is_stable_on_heavy_tails():
    """The payoff: on t(2) (infinite variance) the empirical std chases tail excursions and is
    inflated and wildly seed-dependent, while the median/MAD scale is stable and near the
    Gaussian-equivalent scale. Compare the seed-to-seed spread of the learned sigma."""
    nu = 2.0
    model = Model([EuclideanParameter("x", (1,), centered=True)],
                  {"lp": lambda p: jnp.sum(-0.5 * (nu + 1.0) * jnp.log1p(p["x"] ** 2 / nu))})

    def learned_sigma(center, seed):
        s = nuts(mass_adapt="score", center=center)(model, seed)
        s.warmup(4000)
        return float(np.asarray(s.state.chart_hyperparams[0][1])[0])

    robust = np.array([learned_sigma("robust", sd) for sd in range(5)])
    empirical = np.array([learned_sigma(True, sd) for sd in range(5)])
    assert np.all((robust > 0.8) & (robust < 1.7)), f"robust sigma {robust} not stable/sensible"
    assert robust.std() < 0.3 * empirical.std(), \
        f"robust spread {robust.std():.3f} not << empirical spread {empirical.std():.3f}"


def test_robust_centering_samples_gaussian(artifacts_dir):
    """End-to-end: HMC with robust centering + score-mass samples a large-mean/scale Gaussian."""
    prob = _centered_gaussian([5.0, -3.0], [[4.0, 1.5], [1.5, 2.0]])
    report = evaluate(
        prob,
        {"robust": hmc(n_leapfrog=20, step_size=0.5, mass_adapt="score", center="robust",
                       init=[5.0, -3.0])},
        n_warmup=4000, n_samples=8000, seed=0,
        out_dir=str(artifacts_dir / "robust_centering_gaussian"))
    print("\n" + report.summary())
    report.assert_correct()
