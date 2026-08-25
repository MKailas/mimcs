"""Tests for constant mass-matrix adaptation.

Two diagonal mass adaptations:

* ``mass_adapt="covariance"`` (default) --- the empirical target covariance via a
  stochastic-approximation running estimate (shared ``(n+n0)^{-kappa}`` schedule, k=0.75,
  n0=5). On a Gaussian ``M^{-1} -> diag(Cov)``.
* ``mass_adapt="score"`` --- the diagonal mass fit to the *score* (gradient) covariance by
  SGD on the KL objective, i.e. exactly the constant (``depends_on=[None]``) learned block
  metric of explicit RMHMC. Its minimiser is ``M_d = E[g_d^2]`` (the precision on a
  Gaussian). The KL-SGD has a mild upward bias (shared with ``MetricAdaptation``) from the
  heavy-tailed ``g^2``, so it is checked loosely / against the block metric, not exactly.

Seeds are fixed, so pass/fail is deterministic.
"""

import numpy as np

import pytest

from mimcs.testing import correlated_gaussian, hmc, nuts, evaluate, explicit_rmhmc
from mimcs.hmc import Exp


def test_covariance_mass_converges_to_target_covariance():
    """The (SA) empirical-covariance mass: ``M^{-1} -> diag(Cov)`` on a Gaussian."""
    cov = np.array([[2.0, 1.4], [1.4, 1.5]])
    prob = correlated_gaussian(mean=[1.0, -2.0], cov=cov)
    s = hmc(n_leapfrog=20, step_size=0.5, mass_adapt="covariance")(prob.model, seed=0)
    s.warmup(4000)
    inv_mass = np.asarray(s.state.ham_params["T"])
    ratio = inv_mass / np.diag(cov)
    assert np.all((ratio > 0.8) & (ratio < 1.25)), \
        f"inv_mass {inv_mass} not near diag(cov) {np.diag(cov)}"


def test_dense_rank_one_update_matches_full_sa_covariance():
    """The dense mass keeps only the Cholesky factor, updated rank-one each step; it must
    reproduce the full-matrix stochastic-approximation covariance to the float32 floor."""
    import jax.numpy as jnp
    from mimcs.adaptation._cholesky import chol_update
    from mimcs.adaptation._stochastic import rm_gain

    rng = np.random.default_rng(0)
    d = 4
    true = np.array([[2.0, 1.4, 0.3, 0.1], [1.4, 1.5, 0.2, 0.0],
                     [0.3, 0.2, 1.0, 0.4], [0.1, 0.0, 0.4, 1.2]])
    xs = rng.standard_normal((1500, d)) @ np.linalg.cholesky(true).T + np.array([1, -2, 0, 3])

    mean = np.zeros(d)
    cov = np.eye(d)                 # full-matrix reference
    lmean = np.zeros(d)
    chol = jnp.eye(d)               # rank-one Cholesky
    for n, x in enumerate(xs, 1):
        g = rm_gain(n)
        dc = x - mean
        mean = mean + g * dc
        cov = (1 - g) * cov + g * np.outer(dc, dc)
        dl = x - lmean
        lmean = lmean + g * dl
        chol = jnp.sqrt(1 - g) * chol_update(chol, jnp.sqrt(g / (1 - g)) * jnp.asarray(dl))

    LLt = np.asarray(chol @ chol.T)
    assert np.abs(LLt - cov).max() < 1e-4


def test_dense_mass_samples_gaussian(artifacts_dir):
    """End-to-end: HMC with the dense (rank-one updated) mass samples a Gaussian."""
    prob = correlated_gaussian(mean=[1.0, -2.0], cov=[[2.0, 1.4], [1.4, 1.5]])
    report = evaluate(
        prob, {"dense": hmc(n_leapfrog=5, step_size=0.5, metric="dense")},
        n_warmup=2000, n_samples=8000, seed=0,
        out_dir=str(artifacts_dir / "dense_mass_gaussian"))
    print("\n" + report.summary())
    report.assert_correct()


def test_score_mass_matches_constant_block_metric():
    """The score-covariance KL mass is exactly the constant (depends_on=[None]) learned
    block metric: the two learned diagonal masses must agree."""
    cov = np.array([[2.0, 1.4], [1.4, 1.5]])
    prob = correlated_gaussian(mean=[1.0, -2.0], cov=cov)
    n = 8000

    s = hmc(n_leapfrog=20, step_size=0.5, target_accept=0.8,
            mass_adapt="score")(prob.model, seed=0)
    s.warmup(n)
    mass_score = np.exp(s._sm_blocks["T"].log_mass)

    b = explicit_rmhmc(metrics={"x": Exp()},
                       n_leapfrog=20, step_size=0.5, target_accept=0.8)(prob.model, seed=0)
    b.warmup(n)
    mass_block = np.exp(np.asarray(b.state.ham_params["x"]["b"]))

    rel = np.abs(mass_score / mass_block - 1.0)
    assert np.all(rel < 0.25), \
        f"score mass {mass_score} disagrees with block metric mass {mass_block}"


def test_score_mass_in_ballpark_of_score_covariance():
    """The score mass targets ``E[g^2] = diag(precision)`` on a Gaussian (loose bounds: the
    KL-SGD has a mild documented upward bias)."""
    cov = np.array([[2.0, 1.4], [1.4, 1.5]])
    precision = np.linalg.inv(cov)
    prob = correlated_gaussian(mean=[1.0, -2.0], cov=cov)
    s = hmc(n_leapfrog=20, step_size=0.5, mass_adapt="score")(prob.model, seed=0)
    s.warmup(4000)
    ratio = np.exp(s._sm_blocks["T"].log_mass) / np.diag(precision)
    assert np.all((ratio > 0.5) & (ratio < 2.5)), \
        f"score mass ratio to diag(precision) {ratio} out of range"


def test_score_mass_mean_grad_vanishes_at_stationarity():
    """Gradient mean estimation: the score is mean-zero at stationarity (integration by
    parts), so the running ``mean_grad`` decays to ~0 and the centring becomes inert."""
    prob = correlated_gaussian(mean=[10.0, 10.0], cov=[[1.0, 0.0], [0.0, 1.0]])
    s = hmc(n_leapfrog=20, step_size=0.5, mass_adapt="score")(prob.model, seed=0)
    s.warmup(4000)
    mean_grad = s._sm_blocks["T"].mean_grad
    assert np.abs(mean_grad).max() < 0.15, mean_grad


def test_score_mass_centering_captures_the_transient():
    """Early on, a chain far from a distant mode has a systematic downhill score. Mean estimation
    (on by default) captures it in a running ``mean_grad``, so the mass fits the score *covariance*
    rather than its transient-inflated second moment. Here we check the transient is captured and
    the centred mass stays near the true unit precision (the mass matrix on cov = I)."""
    prob = correlated_gaussian(mean=[10.0, 10.0], cov=[[1.0, 0.0], [0.0, 1.0]])
    on = hmc(n_leapfrog=20, step_size=0.5, mass_adapt="score")(prob.model, seed=0)
    on.warmup(60)
    off = hmc(n_leapfrog=20, step_size=0.5, mass_adapt="score",
              score_mass_center_grad=False)(prob.model, seed=0)
    off.warmup(60)

    on_blk, off_blk = on._sm_blocks["T"], off._sm_blocks["T"]
    assert np.abs(on_blk.mean_grad).max() > 0.1        # centring captures the transient offset
    assert np.all(off_blk.mean_grad == 0.0)            # toggled off -> never updated
    assert np.all(np.abs(np.exp(on_blk.log_mass) - 1.0) < 0.5)   # centred mass ~ true unit precision


def test_score_mass_samples_gaussian(artifacts_dir):
    """HMC with the score-covariance mass adaptation samples a Gaussian correctly, and the
    harness records & plots the clip-threshold trajectory (``_sm_log_clip``)."""
    import os
    prob = correlated_gaussian(mean=[1.0, -2.0], cov=[[2.0, 1.4], [1.4, 1.5]])
    out_dir = str(artifacts_dir / "score_mass_gaussian")
    report = evaluate(
        prob, {"score_mass": hmc(n_leapfrog=20, step_size=0.5, mass_adapt="score")},
        n_warmup=3000, n_samples=8000, seed=0, out_dir=out_dir)
    print("\n" + report.summary())
    report.assert_correct()
    log_clip = report.outputs["score_mass"].warmup_log_clip
    assert log_clip is not None and log_clip.shape == (3000,)
    assert os.path.exists(os.path.join(out_dir, "clip_threshold_score_mass.png"))


def test_score_mass_clip_threshold_recorded():
    """The clip threshold is recorded once per warmup step (``warmup_log_clip()`` reports the
    cross-coordinate mean of each iteration's per-coordinate thresholds; its update uses the
    plain gain ``eta_n``). A diagonal block's threshold is now PER COORDINATE, initialised at
    ``log 1 = 0`` (each coordinate's own gradient is a single scalar, so its clip "dimension" is
    1 -- see ``mimcs/adaptation/score_mass.py``), not the old whole-block ``log d``."""
    prob = correlated_gaussian(mean=[1.0, -2.0], cov=[[2.0, 1.4], [1.4, 1.5]])
    s = hmc(n_leapfrog=20, step_size=0.5, mass_adapt="score")(prob.model, seed=0)
    s.warmup(500)
    traj = s.warmup_log_clip()
    assert traj.shape == (500,)
    assert abs(traj[0]) < 0.3                                # threshold initialised at log 1 = 0
    s.sample(200)
    assert s.warmup_log_clip().shape == (500,)              # sampling does not extend it


def test_dense_score_mass_recovers_score_covariance():
    """Dense score-covariance mass: the mass ``M`` fits ``Cov(score) = precision`` (on a
    Gaussian), so the stored Cholesky ``L`` of ``M^{-1}`` gives ``L L^T -> cov`` (loose bounds:
    the KL-SGD has the same mild upward bias on the mass as the diagonal version)."""
    cov = np.array([[2.0, 1.4], [1.4, 1.5]])
    prob = correlated_gaussian(mean=[1.0, -2.0], cov=cov)
    s = hmc(n_leapfrog=20, step_size=0.5, metric="dense", mass_adapt="score")(prob.model, seed=0)
    s.warmup(4000)
    L = np.asarray(s.state.ham_params["T"])         # lower Cholesky of M^{-1}
    assert L.shape == (2, 2) and np.allclose(np.triu(L, 1), 0.0)    # lower-triangular
    assert np.abs(L @ L.T - cov).max() < 0.6, f"L L^T {L @ L.T} not near cov {cov}"


def test_dense_score_mass_samples_gaussian(artifacts_dir):
    """NUTS with the dense score-covariance mass adaptation samples a Gaussian correctly.
    (NUTS, not fixed-length HMC: a well-whitened dense mass makes the target near-isotropic,
    where fixed-trajectory HMC is resonance-prone -- see test_randomized_hmc.)"""
    prob = correlated_gaussian(mean=[1.0, -2.0], cov=[[2.0, 1.4], [1.4, 1.5]])
    report = evaluate(
        prob, {"dense_score": nuts(metric="dense", mass_adapt="score")},
        n_warmup=3000, n_samples=8000, seed=0, out_dir=str(artifacts_dir / "dense_score_mass"))
    print("\n" + report.summary())
    report.assert_correct()


def test_mass_polyak_defaults_off_and_toggleable():
    """Mass smoothing (Polyak/EMA) is off by default -- not needed for our targets and it slows
    the mass's learning -- but is available via mass_polyak for both mass schemes."""
    prob = correlated_gaussian(mean=[1.0, -2.0], cov=[[2.0, 1.4], [1.4, 1.5]])
    assert hmc(mass_adapt="score")(prob.model, 0)._sm_polyak is False             # default off
    assert hmc(mass_adapt="score", mass_polyak=True)(prob.model, 0)._sm_polyak is True
    assert hmc(mass_adapt="covariance")(prob.model, 0)._mm_polyak is False        # default off
    assert hmc(mass_adapt="covariance", mass_polyak=True)(prob.model, 0)._mm_polyak is True


def test_polyak_freezes_the_raw_iterate_only_at_sampling():
    """During warmup the state holds the raw SGD iterate (so warmup dynamics are unperturbed);
    the Polyak average is frozen in only when sampling begins."""
    prob = correlated_gaussian(mean=[1.0, -2.0], cov=[[2.0, 1.4], [1.4, 1.5]])
    s = nuts(metric="dense", mass_adapt="score", mass_polyak=True)(prob.model, seed=0)
    s.warmup(2000)
    raw = np.asarray(s.state.ham_params["T"]).copy()   # raw iterate during warmup
    s.sample(1)                                         # _finalize_hooks freezes the average
    avg = np.asarray(s.state.ham_params["T"]).copy()
    assert np.abs(raw - avg).max() > 1e-3              # the frozen mass is the (different) average


def test_polyak_stabilises_the_frozen_mass_estimate():
    """Polyak--Ruppert averaging's payoff: the mass frozen for sampling barely moves under
    further warmup, where the raw SGD iterate keeps jittering. Shown on the dense score mass
    (noisiest case); sampling finalises the average (or, off, the raw iterate)."""
    prob = correlated_gaussian(mean=[1.0, -2.0], cov=[[2.0, 1.4], [1.4, 1.5]])

    def drift(polyak):
        s = nuts(metric="dense", mass_adapt="score", mass_polyak=polyak)(prob.model, seed=0)
        s.warmup(2000); s.sample(1)
        L1 = np.asarray(s.state.ham_params["T"]).copy()
        s.warmup(400); s.sample(1)                     # 400 more warmup steps, re-finalise
        L2 = np.asarray(s.state.ham_params["T"]).copy()
        return np.abs(L2 - L1).max()

    on, off = drift(True), drift(False)
    assert on < 0.8 * off, f"Polyak drift {on:.4f} not < raw-iterate drift {off:.4f}"
