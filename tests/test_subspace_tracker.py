"""The held-basis low-rank tracker (``mimcs.adaptation._subspace``) --- honest under autocorrelation.

``_Sanger`` moves its subspace at an effective gain ``lr ||x||^2 ~ lr d``, so on autocorrelated
scores it reads ``lambda ~ 1 + (d - 1) rho^2`` even when the target is isotropic
(``tests/experiments/writeups/shape_estimator_bench.md``). The held basis accumulates
``E[x x^T] Q`` with ``Q`` fixed over a block and reads its eigenvalues out of sample, so
autocorrelation costs it effective sample size but no bias. These tests pin:
* the recursion, checked by hand;
* recovery of a spike on iid scores;
* the null on autocorrelated isotropic scores, with ``_Sanger`` failing the same assertion as the
  control;
* power at the same correlation;
* that choosing the tracker touches no RNG.
"""

import numpy as np
import pytest

from mimcs.adaptation._stochastic import rm_gain, DEFAULT_KAPPA, DEFAULT_N0
from mimcs.adaptation._subspace import _HeldBasisTracker, make_tracker, tracker_kwargs
from mimcs.adaptation.lowrank_mass import _Sanger, LowRankAdaptation
from mimcs.hmc import NUTS, LowRankQuadraticKinetic
from mimcs.adaptation import RobbinsMonroStepSize
from mimcs.samplers import make_sampler_class
from mimcs.testing import correlated_gaussian


def _feed(tr, xs, start=51):
    for i, x in enumerate(xs):
        t = start + i
        tr.step(x, rm_gain(t, DEFAULT_N0, DEFAULT_KAPPA), t)
    return tr


def _ar1(d, rho, n, seed, spike=0.0):
    """``n`` draws of an AR(1) whose stationary law is exactly N(0, I + spike u u^T); returns
    (xs, u)."""
    rng = np.random.default_rng(seed)
    u = rng.standard_normal(d); u /= np.linalg.norm(u)
    s = np.sqrt(1.0 + spike) - 1.0
    z = rng.standard_normal(d)
    xs = np.empty((n, d))
    for t in range(n):
        z = rho * z + np.sqrt(1.0 - rho ** 2) * rng.standard_normal(d)
        xs[t] = z + s * u * (u @ z)
    return xs, u


def test_one_block_follows_the_recursion_by_hand():
    """Initial basis = QR of the first m scores; Y, T the RM recursions (T only after `gap`);
    at the block end W, lam are the Ritz pairs of T / wT in the old basis."""
    d, J, block, gap = 7, 2, 12, 3
    rng = np.random.default_rng(0)
    xs = rng.standard_normal((4 + block, d)) * np.linspace(0.5, 2.0, d)   # small: no clipping
    tr = _HeldBasisTracker(d, J, DEFAULT_N0, DEFAULT_KAPPA, clip_frac=0.1, block=block, gap=gap)
    tr._clip.log_clip_w = 50.0                    # threshold e^50: the clip is inert here
    m = 2 * J
    _feed(tr, xs[:m])
    Q0, _ = np.linalg.qr(xs[:m].T)
    np.testing.assert_allclose(tr.Q, Q0)

    Y = np.zeros((d, m)); T = np.zeros((m, m)); wT = 0.0
    for i, x in enumerate(xs[m:m + block]):
        t = 51 + m + i
        g = rm_gain(t, DEFAULT_N0, DEFAULT_KAPPA)
        y = Q0.T @ x
        Y += g * (np.outer(x, y) - Y)
        if i + 1 > gap:
            T += g * (np.outer(y, y) - T); wT = (1 - g) * wT + g
    _feed(tr, xs[m:m + block], start=51 + m)
    ev, U = np.linalg.eigh(T / wT)
    np.testing.assert_allclose(tr.lam, ev[::-1][:J], rtol=1e-12)
    np.testing.assert_allclose(np.abs(tr.W.T @ (Q0 @ U[:, ::-1][:, :J])), np.eye(J), atol=1e-10)
    Qn, _ = np.linalg.qr(Y)                                             # the power step
    np.testing.assert_allclose(np.abs(tr.Q.T @ Qn), np.eye(m), atol=1e-10)


@pytest.mark.parametrize("d", [50, 500])
def test_recovers_a_spike_from_iid_scores(d):
    xs, u = _ar1(d, 0.0, 4000, seed=1, spike=9.0)
    tr = _feed(_HeldBasisTracker(d, 2, DEFAULT_N0, DEFAULT_KAPPA, 0.1), xs)
    j = int(np.argmax(tr.gamma()))
    assert abs(tr.W[:, j] @ u) > 0.9
    assert 6.0 < tr.gamma()[j] < 10.5


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_autocorrelation_is_not_read_as_shape(seed):
    """Isotropic target, IACT 10 (rho = 9/11): the honest answer is gamma = 0. The held basis reads
    ~0; `_Sanger` on the same stream reads tens --- the control that keeps this non-vacuous."""
    d = 100
    xs, _ = _ar1(d, 9.0 / 11.0, 2000, seed)
    held = _feed(_HeldBasisTracker(d, 2, DEFAULT_N0, DEFAULT_KAPPA, 0.1), xs)
    sanger = _feed(_Sanger(d, 2, DEFAULT_N0, DEFAULT_KAPPA, 0.1, 1.0), xs)
    assert held.gamma().max() < 0.5
    assert sanger.gamma().max() > 10.0


def test_a_real_spike_survives_the_same_autocorrelation():
    """The power arm: gamma = 9 along u at IACT 10 is still found, in the right direction."""
    xs, u = _ar1(100, 9.0 / 11.0, 2000, seed=3, spike=9.0)
    tr = _feed(_HeldBasisTracker(100, 2, DEFAULT_N0, DEFAULT_KAPPA, 0.1), xs)
    j = int(np.argmax(tr.gamma()))
    assert abs(tr.W[:, j] @ u) > 0.9
    assert tr.gamma()[j] > 5.0


def test_tracker_keys_are_validated():
    assert tracker_kwargs({}, "lowrank") == {"block": 50, "oversample": None, "gap": 5}
    with pytest.raises(ValueError, match="lowrank_tracker"):
        tracker_kwargs({"lowrank_tracker": "oja"}, "lowrank")
    with pytest.raises(ValueError, match="must exceed its gap"):
        make_tracker("held_basis", 10, 2, DEFAULT_N0, DEFAULT_KAPPA, 0.1, block=5, gap=5)


def _lowrank_sampler(tracker, seed=0):
    rng = np.random.default_rng(0)
    B = rng.standard_normal((6, 6))
    problem = correlated_gaussian(mean=np.zeros(6), cov=B @ B.T + np.eye(6))
    Cls = make_sampler_class(RobbinsMonroStepSize, LowRankAdaptation, NUTS)
    return Cls(problem.model, init_position=np.zeros(6), seed=seed,
               kinetics=[LowRankQuadraticKinetic(id="T", rank=2)], max_tree_depth=8,
               step_size=0.3, target_accept=0.8, lowrank_min_samples=40,
               lowrank_tracker=tracker)


def test_the_tracker_draws_no_randomness():
    """Through the burn-in both trackers are inert (gamma = 0), so the chains must agree
    bit-for-bit: the tracker choice touches no RNG. After it, they differ (the control)."""
    a, b = _lowrank_sampler("sanger"), _lowrank_sampler("held_basis")
    a.warmup(35); b.warmup(35)
    assert np.array_equal(np.asarray(a.state.coordinate), np.asarray(b.state.coordinate))
    a.warmup(400); b.warmup(400)
    assert not np.array_equal(np.asarray(a.state.coordinate), np.asarray(b.state.coordinate))
    assert isinstance(b._lr_blocks["T"]._tracker, _HeldBasisTracker)
