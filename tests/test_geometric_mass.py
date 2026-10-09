"""The geometric-mean mass target (``mass_target="geometric"``, ``spec.mass_adapt="geometric"``).

The score-mass SGD with its log-det barrier replaced by a spread term fits ``M = S^{-1} # F``: the
geometric mean of the score covariance ``F`` and the inverse spread ``S = E[u u^T] / c`` (see
``mimcs/adaptation/score_mass.py``). ``u`` is the kernel's move (``"difference"``, ``c = 2``) or
the centred position (``"marginal"``, ``c = 1``, Seyboldt's ``nutpie`` target).

The fixed-point tests feed the SGD blocks synthetic streams whose ``F`` and ``S`` are known and
do not commute, so the geometric mean differs from both of its ends; each has a control that the
score target misses. Seeds are fixed, so pass/fail is deterministic.
"""

import numpy as np
import pytest
from scipy.linalg import sqrtm

from mimcs.adaptation._stochastic import rm_gain
from mimcs.adaptation._subspace import _HeldBasisTracker, aux_gamma
from mimcs.adaptation.lowrank_mass import _LowRankBlock, _Sanger
from mimcs.adaptation.score_mass import _ScoreBlock
from mimcs.testing import correlated_gaussian, evaluate, nuts

D = 4


def _geometric_mean(A, B):
    """``A # B = A^{1/2} (A^{-1/2} B A^{-1/2})^{1/2} A^{1/2}``."""
    Ah = np.real(sqrtm(A))
    Aih = np.linalg.inv(Ah)
    return Ah @ np.real(sqrtm(Aih @ B @ Aih)) @ Ah


def _non_commuting_pair(seed=1):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((D, D))
    B = rng.standard_normal((D, D))
    return A @ A.T + 0.5 * np.eye(D), B @ B.T + 0.3 * np.eye(D)       # F, C


def _run_block(block, F, C, n, *, rho=0.0, spread="difference", offset=3.0, seed=2):
    """Feed ``block`` iid scores ``N(0, F)`` and AR(1) positions ``N(offset, C)`` with lag-1
    correlation ``rho``; return the mean of the fitted mass ``M`` over the second half."""
    rng = np.random.default_rng(seed)
    LF, LC = np.linalg.cholesky(F), np.linalg.cholesky(C)
    x_prev = LC @ rng.standard_normal(D)
    masses = []
    for t in range(1, n + 1):
        x = rho * x_prev + np.sqrt(1.0 - rho ** 2) * (LC @ rng.standard_normal(D))
        u = x - x_prev if spread == "difference" else x + offset
        x_prev = x
        out = np.asarray(block.update(LF @ rng.standard_normal(D), t, u))
        if t > n // 2:
            masses.append(np.linalg.inv(out @ out.T) if block.mode == "dense" else 1.0 / out)
    return np.mean(masses, axis=0)


def _block(mode, target="geometric", spread="difference"):
    return _ScoreBlock(mode, D, 5.0, 0.75, 0.1, True, False, target=target, spread=spread)


@pytest.mark.parametrize("spread", ["difference", "marginal"])
def test_diagonal_geometric_target_is_the_geometric_mean(spread):
    """Diagonal: ``M_d = sqrt(F_dd / C_dd)`` for either spread (the marginal one centred by its
    running mean, the offset of 3 notwithstanding). Control: the score target lands on ``F_dd``,
    which is 1.4--3.6x away here.

    The geometric fit is unbiased to 3% *with* the 10% clip on: its gradient is a difference of
    two heavy-tailed terms, so the clip bites on both sides. The score gradient
    ``1/2 (1 - g^2 / M)`` is bounded above, so its clip bites only on large ``g^2`` and leaves
    the mass ~18% low (the Kailas--Vihola--Wallin regularization, intended there)."""
    F, C = _non_commuting_pair()
    expected = np.sqrt(np.diag(F) / np.diag(C))
    M = _run_block(_block("diagonal", spread=spread), F, C, 40_000, spread=spread)
    np.testing.assert_allclose(M, expected, rtol=0.03)
    score = _run_block(_block("diagonal", target="score"), F, C, 40_000)
    np.testing.assert_allclose(score, 0.82 * np.diag(F), rtol=0.06)
    assert np.min(np.abs(score / expected - 1.0)) > 0.3


@pytest.mark.parametrize("spread", ["difference", "marginal"])
def test_dense_geometric_target_is_the_matrix_geometric_mean(spread):
    """Dense: ``M = C^{-1} # F`` (``M C M = F``), checked against ``scipy.linalg.sqrtm``.
    Control: the score target lands on ``F``, 41% away (>10x the tolerance)."""
    F, C = _non_commuting_pair()
    expected = _geometric_mean(np.linalg.inv(C), F)
    np.testing.assert_allclose(expected @ C @ expected, F, atol=1e-10)
    M = _run_block(_block("dense", spread=spread), F, C, 40_000, spread=spread)
    assert np.abs(M - expected).max() / np.abs(expected).max() < 0.03
    score = _run_block(_block("dense", target="score"), F, C, 40_000)
    assert np.abs(score - F).max() / np.abs(F).max() < 0.06
    assert np.abs(expected - F).max() / np.abs(F).max() > 0.3


def test_the_multiplicative_dense_step_fits_the_score_target_too():
    """``score_mass_dense_step="multiplicative"`` takes the score target's step in the whitened
    frame (``X = I - w w^T``): same fixed point as the default Cholesky SGD, within the same ~6%
    clip bias. An unknown step raises."""
    F, C = _non_commuting_pair()
    for step in ("cholesky", "multiplicative"):
        b = _ScoreBlock("dense", D, 5.0, 0.75, 0.1, True, False, dense_step=step)
        M = _run_block(b, F, C, 40_000)
        assert np.abs(M - F).max() / np.abs(F).max() < 0.08, step
    with pytest.raises(ValueError, match="dense_step"):
        _ScoreBlock("dense", D, 5.0, 0.75, 0.1, True, False, dense_step="newton")


def test_lag_one_correlation_shrinks_the_difference_spread():
    """``E[(x_t - x_{t-1})^2] / 2 = Var (1 - rho)``: draws correlated at lag 1 by ``rho = 0.6``
    leave the mass larger by ``1 / sqrt(0.4) = 1.58``. The marginal spread does not see it."""
    F, C = _non_commuting_pair()
    iid = _run_block(_block("diagonal"), F, C, 40_000)
    ar = _run_block(_block("diagonal"), F, C, 40_000, rho=0.6)
    np.testing.assert_allclose(ar / iid, 1.0 / np.sqrt(0.4), rtol=0.05)
    m_iid = _run_block(_block("diagonal", spread="marginal"), F, C, 40_000, spread="marginal")
    m_ar = _run_block(_block("diagonal", spread="marginal"), F, C, 40_000, rho=0.6,
                      spread="marginal")
    np.testing.assert_allclose(m_ar / m_iid, 1.0, rtol=0.06)


def test_rare_mode_switches_reach_the_marginal_spread_but_not_the_difference():
    """Two modes at +-3 on axis 0 (unit width), switching with probability 0.01 per step, and unit
    scores. The difference spread reads the within-mode width, ``E[u^2] / 2 = 1 + 0.01 * 36 / 2``,
    so the inverse mass is ~1.09 on every axis. The marginal one reads the whole variance
    ``1 + 9 = 10`` on axis 0, inverse mass ~sqrt(10) = 3.2: the distance between the modes."""
    rng = np.random.default_rng(3)
    n = 40_000
    mode = 1.0
    x_prev = np.zeros(D)
    blocks = {s: _block("diagonal", spread=s) for s in ("difference", "marginal")}
    inv = {s: [] for s in blocks}
    for t in range(1, n + 1):
        if rng.uniform() < 0.01:
            mode = -mode
        x = rng.standard_normal(D)
        x[0] += 3.0 * mode
        g = rng.standard_normal(D)
        for s, b in blocks.items():
            out = np.asarray(b.update(g, t, x - x_prev if s == "difference" else x))
            if t > n // 2:
                inv[s].append(out)
        x_prev = x
    diff, marg = np.mean(inv["difference"], 0), np.mean(inv["marginal"], 0)
    assert diff[0] < 1.3, diff
    np.testing.assert_allclose(diff[1:], 1.0, rtol=0.08)
    assert marg[0] > 2.6, marg
    np.testing.assert_allclose(marg[1:], 1.0, rtol=0.08)


def test_the_geometric_target_needs_a_spread():
    with pytest.raises(ValueError, match="spread"):
        _block("diagonal").update(np.ones(D), 1)


# --- low-rank -----------------------------------------------------------------------------------

def test_aux_gamma_is_the_geometric_mean_along_each_direction():
    lam = np.array([9.0, 4.0, 0.5])
    np.testing.assert_allclose(aux_gamma(lam, None), [8.0, 3.0, 0.0])
    np.testing.assert_allclose(aux_gamma(lam, np.array([1.0, 4.0, 0.5])), [2.0, 0.0, 0.0])


@pytest.mark.parametrize("tracker", ["held_basis", "sanger"])
def test_an_isotropic_but_inflated_block_gets_no_rank_stiffening(tracker):
    """The funnel's ``x`` block in miniature: scores isotropic at variance 90, positions at unit
    variance. The geometric ``D`` is ~sqrt(90) per coordinate, and the ``D``-whitened score is
    then isotropic at ~sqrt(90) = 9.5: the score's ``gamma = lambda - 1`` would stiffen ``J``
    arbitrary directions by ~8.5, straight back towards the score's scale (control). The
    geometric mean along each direction is ~1, so ``gamma ~ 0``."""
    n, J = 12, 3
    rng = np.random.default_rng(4)
    b = _LowRankBlock(n, J, 5.0, 0.75, 0.1, True, 1.0, 1.0, 50, tracker=tracker,
                      target="geometric", spread="difference")
    x_prev = rng.standard_normal(n)
    for _ in range(6000):
        x = rng.standard_normal(n)
        D_, V = b.update(np.sqrt(90.0) * rng.standard_normal(n), x - x_prev)
        x_prev = x
    # D is the last raw iterate (+-20% per coordinate); its mean is the fixed point.
    np.testing.assert_allclose(np.mean(np.asarray(D_)), np.sqrt(90.0), rtol=0.1)
    assert np.max(b._tracker.gamma()) < 0.5, b._tracker.gamma()
    assert np.min(np.maximum(0.0, b._tracker.lam - 1.0)) > 5.0          # the score's reading


def test_a_correlated_gaussian_low_rank_mass_finds_the_stiff_direction():
    """A Gaussian with one stiff direction: there the geometric mean is the precision, so the
    rank-1 mass recovers ``v^T M v ~ v^T P v`` (the stiff direction) to within the SGD's noise."""
    n = 10
    rng = np.random.default_rng(5)
    v = rng.standard_normal(n)
    v /= np.linalg.norm(v)
    C = np.eye(n) - 0.9 * np.outer(v, v)             # variance 0.1 along v: precision 10
    P = np.linalg.inv(C)
    LC = np.linalg.cholesky(C)
    b = _LowRankBlock(n, 1, 5.0, 0.75, 0.1, True, 1.0, 1.0, 50, target="geometric",
                      spread="difference")
    x_prev = LC @ rng.standard_normal(n)
    for _ in range(8000):
        x = LC @ rng.standard_normal(n)
        D_, V = b.update(P @ x_prev, x - x_prev)      # score of an independent draw: N(0, P)
        x_prev = x
    D_, V = np.asarray(D_), np.asarray(V)
    M = np.diag(D_) + V.T @ V
    assert 6.0 < v @ M @ v < 15.0, v @ M @ v
    w = np.linalg.qr(np.column_stack([v, rng.standard_normal(n)]))[0][:, 1]   # an easy direction
    assert 0.5 < w @ M @ w < 2.0, w @ M @ w


def test_trackers_without_an_aux_stream_are_unchanged():
    """The shaped metric shares the trackers and never passes ``aux``: its ``gamma`` must stay
    ``max(0, lambda - 1)`` bit for bit."""
    rng = np.random.default_rng(6)
    for t in (_HeldBasisTracker(8, 2, 5.0, 0.75, 0.1), _Sanger(8, 2, 5.0, 0.75, 0.1, 1.0)):
        for k in range(1, 300):
            t.step(rng.standard_normal(8) * np.arange(1, 9), rm_gain(k, 5.0, 0.75), k)
        assert t.lam_aux is None
        np.testing.assert_array_equal(t.gamma(), np.maximum(0.0, t.lam - 1.0))


# --- the move the sampler hands the block --------------------------------------------------------

def test_the_move_is_the_kernels_own_displacement():
    """The spread is the coordinate the kernel was handed against the one it returned: after one
    warmup step the block's running spread mean is exactly ``gain_1 * (x_1 - x_0)``."""
    prob = correlated_gaussian(mean=[1.0, -2.0], cov=[[2.0, 1.4], [1.4, 1.5]])
    s = nuts(mass_adapt="geometric")(prob.model, seed=0)
    x0 = np.asarray(s.state.coordinate)
    s.warmup(1)
    x1 = np.asarray(s.state.coordinate)
    assert not np.allclose(x0, x1)
    np.testing.assert_array_equal(np.asarray(s._sm_track.pre), x0)
    (block,) = s._sm_blocks.values()
    np.testing.assert_allclose(block.mean_spread, rm_gain(1, 5.0, 0.75) * (x1 - x0), rtol=1e-6)


def test_unknown_target_or_spread_raises():
    prob = correlated_gaussian()
    with pytest.raises(ValueError, match="mass_target"):
        nuts(mass_adapt="score", mass_target="fisher")(prob.model, seed=0)
    with pytest.raises(ValueError, match="geometric_spread"):
        nuts(mass_adapt="geometric", geometric_spread="lagged")(prob.model, seed=0)


@pytest.mark.parametrize("metric", ["diagonal", "dense"])
def test_geometric_mass_samples_a_correlated_gaussian(metric):
    """End to end: NUTS with the geometric mass samples a correlated Gaussian correctly, and on a
    Gaussian the geometric mean is the covariance itself, so ``M^{-1}`` lands near it."""
    cov = np.array([[2.0, 1.4], [1.4, 1.5]])
    prob = correlated_gaussian(mean=[1.0, -2.0], cov=cov)
    report = evaluate(prob, {"geometric": nuts(mass_adapt="geometric", metric=metric)},
                      n_warmup=1500, n_samples=4000, seed=0)
    print("\n" + report.summary())
    report.assert_correct()


# --- factory and parallel tempering --------------------------------------------------------------

def _spec(model, mass_adapt="geometric", **kw):
    from mimcs.factory import default_spec
    spec = default_spec(model)
    spec.mass_adapt = mass_adapt
    for k, v in kw.items():
        setattr(spec, k, v)
    return spec


def test_factory_geometric_reaches_the_quadratic_and_the_lowrank_blocks():
    """``mass_adapt="geometric"`` is ``ScoreMassAdaptation`` under ``mass_target="geometric"``,
    and unlike ``"score"``/``"covariance"`` it reaches ``lowrank`` blocks too. ``"score"`` leaves
    the low-rank target alone."""
    from mimcs.factory import BlockSpec
    from mimcs.model import EuclideanParameter, Model
    m = Model([EuclideanParameter("a", (3,)), EuclideanParameter("b", (6,))],
              {"lp": lambda p: -0.5 * (p["a"] @ p["a"] + p["b"] @ p["b"])})
    blocks = [BlockSpec(["a"], [m.coord_block("a")], "diagonal"),
              BlockSpec(["b"], [m.coord_block("b")], "lowrank", {"rank": 2})]
    s = _spec(m, blocks=blocks).build(seed=0)
    assert (s._sm_target, s._lr_target) == ("geometric", "geometric")
    s.warmup(60)
    assert s._lr_blocks and all(b.target == "geometric" for b in s._lr_blocks.values())
    s = _spec(m, mass_adapt="score", blocks=blocks).build(seed=0)
    assert (s._sm_target, s._lr_target) == ("score", "score")
    with pytest.raises(ValueError, match="mass_adapt"):
        _spec(m, mass_adapt="geometrical").build(seed=0)


def _pt(model, K=3):
    return _spec(model, base="pt_nuts", tempering_params={"n_temperatures": K}).build(seed=0)


def test_a_rung_move_is_its_own_trajectory_with_the_swap_undone():
    """Rung k's move is where *its* trajectory ended (before the swap) minus where it started.
    With pair (0, 1) swapped, rung 0 now holds rung 1's endpoint, so rung 0's move is
    ``post[1] - pre[0]``. Control: the post-swap difference ``post[0] - pre[0]`` is a replica jump
    between rungs, not a move of either."""
    s = _pt(correlated_gaussian().model)
    K, n = 3, 2
    pre = np.arange(K * n, dtype=float)
    post = pre + 10.0 * np.repeat(np.arange(1, K + 1), n)          # rung k moved by 10 (k + 1)
    swapped = np.asarray(post).reshape(K, n)[[1, 0, 2]].reshape(-1)  # pair (0, 1) exchanged
    s._pt_pre = pre
    state = s.state._replace(coordinate=swapped, diagnostics={
        **s.state.diagnostics, "swap_accepted": np.array([True, False, False])})
    moves = np.asarray(s._rung_moves(state, n))
    np.testing.assert_allclose(moves, np.repeat([[10.0], [20.0], [30.0]], n, axis=1))
    assert not np.allclose(swapped.reshape(K, n)[0] - pre.reshape(K, n)[0], moves[0])


def test_parallel_tempering_hosts_fit_their_own_rungs_moves():
    """End to end under ``pt_nuts``: every host receives its rung's move (a host has no
    ``_preprocess_hooks`` of its own and would raise without it), and after one warmup step its
    running spread mean is exactly ``gain_1`` times that move."""
    s = _pt(correlated_gaussian().model)
    s.warmup(1)
    moves = np.asarray(s._rung_moves(s.state, 2))
    for k, host in enumerate(s._adapt_hosts):
        assert host._sm_target == "geometric" and host._sm_track.pre is None
        (block,) = host._sm_blocks.values()
        np.testing.assert_allclose(block.mean_spread, rm_gain(1, 5.0, 0.75) * moves[k],
                                   rtol=1e-5, atol=1e-7)
    s.warmup(100)
    s.sample(100)


@pytest.mark.parametrize("mode", ["diagonal", "dense"])
def test_a_huge_transient_is_held_to_the_trust_region_and_forgotten(mode):
    """A far-from-mode start: 300 steps of scores ~1e8 (kilpisjarvi's are ~1e9) drive the clip
    threshold up, which alone does not bound the step --- without the trust region the dense factor
    oscillated by +-30 nats a step into singularity. With it no step moves the mass by more than
    ``MAX_LOG_STEP`` nats, it binds through the transient only, and the fixed point is reached."""
    from mimcs.adaptation._geometric import MAX_LOG_STEP
    F, C = _non_commuting_pair()
    rng = np.random.default_rng(7)
    b = _block(mode)
    prev = np.zeros(D) if mode == "diagonal" else np.eye(D)
    for t in range(1, 301):
        b.update(1e8 * rng.standard_normal(D), t, 1e-3 * rng.standard_normal(D))
        # the block's own float64 state (the kinetic's parameter is float32 here, too coarse at
        # mass scales ~1e16): log M, or the factor S of M = S S^T
        cur = b.log_mass.copy() if mode == "diagonal" else b.S.copy()
        if mode == "diagonal":
            assert np.max(np.abs(cur - prev)) <= MAX_LOG_STEP + 1e-9
        else:   # S_prev^{-1} S_new = expm(-t X) is symmetric: its eigenvalues are the mass's moves
            ev = np.linalg.eigvals(np.linalg.solve(prev, cur)).real
            assert np.all(np.abs(2.0 * np.log(ev)) <= MAX_LOG_STEP + 1e-6)
        prev = cur
    # it binds once the clip threshold has climbed (from step ~19), for most of the transient
    assert len(b.capped) > 200 and max(b.capped) <= 300
    n_transient = len(b.capped)
    target = (np.sqrt(np.diag(F) / np.diag(C)) if mode == "diagonal"
              else _geometric_mean(np.linalg.inv(C), F))
    LF, LC = np.linalg.cholesky(F), np.linalg.cholesky(C)
    x_prev = LC @ rng.standard_normal(D)
    masses = []
    for t in range(301, 60_301):
        x = LC @ rng.standard_normal(D)
        out = np.asarray(b.update(LF @ rng.standard_normal(D), t, x - x_prev))
        x_prev = x
        if t > 30_300:
            masses.append(1.0 / out if mode == "diagonal" else np.linalg.inv(out @ out.T))
    assert np.abs(np.mean(masses, 0) - target).max() / np.abs(target).max() < 0.05
    assert max(b.capped) < 30_300, "the trust region still binds at stationarity"
    print(f"{mode}: capped {n_transient} transient steps, {len(b.capped) - n_transient} later")


# --- lag-k spread ---------------------------------------------------------------------------------

def _tracker(k, spread="difference", point="split"):
    from mimcs.adaptation._geometric import SpreadTracker
    return SpreadTracker({"mass_target": "geometric", "geometric_spread": spread,
                          "geometric_lag": k, "geometric_metric_point": point})


class _Lag:
    """Feed a position stream through a :class:`SpreadTracker` window: returns ``x_t - x_{t-k}``
    (fewer steps until ``k`` have arrived), as a plain sampler's mixin would see it."""

    def __init__(self, k):
        self.track, self.prev = _tracker(k), None

    def __call__(self, x):
        from types import SimpleNamespace
        prev = x if self.prev is None else self.prev
        self.prev = x
        return self.track.observe(SimpleNamespace(coordinate=x, start=prev, end=x)).u


def test_the_spread_window_keeps_start_coordinates_and_both_endpoints():
    """The window keeps the last k *start* coordinates: u = end - (start k steps back), and the
    learned metric's two evaluation points are that old start and the new end. k = 1 is exactly
    ``end - start``. A replacement (PT) empties the window after the step."""
    from types import SimpleNamespace as NS
    t = _tracker(3)
    xs = [np.full(2, float(i)) for i in range(7)]          # a chain at 0, 1, 2, ... 6
    outs = [t.observe(NS(coordinate=xs[i + 1], start=xs[i], end=xs[i + 1])) for i in range(5)]
    np.testing.assert_array_equal(outs[0].u, [1, 1])        # one step so far
    np.testing.assert_array_equal(outs[4].u, [3, 3])        # 5 - 2: the window is three steps
    np.testing.assert_array_equal(outs[4].q_old, xs[2])
    np.testing.assert_array_equal(outs[4].q_new, xs[5])
    t.observe(NS(coordinate=xs[6], start=xs[5], end=xs[6], replaced=True))
    assert t._starts == []                                  # a new replica: a new window
    one = _tracker(1).observe(NS(coordinate=xs[3], start=xs[1], end=xs[3]))
    np.testing.assert_array_equal(one.u, xs[3] - xs[1])
    m = _tracker(4, "marginal").observe(NS(coordinate=xs[2]))
    np.testing.assert_array_equal(m.u, xs[2])
    np.testing.assert_array_equal(m.q_old, m.q_new)


@pytest.mark.parametrize("k", [1, 4])
def test_a_lag_k_spread_stiffens_by_sqrt_one_minus_rho_to_the_k(k):
    """AR(1) draws with lag-1 correlation 0.6: the sum of the last k moves is ``x_t - x_{t-k}``,
    whose half second moment is ``Sigma (1 - rho^k)``. So the mass is larger than the iid one by
    ``1 / sqrt(1 - 0.6^k)``: 1.58 at k = 1 and only 1.07 at k = 4. The lag-1 spread's
    slow-coordinate stiffening is what the lag removes."""
    F, C = _non_commuting_pair()

    def run(rho, seed):
        rng = np.random.default_rng(seed)
        LF, LC = np.linalg.cholesky(F), np.linalg.cholesky(C)
        lag, b = _Lag(k), _block("diagonal")
        x_prev = LC @ rng.standard_normal(D)
        out = []
        for t in range(1, 40_001):
            x = rho * x_prev + np.sqrt(1 - rho ** 2) * (LC @ rng.standard_normal(D))
            m = 1.0 / np.asarray(b.update(LF @ rng.standard_normal(D), t, lag(x)))
            x_prev = x
            if t > 20_000:
                out.append(m)
        return np.mean(out, 0)

    ratio = run(0.6, 8) / run(0.0, 9)
    np.testing.assert_allclose(ratio, 1.0 / np.sqrt(1.0 - 0.6 ** k), rtol=0.05)


def test_a_lag_four_spread_still_keeps_rare_mode_switches_out():
    """The two-mode stream of the lag-1 test (modes +-3, switching w.p. 0.01). Over 4 steps a switch
    enters the sum with probability ~0.04, so ``E[u^2] / 2 ~ 1 + 0.04 * 36 / 2``: inverse mass
    ~1.3, against the marginal spread's ~3.2."""
    rng = np.random.default_rng(3)
    lag, b = _Lag(4), _block("diagonal")
    mode, x_prev, inv = 1.0, np.zeros(D), []
    for t in range(1, 40_001):
        if rng.uniform() < 0.01:
            mode = -mode
        x = rng.standard_normal(D)
        x[0] += 3.0 * mode
        out = np.asarray(b.update(rng.standard_normal(D), t, lag(x)))
        x_prev = x
        if t > 20_000:
            inv.append(out)
    inv = np.mean(inv, 0)
    assert 1.1 < inv[0] < 1.6, inv
    np.testing.assert_allclose(inv[1:], 1.0, rtol=0.08)


def test_bad_lags_raise():
    prob = correlated_gaussian()
    for k in (0, 1.5, True):
        with pytest.raises(ValueError, match="geometric_lag"):
            nuts(mass_adapt="geometric", geometric_lag=k)(prob.model, seed=0)


def test_a_rung_lag_sums_its_own_replicas_pre_swap_moves():
    """Under PT each host sums its rung's pre-swap moves *since the rung last received a different
    replica*: the sum is then the current replica's own displacement, a genuine single-chain lag
    difference. Across a replacement the moves are independent rather than anticorrelated, so a sum
    spanning replicas gains a full ``2 Sigma`` per break (measured: lag 16 widened non-switching
    coordinates ~1.6x past the marginal covariance). Over 40 warmup steps with lag 16 and a 3-rung
    ladder, swaps replace replicas many times; each host's buffer must equal the moves it saw after
    its own last replacement (control: it is not the plain last-16 sum)."""
    from mimcs.factory import default_spec
    spec = default_spec(correlated_gaussian().model)
    spec.mass_adapt = "geometric"
    spec.base = "pt_nuts"
    spec.tempering_params = {"n_temperatures": 3}
    spec.algo_kwargs = {**spec.algo_kwargs, "geometric_lag": 16}
    s = spec.build(seed=0)
    moves, replaced = [], []
    for _ in range(40):
        s.warmup(1)
        moves.append(np.asarray(s._rung_moves(s.state, 2)))
        replaced.append(np.asarray(s._rung_replaced(s.state)))
    replaced = np.array(replaced)
    assert replaced.sum() >= 5, "the ladder never swapped: the test would be vacuous"
    differs = 0
    for k, host in enumerate(s._adapt_hosts):
        last = max([t for t in range(40) if replaced[t, k]], default=-1)
        expected = moves[last + 1:][-16:]
        window = host._sm_track._starts
        assert len(window) == len(expected)
        if expected:     # the window's displacement = the sum of this replica's own moves
            end = np.asarray(s._rung_endpoints(s.state, 2)[1][k], float)
            np.testing.assert_allclose(end - window[0], sum(m[k] for m in expected),
                                       rtol=1e-5, atol=1e-6)
        differs += last >= 24                      # a replacement inside the last-16 window
    assert differs >= 1, "no rung was replaced inside the window: the control is vacuous"
