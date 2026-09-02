"""Tests for evidence-informed mass-matrix mode selection (``mimcs.factory.mode_select``).

The selector reads a block's mass structure off the spectrum of the whitened score covariance
``R = D^{-1/2} S D^{-1/2}``. What must hold: the whitened-spectrum math (the ``d>n`` Gram path
matches the full eigendecomposition; the KL-loss decomposition over eigenvalues is exact), the
boundary gates (small-d -> dense, nothing-above-the-bulk -> diagonal, the ``n >= 0.75 d`` rank
guard, ``d>n`` forbids dense, ``J <= d/5``), and that the recovered mode matches ground truth on
clean spiked / isotropic / dense targets.

Every threshold is relative to the **bulk**, never to an absolute number: the statistic is
``max(eig) / bulk_scale`` with ``bulk_scale = median(eig) / mp_median(gamma)``, compared against
the Marchenko-Pastur edge. So the tests that matter are calibration tests --- does pure noise read
as diagonal at every ``(n, d)``, and does real structure still read as structure --- rather than
assertions about any one constant.

Seeds are fixed, so pass/fail is deterministic.
"""

import numpy as np
import pytest

from mimcs.factory.mode_select import (
    select_mass_mode, whitened_spectrum, _kl_benefit, J_MAX_FRAC)


def _draw(C, n, seed=0):
    return np.random.default_rng(seed).standard_normal((n, C.shape[0])) @ np.linalg.cholesky(C).T


def _spiked(d, spikes, seed=1):
    Q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((d, d)))
    C = np.eye(d)
    for j, s in enumerate(spikes):
        C = C + s * np.outer(Q[:, j], Q[:, j])
    return C


# --- whitened-spectrum math -------------------------------------------------- #

def test_gram_path_matches_full_eigendecomposition():
    """For ``n < d`` the selector uses the ``n x n`` Gram; its nonzero eigenvalues must equal the
    full whitened covariance's."""
    X = _draw(_spiked(60, [10.0, 5.0]), 25, seed=0)         # n=25 < d=60
    _, eig_gram, n, d = whitened_spectrum(X)                # Gram path (n<d)
    g = X - X.mean(0)
    Dv = (g * g).mean(0)
    R = (g / np.sqrt(Dv)).T @ (g / np.sqrt(Dv)) / n         # full d x d whitened covariance
    eig_full = np.sort(np.linalg.eigvalsh(R))[::-1]
    assert np.allclose(np.sort(eig_gram)[::-1][:n], eig_full[:n], atol=1e-8)
    assert np.allclose(eig_gram[n:], 0.0)                   # d-n exact zeros


def test_kl_loss_decomposition_is_exact():
    """loss(dense) - loss(diag) == -sum of per-direction KL benefits (the identity the AIC rule
    scans over)."""
    X = _draw(_spiked(40, [8.0, 4.0]), 3000, seed=2)
    _, eig, n, d = whitened_spectrum(X)
    S = np.cov(X.T, bias=True)
    loss_diag = 0.5 * (np.sum(np.log(np.diag(S))) + d)
    loss_dense = 0.5 * (np.linalg.slogdet(S)[1] + d)
    assert np.isclose(loss_dense - loss_diag, -_kl_benefit(eig).sum(), atol=1e-6)


# --- boundary gates ---------------------------------------------------------- #

@pytest.mark.parametrize("rule", ["aic", "mp", "parallel"])
def test_isotropic_is_diagonal(rule):
    X = _draw(np.eye(50), 500, seed=0)
    assert select_mass_mode(X, rule=rule) == ("diagonal", 0)


@pytest.mark.parametrize("rule", ["aic", "mp", "parallel"])
def test_wide_diagonal_is_diagonal(rule):
    """A wide *diagonal* covariance (no correlation) must be diagonal, not dense: the diagonal D
    absorbs the scale and the whitened R is the identity."""
    C = np.diag(np.exp(np.random.default_rng(1).uniform(-3, 3, 50)))
    assert select_mass_mode(_draw(C, 1000, seed=0), rule=rule) == ("diagonal", 0)


@pytest.mark.parametrize("rule", ["aic", "mp", "parallel"])
def test_small_dimension_is_dense(rule):
    assert select_mass_mode(_draw(_spiked(5, [6.0]), 300), rule=rule) == ("dense", 0)


@pytest.mark.parametrize("rule", ["aic", "mp", "parallel"])
def test_high_dim_forbids_dense_and_caps_rank(rule):
    """``d > n`` -> never dense; and any low-rank J respects ``J <= floor(d/5)`` and ``J < n``.

    Kept at ``n = 0.8 d``, just *above* the rank guard, so it still tests what it was written to
    test. Below the guard it would pass vacuously on the diagonal branch.
    """
    d, n = 200, 160                                         # d > n, but above the rank guard
    X = _draw(_spiked(d, [30.0, 18.0, 9.0]), n, seed=0)
    kind, J = select_mass_mode(X, rule=rule)
    assert kind in ("diagonal", "lowrank")                 # not dense
    if kind == "lowrank":
        assert J <= int(J_MAX_FRAC * d) and J < n


@pytest.mark.parametrize("rule", ["aic", "mp", "parallel"])
def test_the_rank_guard_refuses_a_block_with_too_few_rows(rule):
    """The regression test for the production failure this module was reworked for.

    A 500-draw pilot on a 2000-coordinate block got a *correctly detected* ``lowrank(52)`` --- the
    top eigenvalues were 20-36 against a bulk edge of 9.45, with no outlying rows --- and building
    it drove NUTS's step size to 1e-24 at 100% divergences. Detection was never the problem:
    ``n`` rows can reveal far more directions than they can *aim*, and a mass built from misaimed
    directions is wrong exactly in the stiff directions it claims to fix.

    The control is the same law with enough rows: if that also came back diagonal, this test would
    be passing because the signal is absent rather than because the guard fired.
    """
    d = 200
    C = _spiked(d, [30.0, 18.0, 9.0])
    assert select_mass_mode(_draw(C, 50, seed=0), rule=rule) == ("diagonal", 0)
    assert select_mass_mode(_draw(C, int(0.7 * d), seed=0), rule=rule) == ("diagonal", 0)
    kind, _ = select_mass_mode(_draw(C, 5 * d, seed=0), rule=rule)      # control
    assert kind == "lowrank", kind


@pytest.mark.parametrize("rule", ["aic", "mp"])
def test_pure_noise_is_diagonal_at_every_shape(rule):
    """Calibration, the property a single hard-coded threshold cannot have.

    The pure-noise spread tracks the Marchenko-Pastur edge ``(1+sqrt(gamma))^2``, so any fixed
    number is simultaneously too tight at one ``gamma`` and too loose at another --- measured, the
    raw ratio runs 4.6 at ``n = 0.75 d``, 6.0 at ``n = d`` and 2.2 at ``n = 5 d``. A ``gamma``-aware
    edge in units of the estimated bulk scale is flat against it.

    This is also the guard on `_aic_mode`: measurement showed the AIC penalty, not the gate, is
    what suppresses the bulk today, so a future change there would otherwise reintroduce
    over-selection silently.
    """
    for d in (20, 50, 200):
        for frac in (0.75, 1.0, 1.5, 2.0, 5.0):
            n = int(frac * d)
            for seed in range(4):
                X = np.random.default_rng(1000 + seed).standard_normal((n, d))
                kind, J = select_mass_mode(X, rule=rule)
                assert kind == "diagonal", (d, n, seed, kind, J)


def test_the_bulk_scale_correction_is_what_makes_the_gate_fire():
    """Asserted at the **gate**, because the verdict cannot see it.

    Dropping the ``mp_median`` correction changes no final verdict --- ``_aic_mode``'s penalty
    absorbs the bulk downstream either way --- so a test on ``select_mass_mode``'s output passes
    with the correction removed. What the correction buys is that the cheap early exit actually
    fires: the MP median is 0.65 at ``gamma = 1``, so an uncorrected ratio inflates by ~1.5x
    exactly where the bulk is widest, and the gate stops firing on pure noise in the band
    ``0.9 d <= n <= 1.5 d`` --- the band pilots live in. Without this test the module would keep a
    documented gate that never fires, with all the safety resting on AIC by accident.
    """
    from mimcs.factory.mode_select import _bulk_scale, _bulk_edge, _n_computable, DIAGONAL_MARGIN
    corrected_fires = uncorrected_fires = trials = 0
    for d in (50, 200):
        for frac in (1.0, 1.5):
            n = int(frac * d)
            for seed in range(6):
                X = np.random.default_rng(1000 + seed).standard_normal((n, d))
                _, eig, nn, dd = whitened_spectrum(X)
                edge = _bulk_edge(nn, dd, DIAGONAL_MARGIN)
                corrected_fires += eig[0] / _bulk_scale(eig, nn, dd) < edge
                raw = float(np.median(eig[: _n_computable(nn, dd)]))     # the control
                uncorrected_fires += eig[0] / raw < edge
                trials += 1
    assert corrected_fires == trials, (corrected_fires, trials)
    assert uncorrected_fires < trials // 2, (uncorrected_fires, trials)


def test_a_real_spike_still_reads_as_structure():
    """The control for the calibration test above: a gate that says diagonal on everything would
    pass it. One genuine spike, at the same shapes, must not be called diagonal."""
    for d in (20, 50, 200):
        for frac in (1.0, 2.0, 5.0):
            n = int(frac * d)
            X = _draw(_spiked(d, [d / 2.0]), n, seed=0)
            assert select_mass_mode(X)[0] != "diagonal", (d, n)


def test_a_degenerate_pilot_is_refused_not_ranked():
    """A chain that rejects heavily repeats rows, so its scores are rank-deficient in a way the
    *row count* cannot see: 500 draws from 120 distinct states at ``d = 200`` look like
    ``n/d = 2.5`` and are really ``n/d = 0.6``. Left alone that re-admits the over-selection the
    rank guard exists to prevent, through a door the guard does not watch --- and at the extreme
    the bulk scale underflows, every threshold built on it is passed, and the selector would return
    a rank-40 mass fitted from 79 independent rows.

    The control is the same construction with enough distinct states.
    """
    d, n = 200, 500
    rng = np.random.default_rng(5)
    base = rng.normal(size=(500, d))

    def resampled(k):
        return base[:k][np.random.default_rng(5).integers(0, k, size=n)]

    assert select_mass_mode(resampled(80)) == ("diagonal", 0)      # bulk scale underflows
    assert select_mass_mode(resampled(120)) == ("diagonal", 0)     # n_eff/d = 0.60
    assert select_mass_mode(resampled(500))[0] == "lowrank"        # control: full rank, structured


def test_the_bulk_scale_ignores_the_structural_padding():
    """``_eigs_desc`` zero-pads to ``d``, and centring costs one more rank, so a median over the
    raw array is 0 for ``n <= d/2`` and the spread is infinite. The scale is taken over the
    computable eigenvalues instead. The control is the padded median, which must be 0."""
    from mimcs.factory.mode_select import _bulk_scale, _n_computable
    d, n = 200, 60
    X = np.random.default_rng(0).standard_normal((n, d))
    _, eig, nn, dd = whitened_spectrum(X)
    assert _n_computable(nn, dd) == nn - 1                        # centring costs one rank
    assert np.median(eig) == 0.0                                  # the control
    assert np.isfinite(_bulk_scale(eig, nn, dd)) and _bulk_scale(eig, nn, dd) > 0


def test_jmax_counts_spikes_not_the_bulks_upper_half():
    """`jmax` shrinks to what is actually above the bulk, not to a fixed multiple of the median.

    Honest scope: this pins the *counting rule*, not a verdict. The cap is **inert** in every case
    swept --- AR(1) at rho in {0.7, 0.9, 0.95} and isolated spikes at k in {3, 10, 25}, d in
    {60..200}, n/d in {1..20}, on both the AIC and the spike paths --- because AIC's per-direction
    penalty (~2d) is far stricter than the MP edge, so it never asks for more directions than are
    above the bulk. It is kept because it is the right definition of the ceiling and it guards the
    spike rules, not because it was measured to change an answer.

    What the assertion does encode is the measurement that rejected the ``2 * median`` multiplier:
    that counts ~40 of 200 eigenvalues on pure noise at ``n = 0.75 d`` --- the same count it gives
    with 8 real spikes --- because at that ``gamma`` the bulk spans [0.02, 4.64] and ``2 * median``
    sits inside it."""
    from mimcs.factory.mode_select import _bulk_scale, _mp_spikes
    d, n = 200, 150
    X = np.random.default_rng(0).standard_normal((n, d))
    _, eig, nn, dd = whitened_spectrum(X)
    scale = _bulk_scale(eig, nn, dd)
    assert _mp_spikes(eig, nn, dd, scale) <= 2                    # ~0 above the bulk on noise
    naive = int(np.sum(eig > 2.0 * np.median(eig[: nn - 1])))     # the rejected rule
    assert naive > 20, naive


def test_the_edge_cases_do_not_raise():
    """`select_mass_mode` is public and its docstring promises nothing about ``n``; both callers
    gate on row counts but the function must not return nan or divide by zero on its own."""
    assert select_mass_mode(np.random.default_rng(0).standard_normal((1, 5))) == ("diagonal", 0)
    assert select_mass_mode(np.random.default_rng(0).standard_normal((2, 5))) == ("diagonal", 0)
    assert select_mass_mode(np.random.default_rng(0).standard_normal((100, 1))) == ("diagonal", 0)


@pytest.mark.parametrize("rule", ["aic", "mp", "parallel"])
def test_spikes_recovered_as_lowrank(rule):
    """Three isolated spikes in a mid-size block -> low-rank of rank 3."""
    X = _draw(_spiked(80, [20.0, 12.0, 7.0]), 2000, seed=0)
    kind, J = select_mass_mode(X, rule=rule)
    assert kind == "lowrank" and J == 3


# --- the factory refinement rule --------------------------------------------- #

def _one_block_model(d):
    from mimcs import Model
    from mimcs.model import EuclideanParameter
    return Model([EuclideanParameter("x", (d,))], {"lp": lambda p: -0.5 * (p["x"] ** 2).sum()})


def test_mass_mode_rule_selects_from_evidence():
    """`analyze(model, evidence)` reads the block's mass mode off the evidence spectrum: a spiked
    score covariance -> low-rank(3), overriding the dimension-count placeholder."""
    from mimcs.factory import analyze
    model = _one_block_model(30)
    G = _draw(_spiked(30, [20.0, 12.0, 6.0]), 2000, seed=0)
    spec = analyze(model, {"gradients": G, "samples": G[:200]})
    assert spec.blocks[0].kind == "lowrank" and spec.blocks[0].params.get("rank") == 3


def test_mass_mode_rule_is_noop_without_evidence():
    """No gradient evidence -> the rule is inert and the dimension-count default governs (d=30 in
    (20, 50] -> its own dense block)."""
    from mimcs.factory import analyze
    spec = analyze(_one_block_model(30))
    assert spec.blocks[0].kind == "dense"


# --- the divergence gate ----------------------------------------------------- #

def _evidence(G, rate):
    from mimcs.factory.evidence import Evidence, Diagnostics
    return Evidence(samples=G[:200], coordinates=G, gradients=G,
                    diagnostics=Diagnostics(divergence_rate=rate))


def test_mass_mode_rule_skipped_when_evidence_diverged():
    """A pilot that diverged on > 10% of its transitions is not trusted for mode selection: the
    same spiked evidence yields low-rank at a low rate but *no* proposals (dimension-count default
    kept) once the divergence rate crosses the gate."""
    from mimcs.factory import analyze
    from mimcs.factory.rules import mass_mode_rule, MODE_SELECT_MAX_DIVERGENCE_RATE
    model = _one_block_model(30)
    spec = analyze(model)                                    # dimension-count partition (d=30 dense)
    G = _draw(_spiked(30, [20.0, 12.0, 6.0]), 2000, seed=0)
    below = MODE_SELECT_MAX_DIVERGENCE_RATE - 0.02
    above = MODE_SELECT_MAX_DIVERGENCE_RATE + 0.02
    props_ok = mass_mode_rule(spec, _evidence(G, below), model)
    assert any(p.slot.endswith(".kind") and p.value == "lowrank" for p in props_ok)
    assert mass_mode_rule(spec, _evidence(G, above), model) == []


def _funnel_shape_evidence(d=25, N=4000, rho=0.7, rate=0.0, seed=0):
    """A 2-block funnel: v ~ N(0,1); x-scores ~ exp(-v/2) N(0, R) (so cov ~ exp(-v) R, an
    Exp('v')-representable D with a compound-symmetry shape). Returns (model, evidence)."""
    from mimcs import Model
    from mimcs.model import EuclideanParameter
    from mimcs.factory.evidence import Evidence, Diagnostics
    rng = np.random.default_rng(seed)
    R = (1.0 - rho) * np.eye(d) + rho * np.ones((d, d))
    v = rng.standard_normal(N)
    gx = np.exp(-v / 2)[:, None] * (rng.standard_normal((N, d)) @ np.linalg.cholesky(R).T)
    coords = np.column_stack([v, np.zeros((N, d))])         # v at col 0, x coords unused by regression
    grads = np.column_stack([-v, gx])                        # v-score independent of x -> v stays diagonal
    model = Model([EuclideanParameter("v", ()), EuclideanParameter("x", (d,))],
                  {"lp": lambda p: -0.5 * (p["x"] ** 2).sum()})
    ev = Evidence(coordinates=coords, gradients=grads,
                  diagnostics=Diagnostics(divergence_rate=rate))
    return model, ev


def test_learned_metric_shape_skipped_when_evidence_diverged():
    """The learned metric on ``x`` is still adopted on a too-divergent pilot, but its shape stays
    diagonal (no ``shape`` key): the heavy-tailed scores are not trusted to pick a nondiagonal ``A``."""
    from mimcs.factory import analyze
    from mimcs.factory.rules import learned_metric_rule, MODE_SELECT_MAX_DIVERGENCE_RATE
    below = MODE_SELECT_MAX_DIVERGENCE_RATE - 0.02
    above = MODE_SELECT_MAX_DIVERGENCE_RATE + 0.02

    model, ev_ok = _funnel_shape_evidence(rate=below)
    spec = analyze(model)                                    # v diagonal, x its own (dense) block
    params_ok = next(p.value for p in learned_metric_rule(spec, ev_ok, model)
                     if p.slot.endswith(".params"))
    assert params_ok.get("shape") is not None                # a shape is selected without divergence

    model, ev_bad = _funnel_shape_evidence(rate=above)
    spec = analyze(model)
    props_bad = learned_metric_rule(spec, ev_bad, model)
    assert any(p.value == "learned_metric" for p in props_bad)   # metric still adopted
    params_bad = next(p.value for p in props_bad if p.slot.endswith(".params"))
    assert "shape" not in params_bad                             # but left diagonal


def test_divergence_rate_flows_from_a_live_sampler():
    """A live pilot exposes ``divergence_rate`` and the factory's evidence normaliser carries it."""
    from mimcs import Model, make_sampler
    from mimcs.model import EuclideanParameter
    from mimcs.factory.evidence import normalize
    model = Model([EuclideanParameter("x", (3,))], {"lp": lambda p: -0.5 * (p["x"] ** 2).sum()})
    s = make_sampler(model, seed=0)
    s.initialize(); s.warmup(40); s.sample(40)
    assert 0.0 <= s.divergence_rate() <= 1.0
    ev = normalize(model, s)
    assert ev.diagnostics.divergence_rate == s.divergence_rate()
