"""Tests for evidence-informed mass-matrix mode selection (``mimcs.factory.mode_select``).

The selector reads a block's mass structure off the spectrum of the whitened score covariance
``R = D^{-1/2} S D^{-1/2}``. What must hold: the whitened-spectrum math (the ``d>n`` Gram path
matches the full eigendecomposition; the KL-loss decomposition over eigenvalues is exact), the
boundary gates (small-d -> dense, ~1-order -> diagonal, ``d>n`` forbids dense, ``J <= d/5``), and
that the recovered mode matches ground truth on clean spiked / isotropic / dense targets.

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
    """``d > n`` -> never dense; and any low-rank J respects ``J <= floor(d/5)`` and ``J < n``."""
    d, n = 200, 80                                          # d > n
    X = _draw(_spiked(d, [30.0, 18.0, 9.0]), n, seed=0)
    kind, J = select_mass_mode(X, rule=rule)
    assert kind in ("diagonal", "lowrank")                 # not dense
    if kind == "lowrank":
        assert J <= int(J_MAX_FRAC * d) and J < n


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
