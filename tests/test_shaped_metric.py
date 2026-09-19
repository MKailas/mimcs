"""Tests for the shaped (nondiagonal) learned metric ``M(x) = D(x)^{1/2} A D(x)^{1/2}``.

``D(x)`` is a learned mini-language diagonal over other blocks; ``A`` is a *constant* shape --
dense ``K K^T`` or low-rank ``I + sum_j gamma_j v_j v_j^T`` (:class:`mimcs.hmc.ShapedLearnedBlock`,
adapted by :class:`mimcs.adaptation.ShapedMetricAdaptation`). What must hold: the kinetic algebra
(``M^{-1} p``, energy, momentum covariance) for both shapes; that the adaptation recovers ``D(x)``
and ``A`` on a target whose ideal metric has this exact form; that both shapes sample it correctly;
and that ``shape=None`` is the plain diagonal metric.

Seeds are fixed, so pass/fail is deterministic.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from mimcs import Model
from mimcs.model import EuclideanParameter
from mimcs.hmc import build_block, ShapedLearnedBlock, LearnedDiagonalBlock
from mimcs.hmc.metric_expr import Exp as ExprExp
from mimcs.factory import analyze
from mimcs.factory.spec import BlockSpec
from mimcs.testing import evaluate, correlated_gaussian
from mimcs.testing.problems import TargetProblem


# --- a funnel-with-correlation target: ideal metric E[gg^T|v] = e^{-v} R = D(v)^1/2 R D(v)^1/2 with
#     D(v) = e^{-v} (log-linear in v, Exp("v") representable) and R the constant correlation shape.

def funnel_correlated(n: int = 3, scale: float = 1.5, rho: float = 0.4):
    """``v ~ N(0, scale^2)``, ``x | v ~ N(0, e^v R^{-1})`` with ``R`` a compound-symmetry
    correlation (diagonal 1, off-diagonal ``rho``) -- deterministic, so the test is stable. The
    conditional score covariance is ``e^{-v} R = D(v)^{1/2} R D(v)^{1/2}`` (``D(v) = e^{-v}``), so
    the ideal constant shape is exactly ``R``."""
    R = (1.0 - rho) * np.eye(n) + rho * np.ones((n, n))
    Rj = jnp.asarray(R)

    def log_post(p):
        v = jnp.squeeze(p["v"]); x = p["x"]
        return -0.5 * v ** 2 / scale ** 2 - 0.5 * jnp.exp(-v) * (x @ Rj @ x) - 0.5 * n * v

    model = Model([EuclideanParameter("v", ()), EuclideanParameter("x", (n,))], {"lp": log_post})
    chol_Rinv = np.linalg.cholesky(np.linalg.inv(R))

    def sampler(m, rng):
        v = scale * rng.standard_normal(m)
        x = np.exp(v / 2)[:, None] * (rng.standard_normal((m, n)) @ chol_Rinv.T)
        return np.column_stack([v, x])

    problem = TargetProblem(name="funnel_correlated", model=model, dim=n + 1,
                            labels=["v"] + [f"x{i}" for i in range(n)], exact_sampler=sampler)
    return problem, R


def _shaped_builder(shape):
    """A factory-spec builder: v diagonal, x a shaped learned metric depending on v."""
    def build(model, seed):
        spec = analyze(model)
        vs, ve = model.coord_block("v")
        xs, xe = model.coord_block("x")
        spec.blocks = [
            BlockSpec(["v"], [(vs, ve)], "diagonal"),
            BlockSpec(["x"], [(xs, xe)], "learned_metric",
                      params={"metric": ExprExp("v") + ExprExp(), "shape": shape}),
        ]
        spec.terminate = None
        return spec.build(seed=seed)
    return build


def _recovered_A(sampler, n):
    hp = sampler.state.ham_params["x"]["shape"]
    if isinstance(hp, tuple):                      # low-rank: (W, gamma)
        W, gamma = np.asarray(hp[0]), np.asarray(hp[1])
        return np.eye(n) + (W * gamma) @ W.T
    K = np.asarray(hp)                             # dense: K
    return K @ K.T


# --- kinetic algebra (unit) -------------------------------------------------- #

def _block(shape):
    # block "x" at coords [0:4], dependency "v" at [4:6]
    return ShapedLearnedBlock("x", (0, 4), ExprExp("v"), {"v": [(4, 6)]}, shape)


@pytest.mark.parametrize("shape", ["dense", ("lowrank", 2)])
def test_kinetic_matches_dense_reference(shape):
    """velocity = M(x)^{-1} p, energy = 1/2 p^T M^{-1} p + 1/2 logdet M, and a sampled momentum
    factor S with S S^T = M(x) -- all against a plain dense reference."""
    rng = np.random.default_rng(0)
    n = 4
    blk = _block(shape)
    q = jnp.asarray(rng.standard_normal(6))
    p = jnp.asarray(rng.standard_normal(n))
    dp = {"W": [jnp.asarray(rng.standard_normal((n, 2)) * 0.5)], "b": jnp.asarray(rng.standard_normal(n) * 0.3)}
    v = np.asarray(q[4:6])
    D = np.exp(np.asarray(dp["W"][0]) @ v + np.asarray(dp["b"]))

    if shape == "dense":
        K = np.tril(rng.standard_normal((n, n))); np.fill_diagonal(K, np.abs(np.diag(K)) + 0.5)
        A = K @ K.T
        params = {"diag": dp, "shape": jnp.asarray(K)}
    else:
        W, _ = np.linalg.qr(rng.standard_normal((n, 2))); gamma = np.array([1.3, 0.4])
        A = np.eye(n) + (W * gamma) @ W.T
        params = {"diag": dp, "shape": (jnp.asarray(W), jnp.asarray(gamma))}

    M = np.diag(np.sqrt(D)) @ A @ np.diag(np.sqrt(D))
    ref_vel = np.linalg.solve(M, np.asarray(p))
    ref_en = 0.5 * np.asarray(p) @ ref_vel + 0.5 * np.linalg.slogdet(M)[1]
    # `None` is the label vector: these blocks have no discrete dependency, and a metric that
    # declares none never indexes it (mimcs/hmc/metric_encode.py).
    S = np.column_stack([np.asarray(blk._sample_factor(q, None, jnp.asarray(e), params))
                         for e in np.eye(n)])

    assert np.allclose(np.asarray(blk._velocity(q, None, p, params)), ref_vel, atol=1e-3, rtol=1e-3)
    assert np.isclose(float(blk._energy(q, None, p, params)), ref_en, atol=1e-3, rtol=1e-3)
    assert np.allclose(S @ S.T, M, atol=1e-3, rtol=1e-3)


def test_shape_none_is_the_plain_diagonal_block():
    """``build_block`` with ``shape=None`` builds the diagonal metric; a shaped block with ``A=I``
    matches it numerically."""
    problem, _ = funnel_correlated()
    model = problem.model
    diag = build_block(model, "x", ExprExp("v"))
    shaped = build_block(model, "x", ExprExp("v"), shape="dense")
    assert isinstance(diag, LearnedDiagonalBlock) and isinstance(shaped, ShapedLearnedBlock)
    # at A = I (K = I), the shaped energy/velocity equal the diagonal block's
    q = jnp.asarray(np.random.default_rng(0).standard_normal(model.coord_dim))
    n = shaped.size
    p = jnp.asarray(np.random.default_rng(1).standard_normal(n))
    dp = diag.init_params()
    e_diag = float(diag._energy(q, None, p, dp))
    e_shaped = float(shaped._energy(q, None, p, {"diag": dp, "shape": jnp.eye(n)}))
    assert np.isclose(e_diag, e_shaped, atol=1e-4, rtol=1e-4)


# --- adaptation recovery ----------------------------------------------------- #

def test_adaptation_recovers_dense_shape():
    """On the funnel-correlated target the dense fit recovers the ideal constant shape ``R``
    (``K K^T -> corr`` of the whitened score). (The low-rank Sanger fit is covered by
    ``test_lowrank_mass.py``'s eigenstructure test -- the same ``_Sanger`` is reused here.)

    The bound is 0.25, loosened from 0.15 when ``D(x)`` got MetricAdaptation's per-coordinate
    clip: whitening by that noisier raw ``D(x)`` iterate biases ``A``'s diagonal high (5 seeds:
    median max error 0.162, max 0.206, was 0.080; ``writeups/irt_shaped_dx_parity.md``), an
    accepted price for the clip's stability. Still non-vacuous: an unadapted ``A = I`` misses
    ``R`` by 0.4 (the off-diagonal ``rho``)."""
    problem, R = funnel_correlated(n=3, rho=0.4)
    sampler = _shaped_builder("dense")(problem.model, seed=0)
    sampler.initialize(); sampler.warmup(6000); sampler.sample(1000)
    Ahat = _recovered_A(sampler, 3)
    assert np.abs(Ahat - R).max() < 0.25, (Ahat, R)


# --- ergodicity -------------------------------------------------------------- #

def _gaussian_shaped_builder(shape, d):
    """A shaped metric with *constant* D (dep-less ``Exp()``) on a single Gaussian block."""
    def build(model, seed):
        spec = analyze(model)
        spec.blocks = [BlockSpec(["x"], [(0, d)], "learned_metric",
                                 params={"metric": ExprExp(), "shape": shape})]
        spec.terminate = None
        return spec.build(seed=seed)
    return build


@pytest.mark.parametrize("shape", ["dense", ("lowrank", 3)])
def test_shaped_metric_samples_gaussian(shape, artifacts_dir):
    """Strong within-block correlation, constant D: the shaped metric samples a correlated Gaussian
    correctly -- the clean ergodicity + shape-``A`` check (both shapes)."""
    rng = np.random.default_rng(3)
    B = rng.standard_normal((4, 4))
    cov = (B @ B.T + np.eye(4)).tolist()
    problem = correlated_gaussian(mean=[1.0, -2.0, 0.5, 3.0], cov=cov)
    tag = "dense" if shape == "dense" else "lowrank"
    report = evaluate(problem, {f"gauss_{tag}": _gaussian_shaped_builder(shape, 4)},
                      n_warmup=3000, n_samples=12000, seed=0,
                      out_dir=str(artifacts_dir / f"shaped_gauss_{tag}"))
    print("\n" + report.summary())
    report.assert_correct()


def test_shaped_metric_samples_funnel(artifacts_dir):
    """Position-dependent ``D(x)`` (the funnel) with a dense shape -- exercises the explicit
    metric-derivative kick without bias."""
    problem, _ = funnel_correlated(n=3, scale=1.5, rho=0.4)
    report = evaluate(problem, {"shaped_funnel": _shaped_builder("dense")},
                      n_warmup=5000, n_samples=12000, seed=0,
                      out_dir=str(artifacts_dir / "shaped_funnel"))
    print("\n" + report.summary())
    report.assert_correct()


# --- D(x) adaptation parity with MetricAdaptation ------------------------------ #
#
# The shaped adapter used to fit D(x) with its own step: one global gradient-norm clip, an undivided
# shared-leaf gradient, no non-finite guard, the raw iterate frozen. On `irt_2pl` that could not be
# separated from the shape as the cause of a stage-2 collapse, so D(x) now runs MetricAdaptation's
# exact step. These pin the parity and each of the four behaviours it brings.

from mimcs.adaptation.metric import _make_kl_step, _PerUnitClip   # noqa: E402

_SHARED_EXPR = ExprExp("v", shared_weights=(0,)) + ExprExp()


def _parity_blocks(size=5):
    shaped = ShapedLearnedBlock("x", (0, size), _SHARED_EXPR, {"v": [(size, size + 1)]}, ("lowrank", 2))
    plain = LearnedDiagonalBlock("x", (0, size), _SHARED_EXPR, {"v": [(size, size + 1)]})
    return shaped, plain


def _scores(size, steps, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for t in range(steps):
        q = rng.standard_normal(size + 1)
        s = rng.standard_normal(size + 1) * np.exp(-q[size] / 2)
        if t % 7 == 3:
            s[1] *= 60.0                  # a large gradient on one coordinate: exercises the clips
        out.append((jnp.asarray(q), jnp.asarray(s)))
    return out


def test_shaped_dx_step_is_metric_adaptations_step():
    """With A = I, feeding the same (q, score) sequence through the shaped D(x) step and through
    MetricAdaptation's gives bit-identical parameters -- and the old global-norm step does not."""
    size = 5
    shaped, plain = _parity_blocks(size)
    p_plain = plain.init_params()
    p_shaped = shaped.init_params()["diag"]
    shape_I = shaped.init_params()["shape"]
    step_plain = _make_kl_step(lambda p, q, l, s, lr: plain.metric_loss(p, q, l, s), size)
    step_shaped = _make_kl_step(
        lambda dp, sh, q, l, s, lr: shaped.metric_loss({"diag": dp, "shape": sh}, q, l, s), size)
    clip_plain, clip_shaped = _PerUnitClip(p_plain, size), _PerUnitClip(p_shaped, size)
    p_old = p_shaped
    log_clip_old = np.log(size)
    for t, (q, s) in enumerate(_scores(size, 60), start=1):
        lr = (t + 5.0) ** -0.75
        g, gn, sn = step_plain(p_plain, q, None, s, lr)
        p_plain, _, _ = clip_plain.update(p_plain, g, gn, sn, lr, lr, 0.1)
        g, gn, sn = step_shaped(p_shaped, shape_I, q, None, s, lr)
        p_shaped, _, _ = clip_shaped.update(p_shaped, g, gn, sn, lr, lr, 0.1)
        # control: the pre-parity shaped step (global norm, undivided gradient)
        g_old = jax.grad(lambda dp: shaped.metric_loss({"diag": dp, "shape": shape_I}, q, None, s))(p_old)
        gnorm = float(np.sqrt(sum(np.sum(np.asarray(l) ** 2) for l in jax.tree_util.tree_leaves(g_old))))
        thr = np.exp(log_clip_old)
        p_old = jax.tree_util.tree_map(lambda w, gw: w - lr * min(1.0, thr / (gnorm + 1e-12)) * gw,
                                       p_old, g_old)
        log_clip_old += lr * ((1.0 if gnorm > thr else 0.0) - 0.1)
    for a, b in zip(jax.tree_util.tree_leaves(p_plain), jax.tree_util.tree_leaves(p_shaped)):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    assert any(not np.allclose(np.asarray(a), np.asarray(b), atol=1e-6)
               for a, b in zip(jax.tree_util.tree_leaves(p_old), jax.tree_util.tree_leaves(p_shaped)))


def test_per_unit_clip_decouples_coordinates_and_skips_nonfinite():
    """One coordinate's huge gradient does not shrink the others' step (a global norm would), and a
    non-finite coordinate is skipped while the rest still descend."""
    size = 4
    params = {"W": [jnp.zeros((size, 1))], "b": jnp.zeros(size)}
    g = {"W": [jnp.asarray([[0.1], [0.1], [1e4], [0.1]])], "b": jnp.asarray([0.1, 0.1, 1e4, np.nan])}
    rows = np.sqrt(np.asarray(g["W"][0])[:, 0] ** 2 + np.asarray(g["b"]) ** 2)
    clip = _PerUnitClip(params, size)
    new, n_bad, applied = clip.update(params, g, jnp.asarray(rows), [], 0.5, 0.5, 0.1)
    b = np.asarray(new["b"])
    assert applied and n_bad == 1
    assert b[3] == 0.0                                   # the non-finite coordinate did not move
    assert np.isclose(b[0], -0.5 * 0.1) and np.isclose(b[1], -0.5 * 0.1)   # unclipped: full step
    # control: one global norm over the (finite) block would have scaled coordinate 0 by ~1e-4
    global_scale = min(1.0, size / float(np.linalg.norm(rows[np.isfinite(rows)])))
    assert global_scale < 1e-3


def test_shaped_warmup_keeps_shared_leaves_shared_and_freezes_the_ema():
    """In a live warmup the shared weight stays (1, 1), and by default (``mass_ema`` is on for the
    learned metrics) sampling uses the EMA of D(x) rather than the raw iterate."""
    problem, _ = funnel_correlated(n=3, rho=0.4)
    model = problem.model
    spec = analyze(model)
    vs, ve = model.coord_block("v")
    xs, xe = model.coord_block("x")
    spec.blocks = [BlockSpec(["v"], [(vs, ve)], "diagonal"),
                   BlockSpec(["x"], [(xs, xe)], "learned_metric",
                             params={"metric": _SHARED_EXPR, "shape": ("lowrank", 1)})]
    spec.terminate = None
    sampler = spec.build(seed=0)
    sampler.initialize()
    sampler.warmup(300)
    assert np.shape(jax.tree_util.tree_leaves(sampler.state.ham_params["x"]["diag"])[0]) == (1, 1)
    sampler.sample(1)                              # `_finalize_hooks` runs on the first `sample`
    frozen = sampler.state.ham_params["x"]["diag"]
    assert np.shape(jax.tree_util.tree_leaves(frozen)[0]) == (1, 1)
    for a, b in zip(jax.tree_util.tree_leaves(frozen), jax.tree_util.tree_leaves(sampler._shp_ema["x"])):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    assert any(not np.array_equal(np.asarray(a), np.asarray(b)) for a, b in
               zip(jax.tree_util.tree_leaves(frozen), jax.tree_util.tree_leaves(sampler._shp_diag["x"])))
    assert sampler.shaped_nonfinite_count() == 0


def test_score_centring_is_off_by_default_and_reaches_the_shaped_block_when_on():
    """``metric_center_grad`` is read by ``ShapedMetricAdaptation`` too: off by default (no running
    mean at all), and on it tracks one and moves the fit. Regression guard -- the flag used to be
    read only by ``MetricAdaptation``, so it was silently inert on *shaped* blocks, which are
    exactly the blocks the ``irt_2pl`` stage-2 collapse sits in."""
    problem, _ = funnel_correlated(n=3, rho=0.4)
    model = problem.model
    vs, ve = model.coord_block("v")
    xs, xe = model.coord_block("x")
    out = {}
    for center in (False, True):
        spec = analyze(model)
        spec.blocks = [BlockSpec(["v"], [(vs, ve)], "diagonal"),
                       BlockSpec(["x"], [(xs, xe)], "learned_metric",
                                 params={"metric": ExprExp("v") + ExprExp(),
                                         "shape": ("lowrank", 1)})]
        spec.terminate = None
        spec.algo_kwargs = {**spec.algo_kwargs, "metric_center_grad": center}
        sampler = spec.build(seed=0)
        sampler.initialize().warmup(200)
        out[center] = (sampler._shp_center_grad, sampler._shp_mean_grad,
                       jax.tree_util.tree_leaves(sampler.state.ham_params["x"]["diag"]))
    assert out[False][0] is False and out[False][1] is None       # default: nothing is tracked
    assert out[True][0] is True
    mean = np.asarray(out[True][1], float)
    assert np.all(np.isfinite(mean)) and np.abs(mean).max() > 0.0
    assert any(not np.allclose(np.asarray(a), np.asarray(b))      # and it changes D(x)
               for a, b in zip(out[False][2], out[True][2]))


# --- EMA-driven warmup (`mass_ema_warmup`) ------------------------------------ #
#
# Off by default: the raw D(x) iterate drives warmup and whitens A (an EMA of it is frozen for
# sampling: `mass_ema`, on by default for the learned metrics). `mass_ema_warmup`
# makes an exponential moving average of the D(x) parameters (the SGD's Robbins-Monro gain) drive the
# simulation, whiten the shape A, and be frozen for sampling. The keys are shared by every mass
# adaptation (mimcs.adaptation._ema; tests/test_ema.py), so plain and shaped blocks run the same D(x).

from mimcs.adaptation._stochastic import rm_gain, DEFAULT_KAPPA, DEFAULT_N0   # noqa: E402
from mimcs.adaptation.lowrank_mass import _Sanger                            # noqa: E402


def _funnel_sampler(shape, **algo):
    """v diagonal, x a learned metric on v -- plain (``shape=None``) or shaped."""
    problem, _ = funnel_correlated(n=3, rho=0.4)
    model = problem.model
    spec = analyze(model)
    vs, ve = model.coord_block("v")
    xs, xe = model.coord_block("x")
    params = {"metric": ExprExp("v") + ExprExp()}
    if shape is not None:
        params["shape"] = shape
    spec.blocks = [BlockSpec(["v"], [(vs, ve)], "diagonal"),
                   BlockSpec(["x"], [(xs, xe)], "learned_metric", params=params)]
    spec.terminate = None
    spec.algo_kwargs = {**spec.algo_kwargs, **algo}
    return spec.build(seed=0)


def _leaves(tree):
    return [np.asarray(l, dtype=float) for l in jax.tree_util.tree_leaves(tree)]


def _views(sampler, state, shaped):
    """(step count, raw D(x) iterate, its EMA, the D(x) params the chain is simulated with)."""
    if shaped:
        return (sampler._shp_count, sampler._shp_diag.get("x"), sampler._shp_ema.get("x"),
                state.ham_params["x"]["diag"])
    return (sampler._metric_count, sampler._metric_params.get("x"),
            sampler._metric_ema.get("x"), state.ham_params["x"])


@pytest.mark.parametrize("shaped", [False, True])
def test_mass_ema_warmup_is_off_by_default(shaped):
    """Default: the raw iterate is what the chain is simulated with; the EMA (``mass_ema``, on by
    default for the learned metrics) is kept only to be frozen for sampling."""
    sampler = _funnel_sampler(("lowrank", 1) if shaped else None)
    sampler.initialize().warmup(200)
    _, raw, ema, written = _views(sampler, sampler.state, shaped)
    assert ema is not None
    assert all(np.array_equal(a, b) for a, b in zip(_leaves(raw), _leaves(written)))
    assert not all(np.allclose(a, b) for a, b in zip(_leaves(ema), _leaves(written)))


@pytest.mark.parametrize("shaped", [False, True])
def test_mass_ema_warmup_drives_warmup_and_is_frozen(shaped):
    """On: every warmup step simulates with the EMA, which starts at the first iterate and then
    follows ``ema_n = ema_{n-1} + eta_n (raw_n - ema_{n-1})`` with the RM gain ``eta_n`` of the SGD
    step; sampling freezes that EMA -- for a plain and a shaped block alike."""
    sampler = _funnel_sampler(("lowrank", 1) if shaped else None, mass_ema_warmup=True)
    hook, rec = sampler._postprocess_hooks, []

    def recording(state):
        state = hook(state)
        n, raw, ema, written = _views(sampler, state, shaped)
        rec.append((n, _leaves(raw), _leaves(ema), _leaves(written)))
        return state

    sampler._postprocess_hooks = recording
    sampler.initialize().warmup(200)
    del sampler._postprocess_hooks                  # back to the class's hook for sampling
    skipped = sampler.shaped_nonfinite_count() if shaped else sampler.metric_nonfinite_count()
    assert len(rec) == 200 and skipped == 0

    _, raw0, ema0, written0 = rec[0]
    assert all(np.array_equal(a, b) for a, b in zip(raw0, ema0))       # starts at the first iterate
    assert all(np.array_equal(a, b) for a, b in zip(ema0, written0))
    for (_, _, ema_prev, _), (n, raw, ema, written) in zip(rec, rec[1:]):
        eta = rm_gain(n, DEFAULT_N0, DEFAULT_KAPPA)
        for p, r, e in zip(ema_prev, raw, ema):
            np.testing.assert_allclose(e, p + eta * (r - p), rtol=1e-5, atol=1e-6)
        assert all(np.array_equal(a, b) for a, b in zip(ema, written))  # the EMA drives warmup
    # Non-vacuous: by the end the EMA really lags the raw iterate.
    assert any(not np.allclose(a, b, atol=1e-4) for a, b in zip(rec[-1][1], rec[-1][2]))

    sampler.sample(1)                               # `_finalize_hooks` runs on the first `sample`
    frozen = sampler.state.ham_params["x"]["diag"] if shaped else sampler.state.ham_params["x"]
    assert all(np.array_equal(a, b) for a, b in zip(_leaves(frozen), rec[-1][2]))


def test_mass_ema_warmup_whitens_the_shape_by_the_ema(monkeypatch):
    """On: the shape tracker is fed the score whitened by the EMA ``D(x)``, not the raw iterate."""
    seen = []
    step = _Sanger.step

    def spy(self, x_w, lr, count):
        seen.append(np.array(x_w, dtype=float))
        return step(self, x_w, lr, count)

    monkeypatch.setattr(_Sanger, "step", spy)
    sampler = _funnel_sampler(("lowrank", 1), mass_ema_warmup=True)
    sampler.initialize().warmup(200)
    k = next(k for k in sampler.kinetics if k.id == "x")
    state = sampler.state
    labels = getattr(state, "discrete", None)
    score = np.asarray(sum(state.potential_grads.values()), dtype=float)[k.s:k.e]

    def whiten(params):
        return score / np.sqrt(np.asarray(k._D(state.coordinate, labels, params), dtype=float))

    assert seen, "the shape tracker never ran"
    np.testing.assert_allclose(seen[-1], whiten(sampler._shp_ema["x"]), rtol=1e-5)
    assert not np.allclose(seen[-1], whiten(sampler._shp_diag["x"]), rtol=1e-3)   # control


# --- automatic shape selection (the factory rule) ---------------------------- #

def test_whitened_scores_recover_the_shape_correlation():
    """`whitened_scores` h = g / sqrt(D(x)) has correlation ~ R when g ~ N(0, D(x)^1/2 R D(x)^1/2):
    exactly the shape A's target that `select_mass_mode` is then run on."""
    from mimcs.factory.regression import whitened_scores
    d, N = 8, 4000
    rng = np.random.default_rng(0)
    R = 0.5 * np.eye(d) + 0.5 * np.ones((d, d))            # compound-symmetry correlation
    v = rng.standard_normal(N)
    g = np.exp(-v / 2)[:, None] * (rng.standard_normal((N, d)) @ np.linalg.cholesky(R).T)
    coords = np.column_stack([v, np.zeros((N, d))])        # v at column 0
    grads = np.column_stack([np.zeros(N), g])              # block scores at columns 1..d
    params = {"W": [-np.ones((d, 1))], "b": np.zeros(d)}   # D(x) = exp(-v)
    h = whitened_scores(ExprExp("v"), params, list(range(1, d + 1)), {"v": [0]}, coords, grads)
    assert np.abs(np.corrcoef(h.T) - R).max() < 0.08


@pytest.mark.parametrize("rho, expect_shape", [(0.7, True), (0.0, False)])
def test_shape_selected_from_pilot_evidence(rho, expect_shape):
    """A pilot fed back to `analyze`: a funnel with strong within-block correlation gives the x
    learned metric a (non-None) shape; with no correlation the shape is None (plain diagonal
    metric). Exercises the whole rule: regress D(x) -> whiten -> select_mass_mode."""
    from mimcs import make_sampler
    problem, _ = funnel_correlated(n=20, rho=rho)
    pilot = make_sampler(problem.model, seed=0)
    pilot.initialize(); pilot.warmup(1200); pilot.sample(1200)
    spec = analyze(problem.model, pilot)
    xb = next(b for b in spec.blocks if b.names == ["x"])
    assert xb.kind == "learned_metric"
    assert (xb.params.get("shape") is not None) is expect_shape
