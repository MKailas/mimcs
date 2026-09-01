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
    ``test_lowrank_mass.py``'s eigenstructure test -- the same ``_Sanger`` is reused here.)"""
    problem, R = funnel_correlated(n=3, rho=0.4)
    sampler = _shaped_builder("dense")(problem.model, seed=0)
    sampler.initialize(); sampler.warmup(6000); sampler.sample(1000)
    Ahat = _recovered_A(sampler, 3)
    assert np.abs(Ahat - R).max() < 0.15, (Ahat, R)


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
