"""Tests for the diagonal-whitened rank-J (low-rank) quadratic kinetic and its adaptation.

Three layers: (1) the pure ``lowrank`` matrix algebra and the packing identity
``M = diag(D) + V^T V`` for ``V[j] = sqrt(gamma_j) sqrt(D) v_j``; (2) the
``LowRankQuadraticKinetic`` methods against a dense reference; (3) the ``LowRankAdaptation``
Sanger/Oja recovery and the end-to-end sampler on a stiff correlated Gaussian (correctness,
a mixing win over a diagonal mass, and an axis-aligned control where the low-rank part stays
inert).
"""

import types

import numpy as np
import jax.numpy as jnp

from mimcs.hmc import lowrank, NUTS, LowRankQuadraticKinetic
from mimcs.hmc.state import IntegratorState, HamiltonianContext
from mimcs.adaptation import RobbinsMonroStepSize, LowRankAdaptation
from mimcs.adaptation.lowrank_mass import _LowRankBlock
from mimcs.samplers import make_sampler_class
from mimcs.model import Model, EuclideanParameter
from mimcs.testing import correlated_gaussian, evaluate, nuts


def _random_lowrank(d, J, seed):
    """A random SPD mass in the low-rank form and its dense materialization."""
    rng = np.random.default_rng(seed)
    D = rng.uniform(0.5, 3.0, d)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    v = Q[:, :J].T                                   # (J, d) orthonormal rows
    gamma = rng.uniform(0.2, 5.0, J)
    V = np.sqrt(gamma)[:, None] * (np.sqrt(D)[None, :] * v)      # V[j] = sqrt(g) sqrt(D) v_j
    M = np.diag(D) + V.T @ V
    return D, V, gamma, v, M


def test_lowrank_packing_and_matvec_identities():
    D, V, gamma, v, M = _random_lowrank(9, 3, 0)
    # Packing reproduces D^{1/2}(I + sum gamma_j v v^T)D^{1/2}.
    M_ref = (np.diag(np.sqrt(D))
             @ (np.eye(9) + sum(gamma[j] * np.outer(v[j], v[j]) for j in range(3)))
             @ np.diag(np.sqrt(D)))
    assert np.allclose(M, M_ref)

    Dj, Vj = jnp.asarray(D), jnp.asarray(V)
    u = np.random.default_rng(1).standard_normal(9)
    assert np.allclose(np.asarray(lowrank.apply(Dj, Vj, u)), M @ u, rtol=1e-4, atol=1e-5)
    assert np.allclose(np.asarray(lowrank.apply_inv(Dj, Vj, u)),
                       np.linalg.solve(M, u), rtol=1e-4, atol=1e-5)
    assert np.allclose(M @ np.asarray(lowrank.apply_inv(Dj, Vj, u)), u, rtol=1e-4, atol=1e-5)
    # apply_chol is a square-root factor S with S S^T = M (used for momentum p = S z).
    S = np.stack([np.asarray(lowrank.apply_chol(Dj, Vj, jnp.asarray(e)))
                  for e in np.eye(9)], axis=1)
    assert np.allclose(S @ S.T, M, rtol=1e-4, atol=1e-5)


def test_lowrank_kinetic_energy_velocity_sample():
    D, V, gamma, v, M = _random_lowrank(6, 2, 2)
    Dj, Vj = jnp.asarray(D), jnp.asarray(V)
    k = LowRankQuadraticKinetic(id="T", rank=2)
    ctx = HamiltonianContext(chart_hyperparams={}, chart_indices={}, ham_params={"T": (Dj, Vj)})
    rng = np.random.default_rng(3)
    p = rng.standard_normal(6)
    ist = IntegratorState(q=jnp.zeros(6), p=jnp.asarray(p),
                          potential_values={}, potential_grads={}, log_weight=jnp.zeros(()))
    # energy = 1/2 p^T M^{-1} p ; velocity = M^{-1} p
    assert np.isclose(float(k.energy(ist, ctx)), 0.5 * p @ np.linalg.solve(M, p),
                      rtol=1e-4, atol=1e-5)
    vel = np.asarray(k.velocity_into(jnp.zeros(6), ist, ctx))
    assert np.allclose(vel, np.linalg.solve(M, p), rtol=1e-4, atol=1e-5)
    # sample_into maps z -> S z with Cov = M; check the mapping matches apply_chol.
    z = rng.standard_normal(6)
    draw = types.SimpleNamespace(T_momentum=jnp.asarray(z))
    p_new = np.asarray(k.sample_into(jnp.zeros(6), draw, ist.q, ctx))
    assert np.allclose(p_new, np.asarray(lowrank.apply_chol(Dj, Vj, jnp.asarray(z))),
                       rtol=1e-4, atol=1e-5)
    # identity initial mass -> energy = 1/2 |p|^2
    Di, Vi = k.initial_mass_params(6)
    ctx_id = ctx._replace(ham_params={"T": (Di, Vi)})
    assert np.isclose(float(k.energy(ist, ctx_id)), 0.5 * p @ p, rtol=1e-5, atol=1e-6)


def test_lowrank_kinetic_noncontiguous_slices():
    # A block over coordinates {0,1, 5,6} of an 8-vector.
    k = LowRankQuadraticKinetic(id="T", slices=[(0, 2), (5, 7)], rank=1)
    assert k._size(8) == 4
    Di, Vi = k.initial_mass_params(8)
    assert Di.shape == (4,) and Vi.shape == (1, 4)
    x = jnp.asarray(np.arange(8.0))
    assert np.array_equal(np.asarray(k._gather(x)), [0., 1., 5., 6.])
    base = k._scatter(jnp.zeros(8), jnp.asarray([10., 11., 12., 13.]))
    assert np.array_equal(np.asarray(base), [10., 11., 0., 0., 0., 12., 13., 0.])


# --- the Woodbury factors are hoisted out of the trajectory loop ------------------ #

def test_factored_inverse_matches_the_unfactored_one():
    """``inv_factors`` + ``apply_inv_factored`` is exactly the split of ``apply_inv``."""
    D, V, _, _, M = _random_lowrank(12, 3, 5)
    Dj, Vj = jnp.asarray(D), jnp.asarray(V)
    u = jnp.asarray(np.random.default_rng(6).standard_normal(12))
    beta, t = lowrank.inv_factors(Dj, Vj)
    got = np.asarray(lowrank.apply_inv_factored(Dj, beta, t, u))
    assert np.allclose(got, np.asarray(lowrank.apply_inv(Dj, Vj, u)), rtol=1e-6)
    assert np.allclose(got, np.linalg.solve(M, np.asarray(u)), rtol=1e-4, atol=1e-5)


def test_the_context_carries_factors_consistent_with_its_own_mass():
    """The cache is only safe because it is built from the *same* ``ham_params`` it is read
    beside --- ``BaseHMC.context`` fills both from one state, and nothing replaces a context's
    ``ham_params`` afterwards. A stale cache would be a wrong ``M^-1`` of the right shape.

    Warmed up first: at initialization ``gamma = 0`` so ``V`` is all zeros, the rank term is
    inert, and the check would hold no matter what the cache contained.
    """
    problem = _stiff_gaussian(d=6, n_stiff=2, seed=0)
    s = _lowrank_nuts(2)(problem.model, 0)
    s.warmup(300)                                   # let the adaptation put something in V
    ctx = s.context(s.state)
    D, V = ctx.ham_params["T"]
    assert float(jnp.max(jnp.abs(V))) > 0.0, "V is still zero: the test would be vacuous"
    beta, t = ctx.kinetic_cache["T"]
    ref_beta, ref_t = lowrank.inv_factors(D, V)
    assert np.allclose(np.asarray(beta), np.asarray(ref_beta), rtol=1e-6)
    assert np.allclose(np.asarray(t), np.asarray(ref_t), rtol=1e-6)


def test_the_cache_is_used_and_agrees_with_computing_the_factors_inline():
    """Two halves, and the second is what stops the first passing vacuously.

    The cached path must agree with the fallback that computes the factors inline, and with the
    dense reference -- *and* feeding a deliberately wrong cache must change the answer, which
    proves the cache is actually consulted rather than quietly ignored. Note the mass here is a
    genuine low-rank one: at ``V = 0`` the correction term vanishes and *any* cache gives the
    same answer.
    """
    d, J = 8, 2
    D, V, _, _, M = _random_lowrank(d, J, 7)
    Dj, Vj = jnp.asarray(D), jnp.asarray(V)
    k = LowRankQuadraticKinetic(id="T", rank=J)
    bare = HamiltonianContext(chart_hyperparams={}, chart_indices={},
                              ham_params={"T": (Dj, Vj)})
    ctx = bare._replace(kinetic_cache={"T": k.precompute(bare)})
    p = np.random.default_rng(8).standard_normal(d)
    istate = IntegratorState(q=jnp.zeros(d), p=jnp.asarray(p), potential_values={},
                             potential_grads={}, log_weight=jnp.zeros(()))

    cached, fallback = float(k.energy(istate, ctx)), float(k.energy(istate, bare))
    assert np.isclose(cached, fallback, rtol=1e-6)
    assert np.isclose(cached, 0.5 * p @ np.linalg.solve(M, p), rtol=1e-4, atol=1e-5)
    assert np.allclose(np.asarray(k.velocity_into(jnp.zeros(d), istate, ctx)),
                       np.linalg.solve(M, p), rtol=1e-4, atol=1e-5)

    wrong = bare._replace(kinetic_cache={"T": lowrank.inv_factors(Dj * 7.0 + 1.0, Vj)})
    assert not np.isclose(cached, float(k.energy(istate, wrong))), \
        "the cache is not being read: a wrong one made no difference"


def test_lowrank_adaptation_recovers_correlation_eigenstructure():
    """Fed scores ~ N(0, C), the block learns D -> diag(C), the top-J eigenvectors of the
    correlation matrix R = D^{-1/2} C D^{-1/2}, and gamma_j = max(0, lambda_j(R) - 1) >= 0."""
    rng = np.random.default_rng(4)
    d, J = 6, 2
    Dtrue = rng.uniform(0.5, 4.0, d)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    vt = Q[:, :J].T
    C = (np.diag(np.sqrt(Dtrue))
         @ (np.eye(d) + 5.0 * np.outer(vt[0], vt[0]) + 2.0 * np.outer(vt[1], vt[1]))
         @ np.diag(np.sqrt(Dtrue)))
    L = np.linalg.cholesky(C)
    Dc = np.diag(C)
    R = np.diag(1 / np.sqrt(Dc)) @ C @ np.diag(1 / np.sqrt(Dc))
    w_R, V_R = np.linalg.eigh(R)
    top_vals, top_vecs = w_R[::-1][:J], V_R[:, ::-1][:, :J]

    blk = _LowRankBlock(d, J, n0=5.0, kappa=0.75, clip_frac=0.1, center_grad=True,
                        mass_lr_const=1.0, oja_const=2.0, min_samples=200, polyak=False)
    for _ in range(40000):
        D_out, V_out = blk.update(L @ rng.standard_normal(d))
    D_out = np.asarray(D_out)

    gamma = np.maximum(0.0, blk._sanger.lam - 1.0)
    assert np.all(gamma >= 0.0)
    assert np.abs(D_out - Dc).max() / Dc.max() < 0.1               # D -> diag(C)
    assert np.allclose(blk._sanger.lam, top_vals, rtol=0.1)                # whitened eigenvalues
    # subspace angle between learned W and true top-J eigenvectors of R
    Qa, _ = np.linalg.qr(blk._sanger.W)
    Qb, _ = np.linalg.qr(top_vecs)
    smin = np.linalg.svd(Qa.T @ Qb, compute_uv=False).min()
    assert np.degrees(np.arccos(np.clip(smin, -1, 1))) < 10.0
    # the fitted mass reaches the best rank-J-plus-diagonal approximation of C
    M_rec = np.diag(D_out) + V_out.T @ V_out
    best = (np.diag(np.sqrt(Dc)) @ (np.eye(d)
            + sum((top_vals[j] - 1) * np.outer(top_vecs[:, j], top_vecs[:, j])
                  for j in range(J))) @ np.diag(np.sqrt(Dc)))
    assert (np.linalg.norm(M_rec - C) / np.linalg.norm(C)
            <= np.linalg.norm(best - C) / np.linalg.norm(C) + 0.05)


def test_lowrank_adaptation_burn_in_is_diagonal():
    """Before ``min_samples`` the low-rank part is inert (gamma = 0, a pure diagonal mass)."""
    rng = np.random.default_rng(5)
    blk = _LowRankBlock(5, 2, n0=5.0, kappa=0.75, clip_frac=0.1, center_grad=True,
                        mass_lr_const=1.0, oja_const=1.0, min_samples=30, polyak=False)
    for i in range(20):
        D_out, V_out = blk.update(rng.standard_normal(5))
        assert np.allclose(np.asarray(V_out), 0.0)      # gamma == 0 while in burn-in


def test_lowrank_whitened_score_clip_bounds_eigenvalue():
    """The adaptive whitened-score clip keeps the top eigenvalue estimate near the truth under
    heavy-tailed / transient outlier scores; without effective clipping the outliers inflate it."""
    d, J = 6, 2
    v0 = np.eye(d)[0]
    L = np.linalg.cholesky(np.eye(d) + 4.0 * np.outer(v0, v0))    # true top whitened eigenvalue 5

    def peak_lambda(clip_frac):
        rng = np.random.default_rng(0)
        # D frozen at 1 (mass_lr_const=0) so x_w = g (no whitening protection); 3% outliers x1000.
        blk = _LowRankBlock(d, J, n0=5.0, kappa=0.75, clip_frac=clip_frac, center_grad=False,
                            mass_lr_const=0.0, oja_const=1.0, min_samples=5, polyak=False)
        peak = 1.0
        for _ in range(4000):
            g = L @ rng.standard_normal(d)
            if rng.random() < 0.03:
                g = g * 1000.0
            blk.update(g)
            peak = max(peak, float(blk._sanger.lam.max()))
        return peak, float(np.exp(blk._sanger.log_clip_w))

    lam_clip, thr = peak_lambda(0.1)
    lam_noclip, _ = peak_lambda(0.0)                  # threshold only rises -> ~never clips
    assert lam_clip < 6.5                             # stays near the true eigenvalue (5)
    assert lam_clip < lam_noclip - 1.0                # clipping demonstrably tightens it
    assert thr < 50.0                                 # threshold adapted below the outlier scale


def _lowrank_nuts(rank, **kw):
    """A NUTS builder with a single whole-space low-rank kinetic + its adaptation."""
    Cls = make_sampler_class(RobbinsMonroStepSize, LowRankAdaptation, NUTS)

    def build(model, seed):
        return Cls(model, init_position=np.zeros(model.ambient_dim), seed=seed,
                   kinetics=[LowRankQuadraticKinetic(id="T", rank=rank)],
                   max_tree_depth=10, step_size=0.5, target_accept=0.8, **kw)
    return build


def _stiff_gaussian(d=6, n_stiff=2, seed=0):
    """A correlated Gaussian whose covariance has ``n_stiff`` off-axis low-variance
    (precision-stiff) directions -- a diagonal mass cannot precondition them."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    evals = np.ones(d)
    evals[:n_stiff] = 0.03                                  # two very stiff directions
    cov = Q @ np.diag(evals) @ Q.T
    cov = 0.5 * (cov + cov.T)
    return correlated_gaussian(mean=np.zeros(d), cov=cov)


def test_lowrank_nuts_recovers_stiff_gaussian():
    problem = _stiff_gaussian(d=6, n_stiff=2, seed=0)
    report = evaluate(problem, {"lowrank": _lowrank_nuts(2)},
                      n_warmup=2000, n_samples=8000, seed=0, make_plots=False)
    print("\n" + report.summary())
    report.assert_correct()


def test_lowrank_beats_diagonal_on_stiff_gaussian():
    """On a target with off-axis stiff directions a rank-2 mass mixes better than diagonal."""
    problem = _stiff_gaussian(d=6, n_stiff=2, seed=1)
    report = evaluate(
        problem,
        {"lowrank": _lowrank_nuts(2),
         "diagonal": nuts(metric="diagonal", mass_adapt="score", step_size=0.5)},
        n_warmup=2000, n_samples=8000, seed=0, make_plots=False)
    print("\n" + report.summary())
    ess_lr = report.outputs["lowrank"].ess.min()
    ess_diag = report.outputs["diagonal"].ess.min()
    assert ess_lr > 1.3 * ess_diag, f"lowrank min-ESS {ess_lr:.0f} !> 1.3 * diagonal {ess_diag:.0f}"


def test_lowrank_control_axis_aligned_gaussian():
    """On an axis-aligned (diagonal) target the low-rank part stays near-zero: it must not
    manufacture spurious correlation, and it still samples correctly."""
    d = 5
    cov = np.diag(np.array([0.5, 1.0, 2.0, 3.0, 4.0]))
    problem = correlated_gaussian(mean=np.zeros(d), cov=cov)
    build = _lowrank_nuts(2)
    sampler = build(problem.model, seed=0)
    sampler.warmup(2000)
    draws = np.asarray(sampler.sample(8000)["x"])
    emp_cov = np.cov(draws.T)
    assert np.linalg.norm(emp_cov - cov) / np.linalg.norm(cov) < 0.1
    # learned low-rank rows are small (whitened eigenvalues near 1 -> gamma ~ 0)
    _, V = sampler.state.ham_params["T"]
    assert np.linalg.norm(np.asarray(V)) < 0.75
