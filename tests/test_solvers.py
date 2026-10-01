"""Tests for the implicit-integrator fixed-point solvers (Picard, Anderson, Newton), which iterate
to a tolerance and report whether they got there.

Unit level: both find the same fixed point and say they converged; Anderson needs far fewer
evaluations on a stiff contraction; a map with no attracting fixed point is reported as a failure
(which the Riemannian kinetic turns into a divergence); the tolerance tracks x64; a spec typo
raises. Sampling level: on Neal's funnel with the misspecified conformal metric ``exp(-v) I`` the
generalized leapfrog's implicit solve often cannot be solved at all, and Anderson fails less often
than Picard at the same iteration cap.

Seeds are fixed, so pass/fail is deterministic.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from mimcs.hmc import PicardSolver, AndersonSolver, NewtonSolver
from mimcs.hmc.solvers import default_tol, resolve_solver
from mimcs.testing import neal_funnel, draw_samples, rmnuts


def test_anderson_matches_picard_fixed_point():
    a = jnp.array([1.0, -2.0, 0.5])
    g = lambda x: 0.5 * (jnp.cos(x) + a)            # a contraction; unique fixed point
    rp = PicardSolver().solve(g, jnp.zeros(3))
    ra = AndersonSolver().solve(g, jnp.zeros(3))
    assert bool(rp.converged) and bool(ra.converged)
    assert np.allclose(np.asarray(rp.x), np.asarray(ra.x), atol=1e-3)
    for r in (rp, ra):
        assert float(r.residual) <= default_tol()
        assert np.max(np.abs(np.asarray(g(r.x)) - np.asarray(r.x))) < 1e-3


def test_anderson_converges_in_fewer_iterations_on_stiff_maps():
    """40 random 6-d near-linear contractions with spectral radius up to 0.97: Anderson's median
    evaluation count is a fraction of Picard's (measured medians 43.5 vs 13 at depth 3), and both
    converge on every one."""
    rng = np.random.default_rng(1)
    n_p, n_a = [], []
    for _ in range(40):
        Q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
        A = jnp.asarray(Q @ np.diag(rng.uniform(-0.95, 0.97, size=6)) @ Q.T, float)
        c = jnp.asarray(rng.normal(size=6), float)
        g = lambda x: A @ x + 0.1 * jnp.tanh(x) + c
        rp = PicardSolver(max_iter=300).solve(g, jnp.zeros(6))
        ra = AndersonSolver(max_iter=300).solve(g, jnp.zeros(6))
        assert bool(rp.converged) and bool(ra.converged)
        n_p.append(int(rp.n_iter))
        n_a.append(int(ra.n_iter))
    assert np.median(n_a) < 0.5 * np.median(n_p), (np.median(n_a), np.median(n_p))


@pytest.mark.parametrize("solver", [PicardSolver(), AndersonSolver(depth=1)])
def test_no_fixed_point_is_reported_as_failure(solver):
    """An expanding map has no attracting fixed point: the solve runs to ``max_iter`` and says it
    did not converge (the control for the convergence claims above --- a solver that always
    reported success would pass them)."""
    r = solver.solve(lambda x: 2.0 * x + 1.0, jnp.zeros(2))
    assert not bool(r.converged)
    assert int(r.n_iter) == solver.max_iter


@pytest.mark.parametrize("warm_start", [1e-2, None])
def test_newton_matches_anderson_fixed_point(warm_start):
    a = jnp.array([1.0, -2.0, 0.5])
    g = lambda x: 0.5 * (jnp.cos(x) + a)
    rn = NewtonSolver(warm_start=warm_start).solve(g, jnp.zeros(3))
    ra = AndersonSolver().solve(g, jnp.zeros(3))
    assert bool(rn.converged)
    assert float(rn.residual) <= default_tol()
    assert np.allclose(np.asarray(rn.x), np.asarray(ra.x), atol=1e-3)


def test_newton_needs_no_contraction():
    """``g(x) = 2x + 1`` has the *repelling* fixed point ``x = -1``: Picard runs away (control),
    Newton solves it --- on a linear map one Newton step is exact, so the count is the first
    evaluation plus one step of ``d`` Jacobian products and one evaluation."""
    g = lambda x: 2.0 * x + 1.0
    assert not bool(PicardSolver().solve(g, jnp.zeros(2)).converged)
    r = NewtonSolver(warm_start=None).solve(g, jnp.zeros(2))
    assert bool(r.converged)
    assert np.allclose(np.asarray(r.x), -1.0)
    assert int(r.n_iter) == 1 + (2 + 1)


@pytest.mark.parametrize("warm_start", [1e-2, None])
def test_newton_reports_a_missing_root(warm_start):
    """``g(x) = x + 1 + x^2`` means ``F = 1 + x^2 > 0``: no fixed point exists (the implicit kick's
    situation past its critical step), and Newton must say so rather than return a point. (The
    iterates may run off to non-finite values first, which ends the loop early --- as for every
    solver, a NaN residual is never ``<= tol``.)"""
    r = NewtonSolver(warm_start=warm_start).solve(lambda x: x + 1.0 + x ** 2, jnp.zeros(2))
    assert not bool(r.converged)


def test_newton_converges_quadratically_on_a_slow_spiral():
    """A rotation-contraction of radius 0.97 (the drift's situation in a funnel's neck: complex
    eigenvalues of ``J_g`` near the unit circle) plus a weak nonlinearity. Picard needs hundreds of
    evaluations; Newton from the guess needs a handful of steps, each ``d + 1 = 3`` evaluations."""
    th = 1.2
    R = 0.97 * jnp.array([[jnp.cos(th), -jnp.sin(th)], [jnp.sin(th), jnp.cos(th)]])
    c = jnp.array([1.0, -0.5])
    g = lambda x: R @ x + 0.05 * jnp.tanh(x) + c
    rp = PicardSolver(max_iter=1000).solve(g, jnp.zeros(2))
    rn = NewtonSolver(warm_start=None).solve(g, jnp.zeros(2))
    assert bool(rp.converged) and bool(rn.converged)
    assert int(rp.n_iter) > 100
    assert int(rn.n_iter) <= 1 + 5 * 3, int(rn.n_iter)
    assert np.allclose(np.asarray(rn.x), np.asarray(rp.x), atol=1e-3)


def test_solve_is_jittable_and_the_tolerance_follows_the_float_type():
    g = lambda x: 0.5 * jnp.cos(x)
    r = jax.jit(lambda x0: AndersonSolver().solve(g, x0))(jnp.zeros(2))
    assert bool(r.converged)
    eps = float(jnp.finfo(jnp.asarray(0.0, float).dtype).eps)
    assert default_tol() == pytest.approx(eps ** 0.5)


def test_resolve_solver_validates_options():
    assert isinstance(resolve_solver(), AndersonSolver)
    assert resolve_solver("picard", max_iter=5).max_iter == 5
    with pytest.raises(ValueError, match="unknown solver option"):
        resolve_solver("picard", depth=3)            # Anderson-only option
    newton = resolve_solver("newton", warm_start=None, max_iter=12)
    assert isinstance(newton, NewtonSolver) and newton.warm_start is None
    with pytest.raises(ValueError, match="unknown solver option"):
        resolve_solver("newton", mixing=0.5)          # Anderson-only option
    with pytest.raises(ValueError, match="unknown solver"):
        resolve_solver("broyden")
    with pytest.raises(ValueError, match="alongside a solver object"):
        resolve_solver(PicardSolver(), max_iter=3)


def test_anderson_fails_less_than_picard_on_stiff_metric():
    """Neal's funnel with the misspecified conformal metric ``exp(-v) I``: at a step the adapted
    step size reaches, the implicit solve frequently has no reachable fixed point, and an unsolved
    step is rejected as a divergence --- so here *every* divergence is a solver failure. Anderson
    converges where Picard cannot more often, at the same cap of 8 evaluations.

    Averaged over 4 seeds because the per-seed gap varies (measured failure rates, Picard vs
    Anderson: .67/.50, .75/.55, .52/.51, .57/.58 --- one seed reversed; means .63 vs .53).
    Raising the cap to 30 did not lower either, so the failures are the implicit map not
    contracting at that step size, not slow convergence."""
    fun = neal_funnel(dim=2, scale=3.0)
    conformal = lambda q: jnp.exp(-q[0]) * jnp.eye(2)
    fails = {"picard": [], "anderson": []}
    for seed in range(4):
        for name in fails:
            s = rmnuts(metric=conformal, max_tree_depth=9, step_size=0.4, target_accept=0.8,
                       solver=name, max_iter=8)(fun.model, seed=seed)
            draw_samples(s, 500, 1500)
            rate = s.fixed_point_failure_rate(include_warmup=True)
            # every divergence here is a failed solve (and the counter is live, not stuck at 0)
            assert rate == pytest.approx(s.divergence_rate(include_warmup=True))
            fails[name].append(rate)
    print(f"\nfailure rates: {fails}")
    assert np.mean(fails["anderson"]) < np.mean(fails["picard"]) - 0.05, fails
