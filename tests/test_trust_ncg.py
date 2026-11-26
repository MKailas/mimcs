"""The trust-region Newton-CG (``mimcs.optim.trust_ncg`` and DADVI's ``optimizer="trust_ncg"``).

Both are checked in x64 against SciPy, the implementation they follow: Steihaug's CG against
``scipy.optimize._trustregion_ncg.CGSteihaugSubproblem`` on quadratics built to take each of its
exits (each case asserts the exit, so none passes vacuously), and a whole DADVI fit against
``scipy.optimize.minimize(method="trust-ncg")`` on the same fixed-draw objective, from starts
far enough out that steps are rejected and the radius shrinks. Then the departures: non-finite
proposals are rejected, the Newton-decrement stop, the CG cap, and the secant read-out.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.optimize as so
from scipy.optimize._trustregion_ncg import CGSteihaugSubproblem

from mimcs import config
from mimcs.optim.trust_ncg import steihaug_cg
from mimcs.vi import dadvi
from mimcs.vi.deterministic import (READOUTS, STOP_REASONS, _init, _objective, _pattern,
                                    _tr_step, dadvi_options)


@pytest.fixture
def x64():
    was = config.x64_enabled()
    config.enable_x64(True)
    try:
        yield
    finally:
        config.enable_x64(was)


def _scipy_step(g, H, radius):
    sp = CGSteihaugSubproblem(np.zeros_like(g), lambda x: 0.0, lambda x: g,
                              hessp=lambda x, p: H @ p)
    return sp.solve(radius)


def _ours(g, H, radius, max_iters=1000):
    Hj = jnp.asarray(H)
    return jax.jit(lambda g, r: steihaug_cg(g, lambda v: Hj @ v, r, max_iters))(
        jnp.asarray(g), radius)


@pytest.mark.parametrize("case", ["interior", "boundary", "negative_curvature"])
def test_steihaug_cg_is_scipys(x64, case):
    """Same step as SciPy on each exit, and H p carried exactly (so m(p) needs no product)."""
    rng = np.random.default_rng(0)
    n = 12
    for rep in range(4):
        A = rng.normal(size=(n, n))
        g = rng.normal(size=n)
        if case == "negative_curvature":
            H = -(A @ A.T) - 0.1 * np.eye(n)          # every direction: the first d.Hd < 0
            radius = 2.0
        else:
            H = A @ A.T + 0.1 * np.eye(n)
            radius = 1e6 if case == "interior" else 0.05
        p_ref, hit_ref = _scipy_step(g, H, radius)
        r = _ours(g, H, radius)
        assert bool(r.hits_boundary) == hit_ref == (case != "interior")
        assert not bool(r.capped)
        np.testing.assert_allclose(np.asarray(r.p), p_ref, atol=1e-12)
        np.testing.assert_allclose(np.asarray(r.Hp), H @ np.asarray(r.p), atol=1e-10)
        if case == "interior":       # CG stopped on its forcing tolerance, inside the ball
            assert np.linalg.norm(p_ref) < radius and int(r.n_hvp) >= 1
        else:
            np.testing.assert_allclose(np.linalg.norm(np.asarray(r.p)), radius, rtol=1e-12)
        if case == "negative_curvature":
            assert int(r.n_hvp) == 1                # left on the first direction


def test_steihaug_cg_cap_and_zero_gradient(x64):
    """At the cap the current CG iterate is returned (one product: the Cauchy step along -g);
    a zero gradient spends no product. CONTROL: uncapped, the same problem takes more."""
    rng = np.random.default_rng(1)
    n = 20
    A = rng.normal(size=(n, n))
    H = A @ A.T + 1e-3 * np.eye(n)
    g = rng.normal(size=n)
    r = _ours(g, H, 1e6, max_iters=1)
    assert bool(r.capped) and not bool(r.hits_boundary) and int(r.n_hvp) == 1
    np.testing.assert_allclose(np.asarray(r.p), -(g @ g) / (g @ H @ g) * g, rtol=1e-12)
    assert int(_ours(g, H, 1e6).n_hvp) > 1
    z = _ours(np.zeros(n), H, 1.0)
    assert int(z.n_hvp) == 0 and not np.any(np.asarray(z.p))


def _quartic(n=4, seed=1):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    A = jnp.asarray(A @ A.T / n + 0.5 * np.eye(n))
    b = jnp.asarray(rng.normal(size=n))
    return lambda x: -0.5 * x @ A @ x - 0.25 * jnp.sum(x ** 4) + b @ x


@pytest.mark.parametrize("start", [1.5, 4.0, 10.0])
def test_a_dadvi_fit_is_scipys_trust_ncg(x64, start):
    """The whole fit, iterate for iterate, is SciPy's trust-ncg (defaults) on -F: the same final
    eta and the same number of accepted steps, from starts where SciPy rejects steps (the far
    ones) and where it does not. The Newton-decrement stop is off (rel_tol=0), so both stop on
    SciPy's |grad| < 1e-4."""
    n = 4
    lp = _quartic(n)
    key = jax.random.PRNGKey(3)
    x0 = jnp.full(n, start)
    res = dadvi(lp, x0, key, rel_tol=0.0, covariance="diag")
    assert STOP_REASONS[int(res.stop)] == "tolerance"
    g = _pattern(jax.random.split(key)[0], 5, n, x0.dtype, False)
    F = _objective(lp, g)
    phi = jax.jit(lambda e: -F(e))
    grad = jax.jit(jax.grad(phi))
    hvp = jax.jit(lambda e, v: jax.jvp(grad, (e,), (v,))[1])
    path = []
    e0 = np.concatenate([np.asarray(x0), np.zeros(n)])
    ref = so.minimize(lambda e: float(phi(e)), e0, jac=lambda e: np.asarray(grad(e)),
                      hessp=lambda e, v: np.asarray(hvp(e, v)), method="trust-ncg",
                      callback=lambda xk: path.append(np.array(xk)))
    assert ref.success
    accepted = sum(not np.array_equal(a, b) for a, b in zip(path, [e0] + path[:-1]))
    if start >= 4.0:
        assert ref.nit > accepted                    # the far starts exercise rejection
    assert int(res.n_iter) == accepted
    eta = np.concatenate([np.asarray(res.mean), np.asarray(res.log_sd)])
    np.testing.assert_allclose(eta, ref.x, atol=1e-10)
    np.testing.assert_allclose(float(res.elbo), -ref.fun, rtol=1e-12)
    # SciPy spends a product on each model value; the CG's carried H p does not
    assert 0 < int(res.n_hvp) < ref.nhev


def test_a_non_finite_proposal_is_rejected_and_the_radius_shrinks(x64):
    """On a half line, steps whose cloud leaves the support (F = -inf) are rejected: the point and
    F stay, the radius is quartered, and the fit still converges into Gamma(3, 1)'s bulk. Every
    rejection here is such a step (checked against the recomputed proposal), and F never falls."""
    lp = lambda x: (jnp.where(x[0] > 0, 2 * jnp.log(jnp.where(x[0] > 0, x[0], 1.0)) - x[0],
                              -jnp.inf) - 0.5 * jnp.sum(x[1:] ** 2))
    n = 3
    g = _pattern(jax.random.PRNGKey(0), 5, n, jnp.float64, False)
    F = _objective(lp, g)
    vg, grad = jax.value_and_grad(F), jax.grad(F)
    opts = dadvi_options({})
    step = jax.jit(lambda c: _tr_step(vg, grad, c, max_iters=200, rel_tol=opts["rel_tol"],
                                      gtol=opts["gtol"], trust_eta=opts["trust_eta"],
                                      max_trust_radius=opts["max_trust_radius"],
                                      cg_max_iters=2 * n))
    def proposal(c):              # the step _tr_step proposes from c (its CG, recomputed)
        _, lin = jax.linearize(grad, c.eta)
        return c.eta + steihaug_cg(-c.g, lambda v: -lin(v), c.radius, 2 * n).p

    c = _init(vg, jnp.array([0.2, 0.0, 0.0]), 6, -3.0, radius=50.0)
    rejected = 0
    while int(c.stop) == 0:
        nxt = step(c)
        assert float(nxt.f) >= float(c.f)                         # ascent is monotone
        if int(nxt.k) == int(c.k) and int(nxt.stop) == 0:
            assert not np.isfinite(float(F(proposal(c))))          # rejected for leaving
            assert np.array_equal(np.asarray(nxt.eta), np.asarray(c.eta))
            assert float(nxt.radius) == 0.25 * float(c.radius) and float(nxt.f) == float(c.f)
            rejected += 1
        c = nxt
    assert rejected >= 2 and STOP_REASONS[int(c.stop)] == "tolerance"
    assert np.isfinite(float(c.f)) and 1.5 < float(c.eta[0]) < 4.0


def test_the_newton_decrement_stops_a_converged_fit():
    """In float32 a tight gtol is out of reach, but the interior Newton step's predicted gain falls
    below rel_tol |F|: the fit stops on "tolerance". CONTROL: with rel_tol = 0 it runs on to
    max_iters (or the radius floor), never converging."""
    lp = _quartic(6, seed=2)
    x0, key = jnp.full(6, 2.0), jax.random.PRNGKey(0)
    res = dadvi(lp, x0, key, gtol=0.0, covariance="diag")
    assert STOP_REASONS[int(res.stop)] == "tolerance" and int(res.n_iter) < 100
    ctl = dadvi(lp, x0, key, gtol=0.0, rel_tol=0.0, max_iters=300, covariance="diag")
    assert STOP_REASONS[int(ctl.stop)] in ("max_iters", "trust region")
    np.testing.assert_allclose(float(ctl.elbo), float(res.elbo), rtol=1e-5)


def test_a_capped_cg_still_converges_and_is_counted(x64):
    """cg_max_iters=1 makes every step a Cauchy step: the solves are counted as capped, the fit
    still converges (on gtol: a capped step never triggers the Newton-decrement stop), and every
    product is counted at 2 K gradients. CONTROL: uncapped, no solve is capped."""
    lp = _quartic(6, seed=3)
    x0, key = jnp.full(6, 1.0), jax.random.PRNGKey(1)
    capped = dadvi(lp, x0, key, cg_max_iters=1, covariance="diag", max_iters=5000)
    assert int(capped.n_cg_capped) > 0 and int(capped.n_hvp) >= int(capped.n_cg_capped)
    assert STOP_REASONS[int(capped.stop)] == "tolerance"
    full = dadvi(lp, x0, key, covariance="diag")
    assert int(full.n_cg_capped) == 0 and int(full.n_hvp) > 0
    np.testing.assert_allclose(float(capped.elbo), float(full.elbo), rtol=1e-8)
    for r in (capped, full):     # K gradients per F evaluation, 2 K per product
        assert (int(r.n_grad) - 10 * int(r.n_hvp)) % 5 == 0
        assert int(r.n_grad) - 10 * int(r.n_hvp) >= 5 * (2 * int(r.n_iter) + 1)


def test_the_secant_readout_is_available_under_trust_ncg(x64):
    """The trust-region steps feed the same secant history as L-BFGS's, so "lr_lbfgs" is drawn
    from (not the diagonal fallback), with finite log q."""
    lp = _quartic(5, seed=4)
    res = dadvi(lp, jnp.full(5, 3.0), jax.random.PRNGKey(2), covariance="lr_lbfgs")
    assert int(res.n_iter) >= 3 and READOUTS[int(res.readout)] == "lr_lbfgs"
    assert np.all(np.isfinite(np.asarray(res.log_q)))


def test_trust_ncg_options_are_validated():
    o = dadvi_options({})
    assert o["optimizer"] == "trust_ncg" and o["cg_max_iters"] is None
    assert dadvi_options({"optimizer": "lbfgs"})["optimizer"] == "lbfgs"
    for bad, match in (({"optimizer": "bfgs"}, "optimizer must be"),
                       ({"trust_eta": 0.3}, "trust_eta must be"),
                       ({"trust_eta": -0.1}, "trust_eta must be"),
                       ({"initial_trust_radius": 2000.0}, "trust radii"),
                       ({"initial_trust_radius": 0.0}, "trust radii"),
                       ({"gtol": -1.0}, "gtol must be"),
                       ({"gtol": float("nan")}, "gtol must be"),
                       ({"cg_max_iters": 0}, "cg_max_iters must be"),
                       ({"cg_max_iters": 2.5}, "cg_max_iters must be")):
        with pytest.raises(ValueError, match=match):
            dadvi_options(bad)


def test_the_cg_pair_readout_is_exact_on_a_gaussian(x64):
    """secant="cg": CG on -hess F at the fit gives exact curvature pairs; with all of them kept
    (cg_pairs >= 2n) the secant read-out is J^T (-hess F)^-1 J itself, so on a correlated
    Gaussian q = p up to log Z. Its products are counted (2 K each). CONTROL: the steps' secant
    read-out with the default history of 6 < 2n is visibly off."""
    n = 5
    rng = np.random.default_rng(1)
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    C = Q @ np.diag(np.exp(np.linspace(-2.0, 1.5, n))) @ Q.T
    m, P = rng.normal(size=n), jnp.asarray(np.linalg.inv(C))
    lp = lambda x: -0.5 * (x - m) @ P @ (x - m)
    key = jax.random.PRNGKey(2)
    tight = dict(covariance="lr_lbfgs", draws=20000, gtol=0.0, rel_tol=1e-15)
    cg = dadvi(lp, jnp.zeros(n), key, secant="cg", cg_pairs=200, **tight)
    steps = dadvi(lp, jnp.zeros(n), key, secant="steps", **tight)
    default = dadvi(lp, jnp.zeros(n), key, cg_pairs=200, **tight)     # "cg" under trust_ncg
    assert np.array_equal(np.asarray(default.draws), np.asarray(cg.draws))
    assert READOUTS[int(cg.readout)] == "lr_lbfgs"
    w = np.asarray(cg.log_p) - np.asarray(cg.log_q)
    assert np.std(w) < 1e-8
    np.testing.assert_allclose(np.cov(np.asarray(cg.draws).T), C, atol=0.03 * np.abs(C).max())
    assert np.std(np.asarray(steps.log_p) - np.asarray(steps.log_q)) > 0.5         # control
    pairs = int(cg.n_hvp) - int(steps.n_hvp)
    assert pairs == 2 * n                       # CG ends at 2n, exact, on 2n coordinates
    assert int(cg.n_grad) - int(steps.n_grad) == 2 * 5 * pairs
    for bad, match in (({"secant": "lanczos"}, "secant must be"), ({"cg_pairs": 0}, "cg_pairs")):
        with pytest.raises(ValueError, match=match):
            dadvi_options(bad)
