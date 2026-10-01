"""Tests for :mod:`mimcs.hmc.spectral`: matrix functions of a symmetric matrix whose derivatives
survive repeated eigenvalues, and the eigenvalue clamps.

The derivative check runs in x64 against central finite differences of the primal (always valid).
The control is naive autodiff through ``eigh``, which gives NaN at the identity and --- worse --- a
*finite, wrong* derivative on an indefinite matrix with a repeated eigenvalue; without it the
custom JVP would not be needed and the test would be vacuous.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from mimcs import config
from mimcs.hmc.spectral import CLAMPS, sym_matfun, sym_tracefun, resolve_clamp


@pytest.fixture
def x64():
    was = config.x64_enabled()
    config.enable_x64(True)
    try:
        yield
    finally:
        config.enable_x64(was)


def _naive_matfun(H, g):
    lam, Q = jnp.linalg.eigh(0.5 * (H + H.T))
    return (Q * jax.vmap(g)(lam)) @ Q.T


def _matrices(rng):
    A = rng.normal(size=(4, 4))
    Q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    return {"random": A + A.T,
            "near-degenerate": Q @ np.diag([1.0, 1.0 + 1e-9, -0.5, 2.0]) @ Q.T,
            "identity": np.eye(4),
            "indefinite-degenerate": Q @ np.diag([-1.0, -1.0, 0.3, 0.3]) @ Q.T}


@pytest.mark.parametrize("clamp", sorted(CLAMPS))
def test_derivatives_match_finite_differences(x64, clamp):
    rng = np.random.default_rng(0)
    c, b = CLAMPS[clamp], 1.7
    W = jnp.asarray(rng.normal(size=(4, 4)))
    fm = lambda H: jnp.sum(W * sym_matfun(H, lambda x: c.inv_phi(b * x)))
    ft = lambda H: sym_tracefun(H, lambda x: c.log_phi(b * x))
    for label, H in _matrices(rng).items():
        H = jnp.asarray(H)
        E = jnp.asarray(rng.normal(size=(4, 4)))
        E = 0.5 * (E + E.T)
        for f in (fm, ft):
            d = float(jnp.sum(jax.grad(f)(H) * E))
            fd = float((f(H + 1e-5 * E) - f(H - 1e-5 * E)) / 2e-5)
            assert d == pytest.approx(fd, abs=1e-7), (clamp, label)


def test_naive_eigh_autodiff_fails_where_the_custom_rule_does_not(x64):
    """The control: at a repeated eigenvalue naive autodiff is NaN (identity) or silently wrong
    (indefinite, degenerate), while the Daleckii--Krein rule matches finite differences."""
    rng = np.random.default_rng(0)
    c, b = CLAMPS["softplus"], 1.7
    g = lambda x: c.inv_phi(b * x)
    W = jnp.asarray(rng.normal(size=(4, 4)))
    mats = _matrices(rng)
    E = jnp.asarray(rng.normal(size=(4, 4)))
    E = 0.5 * (E + E.T)
    naive = lambda H: jnp.sum(W * _naive_matfun(H, g))
    custom = lambda H: jnp.sum(W * sym_matfun(H, g))
    assert not np.isfinite(float(jnp.sum(jax.grad(naive)(jnp.eye(4)) * E)))
    H = jnp.asarray(mats["indefinite-degenerate"])
    fd = float((custom(H + 1e-5 * E) - custom(H - 1e-5 * E)) / 2e-5)
    d_naive = float(jnp.sum(jax.grad(naive)(H) * E))
    d_custom = float(jnp.sum(jax.grad(custom)(H) * E))
    assert abs(d_naive - fd) > 1e-3            # finite but wrong
    assert d_custom == pytest.approx(fd, abs=1e-7)


def test_clamps_are_positive_and_stable_in_float32():
    """``log phi`` is finite across 80 orders of magnitude of curvature in float32 --- softplus's
    ``phi`` itself underflows to exactly 0 below ``x ~ -88``, which would make the metric singular."""
    x = jnp.asarray([-1e4, -200.0, -30.0, -1.0, -1e-6, 0.0, 1e-6, 1.0, 30.0, 1e4], jnp.float32)
    for c in CLAMPS.values():
        lp = c.log_phi(x)
        assert bool(jnp.all(jnp.isfinite(lp))), c.name
        d = jax.vmap(jax.grad(c.log_phi))(x)
        assert bool(jnp.all(jnp.isfinite(d))), c.name
    # large positive curvature passes through: phi(x) ~ x (softplus) and |x| (softabs)
    for c in CLAMPS.values():
        assert float(jnp.exp(c.log_phi(jnp.asarray(50.0)))) == pytest.approx(50.0, rel=1e-6)
    # the distinguishing behaviour: a negative curvature keeps ~|lambda| under softabs, ~0 under
    # softplus
    assert float(jnp.exp(CLAMPS["softabs"].log_phi(jnp.asarray(-50.0)))) == pytest.approx(50.0)
    assert float(CLAMPS["softplus"].log_phi(jnp.asarray(-50.0))) == pytest.approx(-50.0)


def test_resolve_clamp():
    assert resolve_clamp("softabs") is CLAMPS["softabs"]
    with pytest.raises(ValueError, match="unknown clamp"):
        resolve_clamp("relu")
