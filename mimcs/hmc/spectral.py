"""Spectral functions of a symmetric matrix, with derivatives that survive repeated eigenvalues.

The Hessian metric of :mod:`mimcs.hmc.riemannian` is ``G = Q f(Lambda) Q^T`` for the eigen-
decomposition ``H = Q Lambda Q^T`` of a block Hessian and a positive *clamp* ``f``. Its kinetic
energy is differentiated through ``H`` --- that derivative *is* the metric-derivative kick --- and
naive autodiff through ``jnp.linalg.eigh`` cannot do it: the eigenvector derivative divides by
``lambda_j - lambda_i``, which is ``inf`` for a repeated eigenvalue, and the ``inf * 0`` that
follows turns the whole kick into NaN. Repeated eigenvalues are not exotic here: a symmetric start,
or a block of exchangeable parameters, gives them exactly.

A smooth *function* of ``H`` has a perfectly well-defined derivative even then. The two functions
below carry it as a ``custom_jvp`` (Daleckii--Krein):

* :func:`sym_matfun` --- ``g(H) = Q diag(g(lambda)) Q^T``, with
  ``d g(H) = Q (Gamma o (Q^T dH Q)) Q^T``, where ``Gamma_ij = (g(l_i) - g(l_j)) / (l_i - l_j)`` is
  the first divided difference and ``Gamma_ii = g'(l_i)``. When two eigenvalues are closer than a
  scale-relative ``delta`` the difference quotient cancels catastrophically, so ``g'`` at their
  midpoint is used instead --- an ``O(delta^2)`` error against the ``O(eps / delta)`` it replaces.
* :func:`sym_tracefun` --- ``sum_i h(lambda_i)``, with ``d = sum_i h'(l_i) (Q^T dH Q)_ii`` (only the
  eigenvalue derivative, which never divides).

Both take the scalar function (and nothing traced besides the matrix) as a static argument, so a
clamp's scale ``b`` enters as ``g(b H)`` *outside* them and is differentiated normally.

The clamps themselves are also here --- :data:`CLAMPS` maps a name to a :class:`Clamp` holding
``log phi`` for the unit-scale clamp ``phi``; the metric is ``f(lambda) = phi(b lambda) / b``.
Everything is written in terms of ``log phi`` because the kinetic energy needs ``1 / f`` and
``log f``, and ``phi`` itself underflows for a strongly negative argument under softplus.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array


def _elementwise_grad(fn: Callable) -> Callable:
    """``fn'`` applied elementwise, for a scalar ``fn`` written with safe ``where`` branches."""
    return jax.vmap(jax.grad(fn))


def _near_tolerance(lam: Array) -> Array:
    """The gap below which two eigenvalues count as equal for the divided difference.

    ``eps^(1/3)`` balances the two errors: the difference quotient loses ``~eps * |g| / delta`` to
    cancellation, the midpoint derivative is off by ``~delta^2 * |g'''|``. Scaled by the spectrum's
    size so the rule is unit-free."""
    eps = jnp.finfo(lam.dtype).eps
    return eps ** (1.0 / 3.0) * (1.0 + jnp.max(jnp.abs(lam)))


def divided_differences(lam: Array, g: Callable) -> Array:
    """``Gamma_ij = (g(l_i) - g(l_j)) / (l_i - l_j)``, with ``g'`` at the midpoint for close pairs.

    Both branches are computed on safe inputs (the ``where`` guards the *denominator*, not just the
    result), so neither produces an ``inf`` or ``NaN`` that could leak through a later gradient."""
    gl = jax.vmap(g)(lam)
    dg = _elementwise_grad(g)
    diff = lam[:, None] - lam[None, :]
    close = jnp.abs(diff) <= _near_tolerance(lam)
    safe = jnp.where(close, 1.0, diff)
    quotient = (gl[:, None] - gl[None, :]) / safe
    mid = 0.5 * (lam[:, None] + lam[None, :])
    derivative = dg(mid.reshape(-1)).reshape(mid.shape)
    return jnp.where(close, derivative, quotient)


def _symmetrize(H: Array) -> Array:
    return 0.5 * (H + H.T)


@partial(jax.custom_jvp, nondiff_argnums=(1,))
def sym_matfun(H: Array, g: Callable) -> Array:
    """``g(H) = Q diag(g(lambda)) Q^T`` for symmetric ``H`` (see the module docstring)."""
    lam, Q = jnp.linalg.eigh(_symmetrize(H))
    return (Q * jax.vmap(g)(lam)) @ Q.T


@sym_matfun.defjvp
def _sym_matfun_jvp(g, primals, tangents):
    (H,), (dH,) = primals, tangents
    lam, Q = jnp.linalg.eigh(_symmetrize(H))
    out = (Q * jax.vmap(g)(lam)) @ Q.T
    Ht = Q.T @ _symmetrize(dH) @ Q
    dout = Q @ (divided_differences(lam, g) * Ht) @ Q.T
    return out, dout


@partial(jax.custom_jvp, nondiff_argnums=(1,))
def sym_tracefun(H: Array, h: Callable) -> Array:
    """``sum_i h(lambda_i)`` for symmetric ``H`` (see the module docstring)."""
    return jnp.sum(jax.vmap(h)(jnp.linalg.eigvalsh(_symmetrize(H))))


@sym_tracefun.defjvp
def _sym_tracefun_jvp(h, primals, tangents):
    (H,), (dH,) = primals, tangents
    lam, Q = jnp.linalg.eigh(_symmetrize(H))
    out = jnp.sum(jax.vmap(h)(lam))
    diag = jnp.sum(Q * (_symmetrize(dH) @ Q), axis=0)          # (Q^T dH Q)_ii
    return out, jnp.sum(_elementwise_grad(h)(lam) * diag)


# --- clamps --------------------------------------------------------------------------------- #

def _log_softplus(x: Array) -> Array:
    """``log(log(1 + e^x))``, stable at both ends.

    Below ``x = -20`` softplus is ``e^x (1 - e^x / 2 + ...)``, so its log is ``x - e^x / 2``;
    evaluating ``log(softplus(x))`` there instead would return ``log 0 = -inf`` once ``e^x``
    underflows (``x < ~-88`` in float32), and the metric would be exactly singular."""
    lo = x < -20.0
    x_lo = jnp.where(lo, x, -20.0)
    x_hi = jnp.where(lo, 0.0, x)
    return jnp.where(lo, x_lo - 0.5 * jnp.exp(x_lo), jnp.log(jax.nn.softplus(x_hi)))


def _log_softabs(x: Array) -> Array:
    """``log(x coth x)``, stable at ``0`` (where it is ``x^2 / 3``) and for large ``|x|``
    (where it is ``log|x|``)."""
    ax = jnp.abs(x)
    small = ax < 1e-2
    big = ax > 20.0
    x_mid = jnp.where(small | big, 1.0, ax)
    x_small = jnp.where(small, ax, 0.0)
    x_big = jnp.where(big, ax, 1.0)
    mid = jnp.log(x_mid / jnp.tanh(x_mid))
    return jnp.where(small, x_small ** 2 / 3.0 - x_small ** 4 / 90.0,
                     jnp.where(big, jnp.log(x_big), mid))


@dataclass(frozen=True)
class Clamp:
    """A positive unit-scale function ``phi`` of an eigenvalue, via ``log phi``.

    The metric eigenvalue for curvature ``lambda`` at softness ``1/b`` is ``phi(b lambda) / b``:
    ``b lambda >> 1`` leaves a positive curvature essentially unchanged, ``b |lambda| <~ 1`` is the
    band the clamp reshapes, and ``1/b`` is the floor a flat direction gets (up to a constant)."""

    name: str
    log_phi: Callable[[Array], Array]

    def inv_phi(self, x: Array) -> Array:
        return jnp.exp(-self.log_phi(x))

    def sqrt_phi(self, x: Array) -> Array:
        return jnp.exp(0.5 * self.log_phi(x))


#: ``softplus``: ``phi(x) = log(1 + e^x)``. A negative curvature ``-|lambda|`` keeps mass
#: ``~ e^{-b|lambda|} / b`` --- positive but small, so such directions move fast.
#: ``softabs``: ``phi(x) = x coth x`` (Betancourt 2013). A negative curvature keeps mass ``~|lambda|``.
CLAMPS = {"softplus": Clamp("softplus", _log_softplus),
          "softabs": Clamp("softabs", _log_softabs)}


def resolve_clamp(clamp) -> Clamp:
    if isinstance(clamp, Clamp):
        return clamp
    if clamp not in CLAMPS:
        raise ValueError(f"unknown clamp {clamp!r} (use one of {sorted(CLAMPS)} or a Clamp)")
    return CLAMPS[clamp]
