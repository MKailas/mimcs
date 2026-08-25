"""Rank-one update (and downdate) of a lower-triangular Cholesky factor.

Given ``L`` with ``A = L L^T`` and a vector ``v``, :func:`chol_update` returns the Cholesky
factor of ``A +/- v v^T`` in ``O(d^2)`` --- far cheaper than refactorising ``A`` from
scratch (``O(d^3)``). The dense mass-matrix adaptation uses this to keep the inverse-mass
Cholesky current under the stochastic-approximation covariance update without recomputing
it each warmup step.

The update is the standard column-sweep algorithm, written with ``jax.lax.scan`` over the
rows of ``L^T`` so it is jit-friendly.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array


def _sweep(carry, row):
    """One sweep: fold the residual vector into the next column of the factor.

    ``row`` is a row of ``L^T`` (a column of ``L``); ``alpha``/``beta`` carry the running
    scale across columns, ``r`` the column index, ``residual`` the (shrinking) update vector.
    """
    r, residual, alpha, beta, sign = carry
    below = jnp.arange(row.shape[0]) > r

    a = residual[r] / row[r]
    alpha = (alpha[1], alpha[1] + sign * jnp.square(a))
    beta = (beta[1], jnp.sqrt(alpha[1]))

    residual = jnp.where(below, residual - a * row, residual)
    row = row * (beta[1] / beta[0])
    row = jnp.where(below, row + sign * a / (beta[1] * beta[0]) * residual, row)

    return (r + 1, residual, alpha, beta, sign), row


def chol_update(L: Array, v: Array, downdate: bool = False) -> Array:
    """Cholesky factor of ``L L^T +/- v v^T`` by a rank-one update.

    Args:
        L: lower-triangular Cholesky factor.
        v: the rank-one update vector.
        downdate: subtract ``v v^T`` instead of adding it (the result must remain
            positive definite).

    Returns:
        The lower-triangular Cholesky factor of ``L L^T +/- v v^T``.
    """
    sign = jnp.where(downdate, -1.0, 1.0)
    init = (0, v, (0.0, 1.0), (0.0, 1.0), sign)
    _, updated_rows = jax.lax.scan(_sweep, init, L.T)
    return updated_rows.T
