"""Matrix operations for a diagonal-plus-low-rank symmetric matrix ``A = diag(D) + V^T V``.

``D`` is the diagonal (a ``(d,)`` array) and ``V`` a ``(q, d)`` array whose rows are the
rank-one update vectors, so ``V^T V = sum_j v_j v_j^T`` is a rank-``q`` positive-semidefinite
update. :class:`~mimcs.hmc.LowRankQuadraticKinetic` uses these to apply a rank-``J`` mass and its
inverse / square root without forming the ``d x d`` matrix: energy and velocity need ``A^{-1} u``
(:func:`apply_inv`), and momentum refresh needs ``S u`` with ``S S^T = A`` (:func:`apply_chol`).

The inverse is the Sherman--Morrison--Woodbury recursion over the ``q`` rank-one terms; the
square root is a product of symmetric rank-one factors ``I + a s s^T`` in the ``D^{-1/2}``-whitened
space. Both loop over the (static) rank ``q``, so the routines are jit-friendly when ``q`` is a
Python int. All routines assume ``D > 0`` and that ``A`` stays positive definite (each added
rank-one term keeps it so).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array


def apply(D: Array, V: Array, u: Array) -> Array:
    """Matrix-vector product ``(diag(D) + V^T V) u``.

    Args:
        D: the ``(d,)`` diagonal component.
        V: the ``(q, d)`` low-rank component (rows are the update vectors).
        u: the ``(d,)`` vector to multiply.

    Returns:
        ``A u``.
    """
    return D * u + jnp.matmul(V.T, jnp.matmul(V, u))


def log_det(D: Array, V: Array) -> Array:
    """``log det(diag(D) + V^T V)`` via the matrix-determinant lemma (``V`` is ``(q, d)``).

    ``det(diag(D) + V^T V) = det(D) * det(I_q + V diag(1/D) V^T)``, and the capacitance matrix
    ``I_q + V diag(1/D) V^T`` is only ``q x q`` (``q`` the rank), so this is ``O(q^2 d)`` and needs
    no ``d x d`` factorization. Correct for any ``V`` (does not assume orthonormal rows)."""
    W = V / D                                          # (q, d): row r = v_r / D
    cap = jnp.eye(V.shape[0]) + jnp.matmul(W, V.T)     # I_q + V diag(1/D) V^T
    _, logdet_cap = jnp.linalg.slogdet(cap)
    return jnp.sum(jnp.log(D)) + logdet_cap


def inv_factors(D: Array, V: Array):
    """Sherman--Morrison factors for the inverse: the ``(q, d)`` vectors ``t`` and ``(q,)``
    scalars ``beta`` with ``t[r] = A_{r-1}^{-1} v_r`` and ``beta[r] = 1 / (1 + v_r^T t[r])``
    (``A_r = diag(D) + sum_{i<=r} v_i v_i^T``), so ``A^{-1} = D^{-1} - sum_r beta_r t_r t_r^T``.

    Public because they depend only on ``(D, V)``: where those are constant for a whole
    trajectory the factors should be computed **once** and handed to :func:`apply_inv_factored`,
    rather than rebuilt inside every ``apply_inv``. The recursion is ``O(q^2 d)``, against
    ``O(q d)`` for the application itself, so it dominates when it is not hoisted."""
    d, q = D.shape[0], V.shape[0]
    t = jnp.zeros((q, d))
    t = t.at[0].set(V[0] / D)
    beta = jnp.zeros(q)
    beta = beta.at[0].set(1 / (1 + jnp.sum(t[0] * V[0])))
    for r in range(1, q):
        tt = (V[r] / D
              - jnp.sum(beta[:r] * jnp.sum(t[:r] * V[r], axis=1) * t[:r].T, axis=1))
        t = t.at[r].set(tt)
        beta = beta.at[r].set(1 / (1 + jnp.sum(t[r] * V[r])))
    return beta, t


def apply_inv(D: Array, V: Array, u: Array) -> Array:
    """Inverse matrix-vector product ``(diag(D) + V^T V)^{-1} u``.

    Args:
        D: the ``(d,)`` diagonal component.
        V: the ``(q, d)`` low-rank component (rows are the update vectors).
        u: the ``(d,)`` vector to solve against.

    Returns:
        ``A^{-1} u``, via the Sherman--Morrison--Woodbury recursion.
    """
    return apply_inv_factored(D, *inv_factors(D, V), u)


def apply_inv_factored(D: Array, beta: Array, t: Array, u: Array) -> Array:
    """``A^{-1} u`` from factors :func:`inv_factors` already produced --- the ``O(q d)`` half."""
    return u / D - jnp.sum(beta * jnp.sum(t * u, axis=1) * t.T, axis=1)


def _apply_factor(a: Array, s: Array, g: Array) -> Array:
    """Apply the symmetric rank-one factor ``(I + a s s^T) g``."""
    return g + a * jnp.sum(s * g) * s


def _apply_inv_factor(a: Array, s: Array, g: Array) -> Array:
    """Apply the inverse factor ``(I + a s s^T)^{-1} g`` (Sherman--Morrison of one factor)."""
    return g - a / (1 + a * jnp.sum(jnp.square(s))) * jnp.sum(s * g) * s


def _alpha(s_nsq: Array) -> Array:
    """Coefficient making ``I + a s s^T`` a square root of ``I + s s^T`` (``s_nsq = ||s||^2``).

    ``a`` is the positive root of ``a^2 ||s||^2 + 2 a - 1 = 0``; a Taylor branch is used for
    ``||s||^2 -> 0`` to stay numerically stable."""
    return jnp.where(s_nsq > 1e-8,
                     (jnp.sqrt(1 + s_nsq) - 1) / s_nsq,
                     0.5 - s_nsq / 8)


def _compute_alpha_s(D: Array, V: Array):
    """Square-root factors: the ``(q, d)`` whitened vectors ``s`` and ``(q,)`` coefficients
    ``alpha`` whose product ``prod_r (I + alpha_r s_r s_r^T)`` is a symmetric square root of
    ``I + sum_r (v_r / sqrt(D))(v_r / sqrt(D))^T`` in the ``D^{-1/2}``-whitened space."""
    d, q = D.shape[0], V.shape[0]
    s = jnp.zeros((q, d))
    s = s.at[0].set(V[0] / jnp.sqrt(D))
    alpha = jnp.zeros(q)
    alpha = alpha.at[0].set(_alpha(jnp.sum(jnp.square(s[0]))))
    for r in range(1, q):
        ss = jax.lax.fori_loop(0, r, lambda i, g: _apply_inv_factor(alpha[i], s[i], g),
                               V[r] / jnp.sqrt(D))
        s = s.at[r].set(ss)
        alpha = alpha.at[r].set(_alpha(jnp.sum(jnp.square(ss))))
    return alpha, s


def apply_chol(D: Array, V: Array, u: Array) -> Array:
    """Square-root product ``S u`` with ``S S^T = diag(D) + V^T V``.

    ``S = D^{1/2} prod_r (I + alpha_r s_r s_r^T)`` is a *symmetric* square-root factor (not
    triangular, despite the name), so ``S z`` for ``z ~ N(0, I)`` has covariance ``A``.

    Args:
        D: the ``(d,)`` diagonal component.
        V: the ``(q, d)`` low-rank component (rows are the update vectors).
        u: the ``(d,)`` vector to multiply.

    Returns:
        ``S u``.
    """
    q = V.shape[0]
    alpha, s = _compute_alpha_s(D, V)
    g = jax.lax.fori_loop(0, q, lambda i, g: _apply_factor(alpha[q - 1 - i], s[q - 1 - i], g), u)
    return jnp.sqrt(D) * g


def apply_chol_invT(D: Array, V: Array, u: Array) -> Array:
    """Inverse-transpose square-root product ``S^{-T} u`` (``S S^T = diag(D) + V^T V``).

    Args:
        D: the ``(d,)`` diagonal component.
        V: the ``(q, d)`` low-rank component (rows are the update vectors).
        u: the ``(d,)`` vector to multiply.

    Returns:
        ``S^{-T} u``.
    """
    q = V.shape[0]
    alpha, s = _compute_alpha_s(D, V)
    g = jax.lax.fori_loop(0, q, lambda i, g: _apply_inv_factor(alpha[q - 1 - i], s[q - 1 - i], g), u)
    return g / jnp.sqrt(D)


def apply_chol_inv(D: Array, V: Array, u: Array) -> Array:
    """Inverse square-root product ``S^{-1} u`` (``S S^T = diag(D) + V^T V``).

    Args:
        D: the ``(d,)`` diagonal component.
        V: the ``(q, d)`` low-rank component (rows are the update vectors).
        u: the ``(d,)`` vector to multiply.

    Returns:
        ``S^{-1} u``.
    """
    q = V.shape[0]
    alpha, s = _compute_alpha_s(D, V)
    g = jax.lax.fori_loop(0, q, lambda i, g: _apply_inv_factor(alpha[i], s[i], g),
                          u / jnp.sqrt(D))
    return g
