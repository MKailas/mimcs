"""Stick breaking: the bijection ``R^(K-1) -> interior of the simplex on K parts``.

Two parameter types are built on this, because two constrained objects turn out to be the same
object. A :class:`~mimcs.model.SimplexParameter` *is* a vector of ``K`` parts summing to one. A
doubly-bounded :class:`~mimcs.model.OrderedParameter` ``L < x_1 < ... < x_d < U`` is one too, in
disguise: its ``d + 1`` **gaps** ``(x_1 - L, x_2 - x_1, ..., U - x_d)`` are positive and sum to
``U - L``, so it is a ``(d+1)``-part simplex scaled by the width. Both therefore have exactly
``K - 1`` free coordinates, and share this file.

The construction breaks a unit stick into ``K`` pieces, taking a fraction ``z_k`` of what is
left at each of ``K - 1`` steps::

    z_k = sigmoid(y_k + offset_k)
    p_k = z_k * (1 - sum_{i<k} p_i)          k = 1 .. K-1
    p_K = 1 - sum_{i<K} p_i

The remaining stick after ``k`` steps is ``prod_{i<=k} (1 - z_i)``, a *cumulative product*, so
the whole map vectorizes --- no sequential scan is needed, and working in logs (``log_sigmoid``
and a ``cumsum``) keeps it stable when a part is tiny.

``offsets`` places ``y = 0``. With :func:`default_offsets` it lands on the **uniform** point
``p_k = 1/K``, which is what makes the coordinate origin a well-behaved default for both types
(``Model.default_sample`` evaluates every chart at ``0``): the centre of the simplex, and evenly
spaced points for a doubly-bounded ordered vector.

The Jacobian is triangular (``p_k`` depends only on ``y_1..y_k``), so its log-determinant is the
sum of the diagonal, ``d p_k / d y_k = z_k (1 - z_k) * (remaining stick)``. All functions here
act on the **last axis** and broadcast over any leading batch axes.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array, nn


def default_offsets(n: int) -> Array:
    """Offsets putting ``y = 0`` at the uniform point of the ``(n+1)``-part simplex.

    Step ``k`` should take ``1 / (K - k + 1)`` of the remaining stick to leave every part equal,
    so ``offset_k = -log(n - k)`` with ``k`` counted from zero (the final step is a fair split,
    offset ``0``).
    """
    k = jnp.arange(n, dtype=float)
    return -jnp.log(n - k)


def _log_fractions(y: Array, offsets: Array) -> tuple[Array, Array]:
    """``(log z, log(1 - z))`` for each break, in logs throughout."""
    a = y + offsets
    return nn.log_sigmoid(a), nn.log_sigmoid(-a)


def _log_remaining(log_1mz: Array) -> Array:
    """``log`` of the stick left *before* each break: ``[0, cumsum(log(1-z))[:-1]]``."""
    csum = jnp.cumsum(log_1mz, axis=-1)
    return jnp.concatenate([jnp.zeros_like(csum[..., :1]), csum[..., :-1]], axis=-1)


def forward(y: Array, offsets: Array) -> Array:
    """``(..., K-1)`` coordinates -> ``(..., K)`` parts, positive and summing to one."""
    log_z, log_1mz = _log_fractions(y, offsets)
    log_remaining = _log_remaining(log_1mz)
    head = jnp.exp(log_remaining + log_z)
    last = jnp.exp(jnp.sum(log_1mz, axis=-1, keepdims=True))
    return jnp.concatenate([head, last], axis=-1)


def inverse(p: Array, offsets: Array) -> Array:
    """``(..., K)`` parts -> ``(..., K-1)`` coordinates.

    The stick left before break ``k`` is the *suffix sum* ``sum_{i>=k} p_i``, so both the
    fraction taken and the fraction left are ratios of suffix sums --- computed that way rather
    than as ``1 - prefix``, which cancels catastrophically once the prefix approaches one.
    """
    tail = jnp.cumsum(p[..., ::-1], axis=-1)[..., ::-1]      # tail_k = sum_{i>=k} p_i
    # logit(z_k) = log(p_k / tail_k) - log(tail_{k+1} / tail_k)
    a = jnp.log(p[..., :-1]) - jnp.log(tail[..., 1:])
    return a - offsets


def log_jacobian(y: Array, offsets: Array) -> Array:
    """``log|det d(p_1..p_{K-1}) / dy|``, shape ``y.shape[:-1]`` (the batch axes)."""
    log_z, log_1mz = _log_fractions(y, offsets)
    return jnp.sum(log_z + log_1mz + _log_remaining(log_1mz), axis=-1)
