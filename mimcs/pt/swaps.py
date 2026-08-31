"""Replica exchange: the move that makes tempering worth doing.

Implements the swap section of ``docs/design/13_parallel_tempering.md``. Without swaps the K
chains are K independent runs and the hot ones are wasted work; the swap is what carries a hot
chain's mobility down to the cold chain whose draws are kept.

Only **adjacent** pairs are attempted --- the acceptance ratio decays with the temperature gap, so
non-adjacent swaps almost never accept and cost the same to propose. A sweep alternates between
the even pairs ``(0,1), (2,3), ...`` and the odd pairs ``(1,2), (3,4), ...``: within one sweep the
pairs touch disjoint replicas, so every pair of a sweep is attempted **at once**, and alternating
gives every neighbour a turn.

For the pair ``(k, k+1)``::

    log alpha = (beta_k - beta_{k+1}) * (L(x_{k+1}) - L(x_k))

with ``L`` the sum of the *untempered* tempered-components. Exchanging two states under the
product target is a symmetric proposal, so this ratio is the whole acceptance probability.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from .._logging import get_logger

log = get_logger(__name__)


def swap_log_ratios(untempered: Array, betas: Array, offset: Array) -> Array:
    """``log alpha`` for every adjacent pair, as a ``(K,)`` vector indexed by the pair's lower rung.

    Entries that are not the lower member of an attempted pair are ``-inf`` (never accepted);
    ``offset`` (0 or 1) selects the even or odd pairs.
    """
    K = betas.shape[0]
    lower = jnp.arange(K)
    upper = lower + 1
    in_range = upper < K
    attempted = in_range & (((lower - offset) % 2) == 0)

    hi = jnp.clip(upper, 0, K - 1)
    d_beta = betas - betas[hi]                      # beta_k - beta_{k+1} >= 0
    d_L = untempered[hi] - untempered              # L(x_{k+1}) - L(x_k)
    return jnp.where(attempted, d_beta * d_L, -jnp.inf)


def all_pair_log_ratios(untempered: Array, betas: Array) -> Array:
    """``log alpha`` for *every* adjacent pair, ignoring the even/odd parity --- shape ``(K,)``.

    Ladder adaptation wants an acceptance signal for each pair on every iteration, and the ratio
    costs nothing to evaluate for a pair that is not being swapped this sweep. The last entry is
    padding (there is no pair above the top rung).
    """
    K = betas.shape[0]
    hi = jnp.clip(jnp.arange(K) + 1, 0, K - 1)
    return jnp.where(jnp.arange(K) + 1 < K, (betas - betas[hi]) * (untempered[hi] - untempered),
                     -jnp.inf)


def apply_swaps(per_temperature: Array, accepted_lower: Array) -> Array:
    """Exchange rows of a ``(K, ...)`` array for every accepted adjacent pair.

    ``accepted_lower[k]`` is True when the pair ``(k, k+1)`` swaps. Because a sweep's pairs are
    disjoint, the whole sweep is one gather: build the permutation, then apply it.
    """
    K = accepted_lower.shape[0]
    idx = jnp.arange(K)
    # k swaps up when its own pair accepted; k swaps down when the pair below it accepted.
    up = accepted_lower
    down = jnp.concatenate([jnp.zeros((1,), bool), accepted_lower[:-1]])
    perm = jnp.where(up, idx + 1, jnp.where(down, idx - 1, idx))
    return per_temperature[perm]


def swap_step(coordinate: Array, untempered: Array, betas: Array, uniforms: Array,
              offset: Array, n_temperatures: int, coord_dim: int, discrete: Array | None = None,
              discrete_dim: int = 0):
    """One even/odd sweep of adjacent replica exchanges.

    Returns ``(new_coordinate, new_discrete, accepted_lower, attempted_lower)``. The caller
    re-seeds the potential caches afterwards: a swapped state's cached values were computed at the
    *other* temperature's beta and no longer describe it.

    **A replica is its whole state, labels included.** When a model has discrete parameters, the
    integer block must move under the *same* permutation as the coordinate --- exchanging
    positions while leaving the labels behind would hand each rung a configuration that was drawn
    from no target at all, and nothing downstream would show it. :func:`apply_swaps` is generic
    over the trailing shape precisely so both blocks go through one permutation.
    """
    log_alpha = swap_log_ratios(untempered, betas, offset)
    attempted = jnp.isfinite(log_alpha)
    accepted = attempted & (jnp.log(uniforms) < jnp.minimum(0.0, log_alpha))

    per = coordinate.reshape(n_temperatures, coord_dim)
    new_coord = apply_swaps(per, accepted).reshape(-1)

    new_discrete = discrete
    if discrete is not None and discrete_dim:
        per_z = discrete.reshape(n_temperatures, discrete_dim)
        new_discrete = apply_swaps(per_z, accepted).reshape(-1)
    return new_coord, new_discrete, accepted, attempted
