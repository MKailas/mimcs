"""Lane helpers shared by the samplers that treat each temperature as its own chain.

A *lane* is one temperature's block of the product coordinate: lane ``k`` occupies
``[k*n, (k+1)*n)`` of the ``(K*n,)`` vector (``mimcs/pt/tempering.py``). Two samplers work in
these terms --- :class:`~mimcs.pt.hmc.IndependentAcceptanceMixin`, which accepts per lane, and
:class:`~mimcs.pt.nuts.PerTemperatureNUTSMixin`, which selects per lane --- so the reshapes and
the per-lane energy live here rather than being written twice.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def spread(v: Array, n: int) -> Array:
    """A per-lane vector ``(K,)`` broadcast to the product coordinate ``(K*n,)``."""
    return jnp.repeat(v, n)


def lanes(x: Array, n_temperatures: int, n: int) -> Array:
    """A product vector ``(K*n,)`` viewed as ``(K, n)``, one row per temperature."""
    return x.reshape(n_temperatures, n)


class LaneStateMixin:
    """Per-lane energy and step size over the product coordinate."""

    def _eps(self, step_size: Array) -> Array:
        """Expand a per-temperature step size to the product coordinate.

        A scalar is passed through (the integrator's kick and drift are elementwise, so it
        applies to every lane); a ``(K,)`` vector is repeated to ``(K*n,)``, which is uniform
        within a lane and so satisfies the precondition
        :meth:`~mimcs.pt.kinetics.ProductKinetic._lane_eps` documents.
        """
        if jnp.ndim(step_size) == 0:
            return step_size
        return jnp.repeat(step_size, self.model.base.coord_dim)

    def per_temperature_energy(self, istate, ctx) -> Array:
        """``H_k`` for every temperature --- the potentials' and kinetics' own per-lane terms.

        Note this recomputes the potential terms from ``istate.q`` rather than reading
        ``istate.potential_values``, which is a *sum* over lanes and so cannot be split. That is
        what lets a lane-mixed state carry a stale scalar cache without corrupting any energy.
        """
        total = jnp.zeros((self.model.n_temperatures,))
        for p in self.potentials:
            total = total + p.per_temperature_values(istate.q, ctx)
        for k in self.kinetics:
            total = total + k.per_temperature_energy(istate, ctx)
        return total
