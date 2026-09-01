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


def lane_discrete(ctx, n_temperatures: int):
    """The per-lane view of the context's discrete block --- ``(K, n)``, or ``None``.

    A model's integer parameters are a trajectory constant that each *rung* holds its own copy of,
    so a lane must see its own row and not the whole product block. ``None`` (a continuous model)
    is an empty pytree, which ``jax.vmap`` maps trivially --- the same property
    :meth:`~mimcs.pt.kinetics.ProductKinetic._lanes` relies on for its cache, and what keeps the
    graph of a continuous tempered run exactly what it was.
    """
    z = getattr(ctx, "discrete", None)
    if z is None or z.shape[0] == 0:
        return None
    return z.reshape(n_temperatures, -1)


def per_temperature_potential(potentials, q: Array, ctx, n_temperatures: int) -> Array:
    """``V_k`` for every temperature --- the potentials' own per-lane terms, ``(K,)``.

    The potential half of a per-lane Hamiltonian, and a free function rather than only a method
    because two unrelated places need it: :meth:`LaneStateMixin.per_temperature_energy`, and the
    discrete Gibbs sweep under tempering, whose host
    (:class:`~mimcs.pt.ReplicaExchangeMixin`) cannot assume ``LaneStateMixin`` is in its MRO ---
    joint selection composes without it.

    A move that does not touch the momentum needs exactly this and nothing else: a
    Metropolis-within-Gibbs sweep proposes at fixed ``p``, so its acceptance ratio is a difference
    of ``V_k``, not of ``H_k`` (doc 14).
    """
    total = jnp.zeros((n_temperatures,))
    for p in potentials:
        total = total + p.per_temperature_values(q, ctx)
    return total


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

    def per_temperature_potential(self, q: Array, ctx) -> Array:
        """This sampler's potentials, per lane --- see the module-level function of the same name.

        Note this recomputes from ``q`` rather than reading ``istate.potential_values``, which is
        a *sum* over lanes and so cannot be split.
        """
        return per_temperature_potential(self.potentials, q, ctx, self.model.n_temperatures)

    def per_temperature_energy(self, istate, ctx) -> Array:
        """``H_k`` for every temperature --- the potentials' and kinetics' own per-lane terms.

        Note this recomputes the potential terms from ``istate.q`` rather than reading
        ``istate.potential_values``, which is a *sum* over lanes and so cannot be split. That is
        what lets a lane-mixed state carry a stale scalar cache without corrupting any energy.
        """
        total = self.per_temperature_potential(istate.q, ctx)
        for k in self.kinetics:
            total = total + k.per_temperature_energy(istate, ctx)
        return total
