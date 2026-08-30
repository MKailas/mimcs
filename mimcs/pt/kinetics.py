"""The kinetic energy over the product space: one block structure, applied per temperature.

Implements the kinetic section of ``docs/design/13_parallel_tempering.md``.

Tempering must not cost the sampler its kinetic machinery. Every temperature therefore uses the
**same block structure** --- whatever partition the factory chose for the model (doc 09), with
whatever per-block kinetics it chose: diagonal, dense, low-rank, or a learned position-dependent
metric --- applied **independently at each temperature**, each carrying its own adapted
parameters.

The blocks are defined once, with slices relative to a *single* temperature's coordinate
(``0 .. n``), exactly as for a non-tempered sampler. :class:`ProductKinetic` wraps one such block
and `vmap`s the block's own ``energy`` / ``velocity_into`` / ``sample_into`` over the temperature
axis. **No kinetic class changes**: the block classes are already written against
``(istate, ctx)`` with their own slices and their own ``ham_params[id]``, which is exactly what
`vmap` needs.

What this deliberately is *not*: one diagonal kinetic per temperature spanning that temperature's
whole coordinate. That would throw away the block partition and the learned metrics with it. Nor
is it ``K x B`` ordinary kinetics with offset slices --- correct, and needing no new code at all,
but it unrolls the graph K times instead of vectorising it.

**Both separable and non-separable blocks are supported**, and they differ only in :meth:`flow`.
A separable block's drift is rebuilt from its velocity, so one scatter serves every temperature.
A position-dependent (learned-metric) block has no explicit drift --- its flow drifts its own
coordinates and kicks the dependency momenta --- so that flow is vmapped per lane instead. The
composition is exact, not an approximation: see :meth:`ProductKinetic.flow`.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
from jax import Array

from .._logging import get_logger
from ..rng import DrawComponent
from ..hmc.hamiltonians import KineticHamiltonian
from ..hmc.state import IntegratorState, HamiltonianContext

log = get_logger(__name__)


class ProductKinetic(KineticHamiltonian):
    """One coordinate block's kinetic energy, applied at every temperature.

    ``ham_params[id]`` gains a leading ``K`` axis: row ``k`` is temperature ``k``'s parameters for
    this block. That is what makes per-temperature mass adaptation fall out --- temperature
    ``k``'s adaptation writes row ``k``, and the sampler stacks the per-temperature adaptations
    into the array.
    """

    def __init__(self, inner: KineticHamiltonian, n_temperatures: int, coord_dim: int):
        self.inner = inner
        self.id = inner.id
        self.slices = inner.slices
        self.separable = getattr(inner, "separable", True)
        self.mass_mode = getattr(inner, "mass_mode", None)
        self.n_temperatures = int(n_temperatures)
        self.coord_dim = int(coord_dim)
        self.block_size = inner._size(coord_dim)

    # --- the shared vmap ---

    def _lanes(self, fn, q_flat: Array, p_flat: Array, ctx, extra=None):
        """Run ``fn(istate_k, ctx_k, extra_k)`` at every temperature, batched.

        ``ham_params`` is mapped along its leading axis so each lane sees one temperature's
        parameters; the charts are shared and simply closed over. ``kinetic_cache`` is mapped the
        same way, so a lane's cached quantities are the ones built from *its own* parameters ---
        see :meth:`precompute`. When no block precomputes anything it is ``None``, which `vmap`
        treats as an empty pytree: every lane then sees ``None`` and the graph is what it was
        before the cache existed.
        """
        K, n = self.n_temperatures, self.coord_dim
        q = q_flat.reshape(K, n)
        p = p_flat.reshape(K, n)
        cache = getattr(ctx, "kinetic_cache", None)

        def lane(q_k, p_k, hp_k, extra_k, cache_k):
            istate = IntegratorState(
                q=q_k, p=p_k, potential_values={}, potential_grads={},
                log_weight=jnp.zeros(()), integrator_data={})
            ctx_k = HamiltonianContext(ctx.chart_hyperparams, ctx.chart_indices, hp_k,
                                       kinetic_cache=cache_k)
            return fn(istate, ctx_k, extra_k)

        return jax.vmap(lane)(q, p, ctx.ham_params, extra, cache)

    def precompute(self, ctx):
        """The inner block's per-trajectory cache, at every temperature (leading ``K`` axis).

        ``BaseHMC.context`` duck-types on this method to fill
        :attr:`~mimcs.hmc.state.HamiltonianContext.kinetic_cache` once per kernel call, which is
        how :class:`~mimcs.hmc.LowRankQuadraticKinetic` keeps its ``O(J^2 d)`` Woodbury recursion
        out of the trajectory loop. Without this the wrapper hid the inner block's ``precompute``
        and every tempered leaf rebuilt the factors --- correct, just not hoisted.

        ``None`` when the inner block has nothing to precompute, which is every kinetic but the
        low-rank one; ``context`` drops those entries so the cache stays empty and the emitted
        graph is unchanged. Delegating rather than assuming low-rank is also what keeps a
        *position-dependent* block correct: a shaped metric's factors depend on ``q``, so they are
        not trajectory constants, and such a block simply defines no ``precompute``.

        Vmapped over the **whole** ``ham_params`` dict, exactly as :meth:`_lanes` is --- that
        shared slicing rule is the correctness argument, since it makes row ``k`` of the cache a
        function of row ``k`` of the mass by construction.
        """
        inner = getattr(self.inner, "precompute", None)
        if inner is None:
            return None

        def one(hp_k):
            return inner(HamiltonianContext(ctx.chart_hyperparams, ctx.chart_indices, hp_k))

        return jax.vmap(one)(ctx.ham_params)

    # --- component interface ---

    def per_temperature_energy(self, istate, ctx) -> Array:
        """This block's kinetic energy at each temperature --- shape ``(K,)``.

        :meth:`energy` is this summed; the terms are what a sampler accepting independently at
        each temperature needs (doc 13).
        """
        return self._lanes(lambda ist, c, _: self.inner.energy(ist, c),
                           istate.q, istate.p, ctx, None)

    def energy(self, istate, ctx) -> Array:
        return jnp.sum(self.per_temperature_energy(istate, ctx))

    def velocity_into(self, v: Array, istate, ctx) -> Array:
        """Add this block's velocity, at every temperature, into the product velocity vector.

        Each lane scatters into a zero vector of one temperature's width, so the contributions are
        disjoint across blocks *and* temperatures and adding is the same as setting.
        """
        per = self._lanes(
            lambda ist, c, _: self.inner.velocity_into(jnp.zeros_like(ist.q), ist, c),
            istate.q, istate.p, ctx, None)
        return v + per.reshape(-1)

    def sample_into(self, p: Array, draw, q: Array, ctx) -> Array:
        z = getattr(draw, f"{self.id}_momentum")          # (K, block_size)

        def lane(ist, c, z_k):
            # The block reads its draw by name; hand it one temperature's slice under that name.
            one = SimpleNamespace(**{f"{self.id}_momentum": z_k})
            return self.inner.sample_into(jnp.zeros_like(ist.q), one, ist.q, c)

        per = self._lanes(lane, q, jnp.zeros_like(q), ctx, z)
        return p + per.reshape(-1)

    def _lane_eps(self, eps) -> Array:
        """The step size as a ``(K,)`` vector, one entry per temperature.

        ``eps`` is a scalar under NUTS (joint selection gives one global step) and a ``(K*n,)``
        vector under independent acceptance, where :meth:`~mimcs.pt.hmc.IndependentAcceptanceMixin.
        _eps` builds it as ``jnp.repeat(step_size, coord_dim)`` --- so it is **uniform within a
        temperature** by construction and column 0 of each row is that rung's step. A genuinely
        per-*coordinate* step size is not supported here, and is not supported untempered either:
        the inner flow multiplies a block-sized velocity by ``eps``.
        """
        eps = jnp.asarray(eps)
        if eps.ndim == 0:
            return jnp.full((self.n_temperatures,), eps)
        return eps.reshape(self.n_temperatures, self.coord_dim)[:, 0]

    def flow(self, istate, eps, ctx, use_cache=False):
        """Drift ``q -> q + eps * v`` for a separable block; the inner block's own flow otherwise.

        A **separable** block has an explicit drift built from its velocity, so the product flow is
        one scatter --- the fast path, and what every quadratic mass takes.

        A **non-separable** block (a position-dependent learned metric) has no such drift: its flow
        drifts its own coordinates *and* kicks the dependency momenta by an autodiff force
        (:meth:`mimcs.hmc.block_riemannian._DiagBlock.flow`), so it is run per lane and both ``q``
        and ``p`` come back changed. That is exact rather than an approximation: the product
        Hamiltonian is a sum of K per-temperature terms with **no coupling between rungs** (rungs
        meet only in the swap move), so composing each lane's own flow over disjoint coordinates
        *is* the flow of ``sum_k T_i^(k)``, and reversibility and volume preservation hold
        lane-wise. Each rung's metric depends only on that rung's other coordinates, which is
        automatic here because a lane never sees another temperature's state.
        """
        if self.separable:
            v = self.velocity_into(jnp.zeros_like(istate.q), istate, ctx)
            return istate._replace(q=istate.q + eps * v)

        def lane(ist, c, eps_k):
            out = self.inner.flow(ist, eps_k, c)
            return out.q, out.p

        q, p = self._lanes(lane, istate.q, istate.p, ctx, self._lane_eps(eps))
        return istate._replace(q=q.reshape(-1), p=p.reshape(-1))

    # --- RNG and initial parameters, both gaining the temperature axis ---

    def make_draw_components(self, dim: int) -> list[DrawComponent]:
        return [DrawComponent(f"{self.id}_momentum",
                              (self.n_temperatures, self.block_size), jax.random.normal)]

    def initial_mass_params(self, dim: int):
        one = self.inner.initial_mass_params(self.coord_dim)
        return jax.tree.map(
            lambda x: jnp.broadcast_to(jnp.asarray(x), (self.n_temperatures,) + jnp.shape(x)), one)


def build_product_kinetics(inner_kinetics, n_temperatures: int, coord_dim: int) -> list:
    """Wrap each of the model's kinetic blocks for the product space, preserving the structure."""
    out = [ProductKinetic(k, n_temperatures, coord_dim) for k in inner_kinetics]
    log.debug("product kinetics over %d temperature(s): %s", n_temperatures,
              ", ".join(f"{k.id}[{type(k.inner).__name__}, {k.block_size}d]" for k in out))
    return out
