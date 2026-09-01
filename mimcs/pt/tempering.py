"""The temperature ladder, the tempered product potential, and the product-space model view.

Implements the tempering half of ``docs/design/13_parallel_tempering.md``. Parallel tempering
runs K copies of the model at inverse temperatures ``1 = beta_1 > ... > beta_K >= 0``; the hot
copies flatten the barriers between modes, and swaps carry that mobility down to the cold chain,
whose draws are the ones kept.

Everything here works on the **product coordinate**: the ``(K * n,)`` vector holding temperature
``k``'s ``n`` coordinates at ``[k*n, (k+1)*n)``. That layout is what lets one `vmap` over the
leading temperature axis evaluate all K log-densities and gradients at once --- the whole cost of
a PT step, and the reason it is worth running on a GPU.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from .._logging import get_logger
from ..hmc.hamiltonians import PotentialHamiltonian
from .lanes import lane_discrete

log = get_logger(__name__)


def geometric_ladder(n_temperatures: int, beta_min: float = 0.01) -> Array:
    """``K`` inverse temperatures, geometrically spaced from ``1`` down to ``beta_min``.

    Geometric spacing is the standard default because the swap acceptance between neighbours
    depends on the *ratio* of temperatures rather than their difference, so a geometric ladder
    spreads the acceptance rates evenly when nothing better is known. ``beta_1`` is always exactly
    ``1``: that chain is the target, and its draws are what a run returns.
    """
    K = int(n_temperatures)
    if K < 1:
        raise ValueError(f"a ladder needs at least one temperature, got {K}")
    if K == 1:
        return jnp.ones((1,), float)
    if not 0.0 <= beta_min < 1.0:
        raise ValueError(f"beta_min must be in [0, 1), got {beta_min}")
    if beta_min == 0.0:
        # A geometric ladder cannot reach 0; approach it and place the last rung there.
        betas = jnp.concatenate(
            [jnp.geomspace(1.0, 1e-3, K - 1), jnp.zeros((1,), float)])
    else:
        betas = jnp.geomspace(1.0, beta_min, K)
    return jnp.asarray(betas, float)


class TemperedProductPotential(PotentialHamiltonian):
    """One model potential, evaluated at all K temperatures at once and scaled by ``beta``.

    Wraps a single :class:`~mimcs.hmc.ModelPotential` (or any potential over one temperature's
    coordinate) and presents it as a potential over the product coordinate::

        V(q) = sum_k  w_k * V_inner(q_k),     w_k = beta_k if tempered else 1

    The gradient is the K independent gradients, each scaled by its own ``w_k`` --- computed with
    a single `vmap`, which is the point of the whole design.

    One wrapper **per model component**, so the component structure the model exposes (doc 05)
    survives into the product space; ``component`` and ``inner`` are kept for the cost-aware
    integrator machinery that reads them.

    ``tempered=False`` leaves the component at full strength for every temperature. That is how a
    *power posterior* is expressed (prior untempered, likelihood tempered), and it is **forced**
    for the chart Jacobian: that term is a change of variables, not part of the target, and
    scaling it would sample a different distribution rather than a flatter one.
    """

    def __init__(self, inner: PotentialHamiltonian, betas: Array, coord_dim: int, *,
                 tempered: bool = True):
        self.inner = inner
        self.id = inner.id
        self.component = getattr(inner, "component", None)
        self.betas = jnp.asarray(betas, float)
        self.n_temperatures = int(self.betas.shape[0])
        self.coord_dim = int(coord_dim)
        self.tempered = bool(tempered)

    def _weights_from(self, ctx) -> Array:
        """The per-temperature scaling, preferring the ladder carried in the context.

        The ladder is adapted during warmup, so it must be a *traced* value: closing over a
        concrete ``self.betas`` would make every ladder update retrace the kernel. ``self.betas``
        remains the fallback for a fixed ladder and for direct use outside a sampler.
        """
        betas = getattr(ctx, "betas", None)
        if betas is None:
            betas = self.betas
        return betas if self.tempered else jnp.ones_like(betas)

    def _per_temperature(self, q: Array) -> Array:
        return q.reshape(self.n_temperatures, self.coord_dim)

    def _lane_potentials(self, q: Array, ctx) -> Array:
        """``V_inner(q_k, z_k)`` per temperature --- the one place lanes are actually evaluated.

        Each lane is given **its own** discrete block. Passing the shared ``ctx`` straight through,
        as this did before discrete parameters existed, would hand every rung the whole product
        block: a shape error at best, and at worst every rung evaluating its density at some other
        rung's labels. There is nothing to notice in the output when that goes wrong, so all four
        of this class's vmapped entry points route through here rather than repeating the map.
        """
        zs = lane_discrete(ctx, self.n_temperatures)
        if zs is None:
            # Literally the pre-discrete code path, so a continuous tempered run emits the graph
            # it always did -- and so a context stand-in without `_replace` still works here.
            return jax.vmap(lambda qk: self.inner.potential(qk, ctx))(self._per_temperature(q))

        def one(qk, zk):
            return self.inner.potential(qk, ctx._replace(discrete=zk))

        return jax.vmap(one)(self._per_temperature(q), zs)

    def potential(self, q: Array, ctx) -> Array:
        return jnp.sum(self._weights_from(ctx) * self._lane_potentials(q, ctx))

    def value_and_grad(self, q: Array, ctx):
        """One batched value-and-gradient over the temperature axis.

        Overridden rather than left to ``jax.value_and_grad(self.potential)`` so the batching is
        explicit: this is the single evaluation the whole design exists to share.
        """
        zs = lane_discrete(ctx, self.n_temperatures)
        if zs is None:
            vals, grads = jax.vmap(lambda qk: self.inner.value_and_grad(qk, ctx))(
                self._per_temperature(q))
        else:
            def one(qk, zk):
                return self.inner.value_and_grad(qk, ctx._replace(discrete=zk))

            vals, grads = jax.vmap(one)(self._per_temperature(q), zs)
        w = self._weights_from(ctx)
        return jnp.sum(w * vals), (w[:, None] * grads).reshape(-1)

    def per_temperature_values(self, q: Array, ctx) -> Array:
        """``w_k * V_inner(q_k)`` per temperature --- shape ``(K,)``.

        The scalar :meth:`potential` is this summed. Kept separate because a sampler that accepts
        **independently at each temperature** (RWMH, HMC --- doc 13) needs the terms, not the sum.
        """
        return self._weights_from(ctx) * self._lane_potentials(q, ctx)

    def untempered_values(self, q: Array, ctx) -> Array:
        """``V_inner(q_k)`` per temperature, *unscaled* --- shape ``(K,)``.

        The swap acceptance ratio is built from the raw log-density, not the beta-scaled one the
        potential caches, so the swap step asks for this rather than unpicking the cache.
        """
        return self._lane_potentials(q, ctx)


class ProductModel:
    """A model's-eye view of the ``(K * n,)`` product space.

    The samplers ask their model for a handful of things --- the dimensions, the charts, and the
    coordinate<->sample maps. This provides them over the product coordinate by `vmap`ping the
    real model, so :class:`~mimcs.hmc.NUTS` and friends run over a tempered product **unchanged**.

    The charts themselves are *shared* across temperatures: every copy is the same model, so it
    has the same parameters and the same chart hyperparameters. Only the positions differ.

    **There is deliberately no ``log_prob_at_coordinate`` here.** Over the product space the
    sampled target is the beta-weighted sum the tempered potentials evaluate, and the ladder they
    weight by is *adapted*, so it travels in the Hamiltonian context rather than on the model
    (see :class:`TemperedProductPotential`). A method with ``Model``'s signature could only bake
    in some fixed ladder, and would quietly return the wrong density the moment the ladder moved
    --- so anything needing the product target asks the potentials instead
    (``-sum(state.potential_values.values())`` is that number, already cached in the state).
    """

    def __init__(self, base, n_temperatures: int):
        self.base = base
        self.n_temperatures = int(n_temperatures)
        self.coord_dim = self.n_temperatures * base.coord_dim
        self.ambient_dim = self.n_temperatures * base.ambient_dim
        self.parameters = base.parameters
        self.log_prob_fns = base.log_prob_fns
        self.cheap_components = base.cheap_components
        # Every rung holds its own copy of the labels, so the discrete block is K-fold like the
        # coordinate. The *parameters* are the base's --- their names, supports and sizes describe
        # one rung, which is the view `discrete_block` and everything downstream works in.
        self.discrete_parameters = base.discrete_parameters
        self.discrete_dim = self.n_temperatures * base.discrete_dim

    # --- charts (shared across temperatures) ---

    def init_chart_hyperparams(self):
        return self.base.init_chart_hyperparams()

    def init_chart_indices(self):
        return self.base.init_chart_indices()

    # --- the maps, vmapped over the temperature axis ---

    def coordinate_to_sample(self, coord_flat: Array, chart_hyperparams, chart_indices) -> Array:
        per = coord_flat.reshape(self.n_temperatures, self.base.coord_dim)
        out = jax.vmap(lambda q: self.base.coordinate_to_sample(
            q, chart_hyperparams, chart_indices))(per)
        return out.reshape(-1)

    def sample_to_coordinate(self, sample_flat: Array, chart_hyperparams, chart_indices) -> Array:
        per = sample_flat.reshape(self.n_temperatures, self.base.ambient_dim)
        out = jax.vmap(lambda s: self.base.sample_to_coordinate(
            s, chart_hyperparams, chart_indices))(per)
        return out.reshape(-1)

    def pack_sample(self, sample_dict: dict) -> Array:
        """One temperature's ``{name: value}`` tiled across the ladder --- how a run starts."""
        one = self.base.pack_sample(sample_dict)
        return jnp.tile(one, self.n_temperatures)

    # --- the discrete block: K copies of the base's, laid out the same way ---

    def discrete_block(self, name: str) -> tuple[int, int]:
        """``(start, stop)`` of a discrete parameter's slice **within one rung**.

        Deliberately the base model's block, not an offset into the ``(K * n,)`` product: every
        consumer works in the ``(K, base.discrete_dim)`` view, exactly as
        :class:`~mimcs.pt.adaptation.PerTemperatureAdaptation` already reshapes the coordinate to
        ``(K, base.coord_dim)`` and slices columns.
        """
        return self.base.discrete_block(name)

    def default_discrete(self) -> Array:
        """The base model's starting labels, tiled across the ladder."""
        return self.tile_discrete(self.base.default_discrete())

    def pack_discrete(self, discrete_dict: dict) -> Array:
        return self.tile_discrete(self.base.pack_discrete(discrete_dict))

    def unpack_discrete_draws(self, draws) -> dict:
        """Delegated: the sampler only ever unpacks the *cold* chain's draws."""
        return self.base.unpack_discrete_draws(draws)

    @property
    def discrete_lower(self) -> Array:
        """One lane's elementwise lower bounds --- the base model's, since every rung holds the
        same parameters over the same supports."""
        return self.base.discrete_lower

    @property
    def discrete_upper(self) -> Array:
        """One lane's elementwise upper bounds (see :attr:`discrete_lower`)."""
        return self.base.discrete_upper

    def tile_discrete(self, one_temperature: Array) -> Array:
        """Repeat one temperature's label block across the ladder, **as int32**.

        The integer counterpart of :meth:`tile`, which casts to float and so cannot be reused:
        rounding labels through a float would be silent and, for a large support, lossy.
        """
        return jnp.tile(jnp.asarray(one_temperature, jnp.int32), self.n_temperatures)

    def unpack_draws(self, draws) -> dict:
        """Delegated: the sampler only ever unpacks the *cold* chain's draws."""
        return self.base.unpack_draws(draws)

    def features(self, sample_flat: Array, discrete_flat: Array | None = None) -> Array:
        """The **cold** chain's features --- what a convergence criterion should judge.

        A warmup-termination mixin buffers ``state.sample``, which here spans every temperature.
        The cold chain is the one whose draws are kept, so it is the one whose mixing decides when
        warmup is done; a hot rung mixes more easily and would end warmup early. This matches
        every other user-facing narrowing (:class:`~mimcs.pt.ProductSpaceMixin`), and ``beta_1 = 1``
        is pinned by construction, so the cold chain is always row 0.
        """
        z = None
        if discrete_flat is not None and self.base.discrete_dim:
            z = discrete_flat[:self.base.discrete_dim]
        return self.base.features(sample_flat[:self.base.ambient_dim], z)

    def tile(self, one_temperature: Array) -> Array:
        """Repeat a single temperature's flat vector across the ladder."""
        return jnp.tile(jnp.asarray(one_temperature, float), self.n_temperatures)


def build_tempered_potentials(model, betas, *, tempered=None):
    """One :class:`TemperedProductPotential` per potential of ``model``.

    ``tempered`` names the model components that ``beta`` scales; ``None`` (the default) means
    all of them, which is textbook parallel tempering. Naming a subset gives a power posterior.
    A :class:`~mimcs.hmc.JacobianPotential` is never tempered whatever is asked --- see the class
    docstring.
    """
    from ..hmc.samplers import default_potentials
    from ..hmc.hamiltonians import JacobianPotential

    if tempered is not None:
        unknown = set(tempered) - set(model.log_prob_fns)
        if unknown:
            raise ValueError(
                f"tempered names {sorted(unknown)}, which are not log-density components of this "
                f"model; its components are {list(model.log_prob_fns)}")

    out = []
    for p in default_potentials(model):
        if isinstance(p, JacobianPotential):
            is_tempered = False                     # a change of variables, never the target
        elif tempered is None:
            is_tempered = True
        else:
            is_tempered = getattr(p, "component", None) in set(tempered)
        out.append(TemperedProductPotential(p, betas, model.coord_dim, tempered=is_tempered))
    log.debug("tempered potentials: %s",
              ", ".join(f"{p.id}{'[beta]' if p.tempered else '[fixed]'}" for p in out))
    return out
