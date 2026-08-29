"""Hamiltonian components (axis 1 of the modular HMC design).

Implements the ``Hamiltonian`` protocol from
``docs/design/06_hamiltonian_monte_carlo.md``: each component contributes an additive
term to ``H = sum_i V_i(q) + sum_j T_j(q, p)`` and supplies the elementary flow it
generates. Potentials generate momentum *kicks*; (separable) kinetics generate
position *drifts*. The integrator (see :mod:`mimcs.hmc.integrators`) composes these
flows.

Provided here:

* :class:`ModelPotential` --- ``V = -log pi_component(x(q))`` for one model log-density component,
  and :class:`JacobianPotential` --- the charts' change-of-variables correction.
* :class:`DiagonalQuadraticKinetic`, :class:`DenseQuadraticKinetic` and
  :class:`LowRankQuadraticKinetic` --- quadratic kinetics whose adapted mass lives in
  ``ctx.ham_params[id]`` (the reparameterization principle: adapted parameters live in state).

The remaining kinetics follow the same interfaces from their own modules: the relativistic one in
:mod:`mimcs.hmc.relativistic`, and the position-dependent ones in
:mod:`mimcs.hmc.block_riemannian` (explicit, the supported path) and :mod:`mimcs.hmc.riemannian`
(implicit).
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import jax.scipy.linalg
from jax import Array

from ..rng import DrawComponent
from . import lowrank
from .state import IntegratorState, HamiltonianContext


class Hamiltonian:
    """Protocol: an additive energy term and the elementary flow it generates."""

    id: str

    def energy(self, istate: IntegratorState, ctx: HamiltonianContext) -> Array:
        raise NotImplementedError

    def flow(self, istate: IntegratorState, eps: Array, ctx: HamiltonianContext,
             use_cache: bool = False) -> IntegratorState:
        raise NotImplementedError


# --- potentials ------------------------------------------------------------- #

class PotentialHamiltonian(Hamiltonian):
    """A potential ``V(q)``. Flow is a kick: ``p -> p - eps * grad_q V(q)``.

    The kick consults the gradient cache when ``use_cache`` is set (the leapfrog
    leading half-kick), and otherwise computes a fresh value-and-gradient and writes
    both back into the state's caches (the trailing half-kick).
    """

    def potential(self, q: Array, ctx: HamiltonianContext) -> Array:
        raise NotImplementedError

    def value_and_grad(self, q: Array, ctx: HamiltonianContext):
        return jax.value_and_grad(self.potential)(q, ctx)

    def energy(self, istate, ctx):
        cached = istate.potential_values.get(self.id)
        return cached if cached is not None else self.potential(istate.q, ctx)

    def flow(self, istate, eps, ctx, use_cache=False):
        if use_cache:
            g = istate.potential_grads[self.id]
            return istate._replace(p=istate.p - eps * g)
        v, g = self.value_and_grad(istate.q, ctx)
        return istate._replace(
            p=istate.p - eps * g,
            potential_values={**istate.potential_values, self.id: v},
            potential_grads={**istate.potential_grads, self.id: g},
        )


class ModelPotential(PotentialHamiltonian):
    """``V = -log pi_component(x(q))`` for one named model log-density component."""

    def __init__(self, model, component_name: str):
        self.model = model
        self._name = component_name
        #: the log-density component this potential is for --- what a cost-aware integrator
        #: builder matches against ``Model.cheap_components`` (see ``hmc.split_potentials``).
        self.component = component_name
        self.id = f"V_{component_name}"

    def potential(self, q, ctx):
        sample = self.model.unpack_coordinate(q, ctx.chart_hyperparams, ctx.chart_indices)
        return -self.model.log_prob_fns[self._name](sample)


class JacobianPotential(PotentialHamiltonian):
    """``V = -log|J|`` from the charts' change-of-variables (doc 04).

    Needed whenever any parameter is constrained/manifold-typed; for an all-Euclidean
    model the Jacobian is zero and this potential can be omitted.
    """

    def __init__(self, model):
        self.model = model
        self.id = "V_jacobian"

    def potential(self, q, ctx):
        return -self.model.total_log_jacobian_from_coordinate(
            q, ctx.chart_hyperparams, ctx.chart_indices)


# --- kinetics --------------------------------------------------------------- #

class KineticHamiltonian(Hamiltonian):
    """A kinetic *component* ``T_i(p_i)`` over a coordinate block ``[s, e)``.

    ``BaseHMC`` holds a **list** of these and aggregates them, exactly as it holds a list of
    potentials (see ``mimcs/hmc/samplers.py``); a single whole-space kinetic is a one-element
    list with the default slice ``s=0, e=None`` (to the end). A component reads/writes only its
    block ``[s, e)`` of the momentum and assembles into the full vectors via ``velocity_into`` /
    ``sample_into``. Its mass parameters live in ``ctx.ham_params[self.id]`` (so each component
    needs a **unique** ``id``), and its momentum draw is named ``f"{id}_momentum"`` so multiple
    components never collide.

    Separable components (``separable = True``) inherit a base drift ``flow``; non-separable
    ones (RMHMC) override it. The integrator just composes each component's ``flow`` (it never
    branches on ``separable``).
    """

    separable: bool = True
    mass_mode: str | None = None   # "diagonal" / "dense" / None (see the mass adaptations)
    #: the block's coordinates as a list of ``(s, e)`` slices (possibly *non-contiguous*, so a
    #: block can fuse scattered parameters); ``None`` means the whole coordinate.
    slices = None

    # --- gather / scatter the block's coordinates ---------------------------- #

    def _gather(self, x: Array) -> Array:
        """The block's entries of ``x``, concatenated in slice order (whole ``x`` if None)."""
        if self.slices is None:
            return x
        if len(self.slices) == 1:
            s, e = self.slices[0]
            return x[s:e]
        return jnp.concatenate([x[s:e] for s, e in self.slices])

    def _scatter(self, base: Array, vals: Array) -> Array:
        """Write ``vals`` (in slice order) into the block's entries of ``base``."""
        if self.slices is None:
            return base.at[:].set(vals)
        out, off = base, 0
        for s, e in self.slices:
            n = e - s
            out = out.at[s:e].set(vals[off:off + n])
            off += n
        return out

    def _size(self, dim: int) -> int:
        if self.slices is None:
            return dim
        return sum(e - s for s, e in self.slices)

    # --- component interface ------------------------------------------------- #

    def energy(self, istate, ctx) -> Array:
        raise NotImplementedError

    def velocity_into(self, v: Array, istate, ctx) -> Array:
        """Write this block's velocity ``grad_{p_i} T_i`` into its coordinates of ``v``."""
        raise NotImplementedError

    def flow(self, istate, eps, ctx, use_cache=False):
        assert self.separable, "non-separable kinetics must override flow"
        v = self.velocity_into(jnp.zeros_like(istate.q), istate, ctx)
        return istate._replace(q=istate.q + eps * v)

    def make_draw_components(self, dim: int) -> list[DrawComponent]:
        return [DrawComponent(f"{self.id}_momentum", (self._size(dim),), jax.random.normal)]

    def sample_into(self, p: Array, draw, q: Array, ctx) -> Array:
        """Write this block's refreshed momentum into its coordinates of ``p``. ``q`` is the
        current position (needed when the metric is position-dependent, ``p ~ N(0, G(q))``)."""
        raise NotImplementedError

    def initial_mass_params(self, dim: int):
        raise NotImplementedError


class DiagonalQuadraticKinetic(KineticHamiltonian):
    """``T = 1/2 p_i^T diag(m)^{-1} p_i`` over its block. ``ctx.ham_params[id]`` holds
    ``M^{-1}`` (the block's variance vector). Momentum refresh: ``p = z / sqrt(M^{-1})``."""

    separable = True
    mass_mode = "diagonal"      # M^{-1} stored as the variance vector

    def __init__(self, id: str = "T", slices=None):
        self.id = id
        self.slices = slices

    def _inv_mass(self, ctx) -> Array:
        return ctx.ham_params[self.id]

    def energy(self, istate, ctx):
        return 0.5 * jnp.sum(self._inv_mass(ctx) * self._gather(istate.p) ** 2)

    def velocity_into(self, v, istate, ctx):
        return self._scatter(v, self._inv_mass(ctx) * self._gather(istate.p))

    def sample_into(self, p, draw, q, ctx):
        z = getattr(draw, f"{self.id}_momentum")
        return self._scatter(p, z / jnp.sqrt(self._inv_mass(ctx)))

    def initial_mass_params(self, dim):
        return jnp.ones((self._size(dim),))


class DenseQuadraticKinetic(KineticHamiltonian):
    """``T = 1/2 p_i^T M^{-1} p_i`` with a dense inverse mass over its block.
    ``ctx.ham_params[id]`` holds the lower Cholesky ``L`` of ``M^{-1}``; ``p = L^{-T} z``."""

    separable = True
    mass_mode = "dense"         # M^{-1} stored as its lower Cholesky factor L

    def __init__(self, id: str = "T", slices=None):
        self.id = id
        self.slices = slices

    def _chol(self, ctx) -> Array:
        return ctx.ham_params[self.id]      # lower Cholesky L of M^{-1}

    def energy(self, istate, ctx):
        Lt_p = self._chol(ctx).T @ self._gather(istate.p)
        return 0.5 * jnp.dot(Lt_p, Lt_p)

    def velocity_into(self, v, istate, ctx):
        L = self._chol(ctx)
        return self._scatter(v, L @ (L.T @ self._gather(istate.p)))    # M^{-1} p_i

    def sample_into(self, p, draw, q, ctx):
        # p_i = L^{-T} z  =>  Cov(p_i) = (L L^T)^{-1} = M_i
        z = getattr(draw, f"{self.id}_momentum")
        p_i = jax.scipy.linalg.solve_triangular(self._chol(ctx), z, lower=True, trans="T")
        return self._scatter(p, p_i)

    def initial_mass_params(self, dim):
        return jnp.eye(self._size(dim))


class LowRankQuadraticKinetic(KineticHamiltonian):
    """``T = 1/2 p_i^T M^{-1} p_i`` with a diagonal-whitened rank-``J`` mass over its block.

    The mass ``M = D^{1/2} (I + sum_{j=1}^J gamma_j v_j v_j^T) D^{1/2}`` -- a per-coordinate
    scale ``D`` plus a rank-``J`` stiffening along the top eigen-directions ``v_j`` of the
    ``D^{-1/2}``-whitened score covariance (``gamma_j >= 0``). It sits between the diagonal and
    the dense quadratic kinetics: it captures the leading correlations at ``O(J d)`` storage.

    ``ctx.ham_params[id]`` holds the packed tuple ``(D, V)`` in the :mod:`mimcs.hmc.lowrank`
    convention ``M = diag(D) + V^T V``, where ``V`` has shape ``(rank, size)`` and
    ``V[j] = sqrt(gamma_j) * sqrt(D) * v_j`` (so ``V^T V = D^{1/2}(sum_j gamma_j v_j v_j^T)
    D^{1/2}``). Energy/velocity use ``M^{-1}`` via the Woodbury :func:`~mimcs.hmc.lowrank.apply_inv`;
    momentum ``p = S z`` with ``S S^T = M`` via :func:`~mimcs.hmc.lowrank.apply_chol` (so
    ``Cov(p) = M``). ``rank`` is static (it fixes ``V``'s row count and the unrolled loops in
    ``lowrank``). Because ``gamma_j >= 0`` the rank term can only *stiffen*: whitened directions
    with variance below 1 are left at the diagonal scale ``D``. Adapted by
    :class:`~mimcs.adaptation.LowRankAdaptation`."""

    separable = True
    mass_mode = None            # partitions this block off the diagonal/dense mass adaptations;
                                # the dedicated LowRankAdaptation is its sole adapter

    def __init__(self, id: str = "T", slices=None, rank: int = 4):
        self.id = id
        self.slices = slices
        self.rank = int(rank)

    def precompute(self, ctx):
        """The Woodbury inverse factors, once per trajectory. See :attr:`_inv_factors`."""
        D, V = ctx.ham_params[self.id]
        return lowrank.inv_factors(D, V)

    def _inv_factors(self, D, V, ctx):
        """``(beta, t)`` from the context's per-trajectory cache, or computed if it is absent.

        The factors are a function of ``(D, V)`` alone, so they are constant for a whole
        trajectory --- but ``energy`` and ``velocity_into`` are called several times per leaf, and
        XLA does not hoist the ``O(q^2 d)`` recursion out of the trajectory ``while_loop``. Taking
        them from :attr:`~mimcs.hmc.state.HamiltonianContext.kinetic_cache` computes them once per
        kernel call instead. The fallback keeps the class usable with a hand-built context (a
        direct ``HamiltonianContext(...)``, as several tests construct), where the answer is the
        same and only the cost differs."""
        cache = getattr(ctx, "kinetic_cache", None)
        if cache is not None and self.id in cache:
            return cache[self.id]
        return lowrank.inv_factors(D, V)

    def energy(self, istate, ctx):
        D, V = ctx.ham_params[self.id]
        p_i = self._gather(istate.p)
        beta, t = self._inv_factors(D, V, ctx)
        return 0.5 * jnp.dot(p_i, lowrank.apply_inv_factored(D, beta, t, p_i))

    def velocity_into(self, v, istate, ctx):
        D, V = ctx.ham_params[self.id]
        beta, t = self._inv_factors(D, V, ctx)
        return self._scatter(
            v, lowrank.apply_inv_factored(D, beta, t, self._gather(istate.p)))   # M^{-1} p_i

    def sample_into(self, p, draw, q, ctx):
        # p_i = S z with S S^T = M  =>  Cov(p_i) = M_i
        D, V = ctx.ham_params[self.id]
        z = getattr(draw, f"{self.id}_momentum")
        return self._scatter(p, lowrank.apply_chol(D, V, z))

    def initial_mass_params(self, dim):
        size = self._size(dim)
        return (jnp.ones((size,)), jnp.zeros((self.rank, size)))        # gamma = 0  =>  M = I


# --- total energy ----------------------------------------------------------- #

def total_energy(istate: IntegratorState, potentials, kinetics, ctx) -> Array:
    """``H = sum_i V_i(q) + sum_j T_j(q, p)`` using cached potential values when present.

    ``kinetics`` is the list of kinetic components (a single kinetic may be passed and is
    treated as a one-element list)."""
    if not isinstance(kinetics, (list, tuple)):
        kinetics = (kinetics,)
    v = jnp.zeros(())
    for potential in potentials:
        v = v + potential.energy(istate, ctx)
    for kinetic in kinetics:
        v = v + kinetic.energy(istate, ctx)
    return v
