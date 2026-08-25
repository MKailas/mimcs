"""The ``Model`` class: parameters plus a decomposed log-probability.

Implements ``docs/design/05_model_interface.md``. The decomposed ``log_prob_fns``
(prior, likelihood, ...) are the primary interface; flat pack/unpack helpers and a
coordinate-space density are provided for the samplers.

Samplers operate in *coordinate* space, so the canonical target a sampler sees is
``log_prob_at_coordinate`` --- the model density pulled back through the charts plus
the chart Jacobian correction. ``state.log_prob`` always stores this quantity, so
Metropolis acceptance with a symmetric coordinate-space proposal needs no further
correction (doc 04, "Density Correction").

Parameters may have *parents*: a parameter's chart can depend on the (ambient) values
of other parameters (e.g. a bound that depends on another parameter). The model
therefore evaluates ``from_coordinate`` in topological order, threading already-built
parent samples to each child. The coordinate->sample map is triangular, so the total
log-Jacobian is still the sum of per-parameter terms (doc 04, "Parameters with
parents"); autodiff through this construction yields correct coordinate-space
gradients for HMC.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp
from jax import Array

from .._logging import get_logger
from .parameter import BaseParameter

log = get_logger(__name__)

LogProbFn = Callable[[dict], Array]


class Model:
    """A probabilistic model over a list of typed parameters.

    Args:
        parameters: ordered list of :class:`BaseParameter`.
        log_prob_fns: named scalar log-density components, each a pure function of a
            dict ``{param_name: ambient_array}``. Their sum is the joint log-density.
        cheap_components: the components whose *gradient* is cheap to evaluate --- typically a
            prior, which touches no data. This is what a multi-rate (RESPA) integrator needs in
            order to kick a cheap potential several times per expensive one
            (``docs/design/06_hamiltonian_monte_carlo.md``); it says nothing about the density
            itself, so it never changes what the model *is*. Every component not named here is
            expensive, and the empty default means "nothing is known to be cheap" --- the
            conservative reading, under which a multi-rate builder finds nothing to nest and
            falls back to plain leapfrog, exactly as today. A DSL-compiled model gets this from
            its :class:`~mimcs.dsl.spec.ModelSpec`; a hand-written one passes it directly.
    """

    def __init__(self, parameters: list[BaseParameter], log_prob_fns: dict[str, LogProbFn],
                 *, cheap_components=()):
        self.parameters = list(parameters)
        self.log_prob_fns = dict(log_prob_fns)
        self.cheap_components = frozenset(cheap_components)
        unknown = self.cheap_components - set(self.log_prob_fns)
        if unknown:
            raise ValueError(
                f"cheap_components names {sorted(unknown)}, which are not log-density "
                f"components of this model; its components are {list(self.log_prob_fns)}")

        self._name_to_idx = {p.name: i for i, p in enumerate(self.parameters)}
        self._order = self._topological_order()

        self._ambient_sizes = [int(np.prod(p.ambient_shape)) if p.ambient_shape else 1
                               for p in self.parameters]
        self._coord_sizes = [p.coord_dim for p in self.parameters]
        self._ambient_offsets = np.concatenate([[0], np.cumsum(self._ambient_sizes)])
        self._coord_offsets = np.concatenate([[0], np.cumsum(self._coord_sizes)])

        log.debug("Model: %d parameter(s) [%s], coord_dim %d, ambient_dim %d, log-prob "
                  "component(s) %s, cheap %s", len(self.parameters),
                  ", ".join(f"{p.name}:{type(p).__name__}({p.coord_dim}d)"
                            for p in self.parameters),
                  self.coord_dim, self.ambient_dim, list(self.log_prob_fns),
                  sorted(self.cheap_components) or "(none)")

    # --- component cost ---

    @property
    def expensive_components(self) -> frozenset:
        """The components not declared cheap --- the complement of :attr:`cheap_components`."""
        return frozenset(self.log_prob_fns) - self.cheap_components

    def is_cheap(self, name: str) -> bool:
        """Is this component's gradient cheap to evaluate? (Unknown counts as expensive.)"""
        return name in self.cheap_components

    # --- dependency graph ---

    def _topological_order(self) -> list[int]:
        n = len(self.parameters)
        indeg = [0] * n
        children: list[list[int]] = [[] for _ in range(n)]
        for i, p in enumerate(self.parameters):
            for parent in getattr(p, "parents", ()):
                if parent not in self._name_to_idx:
                    raise ValueError(
                        f"parameter '{p.name}' references unknown parent '{parent}'")
                j = self._name_to_idx[parent]
                indeg[i] += 1
                children[j].append(i)
        queue = [i for i in range(n) if indeg[i] == 0]
        order: list[int] = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for c in children[node]:
                indeg[c] -= 1
                if indeg[c] == 0:
                    queue.append(c)
        if len(order) != n:
            raise ValueError("parameter dependency graph has a cycle")
        return order

    # --- dimensions ---

    def coord_block(self, name: str) -> tuple[int, int]:
        """``(start, stop)`` of parameter ``name``'s slice in the flat coordinate vector."""
        i = self._name_to_idx[name]
        return int(self._coord_offsets[i]), int(self._coord_offsets[i + 1])

    @property
    def coord_dim(self) -> int:
        return int(self._coord_offsets[-1])

    @property
    def ambient_dim(self) -> int:
        return int(self._ambient_offsets[-1])

    # --- features (observables) ---

    @property
    def n_features(self) -> int:
        return sum(p.n_features for p in self.parameters)

    @property
    def feature_names(self) -> list:
        return [n for p in self.parameters for n in p.feature_names()]

    def features(self, sample_flat: Array) -> Array:
        """Observables of a draw: ``(n_features,)``, concatenated over the parameters.

        The layer a convergence diagnostic works in. Each parameter decides which fixed functions
        of its *ambient* value are worth watching (see :meth:`BaseParameter.features`), so a
        diagnostic never has to know what kind of space a parameter lives on. Differentiability is
        not needed, but this is JAX and vmappable: ``jax.vmap(model.features)(draws)``.
        """
        return jnp.concatenate([
            p.features(sample_flat[self._ambient_offsets[i]:self._ambient_offsets[i + 1]])
            for i, p in enumerate(self.parameters)])

    @property
    def ambient_names(self) -> list:
        return [n for p in self.parameters for n in p.ambient_names()]

    def stein_terms(self, sample_flat: Array, score_flat: Array) -> Array:
        """Per-feature Langevin--Stein terms, concatenated over the parameters (``(n_features,)``).

        ``score_flat`` is the ambient score ``grad_x log pi(x)`` (see :meth:`ambient_score`). Each
        parameter maps its own sample and score block to its features' Stein terms
        (:meth:`BaseParameter.stein_terms`); a manifold parameter projects the score internally, so
        only its tangential part matters.
        """
        return jnp.concatenate([
            p.stein_terms(sample_flat[self._ambient_offsets[i]:self._ambient_offsets[i + 1]],
                          score_flat[self._ambient_offsets[i]:self._ambient_offsets[i + 1]])
            for i, p in enumerate(self.parameters)])

    # --- ambient score (for the Stein diagnostic) ---

    def log_prob_flat(self, sample_flat: Array) -> Array:
        """The ambient log-density from a flat sample vector (no chart Jacobian)."""
        return self.log_prob(self.unpack_sample(sample_flat))

    def ambient_score(self, sample_flat: Array, coord_score: Array | None = None,
                      chart_hyperparams: tuple | None = None,
                      chart_indices: tuple | None = None) -> Array:
        """The ambient score ``grad_x log pi(x)``, for the Stein diagnostic.

        The sampler's saved gradients are in *coordinate* space --- ``s_coord = grad_q [log pi(x(q))
        + log|J|]`` --- not what the Stein operator needs. But the model gradient the sampler
        already spent is inside ``s_coord``, and the chain rule ``s_coord = J^T s_ambient +
        grad_q log|J|`` (``J = dx/dq``) recovers the ambient score without touching the model
        again:

        * **Pullback** (``coord_score`` given): subtract the chart's own Jacobian term
          ``grad_q log|J|`` --- itself model-free --- then apply ``(J^T)^{-1}`` as a
          vector--Jacobian product through :meth:`sample_to_coordinate`. For an identity chart this
          is a passthrough (the saved gradient *is* the ambient score); a parent-dependent chart's
          cross terms and a manifold's non-square ``J`` are both handled by the autodiff. Reuses
          ``coord_score`` at the frozen sampling chart; never forms a dense Jacobian.
        * **Recompute** (``coord_score`` None --- a gradient-free sampler): plain
          ``grad log_prob_flat``.

        For a manifold parameter the pullback recovers only the *tangential* score, which is all
        :meth:`BaseParameter.stein_terms` uses --- the two paths therefore agree once that
        projection is applied.
        """
        if coord_score is None:
            return jax.grad(self.log_prob_flat)(sample_flat)
        h = self.init_chart_hyperparams() if chart_hyperparams is None else chart_hyperparams
        c = self.init_chart_indices() if chart_indices is None else chart_indices
        coord = self.sample_to_coordinate(sample_flat, h, c)
        dlog_j = jax.grad(self.total_log_jacobian_from_coordinate)(coord, h, c)
        _, vjp = jax.vjp(lambda s: self.sample_to_coordinate(s, h, c), sample_flat)
        return vjp(coord_score - dlog_j)[0]

    # --- chart state initialization ---

    def init_chart_hyperparams(self) -> tuple:
        return tuple(p.init_hyperparams() for p in self.parameters)

    def init_chart_indices(self) -> tuple:
        return tuple(jnp.zeros((), jnp.int32) for _ in self.parameters)

    # --- a valid default starting point ---

    def default_sample(self) -> Array:
        """A valid flat ambient point: the one at the *origin* of the initial charts.

        Every chart maps its own well-behaved interior point to ``0``, so this is valid by
        construction whatever the parameter types are, and it threads parent-dependent bounds
        through the usual topological order. It gives ``0`` for a Euclidean parameter (so it
        matches a plain zero vector), the midpoint of a doubly-bounded one, ``L + 1`` / ``U - 1``
        for a singly-bounded one, and the antipode of the pole --- the point farthest from the
        chart's singularity --- for a unit vector. Note that a flat ambient zero vector is *not*
        a valid default for either a bounded parameter or a unit vector.
        """
        return self.coordinate_to_sample(
            jnp.zeros((self.coord_dim,), float),
            self.init_chart_hyperparams(),
            self.init_chart_indices())

    # --- pack / unpack ---

    def unpack_sample(self, sample_flat: Array) -> dict:
        out = {}
        for i, p in enumerate(self.parameters):
            lo, hi = self._ambient_offsets[i], self._ambient_offsets[i + 1]
            out[p.name] = jnp.reshape(sample_flat[lo:hi], p.ambient_shape)
        return out

    def unpack_draws(self, draws) -> dict:
        """A *stack* of flat samples -> ``{name: array}``, each ``(n_draws, *ambient_shape)``.

        The plural of :meth:`unpack_sample`, and what :meth:`mimcs.samplers.BaseSampler.get_samples`
        hands back: a scalar parameter comes out ``(n,)``, a vector one ``(n, d)``, and an array
        of unit vectors ``(n, batch, d)``.

        Deliberately written with plain slicing and ``.reshape`` rather than ``jnp``, so the
        array type is **preserved**: numpy draws stay numpy (which is what the sampler stores and
        what plotting or ArviZ will want), and traced or device arrays stay themselves.
        """
        n = len(draws)
        out = {}
        for i, p in enumerate(self.parameters):
            lo, hi = int(self._ambient_offsets[i]), int(self._ambient_offsets[i + 1])
            out[p.name] = draws[:, lo:hi].reshape(n, *p.ambient_shape)
        return out

    def pack_sample(self, sample_dict: dict) -> Array:
        return jnp.concatenate(
            [jnp.reshape(sample_dict[p.name], (self._ambient_sizes[i],))
             for i, p in enumerate(self.parameters)]
        )

    def _parents_of(self, p: BaseParameter, sample_dict: dict) -> dict:
        return {name: sample_dict[name] for name in p.parents}

    def unpack_coordinate(self, coord_flat: Array, chart_hyperparams: tuple,
                          chart_indices: tuple) -> dict:
        """Coordinate vector -> ``{name: ambient_value}``, in topological order."""
        samples: dict = {}
        for i in self._order:
            p = self.parameters[i]
            lo, hi = self._coord_offsets[i], self._coord_offsets[i + 1]
            samples[p.name] = p.from_coordinate(
                coord_flat[lo:hi], chart_hyperparams[i], chart_indices[i],
                self._parents_of(p, samples))
        return samples

    def pack_coordinate(self, sample_dict: dict, chart_hyperparams: tuple,
                        chart_indices: tuple) -> Array:
        parts = [None] * len(self.parameters)
        for i, p in enumerate(self.parameters):
            parts[i] = p.to_coordinate(
                sample_dict[p.name], chart_hyperparams[i], chart_indices[i],
                self._parents_of(p, sample_dict))
        return jnp.concatenate(parts)

    # --- coordinate <-> sample ---

    def coordinate_to_sample(self, coord_flat: Array, chart_hyperparams: tuple,
                             chart_indices: tuple) -> Array:
        return self.pack_sample(
            self.unpack_coordinate(coord_flat, chart_hyperparams, chart_indices))

    def sample_to_coordinate(self, sample_flat: Array, chart_hyperparams: tuple,
                             chart_indices: tuple) -> Array:
        return self.pack_coordinate(
            self.unpack_sample(sample_flat), chart_hyperparams, chart_indices)

    # --- densities ---

    def log_prob_components(self, sample_dict: dict) -> dict:
        return {name: fn(sample_dict) for name, fn in self.log_prob_fns.items()}

    def log_prob(self, sample_dict: dict) -> Array:
        total = jnp.zeros(())
        for fn in self.log_prob_fns.values():
            total = total + fn(sample_dict)
        return total

    def total_log_jacobian(self, coord_flat: Array, sample_dict: dict,
                           chart_hyperparams: tuple, chart_indices: tuple) -> Array:
        """Sum of per-parameter log-Jacobians (parents drawn from ``sample_dict``)."""
        total = jnp.zeros(())
        for i, p in enumerate(self.parameters):
            lo, hi = self._coord_offsets[i], self._coord_offsets[i + 1]
            total = total + p.log_jacobian_det(
                coord_flat[lo:hi], chart_hyperparams[i], chart_indices[i],
                self._parents_of(p, sample_dict))
        return total

    def total_log_jacobian_from_coordinate(self, coord_flat: Array,
                                           chart_hyperparams: tuple,
                                           chart_indices: tuple) -> Array:
        """Total log-Jacobian as a function of the coordinate alone (unpacks samples)."""
        sample_dict = self.unpack_coordinate(coord_flat, chart_hyperparams, chart_indices)
        return self.total_log_jacobian(coord_flat, sample_dict, chart_hyperparams,
                                       chart_indices)

    def log_prob_at_coordinate(self, coord_flat: Array, chart_hyperparams: tuple,
                               chart_indices: tuple) -> Array:
        """Coordinate-space target: ``log pi(from_coordinate(q)) + log|J|``.

        This is the density the sampler's Markov kernel is reversible with respect
        to, and what ``state.log_prob`` stores. Differentiable in ``coord_flat``.
        """
        sample_dict = self.unpack_coordinate(coord_flat, chart_hyperparams, chart_indices)
        logp = self.log_prob(sample_dict)
        logj = self.total_log_jacobian(coord_flat, sample_dict, chart_hyperparams,
                                       chart_indices)
        return logp + logj
