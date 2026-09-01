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

from dataclasses import dataclass

from .._logging import get_logger
from .discrete import BaseDiscreteParameter
from .parameter import BaseParameter

log = get_logger(__name__)

LogProbFn = Callable[[dict], Array]


@dataclass(frozen=True)
class ScanComponent:
    """A log-density component that is a **sum over elements**, plus the element itself.

    An ordinary component is an opaque scalar function: nothing in it says whether it is a sum of
    per-observation terms or one irreducible quantity. That distinction is exactly what a
    single-site discrete sweep needs --- changing one label perturbs one term of a mixture
    likelihood, and recomputing the whole density to find out is the cost
    ``docs/design/14_discrete_parameters.md`` calls the largest available win.

    So a scan component declares it. ``element_fn(values, i, overrides=None)`` is one element's
    contribution; ``Model.log_prob_fns[name]`` is their sum, which is all any continuous sampler
    ever sees. ``overrides`` supplies a scanned name's value for this element directly, so a
    proposal costs no array write.

    ``scanned`` names the arrays the component is elementwise in, and this is the property the
    sweep relies on: for a discrete parameter among them, **element ``i`` is coordinate ``i``**.
    The DSL guarantees it by *scoping* --- inside the body the name denotes the element and the
    array is not reachable --- rather than by analysing index expressions.
    """

    element_fn: Callable      # (values, i, overrides=None) -> scalar
    scanned: tuple            # the names scanned over, in header order
    length: int               # their common leading dimension --- the number of elements


def _concat(parts: list, dtype=float) -> Array:
    """``jnp.concatenate``, but a zero-length result for an empty list.

    A model may legitimately have *no* continuous parameters --- a purely discrete one, sampled
    under :class:`~mimcs.samplers.StaticContinuous` --- and then every flat continuous vector is
    width zero. ``jnp.concatenate([])`` raises instead of producing that.
    """
    if not parts:
        return jnp.zeros((0,), dtype)
    return jnp.concatenate(parts)


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
        discrete_parameters: integer-valued parameters
            (:class:`~mimcs.model.BaseDiscreteParameter`). They are kept in a list of their own,
            with a flat ``int`` layout of their own, and contribute to neither ``coord_dim`` nor
            ``ambient_dim`` --- a discrete parameter has no chart and no gradient, so nothing that
            partitions coordinates, adapts a mass or differentiates a density should ever see one.
            The log-density components *do*: they are functions of a ``{name: value}`` dict, and
            the discrete values are simply further entries in it
            (``docs/design/14_discrete_parameters.md``).
        scan_components: for the components that are a **sum over elements**, the
            :class:`ScanComponent` describing one element. Optional and purely additive: the
            component's entry in ``log_prob_fns`` is the sum, so nothing that reads
            ``log_prob_fns`` behaves differently. Only the discrete Gibbs sweep looks here, to
            recompute one coordinate's contribution instead of the whole density.
        component_reads: per component, the parameter names it reads. A component **absent from
            this dict is treated as reading everything**, which is the conservative reading and
            the reason a hand-written model needs no change: with no entries, every component is
            assumed to depend on every label and the sweep recomputes exactly what it always did.
            A DSL-compiled model supplies the real sets, from
            :attr:`~mimcs.dsl.spec.ComponentSpec.reads`.
    """

    def __init__(self, parameters: list[BaseParameter], log_prob_fns: dict[str, LogProbFn],
                 *, cheap_components=(), discrete_parameters=(), scan_components=None,
                 component_reads=None):
        self.parameters = list(parameters)
        self.discrete_parameters = list(discrete_parameters)
        self.log_prob_fns = dict(log_prob_fns)
        self.cheap_components = frozenset(cheap_components)
        self.scan_components = dict(scan_components or {})
        self.component_reads = dict(component_reads or {})
        for label, names in (("cheap_components", self.cheap_components),
                             ("scan_components", set(self.scan_components)),
                             ("component_reads", set(self.component_reads))):
            unknown = set(names) - set(self.log_prob_fns)
            if unknown:
                raise ValueError(
                    f"{label} names {sorted(unknown)}, which are not log-density "
                    f"components of this model; its components are {list(self.log_prob_fns)}")

        self._name_to_idx = {p.name: i for i, p in enumerate(self.parameters)}
        self._discrete_name_to_idx = {p.name: i for i, p in enumerate(self.discrete_parameters)}
        # Before the topological sort: a discrete parent is not an *unknown* parent, and the sort
        # would otherwise report it as one.
        self._validate_discrete()
        self._order = self._topological_order()

        self._ambient_sizes = [int(np.prod(p.ambient_shape)) if p.ambient_shape else 1
                               for p in self.parameters]
        self._coord_sizes = [p.coord_dim for p in self.parameters]
        self._ambient_offsets = np.concatenate([[0], np.cumsum(self._ambient_sizes)])
        self._coord_offsets = np.concatenate([[0], np.cumsum(self._coord_sizes)])

        self._discrete_sizes = [p.size for p in self.discrete_parameters]
        self._discrete_offsets = np.concatenate([[0], np.cumsum(self._discrete_sizes)])

        log.debug("Model: %d parameter(s) [%s], coord_dim %d, ambient_dim %d, log-prob "
                  "component(s) %s, cheap %s", len(self.parameters),
                  ", ".join(f"{p.name}:{type(p).__name__}({p.coord_dim}d)"
                            for p in self.parameters),
                  self.coord_dim, self.ambient_dim, list(self.log_prob_fns),
                  sorted(self.cheap_components) or "(none)")

    def _validate_discrete(self) -> None:
        """Name collisions, and the stage-1 no-discrete-parent rule.

        A discrete parameter may not be the **parent** of a continuous parameter's chart. That is
        a real restriction, and it is what keeps the Gibbs sweep simple: with it, the total
        log-Jacobian cannot depend on the discrete block, so changing a label moves ``log_prob``
        and nothing else --- ``coordinate`` and ``sample`` are untouched and ``JacobianPotential``
        needs no discrete argument at all. Lifting it means recomputing the sample inside the
        sweep and threading the discrete block into the Jacobian; see doc 14.
        """
        clash = sorted(set(self._name_to_idx) & set(self._discrete_name_to_idx))
        if clash:
            raise ValueError(
                f"parameter name(s) {clash} are declared both continuous and discrete")
        names = [p.name for p in self.discrete_parameters]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"duplicate discrete parameter name(s) {dupes}")
        for p in self.parameters:
            bad = [n for n in getattr(p, "parents", ()) if n in self._discrete_name_to_idx]
            if bad:
                raise NotImplementedError(
                    f"continuous parameter '{p.name}' has discrete parent(s) {bad}. A discrete "
                    f"parameter may not be a chart's parent yet: the chart --- and so the "
                    f"log-Jacobian --- would change when the Gibbs sweep moves the label, which "
                    f"stage 1 assumes it cannot "
                    f"(docs/design/14_discrete_parameters.md, 'What is deferred')")

    def _require_discrete(self, discrete, what: str):
        """The loud-``None`` policy: ``None`` is fine only when there is no discrete block.

        A defaulted-away discrete block would evaluate the density at stale labels --- correct
        shapes, correct dtypes, wrong answer, and nothing raising. So a model that *has* one
        insists on being given it.
        """
        if discrete is None:
            if self.discrete_dim:
                raise ValueError(
                    f"{what} needs this model's discrete parameter(s) "
                    f"{[p.name for p in self.discrete_parameters]}: pass `discrete=` "
                    f"(the sampler threads it through `state.discrete`)")
            return None
        return discrete

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

    def discrete_block(self, name: str) -> tuple[int, int]:
        """``(start, stop)`` of discrete parameter ``name``'s slice in the flat integer vector."""
        i = self._discrete_name_to_idx[name]
        return int(self._discrete_offsets[i]), int(self._discrete_offsets[i + 1])

    @property
    def discrete_dim(self) -> int:
        """Width of the flat integer block --- ``0`` for a purely continuous model."""
        return int(self._discrete_offsets[-1])

    # --- features (observables) ---

    @property
    def n_features(self) -> int:
        return (sum(p.n_features for p in self.parameters)
                + sum(p.n_features for p in self.discrete_parameters))

    @property
    def feature_names(self) -> list:
        return ([n for p in self.parameters for n in p.feature_names()]
                + [n for p in self.discrete_parameters for n in p.feature_names()])

    def features(self, sample_flat: Array, discrete_flat: Array | None = None) -> Array:
        """Observables of a draw: ``(n_features,)``, concatenated over the parameters.

        The layer a convergence diagnostic works in. Each parameter decides which fixed functions
        of its *ambient* value are worth watching (see :meth:`BaseParameter.features`), so a
        diagnostic never has to know what kind of space a parameter lives on. Differentiability is
        not needed, but this is JAX and vmappable: ``jax.vmap(model.features)(draws)``.

        The continuous features come first, then the discrete ones --- the same order as
        :attr:`feature_names`. A discrete parameter contributes its bare value and nothing else
        (see :class:`~mimcs.model.BaseDiscreteParameter`).
        """
        parts = [p.features(sample_flat[self._ambient_offsets[i]:self._ambient_offsets[i + 1]])
                 for i, p in enumerate(self.parameters)]
        if self.discrete_dim:
            d = self._require_discrete(discrete_flat, "features")
            parts += [p.features(d[self._discrete_offsets[i]:self._discrete_offsets[i + 1]])
                      for i, p in enumerate(self.discrete_parameters)]
        return _concat(parts)

    @property
    def ambient_names(self) -> list:
        return ([n for p in self.parameters for n in p.ambient_names()]
                + [n for p in self.discrete_parameters for n in p.ambient_names()])

    @property
    def stein_defined(self) -> np.ndarray:
        """Per feature: is a Langevin--Stein term defined for it? ``(n_features,)`` of bool.

        ``True`` throughout for a continuous parameter. ``False`` for every discrete feature: the
        Stein identity integrates by parts against a density and its score, and a probability mass
        function has neither, so there is no z to report and reporting one anyway would be worse
        than reporting nothing. :func:`mimcs.summary.summarize` masks those rows.

        Not simply "NaN in ``stein_terms``": ``summarize`` drops any draw whose Stein row is
        non-finite, so a NaN column would discard **every** draw and silently empty the diagnostic.
        """
        parts = ([np.ones(p.n_features, bool) for p in self.parameters]
                 + [np.asarray(p.stein_defined(), bool) for p in self.discrete_parameters])
        return np.concatenate(parts) if parts else np.ones(0, bool)

    def stein_terms(self, sample_flat: Array, score_flat: Array) -> Array:
        """Per-feature Langevin--Stein terms, concatenated over the parameters (``(n_features,)``).

        ``score_flat`` is the ambient score ``grad_x log pi(x)`` (see :meth:`ambient_score`). Each
        parameter maps its own sample and score block to its features' Stein terms
        (:meth:`BaseParameter.stein_terms`); a manifold parameter projects the score internally, so
        only its tangential part matters.
        """
        parts = [
            p.stein_terms(sample_flat[self._ambient_offsets[i]:self._ambient_offsets[i + 1]],
                          score_flat[self._ambient_offsets[i]:self._ambient_offsets[i + 1]])
            for i, p in enumerate(self.parameters)]
        if self.discrete_dim:
            # Zeros, not NaN, for the discrete block: they keep the row finite so the
            # non-finite-draw filter in `summarize` still means what it says, and they are never
            # read --- `stein_defined` masks them out of the table.
            parts += [jnp.zeros((p.n_features,), float) for p in self.discrete_parameters]
        return _concat(parts)

    # --- ambient score (for the Stein diagnostic) ---

    def log_prob_flat(self, sample_flat: Array, discrete_flat: Array | None = None) -> Array:
        """The ambient log-density from a flat sample vector (no chart Jacobian)."""
        values = self.unpack_sample(sample_flat)
        if self.discrete_dim:
            values.update(self.unpack_discrete(
                self._require_discrete(discrete_flat, "log_prob_flat")))
        return self.log_prob(values)

    def ambient_score(self, sample_flat: Array, coord_score: Array | None = None,
                      chart_hyperparams: tuple | None = None,
                      chart_indices: tuple | None = None,
                      discrete_flat: Array | None = None) -> Array:
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
            return jax.grad(self.log_prob_flat)(sample_flat, discrete_flat)
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

    def unpack_discrete(self, discrete_flat: Array) -> dict:
        """Flat integer vector -> ``{name: value}`` in each discrete parameter's ambient shape."""
        out = {}
        for i, p in enumerate(self.discrete_parameters):
            lo, hi = self._discrete_offsets[i], self._discrete_offsets[i + 1]
            out[p.name] = jnp.reshape(discrete_flat[lo:hi], p.ambient_shape)
        return out

    def pack_discrete(self, discrete_dict: dict) -> Array:
        """``{name: value}`` -> the flat ``int32`` vector, in declaration order."""
        return _concat([
            jnp.reshape(jnp.asarray(discrete_dict[p.name], jnp.int32), (self._discrete_sizes[i],))
            for i, p in enumerate(self.discrete_parameters)], jnp.int32)

    def unpack_discrete_draws(self, draws) -> dict:
        """A *stack* of flat discrete vectors -> ``{name: array}``, each ``(n_draws, *shape)``.

        The discrete counterpart of :meth:`unpack_draws`, and written the same deliberate way:
        plain slicing and ``.reshape``, so numpy draws stay numpy **and integer draws stay
        integer**. Casting labels to float here would be a quiet way to lose the one property
        that makes them labels.
        """
        n = len(draws)
        out = {}
        for i, p in enumerate(self.discrete_parameters):
            lo, hi = int(self._discrete_offsets[i]), int(self._discrete_offsets[i + 1])
            out[p.name] = draws[:, lo:hi].reshape(n, *p.ambient_shape)
        return out

    def default_discrete(self) -> Array:
        """A valid flat discrete point: each parameter's lower bound.

        The discrete counterpart of :meth:`default_sample`, and valid for any support by
        construction. It is a *poor* place to start a label --- every observation in the first
        category --- so :class:`~mimcs.samplers.DiscreteMetropolisWithinGibbs` randomizes it in
        ``initialize()``.
        """
        return _concat([jnp.reshape(p.default_value(), (self._discrete_sizes[i],))
                        for i, p in enumerate(self.discrete_parameters)], jnp.int32)

    @property
    def discrete_lower(self) -> Array:
        """Elementwise inclusive lower bound of the whole flat discrete block, ``(discrete_dim,)``."""
        return self._discrete_bounds()[0]

    @property
    def discrete_upper(self) -> Array:
        """Elementwise inclusive upper bound of the whole flat discrete block, ``(discrete_dim,)``."""
        return self._discrete_bounds()[1]

    def _discrete_bounds(self):
        return (_concat([jnp.reshape(p.lower, (-1,)) for p in self.discrete_parameters],
                        jnp.int32),
                _concat([jnp.reshape(p.upper, (-1,)) for p in self.discrete_parameters],
                        jnp.int32))

    def pack_sample(self, sample_dict: dict) -> Array:
        return _concat([jnp.reshape(sample_dict[p.name], (self._ambient_sizes[i],))
                        for i, p in enumerate(self.parameters)])

    def _parents_of(self, p: BaseParameter, sample_dict: dict) -> dict:
        return {name: sample_dict[name] for name in p.parents}

    def unpack_coordinate(self, coord_flat: Array, chart_hyperparams: tuple,
                          chart_indices: tuple, discrete_flat: Array | None = None) -> dict:
        """Coordinate vector -> ``{name: ambient_value}``, in topological order.

        With ``discrete_flat`` given, the discrete parameters' values are merged into the result,
        so a log-density component --- a pure function of exactly this dict --- sees them as
        ordinary entries and needed no change to gain them. They are *not* threaded to the charts:
        stage 1 forbids a discrete parameter from being a chart's parent, so no
        ``from_coordinate`` can consult one.
        """
        samples: dict = {}
        if discrete_flat is not None:
            samples.update(self.unpack_discrete(discrete_flat))
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
        return _concat(parts)

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
                               chart_indices: tuple,
                               discrete_flat: Array | None = None) -> Array:
        """Coordinate-space target: ``log pi(from_coordinate(q)) + log|J|``.

        This is the density the sampler's Markov kernel is reversible with respect
        to, and what ``state.log_prob`` stores. Differentiable in ``coord_flat``; the discrete
        block enters as a constant, which is exactly its status inside one HMC trajectory and
        inside one Gibbs proposal.

        ``discrete_flat`` is **required** when the model has discrete parameters --- see
        :meth:`_require_discrete` for why it is not quietly defaulted.
        """
        discrete_flat = self._require_discrete(discrete_flat, "log_prob_at_coordinate")
        sample_dict = self.unpack_coordinate(coord_flat, chart_hyperparams, chart_indices,
                                             discrete_flat)
        logp = self.log_prob(sample_dict)
        logj = self.total_log_jacobian(coord_flat, sample_dict, chart_hyperparams,
                                       chart_indices)
        return logp + logj
