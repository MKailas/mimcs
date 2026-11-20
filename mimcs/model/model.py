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

import functools
from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp
from jax import Array

from dataclasses import dataclass

from .._logging import get_logger
from .discrete import BaseDiscreteParameter
from .parameter import BaseParameter, _index_prefixes

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


#: The kinds a derived quantity may be (:class:`GeneratedQuantities`), in the order a summary lists
#: them. Named after the DSL blocks they come from, which is also how a hand-written model should
#: read them: a *transformed* parameter is a deterministic function of the draw, a *generated*
#: quantity may also consume randomness.
GENERATED_KINDS = ("transformed", "generated")


def _key_bits(key, shape):
    """A :class:`~mimcs.rng.DrawComponent` generator whose draws are themselves **PRNG keys**.

    A raw ``uint32[2]`` is exactly a (legacy, threefry) ``jax.random`` key, so a stack of these is
    one fresh key per step. Keys and not variates are what a generated quantity needs: a
    ``poisson_rng(lam)`` takes its rate from the current draw, and JAX samples it by rejection, so
    no ``(key, shape)`` generator could produce it ahead of time.
    """
    return jax.random.bits(key, shape, jnp.uint32)


@dataclass(frozen=True)
class GeneratedQuantities:
    """The **derived quantities** of a model: values computed from a draw, saved alongside it.

    ``fn(values, key)`` takes the same ``{name: ambient_value}`` dict a log-density component does
    (discrete parameters included) and returns ``{name: array}``, one entry per name in ``kinds``.
    It runs once per *retained* draw, after the kernel --- never inside a trajectory --- so it
    costs one evaluation per saved sample whatever the sampler, and it may be arbitrarily
    expensive without slowing the chain down.

    ``kinds`` maps each output name, in report order, to ``"transformed"`` or ``"generated"``; the
    two differ only in how they are labelled and which ``get_samples`` keyword drops them.

    ``rng``: does ``fn`` consume randomness? If so it receives a fresh PRNG ``key`` per draw ---
    delivered through the sampler's own ``rng_draw`` like every other random number
    (``docs/design/03``) --- and otherwise ``None``.
    """

    fn: Callable                 # (values: dict, key | None) -> {name: array}
    kinds: dict                  # {name: "transformed" | "generated"}, in report order
    rng: bool = False

    def __post_init__(self):
        bad = {n: k for n, k in self.kinds.items() if k not in GENERATED_KINDS}
        if bad:
            raise ValueError(f"unknown generated-quantity kind(s) {bad}; expected one of "
                             f"{list(GENERATED_KINDS)}")


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
        generated: the model's derived quantities (:class:`GeneratedQuantities`) --- what a DSL
            program's ``transformed parameters`` and ``generated quantities`` declare. A sampler
            evaluates them once per retained draw and :meth:`~mimcs.samplers.BaseSampler.get_samples`
            returns them beside the parameters. They never enter the density.
    """

    def __init__(self, parameters: list[BaseParameter], log_prob_fns: dict[str, LogProbFn],
                 *, cheap_components=(), discrete_parameters=(), scan_components=None,
                 component_reads=None, jump_operators=None,
                 generated: GeneratedQuantities | None = None):
        self.parameters = list(parameters)
        self.discrete_parameters = list(discrete_parameters)
        self.log_prob_fns = dict(log_prob_fns)
        self.cheap_components = frozenset(cheap_components)
        self.scan_components = dict(scan_components or {})
        self.component_reads = dict(component_reads or {})
        self.jump_operators = dict(jump_operators or {})
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
        self._validate_jumps()
        self._order = self._topological_order()
        self.generated = generated if (generated is not None and generated.kinds) else None
        self._validate_generated()

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

    #: Jump-operator outputs are restricted to these parameter kinds --- the ones whose
    #: ``to_coordinate`` is a genuine bijection on its domain. See :meth:`_validate_jumps`.
    _JUMP_OUTPUT_KINDS = ("EuclideanParameter", "BoundedParameter")

    def _validate_jumps(self) -> None:
        """What a jump operator may rewrite, and why the list is short.

        Three refusals, each because the corresponding runtime failure is **silent** --- a finite
        density, a plausible trace and a wrong posterior --- so a static error is the only thing
        that would ever report it.

        **The output must be a continuous parameter of this model.** A discrete output would move
        under counting measure with no Jacobian and interact with the sweep's own carry; it is
        deferred rather than wrong.

        **The output's chart must not project.** This is the nastiest case in the design.
        :class:`~mimcs.model.UnitVectorParameter`'s ``to_coordinate`` accepts a vector that is not
        on the sphere and returns a coordinate whose ``from_coordinate`` is a *different* unit
        vector; the same goes for the simplex and, through the shared stick breaking, a
        doubly-bounded :class:`~mimcs.model.OrderedParameter`. A jump writing such an output is
        silently **projected back onto the manifold**, which violates the involution by exactly the
        projection error and reports nothing. A bounded output at least produces ``NaN`` out of
        domain, and a ``NaN`` rejects under Metropolis and masks to ``-inf`` under exact Gibbs ---
        loud, or at worst inert. So the allowed kinds are the ones that are bijections on their
        domain.

        **The output may not be a chart parent.** :meth:`unpack_coordinate` threads parents
        topologically, so moving a parent changes a *child's* ambient value while the child's
        coordinate block stands still --- the map would silently move a parameter the operator
        never named. (The determinant is unharmed: the composed coordinate map stays block
        triangular, so it is the semantics that are wrong, not the arithmetic.)
        """
        by_name = {p.name: p for p in self.parameters}
        for key, op in self.jump_operators.items():
            if key not in self._discrete_name_to_idx:
                raise ValueError(
                    f"jump_operators names '{key}', which is not a discrete parameter of this "
                    f"model; its discrete parameters are "
                    f"{[p.name for p in self.discrete_parameters]}")
            if getattr(op, "parameter", key) != key:
                raise ValueError(
                    f"jump_operators['{key}'] is an operator for '{op.parameter}'. The key and "
                    f"the operator's own parameter must agree --- the sweep addresses operators "
                    f"by the parameter they attach to.")
            for name in op.outputs:
                if name == key:
                    raise ValueError(
                        f"jump operator for '{key}' lists '{key}' itself among its outputs. The "
                        f"sweep moves that coordinate; an operator names only what moves "
                        f"*alongside* it. (Rewriting other coordinates of the same parameter --- a "
                        f"swap or relabel move --- is deferred to blocked updates.)")
                if name in self._discrete_name_to_idx:
                    # Another discrete parameter: allowed. It moves under counting measure, so it
                    # adds no Jacobian term, and the map's returned labels are validated (integral,
                    # in support) at run time, where an invalid one rejects the proposal.
                    continue
                if name not in by_name:
                    raise ValueError(
                        f"jump operator for '{key}' rewrites '{name}', which is not a parameter "
                        f"of this model; its continuous parameters are {list(by_name)}")
                kind = type(by_name[name]).__name__
                if kind not in self._JUMP_OUTPUT_KINDS:
                    raise NotImplementedError(
                        f"jump operator for '{key}' rewrites '{name}', which is a {kind}. A jump "
                        f"output must be one of {list(self._JUMP_OUTPUT_KINDS)}, because those "
                        f"are the charts whose `to_coordinate` is a bijection on its domain. A "
                        f"chart that *projects* --- unit_vector, simplex, a doubly-bounded "
                        f"ordered, and the matrix types --- would silently pull an off-manifold "
                        f"proposal back onto the manifold, breaking detailed balance by exactly "
                        f"the projection error while reporting a finite density and nothing else.")
                children = [q.name for q in self.parameters
                            if name in getattr(q, "parents", ())]
                if children:
                    raise NotImplementedError(
                        f"jump operator for '{key}' rewrites '{name}', which is a chart parent of "
                        f"{children}. Moving a parent changes its children's ambient values while "
                        f"their coordinates stand still, so the map would move parameters it "
                        f"never named --- silently. Name them as outputs of a map that does not "
                        f"depend on them, or drop the parent link.")
            if not op.volume_preserving and all(n in self._discrete_name_to_idx
                                                for n in op.outputs):
                raise ValueError(
                    f"jump operator for '{key}' is declared as scaling, but it rewrites only "
                    f"discrete parameters {list(op.outputs)}. Labels move under counting measure, "
                    f"so there is no volume to scale and no Jacobian to take --- declare it volume "
                    f"preserving (drop `scales`).")

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

    def _validate_generated(self) -> None:
        """A derived quantity is reported in the same dict as the parameters, keyed by name --- so a
        name shared with a parameter would silently overwrite one of the two."""
        if self.generated is None:
            return
        clash = sorted(set(self.generated.kinds)
                       & (set(self._name_to_idx) | set(self._discrete_name_to_idx)))
        if clash:
            raise ValueError(
                f"derived quantity name(s) {clash} are also parameter names; both are reported "
                f"in the same draws dict, so one would overwrite the other")

    # --- derived quantities (transformed parameters / generated quantities) ---

    @property
    def has_generated(self) -> bool:
        return self.generated is not None

    @property
    def generated_names(self) -> list:
        return list(self.generated.kinds) if self.generated is not None else []

    def generated_kind(self, name: str) -> str:
        """``"transformed"`` or ``"generated"``."""
        return self.generated.kinds[name]

    @property
    def generated_draw_components(self) -> list:
        """The randomness the derived quantities need, as draw components for the sampler's buffer.

        One PRNG key per step when :attr:`GeneratedQuantities.rng` is set, none otherwise. The
        sampler *appends* it to its own components, which is stream-neutral: the buffer derives
        component subkeys by ``jax.random.split(key, n)``, whose first ``n`` keys do not depend on
        ``n`` (threefry, partitionable), so the kernel's draws are bit-identical with or without it.
        """
        if self.generated is None or not self.generated.rng:
            return []
        from ..rng import DrawComponent       # local: mimcs.rng is a sampler-side module
        return [DrawComponent("generated_key", (2,), _key_bits, jnp.uint32)]

    def _generated_values(self, sample_flat, discrete_flat) -> dict:
        values = self.unpack_sample(sample_flat)
        if self.discrete_dim:
            values.update(self.unpack_discrete(
                self._require_discrete(discrete_flat, "generated quantities")))
        return values

    @functools.cached_property
    def _generated_layout(self) -> dict:
        """``{name: (block, shape, lo, hi)}`` --- where each output sits in the flat float or int
        block. Found by **tracing** ``fn`` once (``jax.eval_shape``) at the default point, so a
        hand-written model never has to declare shapes, and a trace error surfaces here --- which a
        sampler forces at construction, not after warmup.

        Integer (and boolean) outputs go to an ``int32`` block of their own, floats to a float
        block: the sample/discrete split again, for the same reason --- a count cast to float has
        lost the property that made it a count.
        """
        gen = self.generated
        key = jnp.zeros((2,), jnp.uint32) if gen.rng else None
        values = self._generated_values(
            self.default_sample(), self.default_discrete() if self.discrete_dim else None)
        out = jax.eval_shape(gen.fn, values, key)
        missing = [n for n in gen.kinds if n not in out]
        if missing:
            raise ValueError(f"the generated-quantities function did not return {missing}")
        layout, offs = {}, {"float": 0, "int": 0}
        for name in gen.kinds:
            leaf = out[name]
            if leaf is None or not hasattr(leaf, "shape"):
                raise TypeError(f"generated quantity {name!r} is not an array (got {leaf!r})")
            block = ("int" if jnp.issubdtype(leaf.dtype, jnp.integer)
                     or jnp.issubdtype(leaf.dtype, jnp.bool_) else "float")
            size = int(np.prod(leaf.shape)) if leaf.shape else 1
            layout[name] = (block, tuple(leaf.shape), offs[block], offs[block] + size)
            offs[block] += size
        log.debug("generated quantities: %s", {n: (b, sh) for n, (b, sh, _, _) in layout.items()})
        return {"entries": layout, "float_dim": offs["float"], "int_dim": offs["int"]}

    @property
    def generated_shapes(self) -> dict:
        return {n: sh for n, (_, sh, _, _) in self._generated_layout["entries"].items()}

    @property
    def generated_float_dim(self) -> int:
        return self._generated_layout["float_dim"] if self.generated is not None else 0

    @property
    def generated_int_dim(self) -> int:
        return self._generated_layout["int_dim"] if self.generated is not None else 0

    def generated_flat(self, sample_flat: Array, discrete_flat: Array | None = None,
                       key: Array | None = None) -> tuple:
        """Evaluate the derived quantities of one draw: ``(float_flat, int_flat)``. Jittable."""
        entries = self._generated_layout["entries"]
        out = self.generated.fn(self._generated_values(sample_flat, discrete_flat), key)
        fl = [jnp.reshape(jnp.asarray(out[n], float), (-1,))
              for n, (b, *_) in entries.items() if b == "float"]
        it = [jnp.reshape(jnp.asarray(out[n]).astype(jnp.int32), (-1,))
              for n, (b, *_) in entries.items() if b == "int"]
        return _concat(fl), _concat(it, jnp.int32)

    def unpack_generated_draws(self, float_draws, int_draws, kinds=GENERATED_KINDS) -> dict:
        """Stacks from :meth:`generated_flat` -> ``{name: (n_draws, *shape)}``, for the outputs of
        the given ``kinds``. Plain slicing, so numpy stays numpy and integers stay integers."""
        out = {}
        for name, (block, shape, lo, hi) in self._generated_layout["entries"].items():
            if self.generated.kinds[name] not in kinds:
                continue
            src = float_draws if block == "float" else int_draws
            out[name] = src[:, lo:hi].reshape(len(src), *shape)
        return out

    def generated_ambient_names(self, name: str) -> list:
        """Row labels of one derived quantity, 1-based like every other summary label."""
        return [f"{name}{s}" for s in _index_prefixes(self.generated_shapes[name])]

    # --- per-parameter row blocks (what a summary collapses) ---

    @property
    def ambient_blocks(self) -> list:
        """``(name, start, stop)`` of each parameter's rows in :attr:`ambient_names` ---
        continuous, then discrete, the order the posterior table lists them in."""
        out, k = [], 0
        for p in self.parameters + self.discrete_parameters:
            n = len(p.ambient_names())
            out.append((p.name, k, k + n))
            k += n
        return out

    @property
    def feature_blocks(self) -> list:
        """``(name, start, stop)`` of each parameter's rows in :attr:`feature_names`."""
        out, k = [], 0
        for p in self.parameters + self.discrete_parameters:
            out.append((p.name, k, k + p.n_features))
            k += p.n_features
        return out

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

    @property
    def discrete_init_low(self) -> Array:
        """Elementwise inclusive low end of each coordinate's **starting window**, ``(discrete_dim,)``.

        Equal to :attr:`discrete_lower` for a bounded parameter. An open side instead gets a small
        window next to its finite bound (or around zero), because a uniform start over the sentinel
        range would place the chain a billion units from anything plausible.
        """
        return _concat([jnp.full((p.size,), p.init_range()[0], jnp.int32)
                        for p in self.discrete_parameters], jnp.int32)

    @property
    def discrete_init_high(self) -> Array:
        """Elementwise inclusive high end of each starting window (see :attr:`discrete_init_low`)."""
        return _concat([jnp.full((p.size,), p.init_range()[1], jnp.int32)
                        for p in self.discrete_parameters], jnp.int32)

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

    # --- a partially fixed starting point ---

    def pin_fixed(self, coord_flat: Array, sample_flat: Array, fixed, chart_hyperparams: tuple,
                  chart_indices: tuple) -> tuple[Array, Array]:
        """Overlay the parameters named in ``fixed`` on a coordinate: ``(coordinate, sample)``.

        A **fixed** parameter takes its *sample* from ``sample_flat`` --- exactly, bit for bit ---
        and its coordinate from its chart under its parents' values *in the result*. Every other
        parameter keeps its slice of ``coord_flat`` and maps to a sample under those parents. The
        walk is in topological order, because a chart can depend on a parent: a fixed child of a
        free parent is re-charted under the parent's *new* value, so its coordinate moves while
        its sample stays put. (A coordinate-space mask --- overlaying each fixed value's coordinate
        as computed under the old parents --- gets exactly that case wrong: it was measured off by
        2.8 on a ``(0, sigma)`` child before this was written.)

        This is the one primitive behind a partially supplied start: :class:`~mimcs.adaptation.
        UniformInit` overlays each candidate draw with it, :meth:`complete_sample` fills in a
        partial ``init=``, and an optimizing initializer maps its free coordinates to a full one
        through it (it is pure and differentiable; ``fixed`` is static). With nothing fixed it
        returns ``coord_flat`` itself and the sample that maps to it.

        A fixed value outside its support given its (new) parents gives a non-finite coordinate;
        this does not check --- see :meth:`complete_sample` for the eager check.
        """
        fixed = frozenset(fixed)
        if not fixed:
            return coord_flat, self.coordinate_to_sample(coord_flat, chart_hyperparams,
                                                         chart_indices)
        parts = [None] * len(self.parameters)
        samples: dict = {}
        for i in self._order:
            p = self.parameters[i]
            parents = self._parents_of(p, samples)
            if p.name in fixed:
                a0, a1 = self._ambient_offsets[i], self._ambient_offsets[i + 1]
                x = jnp.reshape(sample_flat[a0:a1], p.ambient_shape)
                parts[i] = jnp.reshape(
                    p.to_coordinate(x, chart_hyperparams[i], chart_indices[i], parents),
                    (p.coord_dim,))
                samples[p.name] = x
            else:
                c = coord_flat[self._coord_offsets[i]:self._coord_offsets[i + 1]]
                parts[i] = c
                samples[p.name] = p.from_coordinate(c, chart_hyperparams[i], chart_indices[i],
                                                    parents)
        return _concat(parts), self.pack_sample(samples)

    def complete_sample(self, values: dict, base_sample: Array | None = None) -> Array:
        """A full flat sample from a **partial** ``{name: value}`` dict, checked against support.

        The named continuous parameters take their values exactly; the rest take ``base_sample``'s
        *coordinate* (default: the chart origin, i.e. :meth:`default_sample`), re-mapped under any
        named parent --- so an unnamed child of a named parent lands inside its new bounds rather
        than keeping a value that may no longer be valid. Names of discrete parameters are ignored
        here (:meth:`complete_discrete` takes that half); any other unknown name raises.

        Raises ``ValueError`` for a value of the wrong size, or one outside its support given its
        parents. Support is checked generically, by a round trip through the chart: the value must
        come back (to float tolerance) with a finite coordinate. That catches a bound violation, a
        value *on* a bound (an infinite coordinate), a unit vector off the sphere and a simplex
        that does not sum to one, without each parameter type having to say what its support is.
        Eager (it reads the check's result), so call it outside ``jit``.
        """
        known = {p.name for p in self.parameters}
        discrete = {p.name for p in self.discrete_parameters}
        unknown = sorted(set(values) - known - discrete)
        if unknown:
            raise ValueError(
                f"init names {unknown}, which are not parameters of this model; its parameters "
                f"are {sorted(known)}" + (f" and the discrete {sorted(discrete)}" if discrete
                                          else ""))
        h, idx = self.init_chart_hyperparams(), self.init_chart_indices()
        base = (self.default_sample() if base_sample is None
                else jnp.reshape(jnp.asarray(base_sample, float), (self.ambient_dim,)))
        given = [n for n in values if n in known]
        if not given:
            return base
        flat = np.array(base)
        for name in given:
            i = self._name_to_idx[name]
            v = np.asarray(values[name], dtype=flat.dtype)
            if v.size != self._ambient_sizes[i]:
                raise ValueError(
                    f"init value for {name!r} has {v.size} element(s); the parameter's ambient "
                    f"shape {self.parameters[i].ambient_shape} has {self._ambient_sizes[i]}")
            flat[self._ambient_offsets[i]:self._ambient_offsets[i + 1]] = v.reshape(-1)
        flat = jnp.asarray(flat)
        coord, sample = self.pin_fixed(self.sample_to_coordinate(base, h, idx), flat, given, h,
                                       idx)
        back = np.asarray(self.coordinate_to_sample(coord, h, idx))
        coord, want = np.asarray(coord), np.asarray(flat)
        for i in self._order:
            p = self.parameters[i]
            if p.name not in given:
                continue
            c0, c1 = self._coord_offsets[i], self._coord_offsets[i + 1]
            a0, a1 = self._ambient_offsets[i], self._ambient_offsets[i + 1]
            if not (np.all(np.isfinite(coord[c0:c1]))
                    and np.allclose(back[a0:a1], want[a0:a1], rtol=1e-4, atol=1e-6)):
                unnamed = [q for q in p.parents if q not in given]
                given_by = (f" given its parent(s) {list(p.parents)}" if p.parents else "")
                if unnamed:
                    given_by += (f" ({unnamed} not supplied, so at their base value; supply them "
                                 f"in init too if this value is meant for another)")
                raise ValueError(
                    f"init value for {p.name!r} ({type(p).__name__}) is outside its "
                    f"support{given_by}: it does not survive a round trip through the chart "
                    f"(value {want[a0:a1]}, mapped back to {back[a0:a1]}, coordinate "
                    f"{coord[c0:c1]}). A value on a bound counts as outside, since its "
                    f"coordinate is infinite.")
        return sample

    def complete_discrete(self, values: dict, base_discrete: Array | None = None) -> Array:
        """The flat label vector from a **partial** dict: the named discrete parameters (validated)
        and ``base_discrete`` (default :meth:`default_discrete`) for the rest. Continuous names are
        ignored --- :meth:`complete_sample` takes that half."""
        base = (self.default_discrete() if base_discrete is None
                else jnp.asarray(base_discrete, jnp.int32).reshape((self.discrete_dim,)))
        out = np.array(base, dtype=np.int32)
        for i, p in enumerate(self.discrete_parameters):
            if p.name in values:
                p.validate(values[p.name])
                v = np.asarray(values[p.name], dtype=np.int32).reshape(-1)
                out[self._discrete_offsets[i]:self._discrete_offsets[i + 1]] = v
        return jnp.asarray(out)

    def coordinate_mask(self, names) -> np.ndarray:
        """``(coord_dim,)`` booleans: the coordinates owned by the named continuous parameters.

        Note what it does **not** say: with a parent-dependent chart, a fixed child's coordinate
        still moves with its free parent (see :meth:`pin_fixed`). It marks *ownership* --- e.g. the
        free coordinates an optimizer may move --- not coordinates that stay constant.
        """
        names = set(names)
        mask = np.zeros((self.coord_dim,), bool)
        for i, p in enumerate(self.parameters):
            if p.name in names:
                mask[self._coord_offsets[i]:self._coord_offsets[i + 1]] = True
        return mask

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

    def log_jacobian(self, coord_flat: Array, chart_hyperparams: tuple, chart_indices: tuple,
                     names) -> Array:
        """The log-Jacobian terms of the *named* parameters only (parents unpacked from
        ``coord_flat``).

        What an optimizing initializer subtracts from :meth:`log_prob_at_coordinate` to get the
        density of the free coordinates given parameters held fixed in *sample* space: a fixed
        child of a free parent keeps its sample while its coordinate moves with the parent, so its
        own chart Jacobian is no part of that density (for a fixed parameter with fixed parents it
        is a constant, and subtracting it changes nothing).
        """
        names = set(names)
        sample_dict = self.unpack_coordinate(coord_flat, chart_hyperparams, chart_indices)
        total = jnp.zeros(())
        for i, p in enumerate(self.parameters):
            if p.name in names:
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
