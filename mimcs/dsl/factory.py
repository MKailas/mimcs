"""``ModelFactory``: a parsed DSL program -> a :class:`~mimcs.model.Model`, given ``data``.

A program declares **one or more** ``model`` blocks. A bare ``model { }`` is the component
``"target"``; ``model prior { }`` is the component ``"prior"``. Each becomes its own closure in
``Model.log_prob_fns``, which is what lets a component-aware sampler treat them differently (a
multi-rate integrator kicking a cheap prior more often than an expensive likelihood); their sum
is the joint density, so splitting a model changes nothing about what it *is*. A ``functions``
block compiles to a call table shared by every closure. ``generated quantities`` is still
parse-accepted and ignored --- with a **warning**, since a program using one compiles and samples
perfectly well while quietly producing none of the quantities it asks for.

Compiling against a dataset is two steps, mirroring the sampler factory: ``analyze(data)``
produces the inspectable :class:`~mimcs.dsl.spec.ModelSpec` --- which binds the data, sizes the
constants and labels each component cheap or expensive --- and ``spec.build()`` lowers it to a
``Model``. ``build(data)`` is the one-liner for both.
"""

from __future__ import annotations

import math

import jax.numpy as jnp

from . import ast
from . import semantics
from .cost import COSTS, classify_components, constant_size
from .errors import DslError, log_compile_error
from .interpreter import (build_component_closure, build_scan_component_closures,
                          run_eager)
from .semantics import plan_parameters
from .spec import ComponentSpec, ModelSpec, ParameterSpec
from ..model import BaseDiscreteParameter, Model, ScanComponent, PARAMETER_KINDS
from .._logging import get_logger

log = get_logger(__name__)

#: A model component may not take this name: :class:`~mimcs.hmc.hamiltonians.ModelPotential`
#: keys its gradient cache by ``f"V_{component}"``, which would collide with
#: :class:`~mimcs.hmc.hamiltonians.JacobianPotential`'s ``"V_jacobian"`` --- silently, since the
#: cache is a plain dict.
_RESERVED_COMPONENTS = frozenset({"jacobian"})


class ModelFactory:
    """A compiled DSL program, ready to be instantiated against a dataset.

    At least one ``model`` block is required; several are allowed when named
    (``model prior { }``), and a bare one is the component ``"target"``. ``generated
    quantities`` is accepted but ignored.
    """

    def __init__(self, program: ast.Program, source: str):
        self.source = source
        self._by_kind: dict[str, ast.Block] = {}
        self._models: dict[str, ast.Block] = {}     # component name -> block, in source order
        self._functions: dict = {}                  # function name -> ast.FuncDef
        try:
            self._collect_blocks(program)
            self._check_statements()
        except DslError as e:
            raise log_compile_error(e if e.source is not None else e.with_source(source))
        log.debug("model factory ready: blocks %s, component(s) %s, function(s) %s",
                  sorted(self._by_kind),
                  {n: len(b.body) for n, b in self._models.items()},
                  sorted(self._functions) or "(none)")

    def _collect_blocks(self, program: ast.Program) -> None:
        """Sort the program's blocks: the model components, the function table, the rest."""
        source = self.source
        for b in program.blocks:
            if b.kind == "model":
                name = b.name or "target"
                if name in _RESERVED_COMPONENTS:
                    raise DslError(
                        f"{name!r} is a reserved model-component name (it would collide with "
                        f"the chart-Jacobian potential)", b.span, source)
                if name in self._models:
                    raise DslError(
                        "duplicate `model` block" if b.name is None
                        else f"duplicate model component {name!r}", b.span, source)
                self._models[name] = b
            elif b.kind == "functions":
                if self._functions:
                    raise DslError("duplicate `functions` block", b.span, source)
                self._functions = semantics.check_functions(b.body)
            elif b.kind == "generated_quantities":
                # WARNING, not DEBUG: the block parses, so the program compiles and samples --- but
                # nothing it computes ever appears in the draws. At DEBUG (the old level, invisible
                # by default) a user porting a Stan program would simply find their generated
                # quantities missing, with no indication why.
                log.warning("`generated quantities` is parsed but NOT implemented: the block is "
                            "ignored and none of its quantities will appear in the draws. Compute "
                            "them from the returned samples instead.")
                continue
            else:
                if b.kind in self._by_kind:
                    raise DslError(f"duplicate `{b.kind}` block", b.span, source)
                self._by_kind[b.kind] = b
        if not self._models:
            raise DslError("a `model` block is required", None, source)

    def _check_statements(self) -> None:
        """Static checks over the non-``functions`` blocks (no data needed, so they fire at
        ``compile_model``): `return` belongs to a function body, calls match their signatures,
        and a density statement in `transformed parameters` needs exactly one component to
        belong to."""
        tp = self._body("transformed_parameters")
        scans = [b for b in self._models.values() if b.scan_over]
        if len(self._models) > 1:
            semantics.check_no_density(tp, "`transformed parameters`")
        elif scans:
            # A single scan component is the one case the rule above does not already cover, and
            # it needs covering for a different reason: `transformed parameters` runs **once**,
            # outside the scan, so a density statement there would contribute once while every
            # statement beside it in the component contributes per element. Counting one thing
            # once and its neighbour n times is not something a reader would expect to have to
            # know, so it is rejected rather than documented.
            semantics.check_no_density(
                tp, "`transformed parameters` of a program with a scan component (it runs once, "
                    "outside the scan, while the component's own statements run per element)")
        for kind in ("transformed_data", "transformed_parameters"):
            semantics.check_no_return(self._body(kind), f"`{kind.replace('_', ' ')}`")
            semantics.check_call_arity(self._body(kind), self._functions)
            semantics.check_loop_forms(self._body(kind), self._functions)
        declared = {d.name for kind in ("data", "transformed_data", "parameters",
                                        "transformed_parameters")
                    for d in self._body(kind) if isinstance(d, ast.VarDecl)}
        for name, block in self._models.items():
            semantics.check_no_return(block.body, f"the `{name}` model component")
            semantics.check_call_arity(block.body, self._functions)
            semantics.check_loop_forms(block.body, self._functions)
            semantics.check_target_names(block.body, name)
            if block.scan_over:
                semantics.check_scan_component(block, declared)

    def _body(self, kind: str) -> list:
        block = self._by_kind.get(kind)
        return block.body if block is not None else []

    def _reject_manifold_types(self, kind: str) -> None:
        """A ``parameter_only`` type names a chart, which only a parameter has.

        The declaration grammar is shared across blocks, so the restriction is enforced here
        rather than in the parser. Which types it covers is the registry's to say
        (:data:`~mimcs.model.PARAMETER_KINDS`) --- ``unit_vector`` is the only one so far.
        """
        for decl in self._body(kind):
            if not isinstance(decl, ast.VarDecl):
                continue
            declared = PARAMETER_KINDS.get(decl.base_type)
            if declared is not None and declared.parameter_only:
                raise DslError(
                    f"`{declared.name}` may only be declared in the `parameters` block, not in "
                    f"`{kind}`", decl.span, self.source)

    def _compile_error(self, e: DslError) -> DslError:
        """Attach this program's source to an error on its way out, and log it once."""
        return log_compile_error(e if e.source is not None else e.with_source(self.source))

    def analyze(self, data: dict | None = None, *, classify=None) -> ModelSpec:
        """Bind ``data`` and produce the (mutable) :class:`~mimcs.dsl.spec.ModelSpec`.

        This is the data-dependent half of compiling: the ``data`` declarations are bound,
        ``transformed data`` is run, the constants are sized, and each ``model`` component is
        labelled cheap or expensive by :func:`mimcs.dsl.cost.classify_components`. The spec can
        then be inspected and mutated --- component costs, parameter chart options --- before
        :meth:`~mimcs.dsl.spec.ModelSpec.build` lowers it to a :class:`~mimcs.model.Model`.

        ``classify`` replaces the cost rule wholesale: any callable taking the spec and setting
        each component's ``cost`` (see :mod:`mimcs.dsl.cost` for the contract).
        """
        try:
            spec = default_model_spec(self, data)
            (classify or classify_components)(spec)
        except DslError as e:
            raise self._compile_error(e)
        log.info("model analysis: %d component(s) [%s]; %d of %d constant(s) large %s, "
                 "%d of %d parameter(s) large %s",
                 len(spec.components),
                 ", ".join(f"{c.name}[{c.cost}]" for c in spec.components),
                 len(spec.large_constants), len(spec.constant_sizes),
                 sorted(spec.large_constants) or "(none)",
                 len(spec.large_parameters), len(spec.parameter_sizes),
                 sorted(spec.large_parameters) or "(none)")
        for line in spec.rationale:
            log.debug("model spec: %s", line)
        return spec

    def build(self, data: dict | None = None) -> Model:
        """Compile against ``data`` --- ``analyze(data).build()``, the one-liner."""
        return self.analyze(data).build()


def default_model_spec(factory: ModelFactory, data: dict | None = None) -> ModelSpec:
    """The baseline spec: the data bound, the constants sized, every decision left at its
    default (every component expensive, every chart option unset) for the rule to fill in."""
    data = dict(data or {})
    log.debug("analyzing the model with data %s", sorted(data) or "(none)")
    for kind in ("data", "transformed_data", "transformed_parameters"):
        factory._reject_manifold_types(kind)

    constants: dict = {}
    for decl in factory._body("data"):
        if decl.name not in data:
            raise DslError(f"missing data {decl.name!r}", decl.span, factory.source)
        constants[decl.name] = data[decl.name]
    constants = run_eager(factory._body("transformed_data"), constants, factory._functions)

    # The eager environment holds more than the program's constants: a `for` in
    # `transformed data` leaves its loop variable bound (the interpreter never unbinds it), and
    # so would any undeclared assignment. Keep only what was *declared*, so `constant_sizes`
    # reads as an honest inventory. Cosmetic for the rule itself --- a loop counter is a scalar
    # it could never call large.
    td = factory._body("transformed_data")
    declared = ({d.name for d in factory._body("data")}
                | {s.name for s in semantics.iter_stmts(td) if isinstance(s, ast.VarDecl)})
    sizes, unmeasurable = {}, []
    for name, value in constants.items():
        if name not in declared:
            continue
        size = constant_size(value)
        if size is None:
            unmeasurable.append(name)
        else:
            sizes[name] = size
    if unmeasurable:
        log.debug("constants of unmeasurable size, left out of the cost rule: %s",
                  sorted(unmeasurable))
    log.debug("constant sizes: %s", sizes)

    # A component runs `transformed parameters` before its own body, so its reads are the union
    # of both -- which is why a large constant read there makes every component expensive.
    tp = factory._body("transformed_parameters")
    # A scan component's scanned arrays appear only in its *header*, which `read_names` (a walk
    # over statements) cannot see. Left out, a per-observation likelihood over a large data array
    # would read as touching nothing large and be labelled **cheap** --- the cost rule's answer
    # would be wrong in the one direction that matters, since the whole point of the label is to
    # find the expensive component.
    components = [ComponentSpec(name, reads=(semantics.read_names(tp + block.body)
                                             | frozenset(block.scan_over or ())))
                  for name, block in factory._models.items()]
    parameters = [ParameterSpec(d.name, kind=d.base_type)
                  for d in factory._body("parameters")]

    # Ambient dimension of each declared parameter, for the cost rule (mimcs/dsl/cost.py) --- a
    # prior touches every parameter it names, so the rule needs to know how big those are too, not
    # just the constants. Planning here (bounds unresolved to chart options) is safe: a bound is a
    # pure expression over constants/other parameter names (semantics._resolve_bound), so it never
    # needs a built parameter to evaluate, and neither centered/adaptive affects ambient shape.
    planned = semantics.plan_parameters(factory._body("parameters"), constants)
    parameter_sizes = {p.name: int(math.prod(p.ambient_shape)) for p in planned}

    return ModelSpec(factory=factory, data=data, constants=constants, constant_sizes=sizes,
                     parameter_sizes=parameter_sizes,
                     shared_reads=semantics.read_names(tp), components=components,
                     parameters=parameters)


def _scan_length(name: str, block, shapes: dict) -> int:
    """The common leading dimension of a scan component's scanned arrays.

    Checked here rather than at ``compile_model``: a `data` array's shape is only known once the
    data is bound, which is the same reason the cost rule lives at this stage. Mismatched lengths
    are an error rather than a broadcast, because the component's element count *is* its length
    and two different answers would make "element `i`" mean two different things.
    """
    lengths = {}
    for xname in block.scan_over:
        shape = shapes.get(xname)
        if not shape:
            raise DslError(
                f"the `{name}` component scans over {xname!r}, which is a scalar. `scan` needs "
                f"arrays: the leading dimension is the number of elements the component has.",
                block.span)
        lengths[xname] = int(shape[0])
    if len(set(lengths.values())) != 1:
        raise DslError(
            f"the `{name}` component scans over arrays of different lengths "
            f"({', '.join(f'{k}: {v}' for k, v in lengths.items())}). They are stepped through "
            f"together, so they must agree on the leading dimension.", block.span)
    return next(iter(lengths.values()))


def build_model(spec: ModelSpec) -> Model:
    """Lower a :class:`~mimcs.dsl.spec.ModelSpec` to a :class:`~mimcs.model.Model`."""
    factory = spec.factory
    declared = list(factory._models)
    named = [c.name for c in spec.components]
    if sorted(named) != sorted(declared):
        raise ValueError(
            f"spec.components {named} does not match the program's `model` blocks {declared}: "
            "set `cost` on the components that are there --- adding or removing one would "
            "silently change the joint density")
    unknown = {c.name: c.cost for c in spec.components if c.cost not in COSTS}
    if unknown:
        raise ValueError(f"unknown component cost(s) {unknown} (expected one of {list(COSTS)})")

    try:
        charts = {p.name: p.chart_options() for p in spec.parameters}
        planned = plan_parameters(factory._body("parameters"), spec.constants, charts)
        # `plan_parameters` returns whatever the registry built, which since `int` became a real
        # type is a mix of continuous parameters and discrete ones. `Model` keeps the two in
        # separate lists with separate flat layouts, so split them here -- declaration order is
        # preserved within each, which is what fixes the coordinate and the discrete layout.
        params = [p for p in planned if not isinstance(p, BaseDiscreteParameter)]
        discrete = [p for p in planned if isinstance(p, BaseDiscreteParameter)]
        param_names = [p.name for p in planned]   # every name a component may read
        # `transformed parameters` depends on the parameters, so it cannot be precomputed:
        # its statements are prepended to *every* component, which is how each component
        # sees the transformed values. (Closures cannot pass values to one another, so there
        # is no way to share the work; the density statements that would then be counted
        # once per component are rejected by ``_check_statements``.)
        tp = factory._body("transformed_parameters")
        shapes = {**{n: jnp.shape(v) for n, v in spec.constants.items()},
                  **{pp.name: pp.ambient_shape for pp in planned}}
        components, scans = {}, {}
        for name, block in factory._models.items():
            if not block.scan_over:
                components[name] = build_component_closure(
                    tp + block.body, param_names, spec.constants, factory._functions, name)
                continue
            length = _scan_length(name, block, shapes)
            components[name], element_fn = build_scan_component_closures(
                tp, block.body, block.scan_over, param_names, spec.constants,
                factory._functions, name)
            scans[name] = ScanComponent(element_fn=element_fn, scanned=tuple(block.scan_over),
                                        length=length)
    except DslError as e:
        raise factory._compile_error(e)

    model = Model(params, components, cheap_components=spec.cheap_components,
                  discrete_parameters=discrete, scan_components=scans,
                  component_reads={c.name: c.reads for c in spec.components})
    log.debug("compiled model: %d parameter(s) %s, coord_dim %d, ambient_dim %d, "
              "%d discrete (dim %d), component(s) %s, cheap %s", len(params),
              [p.name for p in params], model.coord_dim, model.ambient_dim,
              len(discrete), model.discrete_dim, list(components),
              sorted(model.cheap_components) or "(none)")
    return model
