"""The model *prototype*: :class:`ModelSpec`, :class:`ComponentSpec`, :class:`ParameterSpec`.

A ``ModelSpec`` carries every decision that turns a compiled DSL program plus a dataset into a
:class:`~mimcs.model.Model`, in attributes a user can inspect and mutate before building --- the
configuration seam of the DSL, and the sibling of the sampler factory's
:class:`~mimcs.factory.spec.SamplerSpec`::

    data ──▶ factory.analyze(data) ──▶ ModelSpec ──▶ .build() ──▶ Model
                  (the rule)          (prototype)    (lower)

    spec = compile_model(source).analyze(data)
    spec.component("likelihood").cost = "cheap"     # override the cost rule
    spec.parameter("beta").centered = True          # a flag the grammar cannot express
    model = spec.build()

``compile_model(source, data=...)`` and ``factory.build(data)`` remain the one-liners; they are
now exactly ``analyze(data).build()``.

Two things live here that the grammar has no syntax for. **Component cost** --- which log-density
components are cheap to differentiate --- is what a multi-rate integrator needs
(:mod:`mimcs.dsl.cost` decides it by default). **Chart options** --- ``centered`` for a Euclidean
or bounded parameter, ``adaptive`` for a unit vector --- are per-parameter switches that only an
adaptation mixin reads; they are left unset by ``analyze``, which provides the interface rather
than a heuristic.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ComponentSpec:
    """One ``model`` component and what it costs to evaluate.

    ``reads`` is every free name the component's statements read (including the transformed
    parameters prepended to it); ``touches`` is the large-constant subset of that, which is the
    whole explanation of ``cost``. Both are informational --- overriding ``cost`` by hand is the
    supported way to disagree with the rule.
    """

    name: str
    cost: str = "expensive"                # "cheap" | "expensive" (mimcs.dsl.cost.COSTS)
    reads: frozenset = frozenset()         # every free name the component's statements read
    touches: frozenset = frozenset()       # the large constants among them --- why `cost` is that
    params: dict = field(default_factory=dict)   # room for future per-component options


@dataclass
class ParameterSpec:
    """One declared parameter and the chart options the DSL grammar cannot express.

    ``centered`` (Euclidean / bounded) and ``adaptive`` (unit vector) are tri-state: ``None``
    leaves the parameter class's own default alone, which is not the same for the two --- a chart
    is un-centered by default but a unit vector's chart *is* adaptive by default. Setting the
    option that does not belong to a parameter's kind is an error at build time, naming the
    declaration that makes it inapplicable.
    """

    name: str
    kind: str = "real"                     # "real" | "int" | "unit_vector" (as declared)
    centered: bool | None = None           # euclidean/bounded chart; None = the class default
    adaptive: bool | None = None           # unit_vector chart;       None = the class default
    params: dict = field(default_factory=dict)   # extra chart kwargs, passed through verbatim

    def chart_options(self) -> dict:
        """The keyword arguments this parameter's chart should be constructed with."""
        opts = dict(self.params)
        if self.centered is not None:
            opts["centered"] = bool(self.centered)
        if self.adaptive is not None:
            opts["adaptive"] = bool(self.adaptive)
        return opts


@dataclass
class ModelSpec:
    """A mutable, inspectable prototype carrying everything needed to build a ``Model``.

    ``data`` is the dataset as it was bound and ``constants`` the environment the closures see
    (the data plus everything ``transformed data`` computed); ``constant_sizes`` is the element
    count of each measurable constant and ``parameter_sizes`` the *ambient* dimension of each
    declared parameter --- together the inventory the cost rule compares (``mimcs.dsl.cost``).
    Mutating ``data`` after ``analyze`` does nothing: to build against a different dataset,
    analyze it.
    """

    factory: object                                     # the ModelFactory this was analyzed from
    data: dict = field(default_factory=dict)
    constants: dict = field(default_factory=dict)
    constant_sizes: dict = field(default_factory=dict)  # name -> element count (measurable only)
    parameter_sizes: dict = field(default_factory=dict)  # name -> ambient dimension
    large_constants: frozenset = frozenset()
    large_parameters: frozenset = frozenset()
    shared_reads: frozenset = frozenset()               # free names of `transformed parameters`
    components: list = field(default_factory=list)      # ComponentSpec, in source order
    parameters: list = field(default_factory=list)      # ParameterSpec, in declaration order
    rationale: list = field(default_factory=list)

    def component(self, name: str) -> ComponentSpec:
        """The named component's spec (the handle for overriding its ``cost``)."""
        for c in self.components:
            if c.name == name:
                return c
        raise KeyError(f"no model component {name!r} (components: "
                       f"{[c.name for c in self.components]})")

    def parameter(self, name: str) -> ParameterSpec:
        """The named parameter's spec (the handle for its chart options)."""
        for p in self.parameters:
            if p.name == name:
                return p
        raise KeyError(f"no parameter {name!r} (parameters: "
                       f"{[p.name for p in self.parameters]})")

    @property
    def cheap_components(self) -> frozenset:
        """The components currently labelled cheap --- what :meth:`build` hands the model."""
        return frozenset(c.name for c in self.components if c.cost == "cheap")

    def build(self):
        """Instantiate the :class:`~mimcs.model.Model` this spec describes."""
        from .factory import build_model
        return build_model(self)
