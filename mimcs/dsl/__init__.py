"""A Stan-like domain-specific language for specifying models.

A DSL program is compiled to the existing :class:`~mimcs.model.Model` (parameters +
log-probability components). See ``docs/design/08_model_dsl.md`` for the language and the
compiler design. Public entry point::

    from mimcs import compile_model
    model = compile_model(source, data={...})          # one-shot -> Model
    factory = compile_model(source); model = factory.build(data)   # reusable

A ``functions`` block defines pure, value-returning functions callable anywhere in the program,
and each ``model <name>`` block becomes its own entry of ``Model.log_prob_fns`` (a bare
``model`` is ``"target"``) --- the separation a component-aware sampler needs.

Two things worth knowing beyond the language itself:

* **The two-step form has a configuration seam.** ``factory.analyze(data)`` returns a mutable
  :class:`~mimcs.dsl.ModelSpec` before ``build()`` --- it labels each component cheap or expensive
  by a data-size rule (which the multi-rate integrator reads) and carries the per-parameter chart
  options the grammar cannot express (``centered``, ``adaptive``).
* **``scan`` and ``fori_loop`` do not unroll**, which is what lets state-space models compile at
  all. Their lengths must be compile-time constants, and that is precisely what keeps them
  reverse-mode differentiable; ``fori_loop`` is written *as* a ``scan`` so this holds by
  construction, and its range is **inclusive**, unlike ``jax.lax.fori_loop``. A plain ``for``
  still unrolls.
"""

from __future__ import annotations

from .errors import DslError, log_compile_error
from .factory import ModelFactory
from .parser import parse
from .spec import ComponentSpec, ModelSpec, ParameterSpec
from .._logging import get_logger

log = get_logger(__name__)


def compile_model(source: str, data: dict | None = None):
    """Compile DSL ``source``.

    Returns a :class:`~mimcs.model.Model` when ``data`` is given (or the program needs none),
    otherwise a :class:`ModelFactory` whose ``.build(data)`` produces the model.

    A compile error is logged at ERROR (with its source location and caret line) before the
    :class:`DslError` is raised, so a failure is on the record even where the caller swallows it.
    """
    try:
        factory = ModelFactory(parse(source), source)
        if data is not None:
            return factory.build(data)
    except DslError as e:
        raise log_compile_error(e)      # already reported by the failing stage; a no-op then
    return factory


__all__ = ["compile_model", "ModelFactory", "ModelSpec", "ComponentSpec", "ParameterSpec",
           "DslError"]
