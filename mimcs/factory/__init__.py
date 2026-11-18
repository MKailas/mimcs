"""Sampler factory: construct a sampler from a model and earlier sampling results.

The primary user-facing entry point for building samplers (``docs/design/09_sampler_factory.md``)::

    sampler = make_sampler(model)               # a reasonable default
    sampler = make_sampler(model, output)        # tailored to earlier results

For configuration, work with the prototype directly::

    spec = analyze(model, output)   # a SamplerSpec
    spec.blocks[0].kind = "dense"   # inspect / override any decision
    sampler = spec.build()

``*results`` may be raw samples, a testing ``SamplerOutput``, a live sampler, or a
``(samples, coordinates, gradients)`` tuple/dict --- in any combination (see
:func:`mimcs.factory.evidence.normalize`).

Build-time arguments (``seed``, ``init``, ``buffer_size``) go to ``build``; everything else is a
decision recorded on the spec, and ``spec.rationale`` says which rule set what and why. Beyond the
blocks, the spec carries the base algorithm (including the parallel-tempered ``pt_`` counterparts),
the integrator and its options, the mass adaptation, centering, the warmup-termination
criterion, and --- for a model with integer parameters --- which proposal the discrete
discrete sweep gives each integer parameter (``spec.discrete``). See
``docs/reference/sampler_factory.md`` for the field-by-field reference.
"""

from __future__ import annotations

import numpy as np

from .evidence import Evidence, Diagnostics, normalize
from .spec import SamplerSpec, BlockSpec, DiscreteSpec, default_spec
from .rules import (Proposal, analyze_proposals, arbitrate, normalize_block_override,
                    RULES, REFINEMENT_RULES)
from .._logging import get_logger

log = get_logger(__name__)


def analyze(model, *results, blocks=None, recompute_gradients: bool = True,
            param_budget_mult: int | None = None, param_budget_add: int | None = None,
            include_bare: bool | None = None) -> SamplerSpec:
    """Produce the (mutable) :class:`SamplerSpec` the factory's heuristics recommend.

    Two arbitration passes: structural rules (the block partition) first, then refinement rules
    (e.g. the metric regression) against the now-final partition. ``recompute_gradients`` (default
    ``True``) controls whether a live-sampler result whose gradients were not saved has them
    recomputed (needed for the metric-regression rule) or skipped (for a very expensive model).

    ``blocks`` overrides the block partition for the parameters it names --- the escape hatch for
    a model whose natural grouping the size heuristic gets wrong::

        analyze(model, blocks=[("mu", "sigma"), "theta"])

    Each entry is a group of parameter names that becomes one block; a bare string is shorthand
    for a one-name group. Any parameter **not** named is partitioned by the usual rule, so
    specifying part of the model is the normal case. An unknown name, a name in two groups, or an
    empty group raises :class:`ValueError` here, before any rule runs. Only the *grouping* is
    fixed: the refinement rules still choose each block's kind from the evidence, and a forced
    single-parameter block is what makes a learned metric possible where fusion would have ruled
    it out.

    The learned-metric regression's candidate pool can be set per call (``None`` keeps the module
    default in :mod:`mimcs.factory.regression`, and is not recorded):

    * ``param_budget_mult`` / ``param_budget_add`` --- a candidate metric for a block of
      ``block_dim`` coordinates may carry at most ``param_budget_mult * block_dim +
      param_budget_add`` parameters (defaults ``PARAM_BUDGET_MULT = 20``, ``PARAM_BUDGET_ADD = 1``);
    * ``include_bare`` --- offer each position-dependent form without the additive ``+ Exp()``
      floor as well as with it (default ``INCLUDE_BARE_CANDIDATES = True``). ``False`` offers
      floored forms only: a floor bounds the metric from below once the sampler leaves the
      evidence's range, at the cost of the pure funnel, whose ideal metric has none.

    They are recorded in ``spec.metric_regression``.
    """
    regression = _metric_regression_options(param_budget_mult, param_budget_add, include_bare)
    evidence = normalize(model, *results, recompute_gradients=recompute_gradients)
    log.debug("analyzing a model with %d parameter(s) (coord_dim %d)%s against %d result(s)",
              len(model.parameters), model.coord_dim,
              (f" and {len(model.discrete_parameters)} discrete one(s) "
               f"(discrete_dim {model.discrete_dim})") if model.discrete_dim else "",
              len(results))
    spec = default_spec(model, evidence)
    spec.metric_regression = regression
    if regression:
        log.info("metric regression options set by the caller: %s", regression)
    if blocks is not None:
        spec.block_override = normalize_block_override(blocks, model)
        log.info("block partition overridden by the caller: %s",
                 ", ".join("+".join(g) for g in spec.block_override))
    arbitrate(spec, analyze_proposals(spec, evidence, model, RULES))
    arbitrate(spec, analyze_proposals(spec, evidence, model, REFINEMENT_RULES))
    # One source for this summary: SamplerSpec.__str__ (a fifth hand-rolled copy of the
    # ``names[dim,kind]`` idiom is how the four that existed came about).
    log.info("analysis:\n%s", spec)
    return spec


def _metric_regression_options(param_budget_mult, param_budget_add, include_bare) -> dict:
    """``analyze``'s regression keywords, validated, without the ``None`` (default) ones."""
    out = {}
    for name, v, lo in (("param_budget_mult", param_budget_mult, 1),
                        ("param_budget_add", param_budget_add, 0)):
        if v is None:
            continue
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < lo:
            raise ValueError(f"{name} must be an integer >= {lo}, got {v!r}")
        out[name] = int(v)
    if include_bare is not None:
        if not isinstance(include_bare, (bool, np.bool_)):
            raise ValueError(f"include_bare must be True or False, got {include_bare!r}")
        out["include_bare"] = bool(include_bare)
    return out


def make_sampler(model, *results, seed: int = 0, init=None, buffer_size=None, blocks=None,
                 recompute_gradients: bool = True, param_budget_mult: int | None = None,
                 param_budget_add: int | None = None, include_bare: bool | None = None):
    """Build a sampler for ``model``, tailored to ``*results`` (the one-liner).

    ``blocks``, ``recompute_gradients`` and the regression options are forwarded to
    :func:`analyze` --- the first to override the block partition, the second to skip recomputing
    a prior sampler's unsaved gradients on an expensive model, the rest to set the learned-metric
    regression's candidate pool.
    """
    return analyze(model, *results, blocks=blocks,
                   recompute_gradients=recompute_gradients,
                   param_budget_mult=param_budget_mult, param_budget_add=param_budget_add,
                   include_bare=include_bare).build(
                       seed=seed, init=init, buffer_size=buffer_size)


__all__ = [
    "make_sampler", "analyze", "default_spec", "normalize",
    "SamplerSpec", "BlockSpec", "DiscreteSpec", "Evidence", "Diagnostics", "Proposal",
]
