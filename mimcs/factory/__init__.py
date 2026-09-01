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
Metropolis-within-Gibbs sweep uses (``spec.discrete_proposal``). See
``docs/reference/sampler_factory.md`` for the field-by-field reference.
"""

from __future__ import annotations

from .evidence import Evidence, Diagnostics, normalize
from .spec import SamplerSpec, BlockSpec, default_spec
from .rules import (Proposal, analyze_proposals, arbitrate, normalize_block_override,
                    RULES, REFINEMENT_RULES)
from .._logging import get_logger

log = get_logger(__name__)


def analyze(model, *results, blocks=None, recompute_gradients: bool = True) -> SamplerSpec:
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
    """
    evidence = normalize(model, *results, recompute_gradients=recompute_gradients)
    log.debug("analyzing a model with %d parameter(s) (coord_dim %d)%s against %d result(s)",
              len(model.parameters), model.coord_dim,
              (f" and {len(model.discrete_parameters)} discrete one(s) "
               f"(discrete_dim {model.discrete_dim})") if model.discrete_dim else "",
              len(results))
    spec = default_spec(model, evidence)
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


def make_sampler(model, *results, seed: int = 0, init=None, buffer_size=None, blocks=None,
                 recompute_gradients: bool = True):
    """Build a sampler for ``model``, tailored to ``*results`` (the one-liner).

    ``blocks`` and ``recompute_gradients`` are forwarded to :func:`analyze` --- the first to
    override the block partition, the second to skip recomputing a prior sampler's unsaved
    gradients on an expensive model.
    """
    return analyze(model, *results, blocks=blocks,
                   recompute_gradients=recompute_gradients).build(
                       seed=seed, init=init, buffer_size=buffer_size)


__all__ = [
    "make_sampler", "analyze", "default_spec", "normalize",
    "SamplerSpec", "BlockSpec", "Evidence", "Diagnostics", "Proposal",
]
