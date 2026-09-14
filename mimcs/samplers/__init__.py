"""The core sampling loop, mixin composition, and the random-walk MH sampler.

:class:`BaseSampler` owns the preprocess -> kernel -> postprocess loop, the phase
(:class:`Phase`: ``WARMUP`` / ``SAMPLING``), the RNG buffer and the cooperative hook chains;
:func:`make_sampler_class` composes a concrete sampler from adaptation mixins plus a base
algorithm (``docs/design/01`` and ``02``).

The samplers *here* are :class:`RandomWalkMH` and, for the discrete half of a model,
:class:`SystematicScanMetropolisWithinGibbs` or :class:`RandomScanMetropolisWithinGibbs` (the two
kernel-composing scans of the :class:`DiscreteMetropolisWithinGibbs` family) with
:class:`StaticContinuous` (a base algorithm that moves nothing, for a discrete-only model).
The HMC family lives in :mod:`mimcs.hmc` and parallel tempering in :mod:`mimcs.pt`, both
building on the same loop.
"""

from .base import BaseSampler, Phase, make_sampler_class
from .metropolis import RandomWalkMH, MHState
from .gibbs import (DiscreteMetropolisWithinGibbs, SystematicScanMetropolisWithinGibbs,
                    RandomScanMetropolisWithinGibbs, StaticContinuous, StaticState)

__all__ = ["BaseSampler", "Phase", "make_sampler_class", "RandomWalkMH", "MHState",
           "DiscreteMetropolisWithinGibbs", "SystematicScanMetropolisWithinGibbs",
           "RandomScanMetropolisWithinGibbs", "StaticContinuous", "StaticState"]
