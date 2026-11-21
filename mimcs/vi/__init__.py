"""Variational approximations: Pathfinder (single- and multi-path), mean-field ADVI, and
Pareto-smoothed importance sampling. Used by the ``"pathfinder"`` / ``"multipath"`` / ``"advi"``
position initializers (:class:`~mimcs.adaptation.PathfinderInit`,
:class:`~mimcs.adaptation.ADVIInit`); usable on any flat log density on their own."""

from .pathfinder import (PathResult, MultiPathResult, single_path, multi_path,
                         PATHFINDER_DEFAULTS, MULTIPATH_DEFAULTS)
from .meanfield import ADVIResult, advi, ADVI_DEFAULTS
from .psis import psis, resample

__all__ = ["PathResult", "MultiPathResult", "single_path", "multi_path", "PATHFINDER_DEFAULTS",
           "MULTIPATH_DEFAULTS", "ADVIResult", "advi", "ADVI_DEFAULTS", "psis", "resample"]
