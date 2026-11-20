"""Variational approximations: Pathfinder (single- and multi-path) and Pareto-smoothed importance
sampling. Used by the ``"pathfinder"`` / ``"multipath"`` position initializers
(:class:`~mimcs.adaptation.PathfinderInit`); usable on any flat log density on its own."""

from .pathfinder import (PathResult, MultiPathResult, single_path, multi_path,
                         PATHFINDER_DEFAULTS, MULTIPATH_DEFAULTS)
from .psis import psis, resample

__all__ = ["PathResult", "MultiPathResult", "single_path", "multi_path", "PATHFINDER_DEFAULTS",
           "MULTIPATH_DEFAULTS", "psis", "resample"]
