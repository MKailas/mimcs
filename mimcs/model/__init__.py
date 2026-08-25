"""Models and parameter types: what a density is written over, and what composes it.

A parameter is defined by its *charts* --- the maps between the ambient value the model is
written in and the unconstrained coordinate the sampler moves in --- and a
:class:`~mimcs.model.Model` is what composes a list of them with a decomposed log-density.
The two are one logical whole, so they live in one package: :mod:`mimcs.model.model` holds
``Model``, :mod:`mimcs.model.parameter` the ``BaseParameter`` interface, and **each parameter
type its own module** beside them: ``euclidean``, ``bounded`` (with the ``Positive`` / ``Interval``
constructors), ``unit_vector``, ``simplex``, ``ordered``, ``cholesky_cov`` and ``correlation`` (the covariance
and correlation pairs, each sharing a chart between its two views and so sharing a module). The ``_``-prefixed modules hold what several types share ---
``_centering`` (the ``centered=True`` standardization), ``_bounds``, and ``_stick_breaking``
(used by ``simplex`` *and* by a doubly-bounded ``ordered``).
:mod:`mimcs.model.registry`'s ``PARAMETER_KINDS`` is the seam a front end reads: it decides which
DSL keyword builds which class, so adding a type does not mean hunting down branches in the DSL.

The design references are ``docs/design/04_manifold_parameters.md`` (charts, the parameter
types, the density correction) and ``docs/design/05_model_interface.md`` (the ``Model``
class). The chart-method contract is documented on :class:`~mimcs.model.BaseParameter`.

Parameters may have *parents*: a parameter's chart can depend on the ambient values of other
parameters (e.g. ``x ~ Uniform(0, sigma)``, whose bound is another parameter). Because
dependencies form a DAG, the map from coordinates to samples is triangular, so the total
change-of-variables log-Jacobian remains a simple sum of per-parameter terms, each computed
with the parents held fixed (doc 04, "Parameters with parents"). ``Model`` evaluates
``from_coordinate`` in topological order, threading already-built parent samples to each child.
"""

from .parameter import BaseParameter, flat_size
from ._bounds import BoundSpec
from .euclidean import EuclideanParameter
from .bounded import BoundedParameter, PositiveParameter, IntervalParameter
from .unit_vector import UnitVectorParameter, SphereChart
from .simplex import SimplexParameter
from .ordered import OrderedParameter
from .cholesky_cov import CovMatrixParameter, CholeskyFactorCovParameter
from .correlation import CorrMatrixParameter, CholeskyFactorCorrParameter
from .registry import ParameterKind, PARAMETER_KINDS
from .model import Model, LogProbFn

__all__ = [
    "Model", "LogProbFn",
    "BaseParameter", "flat_size",
    "EuclideanParameter",
    "BoundedParameter", "PositiveParameter", "IntervalParameter", "BoundSpec",
    "UnitVectorParameter", "SphereChart",
    "SimplexParameter",
    "OrderedParameter",
    "CovMatrixParameter", "CholeskyFactorCovParameter",
    "CorrMatrixParameter", "CholeskyFactorCorrParameter",
    "ParameterKind", "PARAMETER_KINDS",
]
