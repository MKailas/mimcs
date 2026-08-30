"""The declarable parameter kinds: the seam between a front end and the parameter types.

A front end (today only the DSL, ``mimcs/dsl/``) has to know more about a parameter type than
how to construct it: whether its keyword takes a size argument, whether a declaration may carry
``lower``/``upper``, which chart options apply to it, and whether it may be declared outside
the ``parameters`` block. Before this table each of those lived as a hard-coded ``unit_vector``
branch scattered through the lexer, parser and semantic pass --- six of them --- so adding a
parameter type meant finding all six.

Here instead the parameter types own that knowledge, and the DSL reads it. Adding a type is:
write its module beside this one, add a :class:`ParameterKind` entry here, export it from the
package ``__init__``. Registering a kind *is* what reserves its keyword in the grammar.

The builders take plain Python values --- resolved shapes as ints, bounds as
``float | parent-name str | None`` --- and never DSL objects. That is deliberate and load
bearing: it keeps :mod:`mimcs.model` from importing :mod:`mimcs.dsl`, which would be a cycle. The
DSL evaluates its own constants and resolves its own bounds, then calls :attr:`ParameterKind.build`
last. A builder signals a bad argument with :class:`ValueError`, which the caller is expected to
re-raise with whatever source location it has.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .bounded import BoundedParameter
from .integer import IntegerParameter
from .cholesky_cov import CholeskyFactorCovParameter, CovMatrixParameter
from .correlation import CholeskyFactorCorrParameter, CorrMatrixParameter
from .euclidean import EuclideanParameter
from .ordered import OrderedParameter
from .parameter import BaseParameter
from .simplex import SimplexParameter
from .unit_vector import UnitVectorParameter


@dataclass(frozen=True)
class ParameterKind:
    """One declarable parameter kind: how to build it, and what a front end may say about it.

    Args:
        name: the type keyword a program declares it with (``real``, ``unit_vector``, ...).
        build: ``(name, shape, *, base_sizes, lower, upper, **chart) -> BaseParameter``, over
            already-resolved plain Python values (see the module docstring).
        chart_options: the chart keyword arguments this kind accepts --- ``centered`` for the
            Euclidean/bounded standardization, ``adaptive`` for a unit vector's fitted chart.
            The grammar has no syntax for these; they arrive from a ``ModelSpec``.
        chart_option_hints: rejected option -> a kind-specific explanation, for the case where
            the user reached for the option that another kind spells differently.
        takes_bounds: may a declaration of this kind carry ``<lower=..., upper=...>``?
        n_base_sizes: how many sizes the base type itself carries (``unit_vector[d]`` -> 1).
            ``0`` means the keyword stands alone; an ``array[n]`` prefix is separate from this.
        parameter_only: may it be declared only in the ``parameters`` block? True for a type
            that names a chart, since only a parameter has one.
    """

    name: str
    build: Callable[..., BaseParameter]
    chart_options: tuple = ()
    chart_option_hints: dict = field(default_factory=dict)
    takes_bounds: bool = False
    n_base_sizes: int = 0
    parameter_only: bool = False


def _build_real(name, shape, *, base_sizes=(), lower=None, upper=None, **chart):
    """``real``: unconstrained unless a bound was declared."""
    if lower is None and upper is None:
        return EuclideanParameter(name, shape, **chart)
    return BoundedParameter(name, shape, lower=lower, upper=upper, **chart)


def _build_int(name, shape, *, base_sizes=(), lower=None, upper=None, **chart):
    """``int<lower=L, upper=U>``: a bounded integer parameter, moved by a Gibbs sweep.

    Both bounds are required, constant and integral; :class:`~mimcs.model.IntegerParameter`
    raises with the reason otherwise, and ``plan_parameters`` turns that into an error against
    the declaration.

    Note what this replaces. Until discrete parameters existed, ``int`` was an *alias for*
    ``real`` here, so ``parameters { int<lower=0,upper=1> z; }`` compiled to a continuous
    ``BoundedParameter`` and was sampled by NUTS on a logit link --- accepted, plausible-looking,
    and not what anybody writing it meant. ``int`` in a ``data`` block or a function signature is
    untouched: neither reaches a builder.
    """
    return IntegerParameter(name, shape, lower=lower, upper=upper)


def _build_unit_vector(name, shape, *, base_sizes, lower=None, upper=None, **chart):
    """``unit_vector[d]`` is one point on ``S^(d-1)``; an ``array[n]`` prefix makes it ``n`` of
    them, ambient shape ``(n, d)``, each with its own chart."""
    return UnitVectorParameter(name, base_sizes[0], shape, **chart)


def _build_simplex(name, shape, *, base_sizes, lower=None, upper=None, **chart):
    """``simplex[d]``: one probability vector on ``Delta^(d-1)``, ``array[n]`` for ``n`` of them."""
    return SimplexParameter(name, base_sizes[0], shape, **chart)


def _build_cov_matrix(name, shape, *, base_sizes, lower=None, upper=None, **chart):
    """``cov_matrix[K]``: one ``K x K`` covariance matrix, ``array[n]`` for ``n`` of them."""
    return CovMatrixParameter(name, base_sizes[0], shape, **chart)


def _build_cholesky_factor_cov(name, shape, *, base_sizes, lower=None, upper=None, **chart):
    """``cholesky_factor_cov[K]``: the lower-triangular factor of one covariance matrix."""
    return CholeskyFactorCovParameter(name, base_sizes[0], shape, **chart)


def _build_corr_matrix(name, shape, *, base_sizes, lower=None, upper=None, **chart):
    """``corr_matrix[K]``: one ``K x K`` correlation matrix, ``array[n]`` for ``n`` of them."""
    return CorrMatrixParameter(name, base_sizes[0], shape, **chart)


def _build_cholesky_factor_corr(name, shape, *, base_sizes, lower=None, upper=None, **chart):
    """``cholesky_factor_corr[K]``: the unit-row-norm factor of one correlation matrix."""
    return CholeskyFactorCorrParameter(name, base_sizes[0], shape, **chart)


def _build_ordered(name, shape, *, base_sizes, lower=None, upper=None, **chart):
    """``ordered[d]``: one increasing vector; the bounds constrain its first and last entry."""
    return OrderedParameter(name, base_sizes[0], shape, lower=lower, upper=upper, **chart)


_REAL = dict(build=_build_real, chart_options=("centered",), takes_bounds=True)

_COV_CENTERED_HINT = ("a covariance or correlation matrix has no `centered` flag --- its chart is "
                      "already written in the quantity it is free to choose")

#: Every parameter kind a program may declare, keyed by its type keyword. Iteration order is
#: the order the grammar lists the element types in when a declaration names an unknown one.
PARAMETER_KINDS: dict[str, ParameterKind] = {
    "real": ParameterKind(name="real", **_REAL),
    "int": ParameterKind(
        name="int",
        build=_build_int,
        # No chart options: a discrete parameter has no chart to centre or to fit.
        chart_options=(),
        chart_option_hints={
            "centered": "a discrete parameter has no chart, so nothing to standardize",
            "adaptive": "a discrete parameter has no chart to fit"},
        takes_bounds=True,
        # Deliberately NOT parameter_only: `int` is also how a `data` block declares a size and
        # how a function signature declares an index argument, and neither builds a parameter.
        parameter_only=False),
    "unit_vector": ParameterKind(
        name="unit_vector",
        build=_build_unit_vector,
        chart_options=("adaptive",),
        chart_option_hints={
            "centered": "its chart has no `centered` flag --- set `adaptive` instead (the "
                        "unit-vector mirror of centering)"},
        takes_bounds=False,
        n_base_sizes=1,
        parameter_only=True),
    "simplex": ParameterKind(
        name="simplex",
        build=_build_simplex,
        takes_bounds=False,
        n_base_sizes=1,
        parameter_only=True),
    "cov_matrix": ParameterKind(
        name="cov_matrix",
        build=_build_cov_matrix,
        chart_option_hints={"centered": _COV_CENTERED_HINT},
        takes_bounds=False,
        n_base_sizes=1,
        parameter_only=True),
    "cholesky_factor_cov": ParameterKind(
        name="cholesky_factor_cov",
        build=_build_cholesky_factor_cov,
        chart_option_hints={"centered": _COV_CENTERED_HINT},
        takes_bounds=False,
        n_base_sizes=1,
        parameter_only=True),
    "corr_matrix": ParameterKind(
        name="corr_matrix",
        build=_build_corr_matrix,
        chart_option_hints={"centered": _COV_CENTERED_HINT},
        takes_bounds=False,
        n_base_sizes=1,
        parameter_only=True),
    "cholesky_factor_corr": ParameterKind(
        name="cholesky_factor_corr",
        build=_build_cholesky_factor_corr,
        chart_option_hints={"centered": _COV_CENTERED_HINT},
        takes_bounds=False,
        n_base_sizes=1,
        parameter_only=True),
    "ordered": ParameterKind(
        name="ordered",
        build=_build_ordered,
        # The bounds constrain the first and last entry, so they are written before the size:
        # `ordered<lower=0, upper=1>[d]`.
        takes_bounds=True,
        n_base_sizes=1,
        parameter_only=True),
}
