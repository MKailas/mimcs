"""``IntegerParameter``: a bounded integer-valued parameter on ``{L, ..., U}``."""

from __future__ import annotations

import math

import numpy as np
import jax.numpy as jnp

from .discrete import BaseDiscreteParameter


def _as_int_bound(value, what: str, name: str) -> int:
    """Coerce a declared bound to an exact Python int, or explain why it cannot be.

    Bounds arrive from the DSL through ``_resolve_bound``, which floats every constant
    (``mimcs/dsl/semantics.py``), so ``upper=3`` reaches us as ``3.0``. Rounding it back is
    right; rounding ``3.5`` back would silently move the support, so that raises.
    """
    if value is None:
        raise ValueError(
            f"integer parameter '{name}' needs an explicit {what} bound: write "
            f"`int<lower=..., upper=...> {name};`. Both bounds are required because the "
            f"Metropolis-within-Gibbs sweep proposes from the enumerated support, which an "
            f"unbounded integer does not have")
    if isinstance(value, str):
        raise ValueError(
            f"integer parameter '{name}' has a parameter-dependent {what} bound "
            f"({value!r}). Stage 1 supports constant integer bounds only: the support is "
            f"baked into the sampler's proposal, so it may not vary with another parameter")
    try:
        f = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"integer parameter '{name}' has a non-numeric {what} bound {value!r}")
    if not math.isfinite(f):
        raise ValueError(
            f"integer parameter '{name}' has a non-finite {what} bound ({value!r}); its "
            f"support must be finite")
    if f != round(f):
        raise ValueError(
            f"integer parameter '{name}' has a non-integer {what} bound ({f!r})")
    return int(round(f))


class IntegerParameter(BaseDiscreteParameter):
    """Integer parameter with constant inclusive bounds, ``x in {lower, ..., upper}``.

    Declared ``int<lower=L, upper=U>`` in the DSL, and ``array[n] int<lower=L, upper=U>`` for a
    vector of them --- every element shares the one support. The two canonical uses are a binary
    indicator (``lower=0, upper=1``: spike-and-slab inclusion, a change-point flag) and a
    categorical label (``lower=1, upper=K``: mixture membership, a latent class).

    It has **no chart**: the value the model reads is the value the sampler moves, so this type
    contributes to the model's discrete block and to nothing else --- not to ``coord_dim``, not to
    ``ambient_dim``, and not to any mass matrix. See :mod:`mimcs.model.discrete` for why, and
    ``docs/design/14_discrete_parameters.md`` for the whole picture.

    Both bounds are required. That is not a placeholder for a later relaxation of the type but a
    property of the sampler: :class:`~mimcs.samplers.DiscreteMetropolisWithinGibbs` proposes
    uniformly over the ``n-1`` values a coordinate is *not* currently at, which needs an
    enumerable support. Count-valued (singly-bounded) integers want a different proposal and are
    deferred.

    Args:
        name: parameter name (key in the model's value dict).
        shape: ambient shape; ``()`` for a scalar, ``(n,)`` for a vector of them.
        lower: inclusive lower bound, a constant integer.
        upper: inclusive upper bound, a constant integer.
    """

    def __init__(self, name: str, shape: tuple = (), *, lower=None, upper=None):
        self.name = name
        self.ambient_shape = tuple(shape)
        self.parents = ()

        lo = _as_int_bound(lower, "lower", name)
        hi = _as_int_bound(upper, "upper", name)
        if hi < lo:
            raise ValueError(
                f"integer parameter '{name}' has upper < lower ({hi} < {lo}): its support "
                f"is empty")
        self.lower_value, self.upper_value = lo, hi

        n = self.size
        self.lower = jnp.full((n,), lo, jnp.int32)
        self.upper = jnp.full((n,), hi, jnp.int32)

    def __repr__(self) -> str:
        shape = f", shape={self.ambient_shape}" if self.ambient_shape else ""
        return (f"IntegerParameter({self.name!r}{shape}, "
                f"lower={self.lower_value}, upper={self.upper_value})")

    def validate(self, value) -> None:
        """Raise unless every element of ``value`` is an integer inside the support.

        Used where a *user-supplied* value enters --- an explicit ``init_position`` --- so that a
        label outside its range fails at the call rather than as a silently ``-inf`` density or an
        out-of-bounds gather that JAX clamps instead of raising.
        """
        arr = np.asarray(value)
        if arr.size != self.size:
            raise ValueError(
                f"'{self.name}' expects {self.size} value(s) (shape {self.ambient_shape}), "
                f"got {arr.size}")
        flat = arr.reshape(-1)
        if not np.all(flat == np.round(flat)):
            raise ValueError(f"'{self.name}' is an integer parameter; got non-integer value(s)")
        out = (flat < self.lower_value) | (flat > self.upper_value)
        if np.any(out):
            bad = np.unique(flat[out])[:5]
            raise ValueError(
                f"'{self.name}' has value(s) {list(bad)} outside its support "
                f"[{self.lower_value}, {self.upper_value}]")
