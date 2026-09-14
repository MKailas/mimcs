"""``IntegerParameter``: an integer-valued parameter on ``{L, ..., U}``, either side possibly open."""

from __future__ import annotations

import math

import numpy as np
import jax.numpy as jnp

from .discrete import BaseDiscreteParameter

#: The representable range of an integer parameter, and the **sentinel** an open side's bound array
#: carries. It is ``2**30 - 1`` rather than the int32 limit so the clamp arithmetic of a random-walk
#: proposal cannot overflow: with ``|cur|`` and ``|bound|`` both at most this, the distance to a
#: bound, ``upper - cur``, is at most ``2**31 - 2``. Practically unbounded, and exact in float32
#: (the proposal's step length is clipped in float before its int cast).
INT_BOUND = 2 ** 30 - 1

#: How far past its one finite bound a singly-bounded parameter's starting window reaches, and the
#: half-width of an unbounded one's around zero --- the discrete mirror of ``UniformInit``'s
#: ``U(-2, 2)``: a small, deterministic region near the coordinate origin rather than a draw over a
#: support of ``2**31`` values.
INIT_WINDOW = 4


def _as_int_bound(value, what: str, name: str):
    """Coerce a declared bound to an exact Python int, or ``None`` for an open side.

    Bounds arrive from the DSL through ``_resolve_bound``, which floats every constant
    (``mimcs/dsl/semantics.py``), so ``upper=3`` reaches us as ``3.0``. Rounding it back is
    right; rounding ``3.5`` back would silently move the support, so that raises.
    """
    if value is None:
        return None
    if isinstance(value, str):
        raise ValueError(
            f"integer parameter '{name}' has a parameter-dependent {what} bound "
            f"({value!r}). Only constant integer bounds are supported: the support is baked "
            f"into the sampler's proposal, so it may not vary with another parameter")
    try:
        f = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"integer parameter '{name}' has a non-numeric {what} bound {value!r}")
    if not math.isfinite(f):
        raise ValueError(
            f"integer parameter '{name}' has a non-finite {what} bound ({value!r}); to leave "
            f"that side open, omit the bound instead")
    if f != round(f):
        raise ValueError(
            f"integer parameter '{name}' has a non-integer {what} bound ({f!r})")
    i = int(round(f))
    if abs(i) > INT_BOUND:
        raise ValueError(
            f"integer parameter '{name}' has a {what} bound ({i}) beyond the representable "
            f"range +-{INT_BOUND}")
    return i


class IntegerParameter(BaseDiscreteParameter):
    """Integer parameter with constant inclusive bounds, ``x in {lower, ..., upper}``.

    Declared ``int<lower=L, upper=U>`` in the DSL, and ``array[n] int<lower=L, upper=U>`` for a
    vector of them --- every element shares the one support. The canonical bounded uses are a binary
    indicator (``lower=0, upper=1``) and a categorical label (``lower=1, upper=K``).

    **Either bound may be omitted.** A count is ``int<lower=0>``; a free integer is ``int``. An open
    side has no enumerable support, so such a parameter can only be moved by a random-walk proposal
    (:class:`~mimcs.samplers.discrete_updates.RandomWalkUpdate`), and it is **ordinal** by nature.

    **Ordinal** says the values are ordered, so neighbouring values are similar --- a count, a
    change point, a discretised scale --- rather than unordered categories. It is declared
    (``ordinal int<lower=1, upper=200>``) and it is implied by an open side. It changes nothing
    about the target; it is a statement the sampler factory reads to pick a random walk.

    It has **no chart**: the value the model reads is the value the sampler moves, so this type
    contributes to the model's discrete block and to nothing else. See
    :mod:`mimcs.model.discrete` and ``docs/design/14_discrete_parameters.md``.

    Args:
        name: parameter name (key in the model's value dict).
        shape: ambient shape; ``()`` for a scalar, ``(n,)`` for a vector of them.
        lower: inclusive lower bound, a constant integer, or ``None`` for an open side.
        upper: inclusive upper bound, a constant integer, or ``None`` for an open side.
        ordinal: are the values ordered? Implied (and so ignored) when a side is open.
    """

    def __init__(self, name: str, shape: tuple = (), *, lower=None, upper=None,
                 ordinal: bool = False):
        self.name = name
        self.ambient_shape = tuple(shape)
        self.parents = ()

        lo = _as_int_bound(lower, "lower", name)
        hi = _as_int_bound(upper, "upper", name)
        if lo is not None and hi is not None and hi < lo:
            raise ValueError(
                f"integer parameter '{name}' has upper < lower ({hi} < {lo}): its support "
                f"is empty")
        #: ``None`` on an open side. Deliberately not a sentinel: every Python consumer doing
        #: support arithmetic (a table width, a candidate count) then fails loudly on an unbounded
        #: parameter instead of silently building something ``2**31`` wide.
        self.lower_value, self.upper_value = lo, hi
        self._declared_ordinal = bool(ordinal)

        n = self.size
        self.lower = jnp.full((n,), -INT_BOUND if lo is None else lo, jnp.int32)
        self.upper = jnp.full((n,), INT_BOUND if hi is None else hi, jnp.int32)

    @property
    def bounded(self) -> bool:
        """Are both sides finite, i.e. is the support enumerable?"""
        return self.lower_value is not None and self.upper_value is not None

    @property
    def ordinal(self) -> bool:
        """Declared ordinal, or open on a side (and so ordinal by nature)."""
        return self._declared_ordinal or not self.bounded

    @property
    def n_values(self):
        """Support size ``upper - lower + 1`` as a Python int, or ``None`` when a side is open."""
        if not self.bounded:
            return None
        return self.upper_value - self.lower_value + 1

    def __repr__(self) -> str:
        shape = f", shape={self.ambient_shape}" if self.ambient_shape else ""
        ordinal = ", ordinal=True" if self._declared_ordinal else ""
        return (f"IntegerParameter({self.name!r}{shape}, "
                f"lower={self.lower_value}, upper={self.upper_value}{ordinal})")

    def default_value(self):
        """A valid starting value: the lower bound if finite, else the upper, else zero."""
        if self.lower_value is not None:
            v = self.lower_value
        elif self.upper_value is not None:
            v = self.upper_value
        else:
            v = 0
        return jnp.full((self.size,), v, jnp.int32)

    def init_range(self) -> tuple[int, int]:
        """The inclusive window a randomised start draws from.

        The whole support when bounded; otherwise :data:`INIT_WINDOW` values next to the one finite
        bound, or around zero when both sides are open. A uniform draw over ``+-2**30`` would start
        the chain a billion units from anywhere the posterior plausibly is.
        """
        lo, hi = self.lower_value, self.upper_value
        if lo is not None and hi is not None:
            return lo, hi
        if lo is not None:
            return lo, lo + INIT_WINDOW
        if hi is not None:
            return hi - INIT_WINDOW, hi
        return -INIT_WINDOW // 2, INIT_WINDOW // 2

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
        lo = -INT_BOUND if self.lower_value is None else self.lower_value
        hi = INT_BOUND if self.upper_value is None else self.upper_value
        out = (flat < lo) | (flat > hi)
        if np.any(out):
            bad = np.unique(flat[out])[:5]
            raise ValueError(
                f"'{self.name}' has value(s) {list(bad)} outside its support "
                f"[{self.lower_value}, {self.upper_value}]"
                + ("" if self.bounded else f" (an open side is limited to +-{INT_BOUND})"))
