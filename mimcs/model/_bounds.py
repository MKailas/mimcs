"""Bound specifications, shared by every parameter type that accepts ``lower`` / ``upper``.

A bound may be a constant, the *name of a parent parameter* (whose value becomes the bound), or
an arbitrary callable of the parent values --- see ``docs/design/04_manifold_parameters.md``,
"Parameters with Parents". Resolving a spec yields the accessor and the parent names it implies,
which is what a parameter reports as its :attr:`~mimcs.model.BaseParameter.parents` so that
:class:`~mimcs.model.Model` can evaluate the charts in topological order.

Two types use this: :class:`~mimcs.model.BoundedParameter`, where the bounds constrain a scalar
elementwise, and :class:`~mimcs.model.OrderedParameter`, where they constrain the first and last
entry of an increasing vector.
"""

from __future__ import annotations

import math
from typing import Callable

BoundSpec = "float | int | str | Callable[[dict], Array] | None"


def resolve_bound(spec) -> tuple[Callable | None, tuple]:
    """Turn a bound spec into ``(fn(parents) -> value or None, inferred_parent_names)``.

    Accepts: a finite number (constant bound), ``None``/``inf`` (no bound), a string
    (the name of a parent whose value is the bound), or a callable
    ``fn(parents) -> array`` (an arbitrary function of parent values).
    """
    if spec is None:
        return None, ()
    if isinstance(spec, str):
        name = spec
        return (lambda parents: parents[name]), (name,)
    if callable(spec):
        return spec, ()
    value = float(spec)
    if not math.isfinite(value):
        return None, ()
    return (lambda parents: value), ()
