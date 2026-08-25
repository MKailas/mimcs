"""``OrderedParameter``: a strictly increasing vector, optionally bounded at either end."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from . import _stick_breaking as stick
from ._bounds import BoundSpec, resolve_bound
from .parameter import BaseParameter, flat_size


def _expand(value) -> Array:
    """A bound as a column, so it broadcasts against the ``(..., d)`` component axis."""
    return jnp.asarray(value)[..., None]


class OrderedParameter(BaseParameter):
    """A strictly increasing vector ``x_1 < x_2 < ... < x_d``, sampled unconstrained.

    Ordering is what identifies the components of a mixture or the cutpoints of an ordinal
    model: without it the posterior is invariant under relabelling and multimodal by
    construction. Unlike the sphere and the simplex the constraint costs no dimension --- an
    increasing vector is an *open* subset of ``R^d`` --- so there are ``d`` coordinates for ``d``
    ambient components, and the default ``[x, x^2]`` features and flat Stein terms apply
    unchanged.

    The bounds constrain the two ends: ``lower`` the first entry, ``upper`` the last (and hence,
    with the ordering, every entry). Which link is used depends on which bounds are present,
    mirroring :class:`~mimcs.model.BoundedParameter`:

    * **neither**: ``x_1 = y_1``, and each later entry adds a positive gap ``exp(y_k)``.
      ``log|J| = sum_{k>1} y_k`` (the free first entry contributes nothing).
    * **lower only** ``L``: the same, with a positive first gap too, ``x_1 = L + exp(y_1)``.
      ``log|J| = sum_k y_k``.
    * **upper only** ``U``: the reflection of the lower case, built downward from ``U``. This is
      the ``x -> U - x`` mirror --- the same relationship the reflected-log link has to the log
      link on a scalar. ``log|J| = sum_k y_k``.
    * **both** ``(L, U)``: the ``d + 1`` gaps ``(x_1 - L, x_2 - x_1, ..., U - x_d)`` are positive
      and sum to ``U - L``, so a doubly-bounded ordered vector *is* a scaled ``(d+1)``-part
      simplex. It is built by the same stick breaking
      (:mod:`mimcs.model._stick_breaking`), and ``log|J|`` is the simplex term plus
      ``d log(U - L)`` for the scaling. ``y = 0`` gives evenly spaced points across ``(L, U)``.

    Bounds may be constants, the name of a parent parameter, or a callable of the parent values,
    exactly as for :class:`~mimcs.model.BoundedParameter`; they broadcast over ``batch_shape``
    (they constrain the ends of each vector, not each component).

    **On precision.** A gap is ``exp(y_k)``, so a coordinate deep in the negatives asks for two
    entries closer together than the floats at that magnitude can express --- below about
    ``1.2e-7`` relative in float32 the gap is simply absorbed and the two entries come out
    *equal*, so the ordering stops being strict. Nothing here can prevent that (they are the same
    float), and the log-Jacobian is unaffected because it is computed from the coordinate rather
    than from the differences; but a model that takes ``log(x_{k+1} - x_k)`` will see an infinity
    long before the sampler is in any real trouble. That is a reason to reach for x64 (which
    moves the threshold down by ~9 orders of magnitude), not a reason to guard the chart.

    **On the Stein diagnostic.** The inherited flat :meth:`~mimcs.model.BaseParameter.stein_terms`
    is the right operator here --- the ordered region is an open subset of ``R^d``, not a curved
    manifold --- but its integration by parts drops a boundary term, and this parameter's
    boundary deserves a second look. Besides the bounds, it includes the *internal* faces
    ``x_k = x_{k+1}``, and a density need not vanish there: the order statistics of an i.i.d.
    sample, the canonical ordered target, are perfectly happy with two coordinates coinciding.
    Where that happens the Stein z-scores carry a bias that is a property of the target, not of
    the sampler --- the same caveat :class:`~mimcs.model.BoundedParameter` records at its bounds,
    but easier to trip over, since a repulsive prior between neighbours is the exception rather
    than the rule. R-hat and ESS on the same draws are unaffected.

    Args:
        name: parameter name (key in the model's value dict).
        d: number of components. Must be >= 1.
        batch_shape: shape of an *array of* ordered vectors; ``()`` for a single one. The ambient
            shape is ``batch_shape + (d,)`` and each vector is transformed independently.
        lower, upper: bound specs (number / ``None`` / parent name / callable).
        parents: extra parent names used by callable bounds.
    """

    def __init__(self, name: str, d: int, batch_shape: tuple = (), *, lower=None, upper=None,
                 parents: tuple = ()):
        if int(d) < 1:
            raise ValueError(f"ordered[{d}] is degenerate: it needs at least 1 component (d >= 1)")
        self.name = name
        self.d = int(d)
        self.batch_shape = tuple(batch_shape)
        self.ambient_shape = self.batch_shape + (self.d,)
        self.coord_dim = flat_size(self.ambient_shape)

        self._lower_fn, lparents = resolve_bound(lower)
        self._upper_fn, uparents = resolve_bound(upper)
        self._has_lower = self._lower_fn is not None
        self._has_upper = self._upper_fn is not None

        # unique, order-preserving union of declared + inferred parents
        self.parents = tuple(dict.fromkeys(tuple(parents) + lparents + uparents))

    def _bounds(self, parents: dict | None):
        parents = parents or {}
        L = self._lower_fn(parents) if self._has_lower else None
        U = self._upper_fn(parents) if self._has_upper else None
        return L, U

    def _offsets(self) -> Array:
        """Stick-breaking offsets for the doubly-bounded chart: ``d`` gaps to break, ``d+1`` parts."""
        return stick.default_offsets(self.d)

    # --- link x <-> unconstrained y --- #

    def _link_inverse(self, y, parents):
        L, U = self._bounds(parents)
        if self._has_lower and self._has_upper:
            gaps = stick.forward(y, self._offsets())            # (..., d+1), sums to 1
            return _expand(L) + _expand(U - L) * jnp.cumsum(gaps[..., :-1], axis=-1)
        if self._has_lower:
            return _expand(L) + jnp.cumsum(jnp.exp(y), axis=-1)
        if self._has_upper:
            # Build upward from the top, then reflect: the last entry is U - exp(y_1).
            return _expand(U) - jnp.cumsum(jnp.exp(y), axis=-1)[..., ::-1]
        first_and_gaps = jnp.concatenate([y[..., :1], jnp.exp(y[..., 1:])], axis=-1)
        return jnp.cumsum(first_and_gaps, axis=-1)

    def _link(self, x, parents):
        L, U = self._bounds(parents)
        gaps = x[..., 1:] - x[..., :-1]
        if self._has_lower and self._has_upper:
            width = _expand(U - L)
            parts = jnp.concatenate(
                [x[..., :1] - _expand(L), gaps, _expand(U) - x[..., -1:]], axis=-1) / width
            return stick.inverse(parts, self._offsets())
        if self._has_lower:
            return jnp.log(jnp.concatenate([x[..., :1] - _expand(L), gaps], axis=-1))
        if self._has_upper:
            w = (_expand(U) - x)[..., ::-1]                     # increasing, positive
            return jnp.log(jnp.concatenate([w[..., :1], w[..., 1:] - w[..., :-1]], axis=-1))
        return jnp.concatenate([x[..., :1], jnp.log(gaps)], axis=-1)

    def _link_log_jacobian(self, y, parents):
        L, U = self._bounds(parents)
        if self._has_lower and self._has_upper:
            scaling = jnp.broadcast_to(self.d * jnp.log(U - L), self.batch_shape)
            return jnp.sum(stick.log_jacobian(y, self._offsets())) + jnp.sum(scaling)
        if self._has_lower or self._has_upper:
            return jnp.sum(y)                                   # every gap is an exp
        return jnp.sum(y[..., 1:])                              # the free first entry adds nothing

    # --- chart maps --- #

    def to_coordinate(self, sample, hyperparams=None, chart_index=0, parents=None):
        x = jnp.reshape(sample, self.ambient_shape)
        return jnp.reshape(self._link(x, parents), (self.coord_dim,))

    def from_coordinate(self, coordinate, hyperparams=None, chart_index=0, parents=None):
        y = jnp.reshape(coordinate, self.ambient_shape)
        return self._link_inverse(y, parents)

    def log_jacobian_det(self, coordinate, hyperparams=None, chart_index=0, parents=None):
        y = jnp.reshape(coordinate, self.ambient_shape)
        return self._link_log_jacobian(y, parents)
