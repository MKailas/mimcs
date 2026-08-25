"""``BoundedParameter`` and its convenience constructors ``Positive`` / ``Interval``.

A bound is removed by a *link* to an unconstrained real, so the sampler still works on
``R^d``. Bounds may be constants or functions of a *parent* parameter's value --- see
``docs/design/04_manifold_parameters.md``, "Parameters with Parents".
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax import nn

from . import _centering
from ._bounds import BoundSpec, resolve_bound
from .parameter import BaseParameter, flat_size


class BoundedParameter(BaseParameter):
    """Real parameter with a lower and/or upper bound, sampled in unconstrained space.

    The bound is removed by a *link* ``z = link(x)`` to an unconstrained ``z`` (applied
    elementwise; bounds broadcast over ``shape``):

    * both bounds ``(L, U)``: ``x = L + (U - L) * sigmoid(z)``  (logit link)
    * lower only ``L``:       ``x = L + exp(z)``                 (log link)
    * upper only ``U``:       ``x = U - exp(z)``                 (reflected log link)

    With ``centered=True`` the coordinate is the *standardized* link value
    ``q = (z - mu) / sigma`` --- ``mu``, ``sigma`` adaptive hyperparameters fitted to the
    mean / standard deviation of ``z`` by :class:`mimcs.adaptation.CenteringAdaptation`
    (e.g. for a positive parameter, of ``log(x - L)``). The change of variables then carries
    both the link and the standardization: ``log|dx/dq| = log|dx/dz| + sum log sigma``.

    Bounds may be constants, the name of a parent parameter (its value becomes the
    bound), or a callable of the parent values. Parent names are collected from string
    bounds and the explicit ``parents`` argument.

    Args:
        name: parameter name.
        shape: ambient shape (scalar ``()`` or ``(d,)``).
        lower, upper: bound specs (number / ``None`` / parent name / callable).
        parents: extra parent names used by callable bounds.
        centered: standardize the link value with adaptive ``(mu, sigma)``.
    """

    def __init__(self, name: str, shape: tuple = (), *, lower=None, upper=None,
                 parents: tuple = (), centered: bool = False):
        self.name = name
        self.ambient_shape = tuple(shape)
        self.coord_dim = flat_size(self.ambient_shape)
        self.centered = bool(centered)

        self._lower_fn, lparents = resolve_bound(lower)
        self._upper_fn, uparents = resolve_bound(upper)
        self._has_lower = self._lower_fn is not None
        self._has_upper = self._upper_fn is not None

        # unique, order-preserving union of declared + inferred parents
        self.parents = tuple(dict.fromkeys(tuple(parents) + lparents + uparents))

    def init_hyperparams(self) -> Any:
        if not self.centered:
            return None
        return _centering.init_centering(self.coord_dim)

    def _bounds(self, parents: dict | None):
        parents = parents or {}
        L = self._lower_fn(parents) if self._has_lower else None
        U = self._upper_fn(parents) if self._has_upper else None
        return L, U

    # --- link x <-> unconstrained z (the chart before any standardization) --- #

    def _link(self, x, parents):
        L, U = self._bounds(parents)
        if self._has_lower and self._has_upper:
            t = (x - L) / (U - L)
            return jnp.log(t) - jnp.log1p(-t)
        if self._has_lower:
            return jnp.log(x - L)
        if self._has_upper:
            return jnp.log(U - x)
        return x

    def _link_inverse(self, z, parents):
        L, U = self._bounds(parents)
        if self._has_lower and self._has_upper:
            return L + (U - L) * nn.sigmoid(z)
        if self._has_lower:
            return L + jnp.exp(z)
        if self._has_upper:
            return U - jnp.exp(z)
        return z

    def _link_log_jacobian(self, z, parents):
        L, U = self._bounds(parents)
        if self._has_lower and self._has_upper:
            # d x/dz = (U-L) sigmoid(z)(1-sigmoid(z))
            elem = jnp.log(U - L) - nn.softplus(z) - nn.softplus(-z)
        elif self._has_lower or self._has_upper:
            elem = z                                  # d x/dz = +/- exp(z), log|.| = z
        else:
            elem = jnp.zeros_like(z)
        return jnp.sum(elem)

    def _decenter(self, coordinate, hyperparams):
        """Coordinate -> link value ``z`` (undo the standardization), in ambient shape."""
        z = jnp.reshape(coordinate, (self.coord_dim,))
        return jnp.reshape(_centering.unstandardize(z, hyperparams), self.ambient_shape)

    def to_coordinate(self, sample, hyperparams=None, chart_index=0, parents=None):
        x = jnp.reshape(sample, self.ambient_shape)
        z = jnp.reshape(self._link(x, parents), (self.coord_dim,))
        return _centering.standardize(z, hyperparams)

    def from_coordinate(self, coordinate, hyperparams=None, chart_index=0, parents=None):
        z = self._decenter(coordinate, hyperparams)
        return self._link_inverse(z, parents)

    def log_jacobian_det(self, coordinate, hyperparams=None, chart_index=0, parents=None):
        z = self._decenter(coordinate, hyperparams)
        return self._link_log_jacobian(z, parents) + _centering.log_jacobian(hyperparams)


def PositiveParameter(name: str, shape: tuple = (), lower: float = 0.0, *,
                      centered: bool = False) -> BoundedParameter:
    """Convenience: a parameter bounded below (default ``> 0``), log link."""
    return BoundedParameter(name, shape, lower=lower, upper=None, centered=centered)


def IntervalParameter(name: str, lower, upper, shape: tuple = (), *,
                      centered: bool = False) -> BoundedParameter:
    """Convenience: a parameter on an interval ``(lower, upper)``, logit link."""
    return BoundedParameter(name, shape, lower=lower, upper=upper, centered=centered)
