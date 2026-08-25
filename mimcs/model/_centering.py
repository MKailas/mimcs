"""The centering (standardizing) chart, shared by every parameter type that offers one.

``centered=True`` composes an affine standardization ``q = (z - mu) / sigma`` onto whatever
chart a parameter already has --- the identity for :class:`~mimcs.model.EuclideanParameter`,
the bound-removing link for :class:`~mimcs.model.BoundedParameter`. The hyperparameters
``(mu, sigma)`` are fitted to the chain by
:class:`~mimcs.adaptation.CenteringAdaptation`; ``None`` means the parameter is not centered,
and every function here is then the identity.

It lives on its own so that a new centerable parameter type composes these four calls instead
of hand-rolling the same arithmetic a third time. The hyperparameter pytree is a plain
``(mu, sigma)`` pair, which is what the adaptation mixin reads and writes.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def init_centering(coord_dim: int) -> tuple:
    """Initial ``(mu, sigma)``: the identity chart ``q = z``."""
    return (jnp.zeros((coord_dim,)), jnp.ones((coord_dim,)))


def standardize(z: Array, hyperparams) -> Array:
    """``(z - mu) / sigma`` --- the chart's own value to the standardized coordinate."""
    if hyperparams is None:
        return z
    mu, sigma = hyperparams
    return (z - mu) / sigma


def unstandardize(q: Array, hyperparams) -> Array:
    """``sigma * q + mu`` --- the standardized coordinate back to the chart's own value."""
    if hyperparams is None:
        return q
    mu, sigma = hyperparams
    return sigma * q + mu


def log_jacobian(hyperparams) -> Array:
    """``sum log sigma``: the standardization's contribution to ``log|dx/dq|`` (constant in ``q``)."""
    if hyperparams is None:
        return jnp.zeros(())
    _, sigma = hyperparams
    return jnp.sum(jnp.log(sigma))
