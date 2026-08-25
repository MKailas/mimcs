"""``EuclideanParameter``: an unconstrained parameter on ``R^d``."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from . import _centering
from .parameter import BaseParameter, flat_size


class EuclideanParameter(BaseParameter):
    """Unconstrained parameter on ``R^d``.

    The default chart is the identity. With ``centered=True`` it is instead the
    *centering* (standardizing) chart ``q = (x - mu) / sigma`` --- coordinate-wise, with
    adaptive hyperparameters ``mu`` (mean) and ``sigma`` (standard deviation) fitted to the
    chain by :class:`mimcs.adaptation.CenteringAdaptation`. Standardizing the sample makes
    the coordinate target well-scaled and centred; it pairs naturally with the
    score-covariance mass adaptation, which then handles the remaining (gradient) geometry
    in the standardized coordinates. The change-of-variables term is ``log|dx/dq| = sum
    log sigma``.

    Args:
        name: parameter name (key in the model's value dict).
        shape: ambient shape; scalar ``()`` for a single real, ``(d,)`` for a vector.
        centered: use the adaptive centering chart instead of the identity.
    """

    def __init__(self, name: str, shape: tuple = (), *, centered: bool = False):
        self.name = name
        self.ambient_shape = tuple(shape)
        self.coord_dim = flat_size(self.ambient_shape)
        self.parents = ()
        self.centered = bool(centered)

    def init_hyperparams(self) -> Any:
        if not self.centered:
            return None
        # (mu, sigma) starting at the identity chart (q = x).
        return _centering.init_centering(self.coord_dim)

    def to_coordinate(self, sample, hyperparams=None, chart_index=0, parents=None):
        x = jnp.reshape(sample, (self.coord_dim,))
        return _centering.standardize(x, hyperparams)

    def from_coordinate(self, coordinate, hyperparams=None, chart_index=0, parents=None):
        q = jnp.reshape(coordinate, (self.coord_dim,))
        return jnp.reshape(_centering.unstandardize(q, hyperparams), self.ambient_shape)

    def log_jacobian_det(self, coordinate, hyperparams=None, chart_index=0, parents=None):
        return _centering.log_jacobian(hyperparams)

    def is_euclidean(self) -> bool:
        # A centered chart carries a (constant-in-q) log-Jacobian, so the model needs the
        # JacobianPotential; only the plain identity chart is Jacobian-free.
        return not self.centered
