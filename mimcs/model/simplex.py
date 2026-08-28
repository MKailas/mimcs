"""``SimplexParameter``: a probability vector on the simplex ``Delta^(d-1)``."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from . import _stick_breaking as stick
from .parameter import BaseParameter, _index_prefixes, flat_size


class SimplexParameter(BaseParameter):
    """A probability vector: ``d`` positive components summing to one.

    Like :class:`~mimcs.model.UnitVectorParameter`, its coordinate space has a *lower dimension*
    than its ambient space --- ``d - 1`` free coordinates for ``d`` ambient components --- because
    the sum-to-one constraint removes exactly one degree of freedom.

    The chart is **stick breaking** (:mod:`mimcs.model._stick_breaking`), the standard simplex
    transform: break a unit stick into ``d`` pieces, taking ``sigmoid(y_k + offset_k)`` of what
    remains at each of the ``d - 1`` steps. The offsets place ``y = 0`` at the *uniform* point
    ``x_k = 1/d``, the centre of the simplex, so the coordinate origin is a sensible default
    starting point.

    Unlike the sphere, the simplex is **flat** (it lies in the affine hull ``sum_k x_k = 1``) and
    has a **boundary** (the faces where a component hits zero). Both facts show up in
    :meth:`stein_terms`.

    Args:
        name: parameter name (key in the model's value dict).
        d: number of components; the simplex is ``Delta^(d-1)``. Must be >= 2.
        batch_shape: shape of an *array of* simplices; ``()`` for a single one. The ambient
            shape is ``batch_shape + (d,)`` and each simplex is transformed independently.
    """

    def __init__(self, name: str, d: int, batch_shape: tuple = ()):
        if int(d) < 2:
            raise ValueError(
                f"simplex[{d}] is degenerate: it needs at least 2 components (d >= 2)")
        self.name = name
        self.d = int(d)
        self.batch_shape = tuple(batch_shape)
        self.ambient_shape = self.batch_shape + (self.d,)
        self.coord_dim = flat_size(self.batch_shape) * (self.d - 1)
        self.parents = ()

    def _offsets(self) -> Array:
        return stick.default_offsets(self.d - 1)

    # --- chart maps ---

    def _coord_shape(self) -> tuple:
        return self.batch_shape + (self.d - 1,)

    def to_coordinate(self, sample, hyperparams=None, chart_index=0, parents=None):
        x = jnp.reshape(sample, self.ambient_shape)
        return jnp.reshape(stick.inverse(x, self._offsets()), (self.coord_dim,))

    def from_coordinate(self, coordinate, hyperparams=None, chart_index=0, parents=None):
        y = jnp.reshape(coordinate, self._coord_shape())
        return jnp.reshape(stick.forward(y, self._offsets()), self.ambient_shape)

    def log_jacobian_det(self, coordinate, hyperparams=None, chart_index=0, parents=None):
        y = jnp.reshape(coordinate, self._coord_shape())
        return jnp.sum(stick.log_jacobian(y, self._offsets()))

    # --- features (observables) ---

    @property
    def n_features(self) -> int:
        return flat_size(self.batch_shape) * 2 * (self.d - 1)

    def feature_names(self) -> list:
        names = []
        for pre in _index_prefixes(self.batch_shape):
            names += [f"{self.name}{pre}[{j}]" for j in range(1, self.d)]
            names += [f"{self.name}{pre}[{j}]^2" for j in range(1, self.d)]
        return names

    def features(self, sample: Array) -> Array:
        """``[x_1..x_{d-1}, x_1^2..x_{d-1}^2]`` per simplex --- the last component dropped.

        The generic ``[x, x^2]`` would be rank-deficient: ``sum_k x_k = 1`` exactly, so the ``d``
        *linear* features are collinear with the intercept of any model fitted on them (the
        mirror of :class:`~mimcs.model.UnitVectorParameter`, where it is the squares that are
        collinear, since there the constraint is on ``sum_k x_k^2``).

        Dropping ``x_d`` alone would not be enough. ``x_d^2 = (1 - sum_{k<d} x_k)^2`` expands
        into squares *and cross terms*; for ``d > 2`` the cross terms are outside the feature
        span so ``x_d^2`` is independent, but at ``d = 2`` there are none and ``x_2^2`` is again
        an exact combination of ``1, x_1, x_1^2``. Dropping the last component entirely is
        rank-safe for every ``d``, and loses nothing: it leaves exactly the default features on
        the ``d - 1`` free coordinates, two per degree of freedom.
        """
        x = jnp.reshape(sample, self.ambient_shape)
        free = x[..., :-1]
        per_simplex = jnp.concatenate([free, free ** 2], axis=-1)
        return jnp.reshape(per_simplex, (self.n_features,))

    def ambient_names(self) -> list:
        names = []
        for pre in _index_prefixes(self.batch_shape):
            names += [f"{self.name}{pre}[{j}]" for j in range(1, self.d + 1)]
        return names

    def stein_terms(self, sample: Array, score: Array) -> Array:
        """Langevin--Stein terms on the simplex, matching :meth:`features`' order.

        The simplex is flat, so the Langevin generator of its affine hull is the ordinary one
        restricted to the tangent space ``V = {v : sum_k v_k = 0}``, with orthogonal projector
        ``P = I - (1/d) 1 1^T`` --- subtracting the mean. There is no curvature term (contrast
        :class:`~mimcs.model.UnitVectorParameter`, where the sphere's contributes ``-(d-1) x_i``),
        and with ``L h = Delta_V h + <grad_V h, grad_V log pi>``::

            L x_i   = (P g)_i
            L x_i^2 = 2 (1 - 1/d) + 2 x_i (P g)_i

        using ``Delta_V(x_i^2) = 2 P_ii = 2(1 - 1/d)``. Only ``P g`` appears, which is what makes
        this independent of how ``score`` was obtained: the ambient score is defined only up to
        the constraint's normal direction ``1``, and the projection discards exactly that.

        Unlike the sphere the simplex has a **boundary**, so ``E_pi[L h] = 0`` needs the boundary
        flux to vanish --- true when the density goes to zero on the faces, the same assumption
        :class:`~mimcs.model.BoundedParameter` makes at its bounds. A Dirichlet with any
        concentration below one puts mass at a face and violates it, so a Stein flag on such a
        target is reporting the assumption, not the sampler.
        """
        x = jnp.reshape(sample, self.ambient_shape)
        g = jnp.reshape(score, self.ambient_shape)
        pg = g - jnp.mean(g, axis=-1, keepdims=True)                 # tangential score P g
        free_x, free_pg = x[..., :-1], pg[..., :-1]
        lin = free_pg
        sq = 2.0 * (1.0 - 1.0 / self.d) + 2.0 * free_x * free_pg
        per_simplex = jnp.concatenate([lin, sq], axis=-1)
        return jnp.reshape(per_simplex, (self.n_features,))
