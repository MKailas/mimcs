"""``UnitVectorParameter``: a point on ``S^(d-1)`` in an adaptive stereographic chart.

The first parameter type whose coordinate space has a *different dimension* from its ambient
(sample) space. See ``docs/design/04_manifold_parameters.md``, "``UnitVectorParameter``".
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from .parameter import BaseParameter, _index_prefixes, flat_size


class SphereChart(NamedTuple):
    """Hyperparameters of a :class:`UnitVectorParameter`'s stereographic chart.

    ``householder`` is a *unit* vector ``v`` in ``R^d``; the reflection
    ``H_v y = y - 2 v <v, y>`` is what carries the chart's **pole** ``p = H_v e_d`` onto the
    last basis vector. ``log_scale`` sets the projection plane's offset (see the class
    docstring). Both are batched over the parameter's ``batch_shape``.

    A named tuple rather than a bare pair so that a future *tilted* projection plane (a
    normal free of the pole) can add fields without restructuring the pytree.
    """

    householder: Array   # (*batch, d), unit
    log_scale: Array     # (*batch,)


class UnitVectorParameter(BaseParameter):
    """A unit vector: a point on the sphere ``S^(d-1)`` embedded in ``R^d``.

    This is the first parameter whose coordinate space has a *different dimension* from its
    ambient (sample) space: the ambient value has ``d`` components constrained to unit norm,
    while the chart coordinate is a free point of ``R^(d-1)``.

    The chart is a **stereographic projection**, adaptively placed. Writing ``H_v`` for the
    reflection about ``v`` (an isometry of the sphere, an involution, ``|det| = 1``) and
    ``z = H_v x`` for the rotated sample:

        u = s * z[:-1] / (1 - z[-1])

    Two hyperparameters place it (:class:`SphereChart`):

    * ``householder`` ``v`` defines the **pole** ``p = H_v e_d``, the one point the chart
      cannot represent (``u`` diverges as ``x -> p``). Since ``H_v`` is symmetric,
      ``z[-1] = <x, p>``: the last rotated component *is* the pole alignment.
    * ``log_scale`` ``log s`` sets where the projection plane cuts the sphere. Tying the
      plane's normal to the pole makes its offset ``c`` a pure rescaling, and normalizing so
      that the sphere-plane intersection circle is the *unit* circle of coordinate space
      gives the bijection ``s = sqrt((1-c)/(1+c))``, ``c = (1-s^2)/(1+s^2)``. Hence the
      identity that drives the adaptation::

          |u| < 1   <=>   <x, p> < c

      "inside the unit circle" means "on the far side of the cutting plane from the pole",
      so ``c`` is exactly the quantile of ``<x, p>`` that the chart targets.

    The pole is stored *indirectly*, as the reflection that produces it, because solving
    ``H_v p = e_d`` for ``v`` is singular at ``p = e_d`` (``v ~ p - e_d -> 0``). Carrying ``v``
    instead keeps the chart unconditionally well-defined in JAX --- no guard sits in
    ``from_coordinate``'s gradient path --- and leaves the degenerate inverse to the Python-side
    adaptation, which can branch freely (see
    :class:`mimcs.adaptation.UnitVectorCenteringAdaptation`).

    The chart's hyperparameters are always present (never ``None``), so there is a single chart
    code path; ``adaptive`` only decides whether the adaptation mixin adjusts them, mirroring
    ``centered`` on the Euclidean parameters.

    Args:
        name: parameter name (key in the model's value dict).
        d: number of ambient components; the sphere is ``S^(d-1)``. Must be >= 2.
        batch_shape: shape of an *array of* unit vectors; ``()`` for a single one. The ambient
            shape is ``batch_shape + (d,)`` and each unit vector gets its own hyperparameters.
        adaptive: let :class:`mimcs.adaptation.UnitVectorCenteringAdaptation` fit the chart.
    """

    def __init__(self, name: str, d: int, batch_shape: tuple = (), *, adaptive: bool = True):
        if int(d) < 2:
            raise ValueError(
                f"unit_vector[{d}] is degenerate: it needs at least 2 components (d >= 2)")
        self.name = name
        self.d = int(d)
        self.batch_shape = tuple(batch_shape)
        self.ambient_shape = self.batch_shape + (self.d,)
        self.coord_dim = flat_size(self.batch_shape) * (self.d - 1)
        self.parents = ()
        self.adaptive = bool(adaptive)

    # --- chart placement ---

    def init_hyperparams(self) -> SphereChart:
        # v = e_1 is orthogonal to e_d, so H_v fixes e_d: the pole starts at e_d and
        # log_scale = 0 puts the plane through the equator (c = 0), matching the textbook
        # north-pole stereographic chart up to a sign flip of coordinate 0.
        v = jnp.zeros(self.ambient_shape, float).at[..., 0].set(1.0)
        return SphereChart(householder=v, log_scale=jnp.zeros(self.batch_shape, float))

    def _reflect(self, y: Array, v: Array) -> Array:
        """``H_v y = y - 2 v <v, y>``, over the last axis (``v`` unit)."""
        return y - 2.0 * v * jnp.sum(v * y, axis=-1, keepdims=True)

    # --- features (observables) ---

    @property
    def n_features(self) -> int:
        return flat_size(self.batch_shape) * (2 * self.d - 1)

    def feature_names(self) -> list:
        names = []
        for pre in _index_prefixes(self.batch_shape):
            names += [f"{self.name}{pre}[{j}]" for j in range(self.d)]
            names += [f"{self.name}{pre}[{j}]^2" for j in range(self.d - 1)]
        return names

    def features(self, sample: Array) -> Array:
        """``[x_1..x_d, x_1^2..x_{d-1}^2]`` per unit vector --- one squared term short.

        The generic ``[x, x^2]`` would be rank-deficient here: a unit vector satisfies
        ``sum_j x_j^2 = 1`` exactly, so its ``d`` squared features are collinear with the
        intercept of any model fitted on them. Dropping ``x_d^2`` loses nothing --- it is
        ``1 - sum_{j<d} x_j^2`` --- and leaves the same span at full column rank.
        """
        x = jnp.reshape(sample, self.ambient_shape)
        per_vector = jnp.concatenate([x, x[..., :-1] ** 2], axis=-1)
        return jnp.reshape(per_vector, (self.n_features,))

    def ambient_names(self) -> list:
        names = []
        for pre in _index_prefixes(self.batch_shape):
            names += [f"{self.name}{pre}[{j}]" for j in range(self.d)]
        return names

    def stein_terms(self, sample: Array, score: Array) -> Array:
        """Langevin--Stein terms on the sphere, matching :meth:`features`' order.

        The flat ``1 + x s`` is wrong here: the sphere is a curved *closed* manifold, so the right
        operator is the Langevin generator ``L h = Delta_S h + <grad_S h, grad_S log pi>``, whose
        integral over the (boundary-free) sphere vanishes. Worked out in closed form, per unit
        vector with ``P = I - x x^T`` the tangential projection at ``x``:

            L x_i   = -(d-1) x_i + (P g)_i
            L x_i^2 = 2 - 2 d x_i^2 + 2 x_i (P g)_i

        using ``Delta_S(x_i^k) = Delta_{R^d}(x_i^k) - k(k+d-2) x_i^k`` and
        ``grad_S log pi = P grad log pi``. The projection ``(P g)_i = g_i - x_i <x, g>`` is what
        makes this robust to how ``score`` was obtained: only its tangential part is used, so the
        normal component (arbitrary, extension-dependent) is discarded. ``x_d^2`` is dropped to
        match :meth:`features`.
        """
        x = jnp.reshape(sample, self.ambient_shape)
        g = jnp.reshape(score, self.ambient_shape)
        pg = g - x * jnp.sum(x * g, axis=-1, keepdims=True)          # tangential score P g
        lin = -(self.d - 1) * x + pg
        sq = 2.0 - 2.0 * self.d * x[..., :-1] ** 2 + 2.0 * x[..., :-1] * pg[..., :-1]
        per_vector = jnp.concatenate([lin, sq], axis=-1)
        return jnp.reshape(per_vector, (self.n_features,))

    def pole(self, hyperparams: SphereChart) -> Array:
        """The chart's pole ``p = H_v e_d``: the point it cannot represent. Shape ``ambient_shape``."""
        e_d = jnp.zeros(self.ambient_shape, float).at[..., -1].set(1.0)
        return self._reflect(e_d, hyperparams.householder)

    def plane_offset(self, hyperparams: SphereChart) -> Array:
        """The cutting plane's signed offset ``c = (1-s^2)/(1+s^2)`` in ``(-1, 1)``.

        ``|u| < 1`` holds exactly for the draws with ``<x, pole> < c``.
        """
        s2 = jnp.exp(2.0 * hyperparams.log_scale)
        return (1.0 - s2) / (1.0 + s2)

    # --- chart maps ---

    def to_coordinate(self, sample, hyperparams=None, chart_index=0, parents=None):
        x = jnp.reshape(sample, self.ambient_shape)
        z = self._reflect(x, hyperparams.householder)
        s = jnp.exp(hyperparams.log_scale)[..., None]
        u = s * z[..., :-1] / (1.0 - z[..., -1:])
        return jnp.reshape(u, (self.coord_dim,))

    def from_coordinate(self, coordinate, hyperparams=None, chart_index=0, parents=None):
        u = jnp.reshape(coordinate, self.batch_shape + (self.d - 1,))
        w = u / jnp.exp(hyperparams.log_scale)[..., None]
        r2 = jnp.sum(w * w, axis=-1, keepdims=True)
        z = jnp.concatenate([2.0 * w, r2 - 1.0], axis=-1) / (r2 + 1.0)
        return jnp.reshape(self._reflect(z, hyperparams.householder), self.ambient_shape)

    def log_jacobian_det(self, coordinate, hyperparams=None, chart_index=0, parents=None):
        # The parameterization is the stereographic inverse (conformal factor
        # (2 / (1 + |w|^2))^(d-1)) precomposed with w = u / s (a factor s^-(d-1)); the
        # reflection is an isometry and contributes nothing.
        u = jnp.reshape(coordinate, self.batch_shape + (self.d - 1,))
        log_s = hyperparams.log_scale
        w = u / jnp.exp(log_s)[..., None]
        r2 = jnp.sum(w * w, axis=-1)
        return jnp.sum((self.d - 1) * (jnp.log(2.0) - jnp.log1p(r2) - log_s))
