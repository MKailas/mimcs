"""The ``BaseParameter`` interface every parameter type implements.

The contract is documented in ``docs/design/04_manifold_parameters.md`` and summarized in
this package's ``__init__``; concrete types live one per module beside this one.

Chart methods take ``(value, hyperparams, chart_index, parents)``:

* ``hyperparams`` --- adaptive chart hyperparameters (``None`` for fixed charts).
* ``chart_index`` --- active chart in an atlas (always 0 for the types shipped so far;
  they are all single-chart).
* ``parents`` --- a dict ``{parent_name: ambient_value}`` of the values of this parameter's
  parent parameters. A bound may *depend on a parent's value* (e.g. ``x ~ Uniform(0, sigma)``),
  which is what ``parents`` supplies.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import jax.numpy as jnp
from jax import Array


def flat_size(shape: tuple) -> int:
    """Number of elements in ``shape`` (``1`` for the scalar shape ``()``)."""
    return int(math.prod(shape))


def _index_prefixes(shape: tuple) -> list:
    """One index suffix per element of ``shape``, in ravel order (``[""]`` for a scalar)."""
    if not shape:
        return [""]
    return ["[" + ",".join(str(i) for i in idx) + "]"
            for idx in itertools.product(*(range(s) for s in shape))]


class BaseParameter:
    """Abstract base for a typed model parameter."""

    name: str
    ambient_shape: tuple
    coord_dim: int
    parents: tuple = ()   # names of parameters this one's chart depends on

    # --- hyperparameters ---

    def init_hyperparams(self) -> Any:
        """Initial chart hyperparameters (a pytree), or ``None`` for fixed charts."""
        return None

    # --- chart maps ---

    def to_coordinate(self, sample: Array, hyperparams: Any = None, chart_index=0,
                      parents: dict | None = None) -> Array:
        """Ambient value -> coordinate chart. Returns shape ``(coord_dim,)``."""
        raise NotImplementedError

    def from_coordinate(self, coordinate: Array, hyperparams: Any = None, chart_index=0,
                        parents: dict | None = None) -> Array:
        """Coordinate chart -> ambient value. Returns shape ``ambient_shape``."""
        raise NotImplementedError

    def log_jacobian_det(self, coordinate: Array, hyperparams: Any = None, chart_index=0,
                         parents: dict | None = None) -> Array:
        """``log |det d(from_coordinate)/d(coordinate)|``, parents held fixed.

        (Written as a literal because bare pipes are reST substitution syntax.)
        """
        raise NotImplementedError

    # --- atlas ---

    def n_charts(self) -> int:
        return 1

    def chart_contains(self, sample: Array, chart_index: int) -> bool:
        return True

    def is_euclidean(self) -> bool:
        return False

    # --- features (observables) ---

    @property
    def n_features(self) -> int:
        return 2 * self._ambient_size

    def feature_names(self) -> list:
        """One name per feature, in the order :meth:`features` returns them."""
        idx = _index_prefixes(self.ambient_shape)
        return ([f"{self.name}{s}" for s in idx]
                + [f"{self.name}{s}^2" for s in idx])

    def features(self, sample: Array) -> Array:
        """Observables of this parameter: fixed functions of its *ambient* value.

        A convergence diagnostic never looks at a draw directly, it looks at scalar functions of
        one --- R-hat, Geweke and a classifier alike are computed from these. Sample space is the
        right place to take them from: coordinates are a device for computation (and for a
        manifold parameter they do not even have the same dimension), whereas the ambient value is
        what the model is written in and what the draws are stored as.

        The default is the per-coordinate mean and spread, ``[x_j, x_j^2]``, which is enough to
        catch a chain whose location or scale is still moving. Returns shape ``(n_features,)``,
        and is vmappable over a stack of draws.
        """
        x = jnp.reshape(sample, (-1,))
        return jnp.concatenate([x, x ** 2])

    def ambient_names(self) -> list:
        """One label per ambient coordinate --- the row labels of the posterior summary."""
        return [f"{self.name}{s}" for s in _index_prefixes(self.ambient_shape)]

    def stein_terms(self, sample: Array, score: Array) -> Array:
        """The Langevin--Stein term of each feature: a mean-zero function of the draw.

        Applying the Langevin operator to a feature ``phi`` gives a function whose expectation
        under the target is zero, ``E_pi[A phi] = 0`` (integration by parts). Averaged over a
        sample, each is an estimate of zero whose distance from it --- in units of its Monte Carlo
        standard error --- is a *target-aware* check: unlike R-hat or ESS, it can only be satisfied
        by a sample from the actual target, not by any well-mixed sequence. The operator uses only
        the score ``s = grad log pi`` (the *ambient*-space gradient), so it never needs the
        normalizing constant.

        For a feature that is a function of a single ambient coordinate, ``phi = f(x_j)``, pairing
        it with the ``j``-th direction gives ``A_j phi = f'(x_j) + f(x_j) s_j``. The default
        features ``[x_j, x_j^2]`` therefore give ``[1 + x_j s_j]`` and ``[2 x_j + x_j^2 s_j]``, in
        the same order :meth:`features` returns them. Returns shape ``(n_features,)``, vmappable.

        ``score`` is the ambient score restricted to this parameter's block. (For a bounded
        parameter the boundary term of the integration by parts is assumed to vanish --- true when
        the density decays at the bounds; see the package and design docs.)
        """
        x = jnp.reshape(sample, (-1,))
        s = jnp.reshape(score, (-1,))
        return jnp.concatenate([1.0 + x * s, 2.0 * x + x ** 2 * s])

    @property
    def _ambient_size(self) -> int:
        return flat_size(self.ambient_shape)
