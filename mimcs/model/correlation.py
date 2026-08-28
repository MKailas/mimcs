"""Correlation-matrix parameters: ``CorrMatrixParameter`` and ``CholeskyFactorCorrParameter``.

The correlation counterparts of :mod:`mimcs.model.cholesky_cov`, and the same pair of views: the
matrix ``Omega`` --- symmetric positive definite with a **unit diagonal** --- or its Cholesky factor
``L``, lower triangular with **unit row norms**, so that ``Omega = L L^T``. They are what a
hierarchical model wants when scales and dependence deserve separate priors, the
``diag(sigma) Omega diag(sigma)`` decomposition.

**The chart is Stan's**, canonical partial correlations with a ``tanh`` link and stick breaking on
*sums of squares*. For ``m = K(K-1)/2`` coordinates ``y``, with ``z = tanh(y)``::

    L_11 = 1,  L_i1 = z_i1,  L_ij = z_ij sqrt(1 - sum_{k<j} L_ik^2),  L_ii = sqrt(1 - sum_{k<i} L_ik^2)

Every row then has unit norm by construction, and the diagonal is *determined* rather than free ---
so ``m`` coordinates match the ``m`` free ambient entries, the strict lower triangle. ``y = 0``
gives the identity.

**The Stein operator is the one induced by Brownian motion on the elliptope**, taken as an embedded
submanifold of the SPD symmetric space with the induced affine-invariant metric. Deriving it needed
one departure from :mod:`mimcs.model.cholesky_cov`, which is documented at :meth:`stein_terms` and in
``docs/design/04_manifold_parameters.md``: there is no closed-form Laplacian here, so the operator
is assembled in its divergence form instead. Only the square case is supported.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from .parameter import BaseParameter, _index_prefixes, flat_size


def _strict_tril(K: int):
    """Row-major strict lower-triangle index arrays --- the coordinate order."""
    return jnp.tril_indices(K, -1)


def _l_from_coordinate(y: Array, K: int) -> Array:
    """``(m,)`` coordinates -> the ``(K, K)`` factor ``L`` with unit row norms.

    Stick breaking on sums of squares: each row spends what is left of its unit budget. The rows
    are independent of one another, so this is a scan down the rows in which only the running sum
    of squares is carried.
    """
    z = jnp.tanh(y)
    L = jnp.zeros((K, K), float).at[0, 0].set(1.0)
    idx = 0
    for i in range(1, K):
        remaining = jnp.ones((), float)
        for j in range(i):
            value = z[idx] * jnp.sqrt(remaining)
            L = L.at[i, j].set(value)
            remaining = remaining - value ** 2
            idx += 1
        L = L.at[i, i].set(jnp.sqrt(jnp.clip(remaining, 0.0)))
    return L


def _coordinate_from_l(L: Array, K: int) -> Array:
    """The ``(K, K)`` factor -> ``(m,)`` coordinates: undo the stick breaking, then ``arctanh``."""
    out, idx = [], 0
    for i in range(1, K):
        remaining = jnp.ones((), float)
        for j in range(i):
            out.append(L[i, j] / jnp.sqrt(remaining))
            remaining = remaining - L[i, j] ** 2
            idx += 1
    return jnp.arctanh(jnp.clip(jnp.stack(out), -1.0 + 1e-7, 1.0 - 1e-7)) if out \
        else jnp.zeros((0,), float)


def _log_jacobian_l(y: Array, K: int) -> Array:
    """``log|d(strict lower L)/dy|``.

    Two factors. The link contributes ``sum log(1 - z^2)``; the stick breaking is triangular in
    row-major order --- ``L_ij`` depends only on ``z_ij`` and on earlier entries of its own row ---
    so its determinant is the product of ``dL_ij/dz_ij = sqrt(remaining)``, giving
    ``(1/2) sum log(remaining)``.
    """
    z = jnp.tanh(y)
    total = jnp.sum(jnp.log1p(-z ** 2))
    idx = 0
    for i in range(1, K):
        remaining = jnp.ones((), float)
        for j in range(i):
            total = total + 0.5 * jnp.log(remaining)
            remaining = remaining - (z[idx] * jnp.sqrt(remaining)) ** 2
            idx += 1
    return total


def _project_tangent(omega: Array, V: Array) -> Array:
    """The metric-orthogonal projection of a symmetric ``V`` onto the elliptope's tangent space.

    The tangent space at ``Omega`` is the symmetric matrices with **zero diagonal**, and the
    orthogonal complement in the induced metric ``<A, B> = tr(Omega^-1 A Omega^-1 B)`` turns out to
    be ``{Omega D Omega : D diagonal}``: the residual ``V - P(V)`` is orthogonal to every
    zero-diagonal ``B`` exactly when ``Omega^-1 (V - P(V)) Omega^-1`` is diagonal. Requiring
    ``diag(P(V)) = 0`` then pins ``D``::

        P(V) = V - Omega diag(d) Omega,     (Omega o Omega) d = diag(V)

    with ``o`` the Hadamard product. ``Omega o Omega`` is positive definite by the Schur product
    theorem, so the ``K x K`` solve is well posed.
    """
    d = jnp.linalg.solve(omega * omega, jnp.diag(V))
    return V - omega @ jnp.diag(d) @ omega


class _CorrelationBase(BaseParameter):
    """Shared chart, features and Stein operator for the two correlation types."""

    _feature_prefix: str

    def __init__(self, name: str, K: int, batch_shape: tuple = ()):
        if int(K) < 2:
            raise ValueError(
                f"corr[{K}] is degenerate: a correlation matrix needs at least 2 dimensions "
                f"(K >= 2), since a 1x1 one is the constant [[1]] with nothing to infer")
        self.name = name
        self.K = int(K)
        self.batch_shape = tuple(batch_shape)
        self.m = self.K * (self.K - 1) // 2
        self.ambient_shape = self.batch_shape + (self.K, self.K)
        self.coord_dim = flat_size(self.batch_shape) * self.m
        self.parents = ()

    # --- what a subclass supplies ---

    def _ambient_from_l(self, L: Array) -> Array:
        raise NotImplementedError

    def _l_from_ambient(self, value: Array) -> Array:
        raise NotImplementedError

    def _extra_log_jacobian(self, L: Array) -> Array:
        raise NotImplementedError

    def _omega_score(self, omega: Array, value: Array, score: Array) -> Array:
        """This type's ambient score, as the Lebesgue score in ``Omega``'s strict lower triangle."""
        raise NotImplementedError

    # --- chart maps ---

    def _batch(self, fn, x, out_shape):
        flat = jnp.reshape(x, (-1,) + x.shape[len(self.batch_shape):])
        return jnp.reshape(jax.vmap(fn)(flat), out_shape)

    def to_coordinate(self, sample, hyperparams=None, chart_index=0, parents=None):
        value = jnp.reshape(sample, self.ambient_shape)
        return self._batch(lambda v: _coordinate_from_l(self._l_from_ambient(v), self.K),
                           value, (self.coord_dim,))

    def from_coordinate(self, coordinate, hyperparams=None, chart_index=0, parents=None):
        y = jnp.reshape(coordinate, self.batch_shape + (self.m,))
        return self._batch(lambda c: self._ambient_from_l(_l_from_coordinate(c, self.K)),
                           y, self.ambient_shape)

    def log_jacobian_det(self, coordinate, hyperparams=None, chart_index=0, parents=None):
        y = jnp.reshape(coordinate, (-1, self.m))

        def one(c):
            return _log_jacobian_l(c, self.K) + self._extra_log_jacobian(
                _l_from_coordinate(c, self.K))

        return jnp.sum(jax.vmap(one)(y))

    # --- features (observables) ---

    @property
    def n_features(self) -> int:
        return flat_size(self.batch_shape) * 2 * self.m

    def feature_names(self) -> list:
        idx = [(int(a) + 1, int(b) + 1) for a, b in zip(*_strict_tril(self.K))]   # 1-based labels
        names = []
        for pre in _index_prefixes(self.batch_shape):
            stem = f"{self._feature_prefix}{pre}"
            names += [f"{stem}[{a},{b}]" for a, b in idx]
            names += [f"{stem}[{a},{b}]^2" for a, b in idx]
        return names

    def features(self, sample: Array) -> Array:
        """The **strict** lower triangle of ``Omega`` and its squares --- ``2m`` per matrix.

        The diagonal is excluded because it is constantly 1: a constant feature is a frozen
        coordinate, whose ESS is perfect and whose R-hat is undefined, and feeding one to a
        diagnostic is how a broken chain comes to look excellent. Everything below the diagonal is
        free --- the elliptope's interior is open in ``R^m`` --- so nothing else is dropped, unlike
        :class:`~mimcs.model.SimplexParameter`.
        """
        value = jnp.reshape(sample, self.ambient_shape)
        rows, cols = _strict_tril(self.K)

        def one(v):
            L = self._l_from_ambient(v)
            entries = (L @ L.T)[rows, cols]
            return jnp.concatenate([entries, entries ** 2])

        return self._batch(one, value, (self.n_features,))

    def stein_terms(self, sample: Array, score: Array) -> Array:
        """Langevin--Stein terms from the Brownian motion on the elliptope.

        The correlation matrices sit inside the SPD symmetric space
        (:meth:`mimcs.model.CovMatrixParameter.stein_terms`) as the **embedded submanifold**
        ``{Omega : diag(Omega) = I}``. Its tangent space is the symmetric matrices with zero
        diagonal, and it inherits the affine-invariant metric ``tr(Omega^-1 A Omega^-1 B)``
        unchanged; :func:`_project_tangent` is the orthogonal projection that goes with it, so the
        Riemannian gradient of a feature is ``P(Omega C Omega)`` for ``C = d(phi)/d(Omega)``.

        **Why this is not written as a Laplacian.** For the covariance types the operator is
        ``Delta phi + <grad phi, grad log pi>`` with a closed-form ``Delta``, because the linear
        coordinates are eigenfunctions there. That route does not survive the embedding: on a
        submanifold ``Delta_E`` differs from the ambient Laplacian by the mean curvature, and the
        second fundamental form of the elliptope in this metric is not something the author could
        derive with any confidence. So the operator is assembled in its **divergence form**
        instead. In any coordinates ``x`` whose reference measure is Lebesgue with density ``p``,

            A phi = div_x(v) + v . s,    v = grad_E phi in those coordinates,  s = grad_x log p

        and this is *identically* ``Delta_E phi + <grad phi, grad log pi_riem>`` --- the volume
        element relating ``p`` to ``pi_riem`` cancels exactly the difference between the Euclidean
        and Riemannian divergences. It is therefore the operator that was wanted, only computed
        without ever forming ``Delta_E``. The coordinates ``x`` are the strict lower triangle of
        ``Omega``, and ``div_x`` comes from :func:`jax.jacfwd` --- **exact, not an approximation**,
        at ``m`` extra tangent evaluations per draw. That cost is a diagnostic's, paid once per
        draw in :meth:`~mimcs.samplers.BaseSampler.summary`, never in a kernel; it grows about as
        ``m^2 K^3``, which is comfortable for the sizes a correlation matrix is used at. A closed
        form would be an optimization, not a correction.

        Two things justify trusting it without one. Applied to the SPD geometry the same helper
        reproduces :class:`~mimcs.model.CovMatrixParameter`'s closed forms exactly, and against an
        LKJ target the terms integrate to zero by quadrature (K = 2 and 3) and average to zero over
        exact draws (the tests).

        **No boundary assumption is needed**, as for the covariance types: the singular matrices
        are at infinite distance in this metric --- distances within a submanifold are at least the
        ambient ones, and ``P(K)``'s boundary is already infinitely far.
        """
        value = jnp.reshape(sample, self.ambient_shape)
        g = jnp.reshape(score, self.ambient_shape)
        K, rows, cols = self.K, *_strict_tril(self.K)

        def unvech(x):
            M = jnp.zeros((K, K), float).at[rows, cols].set(x)
            return M + M.T + jnp.eye(K, dtype=float)

        def fields(x):
            """Every feature's Riemannian gradient, in coordinates: shape ``(2m, m)``."""
            omega = unvech(x)
            hadamard_inv = jnp.linalg.inv(omega * omega)

            def one(a, b):
                C = jnp.zeros((K, K), float).at[a, b].add(0.5).at[b, a].add(0.5)
                ambient = omega @ C @ omega
                tangent = ambient - omega @ jnp.diag(
                    hadamard_inv @ jnp.diag(ambient)) @ omega
                v = tangent[rows, cols]
                return jnp.stack([v, 2.0 * omega[a, b] * v])       # phi = Omega_ab, then squared
            return jnp.concatenate([one(int(a), int(b)) for a, b in zip(rows, cols)])

        def one_matrix(v, s_ambient):
            L = self._l_from_ambient(v)
            omega = L @ L.T
            x = omega[rows, cols]
            s = self._omega_score(omega, v, s_ambient)
            divergence = jnp.trace(jax.jacfwd(fields)(x), axis1=1, axis2=2)
            return divergence + fields(x) @ s

        flat_v = jnp.reshape(value, (-1, K, K))
        flat_s = jnp.reshape(g, (-1, K, K))
        return jnp.reshape(jax.vmap(one_matrix)(flat_v, flat_s), (self.n_features,))


class CorrMatrixParameter(_CorrelationBase):
    """A correlation matrix: ``K x K``, positive definite, unit diagonal.

    The ambient value is ``Omega`` itself, so a model reads in the correlations directly. The unit
    diagonal shows up in the posterior summary as ``K`` rows of exact 1s --- structural, like the
    zeros of a Cholesky factor, and excluded from the features for the same reason.

    Args:
        name: parameter name (key in the model's value dict).
        K: matrix dimension. Must be >= 2.
        batch_shape: shape of an *array of* correlation matrices; ``()`` for a single one.
    """

    def __init__(self, name: str, K: int, batch_shape: tuple = ()):
        super().__init__(name, K, batch_shape)
        self._feature_prefix = name

    def _ambient_from_l(self, L):
        return L @ L.T

    def _l_from_ambient(self, value):
        return jnp.linalg.cholesky(value)

    def _extra_log_jacobian(self, L):
        # log|d(strict lower Omega)/d(strict lower L)| = sum_i (K - 1 - i) log L_ii, 0-based i.
        i = jnp.arange(self.K, dtype=float)
        return jnp.sum((self.K - 1 - i)[1:] * jnp.log(jnp.diag(L)[1:]))

    def _omega_score(self, omega, value, score):
        """Only the symmetric part matters, and a coordinate moves *two* entries of ``Omega``.

        Perturbing the coordinate ``Omega_ab`` (``a > b``) moves both ``(a, b)`` and ``(b, a)``, so
        the Lebesgue score in that coordinate is ``2 sym(G)_ab`` --- the factor of two being the
        sort of thing that survives a casual reading, hence a test of its own.
        """
        rows, cols = _strict_tril(self.K)
        return 2.0 * ((score + score.T) / 2.0)[rows, cols]


class CholeskyFactorCorrParameter(_CorrelationBase):
    """The Cholesky factor of a correlation matrix: lower triangular, unit row norms.

    The ambient value is ``L``, as in Stan's ``cholesky_factor_corr``, which is the form a
    multivariate density actually uses --- ``diag(sigma) L`` is the scale matrix of a
    ``multi_normal_cholesky``, with no product ever formed. The strict upper triangle is
    structurally zero and the diagonal is *determined* by the rest of its row; both are visible in
    the posterior summary, and neither reaches a diagnostic, since the features are taken from
    ``Omega = L L^T``.

    Args:
        name: parameter name (key in the model's value dict).
        K: matrix dimension. Must be >= 2.
        batch_shape: shape of an *array of* factors; ``()`` for a single one.
    """

    def __init__(self, name: str, K: int, batch_shape: tuple = ()):
        super().__init__(name, K, batch_shape)
        self._feature_prefix = f"corr({name})"

    def _ambient_from_l(self, L):
        return L

    def _l_from_ambient(self, value):
        return jnp.tril(value)

    def _extra_log_jacobian(self, L):
        return jnp.zeros((), float)          # the ambient value *is* L

    def _omega_score(self, omega, value, score):
        """Change of variables from ``L``'s free entries to ``Omega``'s.

        Both parametrize the same ``m``-dimensional manifold, so the scores differ by the usual
        ``s_Omega = J^-T (s_L - grad_L log|det J|)`` for ``J = d vech_s(Omega) / d vech_s(L)``.
        ``J`` and the log-determinant gradient both come from autodiff rather than from an
        identity: the map is a plain matrix product, so there is nothing to gain by deriving it,
        and an ``m x m`` solve per draw is affordable for a diagnostic.
        """
        rows, cols = _strict_tril(self.K)
        K = self.K

        def omega_free(l_free):
            L = jnp.zeros((K, K), float).at[rows, cols].set(l_free)
            L = L + jnp.diag(jnp.sqrt(jnp.clip(1.0 - jnp.sum(L ** 2, axis=1), 0.0)))
            return (L @ L.T)[rows, cols]

        l_free = jnp.tril(value)[rows, cols]
        J = jax.jacfwd(omega_free)(l_free)
        log_det = jax.grad(lambda f: jnp.linalg.slogdet(jax.jacfwd(omega_free)(f))[1])(l_free)
        return jnp.linalg.solve(J.T, jnp.tril(score)[rows, cols] - log_det)
