"""Covariance-matrix parameters: ``CovMatrixParameter`` and ``CholeskyFactorCovParameter``.

Two types for the same object seen two ways --- a covariance matrix ``Sigma``, or its Cholesky
factor ``L`` with ``Sigma = L L^T`` (Stan's ``cov_matrix`` and ``cholesky_factor_cov``). They differ
only in which of the two is the *ambient* value the model is written in; they share a chart, a
feature set and a Stein operator, which is why they share a module.

**The chart is log-Cholesky.** The ``m = K(K+1)/2`` coordinates are the strict lower triangle of
``L`` together with ``log L_ii``::

    L_ii = exp(z_ii),    L_ij = z_ij  (i > j),    Sigma = L L^T

Positive definiteness holds by construction rather than by a rejection or a repair, the coordinates
are unconstrained, and ``z = 0`` is the identity matrix --- so the charts' origin, which is what
:meth:`mimcs.model.Model.default_sample` returns, is a sensible place to start a chain. This is the
same parametrization :class:`mimcs.adaptation.ScoreMassAdaptation` fits a dense mass in
(``mimcs/adaptation/score_mass.py::_step_dense``) and :class:`mimcs.adaptation._polyak.PolyakLog`
averages in; those are NumPy and internal to adaptation, so the JAX chart here is separate code for
the same idea.

Only the **square** case is supported. Stan also has a rectangular ``cholesky_factor_cov[M, N]``,
whose ``Sigma`` is singular; a singular matrix is not a point of the space :meth:`stein_terms` works
on (see below), so it would be a different type with a different diagnostic rather than a size
option on this one.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax import Array

from .parameter import BaseParameter, _index_prefixes, flat_size


def _tril(K: int):
    """Row-major lower-triangle index arrays --- the coordinate order."""
    return jnp.tril_indices(K)


def _l_from_coordinate(z: Array, K: int) -> Array:
    """``(m,)`` log-Cholesky coordinates -> the ``(K, K)`` factor ``L``."""
    rows, cols = _tril(K)
    packed = jnp.zeros((K, K), float).at[rows, cols].set(z)
    return jnp.tril(packed, -1) + jnp.diag(jnp.exp(jnp.diag(packed)))


def _coordinate_from_l(L: Array, K: int) -> Array:
    """The ``(K, K)`` factor ``L`` -> ``(m,)`` log-Cholesky coordinates."""
    rows, cols = _tril(K)
    packed = jnp.tril(L, -1) + jnp.diag(jnp.log(jnp.diag(L)))
    return packed[rows, cols]


def _log_diag(z: Array, K: int) -> Array:
    """``log L_11 .. log L_KK`` straight from the coordinates (no exp/log round trip)."""
    rows, cols = _tril(K)
    return jnp.zeros((K, K), float).at[rows, cols].set(z).diagonal()


def _sigma_score(L: Array, score_L: Array, K: int) -> Array:
    """``Sigma S Sigma`` from the score with respect to ``L``.

    ``S`` is the score of the density *in* ``Sigma`` in the ``D log p[V] = tr(S V)`` convention,
    and ``Sigma S Sigma`` is the only form of it :meth:`stein_terms` needs (it is the Riemannian
    gradient's data). What arrives instead is ``S_L = d log p_L / dL``, the score of the density in
    ``L``, which differs from ``S`` both by the chain rule through ``Sigma = L L^T`` and by the
    Jacobian of that map.

    Undoing both: with ``B = tril(S_L) - diag((K - i + 1)/L_ii)`` the Jacobian term removed,
    ``tril(2 S L) = B``, and symmetry of ``S`` pins the strict upper triangle that ``tril`` dropped.
    Writing ``Z = 2 L^-T Y`` for ``Y = L^T S L`` (symmetric), ``Z = B + U`` with ``U`` strictly
    upper, and ``L^T Z`` symmetric forces ``L^T U = -triu(L^T B - B^T L, 1)`` --- one triangular
    solve. Then ``Sigma S Sigma = L Y L^T``.
    """
    i = jnp.arange(1, K + 1, dtype=float)
    B = jnp.tril(score_L) - jnp.diag((K - i + 1) / jnp.diag(L))
    R = L.T @ B - B.T @ L                                       # antisymmetric
    U = jax.scipy.linalg.solve_triangular(L.T, -jnp.triu(R, 1), lower=False)
    Y = (L.T @ (B + U)) / 2.0
    return L @ Y @ L.T


class _LogCholeskyCovariance(BaseParameter):
    """Shared chart, features and Stein operator for the two covariance types.

    A subclass says only what the ambient value is: :meth:`_ambient_from_l`, :meth:`_l_from_ambient`
    and the extra Jacobian term the ambient map contributes.
    """

    #: prefix for the feature labels --- the covariance is what the features are taken from,
    #: whether or not it is the ambient value.
    _feature_prefix: str

    def __init__(self, name: str, K: int, batch_shape: tuple = ()):
        if int(K) < 1:
            raise ValueError(f"a covariance matrix needs at least one dimension (K >= 1), got {K}")
        self.name = name
        self.K = int(K)
        self.batch_shape = tuple(batch_shape)
        self.m = self.K * (self.K + 1) // 2
        self.ambient_shape = self.batch_shape + (self.K, self.K)
        self.coord_dim = flat_size(self.batch_shape) * self.m
        self.parents = ()

    # --- what a subclass supplies ---

    def _ambient_from_l(self, L: Array) -> Array:
        raise NotImplementedError

    def _l_from_ambient(self, value: Array) -> Array:
        raise NotImplementedError

    def _extra_log_jacobian(self, log_diag: Array) -> Array:
        """``log|d(ambient)/dL|`` for one matrix, given ``log L_11 .. log L_KK``."""
        raise NotImplementedError

    # --- chart maps ---

    def _batch(self, fn, x, out_shape):
        """Apply a single-matrix function over the batch dimensions."""
        flat = jnp.reshape(x, (-1,) + x.shape[len(self.batch_shape):])
        return jnp.reshape(jax.vmap(fn)(flat), out_shape)

    def to_coordinate(self, sample, hyperparams=None, chart_index=0, parents=None):
        value = jnp.reshape(sample, self.ambient_shape)
        return self._batch(lambda v: _coordinate_from_l(self._l_from_ambient(v), self.K),
                           value, (self.coord_dim,))

    def from_coordinate(self, coordinate, hyperparams=None, chart_index=0, parents=None):
        z = jnp.reshape(coordinate, self.batch_shape + (self.m,))
        return self._batch(lambda c: self._ambient_from_l(_l_from_coordinate(c, self.K)),
                           z, self.ambient_shape)

    def log_jacobian_det(self, coordinate, hyperparams=None, chart_index=0, parents=None):
        z = jnp.reshape(coordinate, (-1, self.m))
        log_diag = jax.vmap(lambda c: _log_diag(c, self.K))(z)
        # log|dL/dz| = sum_i log L_ii, plus whatever the ambient map adds on top.
        return jnp.sum(log_diag) + jnp.sum(jax.vmap(self._extra_log_jacobian)(log_diag))

    # --- features (observables) ---

    @property
    def n_features(self) -> int:
        return flat_size(self.batch_shape) * 2 * self.m

    def feature_names(self) -> list:
        idx = [(int(a) + 1, int(b) + 1) for a, b in zip(*_tril(self.K))]   # 1-based labels
        names = []
        for pre in _index_prefixes(self.batch_shape):
            stem = f"{self._feature_prefix}{pre}"
            names += [f"{stem}[{a},{b}]" for a, b in idx]
            names += [f"{stem}[{a},{b}]^2" for a, b in idx]
        return names

    def features(self, sample: Array) -> Array:
        """The lower triangle of ``Sigma = L L^T`` and its squares --- ``2m`` per matrix.

        The covariance is the observable even when the *factor* is the ambient value: it is what a
        model means and what a reader reports, and two chains can agree on ``Sigma`` while their
        factors differ in sign convention. Nothing is dropped here, unlike
        :class:`~mimcs.model.SimplexParameter` and :class:`~mimcs.model.UnitVectorParameter`: the
        entries of ``vech(Sigma)`` range over an open set with no constraint tying them together,
        so ``[x, x^2]`` over them is already full rank.
        """
        value = jnp.reshape(sample, self.ambient_shape)
        rows, cols = _tril(self.K)

        def one(v):
            L = self._l_from_ambient(v)
            sigma = L @ L.T
            entries = sigma[rows, cols]
            return jnp.concatenate([entries, entries ** 2])

        return self._batch(one, value, (self.n_features,))

    def stein_terms(self, sample: Array, score: Array) -> Array:
        """Langevin--Stein terms from the Brownian motion on the SPD matrices.

        The natural geometry for a covariance matrix is not the flat one. Take ``P(K)``, the
        positive definite matrices with the **affine-invariant** metric
        ``<A, B>_Sigma = tr(Sigma^-1 A Sigma^-1 B)`` --- the Riemannian symmetric space
        ``GL(K)/O(K)``, on which ``Sigma -> A Sigma A^T`` is an isometry for every invertible ``A``.
        Its volume element is ``det(Sigma)^-((K+1)/2) dvech(Sigma)``, so a target with Lebesgue
        density ``p`` has Riemannian density ``pi = p det(Sigma)^((K+1)/2)``, and the generator
        ``L h = Delta h + <grad h, grad log pi>`` gives ``E_pi[L h] = 0``.

        Two facts settle the terms. The linear coordinates are **eigenfunctions** of the
        Laplace--Beltrami operator, ``Delta tr(C Sigma) = ((K+1)/2) tr(C Sigma)`` --- at
        ``Sigma = I`` summing ``(E_k)^2`` over an orthonormal basis of the symmetric matrices gives
        ``((K+1)/2) I``, and the isometries carry that everywhere --- and the product rule with
        ``||grad tr(C Sigma)||^2 = tr((C Sigma)^2)`` gives the squares. With
        ``grad log pi = Sigma S Sigma + ((K+1)/2) Sigma``::

            L Sigma_ab   = (K+1) Sigma_ab + (Sigma S Sigma)_ab
            L Sigma_ab^2 = (2K+3) Sigma_ab^2 + Sigma_aa Sigma_bb + 2 Sigma_ab (Sigma S Sigma)_ab

        Checked by hand at ``K = 1``, where ``P(1)`` is the log-scale line and the two reduce to
        ``2v + v^2 s`` and ``6v^2 + 2v^3 s``.

        **This one needs no boundary assumption.** ``P(K)`` is complete and has no boundary --- the
        singular matrices lie at infinite distance --- so unlike
        :class:`~mimcs.model.SimplexParameter` and :class:`~mimcs.model.BoundedParameter`, whose
        identities hold only if the density vanishes on a face or a bound, nothing here has to be
        assumed about the target's tails beyond integrability. A flag means the draws, not the
        fine print.
        """
        value = jnp.reshape(sample, self.ambient_shape)
        g = jnp.reshape(score, self.ambient_shape)
        rows, cols = _tril(self.K)
        K = self.K

        def one(v, s):
            L = self._l_from_ambient(v)
            sigma = L @ L.T
            sss = self._sigma_score(L, s)
            diag = jnp.diag(sigma)
            lin = (K + 1) * sigma + sss
            sq = (2 * K + 3) * sigma ** 2 + jnp.outer(diag, diag) + 2 * sigma * sss
            return jnp.concatenate([lin[rows, cols], sq[rows, cols]])

        flat_v = jnp.reshape(value, (-1, K, K))
        flat_s = jnp.reshape(g, (-1, K, K))
        return jnp.reshape(jax.vmap(one)(flat_v, flat_s), (self.n_features,))

    def _sigma_score(self, L: Array, score: Array) -> Array:
        """``Sigma S Sigma`` from this type's ambient score."""
        raise NotImplementedError


class CovMatrixParameter(_LogCholeskyCovariance):
    """A covariance matrix: ``K x K``, symmetric and positive definite.

    The ambient value is ``Sigma`` itself, so a model is written in the covariance directly --- a
    multivariate normal reads ``y ~ multi_normal(mu, Sigma)`` with no factorization in sight. The
    sampler still works in the log-Cholesky coordinates of the factor, and the two chart Jacobians
    (``z -> L`` and ``L -> Sigma``) compose to Stan's
    ``K log 2 + sum_i (K - i + 2) log L_ii``.

    Args:
        name: parameter name (key in the model's value dict).
        K: matrix dimension. Must be >= 1.
        batch_shape: shape of an *array of* covariance matrices; ``()`` for a single one. The
            ambient shape is ``batch_shape + (K, K)`` and each matrix is transformed independently.
    """

    def __init__(self, name: str, K: int, batch_shape: tuple = ()):
        super().__init__(name, K, batch_shape)
        self._feature_prefix = name

    def _ambient_from_l(self, L):
        return L @ L.T

    def _l_from_ambient(self, value):
        return jnp.linalg.cholesky(value)

    def _extra_log_jacobian(self, log_diag):
        # log|dSigma/dL| = K log 2 + sum_i (K - i + 1) log L_ii.
        i = jnp.arange(1, self.K + 1, dtype=float)
        return self.K * math.log(2.0) + jnp.sum((self.K - i + 1) * log_diag)

    def _sigma_score(self, L, score):
        """Only the symmetric part of the score is meaningful.

        The ambient value is a symmetric matrix held in a full ``(K, K)`` array, so a perturbation
        of the two off-diagonal entries that is antisymmetric moves nothing; whichever way the
        chart's VJP happens to split a cotangent between ``(i, j)`` and ``(j, i)``, the symmetric
        part is what the geometry sees.
        """
        sigma = L @ L.T
        S = (score + score.T) / 2.0
        return sigma @ S @ sigma


class CholeskyFactorCovParameter(_LogCholeskyCovariance):
    """The Cholesky factor of a covariance matrix: lower triangular, positive diagonal.

    The ambient value is ``L``, as in Stan's ``cholesky_factor_cov``, and a model forms the
    covariance where it needs one (``L * L'`` in the DSL). That is the cheaper way to write a
    multivariate normal --- the factor is what the density actually uses --- and it is why the type
    exists beside :class:`CovMatrixParameter`.

    The ambient array is the full ``(K, K)`` matrix with the strict upper triangle held at zero.
    Those ``K(K-1)/2`` entries are structural: they appear in the posterior summary as rows of
    exact zeros, which is honest about what the parametrization is and keeps the matrix shape the
    DSL needs. The *features* are taken from ``Sigma = L L^T`` regardless, so no diagnostic is
    computed from a frozen coordinate.

    Args:
        name: parameter name (key in the model's value dict).
        K: matrix dimension. Must be >= 1.
        batch_shape: shape of an *array of* factors; ``()`` for a single one. The ambient shape is
            ``batch_shape + (K, K)`` and each factor is transformed independently.
    """

    def __init__(self, name: str, K: int, batch_shape: tuple = ()):
        super().__init__(name, K, batch_shape)
        self._feature_prefix = f"cov({name})"

    def _ambient_from_l(self, L):
        return L

    def _l_from_ambient(self, value):
        return jnp.tril(value)

    def _extra_log_jacobian(self, log_diag):
        return jnp.zeros((), float)          # the ambient value *is* L

    def _sigma_score(self, L, score):
        return _sigma_score(L, jnp.tril(score), self.K)
