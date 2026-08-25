"""Diagonal-whitened rank-J mass adaptation for :class:`~mimcs.hmc.LowRankQuadraticKinetic`.

A mixin (``docs/design/02_sampler_classes.md``) that adapts the mass

    M = D^{1/2} (I + sum_{j=1}^J gamma_j v_j v_j^T) D^{1/2}

of a low-rank quadratic kinetic to the block's **score covariance** ``E[g g^T]`` (``g =
grad_q V``, the target precision). Two pieces are learned together, per block, during warmup:

* **Diagonal ``D``** by the same log-mass SGD as :class:`ScoreMassAdaptation`'s diagonal step
  (``theta = log D``, ``grad = 1/2 (1 - g_c^2 / D)`` on the centred score, adaptive
  gradient-norm clip). Its minimiser is ``D = E[g_c^2]`` --- the diagonal of the score
  covariance, which is exactly the whitening that makes the ``D^{-1/2}``-whitened score
  covariance ``C_w = D^{-1/2} E[g g^T] D^{-1/2}`` have unit diagonal.

* **Rank-J part ``{v_j, gamma_j}``** by **Sanger's rule** (the Generalized Hebbian Algorithm,
  a deflationary Oja) on the whitened score ``x_w = D^{-1/2} g_c``. Sanger tracks the *ordered*
  top-J eigenvectors of ``C_w`` in ``W`` (columns), re-orthonormalized each step by a thin QR;
  the whitened eigenvalues ``lambda_j = E[(v_j^T x_w)^2]`` are tracked by a running mean and
  give ``gamma_j = max(0, lambda_j - 1)``. Because ``C_w`` has unit diagonal its top eigenvalues
  are ``>= 1``, so ``gamma_j >= 0`` holds for free --- which is exactly what the
  ``M = diag(D) + V^T V`` (positive rank-one additions) representation in :mod:`mimcs.hmc.lowrank`
  requires. The clamp at 0 means the rank term can only *stiffen* the top directions; whitened
  directions with variance ``< 1`` are left at the diagonal scale ``D`` (an intentional limit
  of a positive low-rank correction). The whitened score feeding Sanger is itself adaptively
  clipped by its norm (its own quantile threshold, ~10% of steps clipped, as for the diagonal
  SGD), so a transient huge score cannot blow up the eigenvector / eigenvalue estimates.

The two run at slightly separated time scales via a short burn-in (``lowrank_min_samples``):
until then only ``D`` adapts (``gamma = 0``, a pure diagonal mass), so the whitening settles
before the eigenvectors start tracking. All stochastic-approximation math is done on the Python
object in float64; only the packed ``(D, V)`` (with ``V[j] = sqrt(gamma_j) * sqrt(D) * v_j``)
crosses into the JAX state at ``ham_params[kinetic.id]``. The shared schedule ``(n + n0)^{-kappa}``
(kappa=0.75, n0=5) drives every update. With ``mass_polyak=True`` the *diagonal* ``D`` written
for sampling is a Polyak--Ruppert average (log space); the eigenvectors are frozen at their final
estimate (subspace/sign averaging is ill-posed). Adaptation runs during warmup only.
"""

from __future__ import annotations

import math

import numpy as np
import jax.numpy as jnp

from .._logging import get_logger
from ..samplers.base import Phase
from ._stochastic import rm_gain, DEFAULT_KAPPA, DEFAULT_N0
from ._polyak import PolyakLog

log = get_logger(__name__)


class _Sanger:
    """Deflationary-Oja (Sanger/GHA) tracker of the top-``J`` eigen-directions and eigenvalues of a
    ``D^{-1/2}``-whitened score covariance ``C_w``. Shared by the constant-``D`` low-rank mass and
    the shaped metric's constant shape ``A`` --- both fit ``A = I + sum_j gamma_j v_j v_j^T`` (with
    ``gamma_j = max(0, lambda_j - 1) >= 0``) to a whitened score. The caller supplies the already-
    whitened score and the Robbins--Monro gain / step count; the whitened score is adaptively
    clipped by its own norm (own log-quantile threshold) so a transient huge score cannot blow up
    the estimates."""

    def __init__(self, n, J, n0, kappa, clip_frac, oja_const):
        self._n0, self._kappa, self._clip_frac, self._oja_const = n0, kappa, clip_frac, oja_const
        self.W = np.eye(n)[:, :J].copy()             # (n, J): top-J whitened eigenvector est.
        self.lam = np.ones(J)                        # whitened eigenvalue est.  lambda_j
        self.log_clip_w = math.log(n)                # whitened-score clip threshold init: log n

    def step(self, x_w, lr, count):
        """One Sanger step on an already-whitened score ``x_w`` (updates ``W``, ``lam`` in place)."""
        thr = math.exp(self.log_clip_w)
        norm = float(np.sqrt(np.sum(x_w ** 2)))
        scale = min(1.0, thr / (norm + 1e-12))
        self.log_clip_w += rm_gain(count, self._n0, self._kappa) * (
            (1.0 if norm > thr else 0.0) - self._clip_frac)
        x_w = scale * x_w
        y = self.W.T @ x_w                                       # projections v_j^T x_w
        self.W += (self._oja_const * lr) * (
            np.outer(x_w, y) - self.W @ np.triu(np.outer(y, y)))  # Sanger deflation
        self.W, _ = np.linalg.qr(self.W)                        # re-orthonormalize (drift)
        self.lam += lr * (y ** 2 - self.lam)                    # running whitened eigenvalues

    def gamma(self):
        """The rank-``J`` stiffenings ``gamma_j = max(0, lambda_j - 1)`` (>= 0: PD + representable)."""
        return np.maximum(0.0, self.lam - 1.0)


class _LowRankBlock:
    """SGD/Oja state and step for one low-rank kinetic block (size ``n``, rank ``J``)."""

    def __init__(self, n, J, n0, kappa, clip_frac, center_grad, mass_lr_const,
                 oja_const, min_samples, polyak):
        self._n0, self._kappa, self._clip_frac = n0, kappa, clip_frac
        self._center_grad = center_grad
        self._mass_lr_const = mass_lr_const
        self._min_samples = min_samples
        self.mean_grad = np.zeros(n)                 # running score mean (centring)
        self.log_D = np.zeros(n)                     # theta = log D  (D = exp theta)
        self.log_clip = math.log(n)                  # diagonal-SGD clip threshold init: log n
        self._sanger = _Sanger(n, J, n0, kappa, clip_frac, oja_const)   # low-rank eigen-tracker
        self.count = 0
        self._polyak = PolyakLog("diagonal") if polyak else None

    def _adaptive_clip(self, norm, log_clip):
        """Adaptive-quantile clip (as in ``ScoreMassAdaptation``): returns the scale factor and
        the updated log-threshold, converging so a ``clip_frac`` fraction of steps are clipped."""
        thr = math.exp(log_clip)
        scale = min(1.0, thr / (norm + 1e-12))
        exceeded = 1.0 if norm > thr else 0.0
        log_clip += rm_gain(self.count, self._n0, self._kappa) * (exceeded - self._clip_frac)
        return scale, log_clip

    def update(self, score):
        """One warmup step from this block's score; returns the packed ``(D, V)`` for the kinetic."""
        self.count += 1
        lr = rm_gain(self.count, self._n0, self._kappa)

        # Centre the score (E[g] -> 0 at stationarity, so the mass fits the covariance).
        g_c = score - self.mean_grad if self._center_grad else score
        if self._center_grad:
            self.mean_grad += lr * (score - self.mean_grad)

        # Diagonal D: log-mass SGD, minimiser D = E[g_c^2] (reuse of score_mass._step_diagonal).
        D = np.exp(self.log_D)
        grad = 0.5 * (1.0 - g_c ** 2 / D)
        scale, self.log_clip = self._adaptive_clip(float(np.sqrt(np.sum(grad ** 2))), self.log_clip)
        self.log_D -= (self._mass_lr_const * lr) * scale * grad
        D = np.exp(self.log_D)

        # Rank-J part: Sanger/GHA on the whitened score, after the diagonal burn-in. The whitened
        # score is adaptively clipped by its own norm (own threshold) so a transient huge score
        # cannot blow up the eigenvector/eigenvalue estimates -- the diagonal-mass analogue for the
        # low-rank part.
        if self.count > self._min_samples:
            self._sanger.step(g_c / np.sqrt(D), lr, self.count)     # Sanger on the whitened score

        if self._polyak is not None:
            self._polyak.update(D)                                  # tail-average D in log space
        return self._pack(D)

    def _pack(self, D):
        """Pack (D, {v_j}, gamma_j) into lowrank's (D, V) with V[j] = sqrt(gamma_j) sqrt(D) v_j."""
        gamma = self._sanger.gamma()                               # >= 0: PD + representable
        V = np.sqrt(gamma)[:, None] * (np.sqrt(D)[None, :] * self._sanger.W.T)   # (J, n)
        return jnp.asarray(D), jnp.asarray(V)

    def finalized(self):
        """The frozen mass for sampling: Polyak-averaged D (if enabled) with the final W, gamma."""
        D = np.exp(self.log_D)
        if self._polyak is not None and self._polyak.value() is not None:
            D = np.asarray(self._polyak.value(), float)
        return self._pack(D)


class LowRankAdaptation:
    """Mixin: adapt each :class:`~mimcs.hmc.LowRankQuadraticKinetic` block's diagonal-plus-rank-J
    mass to its score covariance (diagonal by log-mass SGD, low-rank by Sanger/Oja)."""

    def _init_hooks(self, **kwargs):
        self._lr_n0 = float(kwargs.get("lowrank_n0", DEFAULT_N0))
        self._lr_kappa = float(kwargs.get("lowrank_kappa", DEFAULT_KAPPA))
        self._lr_clip_frac = float(kwargs.get("lowrank_clip_frac", 0.1))
        self._lr_center_grad = bool(kwargs.get("lowrank_center_grad", True))
        self._lr_mass_lr_const = float(kwargs.get("lowrank_mass_lr_const", 1.0))
        self._lr_oja_const = float(kwargs.get("lowrank_oja_const", 1.0))
        self._lr_min_samples = int(kwargs.get("lowrank_min_samples", 50))
        self._lr_polyak = bool(kwargs.get("mass_polyak", False))
        self._lr_blocks: dict | None = None    # {kinetic id: _LowRankBlock}
        super()._init_hooks(**kwargs)

    def _lowrank_kinetics(self):
        """The low-rank kinetic blocks this mixin adapts (rank attribute, mass_mode None). A
        low-rank *shaped* metric block also has a ``rank`` but is adapted by
        :class:`~mimcs.adaptation.ShapedMetricAdaptation`, so it is excluded here."""
        return [k for k in self.kinetics
                if getattr(k, "rank", None) is not None and k.mass_mode is None
                and not getattr(k, "is_shaped", False)]

    def _postprocess_hooks(self, state):
        state = super()._postprocess_hooks(state)
        if self._phase is not Phase.WARMUP:
            return state
        kinetics = self._lowrank_kinetics()
        if not kinetics:
            return state
        if self._lr_blocks is None:
            self._lr_blocks = {}
        total_grad = sum(state.potential_grads.values())
        new = {}
        for k in kinetics:
            score = np.asarray(k._gather(total_grad), dtype=float)
            if k.id not in self._lr_blocks:
                self._lr_blocks[k.id] = _LowRankBlock(
                    score.shape[0], k.rank, self._lr_n0, self._lr_kappa, self._lr_clip_frac,
                    self._lr_center_grad, self._lr_mass_lr_const, self._lr_oja_const,
                    self._lr_min_samples, self._lr_polyak)
                log.debug("low-rank mass adaptation started on block %r: %d coordinate(s), "
                          "rank %d", k.id, score.shape[0], k.rank)
            new[k.id] = self._lr_blocks[k.id].update(score)
        return state._replace(ham_params={**state.ham_params, **new})

    def _finalize_hooks(self, state):
        """Freeze each block's mass (Polyak-averaged D, final eigenvectors) for sampling."""
        state = super()._finalize_hooks(state)
        if self._lr_blocks:
            frozen = {kid: b.finalized() for kid, b in self._lr_blocks.items()}
            state = state._replace(ham_params={**state.ham_params, **frozen})
            log.debug("froze the low-rank mass of block(s) %s (averaged diagonal, final "
                      "eigenvectors)", sorted(frozen))
        return state
