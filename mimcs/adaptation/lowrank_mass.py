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
  of a positive low-rank correction). The whitened score feeding the tracker is itself
  adaptively clipped by its norm (its own quantile threshold, ~10% of steps clipped, as for the
  diagonal SGD), so a transient huge score cannot blow up the eigenvector / eigenvalue estimates.
  By default the rank-J part is tracked not by Sanger but by the held-basis subspace iteration of
  :mod:`mimcs.adaptation._subspace` (``lowrank_tracker="held_basis"``), which does not read the
  warmup's autocorrelation as anisotropy; Sanger does (``lambda ~ 1 + (d - 1) rho^2``) and is
  kept as ``lowrank_tracker="sanger"``.

Under ``mass_target="geometric"`` (:mod:`mimcs.adaptation.score_mass`, ``spec.mass_adapt=
"geometric"``) both pieces follow the geometric mean of the score covariance and the moves' spread
``S``. ``D`` takes the spread-barrier step (minimiser ``sqrt(c E[g^2] / E[u^2])``). The tracker
still picks its directions from the whitened score, but each ``gamma_j`` is the one-dimensional
geometric mean along ``w_j`` --- ``sqrt(lambda_j / lambda^u_j) - 1``, with ``lambda^u_j`` the
mass-whitened spread's second moment along ``w_j`` (an auxiliary tracker stream). Keeping
``gamma = lambda - 1`` would be wrong here: with a geometric ``D`` the whitened score no longer
has unit diagonal, and on the funnel's ``x`` block it is isotropic at ``~E[e^{-v}]``, so ``J``
arbitrary directions would be stiffened straight back to the score's scale.

The two run at slightly separated time scales via a short burn-in (``lowrank_min_samples``):
until then only ``D`` adapts (``gamma = 0``, a pure diagonal mass), so the whitening settles
before the eigenvectors start tracking. All stochastic-approximation math is done on the Python
object in float64; only the packed ``(D, V)`` (with ``V[j] = sqrt(gamma_j) * sqrt(D) * v_j``)
crosses into the JAX state at ``ham_params[kinetic.id]``. The shared schedule ``(n + n0)^{-kappa}``
(kappa=0.75, n0=5) drives every update. Smoothing is optional and off by default, as for every
mass adaptation except the learned metrics (:mod:`mimcs.adaptation._ema`): ``mass_ema`` keeps an EMA of the *diagonal* ``D``
(log space) and freezes it for sampling, and ``mass_ema_warmup`` also lets it drive warmup --- the
EMA ``D`` is then both what is written and what whitens the score Sanger sees. The eigenvectors
and ``gamma`` are always their final estimate (subspace/sign averaging is ill-posed). Adaptation
runs during warmup only.
"""

from __future__ import annotations

import math

import numpy as np
import jax.numpy as jnp

from .._logging import get_logger
from ..samplers.base import Phase
from ._stochastic import rm_gain, DEFAULT_KAPPA, DEFAULT_N0
from ._ema import LogEMA, ema_options
from ._subspace import _NormClip, aux_gamma, make_tracker, tracker_kwargs
from ._geometric import MAX_LOG_STEP, SPREAD_CONSTANT, SpreadTracker

log = get_logger(__name__)


class _Sanger:
    """Deflationary-Oja (Sanger/GHA) tracker of the top-``J`` eigen-directions and eigenvalues of a
    ``D^{-1/2}``-whitened score covariance ``C_w``. Shared by the constant-``D`` low-rank mass and
    the shaped metric's constant shape ``A`` --- both fit ``A = I + sum_j gamma_j v_j v_j^T`` (with
    ``gamma_j = max(0, lambda_j - 1) >= 0``) to a whitened score. The caller supplies the already-
    whitened score and the Robbins--Monro gain / step count; the whitened score is adaptively
    clipped by its own norm (:class:`~mimcs.adaptation._subspace._NormClip`) so a transient huge
    score cannot blow up the estimates.

    **Reads autocorrelation as anisotropy.** Its subspace moves at an effective gain
    ``lr ||x||^2 ~ lr d``, O(1) through warmup, so on autocorrelated scores ``lambda`` approaches
    ``1 + (d - 1) rho^2`` even on an isotropic target. The held-basis tracker
    (``*_tracker="held_basis"``, the default) does not; see :mod:`mimcs.adaptation._subspace`.
    """

    def __init__(self, n, J, n0, kappa, clip_frac, oja_const):
        self._oja_const = oja_const
        self.W = np.eye(n)[:, :J].copy()             # (n, J): top-J whitened eigenvector est.
        self.lam = np.ones(J)                        # whitened eigenvalue est.  lambda_j
        self._clip = _NormClip(n, n0, kappa, clip_frac)   # whitened-score clip, init log n
        self.lam_aux = None                          # optional aux stream's moments along W
        self._aux_clip = _NormClip(n, n0, kappa, clip_frac)

    @property
    def log_clip_w(self):
        return self._clip.log_clip_w

    def step(self, x_w, lr, count, aux=None):
        """One Sanger step on an already-whitened score ``x_w`` (updates ``W``, ``lam`` in place);
        ``aux`` is an optional second whitened stream read along ``W`` (see ``aux_gamma``)."""
        x_w = self._clip(x_w, count)
        y = self.W.T @ x_w                                       # projections v_j^T x_w
        self.W += (self._oja_const * lr) * (
            np.outer(x_w, y) - self.W @ np.triu(np.outer(y, y)))  # Sanger deflation
        self.W, _ = np.linalg.qr(self.W)                        # re-orthonormalize (drift)
        self.lam += lr * (y ** 2 - self.lam)                    # running whitened eigenvalues
        if aux is not None:
            y_aux = self.W.T @ self._aux_clip(np.asarray(aux, dtype=float), count)
            if self.lam_aux is None:
                self.lam_aux = np.ones(len(self.lam))
            self.lam_aux += lr * (y_aux ** 2 - self.lam_aux)

    def gamma(self):
        """The rank-``J`` stiffenings (>= 0: PD + representable); see ``_subspace.aux_gamma``."""
        return aux_gamma(self.lam, self.lam_aux)


class _LowRankBlock:
    """SGD/Oja state and step for one low-rank kinetic block (size ``n``, rank ``J``)."""

    def __init__(self, n, J, n0, kappa, clip_frac, center_grad, mass_lr_const,
                 oja_const, min_samples, ema=False, ema_warmup=False, tracker="held_basis",
                 tracker_kwargs=None, target="score", spread="difference"):
        self._n0, self._kappa, self._clip_frac = n0, kappa, clip_frac
        # Geometric target (score_mass.py): D by the spread-barrier step, gamma by the geometric
        # mean along each tracked direction (the tracker's auxiliary stream, ``aux_gamma``).
        self.target, self._spread = target, spread
        self._c = SPREAD_CONSTANT[spread]
        self.mean_spread = np.zeros(n)
        self.capped: list = []          # iterations where the trust region bound (_geometric)
        self._center_grad = center_grad
        self._mass_lr_const = mass_lr_const
        self._min_samples = min_samples
        self.mean_grad = np.zeros(n)                 # running score mean (centring)
        self.log_D = np.zeros(n)                     # theta = log D  (D = exp theta)
        self.log_clip = math.log(n)                  # diagonal-SGD clip threshold init: log n
        self._tracker = make_tracker(tracker, n, J, n0, kappa, clip_frac, oja_const,
                                     **(tracker_kwargs or {}))   # low-rank eigen-tracker
        self.count = 0
        # EMA of D (log space) under mass_ema; under mass_ema_warmup it also drives warmup.
        self._ema = LogEMA("diagonal") if (ema or ema_warmup) else None
        self._ema_warmup = bool(ema_warmup)

    def _adaptive_clip(self, norm, log_clip):
        """Adaptive-quantile clip (as in ``ScoreMassAdaptation``): returns the scale factor and
        the updated log-threshold, converging so a ``clip_frac`` fraction of steps are clipped."""
        thr = math.exp(log_clip)
        scale = min(1.0, thr / (norm + 1e-12))
        exceeded = 1.0 if norm > thr else 0.0
        log_clip += rm_gain(self.count, self._n0, self._kappa) * (exceeded - self._clip_frac)
        return scale, log_clip

    def update(self, score, spread=None):
        """One warmup step from this block's score (and, for the geometric target, its raw spread
        vector); returns the packed ``(D, V)`` for the kinetic."""
        self.count += 1
        lr = rm_gain(self.count, self._n0, self._kappa)

        # Centre the score (E[g] -> 0 at stationarity, so the mass fits the covariance).
        g_c = score - self.mean_grad if self._center_grad else score
        if self._center_grad:
            self.mean_grad += lr * (score - self.mean_grad)
        u = None
        if self.target == "geometric":
            if spread is None:
                raise ValueError("the geometric mass target needs a spread vector")
            centre = self._center_grad or self._spread == "marginal"
            u = spread - self.mean_spread if centre else spread
            if centre:
                self.mean_spread += lr * (spread - self.mean_spread)

        # Diagonal D: log-mass SGD, minimiser D = E[g_c^2] (reuse of score_mass._step_diagonal),
        # or the geometric mean sqrt(c E[g^2] / E[u^2]) with the spread term as the barrier.
        D = np.exp(self.log_D)
        grad = 0.5 * ((1.0 if u is None else D * u ** 2 / self._c) - g_c ** 2 / D)
        scale, self.log_clip = self._adaptive_clip(float(np.sqrt(np.sum(grad ** 2))), self.log_clip)
        step = (self._mass_lr_const * lr) * scale * grad
        if u is not None:                                         # trust region (_geometric)
            if np.any(np.abs(step) > MAX_LOG_STEP):
                self.capped.append(self.count)
            step = np.clip(step, -MAX_LOG_STEP, MAX_LOG_STEP)
        self.log_D -= step
        D = np.exp(self.log_D)
        if self._ema is not None:
            self._ema.update(D, lr)                                 # EMA of D in log space
            if self._ema_warmup:
                D = np.asarray(self._ema.value(), float)            # the EMA drives warmup

        # Rank-J part: Sanger/GHA on the whitened score, after the diagonal burn-in. The whitened
        # score is adaptively clipped by its own norm (own threshold) so a transient huge score
        # cannot blow up the eigenvector/eigenvalue estimates -- the diagonal-mass analogue for the
        # low-rank part.
        if self.count > self._min_samples:
            # Tracker on the whitened score; the geometric target also hands it the mass-whitened
            # spread sqrt(D) u / sqrt(c), whose moments along each direction set gamma.
            aux = None if u is None else np.sqrt(D / self._c) * u
            self._tracker.step(g_c / np.sqrt(D), lr, self.count, aux)
        return self._pack(D)

    def _pack(self, D):
        """Pack (D, {v_j}, gamma_j) into lowrank's (D, V) with V[j] = sqrt(gamma_j) sqrt(D) v_j."""
        gamma = self._tracker.gamma()                               # >= 0: PD + representable
        V = np.sqrt(gamma)[:, None] * (np.sqrt(D)[None, :] * self._tracker.W.T)   # (J, n)
        return jnp.asarray(D), jnp.asarray(V)

    def finalized(self):
        """The frozen mass for sampling: the EMA of D (under ``mass_ema``) or the raw D, with the
        final W, gamma."""
        D = np.exp(self.log_D)
        if self._ema is not None and self._ema.value() is not None:
            D = np.asarray(self._ema.value(), float)
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
        self._lr_ema, self._lr_ema_warmup = ema_options(kwargs)   # both off by default
        # The rank-J tracker (``_subspace.TRACKERS``) and the held basis's knobs.
        self._lr_tracker = str(kwargs.get("lowrank_tracker", "held_basis"))
        self._lr_tracker_kwargs = tracker_kwargs(kwargs, "lowrank")
        self._lr_blocks: dict | None = None    # {kinetic id: _LowRankBlock}
        # The fitted target, shared with ScoreMassAdaptation (``mass_target``, :mod:`._geometric`).
        # The spread's options and window (the kernel's start coordinates; :mod:`._geometric`).
        self._lr_track = SpreadTracker(kwargs)
        self._lr_target, self._lr_spread = self._lr_track.target, self._lr_track.spread
        super()._init_hooks(**kwargs)

    def _lowrank_kinetics(self):
        """The low-rank kinetic blocks this mixin adapts (rank attribute, mass_mode None). A
        low-rank *shaped* metric block also has a ``rank`` but is adapted by
        :class:`~mimcs.adaptation.ShapedMetricAdaptation`, so it is excluded here."""
        return [k for k in self.kinetics
                if getattr(k, "rank", None) is not None and k.mass_mode is None
                and not getattr(k, "is_shaped", False)]

    def _preprocess_hooks(self, state):
        state = super()._preprocess_hooks(state)
        self._lr_track.capture(state)
        return state

    def _postprocess_hooks(self, state):
        spread = None               # read off the kernel's output, before other hooks run
        if self._lr_track.active and self._phase is Phase.WARMUP:
            spread = self._lr_track.observe(state).u     # a constant mass needs only u
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
                    self._lr_min_samples, self._lr_ema, self._lr_ema_warmup,
                    self._lr_tracker, self._lr_tracker_kwargs,
                    target=self._lr_target, spread=self._lr_spread)
                log.debug("low-rank mass adaptation started on block %r: %d coordinate(s), "
                          "rank %d, %s tracker, %s target", k.id, score.shape[0], k.rank,
                          self._lr_tracker, self._lr_target)
            u = None if spread is None else np.asarray(k._gather(spread), dtype=float)
            new[k.id] = self._lr_blocks[k.id].update(score, u)
        return state._replace(ham_params={**state.ham_params, **new})

    def _finalize_hooks(self, state):
        """Freeze each block's mass (the EMA of D under ``mass_ema``, final eigenvectors)."""
        state = super()._finalize_hooks(state)
        if self._lr_blocks:
            frozen = {kid: b.finalized() for kid, b in self._lr_blocks.items()}
            state = state._replace(ham_params={**state.ham_params, **frozen})
            log.debug("froze the low-rank mass of block(s) %s (%s diagonal, final "
                      "eigenvectors)", sorted(frozen), "EMA" if self._lr_ema else "raw")
        return state
