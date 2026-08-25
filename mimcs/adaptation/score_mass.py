"""Score-covariance (KL) mass-matrix adaptation --- diagonal or dense, per block.

A mixin (``docs/design/02_sampler_classes.md``) that adapts the mass ``M`` of an ordinary
quadratic kinetic to the covariance of the score (the potential gradient ``g = grad_q V``) by
SGD on the KL objective. For a **diagonal** mass,

    L(M) = 1/2 sum_d ( log M_d + g_d^2 / M_d ),

whose per-sample minimiser is ``M_d = g_d^2`` and expected-loss minimiser the score second
moment ``M_d = E[g_d^2]``; the kinetic stores ``M^{-1}``, so the mixin maintains ``theta = log
M`` and writes ``exp(-theta)``. For a **dense** mass the analogous objective is
``L(M) = 1/2 (log det M + g^T M^{-1} g)``, minimised at ``M = E[g g^T]`` (the score covariance).
We SGD on the lower Cholesky ``K`` of the **mass itself** (``M = K K^T``) --- the loss is
``sum log K_ii + 1/2 |K^{-1} g|^2``, with (writing ``w = K^{-1} s``, ``a = K^{-T} w = M^{-1} s``)
gradient ``-a w^T`` on the off-diagonal and ``1 - K_ii a_i w_i`` on ``log K_ii``, and optimum
``K K^T = Cov(g)``. Its diagonal is carried in **log** form (log-Cholesky) to stay positive; each
step costs two triangular solves. :class:`~mimcs.hmc.DenseQuadraticKinetic` stores the lower
Cholesky ``L`` of ``M^{-1}``, formed from ``K`` per step (``chol(inv(K K^T))``, stable because
``K`` stays well-conditioned).

Parameterizing the **mass** rather than the **inverse** mass is what makes dense adaptation
robust. SGD on ``chol(M^{-1})`` has a gradient ``s u^T`` with ``u = L^T s``, whose norm *grows*
with ``||L||``; under the adaptive clip this is a positive feedback that drives the iterate to a
singular / catastrophically ill-conditioned mass on stiff targets (the enormous scores typical
early in warmup) and never recovers. The ``K`` gradient instead carries ``K^{-1}/K^{-T}``, so its
norm *shrinks* as ``||K||`` grows: an early overshoot relaxes back rather than running away. (The
diagonal mode has no off-diagonal to run away and needs no such care; it stays on ``log M``.)

Like :class:`MassMatrixAdaptation` the mixin is **list-aware**: it adapts *every* diagonal/dense
kinetic block over that block's own coordinate slice (each block keeps its own SGD state; a
single whole-space kinetic is just the one-block case). SGD uses the shared schedule
``(n + n0)^{-kappa}`` (``kappa = 0.75``, ``n0 = 5``; :mod:`mimcs.adaptation._stochastic`). Two
Kailas--Vihola--Wallin regularizations shape the score before it enters the loss (both applying
to either mode):

* **Gradient clipping.** The KL gradient is clipped to an adaptive threshold so a target
  fraction (default 10%) of steps are clipped --- an online log-scale quantile of the gradient
  norm, ``log_clip += gain * (exceeded - clip_frac)``, updated with the converging gain
  ``(n + n0)^{-kappa}`` so heavy-tailed ``g^2`` cannot bias the mass upward from the first steps.
  For a **dense** block this is (unchanged) one threshold on the whole Cholesky-gradient norm,
  dimension-aware init ``log d`` (there is genuine cross-coordinate structure to protect: an
  off-diagonal entry mixes a *pair* of coordinates, so it cannot be clipped alone). For a
  **diagonal** block it is instead ``d`` **independent** thresholds, one per coordinate, each
  tracking only that coordinate's own gradient component (init ``log 1 = 0``, since a lone
  scalar's "norm" is itself) --- so one coordinate's large gradient no longer scales down every
  other coordinate's update. This makes the diagonal mass adapt as ``d`` separate one-coordinate
  score-mass problems, exactly as if each coordinate were its own tiny dense block, rather than
  one norm-coupled ``d``-vector.
* **Gradient mean estimation** (``score_mass_center_grad``, default on). A running mean
  ``mean_grad`` of the score is maintained (same schedule) and the loss uses the *centred*
  score ``g - mean_grad``, so the mass fits the score *covariance* rather than its second
  moment. At stationarity the score is mean-zero (integration by parts), so ``mean_grad -> 0``
  and the centring is inert; early on it removes the transient downhill offset that would
  otherwise inflate the mass and over-damp motion toward the mode.

Each block's SGD state (mass parameters, mean-gradient, clip threshold, clip-threshold history)
lives on the Python object; only the mass parameters cross into the JAX state. Adaptation runs
during warmup only. A block's clip-threshold trajectory is recorded for inspection (see
:meth:`warmup_log_clip`).

Smoothing of the mass estimate is **optional and off by default** (``mass_polyak``): the raw SGD
iterate is used directly. When enabled the smoothed mass is an exponential moving average of the
raw estimate (the Kailas--Vihola--Wallin scheme; see :class:`_ScoreBlock`), frozen for sampling,
and with ``score_mass_polyak_warmup`` also used to drive warmup. It is kept for cases that might
benefit but is not needed for the targets in our repertoire and slows the mass's learning.
"""

from __future__ import annotations

import math

import numpy as np
import jax.numpy as jnp
from scipy.linalg import solve_triangular

from .._logging import get_logger
from ..samplers.base import Phase
from ._stochastic import rm_gain, DEFAULT_KAPPA, DEFAULT_N0

log = get_logger(__name__)


class _ScoreBlock:
    """The KL-SGD score-mass state and step for one diagonal/dense kinetic block.

    Smoothing follows the Kailas--Vihola--Wallin paper: the smoothed mass is an **exponential
    moving average** of the raw SGD estimate with the Robbins--Monro gain, ``M_n = eta_n Mhat_n
    + (1 - eta_n) M_{n-1}`` (a linear EMA in mass space), and the clip threshold is initialised
    at ``log d`` and updated with the plain gain ``eta_n``.
    """

    def __init__(self, mode, d, n0, kappa, clip_frac, center_grad, smooth,
                 mass_lr_const=1.0):
        self.mode = mode
        self._mass_lr_const = float(mass_lr_const)      # c in the mass SGD lr  c*(n0+n)^-kappa
        self._n0, self._kappa, self._clip_frac = n0, kappa, clip_frac
        self._center_grad = center_grad
        self.mean_grad = np.zeros(d)                    # running mean of the score (centring)
        if mode == "dense":
            self.K = np.eye(d)                          # K = chol(M); K = I => M = K K^T = I at start
            self.K_logdiag = np.zeros(d)                # log of K's diagonal (keeps K PD)
            self.log_clip = math.log(d)                  # one threshold for the whole block
        else:
            self.log_mass = np.zeros(d)                 # theta = log(diagonal mass M)
            # ONE clip threshold PER COORDINATE, not one for the whole block: each coordinate's
            # own KL gradient is already a single scalar (d/dtheta_d), so its own "clip norm" has
            # dimension 1 and the dimension-aware init is log(1) = 0. Tracking d independent
            # thresholds (rather than one threshold on the d-vector's norm) means one coordinate
            # with a large gradient no longer scales down every other coordinate's update --- the
            # diagonal mass adapts as d independent one-coordinate problems (see module docstring).
            self.log_clip = np.zeros(d)
        self.log_clip_history: list = []
        self._smooth = smooth
        self.mass_ema = None                            # EMA of the raw mass M (paper's smoother)

    def _clip_scale(self, gnorm, count):
        """Adaptive gradient-norm clip: ~clip_frac clipped, threshold converges. Records it.

        Scalar for a dense block (one threshold for the whole Cholesky gradient). A ``(d,)``
        array for a diagonal block: each coordinate tracks its own threshold from its own
        gradient component, independently of every other coordinate.
        """
        thr = np.exp(self.log_clip)
        scale = np.minimum(1.0, thr / (gnorm + 1e-12))
        exceeded = np.asarray(gnorm > thr, dtype=float)
        self.log_clip = self.log_clip + rm_gain(count, self._n0, self._kappa) * (
            exceeded - self._clip_frac)
        self.log_clip_history.append(self.log_clip)
        return scale

    def update(self, score, count):
        """One SGD step from this block's score; returns the kinetic's mass param (``M^{-1}``
        diagonal, or the lower Cholesky ``L`` of ``M^{-1}`` for a dense block)."""
        # Centre the score by its running mean (E[score] -> 0 at stationarity): the mass then
        # fits the score covariance rather than its second moment.
        centred = score - self.mean_grad if self._center_grad else score
        lr = rm_gain(count, self._n0, self._kappa)
        # Mass-parameter SGD learning rate: c * (n0 + n)^{-kappa}. c defaults to 1; c = 1/d
        # normalizes the per-step change of the mass norm to ~1 rather than ~d (the clipped step
        # norm is lr*thr, and thr can be O(d)). The score-mean/EMA smoothers keep the bare schedule.
        mass_lr = self._mass_lr_const * lr
        mass = (self._step_dense(centred, mass_lr, count) if self.mode == "dense"
                else self._step_diagonal(centred, mass_lr, count))
        if self._center_grad:
            self.mean_grad += lr * (score - self.mean_grad)      # SA running mean
        if self._smooth:
            M = self._raw_mass()                                 # the raw mass estimate Mhat_n
            self.mass_ema = M if self.mass_ema is None else self.mass_ema + lr * (M - self.mass_ema)
        return mass

    def _raw_mass(self):
        """The current raw mass ``Mhat`` (diagonal vector, or dense matrix)."""
        if self.mode == "dense":
            return self.K @ self.K.T                             # M = K K^T
        return np.exp(self.log_mass)                             # M = exp(theta)

    def _step_diagonal(self, centred, lr, count):
        M = np.exp(self.log_mass)
        grad = 0.5 * (1.0 - centred ** 2 / M)                    # dL/dtheta, theta = log M
        # Coordinate-wise clip: each coordinate's gradient (a single scalar) is clipped against
        # its OWN threshold, not the norm of the whole d-vector -- see _clip_scale / __init__.
        scale = self._clip_scale(np.abs(grad), count)
        self.log_mass -= lr * scale * grad
        return jnp.asarray(np.exp(-self.log_mass))               # M^{-1} for the kinetic

    def _step_dense(self, centred, lr, count):
        # SGD on the log-Cholesky of the MASS itself, M = K K^T (Kailas--Vihola--Wallin). Loss
        # sum log K_ii + 1/2 |K^{-1} s|^2, optimum K K^T = Cov(s). With w = K^{-1} s and
        # a = K^{-T} w = M^{-1} s, the gradient of the quadratic term wrt K is -a w^T and the
        # log-diagonal barrier is 1 - K_ii a_i w_i.
        #
        # This is the mass, not the inverse mass: the alternative of SGD on chol(M^{-1}) has a
        # gradient s u^T with u = L^T s, whose norm GROWS with ||L|| -- a positive feedback that,
        # under the adaptive gradient clip, drives the iterate to a singular/ill-conditioned mass
        # on stiff targets (large early scores) and never recovers. Here the gradient carries
        # K^{-1}/K^{-T}, so its norm SHRINKS as ||K|| grows: an overshoot relaxes instead of
        # running away, which makes adaptation robust to the huge scores typical early in warmup.
        K = self.K
        w = solve_triangular(K, centred, lower=True)             # w = K^{-1} s
        a = solve_triangular(K.T, w, lower=False)                # a = K^{-T} w = M^{-1} s
        outer = np.outer(a, w)                                   # a w^T
        g_off = -np.tril(outer, -1)                              # d(1/2|K^{-1}s|^2)/dK offdiag = -(a w^T)
        g_ell = 1.0 - np.diag(K) * np.diag(outer)               # d/d(log K_ii) = 1 - K_ii a_i w_i
        scale = self._clip_scale(float(np.sqrt(np.sum(g_off ** 2) + np.sum(g_ell ** 2))), count)
        self.K_logdiag -= lr * scale * g_ell
        K = np.tril(K, -1) - lr * scale * g_off
        np.fill_diagonal(K, np.exp(self.K_logdiag))
        self.K = K
        # kinetic stores L = chol(M^{-1}); stable to form since K stays well-conditioned.
        return jnp.asarray(np.linalg.cholesky(np.linalg.inv(K @ K.T)))

    def frozen(self):
        """The smoothed mass as ``M^{-1}`` (for the kinetic) --- to drive warmup or freeze for
        sampling --- or ``None`` when smoothing is off / not yet started."""
        if not self._smooth or self.mass_ema is None:
            return None
        if self.mode == "dense":
            return jnp.asarray(np.linalg.cholesky(np.linalg.inv(self.mass_ema)))   # chol of M^{-1}
        return jnp.asarray(1.0 / self.mass_ema)                                     # M^{-1}


class ScoreMassAdaptation:
    """Mixin: SGD-adapt each diagonal/dense block's mass to its score covariance (KL)."""

    def _init_hooks(self, **kwargs):
        self._sm_n0 = float(kwargs.get("score_mass_n0", DEFAULT_N0))
        self._sm_kappa = float(kwargs.get("score_mass_kappa", DEFAULT_KAPPA))
        self._sm_clip_frac = float(kwargs.get("score_mass_clip_frac", 0.1))
        self._sm_center_grad = bool(kwargs.get("score_mass_center_grad", True))
        # Constant multiplier c on the mass SGD learning rate c*(n0+n)^-kappa. Default 1; may be
        # a float or the string "1/d" (resolved per block to 1/dim).
        self._sm_lr_const = kwargs.get("score_mass_lr_const", 1.0)
        # Mass smoothing (EMA of the raw estimate) is OPTIONAL and OFF BY DEFAULT: it is not
        # needed for any target in our repertoire and it slows the mass's learning, but it is
        # kept for cases where it might help. ``mass_polyak`` freezes the EMA for sampling;
        # ``score_mass_polyak_warmup`` (experimental) additionally lets the EMA DRIVE warmup
        # (Polyak as an early-warmup regularizer) and implies ``mass_polyak``.
        self._sm_polyak_warmup = bool(kwargs.get("score_mass_polyak_warmup", False))
        self._sm_polyak = bool(kwargs.get("mass_polyak", False)) or self._sm_polyak_warmup
        self._sm_count = 0
        self._sm_blocks: dict | None = None    # {kinetic id: _ScoreBlock}
        super()._init_hooks(**kwargs)

    def _score_kinetics(self):
        """The diagonal- and dense-mass kinetic blocks this mixin adapts (may be empty when
        every block carries a learned metric, adapted by :class:`MetricAdaptation` instead)."""
        return [k for k in self.kinetics if k.mass_mode in ("diagonal", "dense")]

    def warmup_log_clip(self, block_id=None) -> np.ndarray:
        """A block's clip-threshold trajectory (the sole / first block if ``block_id`` is None).

        One scalar per warmup iteration. A diagonal block's threshold is now PER COORDINATE (a
        ``(d,)`` vector each iteration, see :class:`_ScoreBlock`); this reports the
        cross-coordinate mean at each iteration, so a caller that just wants "is it converging"
        (the diagnostic plot, the tests) still gets a single 1-D trajectory. The raw per-coordinate
        history is on the block object itself (``self._sm_blocks[block_id].log_clip_history``) for
        anyone who needs the detail.
        """
        blocks = self._sm_blocks or {}
        if block_id is None:
            block_id = next(iter(blocks), None)
        hist = blocks[block_id].log_clip_history if block_id in blocks else []
        return np.asarray([np.mean(h) for h in hist], dtype=float)

    def _postprocess_hooks(self, state):
        state = super()._postprocess_hooks(state)
        if self._phase is not Phase.WARMUP:
            return state

        kinetics = self._score_kinetics()
        if not kinetics:            # every block is a learned metric -> nothing for this mixin
            return state
        if self._sm_blocks is None:
            self._sm_blocks = {}
        self._sm_count += 1
        total_grad = sum(state.potential_grads.values())
        new = {}
        for k in kinetics:
            score = np.asarray(k._gather(total_grad), dtype=float)
            if k.id not in self._sm_blocks:
                c = self._sm_lr_const
                c = 1.0 / score.shape[0] if isinstance(c, str) and c == "1/d" else float(c)
                self._sm_blocks[k.id] = _ScoreBlock(
                    k.mass_mode, score.shape[0], self._sm_n0, self._sm_kappa,
                    self._sm_clip_frac, self._sm_center_grad, self._sm_polyak,
                    mass_lr_const=c)
                log.debug("score-mass adaptation started on block %r: %s mass over %d "
                          "coordinate(s), lr constant %.3g, centring %s, smoothing %s",
                          k.id, k.mass_mode, score.shape[0], c,
                          "on" if self._sm_center_grad else "off",
                          "on" if self._sm_polyak else "off")
            b = self._sm_blocks[k.id]
            raw = b.update(score, self._sm_count)                     # advance the raw iterate
            # warmup uses the raw iterate, unless polyak_warmup: then the suffix average drives it.
            new[k.id] = jnp.asarray(b.frozen()) if self._sm_polyak_warmup else raw
        return state._replace(ham_params={**state.ham_params, **new})

    def _finalize_hooks(self, state):
        """Freeze each block's Polyak-averaged mass for sampling (see :mod:`._polyak`)."""
        state = super()._finalize_hooks(state)
        if self._sm_blocks:
            frozen = {kid: jnp.asarray(b.frozen())
                      for kid, b in self._sm_blocks.items() if b.frozen() is not None}
            if frozen:
                state = state._replace(ham_params={**state.ham_params, **frozen})
                log.debug("froze the smoothed score mass of block(s) %s after %d update(s)",
                          sorted(frozen), self._sm_count)
            else:
                log.debug("score mass frozen at its last raw iterate (%d update(s); smoothing "
                          "is off)", self._sm_count)
        return state
