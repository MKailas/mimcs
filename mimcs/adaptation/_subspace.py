"""Streaming low-rank eigen-trackers for a whitened score, shared by the low-rank mass and the shaped
metric's low-rank shape (``A = I + sum_j gamma_j v_j v_j^T``, ``gamma_j = max(0, lambda_j - 1)``).

Two trackers, one interface (``step(x_w, lr, count)``, ``W`` (n, J), ``lam`` (J), ``gamma()``,
``log_clip_w``):

* :class:`_HeldBasisTracker` --- streaming subspace iteration with the basis **held fixed** over
  a block of steps. **The default** (``lowrank_tracker`` / ``shaped_tracker = "held_basis"``).
* :class:`~mimcs.adaptation.lowrank_mass._Sanger` --- Sanger's rule (a deflationary Oja), the
  original tracker, kept selectable (``"sanger"``) for comparison.

Why the second exists (``tests/experiments/writeups/shape_direction.md``,
``shape_estimator_bench.md``). A warmup's scores are autocorrelated, and Sanger reads that
autocorrelation as anisotropy. Its subspace update ``W += lr (x y^T - ...)`` moves along the
current score at an effective gain ``lr ||x||^2 ~ lr d``, which is O(1) through warmup (the
classical Oja stability condition is ``lr ||x||^2 << 1``). So ``W`` is roughly the last score's
direction, and ``lambda = E[(W_{t-1}^T x_t)^2] ~ 1 + (d - 1) rho^2`` for a lag-1 score correlation
``rho``:

* on an exactly isotropic stream it reads ``gamma`` 31 at IACT 10 (d = 100) and 249 at IACT 3
  (d = 1000; the law gives 251);
* on `irt_2pl`'s real warmup scores it reads 21-38 before the collapse, where a time-shuffled
  replay of the same scores reads 2.5.

The target --- the marginal second moment ``E[x x^T]`` --- is unchanged by autocorrelation. Only the
estimator's coupling to its own data is at fault, and that is what holding the basis removes. With
``Q`` fixed for a block, ``x (Q^T x)^T`` is an unbiased estimate of ``E[x x^T] Q`` however
correlated the draws (Mitliagkas, Caramanis & Jain 2013, memory-limited streaming PCA). The
eigenvalues are Rayleigh-Ritz values of ``Q^T x x^T Q`` read only on samples taken ``gap`` steps
after ``Q`` was last chosen --- out of sample --- so they carry neither the chasing bias nor the
in-sample (Marchenko-Pastur) inflation of a basis fitted to the same draws. A spiked-covariance
shrinkage gate was tried on top and *over*-corrected, so it is not here.
"""

from __future__ import annotations

import math

import numpy as np

from ._stochastic import rm_gain

TRACKERS = ("sanger", "held_basis")


class _NormClip:
    """Adaptive clip of a whitened score by its norm, with its own log-quantile threshold.

    The threshold starts at ``log n`` and converges so that a ``clip_frac`` fraction of steps are
    clipped, so one transient huge score cannot blow up the estimates. This is the clip that used to
    live inline in ``_Sanger.step``, shared so both trackers clip identically.
    """

    def __init__(self, n, n0, kappa, clip_frac):
        self._n0, self._kappa, self._clip_frac = n0, kappa, clip_frac
        self.log_clip_w = math.log(n)

    def __call__(self, x_w, count):
        thr = math.exp(self.log_clip_w)
        norm = float(np.sqrt(np.sum(x_w ** 2)))
        scale = min(1.0, thr / (norm + 1e-12))
        self.log_clip_w += rm_gain(count, self._n0, self._kappa) * (
            (1.0 if norm > thr else 0.0) - self._clip_frac)
        return scale * x_w


class _HeldBasisTracker:
    """Top-``J`` eigen-directions and eigenvalues of a whitened score's second moment, by streaming
    subspace iteration with a held basis.

    State: an orthonormal basis ``Q`` (n, m), with ``m = J + oversample`` (default ``2J``, capped at
    ``n``); ``Y``, the RM average of ``x (Q^T x)^T`` (~ ``E[x x^T] Q``); and ``T``, the RM average
    of ``(Q^T x)(Q^T x)^T`` (~ ``Q^T E[x x^T] Q``) with its total weight for bias correction.

    * Per step, O(n m): clip; ``y = Q^T x``; update ``Y``. Update ``T`` only if the step is more
      than ``gap`` steps after the last basis change.
    * Every ``block`` steps, O(n m^2):
      1. Rayleigh-Ritz on ``T``: its eigen-pairs give ``W = Q U[:, :J]`` and ``lam``.
      2. One power step ``Q' = qr(Y)``.
      3. The memory is rotated into the new basis with ``R = Q^T Q'`` (``Y <- Y R``,
         ``T <- R^T T R``), as in History PCA.

    Before the first block ends, ``W`` holds the first ``J`` axes and ``lam = 1`` (so ``gamma = 0``).
    The initial basis is the QR of the first ``m`` scores, which is data-driven and deterministic
    and draws nothing from the sampler's RNG stream. ``lr`` is the caller's Robbins-Monro gain, as
    for ``_Sanger``.
    """

    def __init__(self, n, J, n0, kappa, clip_frac, block=50, oversample=None, gap=5):
        self.n, self.J = int(n), int(J)
        self.m = min(self.n, self.J + (self.J if oversample is None else int(oversample)))
        self.block, self.gap = int(block), int(gap)
        if self.block <= self.gap:
            raise ValueError(f"held-basis block ({block}) must exceed its gap ({gap})")
        self._clip = _NormClip(self.n, n0, kappa, clip_frac)
        self.W = np.eye(self.n)[:, :self.J].copy()
        self.lam = np.ones(self.J)
        self.Q = None
        self._init_buf = []
        self.Y = np.zeros((self.n, self.m))
        self.T = np.zeros((self.m, self.m))
        self.wT = 0.0                   # total weight of T's RM average (bias correction)
        self.since = 0                  # steps since the basis last changed

    @property
    def log_clip_w(self):
        return self._clip.log_clip_w

    def gamma(self):
        """The rank-``J`` stiffenings ``gamma_j = max(0, lambda_j - 1)`` (>= 0: PD + representable)."""
        return np.maximum(0.0, self.lam - 1.0)

    def step(self, x_w, lr, count):
        """One step on an already-whitened score ``x_w`` with RM gain ``lr``."""
        x = self._clip(np.asarray(x_w, dtype=float), count)
        if self.Q is None:
            self._init_buf.append(x.copy())
            if len(self._init_buf) >= self.m:
                self.Q, _ = np.linalg.qr(np.stack(self._init_buf, axis=1))
                self._init_buf = None
            return
        y = self.Q.T @ x
        self.Y += lr * (np.outer(x, y) - self.Y)
        self.since += 1
        if self.since > self.gap:
            self.T += lr * (np.outer(y, y) - self.T)
            self.wT = (1.0 - lr) * self.wT + lr
        if self.since >= self.block:
            self._block_end()

    def _block_end(self):
        self.since = 0
        if self.wT > 0.0:
            That = self.T / self.wT
            ev, U = np.linalg.eigh(0.5 * (That + That.T))
            ev, U = ev[::-1], U[:, ::-1]
            self.W = self.Q @ U[:, :self.J]
            self.lam = ev[:self.J].copy()
        Qn, _ = np.linalg.qr(self.Y)
        R = self.Q.T @ Qn
        self.Y = self.Y @ R
        self.T = R.T @ self.T @ R
        self.Q = Qn


def make_tracker(kind, n, J, n0, kappa, clip_frac, oja_const=1.0, block=50, oversample=None,
                 gap=5):
    """Build the named low-rank tracker (``"sanger"`` or ``"held_basis"``) for an ``n``-dimensional
    whitened score. ``oja_const`` is read only by Sanger; ``block``, ``oversample`` and ``gap`` only
    by the held basis."""
    if kind == "sanger":
        from .lowrank_mass import _Sanger
        return _Sanger(n, J, n0, kappa, clip_frac, oja_const)
    if kind == "held_basis":
        return _HeldBasisTracker(n, J, n0, kappa, clip_frac, block, oversample, gap)
    raise ValueError(f"unknown low-rank tracker {kind!r}; expected one of {TRACKERS}")


def tracker_kwargs(kwargs, prefix):
    """The held basis's knobs from ``algo_kwargs`` under ``prefix`` (``"lowrank"`` / ``"shaped"``):
    ``<prefix>_block`` (50), ``<prefix>_oversample`` (``None`` -> ``J``), ``<prefix>_gap`` (5).
    The tracker kind itself is ``<prefix>_tracker``, validated when the tracker is built."""
    tracker = kwargs.get(f"{prefix}_tracker", "held_basis")
    if tracker not in TRACKERS:
        raise ValueError(f"algo_kwargs {prefix}_tracker={tracker!r}; expected one of {TRACKERS}")
    over = kwargs.get(f"{prefix}_oversample")
    return {"block": int(kwargs.get(f"{prefix}_block", 50)),
            "oversample": None if over is None else int(over),
            "gap": int(kwargs.get(f"{prefix}_gap", 5))}
