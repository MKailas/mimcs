"""Adaptation for the shaped (nondiagonal) learned metric ``M(x) = D(x)^{1/2} A D(x)^{1/2}``.

Adapts each :class:`~mimcs.hmc.block_riemannian.ShapedLearnedBlock` during warmup, **decoupled** and
reusing the existing adapters (``docs/design/07_riemannian_hmc.md``):

* **``D(x)``** by *exactly* the diagonal metric KL-SGD of :class:`~mimcs.adaptation.MetricAdaptation`
  --- the same jitted step (:func:`~mimcs.adaptation.metric._make_kl_step`) and the same per-unit
  clip (:class:`~mimcs.adaptation.metric._PerUnitClip`): one adaptive threshold per target coordinate
  and one per shared leaf, a shared leaf's gradient divided by the coordinates it serves, a
  non-finite unit skipped rather than descended on, and the Polyak--Ruppert average of the iterate
  frozen for sampling. Its minimiser is the conditional gradient second moment
  ``diag E[g g^T | q_{-i}]``.

  This used to be a separate, cruder step --- one global gradient-norm clip, an undivided
  shared-leaf gradient, no guard, the raw iterate frozen --- and that difference, not the shape, is
  what the `irt_2pl` theta-shape ablation could not rule out (``tests/experiments/writeups/
  irt_two_stage_v2.md``, ``irt_shape_spectra.md``). Sharing the code is what keeps the two from
  drifting apart again.
* **``A``** by feeding the ``D(x)^{-1/2}``-whitened block score to the existing shape adapter --- a
  dense :class:`~mimcs.adaptation.score_mass._ScoreBlock` (``A = K K^T``) or the Sanger eigen-tracker
  :class:`~mimcs.adaptation.lowrank_mass._Sanger` (``A = I + sum_j gamma_j v_j v_j^T``). Because
  ``D`` fits the diagonal, the whitened score has ~unit-variance coordinates, so ``A`` is a
  **correlation** matrix --- well conditioned, which is what keeps the shape adapter stable.

A short burn-in (``shaped_min_samples``) lets ``D(x)`` settle before ``A`` starts, mirroring
:class:`~mimcs.adaptation.LowRankAdaptation`. As in ``MetricAdaptation`` the block metric is
*conditional* (its conditional score mean is ~zero), so by default the score is used uncentred ---
both ``D`` and ``A`` fit second moments, and the transient large scores are handled by the adapters'
own adaptive clips. ``metric_center_grad`` (the same flag ``MetricAdaptation`` reads, **off** by
default and for the same reason) subtracts a running score mean instead, so they fit covariances;
one centred score feeds *both* ``D(x)`` and ``A``'s whitening, as ``_LowRankBlock`` centres once for
its diagonal and its Sanger step. By default the whitening for ``A`` uses the *raw* ``D(x)``
iterate, as ``MetricAdaptation`` drives warmup with its raw iterate. ``metric_ema_warmup`` (again
``MetricAdaptation``'s key, off by default) makes an exponential moving average of the ``D(x)``
parameters drive the simulation, whiten ``A`` and be frozen for sampling, so the shape is fitted
under the metric the chain actually runs with. The parameters live in ``ham_params[id]`` as
``{"diag": ..., "shape": ...}``; this mixin owns that whole entry (the diagonal ``MetricAdaptation``
skips shaped blocks, which are not ``is_learned``). Adaptation runs during warmup only.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from .._logging import get_logger
from ..samplers.base import Phase
from ._stochastic import rm_gain, DEFAULT_KAPPA, DEFAULT_N0
from .metric import _make_kl_step, _PerUnitClip, _running_mean, _ema
from .score_mass import _ScoreBlock
from .lowrank_mass import _Sanger

log = get_logger(__name__)


class ShapedMetricAdaptation:
    """Mixin: adapt each shaped-metric block's ``D(x)`` (diagonal KL) and constant shape ``A``."""

    def _init_hooks(self, **kwargs):
        self._shp_kappa = float(kwargs.get("shaped_kappa", DEFAULT_KAPPA))
        self._shp_n0 = float(kwargs.get("shaped_n0", DEFAULT_N0))
        self._shp_clip_frac = float(kwargs.get("shaped_clip_frac", 0.1))
        self._shp_oja_const = float(kwargs.get("shaped_oja_const", 1.0))
        self._shp_min_samples = int(kwargs.get("shaped_min_samples", 50))
        # The same flag `MetricAdaptation` reads: D(x) is frozen for sampling as its Polyak average.
        self._shp_polyak = bool(kwargs.get("mass_polyak", True))
        # Likewise `metric_center_grad` (off by default): fit covariances rather than second moments.
        self._shp_center_grad = bool(kwargs.get("metric_center_grad", False))
        self._shp_mean_grad = None        # running mean of the score (centring)
        # And `metric_ema_warmup` (off by default): an EMA of D(x) drives warmup and whitens A.
        self._shp_ema_warmup = bool(kwargs.get("metric_ema_warmup", False))
        self._shp_ema: dict = {}          # EMA of the D-expr params per block id
        self._shp_count = 0
        self._shp_diag: dict = {}         # raw D-expr params per block id (Python-side SGD iterate)
        self._shp_clips: dict = {}        # _PerUnitClip per block id
        self._shp_step_fns: dict = {}     # jitted D-KL grad step per block id
        self._shp_shape: dict = {}        # shape adapter per block id (_ScoreBlock or _Sanger)
        self._shp_avg: dict = {}          # Polyak average of the D-expr params per block id
        self._shp_nonfinite: dict = {}    # skipped (non-finite) D(x) units per block id
        super()._init_hooks(**kwargs)

    def shaped_nonfinite_count(self, block_id: str | None = None) -> int:
        """Non-finite ``D(x)`` gradient units skipped (as :meth:`MetricAdaptation.metric_nonfinite_count`)."""
        if block_id is not None:
            return int(self._shp_nonfinite.get(block_id, 0))
        return int(sum(self._shp_nonfinite.values()))

    def _shaped_blocks(self):
        return [k for k in self.kinetics if getattr(k, "is_shaped", False)]

    def _make_diag_step(self, block):
        """The shared KL step on the ``D(x)`` params, with the current shape as an argument."""
        return _make_kl_step(
            lambda dp, shape, q, labels, score, lr: block.metric_loss(
                {"diag": dp, "shape": shape}, q, labels, score),
            block.size)

    def _new_shape_adapter(self, block):
        if block.shape_kind == "dense":
            return _ScoreBlock("dense", block.size, self._shp_n0, self._shp_kappa,
                               self._shp_clip_frac, False, False)     # center off: we whiten here
        return _Sanger(block.size, block.rank, self._shp_n0, self._shp_kappa,
                       self._shp_clip_frac, self._shp_oja_const)

    def _postprocess_hooks(self, state):
        state = super()._postprocess_hooks(state)
        if self._phase is not Phase.WARMUP:
            return state
        blocks = self._shaped_blocks()
        if not blocks:
            return state

        self._shp_count += 1
        lr = rm_gain(self._shp_count, self._shp_n0, self._shp_kappa)
        q = state.coordinate
        labels = getattr(state, "discrete", None)
        total = sum(state.potential_grads.values())        # total potential gradient (the score)
        score_np = np.asarray(total, dtype=float)

        # Centre the score by its running mean (E[score] -> 0 at stationarity), exactly as
        # `MetricAdaptation` does under the same flag. The *same* centred score then feeds D(x)'s KL
        # step and A's whitening, so the shape sees the covariance its `lambda_j` is meant to track.
        if self._shp_center_grad:
            if self._shp_mean_grad is None:
                self._shp_mean_grad = np.zeros_like(score_np)
            delta = score_np - self._shp_mean_grad
            self._shp_mean_grad += lr * delta              # SA running mean
            score_np = delta
            total = jnp.asarray(delta)

        lr_j = jnp.asarray(lr, float)
        new_ham = dict(state.ham_params)

        for k in blocks:
            if k.id not in self._shp_diag:
                self._shp_diag[k.id] = state.ham_params[k.id]["diag"]
                self._shp_clips[k.id] = _PerUnitClip(self._shp_diag[k.id], k.size)
                self._shp_step_fns[k.id] = self._make_diag_step(k)
                self._shp_shape[k.id] = self._new_shape_adapter(k)
                log.debug("shaped-metric adaptation started on block %r: %d coordinate(s), "
                          "%s shape, which starts after a %d-iteration D(x) burn-in",
                          k.id, k.size, k.shape_kind, self._shp_min_samples)
            shape_params = state.ham_params[k.id]["shape"]

            # 1) D(x): one per-unit clipped KL-SGD step on the raw D-expr iterate (MetricAdaptation's).
            diag = self._shp_diag[k.id]
            g, gnorm, shared_norms = self._shp_step_fns[k.id](diag, shape_params, q, labels,
                                                             total, lr_j)
            gain = rm_gain(self._shp_count, self._shp_n0, self._shp_kappa)
            diag, n_bad, applied = self._shp_clips[k.id].update(
                diag, g, gnorm, shared_norms, lr, gain, self._shp_clip_frac)
            if n_bad:
                self._shp_nonfinite[k.id] = self._shp_nonfinite.get(k.id, 0) + n_bad
                log.debug("shaped metric %r: skipped %d non-finite D(x) KL-gradient unit(s) at "
                          "warmup iteration %d (%d skipped so far)", k.id, n_bad,
                          self._shp_count, self._shp_nonfinite[k.id])
            if applied:
                self._shp_diag[k.id] = diag
                if self._shp_ema_warmup:
                    self._shp_ema[k.id] = (diag if k.id not in self._shp_ema else
                                           _ema(self._shp_ema[k.id], diag, lr))
                elif self._shp_polyak:
                    self._shp_avg[k.id] = (diag if k.id not in self._shp_avg else
                                           _running_mean(self._shp_avg[k.id], diag,
                                                         self._shp_count))
            # The D(x) the chain is simulated with and that whitens A: the EMA under
            # `metric_ema_warmup` (A is then fitted under the metric the chain actually runs
            # with), else the raw iterate.
            drive = self._shp_ema.get(k.id, diag) if self._shp_ema_warmup else diag

            # 2) whiten the block score by that D(x); 3) adapt A after the D burn-in.
            adapter = self._shp_shape[k.id]
            if self._shp_count > self._shp_min_samples:
                # The second, **eager** metric evaluation --- outside the jitted step --- so it
                # needs the labels too, or `D` here would silently be evaluated under different
                # labels from the step's.
                D = np.asarray(k._D(q, labels, drive), dtype=float)
                h = score_np[k.s:k.e] / np.sqrt(D)          # D(x)^{-1/2}-whitened block score
                if k.shape_kind == "dense":
                    adapter.update(h, self._shp_count)       # _ScoreBlock: K K^T = Cov(h) = A
                    shape_out = jnp.asarray(adapter.K)
                else:
                    adapter.step(h, lr, self._shp_count)     # _Sanger: A = I + sum gamma_j v_j v_j^T
                    shape_out = (jnp.asarray(adapter.W), jnp.asarray(adapter.gamma()))
            else:
                shape_out = shape_params                     # A = I while D(x) settles

            new_ham[k.id] = {"diag": drive, "shape": shape_out}

        return state._replace(ham_params=new_ham)

    def _finalize_hooks(self, state):
        """Freeze each shaped block's Polyak-averaged ``D(x)`` for sampling (or, under
        ``metric_ema_warmup``, the EMA that drove warmup), keeping its shape."""
        state = super()._finalize_hooks(state)
        frozen, what = ((self._shp_ema, "EMA") if self._shp_ema_warmup else
                        (self._shp_avg if self._shp_polyak else {}, "Polyak-averaged"))
        if frozen:
            ham = dict(state.ham_params)
            for bid, avg in frozen.items():
                ham[bid] = {"diag": avg, "shape": ham[bid]["shape"]}
            state = state._replace(ham_params=ham)
            log.debug("froze the %s D(x) of shaped block(s) %s after %d update(s)",
                      what, sorted(frozen), self._shp_count)
        skipped = self.shaped_nonfinite_count()
        if skipped:
            log.warning(
                "shaped metric: %d D(x) adaptation unit(s) over %d warmup iteration(s) were "
                "skipped for a non-finite KL gradient (per block: %s) --- usually a pathological "
                "initial fit.", skipped, self._shp_count, dict(self._shp_nonfinite))
        return state
