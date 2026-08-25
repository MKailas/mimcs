"""Adaptation for the shaped (nondiagonal) learned metric ``M(x) = D(x)^{1/2} A D(x)^{1/2}``.

Adapts each :class:`~mimcs.hmc.block_riemannian.ShapedLearnedBlock` during warmup, **decoupled** and
reusing the existing adapters (``docs/design/07_riemannian_hmc.md``):

* **``D(x)``** by the diagonal metric KL-SGD (as :class:`~mimcs.adaptation.MetricAdaptation`): SGD on
  the block's ``metric_loss`` (the diagonal KL over ``D(x)`` only), with an adaptive gradient-norm
  clip. Its minimiser is the conditional gradient second moment ``diag E[g g^T | q_{-i}]``.
* **``A``** by feeding the ``D(x)^{-1/2}``-whitened block score to the existing shape adapter --- a
  dense :class:`~mimcs.adaptation.score_mass._ScoreBlock` (``A = K K^T``) or the Sanger eigen-tracker
  :class:`~mimcs.adaptation.lowrank_mass._Sanger` (``A = I + sum_j gamma_j v_j v_j^T``). Because
  ``D`` fits the diagonal, the whitened score has ~unit-variance coordinates, so ``A`` is a
  **correlation** matrix --- well conditioned, which is what keeps the shape adapter stable.

A short burn-in (``shaped_min_samples``) lets ``D(x)`` settle before ``A`` starts, mirroring
:class:`~mimcs.adaptation.LowRankAdaptation`. As in ``MetricAdaptation`` the block metric is
*conditional* (its conditional score mean is ~zero), so the score is used uncentred --- both ``D``
and ``A`` fit second moments, and the transient large scores are handled by the adapters' own
adaptive clips. The parameters live in ``ham_params[id]`` as ``{"diag": ..., "shape": ...}``; this
mixin owns that whole entry (the diagonal ``MetricAdaptation`` skips shaped blocks, which are not
``is_learned``). Adaptation runs during warmup only.
"""

from __future__ import annotations

import math

import numpy as np
import jax
import jax.numpy as jnp

from .._logging import get_logger
from ..samplers.base import Phase
from ._stochastic import rm_gain, DEFAULT_KAPPA, DEFAULT_N0
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
        self._shp_count = 0
        self._shp_diag: dict = {}         # raw D-expr params per block id (Python-side SGD iterate)
        self._shp_log_clip: dict = {}     # D-KL adaptive clip threshold per block id
        self._shp_step_fns: dict = {}     # jitted D-KL grad step per block id
        self._shp_shape: dict = {}        # shape adapter per block id (_ScoreBlock or _Sanger)
        super()._init_hooks(**kwargs)

    def _shaped_blocks(self):
        return [k for k in self.kinetics if getattr(k, "is_shaped", False)]

    def _make_diag_step(self, block):
        """A jitted step: the diagonal KL-loss gradient wrt the D-expr params, and its norm."""
        def step(diag, shape, q, score, lr):
            g = jax.grad(
                lambda dp: block.metric_loss({"diag": dp, "shape": shape}, q, score))(diag)
            gnorm = jnp.sqrt(sum(jnp.sum(leaf ** 2) for leaf in jax.tree_util.tree_leaves(g)))
            return g, gnorm
        return jax.jit(step)

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
        total = sum(state.potential_grads.values())        # total potential gradient (the score)
        score_np = np.asarray(total, dtype=float)
        lr_j = jnp.asarray(lr, float)
        new_ham = dict(state.ham_params)

        for k in blocks:
            if k.id not in self._shp_diag:
                self._shp_diag[k.id] = state.ham_params[k.id]["diag"]
                self._shp_log_clip[k.id] = math.log(k.size)
                self._shp_step_fns[k.id] = self._make_diag_step(k)
                self._shp_shape[k.id] = self._new_shape_adapter(k)
                log.debug("shaped-metric adaptation started on block %r: %d coordinate(s), "
                          "%s shape, which starts after a %d-iteration D(x) burn-in",
                          k.id, k.size, k.shape_kind, self._shp_min_samples)
            shape_params = state.ham_params[k.id]["shape"]

            # 1) D(x): one clipped KL-SGD step on the raw D-expr iterate.
            diag = self._shp_diag[k.id]
            g, gnorm = self._shp_step_fns[k.id](diag, shape_params, q, total, lr_j)
            gn = float(gnorm)
            thr = math.exp(self._shp_log_clip[k.id])
            scale = lr * min(1.0, thr / (gn + 1e-12))
            diag = jax.tree_util.tree_map(lambda w, gw: w - scale * gw, diag, g)
            self._shp_diag[k.id] = diag
            self._shp_log_clip[k.id] += rm_gain(self._shp_count, self._shp_n0, self._shp_kappa) * (
                (1.0 if gn > thr else 0.0) - self._shp_clip_frac)

            # 2) whiten the block score by the current D(x); 3) adapt A after the D burn-in.
            adapter = self._shp_shape[k.id]
            if self._shp_count > self._shp_min_samples:
                D = np.asarray(k._D(q, diag), dtype=float)
                h = score_np[k.s:k.e] / np.sqrt(D)          # D(x)^{-1/2}-whitened block score
                if k.shape_kind == "dense":
                    adapter.update(h, self._shp_count)       # _ScoreBlock: K K^T = Cov(h) = A
                    shape_out = jnp.asarray(adapter.K)
                else:
                    adapter.step(h, lr, self._shp_count)     # _Sanger: A = I + sum gamma_j v_j v_j^T
                    shape_out = (jnp.asarray(adapter.W), jnp.asarray(adapter.gamma()))
            else:
                shape_out = shape_params                     # A = I while D(x) settles

            new_ham[k.id] = {"diag": diag, "shape": shape_out}

        return state._replace(ham_params=new_ham)
