"""Adapt the softness of a Hessian-metric block (:class:`~mimcs.hmc.riemannian.HessianMetric`).

The metric clamps the block Hessian's eigenvalues as ``f(lambda) = phi(b lambda) / b``. ``1/b`` is a
*curvature scale*: eigenvalues well above it pass through essentially unchanged, those around it or
below (negative ones included) are reshaped into a positive mass. So ``1/b`` should sit **below the
real positive curvatures** --- not distorting them --- but not so far below that a negative
curvature of comparable size gets no mass at all.

This mixin sets it from the spectrum the chain actually visits. Each warmup iteration it reads the
block Hessian's eigenvalues at the current draw and tracks, by stochastic approximation on the log
scale, the ``softness_quantile`` (default 0.1) quantile ``q`` of the **positive** ones:

    log q  +=  gain_n * (mean_j 1[lambda_j > q] - (1 - quantile))    over the positive lambda_j,

the same running log-quantile the learned-metric clip threshold uses (``_PerUnitClip``), and then
sets ``1/b = q / softness_ratio`` (default 3). With softplus, ``phi(3) / 3 = 1.016``: 90% of the
positive curvatures seen are distorted by under 2%, while a negative curvature of the size of that
10% quantile keeps about 5% of it as mass.

The first update initializes ``log q`` at that draw's own quantile rather than walking there from
the default ``1/b``; a draw with no positive eigenvalue leaves everything unchanged. The gain is
the shared Robbins--Monro ``(n + n0)^{-kappa}`` (``softness_adapt_kappa`` / ``softness_adapt_n0``
in ``algo_kwargs``). ``1/b`` lives in ``ham_params[id]["log_softness"]`` and is frozen, like every
adapted quantity, when warmup ends --- the last iterate *is* the running quantile estimate, so there
is no separate average to freeze.
"""

from __future__ import annotations

import math

import jax
import numpy as np
import jax.numpy as jnp

from .._logging import get_logger
from ..samplers.base import Phase
from ._stochastic import rm_gain, DEFAULT_KAPPA, DEFAULT_N0

log = get_logger(__name__)


class HessianSoftnessAdaptation:
    """Mixin: adapt each Hessian-metric block's softness ``1/b`` during warmup."""

    def _init_hooks(self, **kwargs):
        self._softness_kappa = float(kwargs.get("softness_adapt_kappa", DEFAULT_KAPPA))
        self._softness_n0 = float(kwargs.get("softness_adapt_n0", DEFAULT_N0))
        self._softness_count = 0
        self._softness_log_q: dict = {}         # running log-quantile per block id
        self._softness_spectrum_fns: dict = {}  # jitted spectrum per block id
        super()._init_hooks(**kwargs)

    def _softness_blocks(self):
        return [k for k in self.kinetics if getattr(k, "adapts_softness", False)]

    def _spectrum_fn(self, block):
        fn = self._softness_spectrum_fns.get(block.id)
        if fn is None:
            fn = jax.jit(lambda q, ctx: block.spectrum(q, ctx))
            self._softness_spectrum_fns[block.id] = fn
        return fn

    def softness(self, block_id: str) -> float:
        """The block's current ``1/b``."""
        return float(np.exp(np.asarray(self.state.ham_params[block_id]["log_softness"])))

    def _postprocess_hooks(self, state):
        state = super()._postprocess_hooks(state)
        if self._phase is not Phase.WARMUP:
            return state
        blocks = self._softness_blocks()
        if not blocks:
            return state
        self._softness_count += 1
        gain = rm_gain(self._softness_count, self._softness_n0, self._softness_kappa)
        # The context the kernel integrated with --- charts, labels and all --- but no kinetic cache
        # (the spectrum reads the potentials only).
        ctx = self.context(state, kinetic_cache=False)
        new_ham = dict(state.ham_params)
        for k in blocks:
            metric = k.metric
            lam = np.asarray(self._spectrum_fn(k)(state.coordinate, ctx), dtype=float)
            pos = lam[np.isfinite(lam) & (lam > 0)]
            if pos.size == 0:
                continue
            target = 1.0 - metric.softness_quantile
            log_q = self._softness_log_q.get(k.id)
            if log_q is None:
                log_q = math.log(float(np.quantile(pos, metric.softness_quantile)))
                log.debug("softness adaptation started on block %r: first spectrum %s, 1/b "
                          "starts at %.4g", k.id, np.array2string(lam, precision=3),
                          math.exp(log_q) / metric.softness_ratio)
            else:
                log_q += gain * (float(np.mean(pos > math.exp(log_q))) - target)
            self._softness_log_q[k.id] = log_q
            params = dict(new_ham[k.id])
            params["log_softness"] = jnp.asarray(log_q - math.log(metric.softness_ratio), float)
            new_ham[k.id] = params
        return state._replace(ham_params=new_ham)

    def _finalize_hooks(self, state):
        state = super()._finalize_hooks(state)
        for k in self._softness_blocks():
            if k.id in self._softness_log_q:
                log.info("Hessian metric %r: softness 1/b frozen at %.4g after %d warmup "
                         "update(s)", k.id,
                         float(np.exp(np.asarray(state.ham_params[k.id]["log_softness"]))),
                         self._softness_count)
            else:
                log.warning("Hessian metric %r: no warmup draw had a positive curvature in this "
                            "block, so its softness 1/b was never adapted and stays at its "
                            "initial value", k.id)
        return state
