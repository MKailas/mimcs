"""Diagonal proposal-covariance adaptation.

A mixin (``docs/design/02_sampler_classes.md``) that learns the running mean and
diagonal variance of the chain in coordinate space and sets the per-coordinate
``proposal_scale`` to the running standard deviation. Combined with
:class:`RobbinsMonroStepSize`, which supplies the global scale, this is a diagonal
adaptive-Metropolis scheme: ``proposal_scale`` captures the relative scales of the
coordinates while ``step_size`` finds the overall magnitude.

The variance is a *stochastic-approximation* running estimate with the shared
Robbins--Monro gain ``(n + n0)^{-kappa}`` (``kappa = 0.75``, ``n0 = 5``; see
:mod:`mimcs.adaptation._stochastic`): ``var <- var + gain * (delta^2 - var)``, weighting
recent draws more than an equal-weight average and staying consistent with the mass-matrix
and metric adaptations.

The running moments live on the Python object. Only the resulting ``proposal_scale`` is
written into the JAX state. Adaptation is applied during warmup only, and the scale is
updated lazily once enough samples have accumulated to give a non-degenerate estimate.
"""

from __future__ import annotations

import jax.numpy as jnp

from .._logging import get_logger
from ..samplers.base import Phase
from ._stochastic import rm_gain, DEFAULT_KAPPA, DEFAULT_N0

log = get_logger(__name__)


class DiagonalCovarianceAdaptation:
    """Mixin: adapt ``state.proposal_scale`` from the running diagonal covariance."""

    def _init_hooks(self, **kwargs):
        self._cov_floor = float(kwargs.get("cov_floor", 1e-8))
        self._cov_min_samples = int(kwargs.get("cov_min_samples", 10))
        self._cov_n0 = float(kwargs.get("cov_adapt_n0", DEFAULT_N0))
        self._cov_kappa = float(kwargs.get("cov_adapt_kappa", DEFAULT_KAPPA))
        self._cov_count = 0
        self._cov_mean = None   # initialized lazily once the dimension is known
        self._cov_var = None
        super()._init_hooks(**kwargs)

    def _postprocess_hooks(self, state):
        state = super()._postprocess_hooks(state)
        if self._phase is not Phase.WARMUP:
            return state

        x = state.coordinate
        if self._cov_mean is None:
            log.debug("diagonal covariance adaptation started over %d coordinate(s); the "
                      "proposal scale is written from iteration %d on", x.shape[0],
                      self._cov_min_samples + 1)
            self._cov_mean = jnp.zeros_like(x)
            self._cov_var = jnp.zeros_like(x)

        # Stochastic-approximation running mean / variance (decreasing gain).
        self._cov_count += 1
        gain = rm_gain(self._cov_count, self._cov_n0, self._cov_kappa)
        delta = x - self._cov_mean
        self._cov_mean = self._cov_mean + gain * delta
        self._cov_var = self._cov_var + gain * (delta * delta - self._cov_var)

        if self._cov_count > self._cov_min_samples:
            scale = jnp.sqrt(self._cov_var + self._cov_floor)
            state = state._replace(proposal_scale=scale)
        return state
