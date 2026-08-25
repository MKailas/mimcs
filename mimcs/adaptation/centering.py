"""Centering (standardizing) reparametrization adaptation.

A mixin (``docs/design/02_sampler_classes.md``) that adapts the ``(mu, sigma)``
hyperparameters of every parameter created with ``centered=True`` (an
:class:`~mimcs.model.EuclideanParameter` or :class:`~mimcs.model.BoundedParameter`).
A centered parameter's coordinate is the *standardized link value* ``q = (z - mu) / sigma``,
where ``z = link(x)`` is the unconstrained value the chart produces before standardizing
(identity for Euclidean, ``log(x - L)`` / logit for bounded). Fitting ``mu`` to the location of
``z`` and ``sigma`` to its scale makes the coordinate target centred and unit-scaled. The link
value ``z`` is chart-invariant (a function of the sample alone), computed as the coordinate
under the identity standardization ``(mu, sigma) = (0, 1)``.

Two estimators of ``(location, scale)`` share the machinery here (:class:`_CenteringBase`),
both on the shared Robbins--Monro schedule ``(n + n0)^{-kappa}`` (``kappa = 0.75``, ``n0 = 5``;
:mod:`mimcs.adaptation._stochastic`):

* :class:`CenteringAdaptation` --- the diagonal running **mean / standard deviation** (the same
  estimator as the mass-matrix adaptation): ``mean <- mean + gain (z - mean)``,
  ``var <- (1 - gain) var + gain (z - mean)^2``, then ``mu = mean``, ``sigma = sqrt(var)``.
* :class:`RobustCenteringAdaptation` --- a **quantile** estimator robust to heavy tails: the
  **median** for location and the **median absolute deviation** (MAD) for scale. Empirical
  variance is very sensitive to how far into the tails the chain happens to reach; the median
  and MAD are not. See that class for the parametrization.

The two objectives are distinct: centering fits the location/scale of the *sample*, while
the score-covariance mass adaptation (:class:`ScoreMassAdaptation`) fits the *gradient*
covariance in the resulting coordinates --- so they are meant to be used together. (With
the empirical-covariance mass adaptation the two would target the same scale and centering
adds little.)

Changing a chart relabels the coordinate of a fixed physical point, so on each update the
mixin recomputes the coordinate from the (unchanged) sample and refreshes the cached
potential values/gradients at the new coordinate. Adaptation runs during warmup only.
"""

from __future__ import annotations

import jax.numpy as jnp

from .._logging import get_logger
from ..samplers.base import Phase
from ._stochastic import rm_gain, DEFAULT_KAPPA, DEFAULT_N0

log = get_logger(__name__)

# median(|z - median|) = MAD_INV_STD^{-1} * sigma for a Gaussian (Phi^{-1}(0.75) = 0.6744898),
# so sigma = MAD * 1.4826022 recovers the standard deviation from the MAD.
MAD_TO_STD = 1.0 / 0.6744897501960817


class _CenteringBase:
    """Shared machinery: locate the centered parameters, run the estimator, rechart.

    Subclasses supply the location/scale estimator via ``_center_init`` (allocate state from the
    first link value ``z``), ``_center_update`` (fold ``z`` in with the current gain) and
    ``_center_location_scale`` (return the ``(mu, sigma)`` vectors to standardize with).
    """

    #: which location/scale estimator the subclass uses, for the log message.
    _estimator_name = "location/scale"

    def _init_hooks(self, **kwargs):
        self._center_min_samples = int(kwargs.get("center_min_samples", 50))
        self._center_n0 = float(kwargs.get("center_adapt_n0", DEFAULT_N0))
        self._center_kappa = float(kwargs.get("center_adapt_kappa", DEFAULT_KAPPA))
        self._center_floor = float(kwargs.get("center_floor", 1e-8))
        # (param index, coordinate slice) for each centered parameter, and the identity
        # standardization (mu, sigma) = (0, 1) used to read off the link value z.
        offs = self.model._coord_offsets
        self._center_params = [
            (i, int(offs[i]), int(offs[i + 1]))
            for i, p in enumerate(self.model.parameters)
            if getattr(p, "centered", False)]
        self._identity_hyperparams = self.model.init_chart_hyperparams()
        self._center_count = 0
        self._center_ready = False
        super()._init_hooks(**kwargs)
        if self._center_params:
            log.debug("centering adaptation (%s): standardizing %d parameter(s) %s from "
                      "iteration %d on", self._estimator_name, len(self._center_params),
                      [self.model.parameters[i].name for i, _, _ in self._center_params],
                      self._center_min_samples + 1)
        else:
            log.debug("centering adaptation (%s): inert --- the model declares no "
                      "centered=True parameter", self._estimator_name)

    def _postprocess_hooks(self, state):
        state = super()._postprocess_hooks(state)
        if self._phase is not Phase.WARMUP or not self._center_params:
            return state

        # The link value z = link(x): the coordinate under identity standardization.
        z = self.model.sample_to_coordinate(
            state.sample, self._identity_hyperparams, state.chart_indices)
        if not self._center_ready:
            self._center_init(z)
            self._center_ready = True

        self._center_count += 1
        gain = rm_gain(self._center_count, self._center_n0, self._center_kappa)
        self._center_update(z, gain)

        if self._center_count <= self._center_min_samples:
            return state

        mu, sigma = self._center_location_scale()
        new_hyperparams = list(state.chart_hyperparams)
        # Rebuild from the live coordinate and touch only the centered slices. Starting from
        # ``z`` instead would be equivalent for a fixed chart, but ``z`` is computed under the
        # *initial* hyperparameters, so for any other adaptive chart (a unit vector's pole, say)
        # it is a stale coordinate --- writing it back would silently move the chain.
        new_coordinate = state.coordinate
        for i, lo, hi in self._center_params:
            mu_i, sigma_i = mu[lo:hi], sigma[lo:hi]
            new_hyperparams[i] = (mu_i, sigma_i)
            new_coordinate = new_coordinate.at[lo:hi].set((z[lo:hi] - mu_i) / sigma_i)
        return self._recharted_state(state, tuple(new_hyperparams), new_coordinate)

    def _recharted_state(self, state, new_hyperparams, new_coordinate):
        """Refresh the cached densities at the new chart (keeps the sample fixed)."""
        return self.state_at_coordinate(
            state, new_coordinate, sample=state.sample, hyperparams=new_hyperparams)


class CenteringAdaptation(_CenteringBase):
    """Mixin: adapt centered charts to the sample **mean / standard deviation**."""

    _estimator_name = "mean/sd"

    def _center_init(self, z):
        self._center_mean = jnp.zeros_like(z)
        self._center_var = jnp.ones_like(z)

    def _center_update(self, z, gain):
        delta = z - self._center_mean
        self._center_mean = self._center_mean + gain * delta
        self._center_var = self._center_var + gain * (delta ** 2 - self._center_var)

    def _center_location_scale(self):
        return self._center_mean, jnp.sqrt(self._center_var + self._center_floor)


class RobustCenteringAdaptation(_CenteringBase):
    """Mixin: adapt centered charts to the **median** and **MAD** (heavy-tail robust).

    Empirical mean/variance are sensitive to how far into the tails the chain reaches --- a
    single deep excursion inflates the variance for a long time --- so on heavy-tailed targets
    ``CenteringAdaptation`` standardizes to a scale that drifts with the sampled extremes. This
    estimator instead tracks two order statistics of each 1-D marginal by Robbins--Monro
    stochastic-quantile updates, both robust to the tails:

    * **location** --- the median ``m``. A quantile step ``m <- m + gain * s * (1[z > m] - 1/2)``
      converges to ``median(z)`` (its expected drift is ``P(z > m) - 1/2``, zero at the median).
      The step is scaled by the current MAD ``s`` so convergence is scale-invariant --- otherwise
      a large-magnitude coordinate's median crawls, since (unlike a log-scale) the location step
      is in absolute units.
    * **scale** --- the MAD ``s = median(|z - m|)``, adapted on the **log scale** (as the
      gradient-clip threshold is): ``log s <- log s + gain * (1[|z - m| > s] - 1/2)``, whose
      fixed point is ``P(|z - m| > s) = 1/2``. The log scale keeps ``s`` strictly positive, so
      --- unlike adapting the 25th/75th percentiles and taking their difference --- the scale can
      never overshoot to zero-or-negative (e.g. when the sampler barely moves); a floor guards
      the residual collapse toward zero.

    ``sigma = MAD * 1.4826`` rescales the MAD to a standard deviation (matching the Gaussian, so
    a Gaussian target is standardized identically to :class:`CenteringAdaptation` in the mean);
    ``mu = m``. This trades some efficiency on light-tailed targets (the MAD is a noisier scale
    estimate there) for insensitivity to heavy tails.
    """

    _estimator_name = "median/MAD"

    def _center_init(self, z):
        self._rc_median = jnp.zeros_like(z)
        self._rc_log_mad = jnp.zeros_like(z)          # log MAD; s = 1 at start

    def _center_update(self, z, gain):
        m = self._rc_median
        s = jnp.exp(self._rc_log_mad)
        above = (z > m).astype(z.dtype)               # 1[z > m]
        self._rc_median = m + gain * s * (above - 0.5)                       # median step (scaled)
        exceeds = (jnp.abs(z - m) > s).astype(z.dtype)                       # 1[|z - m| > s]
        self._rc_log_mad = self._rc_log_mad + gain * (exceeds - 0.5)         # log-MAD step

    def _center_location_scale(self):
        sigma = MAD_TO_STD * jnp.exp(self._rc_log_mad)
        return self._rc_median, jnp.maximum(sigma, self._center_floor)
