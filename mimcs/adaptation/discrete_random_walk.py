"""Scale adaptation for the discrete random walk, after Vihola's robust adaptive Metropolis.

A mixin (``docs/design/02_sampler_classes.md``) that adapts each random-walk coordinate's log scale
``rho`` --- the mean step is ``1/p = 1 + exp(rho)`` --- toward a target acceptance probability:

    rho  <-  clip(rho + gain_n * (alpha_bar - target), rho_min, rho_max)

per coordinate and per lane, where ``alpha_bar`` is that coordinate's mean acceptance probability
over the iteration's **genuine** proposals. The gain is the step-size mixin's schedule,
``rate * (n + n0)^-kappa`` with ``rate = 1/sqrt(target (1 - target))``, because the signal is the
same kind of thing: an acceptance probability driven toward a scalar target.

**Why 1/3.** On an exactly enumerated kernel, the acceptance rate at the IACT-optimal scale is 0.325
for a (discretised) Laplace target. It is higher, about 0.44, for Gaussian-shaped ones --- but
aiming at 1/3 there costs only ~8% in integrated autocorrelation time (5.31 -> 5.76), so one default
serves both.

**Why only genuine proposals.** At a bound, a step pointing outward clamps to the current value.
Its acceptance probability is 1 and says nothing about the scale, and counting it is not a small
bias: next to a bound holding most of the mass the signal then stays above the target *at every
scale*, and the scale runs away to its cap. Measured on the exact kernel for Poisson(0.1), that
lands at IACT 3.2e5 against a best of 2.71. Excluding those no-ops, the same target has no scale
reaching 1/3 at all (the signal plateaus at 0.17), so ``rho`` settles on its floor --- the +-1
walk, at IACT 3.40. That is the intended degradation, and the warmup-end report says so. On
Poisson(0.5) and Poisson(1) the two choices settle at IACT ~3.1 and ~3.2 against ~12 and ~7.

Adaptation runs during warmup only. The statistics it reads are written by the sweep into the
parameter's own proposal entry, ``state.discrete_proposal_params[name]`` (see
:class:`~mimcs.samplers.discrete_updates.RandomWalkUpdate`); this mixin moves ``log_scale`` and
nothing else.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from .._logging import get_logger
from ..samplers.base import Phase

log = get_logger(__name__)

#: default target acceptance probability of a genuine proposal (see the module docstring)
DEFAULT_TARGET_ACCEPT = 1.0 / 3.0

#: weight of each iteration in the pooled acceptance the warmup-end report quotes --- a report
#: figure only, never fed back into the adaptation
_REPORT_EMA = 0.02


class DiscreteRandomWalkAdaptation:
    """Mixin: adapt each random-walk coordinate's proposal scale toward a target acceptance.

    Compose it **left of** :class:`~mimcs.samplers.DiscreteMetropolisWithinGibbs`::

        cls = make_sampler_class(DiscreteRandomWalkAdaptation, DiscreteMetropolisWithinGibbs, NUTS)

    It owns only the parameters whose update method is ``"random_walk"``, read from the same
    ``discrete_update`` kwarg the sweep reads, and is inert on every other model. It adds no RNG draw
    components, so composing it is stream-neutral.

    Args:
        discrete_rw_target_accept: target acceptance probability of a genuine proposal (1/3).
        discrete_rw_adapt_rate: gain multiplier (default ``1/sqrt(target (1 - target))``).
        discrete_rw_adapt_kappa: gain decay exponent (0.6, the step-size mixin's).
        discrete_rw_adapt_n0: gain offset (5.0).
    """

    def _init_hooks(self, **kwargs):
        t = float(kwargs.get("discrete_rw_target_accept", DEFAULT_TARGET_ACCEPT))
        if not 0.0 < t < 1.0:
            raise ValueError(f"discrete_rw_target_accept must be in (0, 1), got {t!r}")
        self._rw_target = t
        self._rw_rate = float(kwargs.get("discrete_rw_adapt_rate", 1.0 / math.sqrt(t * (1.0 - t))))
        self._rw_kappa = float(kwargs.get("discrete_rw_adapt_kappa", 0.6))
        self._rw_n0 = float(kwargs.get("discrete_rw_adapt_n0", 5.0))
        self._rw_methods = kwargs.get("discrete_update")
        self._rw_count = 0
        self._rw_walks = None           # lazily: the model's random-walk updaters
        self._rw_update = None          # cached jit
        self._rw_pooled = {}            # {name: EMA of pooled acceptance, a JAX scalar}
        super()._init_hooks(**kwargs)

    def _rw_owned(self) -> list:
        """The random-walk updaters, derived from ``(model, methods)`` exactly as the sweep does."""
        if self._rw_walks is None:
            from ..samplers.discrete_updates import build_discrete_updaters
            self._rw_walks = [u for u in build_discrete_updaters(self.model, self._rw_methods)
                              if u.records_accept]
        return self._rw_walks

    def _rw_make_update(self, walks):
        """One jitted step for every random-walk parameter; the dict structure is fixed for the run."""
        bounds = [(u.name, *u.log_scale_bounds) for u in walks]
        target, ema = self._rw_target, _REPORT_EMA

        def update(entries, pooled, gain):
            new, new_pooled = {}, {}
            for name, lo, hi in bounds:
                e = entries[name]
                n = e["n_proposed"]
                alpha_bar = e["accept_sum"] / jnp.maximum(n, 1.0)
                # A coordinate with no genuine proposal this iteration (every step clamped onto
                # itself) carries no information about its scale, so it is left where it is.
                step = jnp.where(n > 0, gain * (alpha_bar - target), 0.0)
                new[name] = {**e, "log_scale": jnp.clip(e["log_scale"] + step, lo, hi)}
                tot = jnp.sum(n)
                pooled_now = jnp.where(tot > 0, jnp.sum(e["accept_sum"]) / jnp.maximum(tot, 1.0),
                                       pooled[name])
                new_pooled[name] = (1.0 - ema) * pooled[name] + ema * pooled_now
            return new, new_pooled

        return jax.jit(update)

    def _postprocess_hooks(self, state):
        state = super()._postprocess_hooks(state)
        if not self.model.discrete_dim or self._phase is not Phase.WARMUP:
            return state
        walks = self._rw_owned()
        if not walks:
            return state
        if self._rw_update is None:
            self._rw_update = self._rw_make_update(walks)
            self._rw_pooled = {u.name: jnp.asarray(self._rw_target, float) for u in walks}
            log.debug("discrete random-walk adaptation over %s: target acceptance %.3f, gain "
                      "%.3g (n + %.1f)^-%.2f", [u.name for u in walks], self._rw_target,
                      self._rw_rate, self._rw_n0, self._rw_kappa)
        self._rw_count += 1
        gain = self._rw_rate * (self._rw_count + self._rw_n0) ** (-self._rw_kappa)
        params = state.discrete_proposal_params
        entries, self._rw_pooled = self._rw_update(
            {u.name: params[u.name] for u in walks}, self._rw_pooled, gain)
        # Merged, not replaced: another method's proposal entry must not be clobbered.
        return state._replace(discrete_proposal_params={**params, **entries})

    def _warmup_end_hooks(self, completed: int, stopped: bool) -> None:
        """Report where each random walk's scale settled, and say plainly when it hit a clip."""
        super()._warmup_end_hooks(completed, stopped)
        if not self._rw_count:
            return
        params = self.state.discrete_proposal_params
        for u in self._rw_owned():
            rho = np.asarray(params[u.name]["log_scale"], dtype=float)      # (L, size)
            lo, hi = u.log_scale_bounds
            step = 1.0 + np.exp(rho)
            per_rung = np.median(step, axis=-1)
            pooled = float(self._rw_pooled[u.name])
            log.info("discrete random walk '%s' after %d update(s): median mean step %s "
                     "(range %.3g..%.3g), acceptance %.2f against target %.2f",
                     u.name, self._rw_count,
                     f"{per_rung[0]:.3g}" if len(per_rung) == 1
                     else np.array2string(per_rung, precision=3),
                     float(step.min()), float(step.max()), pooled, self._rw_target)
            floor = float(np.mean(rho <= lo + 1e-6))
            if floor > 0:
                log.info("discrete random walk '%s': %.0f%% of coordinate(s) sit at the +-1 walk. "
                         "That is the expected outcome next to a bound holding most of the mass, "
                         "where no step size reaches the target acceptance; it is the intended "
                         "fallback, not a failure.", u.name, 100 * floor)
            cap = float(np.mean(rho >= hi - 1e-6))
            if cap > 0 and not u.bounded:
                log.warning("discrete random walk '%s': %.0f%% of coordinate(s) reached the "
                            "largest mean step (%.3g) and still accept above target. A scale that "
                            "wants to be larger suggests an improper or extraordinarily wide "
                            "posterior for this integer.", u.name, 100 * cap, 1.0 + math.exp(hi))
