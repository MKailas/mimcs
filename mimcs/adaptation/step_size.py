"""Step-size adaptation by Robbins--Monro stochastic approximation.

A mixin (``docs/design/02_sampler_classes.md``) that nudges a scalar ``step_size``
toward a value achieving a target acceptance probability. After each warmup step it
applies, on the log scale,

    log step_size  <-  log step_size + gain_n * (accept_prob - target_accept),

with a decreasing gain ``gain_n = rate * (n + n0)^(-kappa)``, ``kappa in (0.5, 1]``. The
decreasing gain gives a diminishing adaptation, so the procedure converges while the
chain is run only during warmup here (adaptation is frozen during sampling). The offset
``n0`` damps the very first (noisiest) updates.

Two defaults are worth noting:

* ``rate`` defaults to ``1 / sqrt(alpha0 (1 - alpha0))`` with ``alpha0 = target_accept``.
  This normalizes the simplest model of the "gradient": if ``accept_prob`` were Bernoulli
  --- one with probability ``alpha0`` and zero otherwise --- its standard deviation is
  ``sqrt(alpha0 (1 - alpha0))``, so the default rate makes the expected log-step update of
  unit scale regardless of the target.
* ``accelerated`` (Kesten's accelerated stochastic approximation, 1958): the step counter
  ``n`` advances *only when the sign of the error ``accept_prob - target_accept`` flips*
  between iterations. While the error keeps one sign --- a monotone approach to the target
  --- the gain stays large (fast movement); once it starts oscillating around the target the
  counter advances and the gain decays. Only the previous sign need be stored.

Adaptation bookkeeping (the step counter, the previous error sign) lives on the Python
object, never in the JAX state. Only the resulting ``step_size`` is written back.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np

from .._logging import get_logger
from ..samplers.base import Phase

log = get_logger(__name__)


class RobbinsMonroStepSize:
    """Mixin: adapt ``state.step_size`` to hit a target acceptance probability."""

    def _init_hooks(self, **kwargs):
        self._target_accept = float(kwargs.get("target_accept", 0.234))
        # Default rate normalizes a Bernoulli(alpha0) accept-prob "gradient".
        a0 = self._target_accept
        default_rate = 1.0 / math.sqrt(a0 * (1.0 - a0))
        self._ss_rate = float(kwargs.get("step_size_adapt_rate", default_rate))
        self._ss_kappa = float(kwargs.get("step_size_adapt_kappa", 0.6))
        self._ss_n0 = float(kwargs.get("step_size_adapt_n0", 5.0))
        self._ss_accelerated = bool(kwargs.get("accelerated", False))
        self._ss_count = 0
        self._ss_prev_sign = 0          # last nonzero sign of the error (Kesten trick)
        super()._init_hooks(**kwargs)
        log.debug("step-size adaptation: target acceptance %.3f, gain %.3g (n + %.1f)^-%.2f%s",
                  self._target_accept, self._ss_rate, self._ss_n0, self._ss_kappa,
                  ", Kesten-accelerated" if self._ss_accelerated else "")

    def _warmup_end_hooks(self, completed: int, stopped: bool) -> None:
        super()._warmup_end_hooks(completed, stopped)
        # Usually a scalar, but a parallel-tempering chain that accepts independently at each
        # temperature adapts one step size per rung (doc 13); report the worst of them, which is
        # the one that would stall the chain.
        steps = np.atleast_1d(np.asarray(self.state.step_size, float))
        step = float(steps.min())
        gain = self._ss_rate * (self._ss_count + self._ss_n0) ** (-self._ss_kappa)
        if not math.isfinite(step) or step <= 0.0:
            log.warning("step-size adaptation ended at a non-positive or non-finite step size "
                        "(%g) after %d update(s); the chain cannot move from here", step,
                        self._ss_count)
        else:
            log.debug("step-size adaptation ended at %.4g after %d update(s) (final gain %.3g)",
                      step, self._ss_count, gain)

    def _step_size_signal(self, state):
        """The acceptance signal step-size adaptation is driven toward ``target_accept``. The base
        mixin uses the real trajectory acceptance; a line-search variant overrides this to use a
        coarse-level proxy (see :class:`LineSearchStepSizeAdaptation`)."""
        return state.diagnostics["accept_prob"]

    def _postprocess_hooks(self, state):
        state = super()._postprocess_hooks(state)
        if self._phase is not Phase.WARMUP:
            return state

        error = self._step_size_signal(state) - self._target_accept
        if self._ss_accelerated:
            # Advance the counter only when the error changes sign (Kesten).
            sign = int(math.copysign(1.0, float(error))) if float(error) != 0.0 else 0
            if self._ss_prev_sign != 0 and sign != 0 and sign != self._ss_prev_sign:
                self._ss_count += 1
            if sign != 0:
                self._ss_prev_sign = sign
        else:
            self._ss_count += 1

        gain = self._ss_rate * (self._ss_count + self._ss_n0) ** (-self._ss_kappa)
        new_log = jnp.log(state.step_size) + gain * error
        return state._replace(step_size=jnp.exp(new_log))


class LineSearchStepSizeAdaptation(RobbinsMonroStepSize):
    """Step-size adaptation for the WALNUTS line-search integrators.

    Ordinary acceptance-driven adaptation fails for :class:`~mimcs.hmc.LineSearchIntegrator`: the
    integrator refines until the energy error is within budget, so the real acceptance is ~1
    regardless of the macro step size, and ``step_size`` runs away upward (forcing the finest
    scales every step). Instead this mixin drives ``step_size`` toward ``target_accept`` using the
    integrator's *proxy* acceptance (``state.proxy_accept_prob``) --- the acceptance the coarsest
    level would give if adaptive integration were not needed (``docs/design/06``). The real
    acceptance still governs the actual Metropolis / multinomial accept; only adaptation changes.
    """

    def _step_size_signal(self, state):
        return state.diagnostics["proxy_accept_prob"]
