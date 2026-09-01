"""Learned-marginal proposals for discrete parameters.

A mixin (``docs/design/02_sampler_classes.md``) that learns each discrete coordinate's **marginal
pmf** during warmup and hands it to
:class:`~mimcs.samplers.DiscreteMetropolisWithinGibbs`, which then proposes proportional to it
instead of uniformly over the values the coordinate is not currently at.

The motivation is waste. In a ``k``-component mixture an ambiguous observation has posterior mass
on perhaps two labels, so a uniform proposal spends ``(k-2)/(k-1)`` of its attempts on labels of
essentially zero density --- each costing a full log-density evaluation and each certain to be
rejected. Proposing from the learned marginal spends them where the density is. The waste, and so
the gain, grows with ``k``; at ``k = 2`` there is nothing to gain and this mixin provably changes
nothing (see below).

**The proposal becomes asymmetric, and the sweep carries the Hastings term for it.** With
``q(a -> b) = p_b / (1 - p_a)``, the correction is ``g(cur) - g(prop)`` for
``g(v) = log p_v + log1p(-p_v)``. Two consequences fall out of that algebra rather than being
arranged: it is identically **zero for a binary coordinate** (where ``p_b = 1 - p_a``, so
``g(a) = g(b)``), and identically zero for a **uniform** table. So binary parameters and
un-adapted runs are untouched, exactly.

**Adaptation runs during warmup only**, and the estimate uses the library's shared Robbins--Monro
gain (:mod:`mimcs.adaptation._stochastic`), which suits a pmf unusually well: the update

    p_hat <- p_hat + gain * (onehot(z) - p_hat)

is a convex combination of two points on the simplex, so it **stays on the simplex** with no
renormalization and no clipping, and it starts at uniform --- which is the unadapted proposal
exactly.

The running estimate lives on the Python object; only the regularized table is written into the
JAX state, keyed per parameter (``state.discrete_proposal_params``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .._logging import get_logger
from ..samplers.base import Phase
from ._stochastic import rm_gain, DEFAULT_KAPPA, DEFAULT_N0

log = get_logger(__name__)

#: Above this support width the learned table stops being a good idea --- the counts spread thin
#: over many values --- so the mixin says so once and carries on. Per parameter, because one wide
#: parameter says nothing about a narrow one in the same model.
WIDE_SUPPORT = 64


class DiscreteMarginalAdaptation:
    """Mixin: adapt the discrete proposal to each coordinate's learned marginal pmf.

    Compose it **left of** :class:`~mimcs.samplers.DiscreteMetropolisWithinGibbs`, which is the
    sweep that reads what this writes::

        cls = make_sampler_class(RobbinsMonroStepSize, DiscreteMarginalAdaptation,
                                 DiscreteMetropolisWithinGibbs, NUTS)

    Inert on a model with no discrete parameters, and it adds **no RNG draw components** --- the
    sweep already draws the uniform an inverse-CDF search consumes --- so composing it is
    stream-neutral.

    Args:
        discrete_lambda: how much uniform to mix into the learned marginal,
            ``p = (1 - lambda) * p_hat + lambda / n_i`` (default 0.05). This is not cosmetic: a
            zero entry would make a value **unproposable**, which does not break detailed balance
            but does break *irreducibility* --- the chain would then target the posterior
            restricted to whatever it happened to visit during warmup, and no diagnostic in this
            library would flag it. Mixing bounds ``p`` away from both 0 and 1 (the upper bound
            matters too: the Hastings term contains ``log1p(-p)``). ``lambda = 1`` recovers the
            uniform proposal exactly, so the knob spans the whole design; ``lambda = 0`` is
            refused.
        discrete_min_samples: iterations to accumulate before writing a table (default 10).
        discrete_adapt_n0, discrete_adapt_kappa: the shared Robbins--Monro gain schedule.
    """

    def _init_hooks(self, **kwargs):
        self._dm_lambda = float(kwargs.get("discrete_lambda", 0.05))
        if not 0.0 < self._dm_lambda <= 1.0:
            raise ValueError(
                f"discrete_lambda must be in (0, 1], got {self._dm_lambda!r}. It may not be 0: a "
                f"value the chain never visited during warmup would get proposal probability 0 "
                f"and become unreachable, which silently restricts the target rather than "
                f"raising. Use a small positive value (default 0.05); 1.0 is the uniform "
                f"proposal.")
        self._dm_min_samples = int(kwargs.get("discrete_min_samples", 10))
        self._dm_n0 = float(kwargs.get("discrete_adapt_n0", DEFAULT_N0))
        self._dm_kappa = float(kwargs.get("discrete_adapt_kappa", DEFAULT_KAPPA))
        self._dm_count = 0
        self._dm_hat: dict | None = None      # lazily allocated once the supports are known
        self._dm_update = None                # cached jit; see _dm_make_update
        super()._init_hooks(**kwargs)

    # --- the lane axis (1 ordinarily, one per rung under tempering) ---

    @property
    def _dm_lanes(self) -> int:
        return int(getattr(self.model, "n_temperatures", 1))

    @property
    def _dm_lane_dim(self) -> int:
        return self.model.discrete_dim // self._dm_lanes

    # --- the update, compiled once ---

    def _dm_make_update(self):
        """One jitted step for the whole model: fold in the draw, then regularize.

        Compiled once and reused, rather than dispatched primitive by primitive per parameter per
        iteration. Eager JAX dispatch is the tax this codebase has paid before (the tempered
        ladder reseed, the classifier's logistic fit), and it is entirely avoidable here: the dict
        structure is fixed for the run, so one ``jax.jit`` covers every parameter.
        """
        blocks = [(p.name, *self.model.discrete_block(p.name), int(p.lower_value),
                   int(p.upper_value - p.lower_value + 1))
                  for p in self.model.discrete_parameters]
        lam = self._dm_lambda
        L, n = self._dm_lanes, self._dm_lane_dim

        def update(hat, discrete, gain):
            # `(L, n)`: one lane untempered, one per rung under tempering. `discrete_block` gives
            # the slice within a *lane*, so the column slice is the same at every temperature.
            z = discrete.reshape(L, n)
            new_hat, tables = {}, {}
            for name, start, stop, lo, ni in blocks:
                onehot = jax.nn.one_hot(z[:, start:stop] - lo, ni, dtype=float)   # (L, size, ni)
                h = hat[name] + gain * (onehot - hat[name])
                new_hat[name] = h
                tables[name] = (1.0 - lam) * h + lam / ni
            return new_hat, tables

        return jax.jit(update)

    # --- adaptation ---

    def _postprocess_hooks(self, state):
        state = super()._postprocess_hooks(state)
        model = self.model
        if not model.discrete_dim or self._phase is not Phase.WARMUP:
            return state

        if self._dm_hat is None:
            self._dm_hat = {
                p.name: jnp.full((self._dm_lanes, p.size,
                                  int(p.upper_value - p.lower_value + 1)),
                                 1.0 / int(p.upper_value - p.lower_value + 1), float)
                for p in model.discrete_parameters}
            self._dm_update = self._dm_make_update()
            wide = [(p.name, int(p.upper_value - p.lower_value + 1))
                    for p in model.discrete_parameters
                    if int(p.upper_value - p.lower_value + 1) > WIDE_SUPPORT]
            for name, ni in wide:
                size = next(p.size for p in model.discrete_parameters if p.name == name)
                log.warning(
                    "discrete marginal adaptation: '%s' has %d values, so its table is %d x %d "
                    "(%.3g MB) and each value will collect only ~1/%d of the draws. The learned "
                    "marginal is unlikely to be worth much this wide --- consider leaving this "
                    "mixin off, or raising the warmup length.",
                    name, ni, size, ni, size * ni * 4 / 1e6, ni)
            log.debug("discrete marginal adaptation over %d parameter(s) %s; tables written from "
                      "iteration %d on, lambda %.3g", len(model.discrete_parameters),
                      [p.name for p in model.discrete_parameters], self._dm_min_samples + 1,
                      self._dm_lambda)

        self._dm_count += 1
        gain = rm_gain(self._dm_count, self._dm_n0, self._dm_kappa)
        self._dm_hat, tables = self._dm_update(self._dm_hat, state.discrete, gain)

        if self._dm_count > self._dm_min_samples:
            # Merged, not replaced: dict-valued state fields are shared by convention, so another
            # mixin adapting a different parameter's proposal must not be clobbered.
            state = state._replace(
                discrete_proposal_params={**state.discrete_proposal_params, **tables})
        return state

    # --- reporting ---

    def _warmup_end_hooks(self, completed: int, stopped: bool) -> None:
        """Report how far the learned marginals actually moved off uniform.

        The number that matters is the **normalized entropy**: 1 means the marginals came out
        uniform, i.e. the adaptation found nothing and this mixin bought nothing. That is the
        expected and correct outcome on a well-separated problem, and it is what distinguishes
        "no gain because there was none to get" from "no gain because the estimator is broken" ---
        a distinction a timing comparison alone cannot make.
        """
        super()._warmup_end_hooks(completed, stopped)
        if self._dm_hat is None:
            return
        for name, h in self._dm_hat.items():
            p = np.asarray(h, dtype=float)                     # (L, size, ni)
            ni = p.shape[-1]
            if ni < 2:
                continue
            ent = -np.sum(np.where(p > 0, p * np.log(np.maximum(p, 1e-300)), 0.0), axis=-1)
            per_lane = np.mean(ent, axis=-1) / np.log(ni)      # (L,)
            norm = float(per_lane[0])                          # the cold chain, or the only one
            if len(per_lane) > 1:
                log.info("discrete marginal '%s' after %d update(s): mean normalized entropy "
                         "per rung %s (1.0 = uniform; a hotter rung should be flatter)",
                         name, self._dm_count, np.array2string(per_lane, precision=3))
            else:
                log.info("discrete marginal '%s' after %d update(s): mean normalized entropy "
                         "%.3f (1.0 = uniform, i.e. nothing learned)",
                         name, self._dm_count, norm)
            if norm > 0.999:
                log.warning(
                    "discrete marginal '%s' is still uniform to within rounding: the learned "
                    "proposal is doing nothing, so this mixin is pure overhead on this model.",
                    name)
