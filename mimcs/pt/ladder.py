"""Adapting the temperature ladder (Miasojedow, Moulines & Vihola 2013).

A hand-set ladder cannot be right across models, and `hmm_gaussian` showed why in the sharpest
possible way: with `beta_min = 0.02` its adjacent rungs are thousands of nats apart, swap
acceptance is **exactly zero on every pair**, and the run is not sampling at all. The usable
temperature range narrows as the data grows — a `beta_min` that suits a 2-d Gaussian is useless
at T = 500 observations — so the ladder has to be found rather than guessed.

**The parametrization** (MMV, §3) is what makes this work. Adapt in *temperature* rather than
inverse temperature, and in the **log-gaps** between neighbours::

    T_1 = 1,   T_{k+1} = T_k + exp(rho_k),   beta_k = 1 / T_k

Every ``rho`` in ``R^{K-1}`` gives a valid ladder: temperatures are automatically ordered and
strictly increasing, and ``beta_1 = 1`` is fixed by construction. No constraint has to be
enforced, and no update can produce a crossed or duplicated rung — which an unconstrained update
of the betas themselves certainly could.

**The top rung is held fixed by default, and this is not a detail.** With the gaps free, the
literal MMV form, ``rho`` provably runs away on any target whose hot end goes *flat*: once two
adjacent tempered densities are both nearly flat their log-densities barely differ, so widening
the gap stops lowering that pair's swap acceptance. The signal ``alpha - alpha*`` stays positive
however far apart the rungs are pushed, and nothing arrests the growth. Measured on the bimodal
test target: in 200 warmup iterations ``beta_min`` ran 0.02 -> 1.7e-4 and was still moving, at
which point the hot chain's target is flat enough that NUTS never U-turns and every iteration
builds a maximum-depth tree — the run does not diverge, it simply stops making progress.

So the gaps are instead *shares* of a fixed total temperature range (a softmax over ``rho``),
which pins both endpoints and adapts only the interior spacing. That also matches how the two
quantities differ in kind: ``beta_min`` says how deep a barrier the model has, which is a
modelling judgement the user can make, while the spacing is the tuning problem they cannot.
``adapt_beta_min=True`` restores the unbounded form, with the above as the warning label; the
range then needs an external cap.

**When freeing it is safe** turns on whether an untempered component holds the hot end down. Temper
everything and ``beta -> 0`` is a genuinely flat target, which is the runaway above. Temper only
the likelihood and the hot rung *is* the prior --- proper, finite width, and still answering the
acceptance signal --- which is why `hmm_gaussian`'s free ladder converged rather than ran away.
Note also that with *both* endpoints pinned the gaps are shares of a fixed range, so a ladder none
of whose pairs swap cannot move at all: a uniform ``alpha - alpha*`` cancels. See "When to free
``beta_min``" in ``docs/design/13_parallel_tempering.md`` for the guidance and the numbers.

**The update** is Robbins--Monro on each gap, driving that pair's swap acceptance toward
``alpha* = 0.234``::

    rho_k  <-  rho_k + gain_n * (alpha_k - alpha*)

The target is not arbitrary: 0.234 is the optimal swap acceptance from the diffusion limit of
Atchadé, Roberts & Rosenthal (2011), the parallel-tempering analogue of the familiar
random-walk Metropolis result. Widening a gap lowers that pair's acceptance and narrowing it
raises it, so the sign is right: too many accepted swaps pushes the rungs apart, too few pulls
them together.

Two implementation choices worth stating:

* The signal is the swap **acceptance probability**, not the binary accept/reject. It is
  available for nothing (the ratio is computed anyway), has far lower variance, and is what MMV
  use.
* Every pair is updated every iteration, using the ratio it *would* have accepted with, rather
  than only the half attempted by this sweep's even/odd parity. The unattempted pairs' ratios
  cost nothing to evaluate, and using them keeps all the gaps adapting at the same rate.

Adaptation runs during warmup only and the gain decays, so diminishing adaptation holds. The
ladder is carried in the state as a *traced* value (`HamiltonianContext.betas`) precisely so it
can change without retracing the kernel every step.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .._logging import get_logger
from ..adaptation._stochastic import rm_gain, DEFAULT_N0, DEFAULT_KAPPA
from ..samplers.base import Phase

log = get_logger(__name__)

#: Optimal swap acceptance from the PT diffusion limit (Atchadé, Roberts & Rosenthal 2011).
OPTIMAL_SWAP_ACCEPT = 0.234

#: Key under which the ladder rides in ``state.ham_params``. Reserved: no kinetic may use it.
BETAS_KEY = "__betas"


def betas_from_rho(rho: Array, t_max: float | None = None) -> Array:
    """``rho`` (the ``K-1`` log-gaps) -> the inverse temperatures, ``beta_1 = 1`` by construction.

    With ``t_max`` given the gaps are *shares* of the fixed total range (a softmax), so the top
    rung stays at ``beta = 1 / t_max`` and only the interior spacing moves. With ``t_max=None``
    the gaps are free, which is the literal MMV form --- see the runaway warning on
    :class:`LadderAdaptation`.
    """
    if t_max is None:
        temperatures = jnp.concatenate([jnp.ones((1,)), 1.0 + jnp.cumsum(jnp.exp(rho))])
    else:
        shares = jax.nn.softmax(rho)
        temperatures = jnp.concatenate(
            [jnp.ones((1,)), 1.0 + (t_max - 1.0) * jnp.cumsum(shares)])
    return 1.0 / temperatures


def rho_from_betas(betas, t_max: float | None = None) -> Array:
    """The inverse of :func:`betas_from_rho`; the ladder must start at ``beta = 1``."""
    betas = jnp.asarray(betas, float)
    temperatures = 1.0 / betas
    gaps = jnp.diff(temperatures)
    if not bool(jnp.all(gaps > 0)):
        raise ValueError(f"the ladder must be strictly decreasing in beta, got {np.asarray(betas)}")
    if t_max is None:
        return jnp.log(gaps)
    return jnp.log(gaps / (t_max - 1.0))     # softmax recovers the shares: they already sum to 1


class LadderAdaptation:
    """Mixin: adapt the temperature ladder toward a target swap acceptance.

    Composed onto a parallel tempering sampler; inert outside warmup. ``beta = 0`` is not
    reachable by this parametrization (it would need an infinite temperature), so a ladder that
    wants a genuinely flat top rung should keep it fixed instead.
    """

    def _init_hooks(self, **kwargs):
        super()._init_hooks(**kwargs)
        self._ladder_adapt = bool(kwargs.get("adapt_ladder", True)) and self.n_temperatures > 1
        # A flat top rung is a legitimate ladder but not an adaptable one: the parametrization is
        # in *temperatures* ``1/beta``, so ``beta_min = 0`` makes ``t_max`` infinite and the very
        # first ``rho`` update produces ``-inf`` and then ``nan``, silently, on every interior gap.
        # The class docstring already says such a ladder "should keep it fixed instead"; this is
        # that advice enforced, because nothing downstream would report the NaN --- it would just
        # be written into the potential caches by the reseed below.
        if self._ladder_adapt and not kwargs.get("adapt_beta_min", False):
            if float(jnp.asarray(self.betas)[-1]) <= 0.0:
                log.warning(
                    "ladder adaptation disabled: beta_min is 0, which this parametrization "
                    "cannot represent (it is an infinite temperature), and adapting from it "
                    "yields a NaN ladder. The ladder is held fixed; pass a small positive "
                    "beta_min to adapt it, or adapt_ladder=False to silence this.")
                self._ladder_adapt = False
        self._swap_target = float(kwargs.get("swap_target_accept", OPTIMAL_SWAP_ACCEPT))
        self._ladder_n0 = float(kwargs.get("ladder_adapt_n0", DEFAULT_N0))
        self._ladder_kappa = float(kwargs.get("ladder_adapt_kappa", DEFAULT_KAPPA))
        self._ladder_count = 0
        # Endpoints held by default: `beta_min` is a modelling statement (how deep is the barrier),
        # while the spacing is the tuning problem. Freeing them lets `rho` run away -- see above.
        self._ladder_t_max = (None if kwargs.get("adapt_beta_min", False)
                              else float(1.0 / jnp.asarray(self.betas)[-1]))
        self._rho = (rho_from_betas(self.betas, self._ladder_t_max)
                     if self._ladder_adapt else None)
        if self._ladder_adapt:
            log.info("ladder adaptation: %d interior gap(s) toward swap acceptance %.3f "
                     "(Miasojedow-Moulines-Vihola), beta_min %s",
                     self.n_temperatures - 1, self._swap_target,
                     "free" if self._ladder_t_max is None else f"held at {1/self._ladder_t_max:.4g}")

    def _postprocess_hooks(self, state):
        state = super()._postprocess_hooks(state)
        if not self._ladder_adapt or self._phase is not Phase.WARMUP:
            return state

        alpha = jnp.asarray(state.diagnostics["swap_accept_prob"])[:self.n_temperatures - 1]
        self._ladder_count += 1
        gain = rm_gain(self._ladder_count, self._ladder_n0, self._ladder_kappa)
        self._rho = self._rho + gain * (alpha - self._swap_target)
        betas = betas_from_rho(self._rho, self._ladder_t_max)
        self.betas = betas                       # keep the Python-side copy in step, for reporting
        state = state._replace(ham_params={**state.ham_params, BETAS_KEY: betas})
        return self._reseed_at_new_betas(state)

    def _reseed_at_new_betas(self, state):
        """Recompute the cached potential values and gradients at the ladder just written.

        **This is not an optimization; without it the ladder adaptation destroys the chain.** The
        cache holds ``w_k V_k`` and ``w_k grad V_k`` evaluated at the *previous* beta, and the
        next trajectory's leading half-kick reads the gradient straight back out of it
        (``cached_gradient=True``), while ``log_prob`` is the acceptance baseline. Move beta and
        those numbers describe a Hamiltonian nobody is integrating: the step is charged a
        spurious energy error of order ``delta_beta * V``, every step.

        It is the same invalidation a swap causes, for the same reason, and
        :meth:`~mimcs.pt.ReplicaExchangeMixin._swap` has always re-seeded after one --- moving a
        replica to another rung's beta and moving a rung's beta under a replica are the same
        event seen from two sides.

        What hid it is the *scale* of ``V``. On a 2-d Gaussian ``delta_beta * V`` is ~1e-3 nats
        and invisible, which is what every ladder test used. On `hmm_gaussian` (T = 500,
        ``V ~ 1e5``) it is ~300 nats per step: acceptance collapses to zero, Robbins--Monro drives
        the step size to ~1e-13 within 25 iterations, and NUTS then builds a maximum-depth tree
        every step without ever U-turning. The ladder drifting afterwards is a *consequence* of
        that frozen chain, not the cause.

        Costs one vmapped value-and-gradient per warmup iteration, beside the swap's own.
        """
        # Carries the ladder just written, and no kinetic cache: this runs **eagerly**, once per
        # warmup iteration, and reseeds the potentials only. Building the cache here dispatches
        # the low-rank Woodbury recursion op by op and costs more than the hoist ever saves.
        ctx = self.context(state, kinetic_cache=False)
        values, grads = self._reseed(state.coordinate, ctx)
        return state._replace(
            log_prob=-sum(values.values()),
            potential_values=values,
            potential_grads=grads)

    def _warmup_end_hooks(self, completed: int, stopped: bool) -> None:
        super()._warmup_end_hooks(completed, stopped)
        if self._ladder_adapt:
            log.info("ladder adaptation ended after %d update(s): betas %s",
                     self._ladder_count,
                     np.array2string(np.asarray(self.betas), precision=4))
