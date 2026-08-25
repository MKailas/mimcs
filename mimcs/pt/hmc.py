"""Independent per-temperature acceptance, for the samplers that can have it.

Implements the "exactly independent" half of ``docs/design/13_parallel_tempering.md``.

RWMH and HMC propose and accept in one shot, so nothing forces the temperatures to share an
accept/reject decision --- and they should not. Each temperature is then **exactly its own valid
chain**, sharing only the vectorised gradient evaluation. A product-space Metropolis step, which
accepts or rejects all K together, would mix far worse: one badly-behaved temperature would veto
every proposal for the rest.

(NUTS is the exception, and gets no mixin here. A trajectory has no fixed length, so the
temperatures must agree on when to stop doubling; the trajectory is built once in the product
space and one leaf is selected jointly. See doc 13 for why independent *selection* from a shared
trajectory is not obviously valid.)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from .._logging import get_logger
from ..rng import DrawComponent

log = get_logger(__name__)


class IndependentAcceptanceMixin:
    """Accept or reject at each temperature separately, on that temperature's own energy.

    Composed above a fixed-trajectory base (``HMC``, ``RandomizedHMC``); it overrides only the
    Metropolis test, leaving the integration exactly as the base performs it.
    """

    def make_draw_components(self, model, **kwargs):
        comps = super().make_draw_components(model, **kwargs)
        # one acceptance uniform per temperature, replacing the base's single scalar
        return [DrawComponent("accept_threshold", (model.n_temperatures,), jax.random.uniform)
                if c.name == "accept_threshold" else c for c in comps]

    def init_diagnostics(self) -> dict:
        K = self.model.n_temperatures
        d = super().init_diagnostics()
        return {**d, "accept_prob": jnp.zeros((K,)), "accepted": jnp.zeros((K,), bool)}

    def _eps(self, step_size: Array) -> Array:
        """Expand a per-temperature step size to the product coordinate.

        Independent acceptance gives a per-temperature acceptance signal, so an ordinary
        step-size mixin adapts a **vector** of K step sizes here --- unlike the joint-selection
        NUTS path, where one acceptance drives one global step. The integrator takes it as-is:
        its kick and drift are elementwise (doc 13).
        """
        if jnp.ndim(step_size) == 0:
            return step_size
        return jnp.repeat(step_size, self.model.base.coord_dim)

    def per_temperature_energy(self, istate, ctx) -> Array:
        """``H_k`` for every temperature --- the potentials' and kinetics' own per-lane terms."""
        total = jnp.zeros((self.model.n_temperatures,))
        for p in self.potentials:
            total = total + p.per_temperature_values(istate.q, ctx)
        for k in self.kinetics:
            total = total + k.per_temperature_energy(istate, ctx)
        return total

    def _integrate_and_accept(self, state, istate0, ctx, n_steps):
        from ..hmc.integrators import init_integrator_state

        H0 = self.per_temperature_energy(istate0, ctx)
        proposed = self.integrator.integrate(istate0, self._eps(state.step_size), n_steps, ctx)
        H1 = self.per_temperature_energy(proposed, ctx)

        log_alpha = H0 - H1                      # leapfrog is deterministic: log_weight stays 0
        accept_prob = jnp.where(
            jnp.isfinite(log_alpha), jnp.minimum(1.0, jnp.exp(log_alpha)), 0.0)
        accepted = state.rng_draw.accept_threshold < accept_prob

        # Mix the accepted temperatures' endpoints with the rejected ones' starting points. Only
        # (q, p) can be selected this way: the cached potential *values* are sums over
        # temperatures, so a per-temperature choice leaves them describing neither state. Re-seed
        # them instead --- one extra evaluation per step, and unambiguously right.
        mask = jnp.repeat(accepted, self.model.base.coord_dim)
        q = jnp.where(mask, proposed.q, istate0.q)
        p = jnp.where(mask, proposed.p, istate0.p)
        chosen = init_integrator_state(self.potentials, q, p, ctx)

        grad_evals = proposed.integrator_data.get("grad_evals", jnp.zeros(()))
        return chosen, {"accept_prob": accept_prob, "accepted": accepted,
                        "grad_evals": grad_evals, "mean_refine": jnp.zeros(()),
                        "proxy_accept_prob": jnp.mean(accept_prob)}
