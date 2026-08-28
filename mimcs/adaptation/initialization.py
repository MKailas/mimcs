"""Sampler initialization mixins: a reasonable starting position and step size.

These mixins contribute to the ``_initialize_hooks`` chain that :meth:`BaseSampler.initialize`
runs once, before warmup (adaptation never runs during it). They are inert unless ``initialize``
is called. Kept as separate mixins behind one cooperative hook so a future joint initializer
(e.g. Pathfinder --- position + metric + step size) can slot in the same way.

* :class:`UniformInit` --- draw the initial *coordinate* (the unconstrained space the sampler
  works in) from ``U(-radius, radius)`` per dimension (Stan's default ``radius = 2``), retrying
  until the draw has finite target density. Appropriate for our Euclidean and bounded parameters,
  whose coordinate is the identity / log / logit link.
* :class:`StepSizeLineSearch` --- a backtracking line search for the initial step size: with one
  random momentum, halve the step from its starting value until a single-leapfrog (MALA)
  acceptance reaches a (high, conservative) target, so the chain starts moving; warmup's
  Robbins--Monro adaptation tunes it from there.

Both are HMC-specific (they use the Hamiltonian machinery: ``state_at_coordinate``,
``sample_momentum``, the integrator). Draws use a dedicated PRNG substream derived from the
sampler seed, independent of the per-step RNG buffer.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .._logging import get_logger

log = get_logger(__name__)


def _init_key(seed: int, tag: int):
    """A dedicated PRNG substream for initialization (independent of the per-step buffer)."""
    return jax.random.fold_in(jax.random.PRNGKey(int(seed)), tag)


class UniformInit:
    """Mixin: initialize the coordinate from ``U(-radius, radius)`` (finite-density retry)."""

    def _init_hooks(self, **kwargs):
        self._init_radius = float(kwargs.get("init_radius", 2.0))
        self._init_max_tries = int(kwargs.get("init_max_tries", 100))
        super()._init_hooks(**kwargs)

    def _initialize_hooks(self, state):
        """Draw until the **sampler's own** target is finite at the candidate.

        The test is ``state_at_coordinate(...).log_prob``, i.e. ``-sum(potential_values)``, rather
        than ``model.log_prob_at_coordinate``. For an ordinary sampler the two are the same number
        (the potentials *are* the model's components plus the chart Jacobian), but they part
        company as soon as the sampler's target is not the model's: parallel tempering samples a
        beta-weighted product whose ladder is carried in the context, and asking the
        :class:`~mimcs.pt.ProductModel` for a coordinate-space density has no good answer (doc 13).
        The candidate state is built either way, so the accepted draw costs nothing extra.
        """
        state = super()._initialize_hooks(state)
        model, r = self.model, self._init_radius
        key = _init_key(self._seed, 1)
        for attempt in range(1, self._init_max_tries + 1):
            key, sub = jax.random.split(key)
            u = jax.random.uniform(sub, (model.coord_dim,), minval=-r, maxval=r)
            candidate = self.state_at_coordinate(state, u)     # new physical point at coordinate u
            log_prob = float(candidate.log_prob)
            if np.isfinite(log_prob):
                log.debug("UniformInit: coordinate drawn from U(-%g, %g)^%d on attempt %d "
                          "(log density %.6g)", r, r, model.coord_dim, attempt, log_prob)
                return candidate
            log.debug("UniformInit: attempt %d gave a non-finite density; redrawing", attempt)
        log.error("UniformInit: no finite-density coordinate in U(-%g, %g)^%d after %d tries",
                  r, r, model.coord_dim, self._init_max_tries)
        raise RuntimeError(
            f"UniformInit: no finite-density coordinate found in "
            f"U(-{r}, {r})^{model.coord_dim} after {self._init_max_tries} tries")


class StepSizeLineSearch:
    """Mixin: initialize the step size by a backtracking MALA line search from ``step_size``.

    The search is scalar --- one MALA probe gives one acceptance --- but ``step_size`` need not be:
    parallel tempering with independent per-temperature acceptance carries a ``(K,)`` vector. The
    starting point is then the largest entry (the search only ever shrinks) and the result is
    written back with the *same shape*, since the state's shape is fixed once the kernel is traced.
    """

    def _init_hooks(self, **kwargs):
        step0 = jnp.asarray(kwargs.get("step_size", 0.5), float)
        self._init_step0 = float(jnp.max(step0))                        # start / cap (shrink-only)
        self._init_target_accept = float(kwargs.get("init_target_accept", 0.9))
        self._init_step_min = float(kwargs.get("init_step_size_min", 1e-6))
        self._init_max_halvings = int(kwargs.get("init_step_size_max_halvings", 60))
        super()._init_hooks(**kwargs)

    def _initialize_hooks(self, state):
        state = super()._initialize_hooks(state)       # position is set first (UniformInit deeper)
        from ..hmc.integrators import init_integrator_state

        ctx = self.context(state)
        # One random momentum at the current coordinate; the same p is used for every candidate
        # step, making the line search deterministic given the draw.
        p = self.sample_momentum(self._init_momentum_draw(), state.coordinate, ctx)
        istate0 = init_integrator_state(self.potentials, state.coordinate, p, ctx)
        H0 = self.total_energy(istate0, ctx)

        # ``integrate`` is a ``lax.while_loop``; bound eagerly it rebuilds its cond/body jaxprs
        # and misses the dispatch cache on *every* call, at any shape (the same pathology
        # documented for ``mimcs.optim.minimize``). The search probes it up to
        # ``init_step_size_max_halvings`` times at one shape, so one local ``jax.jit`` --- with
        # ``eps`` **traced**, so the halvings share a compilation --- pays for itself on the
        # second probe. Measured: ``initialize()`` 810 -> 121 ms on a 2-d Gaussian, 4.6 -> 0.9 s
        # on a 200-d funnel. Local to this call: ``istate0``/``H0``/``ctx`` are fixed for the
        # search, so closing over them is correct here rather than a cache hazard.
        @jax.jit
        def _accept(eps):
            proposed = self.integrator.integrate(istate0, eps, 1, ctx)
            log_alpha = H0 - self.total_energy(proposed, ctx)
            return jnp.where(jnp.isfinite(log_alpha),
                             jnp.minimum(1.0, jnp.exp(log_alpha)), 0.0)

        def mala_accept(eps):
            return float(_accept(jnp.asarray(eps, float)))

        eps, halvings, accept = self._init_step0, 0, float("nan")
        for _ in range(self._init_max_halvings):
            if eps <= self._init_step_min:
                log.warning("StepSizeLineSearch: the search hit its floor %g without reaching a "
                            "MALA acceptance of %.2f (last measured %s); the target is very "
                            "stiff at the starting point", self._init_step_min,
                            self._init_target_accept,
                            "none" if np.isnan(accept) else f"{accept:.3f}")
                break
            accept = mala_accept(eps)
            if accept >= self._init_target_accept:
                break
            eps *= 0.5
            halvings += 1
        log.info("initial step size %.4g (%d halving(s) from %.4g; MALA acceptance %.3f, "
                 "target %.2f)", eps, halvings, self._init_step0, accept,
                 self._init_target_accept)
        return state._replace(step_size=jnp.full_like(state.step_size, eps))

    def _init_momentum_draw(self):
        """A one-off ``rng_draw`` (momentum components) from a dedicated init substream."""
        subkeys = jax.random.split(_init_key(self._seed, 2), len(self._draw_components))
        return self._rng_draw_class(**{
            comp.name: comp.generator(subkeys[i], comp.shape).astype(comp.dtype)
            for i, comp in enumerate(self._draw_components)})
