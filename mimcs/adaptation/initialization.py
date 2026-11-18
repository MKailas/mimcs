"""Sampler initialization mixins: a reasonable starting position and step size.

These mixins contribute to the ``_initialize_hooks`` chain that :meth:`BaseSampler.initialize`
runs once, before warmup (adaptation never runs during it). They are inert unless ``initialize``
is called. There are two independent jobs, and so two kinds of mixin:

* a **position** initializer --- :class:`UniformInit`; a future Pathfinder is the other --- sets
  the starting point. At most one belongs on a sampler (the factory's ``spec.position_init`` slot
  makes that structural);
* a **step-size** initializer --- :class:`StepSizeLineSearch` --- sets the starting step for the
  position it is handed (``spec.step_size_init``). It sits *left of* the position initializer in
  the MRO, so it runs after it.

**The fixed set.** ``sampler.init_fixed`` names the parameters whose start the caller supplied
(``build(init=...)``, or the ``init_fixed`` constructor kwarg). A position initializer must keep
them, and does so in *sample* space: :meth:`~mimcs.model.Model.pin_fixed` overlays them on each
candidate, re-charting a fixed child under its free parent's new value. The fixed values are
already in the state --- ``make_initial_state`` built it from them --- so the mixins need only the
names.

* :class:`UniformInit` --- draw the initial *coordinate* (the unconstrained space the sampler
  works in) from ``U(-radius, radius)`` per dimension (Stan's default ``radius = 2``), overlay the
  fixed set, and retry until the candidate has a finite target density *and* a finite gradient (as
  Stan does). Every chart maps its own well-behaved interior point to 0, so the box is a sensible
  neighbourhood for every parameter type, not only the Euclidean and bounded ones.
* :class:`StepSizeLineSearch` --- a backtracking line search for the initial step size: with one
  random momentum, halve the step from its starting value until a single-leapfrog (MALA)
  acceptance reaches a (high, conservative) target, so the chain starts moving; warmup's
  Robbins--Monro adaptation tunes it from there.

Both are HMC-specific (they use the Hamiltonian machinery: ``state_at_coordinate``,
``sample_momentum``, the integrator). Draws use a dedicated PRNG substream derived from the
sampler seed, independent of the per-step RNG buffer. The discrete labels are drawn by the Gibbs
sweep's own ``_initialize_hooks`` (it sits deeper, so the labels are in place before the position
is tested against them), which honours the same fixed set.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from .._logging import get_logger

log = get_logger(__name__)


def _init_key(seed: int, tag: int):
    """A dedicated PRNG substream for initialization (independent of the per-step buffer)."""
    return jax.random.fold_in(jax.random.PRNGKey(int(seed)), tag)


def _all_finite(grads: dict) -> bool:
    """Are all cached potential gradients finite? One host sync for the lot, not one per potential.

    Under parallel tempering each entry is the rungs' gradients flattened together, so one bad
    rung rejects the candidate --- the right call, since every rung starts from it.
    """
    if not grads:
        return True
    return bool(jnp.all(jnp.stack([jnp.all(jnp.isfinite(g)) for g in grads.values()])))


def fixed_continuous(sampler) -> list[str]:
    """The *continuous* parameters in ``sampler.init_fixed``, in declaration order --- the ones a
    position initializer holds (the labels are the sweep's business)."""
    fixed = sampler.init_fixed
    return [p.name for p in sampler.model.parameters if p.name in fixed]


def _check(name: str, value, ok: bool, need: str):
    if not ok:
        raise ValueError(f"{name} must be {need}, got {value!r}")


class UniformInit:
    """Mixin: initialize the coordinate from ``U(-radius, radius)``, keeping ``init_fixed``
    (retry to finite density and gradient).

    Kwargs: ``init_radius`` (default 2.0) and ``init_max_tries`` (default 100); the factory sets
    them from ``spec.position_init_params`` (``radius``, ``max_tries``).
    """

    def _init_hooks(self, **kwargs):
        self._init_radius = float(kwargs.get("init_radius", 2.0))
        self._init_max_tries = kwargs.get("init_max_tries", 100)
        _check("init_radius", self._init_radius,
               math.isfinite(self._init_radius) and self._init_radius > 0, "finite and > 0")
        _check("init_max_tries", self._init_max_tries,
               isinstance(self._init_max_tries, (int, np.integer)) and self._init_max_tries >= 1,
               "an integer >= 1")
        self._init_max_tries = int(self._init_max_tries)
        super()._init_hooks(**kwargs)

    def _initialize_hooks(self, state):
        """Draw until the **sampler's own** target *and its gradient* are finite at the candidate.

        **The fixed set is overlaid on every draw.** The full ``U(-r, r)`` coordinate is drawn
        whatever is fixed, and :meth:`~mimcs.model.Model.pin_fixed` then puts the fixed
        parameters back (in sample space, exactly). So the free coordinates are the very ones an
        unfixed run would draw, and with nothing fixed the path is the old one, bit for bit. When
        *every* continuous parameter is fixed there is nothing to draw: the supplied point is kept,
        and refused (``RuntimeError``) if its density or gradient is not finite, since no warmup
        recovers from that start.

        **The gradient is checked too**, because a finite density does not imply one: a model using
        ``where`` over a branch that is ``NaN`` where it is not selected has a finite density and a
        ``NaN`` gradient there, and so can a density that is merely enormous in float32 (the cscm
        Gamma GLM at a U(-2, 2) start: ``lp ~ -1e12``, gradient ``NaN``). HMC's first leapfrog reads
        that gradient, and the first non-finite score then reaches the mass adaptation. The
        candidate state already caches the gradients (``potential_grads``), so the check is free.

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
        fixed = fixed_continuous(self)
        kept = f"; keeping the supplied {', '.join(fixed)}" if fixed else ""
        if fixed and len(fixed) == len(model.parameters):
            if not (np.all(np.isfinite(np.asarray(state.log_prob)))
                    and _all_finite(state.potential_grads)):
                log.error("UniformInit: every continuous parameter was supplied, and the density "
                          "or its gradient is not finite there")
                raise RuntimeError(
                    f"UniformInit: every continuous parameter was supplied ({', '.join(fixed)}), "
                    f"so there is nothing to draw, but the target density (log density "
                    f"{np.asarray(state.log_prob)}) or its gradient is not finite at the supplied "
                    f"point. No sampler can start there; supply a different point, or leave some "
                    f"parameters out of init= so they are drawn.")
            log.info("initial position: every continuous parameter supplied (%s); nothing drawn",
                     ", ".join(fixed))
            return state
        h, idx = state.chart_hyperparams, state.chart_indices
        key = _init_key(self._seed, 1)
        for attempt in range(1, self._init_max_tries + 1):
            key, sub = jax.random.split(key)
            u = jax.random.uniform(sub, (model.coord_dim,), minval=-r, maxval=r)
            if fixed:
                coord, sample = model.pin_fixed(u, state.sample, fixed, h, idx)
                candidate = self.state_at_coordinate(state, coord, sample=sample)
            else:
                candidate = self.state_at_coordinate(state, u)  # new physical point at u
            log_prob = float(jnp.sum(candidate.log_prob))
            if not np.isfinite(log_prob):
                log.debug("UniformInit: attempt %d gave a non-finite density; redrawing", attempt)
                continue
            if not _all_finite(candidate.potential_grads):
                log.debug("UniformInit: attempt %d gave a finite density (%.6g) but a non-finite "
                          "gradient; redrawing", attempt, log_prob)
                continue
            log.info("initial position: U(-%g, %g) draw over %d coordinate(s) on attempt %d "
                     "(log density %.6g)%s", r, r, model.coord_dim, attempt, log_prob, kept)
            return candidate
        hint = (f" with {', '.join(fixed)} held at their supplied values --- those may make the "
                f"density non-finite whatever the other parameters are" if fixed else "")
        log.error("UniformInit: no coordinate with a finite density and gradient in "
                  "U(-%g, %g)^%d after %d tries%s", r, r, model.coord_dim,
                  self._init_max_tries, kept)
        raise RuntimeError(
            f"UniformInit: no coordinate with a finite density and gradient found in "
            f"U(-{r}, {r})^{model.coord_dim} after {self._init_max_tries} tries{hint}")


class StepSizeLineSearch:
    """Mixin: initialize the step size by a backtracking MALA line search from ``step_size``.

    Kwargs: the starting step is the sampler's ``step_size`` (default 0.5); ``init_target_accept``
    (0.9), ``init_step_size_min`` (1e-6) and ``init_step_size_max_halvings`` (60) configure the
    search. The factory sets the last three from ``spec.step_size_init_params``
    (``target_accept``, ``step_size_min``, ``max_halvings``).

    The search only ever **shrinks**: a target much wider than the identity mass assumes starts at
    ``step_size`` rather than above it, and warmup's adaptation has to grow it from there.

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
        self._init_max_halvings = kwargs.get("init_step_size_max_halvings", 60)
        _check("init_target_accept", self._init_target_accept,
               0.0 < self._init_target_accept <= 1.0, "in (0, 1]")
        _check("init_step_size_min", self._init_step_min,
               math.isfinite(self._init_step_min) and self._init_step_min > 0, "finite and > 0")
        _check("init_step_size_max_halvings", self._init_max_halvings,
               isinstance(self._init_max_halvings, (int, np.integer))
               and self._init_max_halvings >= 1, "an integer >= 1")
        self._init_max_halvings = int(self._init_max_halvings)
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
