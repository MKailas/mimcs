"""Sampler initialization mixins: a reasonable starting position and step size.

These mixins contribute to the ``_initialize_hooks`` chain that :meth:`BaseSampler.initialize`
runs once, before warmup (adaptation never runs during it). They are inert unless ``initialize``
is called. There are two independent jobs, and so two kinds of mixin:

* a **position** initializer --- :class:`UniformInit`, :class:`PathfinderInit`, :class:`ADVIInit`
  or :class:`DADVIInit` --- sets the starting point. At most one belongs on a sampler (the
  factory's ``spec.position_init`` slot makes that structural);
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
* :class:`PathfinderInit` --- start at a draw from Pathfinder's normal approximation (Zhang et
  al. 2022; single- or multi-path, :mod:`mimcs.vi`), fitted over the free coordinates given the
  fixed set.
* :class:`ADVIInit` --- start at a draw from a mean-field ADVI fit (plain variational inference
  by Adam, :func:`mimcs.vi.advi`) over the same target.
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
from typing import Callable, NamedTuple

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


def _radius_and_tries(self, kwargs):
    """``init_radius`` / ``init_max_tries``, validated: the start box both position initializers
    draw from (Stan's ``U(-2, 2)``; Pathfinder's ``pi_0``)."""
    self._init_radius = float(kwargs.get("init_radius", 2.0))
    self._init_max_tries = kwargs.get("init_max_tries", 100)
    _check("init_radius", self._init_radius,
           math.isfinite(self._init_radius) and self._init_radius > 0, "finite and > 0")
    _check("init_max_tries", self._init_max_tries,
           isinstance(self._init_max_tries, (int, np.integer)) and self._init_max_tries >= 1,
           "an integer >= 1")
    self._init_max_tries = int(self._init_max_tries)


def _keep_supplied(self, state, fixed, who: str):
    """Every continuous parameter was supplied: keep the point, refusing a non-finite one."""
    if not (np.all(np.isfinite(np.asarray(state.log_prob)))
            and _all_finite(state.potential_grads)):
        log.error("%s: every continuous parameter was supplied, and the density or its gradient "
                  "is not finite there", who)
        raise RuntimeError(
            f"{who}: every continuous parameter was supplied ({', '.join(fixed)}), so there is "
            f"nothing to draw, but the target density (log density "
            f"{np.asarray(state.log_prob)}) or its gradient is not finite at the supplied point. "
            f"No sampler can start there; supply a different point, or leave some parameters "
            f"out of init= so they are drawn.")
    log.info("initial position: every continuous parameter supplied (%s); nothing drawn",
             ", ".join(fixed))
    return state


class UniformInit:
    """Mixin: initialize the coordinate from ``U(-radius, radius)``, keeping ``init_fixed``
    (retry to finite density and gradient).

    Kwargs: ``init_radius`` (default 2.0) and ``init_max_tries`` (default 100); the factory sets
    them from ``spec.position_init_params`` (``radius``, ``max_tries``).
    """

    def _init_hooks(self, **kwargs):
        _radius_and_tries(self, kwargs)
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
            return _keep_supplied(self, state, fixed, "UniformInit")
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


class PathfinderInit:
    """Mixin: start at a Pathfinder draw (Zhang et al. 2022), keeping ``init_fixed``.

    Single-path (``init_pathfinder_paths = 1``, the ``"pathfinder"`` slot value) runs one L-BFGS
    path from a ``U(-radius, radius)`` start and starts the chain at a draw from the
    ELBO-maximising normal approximation along it; multi-path (``"multipath"``, 20 paths by
    default) runs several and Pareto-smoothed-importance-resamples the pooled draws
    (:mod:`mimcs.vi`). The paper's defaults throughout; see :data:`mimcs.vi.PATHFINDER_DEFAULTS`.

    **The target is the free coordinates.** Pathfinder optimizes over the coordinates of the
    parameters *not* in ``init_fixed``: each free vector is scattered into the state's coordinate
    and :meth:`~mimcs.model.Model.pin_fixed` re-charts the fixed parameters under it, so the
    density is the *sample-space* conditional of the free parameters given the fixed ones --- the
    coordinate-space density minus the fixed parameters' own chart Jacobians
    (:meth:`~mimcs.model.Model.log_jacobian`), which for a fixed child of a free parent is not a
    constant. The labels are held where the Gibbs sweep's draw (or ``init=``) put them: it sits
    deeper and runs first.

    **Parallel tempering.** Pathfinder runs once, on the base (``beta = 1``) posterior with the
    cold rung's labels, and each rung starts at its *own* approximate draw --- the next valid one
    in order --- as :class:`UniformInit` draws each rung independently.

    Each start is the next draw (in order) whose density and gradient are finite; if none is, the
    path's last iterate is used, with a WARNING. ``sampler.init_report`` records what happened
    (iterations, the selected iteration, ELBO, Pareto k, distinct draws, evaluations in the
    sampler's gradient-evaluation units, wall time).

    Kwargs: ``init_radius`` / ``init_max_tries`` (the start box, as for :class:`UniformInit`),
    ``init_pathfinder_paths`` (1), ``init_pathfinder_resample`` (100) and
    ``init_pathfinder_<option>`` for each of :data:`mimcs.vi.PATHFINDER_DEFAULTS`. The factory sets
    them from ``spec.position_init_params``.
    """

    def _init_hooks(self, **kwargs):
        from ..vi.pathfinder import PATHFINDER_DEFAULTS, pathfinder_options
        _radius_and_tries(self, kwargs)
        self._pf_paths = kwargs.get("init_pathfinder_paths", 1)
        self._pf_resample = kwargs.get("init_pathfinder_resample", 100)
        for name in ("init_pathfinder_paths", "init_pathfinder_resample"):
            v = kwargs.get(name, 1)
            _check(name, v, isinstance(v, (int, np.integer)) and v >= 1, "an integer >= 1")
        self._pf_paths, self._pf_resample = int(self._pf_paths), int(self._pf_resample)
        self._pf_options = pathfinder_options(
            {k: kwargs[f"init_pathfinder_{k}"] for k in PATHFINDER_DEFAULTS
             if f"init_pathfinder_{k}" in kwargs})
        self.init_report = None
        super()._init_hooks(**kwargs)

    def _initialize_hooks(self, state):
        import time
        from ..vi.pathfinder import single_path, multi_path
        from ..vi.psis import psis

        state = super()._initialize_hooks(state)
        fixed = fixed_continuous(self)
        who = "multi-path Pathfinder" if self._pf_paths > 1 else "Pathfinder"
        if fixed and len(fixed) == len(self.model.parameters):
            return _keep_supplied(self, state, fixed, who)
        t0 = time.perf_counter()
        target = _free_target(self, state, fixed)
        key_start, key_path = jax.random.split(_init_key(self._seed, 3))
        starts, n_checked = _pi0_starts(self, target, key_start, self._pf_paths, who, fixed)

        if self._pf_paths == 1:
            res = single_path(target.logp, starts[0], key_path, **self._pf_options)
            candidates = np.asarray(res.draws)
            paths = jax.tree_util.tree_map(lambda a: np.asarray(a)[None], res)
            khat = (float(psis(np.asarray(res.log_p) - np.asarray(res.log_q))[1])
                    if not bool(res.failed) else math.nan)
            distinct = int(len(np.unique(candidates, axis=0)))
        else:
            mres = multi_path(target.logp, jnp.stack(starts), key_path,
                              resample_size=self._pf_resample, **self._pf_options)
            candidates = np.asarray(mres.draws)
            paths = jax.tree_util.tree_map(np.asarray, mres.paths)
            khat, distinct = mres.khat, mres.distinct

        best = int(np.argmax(np.where(np.isfinite(paths.elbo), paths.elbo, -np.inf)))
        candidate, checked = _start_from_draws(
            self, state, target, candidates, lambda: jnp.asarray(paths.last[best]),
            "the last L-BFGS iterate", who)
        n_checked += checked

        units = len(getattr(self, "potentials", ())) or 1
        n_grad = int(np.sum(paths.n_grad)) + n_checked
        self.init_report = dict(
            method="multipath" if self._pf_paths > 1 else "pathfinder",
            paths=self._pf_paths, n_iter=[int(v) for v in paths.n_iter],
            l_star=[int(v) for v in paths.l_star], elbo=[float(v) for v in paths.elbo],
            failed=int(np.sum(paths.failed)), khat=khat, distinct=distinct,
            grad_evals=n_grad * units, value_evals=int(np.sum(paths.n_value)) * units,
            seconds=time.perf_counter() - t0)
        log.info("initial position: %s over %d free coordinate(s): %s iteration(s), best ELBO "
                 "%.6g at iteration %d, Pareto k %.2f, %d distinct draw(s), %d gradient + %d "
                 "density evaluation(s), %.2f s; log density at the start %.6g%s",
                 who, len(target.free),
                 paths.n_iter[0] if self._pf_paths == 1 else
                 f"{int(np.min(paths.n_iter))}-{int(np.max(paths.n_iter))}",
                 float(paths.elbo[best]), int(paths.l_star[best]), khat, distinct,
                 self.init_report["grad_evals"], self.init_report["value_evals"],
                 self.init_report["seconds"], float(jnp.sum(candidate.log_prob)),
                 f"; keeping the supplied {', '.join(fixed)}" if fixed else "")
        if self.init_report["failed"]:
            log.info("%s: %d of %d path(s) failed (no finite ELBO)", who,
                     self.init_report["failed"], self._pf_paths)
        return candidate


class ADVIInit:
    """Mixin: start at a draw from a mean-field ADVI fit (:func:`mimcs.vi.advi`), keeping
    ``init_fixed``.

    Plain variational inference: a diagonal normal over the free coordinates, fitted by Adam on a
    reparameterised ``K``-draw ELBO from a ``U(-radius, radius)`` start (the same start a
    ``"pathfinder"`` run with the same seed takes), stopped once ``patience`` iterations pass
    without the ELBO estimate improving. The target, the fixed set, parallel tempering and the
    choice of start among the draws are exactly as for :class:`PathfinderInit`; the fallback, if
    no draw is valid, is the fit's mean. ``sampler.init_report`` records the iterations, the best
    ELBO estimate and its iteration, skipped iterations, Pareto k of the draws, distinct draws,
    evaluations (in the sampler's gradient-evaluation units) and wall time.

    Kwargs: ``init_radius`` / ``init_max_tries`` and ``init_advi_<option>`` for each of
    :data:`mimcs.vi.ADVI_DEFAULTS`. The factory sets them from ``spec.position_init_params``.
    """

    def _init_hooks(self, **kwargs):
        from ..vi.meanfield import ADVI_DEFAULTS, advi_options
        _radius_and_tries(self, kwargs)
        self._advi_options = advi_options(
            {k: kwargs[f"init_advi_{k}"] for k in ADVI_DEFAULTS if f"init_advi_{k}" in kwargs})
        self.init_report = None
        super()._init_hooks(**kwargs)

    def _initialize_hooks(self, state):
        import time
        from ..vi.meanfield import advi
        from ..vi.psis import psis

        state = super()._initialize_hooks(state)
        fixed = fixed_continuous(self)
        who = "ADVI"
        if fixed and len(fixed) == len(self.model.parameters):
            return _keep_supplied(self, state, fixed, who)
        t0 = time.perf_counter()
        target = _free_target(self, state, fixed)
        key_start, key_fit = jax.random.split(_init_key(self._seed, 3))   # Pathfinder's start
        starts, n_checked = _pi0_starts(self, target, key_start, 1, who, fixed)
        res = advi(target.logp, starts[0], key_fit, **self._advi_options)
        failed = bool(res.failed)
        candidates = np.asarray(res.draws)
        khat = (float(psis(np.asarray(res.log_p) - np.asarray(res.log_q))[1])
                if not failed else math.nan)
        candidate, checked = _start_from_draws(
            self, state, target, candidates, lambda: res.mean, "the fit's mean", who)
        n_checked += checked

        units = len(getattr(self, "potentials", ())) or 1
        self.init_report = dict(
            method="advi", n_iter=int(res.n_iter), best_iter=int(res.best_iter),
            elbo=float(res.elbo), skipped=int(res.n_skipped),
            dropped=int(res.n_dropped), failed=int(failed), khat=khat,
            distinct=int(len(np.unique(candidates, axis=0))),
            grad_evals=(int(res.n_grad) + n_checked) * units,
            value_evals=int(res.n_value) * units, seconds=time.perf_counter() - t0)
        log.info("initial position: ADVI (mean-field) over %d free coordinate(s): %d "
                 "iteration(s), best ELBO estimate %.6g at iteration %d, %d skipped, Pareto k "
                 "%.2f, %d gradient + %d density evaluation(s), %.2f s; log density at the start "
                 "%.6g%s", len(target.free), int(res.n_iter), float(res.elbo),
                 int(res.best_iter), int(res.n_skipped), khat, self.init_report["grad_evals"],
                 self.init_report["value_evals"], self.init_report["seconds"],
                 float(jnp.sum(candidate.log_prob)),
                 f"; keeping the supplied {', '.join(fixed)}" if fixed else "")
        if failed:
            log.warning("ADVI: no iteration had a finite density and gradient at all of its draws;"
                        " the fit never moved from its start")
        return candidate


class DADVIInit:
    """Mixin: start at a draw from a fixed-draw mean-field VI fit (:func:`mimcs.vi.dadvi`),
    keeping ``init_fixed``.

    The mean-field ELBO with its ``K`` reparameterisation draws fixed once, maximised by the
    trust-region Newton-CG (``optimizer="trust_ncg"``, the default) or L-BFGS from a
    ``U(-radius, radius)`` start (the start a ``"pathfinder"`` run with the same seed takes); the
    draws come from its linear-response covariance (``covariance="lr"``, from the exact Hessian;
    ``"lr_lbfgs"``, from the secant pairs of the fit's steps) or the fitted diagonal normal. The
    target, the fixed set, parallel tempering and the choice of start among the draws are exactly
    as for :class:`PathfinderInit`; the fallback, if no draw is valid, is the fit's centroid.
    ``sampler.init_report`` records the optimizer, the iterations (accepted steps), the
    Hessian-vector products and capped CG solves (``trust_ncg``) and why the fit stopped, the
    objective at the fit (``elbo``, in-sample) and the mean ``log p - log q`` of the draws
    (``elbo_out``), the read-out drawn from, Pareto k, distinct draws, evaluations (in the
    sampler's gradient-evaluation units) and wall time.

    Kwargs: ``init_radius`` / ``init_max_tries`` and ``init_dadvi_<option>`` for each of
    :data:`mimcs.vi.DADVI_DEFAULTS`. The factory sets them from ``spec.position_init_params``.
    """

    def _init_hooks(self, **kwargs):
        from ..vi.deterministic import DADVI_DEFAULTS, dadvi_options
        _radius_and_tries(self, kwargs)
        self._dadvi_options = dadvi_options(
            {k: kwargs[f"init_dadvi_{k}"] for k in DADVI_DEFAULTS if f"init_dadvi_{k}" in kwargs})
        self.init_report = None
        super()._init_hooks(**kwargs)

    def _initialize_hooks(self, state):
        import time
        from ..vi.deterministic import READOUTS, STOP_REASONS, dadvi
        from ..vi.psis import psis

        state = super()._initialize_hooks(state)
        fixed = fixed_continuous(self)
        who = "DADVI"
        if fixed and len(fixed) == len(self.model.parameters):
            return _keep_supplied(self, state, fixed, who)
        t0 = time.perf_counter()
        target = _free_target(self, state, fixed)
        key_start, key_fit = jax.random.split(_init_key(self._seed, 3))   # Pathfinder's start
        starts, n_checked = _pi0_starts(self, target, key_start, 1, who, fixed)
        res = dadvi(target.logp, starts[0], key_fit, **self._dadvi_options)
        failed = bool(res.failed)
        candidates = np.asarray(res.draws)
        log_w = np.asarray(res.log_p) - np.asarray(res.log_q)
        khat = float(psis(log_w)[1]) if not failed else math.nan
        elbo_out = float(np.mean(log_w)) if not failed else -math.inf
        candidate, checked = _start_from_draws(
            self, state, target, candidates, lambda: res.center, "the fit's centroid", who)
        n_checked += checked

        units = len(getattr(self, "potentials", ())) or 1
        stop = STOP_REASONS[int(res.stop)]
        self.init_report = dict(
            method="dadvi", optimizer=self._dadvi_options["optimizer"], n_iter=int(res.n_iter),
            n_hvp=int(res.n_hvp), n_cg_capped=int(res.n_cg_capped), stop=stop,
            elbo=float(res.elbo),
            elbo_out=elbo_out, khat=khat, distinct=int(len(np.unique(candidates, axis=0))),
            readout=READOUTS[int(res.readout)], lr_failed=int(bool(res.lr_failed)),
            lr_capped=int(bool(res.lr_capped)), lr_skipped=int(bool(res.lr_skipped)),
            omega0=float(res.omega0), failed=int(failed),
            grad_evals=(int(res.n_grad) + n_checked) * units,
            value_evals=int(res.n_value) * units, seconds=time.perf_counter() - t0)
        log.info("initial position: DADVI (fixed-draw mean-field VI) over %d free coordinate(s): "
                 "%d %s step(s) (stopped: %s), objective %.6g, draws' ELBO %.6g, Pareto "
                 "k %.2f, %d gradient + %d density evaluation(s), %.2f s; log density at the "
                 "start %.6g%s", len(target.free), int(res.n_iter),
                 {"trust_ncg": "trust-region Newton-CG", "lbfgs": "L-BFGS"}[
                     self._dadvi_options["optimizer"]], stop, float(res.elbo),
                 elbo_out, khat, self.init_report["grad_evals"], self.init_report["value_evals"],
                 self.init_report["seconds"], float(jnp.sum(candidate.log_prob)),
                 f"; keeping the supplied {', '.join(fixed)}" if fixed else "")
        if res.lr_capped:
            from ..vi.deterministic import lr_memory_gb
            log.info("DADVI: the exact Hessian over %d free coordinates would take about %.2g GiB, "
                     "over lr_max_gb %g; drew from the lr_lbfgs (secant) read-out",
                     len(target.free),
                     lr_memory_gb(len(target.free)), self._dadvi_options["lr_max_gb"])
        if failed:
            log.warning("DADVI: no step was possible from the start (stopped: %s); starting "
                        "near the start point", stop)
        elif res.lr_skipped:
            log.warning("DADVI: the fit did not converge (stopped: %s), so the exact Hessian was "
                        "not formed; drew from the %s read-out instead", stop,
                        READOUTS[int(res.readout)])
        elif res.lr_failed:
            log.warning("DADVI: the %s covariance was not positive definite; drew from the %s "
                        "read-out instead", self._dadvi_options["covariance"],
                        READOUTS[int(res.readout)])
        return candidate


class _FreeTarget(NamedTuple):
    """A variational initializer's target: the free coordinates' log density (see
    :class:`PathfinderInit`), and how to turn a free vector back into a rung's state."""
    free: np.ndarray
    lanes: int
    full: Callable          # free vector -> (coordinate, sample or None) of one rung
    logp: Callable          # free vector -> log density (traceable)
    valid: Callable         # free vector -> bool: finite density and gradient
    fixed: list


def _free_target(self, state, fixed) -> _FreeTarget:
    """The sample-space conditional of the free parameters given the fixed ones, on the base
    posterior with the cold rung's labels (:class:`PathfinderInit`)."""
    model = self.model
    base = getattr(model, "base", model)
    lanes = int(getattr(model, "n_temperatures", 1))
    h, idx = state.chart_hyperparams, state.chart_indices
    coord0 = state.coordinate[:base.coord_dim]
    sample0 = state.sample[:base.ambient_dim]
    disc0 = state.discrete[:base.discrete_dim] if base.discrete_dim else None
    free = np.flatnonzero(~base.coordinate_mask(fixed))

    def full(u):
        if not fixed:
            return u, None
        return base.pin_fixed(coord0.at[free].set(u), sample0, fixed, h, idx)

    def logp(u):
        c, _ = full(u)
        lp = base.log_prob_at_coordinate(c, h, idx, disc0)
        return lp - base.log_jacobian(c, h, idx, fixed) if fixed else lp

    vg = jax.jit(jax.value_and_grad(logp))

    def valid(u):
        v, g = vg(u)
        return bool(jnp.isfinite(v) & jnp.all(jnp.isfinite(g)))

    return _FreeTarget(free, lanes, full, logp, valid, fixed)


def _pi0_starts(self, target: _FreeTarget, key, count: int, who: str, fixed):
    """``count`` starts from ``pi_0 = U(-r, r)`` over the free coordinates, each redrawn to a
    finite density and gradient. Returns ``(starts, candidates checked)``."""
    r, n_free = self._init_radius, len(target.free)
    starts, n_checked = [], 0
    for _ in range(count):
        for _attempt in range(1, self._init_max_tries + 1):
            key, sub = jax.random.split(key)
            u = jax.random.uniform(sub, (n_free,), minval=-r, maxval=r)
            n_checked += 1
            if target.valid(u):
                starts.append(u)
                break
        else:
            log.error("%s: no start with a finite density and gradient in U(-%g, %g)^%d "
                      "after %d tries", who, r, r, n_free, self._init_max_tries)
            raise RuntimeError(
                f"{who}: no start with a finite density and gradient found in "
                f"U(-{r}, {r})^{n_free} after {self._init_max_tries} tries"
                + (f" with {', '.join(fixed)} held at their supplied values" if fixed else ""))
    return starts, n_checked


def _start_from_draws(self, state, target: _FreeTarget, candidates, fallback, what: str, who):
    """One start per lane: the next candidate (in order) that is valid, or ``fallback()`` (with a
    WARNING) if none is. Returns ``(state at the start, candidates checked)``."""
    chosen, n_checked = [], 0
    for u in candidates:
        if len(chosen) == target.lanes:
            break
        n_checked += 1
        if target.valid(jnp.asarray(u)):
            chosen.append(jnp.asarray(u))
    if not chosen:
        log.warning("%s: no approximate draw has a finite density and gradient; starting at "
                    "%s instead", who, what)
        chosen = [jnp.asarray(fallback())]
    while len(chosen) < target.lanes:                                 # cycle if short
        chosen.append(chosen[len(chosen) % max(1, len(chosen))])

    rungs = [target.full(u) for u in chosen]
    coord = jnp.concatenate([c for c, _ in rungs])
    sample = None if not target.fixed else jnp.concatenate([s for _, s in rungs])
    candidate = self.state_at_coordinate(state, coord, sample=sample)
    if not (np.isfinite(float(jnp.sum(candidate.log_prob)))
            and _all_finite(candidate.potential_grads)):
        log.error("%s: the chosen start has a non-finite density or gradient", who)
        raise RuntimeError(f"{who}: the chosen start has a non-finite density or gradient "
                           f"under the sampler's own target")
    return candidate, n_checked
