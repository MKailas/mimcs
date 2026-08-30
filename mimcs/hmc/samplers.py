"""Basic Hamiltonian Monte Carlo (axis 3 of the modular HMC design).

Implements ``BaseHMC`` and the fixed-length ``HMC`` sampler from
``docs/design/06_hamiltonian_monte_carlo.md``, built on the shared sampling loop
(``mimcs.samplers.base``) and the integrator/Hamiltonian components.

By default ``BaseHMC`` assembles sensible components from the model --- one
``ModelPotential`` per log-density component (plus a ``JacobianPotential`` when any
parameter is constrained), a single diagonal or dense kinetic over the whole
coordinate, and a leapfrog integrator --- so ``HMC(model, ...)`` just works; the
components can also be supplied explicitly for customization.

Gradient caching across kernel calls: the sampler ``State`` carries the per-potential
value/gradient caches at the current coordinate, so each kernel call seeds the
``IntegratorState`` without recomputing the gradient (the accepted state's gradient
becomes the next step's leading-kick cache). This relies on ``preprocess`` not moving
the coordinate or changing the charts --- true for the step-size and mass-matrix
adaptations here (a future chart adaptation would need to refresh the cache).
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .._logging import get_logger
from ..rng import DrawComponent, zero_draw
from ..samplers.base import BaseSampler

log = get_logger(__name__)
from .state import IntegratorState, HamiltonianContext
from .hamiltonians import (
    ModelPotential, JacobianPotential, DiagonalQuadraticKinetic,
    DenseQuadraticKinetic, total_energy)
from .integrators import leapfrog, init_integrator_state


class HMCState(NamedTuple):
    """State for Hamiltonian Monte Carlo."""

    coordinate: Array          # position in coordinate space, flat (n,)
    sample: Array              # position in ambient space, flat
    log_prob: Array            # scalar: coordinate-space target log-density at `coordinate`
    rng_draw: Any              # typed RngDraw NamedTuple (momentum, accept_threshold)
    chart_hyperparams: tuple
    chart_indices: tuple
    momentum: Array            # last sampled momentum (diagnostic)
    step_size: Array           # scalar leapfrog step size
    ham_params: dict           # adapted component params, keyed by component id (mass)
    potential_values: dict     # cached V_i at `coordinate`
    potential_grads: dict      # cached grad V_i at `coordinate`
    diagnostics: dict = {}     # per-transition diagnostics filled by the kernel (accept_prob,
                               # accepted, grad_evals, ...; NUTS adds diverging/tree_depth/n_leaves)


def default_potentials(model):
    """One ModelPotential per log-density component; a JacobianPotential if needed.

    Ids must be distinct: a potential's id keys its slot in the ``potential_values`` /
    ``potential_grads`` caches, and the leading half-kick of each step reads that slot back
    (``cached_gradient=True``). Two potentials sharing an id would therefore not fail --- one
    would silently integrate with the other's gradient --- so the collision is checked here.
    """
    potentials = [ModelPotential(model, name) for name in model.log_prob_fns]
    if any(not p.is_euclidean() for p in model.parameters):
        potentials.append(JacobianPotential(model))
    ids = [p.id for p in potentials]
    if len(set(ids)) != len(ids):
        raise ValueError(
            f"duplicate potential id(s) in {ids}: each Hamiltonian component needs a unique id "
            "(it keys the per-component gradient cache)")
    return potentials


def split_potentials(model, potentials=None):
    """Split ``potentials`` into ``(cheap, expensive)`` for a multi-rate integrator.

    ``model.cheap_components`` (doc 05) names the components whose gradient is cheap. A
    :class:`JacobianPotential` is cheap **by construction** --- it never touches data --- and is
    not a model component, so it can never be named there; it always joins the cheap group.
    Everything else is expensive, which is the conservative reading and matches
    :meth:`mimcs.model.Model.is_cheap`. Declaration order is preserved within each group, so the
    resulting op list is deterministic.

    This is a statement about the *split*, not about whether taking it is worthwhile: a model
    that declares nothing cheap yields ``([V_jacobian], everything else)``, and deciding that a
    lone Jacobian is not a reason to go multi-rate belongs to the caller (see
    :func:`mimcs.factory.rules.multirate_integrator_rule`).
    """
    if potentials is None:
        potentials = default_potentials(model)
    cheap, expensive = [], []
    for p in potentials:
        # A potential may be *wrapped* --- parallel tempering's TemperedProductPotential holds one
        # model component per wrapper and keeps it as ``inner`` --- and cheapness is a property of
        # the component underneath, not of the wrapper. Without unwrapping, every tempered
        # component reads as expensive and the split degenerates.
        inner = getattr(p, "inner", p)
        is_cheap = (isinstance(inner, JacobianPotential)
                    or (isinstance(inner, ModelPotential) and model.is_cheap(inner.component)))
        (cheap if is_cheap else expensive).append(p)
    return cheap, expensive


def make_kinetic(metric: str, id: str = "T"):
    if metric == "diagonal":
        return DiagonalQuadraticKinetic(id)
    if metric == "dense":
        return DenseQuadraticKinetic(id)
    raise ValueError(f"unknown metric '{metric}' (expected 'diagonal' or 'dense')")


class BaseHMC(BaseSampler):
    """Base HMC sampler: holds the Hamiltonian components and integrator.

    Subclasses implement ``build_trajectory_and_select`` (how a trajectory is built and
    a point chosen). ``BaseHMC`` provides momentum refresh, total energy, the context,
    and the cache-seeding initial state.
    """

    state_class = HMCState

    #: Does this sampler give the integrator per-step randomness? A randomized integrator (see
    #: :class:`~mimcs.hmc.MarkovianLineSearchIntegrator`, whose ``n_rng_per_step`` says how many
    #: draws it wants per step) can only be *randomized* if its host declares the draws for it.
    #: ``BaseHMC`` builds a whole trajectory with one ``integrate`` call and has nowhere to put
    #: per-step coins, so a randomized integrator silently runs deterministically here --- which
    #: is why the factory refuses that combination. :class:`~mimcs.hmc.BaseNUTS` overrides this to
    #: True: it calls ``step`` per leaf and declares a coin array (doc 06). Flipping this to True
    #: is what a future randomized fixed-length integrator (WAL-HMC) would do.
    supplies_integrator_rng = False

    def __init__(self, model, init_position, *, step_size: float = 0.5,
                 metric: str = "diagonal", potentials=None, kinetic=None, kinetics=None,
                 integrator=None, seed: int = 0, buffer_size: int = 1024, **kwargs):
        # Components must exist before super().__init__(), which queries the kinetics
        # for their RNG draw components and builds the initial state. ``kinetics`` is a list
        # of kinetic components (one per coordinate block); a single ``kinetic`` or a
        # ``metric`` string is wrapped into a one-element whole-space list.
        self.potentials = potentials if potentials is not None else default_potentials(model)
        if kinetics is not None:
            self.kinetics = list(kinetics)
        elif kinetic is not None:
            self.kinetics = [kinetic]
        else:
            self.kinetics = [make_kinetic(metric)]
        self.integrator = (integrator if integrator is not None
                           else leapfrog(self.potentials, self.kinetics))
        log.debug("HMC components: potentials %s, kinetics %s, integrator %s, initial step "
                  "size %.4g", [type(p).__name__ for p in self.potentials],
                  [f"{k.id}[{type(k).__name__}]" for k in self.kinetics],
                  type(self.integrator).__name__, step_size)
        super().__init__(model, init_position, seed=seed, buffer_size=buffer_size,
                         step_size=step_size, **kwargs)

    # --- RNG: each kinetic declares its (id-namespaced) draws; HMC adds the accept threshold ---

    def make_draw_components(self, model, **kwargs):
        comps = []
        for k in self.kinetics:
            comps.extend(k.make_draw_components(model.coord_dim))
        comps.append(DrawComponent("accept_threshold", (), jax.random.uniform))
        return comps

    def init_diagnostics(self) -> dict:
        z = jnp.zeros(())
        return {**super().init_diagnostics(), "accept_prob": z, "accepted": jnp.asarray(False),
                "grad_evals": z, "mean_refine": z, "proxy_accept_prob": z}

    # --- shared services: aggregate the kinetic components ---

    def context(self, state, *, kinetic_cache: bool = True) -> HamiltonianContext:
        """The per-trajectory constants the Hamiltonian components read.

        Built **once per kernel call**, before the trajectory loop, so anything placed here is a
        loop constant. A kinetic may define ``precompute(ctx)`` to put a quantity that depends
        only on ``ham_params`` into ``kinetic_cache`` --- see
        :class:`LowRankQuadraticKinetic`, whose Woodbury factors would otherwise be rebuilt on
        every leaf because XLA will not hoist them out of the ``while_loop``. Returning ``None``
        from ``precompute`` means *nothing to cache* and contributes no entry --- which is how a
        wrapper that delegates to a block that may or may not precompute anything
        (:class:`~mimcs.pt.ProductKinetic`) leaves the cache untouched rather than storing a
        ``None`` under an id a reader would then try to unpack.

        ``kinetic_cache=False`` skips that work, for a caller that reads only the potentials.
        It is a **performance** switch, never a correctness one: every consumer falls back to
        computing its own factors, so the numbers are the same either way. It matters because
        ``kernel`` is jitted --- there ``precompute`` is traced once and costs nothing --- while
        the reseeding callers run **eagerly**, dispatching the ``O(J^2 d)`` recursion primitive by
        primitive. Measured at rank 8, ``d = 200``, ``K = 4``: 43.8 ms per eager call against
        0.10 ms traced, and the tempered ladder reseeds once per warmup iteration, which turned
        this optimization into a 3.6x warmup *regression* before the opt-out existed.
        """
        ctx = HamiltonianContext(state.chart_hyperparams, state.chart_indices, state.ham_params)
        if not kinetic_cache:
            return ctx
        cache = {}
        for k in self.kinetics:
            precompute = getattr(k, "precompute", None)
            if precompute is not None:
                value = precompute(ctx)
                if value is not None:
                    cache[k.id] = value
        return ctx._replace(kinetic_cache=cache) if cache else ctx

    def total_energy(self, istate, ctx) -> Array:
        return total_energy(istate, self.potentials, self.kinetics, ctx)

    def _current_score(self, state) -> Array:
        """The total score at the current coordinate: the gradient of the log-density. The state
        caches each potential's gradient ``grad V_i`` (with ``log pi = -sum V_i``), so the score
        is ``-sum grad V_i`` --- already computed, hence nearly free to save."""
        return -sum(state.potential_grads.values())

    def kinetic_velocity(self, istate, ctx) -> Array:
        """Aggregate velocity ``grad_p sum_j T_j`` over the kinetic components (drift + U-turn)."""
        v = jnp.zeros_like(istate.q)
        for k in self.kinetics:
            v = k.velocity_into(v, istate, ctx)
        return v

    def sample_momentum(self, draw, q, ctx) -> Array:
        """Refresh the full momentum, each component filling its coordinate block."""
        p = jnp.zeros((self.model.coord_dim,))
        for k in self.kinetics:
            p = k.sample_into(p, draw, q, ctx)
        return p

    # --- initial state ---

    def make_initial_state(self, init_position) -> HMCState:
        from ..samplers.metropolis import _as_sample_flat
        model = self.model
        h = model.init_chart_hyperparams()
        c = model.init_chart_indices()

        sample = _as_sample_flat(model, init_position)
        coordinate = model.sample_to_coordinate(sample, h, c)

        ham_params = {k.id: k.initial_mass_params(model.coord_dim) for k in self.kinetics}
        ctx = HamiltonianContext(h, c, ham_params)

        seed_state = init_integrator_state(
            self.potentials, coordinate, jnp.zeros((model.coord_dim,)), ctx)
        log_prob = -sum(seed_state.potential_values.values())

        return HMCState(
            coordinate=coordinate,
            sample=sample,
            log_prob=log_prob,
            rng_draw=zero_draw(self._rng_draw_class, self._draw_components),
            chart_hyperparams=h,
            chart_indices=c,
            momentum=jnp.zeros((model.coord_dim,)),
            step_size=jnp.asarray(self._kwargs.get("step_size", 0.5), float),
            ham_params=ham_params,
            potential_values=seed_state.potential_values,
            potential_grads=seed_state.potential_grads,
            diagnostics=self.init_diagnostics(),
        )

    def _reseed_caches(self, coordinate, ctx):
        """``(potential_values, potential_grads)`` at ``coordinate`` under ``ctx``.

        The one place the potential caches are refreshed outside the kernel, shared by chart
        recharting (:meth:`state_at_coordinate`) and by a moved temperature ladder
        (:meth:`mimcs.pt.LadderAdaptation._reseed_at_new_betas`).

        **Both arguments are traced; only ``self.potentials`` is closed over.** That split is the
        whole correctness argument, and it is the same one that makes ``_kernel_jit`` safe: the
        potentials are *structure*, while every adapted quantity --- the chart hyperparameters, the
        mass, the ladder --- reaches them through ``ctx``. Close over ``ctx`` instead and the jit
        bakes in the **first** call's chart and ladder as compile-time constants; every later
        reseed would then refresh the cache against a chart nobody is integrating, with the right
        shapes, the right dtypes, and no error. That is precisely the failure this method exists to
        prevent (see :meth:`~mimcs.pt.LadderAdaptation._reseed_at_new_betas` for what a stale cache
        costs), so it must not be "simplified" into a closure.
        """
        seed = init_integrator_state(
            self.potentials, coordinate, jnp.zeros_like(coordinate), ctx)
        return seed.potential_values, seed.potential_grads

    def _reseed(self, coordinate, ctx):
        """:meth:`_reseed_caches` under a **cached** ``jax.jit``.

        Built lazily and kept on the instance, so it survives mixin ``__init__`` ordering and is a
        cache hit from the second call on. Eagerly, this is one full ``value_and_grad`` of every
        potential dispatched primitive by primitive --- measured ~275x slower than the compiled
        form, and it runs *once per warmup iteration* whenever a chart or the ladder adapts.
        Building the ``jax.jit`` per call would retrace every time and be slower than the eager
        version it replaced.
        """
        fn = getattr(self, "_reseed_jit", None)
        if fn is None:
            fn = jax.jit(self._reseed_caches)
            self._reseed_jit = fn
        return fn(coordinate, ctx)

    def state_at_coordinate(self, state, coordinate, *, sample=None, hyperparams=None):
        """Rebuild ``state`` at ``coordinate``: refresh the cached potential values/gradients and
        ``log_prob`` there. ``hyperparams`` defaults to the state's charts; ``sample`` defaults to
        the physical point ``coordinate`` maps to under them (pass the existing ``sample`` to
        instead *relabel* a fixed point, as chart adaptation does). Used by initialization (a new
        starting coordinate) and by :class:`CenteringAdaptation` (recharting)."""
        h = state.chart_hyperparams if hyperparams is None else hyperparams
        # Through ``self.context`` rather than the constructor, so a mixin that adds to the
        # context is honoured here too --- parallel tempering carries the (adapted, traced)
        # ladder there, and building the context by hand would silently evaluate the potentials
        # at whatever ladder they were constructed with.
        # No kinetic cache: this path evaluates the potentials only, and it runs eagerly (chart
        # adaptation calls it per warmup iteration), where building one is pure overhead.
        ctx = self.context(state._replace(chart_hyperparams=h), kinetic_cache=False)
        values, grads = self._reseed(coordinate, ctx)
        # ``sample is None`` is a branch on the *caller's* intent, so it stays out here in Python:
        # recomputing the sample unconditionally would let it win over the one chart adaptation
        # deliberately holds fixed, physically moving the chain on every rechart.
        if sample is None:
            sample = self.model.coordinate_to_sample(coordinate, h, state.chart_indices)
        return state._replace(
            coordinate=coordinate,
            sample=sample,
            chart_hyperparams=h,
            potential_values=values,
            potential_grads=grads,
            log_prob=-sum(values.values()))

    # --- kernel ---

    def kernel(self, state: HMCState) -> HMCState:
        ctx = self.context(state)
        # refresh momentum from the kinetic components' distributions
        p0 = self.sample_momentum(state.rng_draw, state.coordinate, ctx)
        # seed the integrator state from the cached gradients at the current coordinate
        istate0 = IntegratorState(
            q=state.coordinate, p=p0,
            potential_values=state.potential_values,
            potential_grads=state.potential_grads,
            log_weight=jnp.zeros(()),
            integrator_data=self.integrator.init_integrator_data())

        chosen, diagnostics = self.build_trajectory_and_select(state, istate0, ctx)

        new_coordinate = chosen.q
        new_sample = self.model.coordinate_to_sample(
            new_coordinate, state.chart_hyperparams, state.chart_indices)
        new_log_prob = -sum(chosen.potential_values.values())
        return state._replace(
            coordinate=new_coordinate,
            sample=new_sample,
            log_prob=new_log_prob,
            momentum=chosen.p,
            potential_values=chosen.potential_values,
            potential_grads=chosen.potential_grads,
            diagnostics=diagnostics,
        )

    def build_trajectory_and_select(self, state, istate0, ctx):
        """Return ``(chosen, diagnostics)`` where ``diagnostics`` matches :meth:`init_diagnostics`'s
        schema (``accept_prob``, ``accepted``, ``grad_evals``, ``mean_refine``, ``proxy_accept_prob``;
        NUTS adds ``diverging``/``tree_depth``/``n_leaves``). Subclass duty. ``proxy_accept_prob`` is
        the line-search step-size-adaptation signal (equals ``accept_prob`` for ordinary integrators),
        while ``accept_prob`` governs the actual accept."""
        raise NotImplementedError

    def _proxy_signal(self, proposed, fallback_accept):
        """From a proposed trajectory's ``integrator_data``, the step-size-adaptation acceptance
        proxy ``min(1, exp(-proxy_energy))`` and mean refinement level. Falls back to the real
        acceptance for integrators that emit no proxy."""
        if not self.integrator.emits_step_size_proxy:
            return fallback_accept, jnp.zeros(())
        d = proposed.integrator_data
        proxy_accept = jnp.where(
            jnp.isfinite(d["proxy_energy"]), jnp.minimum(1.0, jnp.exp(-d["proxy_energy"])), 0.0)
        mean_refine = d["refine_sum"] / jnp.maximum(d["n_steps"], 1.0)
        return proxy_accept, mean_refine

    def _integrate_and_accept(self, state, istate0, ctx, n_steps):
        """Integrate ``n_steps`` leapfrog steps and Metropolis accept/reject.

        ``n_steps`` may be a Python int (fixed-length HMC) or a traced scalar
        (randomized integration time). A divergent trajectory (non-finite energy) is
        treated as acceptance prob 0: this rejects it and, crucially, makes step-size
        adaptation shrink the step.
        """
        H0 = self.total_energy(istate0, ctx)
        proposed = self.integrator.integrate(istate0, state.step_size, n_steps, ctx)
        H1 = self.total_energy(proposed, ctx)

        # general acceptance; for leapfrog log_weight stays 0 -> min(1, exp(-dH))
        log_alpha = H0 - H1 + (proposed.log_weight - istate0.log_weight)
        accept_prob = jnp.where(
            jnp.isfinite(log_alpha), jnp.minimum(1.0, jnp.exp(log_alpha)), 0.0)
        accepted = state.rng_draw.accept_threshold < accept_prob

        chosen = jax.tree.map(lambda a, b: jnp.where(accepted, a, b), proposed, istate0)
        proxy_accept, mean_refine = self._proxy_signal(proposed, accept_prob)
        grad_evals = proposed.integrator_data.get("grad_evals", jnp.zeros(()))
        diagnostics = {"accept_prob": accept_prob, "accepted": accepted, "grad_evals": grad_evals,
                       "mean_refine": mean_refine, "proxy_accept_prob": proxy_accept}
        return chosen, diagnostics

    def mean_refinements(self) -> float:
        """Mean line-search refinement level over the run (all phases, as before; nan if the
        integrator emits none)."""
        if not self.integrator.emits_step_size_proxy:
            return float("nan")
        v = self._diag_values("mean_refine", warmup=True, sampling=True)
        return float(np.mean(v)) if v.size else float("nan")


class HMC(BaseHMC):
    """Fixed-length HMC: integrate ``n_leapfrog`` steps, then Metropolis accept/reject."""

    def __init__(self, *args, n_leapfrog: int = 20, **kwargs):
        self.n_leapfrog = int(n_leapfrog)
        super().__init__(*args, **kwargs)

    def build_trajectory_and_select(self, state, istate0, ctx):
        return self._integrate_and_accept(state, istate0, ctx, self.n_leapfrog)


class RandomizedHMC(BaseHMC):
    """HMC with randomized integration time.

    Each iteration draws the number of leapfrog steps uniformly from
    ``{T, T+1, ..., 2T}`` where ``T = n_leapfrog``. Randomizing the trajectory length
    (independently of the current state) keeps the target invariant --- it is a
    mixture of valid HMC kernels --- while breaking the periodic-orbit resonances that
    can stall fixed-length HMC on well-conditioned targets.

    ``T`` is taken as input; there is no consensus criterion for adapting it.
    """

    def __init__(self, *args, n_leapfrog: int = 20, **kwargs):
        self.n_leapfrog = int(n_leapfrog)   # the lower bound T
        super().__init__(*args, **kwargs)

    def make_draw_components(self, model, **kwargs):
        comps = super().make_draw_components(model, **kwargs)
        comps.append(DrawComponent("traj_uniform", (), jax.random.uniform))
        return comps

    def build_trajectory_and_select(self, state, istate0, ctx):
        T = self.n_leapfrog
        # u in [0, 1) -> integer in {T, ..., 2T} (T+1 equally likely values)
        u = state.rng_draw.traj_uniform
        n_steps = T + jnp.floor(u * (T + 1)).astype(jnp.int32)
        return self._integrate_and_accept(state, istate0, ctx, n_steps)
