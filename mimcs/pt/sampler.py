"""``parallel_tempering(model, ...)``: the assembled sampler.

Implements the architecture of ``docs/design/13_parallel_tempering.md``. A PT sampler is an
ordinary sampler (NUTS by default) pointed at product-space components, plus a swap move. The
class is composed the same way every other sampler here is --- mixins over a base algorithm
(doc 02) --- so the adaptation mixins stack on top unchanged.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .._logging import get_logger
from ..rng import DrawComponent
from ..samplers.base import make_sampler_class
from ..samplers.gibbs import DiscreteMetropolisWithinGibbs
from ..hmc.nuts import BaseNUTS, NUTS, DEFAULT_DIVERGENCE_THRESHOLD
from ..hmc.simple_nuts import SimpleNUTS
from ..hmc.line_search import LineSearchIntegrator, MarkovianLineSearchIntegrator
from ..hmc.samplers import make_kinetic
from ..hmc.state import HamiltonianContext
from ..hmc.integrators import init_integrator_state
from .adaptation import PerTemperatureAdaptation
from .hmc import IndependentAcceptanceMixin
from .nuts import PerTemperatureNUTSMixin, PerTemperatureSimpleNUTSMixin
from .kinetics import build_product_kinetics
from .ladder import LadderAdaptation, BETAS_KEY
from .product import ProductSpaceMixin
from .lanes import per_temperature_potential
from .swaps import swap_step, all_pair_log_ratios
from .tempering import ProductModel, build_tempered_potentials, geometric_ladder

log = get_logger(__name__)


class ReplicaExchangeMixin:
    """Adds an adjacent-pair replica-exchange sweep after each product-space transition.

    The sweep's parity (even or odd pairs) is drawn at random each step rather than alternated
    from a counter: which disjoint set of pairs to attempt is a choice made independently of the
    state, so randomizing it is valid and saves carrying a parity through the state.
    """

    def _init_hooks(self, **kwargs):
        self.betas = jnp.asarray(kwargs["betas"], float)
        self.n_temperatures = int(self.betas.shape[0])
        #: the potentials beta actually scales --- what the swap ratio is built from.
        self._tempered_potentials = [p for p in self.potentials if p.tempered]
        super()._init_hooks(**kwargs)
        log.info("parallel tempering: %d temperature(s), betas %s; tempered component(s) %s",
                 self.n_temperatures,
                 np.array2string(np.asarray(self.betas), precision=4),
                 [p.id for p in self._tempered_potentials] or "(none)")

    def make_draw_components(self, model, **kwargs):
        comps = super().make_draw_components(model, **kwargs)
        K = model.n_temperatures
        comps.append(DrawComponent("swap_uniform", (K,), jax.random.uniform))
        comps.append(DrawComponent("swap_parity", (), jax.random.uniform))
        return comps

    def init_diagnostics(self) -> dict:
        K = self.model.n_temperatures
        return {**super().init_diagnostics(),
                "swap_accepted": jnp.zeros((K,), bool),
                "swap_attempted": jnp.zeros((K,), bool),
                "swap_accept_prob": jnp.zeros((K,))}

    def make_initial_state(self, init_position):
        """Seed the ladder into ``ham_params``, so it is traced rather than closed over."""
        state = super().make_initial_state(init_position)
        return state._replace(ham_params={**state.ham_params, BETAS_KEY: self.betas})

    def context(self, state, **kwargs):
        """The usual context, plus the (traced) ladder the tempered potentials scale by."""
        ctx = super().context(state, **kwargs)
        return ctx._replace(betas=state.ham_params[BETAS_KEY])

    def state_betas(self, state) -> Array:
        return state.ham_params[BETAS_KEY]

    # --- the swap move ---

    def _untempered_log_density(self, coordinate: Array, ctx) -> Array:
        """``L(x_k)`` per temperature: the log-density of the components beta scales, unscaled."""
        total = jnp.zeros((self.n_temperatures,))
        for p in self._tempered_potentials:
            total = total + p.untempered_values(coordinate, ctx)
        return -total                                   # V = -log pi

    def _discrete_log_prob(self, state, discrete):
        """The **per-rung** tempered log-density at the current position and the given labels.

        The tempered override of the Gibbs sweep's density hook. It cannot go through
        ``model.log_prob_at_coordinate``, because a :class:`~mimcs.pt.ProductModel` deliberately
        has none: over the product space the sampled target is the beta-weighted sum the tempered
        potentials evaluate, and the ladder they weight by is *adapted*, so it travels in the
        Hamiltonian context rather than on the model.

        Returns ``(K,)`` --- one log-density per rung, each at that rung's own labels and its own
        beta. That is what makes the sweep's per-lane acceptance a proper Metropolis test against
        each rung's own target.
        """
        ctx = self.context(state, kinetic_cache=False)._replace(discrete=discrete)
        return -per_temperature_potential(
            self.potentials, state.coordinate, ctx, self.n_temperatures)

    def _swap(self, state):
        ctx = self.context(state, kinetic_cache=False)   # tempered potentials only
        betas = self.state_betas(state)
        # Computed *before* the permutation, which is what the ratio wants: the density of each
        # rung's current state, its labels included.
        untempered = self._untempered_log_density(state.coordinate, ctx)
        offset = (state.rng_draw.swap_parity < 0.5).astype(jnp.int32)
        new_coord, new_discrete, accepted, attempted = swap_step(
            state.coordinate, untempered, betas, state.rng_draw.swap_uniform, offset,
            self.n_temperatures, self.model.base.coord_dim,
            discrete=state.discrete, discrete_dim=self.model.base.discrete_dim)
        # Every pair's acceptance probability, including the half this sweep does not attempt:
        # the ladder adaptation wants a signal for each gap on each iteration (see `ladder`).
        all_alpha = jnp.exp(jnp.minimum(0.0, all_pair_log_ratios(untempered, betas)))

        # A swapped replica's cached values/gradients were computed at the other temperature's
        # beta, so they no longer describe it: re-seed them. One vmapped evaluation, negligible
        # beside the trajectory that precedes it (doc 13).
        #
        # Against the **post-swap** labels: `ctx` was built from the pre-swap state, so re-seeding
        # with it would leave every rung's cached gradient describing labels it no longer holds --
        # right shapes, plausible numbers, wrong density on the very next trajectory.
        seed_ctx = ctx if new_discrete is state.discrete else ctx._replace(discrete=new_discrete)
        seed = init_integrator_state(
            self.potentials, new_coord, jnp.zeros_like(new_coord), seed_ctx)
        new_sample = self.model.coordinate_to_sample(
            new_coord, state.chart_hyperparams, state.chart_indices)
        # `discrete_proposal_params` is deliberately NOT permuted: a proposal table describes a
        # *temperature*, not a state. After a swap rung k holds a different replica but still
        # targets pi^beta_k, so its table still approximates the right marginal. Swapping them
        # would be quiet and wrong.
        return state._replace(
            coordinate=new_coord,
            sample=new_sample,
            discrete=new_discrete,
            log_prob=-sum(seed.potential_values.values()),
            potential_values=seed.potential_values,
            potential_grads=seed.potential_grads,
            diagnostics={**state.diagnostics, "swap_accepted": accepted,
                         "swap_attempted": attempted, "swap_accept_prob": all_alpha})

    def kernel(self, state):
        return self._swap(super().kernel(state))

    # --- diagnostics ---

    def swap_rates(self, *, include_warmup: bool = False,
                   include_sampling: bool = True) -> np.ndarray:
        """Acceptance rate of each adjacent pair, ``(K-1,)`` indexed by the lower rung.

        The quantity any ladder tuning is built on: a ladder whose adjacent rates are near zero is
        a PT run doing nothing, and one whose rates are near one has rungs to spare.
        """
        acc = self._diag_values("swap_accepted", warmup=include_warmup,
                                sampling=include_sampling)
        att = self._diag_values("swap_attempted", warmup=include_warmup,
                                sampling=include_sampling)
        if len(acc) == 0:
            return np.full((self.n_temperatures - 1,), np.nan)
        acc, att = np.asarray(acc), np.asarray(att)
        with np.errstate(invalid="ignore", divide="ignore"):
            rate = acc.sum(axis=0) / np.maximum(att.sum(axis=0), 1)
        return rate[:self.n_temperatures - 1]

    def _sample_end_hooks(self, state):
        state = super()._sample_end_hooks(state)
        rates = self.swap_rates()
        if rates.size == 0:                     # a single rung has no pairs to exchange
            return state
        log.info("parallel tempering: adjacent swap acceptance %s",
                 np.array2string(rates, precision=3))
        if np.nanmin(rates) < 0.01:
            log.warning(
                "parallel tempering: adjacent swap rate %s is near zero --- the ladder is "
                "effectively disconnected there, so the hot chains are not helping the cold one; "
                "add temperatures or raise beta_min", np.array2string(rates, precision=3))
        return state


#: Integrators whose refinement level is chosen from the **summed** product Hamiltonian, so a
#: lane's realized step size depends on the other lanes' positions. That coupling is exactly what
#: ``selection="independent"`` exists to remove (see :mod:`mimcs.pt.nuts`), and it would silently
#: invalidate the construction rather than fail loudly, so it is refused rather than warned about.
_COUPLED_INTEGRATORS = (LineSearchIntegrator, MarkovianLineSearchIntegrator)


def _selection_mixins(base, selection: str, integrator):
    """The acceptance/selection mixin(s) for this base, ``selection`` setting and integrator.

    ``"auto"`` prefers per-temperature selection for NUTS -- it is a clean win on every target
    measured (doc 13) -- but falls back to joint whenever a **coupled** integrator is in use, since
    the two are incompatible and joint is the one that works with a line search. An *explicit*
    ``"independent"`` with such an integrator raises instead of silently downgrading.
    """
    if selection not in ("auto", "joint", "independent"):
        raise ValueError(
            f"selection must be 'auto', 'joint' or 'independent'; got {selection!r}")
    if selection == "joint":
        return ()
    if not issubclass(base, BaseNUTS):
        # a fixed-trajectory base accepts independently at every temperature, always
        return (IndependentAcceptanceMixin,)
    coupled = isinstance(integrator, _COUPLED_INTEGRATORS)
    if coupled:
        if selection == "independent":
            _reject_coupled_integrator(integrator)
        log.info("parallel tempering: selection='auto' falls back to joint selection because "
                 "%s couples the temperatures (see mimcs.pt.nuts)", type(integrator).__name__)
        return ()
    return (PerTemperatureSimpleNUTSMixin if issubclass(base, SimpleNUTS)
            else PerTemperatureNUTSMixin,)


def _reject_coupled_integrator(integrator) -> None:
    """Refuse an integrator whose step couples the lanes (checked on the built instance)."""
    raise ValueError(
        f"selection='independent' cannot use {type(integrator).__name__}: a line search picks "
        f"one refinement level from the summed product Hamiltonian, so each lane's realized "
        f"step size depends on the other lanes' positions and its trajectory is no longer an "
        f"ordinary NUTS orbit for pi^beta_k --- which is the whole basis of per-lane "
        f"selection (see mimcs.pt.nuts). Use the default leapfrog, or selection='joint'.")


def parallel_tempering(model, init_position=None, *, n_temperatures: int = 4, betas=None,
                       beta_min: float = 0.01, tempered=None, base=NUTS, kinetics=None,
                       metric: str = "diagonal", step_size=0.5, seed: int = 0,
                       adapt_ladder: bool = True, selection: str = "auto",
                       per_temperature_step_size: bool = False,
                       adapt_mixins=(), extra_mixins=(),
                       integrator=None, **kwargs):
    """Build a parallel tempering sampler over ``model``.

    Args:
        n_temperatures / betas / beta_min: the ladder. Give ``betas`` explicitly, or a count and a
            minimum for a geometric ladder. ``beta_1`` is always 1 --- that chain is the target.
        tempered: names of the model's log-density components that ``beta`` scales; ``None``
            (default) means all of them. Naming a subset gives a power posterior.
        base: the base sampler class the product-space chain runs (default :class:`~mimcs.hmc.NUTS`).
        kinetics: the model's *own* block structure, one entry per coordinate block, with slices
            relative to a single temperature. Each is applied at every temperature with its own
            adapted parameters. Defaults to one whole-space block of ``metric``.
        selection: how the temperatures pick their next point --- ``"independent"`` (each rung
            builds its own trajectory and selects from it; doc 13), ``"joint"`` (one trajectory in
            the product space, one shared leaf index), or ``"auto"`` (the default): independent
            for a NUTS base, falling back to joint when the integrator couples the temperatures,
            which a line search does. An explicit ``"independent"`` with such an integrator raises
            rather than silently downgrading. A fixed-trajectory base (HMC, RWMH) always accepts
            independently, whatever this says.
        per_temperature_step_size: adapt one step size per rung instead of one global. Needs a
            per-rung acceptance signal, so it requires independent selection/acceptance.
            **Off by default and worth leaving off**: it is 1.5-2x on uniform geometry but
            inflates the step until a funnel's neck cannot be integrated, which shows up as
            divergences and an under-dispersed marginal rather than as a worse ESS (doc 13).
        adapt_mixins: adaptation mixins run **per temperature** on that temperature's own
            slice --- the mass adaptations belong here (see :mod:`mimcs.pt.adaptation`).
        extra_mixins: mixins composed onto the product chain itself, for quantities that are
            genuinely global. The step size is one --- per-temperature *width* is absorbed by the
            per-temperature mass, so one step suits them all (see ``per_temperature_step_size``
            for the opt-out and why it is an opt-out).
        integrator: builder ``(potentials, kinetics, K) -> integrator`` over the *product*
            components, which is why it is a callable rather than an instance --- those components
            are built here. Defaults to product leapfrog;
            :func:`~mimcs.pt.integrators.product_line_search` gives WALNUTS over the product space.

    Returns a sampler whose ``get_samples()`` gives the **cold chain's** draws, so it plugs into
    ``summary()`` and the evaluation harness like any other run; ``get_samples_all()`` gives every
    temperature's.
    """
    betas = geometric_ladder(n_temperatures, beta_min) if betas is None else jnp.asarray(
        betas, float)
    K = int(betas.shape[0])
    if float(betas[0]) != 1.0:
        raise ValueError(
            f"the first rung of the ladder must be beta = 1 (the target); got {float(betas[0])}")
    if K > 1 and not bool(jnp.all(jnp.diff(betas) < 0)):
        raise ValueError(f"betas must be strictly decreasing, got {np.asarray(betas)}")

    pmodel = ProductModel(model, K)
    potentials = build_tempered_potentials(model, betas, tempered=tempered)
    inner = list(kinetics) if kinetics is not None else [make_kinetic(metric)]
    pkinetics = build_product_kinetics(inner, K, model.coord_dim)

    if init_position is None:
        init = np.tile(np.asarray(model.default_sample(), float), K)
    else:
        init = np.tile(np.asarray(init_position, float).reshape(-1), K)

    if integrator is not None:
        kwargs["integrator"] = integrator(potentials, pkinetics, K)
    independent = _selection_mixins(base, selection, kwargs.get("integrator"))
    per_lane_nuts = any(issubclass(m, PerTemperatureNUTSMixin) for m in independent)
    if per_temperature_step_size and not (independent or per_lane_nuts):
        raise ValueError(
            "per_temperature_step_size needs a per-rung acceptance signal, which only "
            "independent acceptance (HMC/RWMH) or selection='independent' (NUTS) provides; "
            "joint selection has one acceptance for the whole product chain (doc 13)")
    if per_lane_nuts:
        kwargs["per_temperature_step_size"] = per_temperature_step_size
    vector_step = bool(independent) and (not per_lane_nuts or per_temperature_step_size)
    if vector_step and jnp.ndim(step_size) == 0:
        # A per-rung acceptance signal drives a per-rung step, so the state must carry the vector
        # from the start (its shape is fixed once the kernel is traced).
        step_size = jnp.full((K,), float(step_size))
    # A model with integer parameters gets the Gibbs sweep. Two things fix where it goes.
    # It must sit **inside** the replica exchange, so each rung sweeps its own labels at its own
    # temperature and the swap then moves whole replicas, labels included. And it must sit
    # **before** the selection mixins, because `PerTemperatureNUTSMixin.make_draw_components`
    # deliberately terminates the cooperative chain rather than calling `super()` -- anything to
    # its right never gets asked for its draws, and the sweep would find no uniforms to consume.
    gibbs = (DiscreteMetropolisWithinGibbs,) if getattr(model, "discrete_dim", 0) else ()
    Cls = make_sampler_class(*extra_mixins, LadderAdaptation, ReplicaExchangeMixin,
                             PerTemperatureAdaptation,
                             *gibbs, *independent, ProductSpaceMixin, base,
                             name=f"ParallelTempering{base.__name__}")
    if issubclass(base, BaseNUTS):
        # The joint test is ``max(H) - min(H)`` over the product Hamiltonian, a sum of K terms, so
        # a threshold calibrated for one chain flags K-fold ranges that are perfectly ordinary
        # (measured on the funnel: scaling by K cut median divergences 488 -> 304 with the neck
        # depth unchanged, i.e. those were false positives). Per-lane selection instead tests
        # ``max_k (h_max_k - h_min_k)``, one Hamiltonian's range, so the scaling must come off.
        kwargs.setdefault("divergence_threshold",
                          DEFAULT_DIVERGENCE_THRESHOLD * (1 if per_lane_nuts else K))
    return Cls(pmodel, init, potentials=potentials, kinetics=pkinetics, betas=betas,
               adapt_ladder=adapt_ladder, adapt_mixins=tuple(adapt_mixins),
               step_size=step_size, seed=seed, **kwargs)
