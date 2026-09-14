"""Metropolis-within-Gibbs scans over a model's discrete parameters.

Implements the sampler half of ``docs/design/14_discrete_parameters.md``. Four classes:

* :class:`DiscreteMetropolisWithinGibbs` --- the **family**: everything a scan over the discrete
  coordinates needs except the order it visits them in. Not composable on its own.
* :class:`SystematicScanMetropolisWithinGibbs` --- a **kernel-composing mixin** that visits every
  coordinate of every parameter in declaration order after whatever continuous kernel it is
  composed over, so ``make_sampler_class(RobbinsMonroStepSize,
  SystematicScanMetropolisWithinGibbs, NUTS)`` is a NUTS sampler that also moves labels. Composing
  two ``pi``-invariant kernels leaves ``pi`` invariant, which is the whole argument for why this is
  allowed to be so simple.
* :class:`RandomScanMetropolisWithinGibbs` --- its sibling: each jump picks a coordinate uniformly at
  random from all of them. Every jump is reversible, so the kernel is; a fixed order is only
  ``pi``-invariant. It is also where a blocked update will attach, as another thing a jump can pick.
* :class:`StaticContinuous` --- a base algorithm that does nothing to the continuous block, so a
  model that is *only* discrete has something to compose a scan over.

**Each parameter is moved by its own method.** What a scan supplies is the visiting order, the lane
axis, the RNG indexing and the restricted density; *how* a coordinate moves belongs to a
:class:`~mimcs.samplers.discrete_updates.DiscreteUpdate` held per parameter in
:attr:`~DiscreteMetropolisWithinGibbs.discrete_updaters` --- the discrete peer of
``BaseHMC.kinetics`` --- and both scans use the same ones. The family keeps the
Metropolis-within-Gibbs name because that remains the default method and the one described below;
exact conditional Gibbs, the random walk and custom jump operators are the others.

**A new mixin category.** Every other mixin in the library cooperates through the ``_*_hooks``
chain and never touches ``kernel``. This one overrides ``kernel`` and calls ``super().kernel``.
That is not a special case bolted on: ``BaseSampler.__init__`` jits the MRO-resolved bound method
(``jax.jit(self.kernel)``), so the composition compiles as a single function, no base algorithm
needed editing, and the ordering rule is the usual one --- mixins before the base algorithm.

**The proposal.** At each coordinate in turn, propose uniformly among the ``n-1`` values it is
*not* currently at::

    n_i    = upper_i - lower_i + 1
    offset = 1 + floor(u * (n_i - 1))            # uniform on 1 .. n_i-1
    prop   = lower_i + ((cur - lower_i) + offset) mod n_i

which is symmetric --- ``q(a -> b) = q(b -> a) = 1/(n_i - 1)`` --- so acceptance is the plain
ratio ``min(1, pi(prop)/pi(cur))`` with no Hastings term. A binary coordinate always proposes the
flip, which is what one wants and needs no special case. Neither does ``n_i = 1``: the formula
proposes the current value, a no-op that is accepted and counted as no move.

Verified before it was written: detailed balance holds to the resolution of the check
(``max |pi_i K_ij - pi_j K_ji| ~ 1e-6``, the u-grid granularity), while the three controls --- a
missing acceptance test, an inverted ratio, and an asymmetric proposal used without a Hastings
correction --- fail it by 2e-2 to 1.7e-1. The float32 rounding edge that would make
``floor(u*(n-1))`` reach ``n-1`` and collapse the proposal to the current value **does not
occur**: exhaustively, for every ``n`` in 2..200000 and the largest representable ``u < 1`` in
both float32 and float64, the product rounds down. So there is no clamp here, deliberately.

**Cost.** One full log-density evaluation per discrete coordinate per sweep, plus one to seed the
sweep and one gradient to refresh the caches afterwards. Gradient-free, so each is cheaper than a
leapfrog step, but the count is ``discrete_dim`` --- a model with hundreds of labels is
sweep-dominated. Evaluating only the components that actually depend on the coordinate is the
obvious fix and is deferred (doc 14).
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .._logging import get_logger
from ..rng import DrawComponent, zero_draw
from .base import BaseSampler
from .discrete_updates import SweepEnv, build_discrete_updaters, check_jump_balance

log = get_logger(__name__)


# --- component- and coordinate-restricted recomputation (doc 14) --------------------------- #
#
# Module-level rather than methods, because the *factory rule* that picks each parameter's update
# method needs this analysis and has no sampler instance to ask. It is a pure function of the
# model in either case: which components read what is a property of the program, not of the traced
# label being moved.

def restriction_plan(model, pname: str):
    """Which components a move of discrete parameter ``pname`` actually needs --- or ``None``.

    Three groups, decided **statically** from the model:

    * *skipped* --- ``component_reads`` says the component does not read ``pname``, so its
      contribution to the acceptance ratio cancels exactly. A component with no recorded reads
      counts as reading everything, which is why a hand-written model loses nothing and gains
      nothing.
    * *fast* --- a :class:`~mimcs.model.ScanComponent` scanned over ``pname``: moving one label
      perturbs one element, so the difference costs ``O(1)`` instead of ``O(n)``.
    * *slow* --- everything else, evaluated in full at both label settings.

    ``None`` means nothing would be gained, and the caller then runs the original full-density
    sweep **verbatim**. That is what keeps every model without a scan component --- which is
    every model that existed before this --- bit-for-bit unchanged.
    """
    fast, slow, skipped = [], [], []
    for comp in model.log_prob_fns:
        reads = getattr(model, "component_reads", {}).get(comp)
        if reads is not None and pname not in reads:
            skipped.append(comp)
            continue
        sc = getattr(model, "scan_components", {}).get(comp)
        (fast if sc is not None and pname in sc.scanned else slow).append(comp)
    if not fast and not skipped:
        return None
    return fast, slow


def only_in_scan_components(model, pname: str) -> bool:
    """Is every component that reads ``pname`` elementwise in it?

    The condition under which evaluating the density at *all* ``n_i`` values costs ``n_i`` pieces
    of ``O(1)`` element work rather than ``n_i`` whole-density evaluations --- which is what makes
    exact conditional Gibbs affordable on a support far wider than it otherwise would be
    (:data:`EXACT_MAX_VALUES_ELEMENTWISE` against :data:`EXACT_MAX_VALUES`).

    Note this asks :func:`restriction_plan`, **not** :func:`restriction_plans`: the latter
    normalises a ``None`` into an all-``slow`` plan whenever some *other* parameter gains, so
    reading ``slow == []`` off it would report a plain model's parameters as elementwise.

    **A custom jump operator destroys the property**, so a jump parameter answers ``False`` however
    its components are written. The map rewrites whole continuous arrays, so every component
    reading one of them needs a *full* evaluation per candidate --- and the chart Jacobian no
    longer cancels, which is what puts a jump on the full-density path to begin with
    (:meth:`DiscreteMetropolisWithinGibbs._discrete_delta`). Without this clause, adding a
    ``proposal`` block to a working model would silently take a 64-valued scanned parameter from
    64 pieces of ``O(1)`` element work to 64 whole densities per coordinate per sweep.
    """
    if pname in getattr(model, "jump_operators", {}):
        return False
    plan = restriction_plan(model, pname)
    return plan is not None and bool(plan[0]) and not plan[1]


def restriction_plans(model, force: bool = False) -> dict:
    """``{parameter name: plan}`` when restriction is in force, else ``{}``.

    The switch is **per model, not per parameter**, because the two paths carry different state:
    the full path threads the running log density through the loop, while the restricted one
    computes differences and never forms a total. Mixing them inside one sweep would mean carrying
    both.

    ``force`` is how an update method that cannot use a running total --- exact conditional Gibbs
    draws from a softmax of *differences* --- puts the whole sweep on the delta path even on a
    model where no component analysis gains anything.
    """
    plans = {p.name: restriction_plan(model, p.name) for p in model.discrete_parameters}
    if not force and all(v is None for v in plans.values()):
        return {}
    # A parameter that gains nothing still runs through the restricted path, with every component
    # slow. It costs one extra evaluation per coordinate there, which is the price of not carrying
    # two kinds of state; in practice a model with a scan component over its labels has no other
    # component reading them, so `slow` is empty.
    return {k: (v if v is not None else ([], list(model.log_prob_fns)))
            for k, v in plans.items()}


class DiscreteMetropolisWithinGibbs:
    """Metropolis-within-Gibbs over the model's discrete coordinates --- the family of both scans.

    Holds everything but the visiting order: the per-parameter updaters, the labels' starting
    draw, the lane axis, the density and restricted-difference hooks a tempered sampler overrides,
    and the scan's setup and write-back (:meth:`_sweep_setup`, :meth:`_sweep_finish`). A subclass
    supplies :meth:`_discrete_sweep` and the RNG draws it consumes. Composing this class itself
    raises; compose one of its two subclasses, **before** the base algorithm::

        cls = make_sampler_class(RobbinsMonroStepSize, SystematicScanMetropolisWithinGibbs, NUTS)

    Inert on a model with no discrete parameters: it adds no RNG draw components, no diagnostics
    and no work, so composing it defensively costs nothing and changes no numbers. That is not
    only tidiness --- :class:`~mimcs.rng.RNGBuffer` splits its key into one subkey **per draw
    component**, so adding a component renumbers every other component's stream. Adding none is
    what keeps a continuous run bit-identical to one built without this mixin.

    Args:
        discrete_sweeps: how many sweeps' worth of updates to run per iteration (default 1): full
            scans for the systematic scan, and ``discrete_sweeps * n`` jumps for the random one.
            More per continuous update trade log-density evaluations for better-mixed labels; the
            useful setting is problem-dependent, and adaptation of it is deferred.
    """

    handles_discrete = True

    def _init_hooks(self, **kwargs):
        if type(self)._discrete_sweep is DiscreteMetropolisWithinGibbs._discrete_sweep:
            raise TypeError(
                "DiscreteMetropolisWithinGibbs is the family of the discrete scans and has no "
                "visiting order of its own. Compose SystematicScanMetropolisWithinGibbs (every "
                "coordinate in declaration order, the long-standing behaviour) or "
                "RandomScanMetropolisWithinGibbs (a uniformly random coordinate per jump) instead.")
        self._n_discrete_sweeps = int(kwargs.get("discrete_sweeps", 1))
        if self._n_discrete_sweeps < 1:
            raise ValueError(
                f"discrete_sweeps must be >= 1, got {self._n_discrete_sweeps!r}")
        #: one :class:`~mimcs.samplers.discrete_updates.DiscreteUpdate` per discrete parameter, in
        #: declaration order --- the discrete peer of ``BaseHMC.kinetics``. Built from the same
        #: module-level function :class:`~mimcs.adaptation.DiscreteMarginalAdaptation` uses, so
        #: neither mixin depends on which the MRO initialises first.
        self.discrete_updaters = build_discrete_updaters(
            self.model, kwargs.get("discrete_update"))
        #: the random walk's starting log scale ``rho`` (mean step ``1 + e^rho``; 0 is a mean of 2),
        #: read here rather than by the adaptation because the walk needs a scale with or without it.
        #: Either one float for every walk, or ``{name: float | (size,) array}`` --- the factory's
        #: per-coordinate scales from evidence --- where a name left out starts at 0.
        init = kwargs.get("discrete_rw_init_log_scale", 0.0)
        self._rw_init_log_scale = (dict(init) if isinstance(init, dict) else float(init))
        if any(u.kind != "metropolis" for u in self.discrete_updaters):
            log.info("discrete update methods: %s",
                     ", ".join(f"{u.name}={u.kind}("
                               + (f"{u.n_values} values" if u.bounded else "unbounded") + ")"
                               for u in self.discrete_updaters))
        return super()._init_hooks(**kwargs)

    # --- diagnostics ---

    def init_diagnostics(self) -> dict:
        d = super().init_diagnostics()
        if not self.model.discrete_dim:
            return d
        L = self._n_lanes
        shape = () if L == 1 else (L,)
        return {**d,
                "discrete_accept_prob": jnp.zeros(shape),
                "discrete_moves": jnp.zeros(shape, jnp.int32)}

    # --- initialization ---

    def _initialize_hooks(self, state):
        """Start the labels uniformly at random over their support.

        ``Model.default_discrete()`` is the lower bound of every coordinate --- every observation
        in the first cluster, every indicator off. That is a valid point and a bad starting one,
        in the same way that a flat zero vector is a valid but bad continuous start, which is what
        :class:`~mimcs.adaptation.UniformInit` exists to fix.
        """
        state = super()._initialize_hooks(state)
        model = self.model
        if not model.discrete_dim:
            return state
        key = jax.random.PRNGKey(self._seed + 0x0D15C)   # a stream of its own, like UniformInit's
        # The starting windows describe **one** lane, so tile them across the lanes: every rung
        # holds its own copy of the same parameters, and each starts from its own random draw. A
        # window is the whole support for a bounded parameter -- exactly the old draw -- and a small
        # region next to the finite bound (or zero) for an open side.
        L = self._n_lanes
        lower = jnp.tile(model.discrete_init_low, L)
        upper = jnp.tile(model.discrete_init_high, L)
        z = jax.random.randint(key, (model.discrete_dim,), lower, upper + 1).astype(jnp.int32)
        state = state._replace(discrete=z)
        return self._after_discrete(state, self._discrete_log_prob(state, z))

    # --- the sweep ---

    def kernel(self, state):
        """The composed kernel: the continuous algorithm's step, then the discrete sweep."""
        state = super().kernel(state)
        if not self.model.discrete_dim:
            return state
        return self._discrete_sweep(state)

    # --- the lane axis: one lane untempered, one per temperature under PT ---

    @property
    def _n_lanes(self) -> int:
        """How many independent copies of the discrete block the state carries.

        ``1`` for an ordinary model. Under parallel tempering it is the number of rungs: every
        temperature holds its own labels and targets its own ``pi^beta_k``, so the sweep runs at
        each independently (doc 13, doc 14).
        """
        return int(getattr(self.model, "n_temperatures", 1))

    @property
    def _lane_discrete_dim(self) -> int:
        """The width of **one** lane's discrete block."""
        return self.model.discrete_dim // self._n_lanes

    def _discrete_log_prob(self, state, discrete):
        """The target at the current position and the given labels --- ``(L,)``.

        One value per lane. Untempered that is the coordinate-space log-density; a tempered
        sampler overrides this, because a ``ProductModel`` deliberately has no
        ``log_prob_at_coordinate`` (the ladder it would need is adapted, so it travels in the
        Hamiltonian context rather than on the model).
        """
        lp = self.model.log_prob_at_coordinate(
            state.coordinate, state.chart_hyperparams, state.chart_indices, discrete)
        return jnp.reshape(lp, (1,))

    # --- component- and coordinate-restricted recomputation (doc 14) ---

    def _init_state_hooks(self, state):
        """Verify any jump operator's balance conditions, once, on the real initial state.

        **Here and not in** ``_initialize_hooks``: ``initialize()`` is optional, so a check living
        there would silently never run for a user who goes straight to ``warmup()`` --- the same
        class of silence it exists to prevent. This hook runs unconditionally from
        ``BaseSampler.__init__``, eagerly, before the kernel is jitted.
        """
        state = super()._init_state_hooks(state)
        state = self._init_random_walk_entries(state)
        check_jump_balance(self, state)
        return state

    def _init_random_walk_entries(self, state):
        """Give each random-walk parameter its proposal entry: a log scale plus the statistics slots.

        Written here, once, before the kernel is traced, so the pytree structure of
        ``discrete_proposal_params`` is fixed for the run. It *replaces* whatever the base sampler's
        ``make_initial_state`` put under that name --- a uniform table for a bounded ordinal
        parameter, which a random walk never reads, and nothing at all for an open side.
        """
        walks = [u for u in self.discrete_updaters if u.records_accept]
        if not walks:
            return state
        L = self._n_lanes
        params = dict(state.discrete_proposal_params)
        init = self._rw_init_log_scale
        for u in walks:
            lo, hi = u.log_scale_bounds
            rho0 = np.asarray(init.get(u.name, 0.0) if isinstance(init, dict) else init, float)
            if rho0.shape not in ((), (u.size,)) or not np.all(np.isfinite(rho0)):
                raise ValueError(
                    f"discrete_rw_init_log_scale for {u.name!r} must be a finite scalar or have "
                    f"one entry per coordinate, shape ({u.size},); got shape {rho0.shape}")
            # One lane's scales, tiled over the lanes: under tempering every rung starts from the
            # same scale and adapts its own from there.
            rho0 = np.broadcast_to(np.clip(rho0, lo, hi), (L, u.size))
            params[u.name] = {"log_scale": jnp.asarray(rho0, float),
                              "accept_sum": jnp.zeros((L, u.size), float),
                              "n_proposed": jnp.zeros((L, u.size), float)}
        return state._replace(discrete_proposal_params=params)

    def _restriction_plan(self, pname: str):
        """This sampler's model's plan for ``pname`` --- see :func:`restriction_plan`."""
        return restriction_plan(self.model, pname)

    def _restricted(self, force: bool = False) -> dict:
        """This sampler's model's plans --- see :func:`restriction_plans`."""
        return restriction_plans(self.model, force=force)

    def _sweep_context(self, state):
        """Whatever the delta hook needs that does not change during the sweep.

        The continuous half of the value dict is the whole of it here: labels cannot reach a chart
        (``Model._validate_discrete`` forbids a discrete chart parent), so ``from_coordinate`` gives
        the same answer at every coordinate of the sweep. Unpacking it **once per sweep** instead
        of once per coordinate is a saving independent of any component analysis --- and the same
        rule is why the chart Jacobian is absent from the delta entirely: it cannot depend on a
        label, so it cancels.
        """
        return self.model.unpack_coordinate(
            state.coordinate, state.chart_hyperparams, state.chart_indices, None)

    def _discrete_delta(self, state, sweep_ctx, z, pname, index, cur, prop, plan):
        """``log pi(prop) - log pi(cur)`` for one coordinate --- ``(L,)``.

        The restricted counterpart of :meth:`_discrete_log_prob`, and the reason the sweep can stop
        evaluating the whole density per coordinate. A tempered sampler overrides it, for the same
        reason it overrides the density hook.

        Only the *difference* is ever formed: every component that does not read this parameter
        cancels, and is never evaluated at all.

        ``index`` is the coordinate's position **within ``pname``'s own block**, not within the
        model's flat discrete array. The two coincide only for the first discrete parameter, so
        passing the flat index instead is invisible on every single-parameter model and silently
        wrong on the rest: a second parameter would index past the end of its own array, and
        ``.at[i].set`` **clamps** rather than raising. It is also what
        :class:`~mimcs.model.ScanComponent` means by "element ``i`` is coordinate ``i``".
        """
        model = self.model
        fast, slow = plan
        values = {**sweep_ctx, **model.unpack_discrete(z[0])}
        total = jnp.zeros(())
        for comp in fast:
            f = model.scan_components[comp].element_fn
            total = total + (f(values, index, {pname: prop[0]})
                             - f(values, index, {pname: cur[0]}))
        if slow:
            # A component that reads the labels without being elementwise in them needs both
            # settings in full, at this parameter's own flat block (see `index` above).
            arr = values[pname]
            flat = jnp.reshape(arr, (-1,))
            v_cur = {**values, pname: jnp.reshape(flat.at[index].set(cur[0]), arr.shape)}
            v_prop = {**values, pname: jnp.reshape(flat.at[index].set(prop[0]), arr.shape)}
            for comp in slow:
                fn = model.log_prob_fns[comp]
                total = total + (fn(v_prop) - fn(v_cur))
        return jnp.reshape(total, (1,))

    def _exit_log_prob(self, state, z, lp, plans):
        """The log-density to hand :meth:`_after_discrete`, at the state the sweep *ended* in.

        On the full path the carry already holds it. On the delta path there is no running total,
        so it is evaluated once here --- and it must be evaluated against ``state``, which by now
        carries the moved labels **and** any moved coordinate, rather than against the pre-sweep
        state the sweep otherwise closes over.
        """
        if not plans:
            return lp
        return self._discrete_log_prob(state, z.reshape(-1))

    def _after_jump(self, state, coordinate):
        """Write a moved coordinate back into the state, with everything derived from it.

        Only reached when some update method moves the continuous block. ``state.sample`` is the
        thing that makes this more than bookkeeping: it is what :meth:`_retained_sample` stores, so
        a stale one means **every recorded continuous draw is the pre-jump value** --- a wrong
        posterior behind clean-looking traces, and a jump that appears to do nothing. Two
        adaptations also read it and write it straight back
        (:class:`~mimcs.adaptation.CenteringAdaptation` and the unit-vector one), which would weld
        an inconsistent ``(coordinate, sample)`` pair into the state permanently.

        :meth:`_after_discrete` then refreshes the potential caches; it already takes the
        coordinate from the state, so it needs this to have run first and nothing else.
        """
        return state._replace(
            coordinate=coordinate,
            sample=self.model.coordinate_to_sample(
                coordinate, state.chart_hyperparams, state.chart_indices))

    def _discrete_sweep(self, state):
        """One iteration's worth of discrete updates, in every lane --- the visiting order.

        Supplied by each scan; see :class:`SystematicScanMetropolisWithinGibbs` and
        :class:`RandomScanMetropolisWithinGibbs`. Both run between :meth:`_sweep_setup` and
        :meth:`_sweep_finish`.
        """
        raise NotImplementedError

    def _sweep_setup(self, state):
        """What every scan holds constant, and its starting carry --- ``(env, plans,
        moves_coordinate, carry)``.

        The **lane** axis is leading throughout: ``z`` is ``(L, n)``, the density is ``(L,)``, and
        a coordinate step updates the same column in every lane at once. Lanes accept
        **independently**, which is what makes this right under tempering --- each rung is its own
        chain against its own target, exactly as
        :class:`~mimcs.pt.hmc.IndependentAcceptanceMixin` treats the continuous half. With ``L = 1``
        every array simply has a leading axis of one and the arithmetic is unchanged.

        **Which path the sweep takes** is still a per-model choice, because the two carry different
        state: the full path threads a running log density, the delta path never forms a total. A
        method that cannot maintain a total (``forms_running_total = False``, i.e. exact
        conditional Gibbs) therefore forces the whole sweep onto the delta path, where a parameter
        that gains nothing from component analysis runs with every component slow and costs one
        extra evaluation per coordinate. A model whose every parameter is Metropolis-updated *and*
        whose components offer no restriction still runs the original full-density code, which is
        what keeps such a model bit-for-bit unchanged.
        """
        L, n = self._n_lanes, self._lane_discrete_dim
        updaters = self.discrete_updaters
        moves_coordinate = any(u.moves_coordinate for u in updaters)
        force = any(not u.forms_running_total for u in updaters)
        plans = self._restricted(force=force)

        env = SweepEnv(
            sampler=self, state=state,
            # `None` when some method moves the coordinate: the once-per-sweep unpacking is then
            # stale from the first accepted move on, and every updater rebuilds it from the carried
            # coordinate instead. Gating here rather than in the updaters is what keeps a model
            # with no jump on the original path, and so bit-identical.
            sweep_ctx=(None if moves_coordinate
                       else (self._sweep_context(state) if plans else None)),
            plans=plans, tables=state.discrete_proposal_params,
            u_prop=state.rng_draw.discrete_proposal,           # (sweeps * n, L)
            u_acc=state.rng_draw.discrete_accept,
            n_lanes=L, lane_dim=n)

        def logp(z):
            return self._discrete_log_prob(state, z.reshape(-1))

        z0 = state.discrete.reshape(L, n)
        # The coordinate rides in the carry so a method that moves it (a custom jump operator) has
        # somewhere to put it. It costs nothing when nothing moves it --- an untouched array
        # threaded through a `fori_loop` is not copied --- and carrying it unconditionally is what
        # keeps one signature for every method rather than two.
        x0 = state.coordinate.reshape(L, -1)
        # The full path seeds the running density; the delta path has no use for it and pays one
        # evaluation at the exit instead of one here plus `n` inside the loop.
        #
        # The last slot is per-coordinate acceptance statistics for the methods that keep them (a
        # random walk), reset every kernel call. It is `{}` for every other model, which leaves the
        # arithmetic -- and so the draws -- exactly as they were.
        stats0 = {u.name: (jnp.zeros((L, u.size)), jnp.zeros((L, u.size)))
                  for u in updaters if u.records_accept}
        carry = (z0, x0, jnp.zeros((L,)) if plans else logp(z0),
                 jnp.zeros((L,)), jnp.zeros((L,), jnp.int32), stats0)
        return env, plans, moves_coordinate, carry

    def _sweep_finish(self, state, carry, plans, moves_coordinate, n_steps: int):
        """Write a finished scan back into the state: the labels, the random-walk statistics, a
        moved coordinate, the refreshed caches and the diagnostics.

        ``n_steps`` is how many proposals the scan made in all, the denominator of
        ``discrete_accept_prob``: ``sweeps * n`` for the systematic scan, the jump count for the
        random one.
        """
        L = self._n_lanes
        z, x, lp, alpha_sum, moved, stats = carry
        state = state._replace(discrete=z.reshape(-1))
        if stats:
            params = dict(state.discrete_proposal_params)
            for name, (acc, n_prop) in stats.items():
                params[name] = {**params[name], "accept_sum": acc, "n_proposed": n_prop}
            state = state._replace(discrete_proposal_params=params)
        if moves_coordinate:
            state = self._after_jump(state, x.reshape(-1))
        state = self._after_discrete(state, self._exit_log_prob(state, z, lp, plans))
        # One lane means an ordinary sampler, whose diagnostics are scalars; L > 1 keeps the lane
        # axis, matching how a tempered run reports its acceptance per rung.
        squeeze = (lambda x: x[0]) if L == 1 else (lambda x: x)
        return state._replace(diagnostics={
            # Merged into the dict the base kernel returned, not into `init_diagnostics()`: a
            # kernel *replaces* the diagnostics dict, so anything not added here is never recorded.
            **state.diagnostics,
            # Note an exact-Gibbs coordinate contributes 1.0 here (its acceptance probability is 1
            # by construction), so this column reads 1.00 for an all-exact model and
            # `discrete_moves` is then the only one that can catch a frozen label.
            "discrete_accept_prob": squeeze(alpha_sum / n_steps),
            "discrete_moves": squeeze(moved),
        })


class SystematicScanMetropolisWithinGibbs(DiscreteMetropolisWithinGibbs):
    """Systematic-scan Metropolis-within-Gibbs: every coordinate of every parameter, in order.

    The library's long-standing scan, and what the factory composes. Mix in **before** the base
    algorithm::

        cls = make_sampler_class(RobbinsMonroStepSize, SystematicScanMetropolisWithinGibbs, NUTS)

    ``pi``-invariant, not reversible: each coordinate update is reversible, a fixed order of them is
    not. :class:`RandomScanMetropolisWithinGibbs` is the reversible sibling.
    """

    def make_draw_components(self, model, **kwargs):
        components = super().make_draw_components(model, **kwargs)
        n = model.discrete_dim
        if n == 0:
            return components          # stream-neutral: see the family's docstring
        sweeps = int(kwargs.get("discrete_sweeps", 1))
        # (sweeps * one lane's width, lanes) -- the trailing-lane convention `pt/nuts.py` uses for
        # its `(J, K)` tree draws, so `u[t]` is the `(L,)` vector one coordinate step needs. For
        # L = 1 this is a reshape of the flat request and threefry fills it identically, so an
        # untempered run's stream is unmoved (checked, not assumed).
        L = int(getattr(model, "n_temperatures", 1))
        shape = (sweeps * (n // L), L)
        return components + [
            DrawComponent("discrete_proposal", shape, generator=jax.random.uniform),
            DrawComponent("discrete_accept", shape, generator=jax.random.uniform),
        ]

    def _discrete_sweep(self, state):
        """One pass (or ``discrete_sweeps`` passes) over every discrete coordinate, in every lane.

        Structured as a **Python loop over the updaters** wrapping a ``fori_loop`` over each
        parameter's own coordinates, rather than one flat loop over the block. Two reasons, and
        both need the parameter to be statically known: its ``n_i`` is then a Python int, so every
        candidate axis is statically sized --- no padding to a global maximum and no masking ---
        and its *update method* is a Python object, so dispatching on it costs nothing at run time.
        The updater loop is static and, in practice, one iteration long.

        Coordinate ``c`` of the parameter starting at ``start`` in sweep ``s`` reads draw row
        ``s * n + start + c``: the global sweep step, so the draw order is the flat sweep's.
        """
        env, plans, moves_coordinate, carry = self._sweep_setup(state)
        n = self._lane_discrete_dim
        updaters = self.discrete_updaters

        def sweep(s_idx, outer):
            """One full pass over every parameter, in declaration order."""
            for u in updaters:
                prep = u.prepare(env)
                outer = jax.lax.fori_loop(
                    0, u.size,
                    lambda c, carry, _u=u, _p=prep: _u.step(
                        env, _p, s_idx * n + (_u.start + c), c, carry),
                    outer)
            return outer

        carry = jax.lax.fori_loop(0, self._n_discrete_sweeps, sweep, carry)
        return self._sweep_finish(state, carry, plans, moves_coordinate,
                                  self._n_discrete_sweeps * n)


class RandomScanMetropolisWithinGibbs(DiscreteMetropolisWithinGibbs):
    """Random-scan Metropolis-within-Gibbs: each jump updates one uniformly chosen coordinate.

    Mix in **before** the base algorithm, exactly where the systematic scan goes::

        cls = make_sampler_class(RobbinsMonroStepSize, RandomScanMetropolisWithinGibbs, NUTS)

    The coordinate is drawn uniformly from **all** discrete coordinates, so a parameter is chosen in
    proportion to its size, and then moved by that parameter's own update method --- the same
    :class:`~mimcs.samplers.discrete_updates.DiscreteUpdate` objects, and so the same proposals and
    adaptations, as the systematic scan. The choice does not depend on the state, so each jump is a
    reversible ``pi``-invariant kernel and so is their product over an iteration's i.i.d. choices.
    This is the base a blocked update attaches to: another unit a jump can pick.

    **The coordinate is shared across lanes.** Under tempering every rung updates the same column
    in a jump. That is valid for the same reason: the choice is independent of every rung's state,
    so each rung is still an exact random-scan chain --- and it keeps "update column ``i`` in every
    lane" the updaters' contract.

    **Dispatch** is a ``lax.switch`` over the parameters' update methods, one branch per parameter,
    each still statically sized by its own support; with one discrete parameter there is no switch.

    Args:
        discrete_jumps: jumps per iteration. Default ``discrete_sweeps * n`` for ``n`` coordinates
            (one lane's), so one sweep's worth --- as many jumps as coordinates --- and the same
            density-evaluation budget as the systematic scan. A random scan leaves a fraction of
            about ``e^-1`` of the coordinates unvisited in an iteration of ``n`` jumps.
    """

    @staticmethod
    def _jumps_for(model, kwargs) -> int:
        """Jumps per iteration, from the model and the kwargs alone --- ``make_draw_components``
        needs the number before any instance state exists."""
        L = int(getattr(model, "n_temperatures", 1))
        n = int(model.discrete_dim) // L
        jumps = kwargs.get("discrete_jumps")
        if jumps is None:
            return int(kwargs.get("discrete_sweeps", 1)) * n
        if int(jumps) != jumps or int(jumps) < 1:
            raise ValueError(f"discrete_jumps must be a positive integer, got {jumps!r}")
        return int(jumps)

    def _init_hooks(self, **kwargs):
        self._n_discrete_jumps = self._jumps_for(self.model, kwargs)
        return super()._init_hooks(**kwargs)

    def make_draw_components(self, model, **kwargs):
        components = super().make_draw_components(model, **kwargs)
        if model.discrete_dim == 0:
            return components          # stream-neutral: see the family's docstring
        J = self._jumps_for(model, kwargs)
        L = int(getattr(model, "n_temperatures", 1))
        return components + [
            # Same names and trailing-lane layout as the systematic scan's, so every updater reads
            # row `t` of them unchanged; here `t` is the jump number.
            DrawComponent("discrete_proposal", (J, L), generator=jax.random.uniform),
            DrawComponent("discrete_accept", (J, L), generator=jax.random.uniform),
            # One per jump, **not** per lane: see the class docstring.
            DrawComponent("discrete_index", (J,), generator=jax.random.uniform),
        ]

    def _discrete_sweep(self, state):
        """``discrete_jumps`` jumps, each at a uniformly drawn coordinate, in every lane."""
        env, plans, moves_coordinate, carry = self._sweep_setup(state)
        n = self._lane_discrete_dim
        J = self._n_discrete_jumps
        updaters = self.discrete_updaters
        # Hoisted out of the jump loop, as the systematic scan hoists them out of each parameter's.
        preps = [u.prepare(env) for u in updaters]
        starts = jnp.asarray([u.start for u in updaters], jnp.int32)
        u_index = state.rng_draw.discrete_index                  # (J,)
        branches = [
            (lambda op, _u=u, _p=p: _u.step(env, _p, op[0], op[1], op[2]))
            for u, p in zip(updaters, preps)]

        def jump(j, carry):
            # `min` guards the float rounding of `u * n` up to `n` at the top of the unit interval.
            g = jnp.minimum(jnp.floor(u_index[j] * n).astype(jnp.int32), n - 1)
            if len(updaters) == 1:
                return branches[0]((j, g, carry))
            k = jnp.searchsorted(starts, g, side="right") - 1
            return jax.lax.switch(k, branches, (j, g - starts[k], carry))

        carry = jax.lax.fori_loop(0, J, jump, carry)
        return self._sweep_finish(state, carry, plans, moves_coordinate, J)


class StaticState(NamedTuple):
    """State for :class:`StaticContinuous`: a position that does not move, and its labels."""

    coordinate: Array
    sample: Array
    discrete: Array
    discrete_proposal_params: dict
    log_prob: Array
    rng_draw: Any
    chart_hyperparams: tuple
    chart_indices: tuple
    diagnostics: dict = {}


class StaticContinuous(BaseSampler):
    """A base algorithm that leaves the continuous coordinates exactly where they are.

    Composed under either scan of :class:`DiscreteMetropolisWithinGibbs` it gives a **discrete-only** sampler,
    which is what a model with no continuous parameters needs --- and what makes the sweep
    testable against an exactly enumerable target, since with the continuous block frozen the
    chain's stationary distribution is a pmf one can write down and compare against.

    It is not a general-purpose "hold these parameters fixed" facility: it freezes *everything*
    continuous, and a model with continuous parameters composed under it will simply never move
    them.
    """

    state_class = StaticState

    def _init_hooks(self, **kwargs):
        # A jump operator moves the continuous block, which is exactly what this class promises not
        # to do. Silently ignoring it would leave the map inert and the chain sampling the *wrong*
        # target -- a jump-aware acceptance ratio against a frozen coordinate -- so it raises.
        jumps = getattr(self.model, "jump_operators", {})
        if jumps:
            raise TypeError(
                f"{type(self).__name__} freezes every continuous parameter, but this model's "
                f"jump operator(s) for {sorted(jumps)} move continuous parameters alongside the "
                f"label. Compose a sampler that moves the continuous block: "
                f"make_sampler_class(..., SystematicScanMetropolisWithinGibbs, NUTS).")
        return super()._init_hooks(**kwargs)

    def make_draw_components(self, model, **kwargs):
        return []

    def make_initial_state(self, init_position) -> StaticState:
        from .metropolis import (_as_discrete_flat, _as_sample_flat,
                                 uniform_discrete_proposal_params)
        model = self.model
        h = model.init_chart_hyperparams()
        c = model.init_chart_indices()
        sample = _as_sample_flat(model, init_position)
        discrete = _as_discrete_flat(model, init_position)
        coordinate = model.sample_to_coordinate(sample, h, c)
        return StaticState(
            coordinate=coordinate,
            sample=sample,
            discrete=discrete,
            discrete_proposal_params=uniform_discrete_proposal_params(model),
            log_prob=model.log_prob_at_coordinate(coordinate, h, c, discrete),
            rng_draw=zero_draw(self._rng_draw_class, self._draw_components),
            chart_hyperparams=h,
            chart_indices=c,
            diagnostics=self.init_diagnostics(),
        )

    def kernel(self, state: StaticState) -> StaticState:
        return state._replace(diagnostics=dict(state.diagnostics))
