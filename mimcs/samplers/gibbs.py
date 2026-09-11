"""A systematic-scan Gibbs sweep over a model's discrete parameters.

Implements the sampler half of ``docs/design/14_discrete_parameters.md``. Two classes:

* :class:`DiscreteMetropolisWithinGibbs` --- a **kernel-composing mixin**. It sweeps the discrete
  coordinates after whatever continuous kernel it is composed over, so
  ``make_sampler_class(RobbinsMonroStepSize, DiscreteMetropolisWithinGibbs, NUTS)`` is a NUTS
  sampler that also moves labels. Composing two ``pi``-invariant kernels leaves ``pi`` invariant,
  which is the whole argument for why this is allowed to be so simple.
* :class:`StaticContinuous` --- a base algorithm that does nothing to the continuous block, so a
  model that is *only* discrete has something to compose the mixin over.

**Each parameter is moved by its own method.** What the sweep supplies is the scan, the lane axis,
the RNG indexing and the restricted density; *how* a coordinate moves belongs to a
:class:`~mimcs.samplers.discrete_updates.DiscreteUpdate` held per parameter in
:attr:`~DiscreteMetropolisWithinGibbs.discrete_updaters` --- the discrete peer of
``BaseHMC.kinetics``. The class keeps its name because Metropolis-within-Gibbs remains the default
method and the one described below; exact conditional Gibbs is the other, and a custom jump
operator will be the third.

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
from jax import Array

from .._logging import get_logger
from ..rng import DrawComponent, zero_draw
from .base import BaseSampler
from .discrete_updates import SweepEnv, build_discrete_updaters

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
    """
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
    """Deterministic-scan Metropolis-within-Gibbs over the model's discrete coordinates.

    Mix in **before** the base algorithm::

        cls = make_sampler_class(RobbinsMonroStepSize, DiscreteMetropolisWithinGibbs, NUTS)

    Inert on a model with no discrete parameters: it adds no RNG draw components, no diagnostics
    and no work, so composing it defensively costs nothing and changes no numbers. That is not
    only tidiness --- :class:`~mimcs.rng.RNGBuffer` splits its key into one subkey **per draw
    component**, so adding a component renumbers every other component's stream. Adding none is
    what keeps a continuous run bit-identical to one built without this mixin.

    Args:
        discrete_sweeps: how many full scans of the discrete coordinates to run per iteration
            (default 1). More sweeps per continuous update trade log-density evaluations for
            better-mixed labels; the useful setting is problem-dependent, and adaptation of it is
            deferred.
    """

    handles_discrete = True

    def _init_hooks(self, **kwargs):
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
        if any(u.kind != "metropolis" for u in self.discrete_updaters):
            log.info("discrete update methods: %s",
                     ", ".join(f"{u.name}={u.kind}({u.n_values} values)"
                               for u in self.discrete_updaters))
        return super()._init_hooks(**kwargs)

    # --- RNG ---

    def make_draw_components(self, model, **kwargs):
        components = super().make_draw_components(model, **kwargs)
        n = model.discrete_dim
        if n == 0:
            return components          # stream-neutral: see the class docstring
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
        # `discrete_lower/upper` describe **one** lane, so tile them across the lanes: every rung
        # holds its own copy of the same parameters, and each starts from its own random draw.
        L = self._n_lanes
        lower = jnp.tile(model.discrete_lower, L)
        upper = jnp.tile(model.discrete_upper, L)
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
            # settings in full --- `index` is within this parameter, so it indexes its flat block.
            arr = values[pname]
            flat = jnp.reshape(arr, (-1,))
            v_cur = {**values, pname: jnp.reshape(flat.at[index].set(cur[0]), arr.shape)}
            v_prop = {**values, pname: jnp.reshape(flat.at[index].set(prop[0]), arr.shape)}
            for comp in slow:
                fn = model.log_prob_fns[comp]
                total = total + (fn(v_prop) - fn(v_cur))
        return jnp.reshape(total, (1,))

    def _discrete_sweep(self, state):
        """One pass (or ``discrete_sweeps`` passes) over every discrete coordinate, in every lane.

        Structured as a **Python loop over the updaters** wrapping a ``fori_loop`` over each
        parameter's own coordinates, rather than one flat loop over the block. Two reasons, and
        both need the parameter to be statically known: its ``n_i`` is then a Python int, so every
        candidate axis is statically sized --- no padding to a global maximum and no masking ---
        and its *update method* is a Python object, so dispatching on it costs nothing at run time.
        The updater loop is static and, in practice, one iteration long.

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
        force = any(not u.forms_running_total for u in updaters)
        plans = self._restricted(force=force)

        env = SweepEnv(
            sampler=self, state=state,
            sweep_ctx=self._sweep_context(state) if plans else None,
            plans=plans, tables=state.discrete_proposal_params,
            u_prop=state.rng_draw.discrete_proposal,           # (sweeps * n, L)
            u_acc=state.rng_draw.discrete_accept,
            n_lanes=L, lane_dim=n)

        def logp(z):
            return self._discrete_log_prob(state, z.reshape(-1))

        def sweep(s_idx, outer):
            """One full pass over every parameter, in declaration order."""
            for u in updaters:
                prep = u.prepare(env)
                outer = jax.lax.fori_loop(
                    0, u.size,
                    lambda c, carry, _u=u, _p=prep: _u.step(env, _p, s_idx, c, carry),
                    outer)
            return outer

        z0 = state.discrete.reshape(L, n)
        # The full path seeds the running density; the delta path has no use for it and pays one
        # evaluation at the exit instead of one here plus `n` inside the loop.
        carry = (z0, jnp.zeros((L,)) if plans else logp(z0),
                 jnp.zeros((L,)), jnp.zeros((L,), jnp.int32))
        z, lp, alpha_sum, moved = jax.lax.fori_loop(
            0, self._n_discrete_sweeps, sweep, carry)

        state = state._replace(discrete=z.reshape(-1))
        state = self._after_discrete(state, logp(z) if plans else lp)
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
            "discrete_accept_prob": squeeze(alpha_sum / (self._n_discrete_sweeps * n)),
            "discrete_moves": squeeze(moved),
        })


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

    Composed under :class:`DiscreteMetropolisWithinGibbs` it gives a **discrete-only** sampler,
    which is what a model with no continuous parameters needs --- and what makes the sweep
    testable against an exactly enumerable target, since with the continuous block frozen the
    chain's stationary distribution is a pmf one can write down and compare against.

    It is not a general-purpose "hold these parameters fixed" facility: it freezes *everything*
    continuous, and a model with continuous parameters composed under it will simply never move
    them.
    """

    state_class = StaticState

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
