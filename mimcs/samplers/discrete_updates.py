"""How one discrete parameter gets moved --- one object per parameter, chosen per parameter.

:class:`~mimcs.samplers.DiscreteMetropolisWithinGibbs` used to apply a single update rule to every
discrete parameter of a model. These are the units that let it apply a different one to each, the
way :class:`~mimcs.hmc.KineticHamiltonian` lets a different kinetic act on each block of continuous
coordinates --- a list on the sampler, each element owning its own slice, each adaptation filtering
the list down to the elements it owns.

Two methods so far:

* :class:`MetropolisUpdate` --- the deterministic-scan Metropolis-within-Gibbs step (doc 14), which
  proposes among the ``n_i - 1`` values the coordinate is *not* at and accepts on the ratio. It is
  the default and the only one that uses ``state.discrete_proposal_params``.
* :class:`ExactGibbsUpdate` --- draw from the exact conditional over all ``n_i`` values. No
  proposal, no acceptance test, nothing to adapt.

**Why objects rather than a ``{name: method}`` string dispatch.** The next method in doc 14's
deferred list is a custom *jump operator*, which moves continuous parameters alongside the label
and carries ``|det dT/dx|`` in the ratio. That changes the carry (the coordinate enters), the
acceptance ratio, and the per-parameter configuration (the map ``T`` and the parameters it acts
on). A string cannot carry ``T`` and an ``if`` in the sweep body cannot carry a different carry; an
object absorbs both, and is the single place the extension gets written.

**The sweep environment is passed in, never closed over.** :class:`SweepEnv` carries the *sampler*,
not bound copies of its hooks, because ``_discrete_delta`` / ``_discrete_log_prob`` /
``_sweep_context`` are overridden by :class:`~mimcs.pt.ParallelTemperingSampler` and must resolve
through the MRO at call time. This is the rule ``Hamiltonian.flow(istate, eps, ctx)`` already
follows: the unit takes its context as an argument.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .._logging import get_logger

log = get_logger(__name__)


#: Narrowest support that gets exact conditional Gibbs. **Below this the Metropolis sweep wins**,
#: which is the opposite of what the cost arithmetic alone suggests and is why this is a floor
#: rather than the cap it started as.
#:
#: Two reasons, one proved and one measured. At ``n_i = 2`` the proposal is forced, so the
#: Metropolis arm always proposes the flip and Peskun-dominates a Gibbs draw (asymptotic-variance
#: ratios 5.0 at ``pi_a = 0.6``, unbounded at 0.5, for half the evaluations). At ``n_i = 3`` the
#: learned marginal is still a good enough stand-in for each coordinate's conditional that the same
#: domination shows end to end: 0.91x label ESS and 0.88x ESS/second over 8 paired seeds --- and
#: understated, because 18 of the Metropolis arm's labels were ESS-censored at the draw count while
#: none of the exact arm's were. From ``n_i = 4`` the gap reverses and grows monotonically
#: (1.30x / 1.28x / 1.98x / 2.91x label ESS at k = 4 / 5 / 8 / 16).
#:
#: The mechanism: the table learns a coordinate's **marginal**, not its **conditional**, and the
#: two drift apart as the support widens. See ``tests/experiments/writeups/discrete_exact.md``.
EXACT_MIN_VALUES = 4

#: Widest support that gets exact conditional Gibbs when the density must be evaluated **in full**
#: at each candidate --- ``n_i - 1`` whole-density evaluations against the proposal's one, and
#: ``vmap`` over the candidate axis materialises that many copies of the modified parameter array
#: inside the ``fori_loop`` body.
#:
#: Measured at ``n_i = 8``, where exact still wins 1.49x on ESS/second despite costing 1.35x the
#: wall clock. PLACEHOLDER above that: the mixing gain and the ``O(n_i)`` cost both grow, and which
#: wins has not been measured past 8.
EXACT_MAX_VALUES = 8

#: Widest support that gets exact conditional Gibbs when every component reading the parameter is
#: **elementwise** in it (:func:`~mimcs.samplers.gibbs.only_in_scan_components`), so each candidate
#: costs ``O(1)`` element work instead of a whole density. Measured to ``n_i = 16`` (2.70x
#: ESS/second at no wall-clock cost, 8/8 seeds) and extrapolated from a monotone trend to 64.
#:
#: It coincides with :data:`~mimcs.adaptation.discrete_marginal.WIDE_SUPPORT` and is deliberately
#: **not** defined in terms of it: the two price unrelated trades --- that one is "a table this
#: wide cannot be estimated from the draws", this one is "this many restricted evaluations are
#: affordable" --- and tying them would make one move whenever the other was retuned.
EXACT_MAX_VALUES_ELEMENTWISE = 64

#: The selectable update methods, by the name a spec/kwarg uses.
DISCRETE_METHODS = ("metropolis", "exact")


class SweepEnv(NamedTuple):
    """Everything one sweep holds constant, handed to each updater's step.

    ``sampler`` is the live sampler so that hooks resolve through the MRO (see the module
    docstring). ``u_acc`` is present for every updater even though only
    :class:`MetropolisUpdate` reads it --- see :meth:`ExactGibbsUpdate.step`.
    """

    sampler: Any
    state: Any
    sweep_ctx: Any
    plans: dict
    tables: dict
    u_prop: Array
    u_acc: Array
    n_lanes: int
    lane_dim: int


class DiscreteUpdate:
    """Base: one discrete parameter and the rule that moves it.

    Subclasses set :attr:`kind`, :attr:`uses_proposal_table` and :attr:`forms_running_total`, and
    implement :meth:`prepare` and :meth:`step`. Instances are plain Python objects built once per
    sampler; they are never traced, and they hold no JAX state.
    """

    #: the tag a factory rule and an adaptation filter on, as ``KineticHamiltonian.mass_mode`` is
    kind: str = ""

    #: does this method read ``state.discrete_proposal_params[name]``? The filter
    #: :class:`~mimcs.adaptation.DiscreteMarginalAdaptation` uses to decide what it owns.
    uses_proposal_table: bool = False

    #: can the sweep thread a running log-density through this method? ``False`` puts the **whole**
    #: sweep on the delta path (see ``gibbs.restriction_plans``), because the two carries cannot be
    #: mixed inside one loop.
    forms_running_total: bool = True

    #: does this method move the **continuous** coordinate alongside the label? A custom jump
    #: operator does. When any updater in a sweep sets this, three things the sweep otherwise
    #: computes once from the pre-sweep state become stale after the first accepted move --- the
    #: unpacked continuous values, the state each density hook is called with, and the sample ---
    #: so the sweep rebuilds them per coordinate instead. Gating on this flag is what keeps a model
    #: with no jump on the original path, and so bit-identical.
    #:
    #: It implies ``forms_running_total = False``: a coordinate-moving method needs the density at
    #: the *carried* coordinate, and the running total on the plans path is seeded to zero.
    moves_coordinate: bool = False

    def __init__(self, parameter, start: int):
        self.name = parameter.name
        self.lower = int(parameter.lower_value)
        #: a Python int, so every candidate axis below is statically sized --- no padding to a
        #: global maximum and no masking
        self.n_values = int(parameter.upper_value - parameter.lower_value + 1)
        self.size = int(parameter.size)
        self.start = int(start)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r}, {self.n_values} values)"

    def prepare(self, env: SweepEnv):
        """Per-parameter constants, computed **once per sweep** rather than per coordinate.

        XLA eliminates common subexpressions *within* a loop body but does not lift them out of the
        loop, so anything depending only on the parameter belongs here.
        """
        return ()

    def step(self, env: SweepEnv, prep, s_idx, c, carry):
        """One coordinate of this parameter, in every lane.

        ``carry`` is ``(z, x, lp, alpha_sum, moved)``: the labels ``(L, lane_dim)``, the continuous
        coordinate ``(L, coord_dim // L)``, the running log-density (meaningless, and seeded to
        zero, on the delta path), the summed acceptance probability and the move count. A method
        that does not move the coordinate passes ``x`` straight through.
        """
        raise NotImplementedError

    # --- shared bookkeeping -----------------------------------------------------------------

    def _rng_index(self, env: SweepEnv, s_idx, c):
        """The **global** sweep step, so the draw order matches the flat sweep this replaced.

        Every method indexes the same way whatever it consumes: a method that reads one uniform
        where another reads two must still leave the other's row alone, or a mixed model's two
        halves would diverge from their single-method counterparts with nothing raising.
        """
        return s_idx * env.lane_dim + (self.start + c)


class MetropolisUpdate(DiscreteUpdate):
    """Metropolis-within-Gibbs: propose among the other values, accept on the ratio.

    The method doc 14 describes and the library's default. With a uniform proposal table the
    candidate draw reduces exactly to the unadapted ``1 + floor(u * (n_i - 1))`` offset and the
    Hastings term is identically zero, so an un-adapted run is untouched by the table machinery.
    """

    kind = "metropolis"
    uses_proposal_table = True
    forms_running_total = True

    def prepare(self, env: SweepEnv):
        tbl = env.tables[self.name]                              # (L, size_p, ni)
        # Hoisted out of the coordinate loop: depends only on the table, a per-kernel-call
        # constant. XLA does not lift this out of the loop on its own.
        g = jnp.log(tbl) + jnp.log1p(-tbl)
        cand_offsets = jnp.arange(1, self.n_values, dtype=jnp.int32)   # 1 .. ni-1, compile-time
        return tbl, g, cand_offsets

    def _propose(self, env: SweepEnv, prep, t, c, cur):
        """Draw a candidate among the ``n_i - 1`` values this coordinate is **not** at.

        Split out of :meth:`step` so a jump-carrying variant reuses the proposal -- and therefore
        the Hastings term below stays the same one -- rather than copying it. The map a jump
        applies is deterministic given the drawn value, so it changes the ratio only through its
        Jacobian and leaves ``q`` untouched.
        """
        tbl, _, cand_offsets = prep
        lo, ni, L = self.lower, self.n_values, env.n_lanes
        # Candidates in cyclic order from cur+1, weighted by each lane's own learned marginal.
        cand = lo + jnp.mod((cur[:, None] - lo) + cand_offsets, ni)   # (L, ni-1)
        w = jnp.take_along_axis(tbl[:, c, :], cand - lo, axis=1)      # (L, ni-1)
        cw = jnp.cumsum(w, axis=1)
        total = cw[:, -1] if ni > 1 else jnp.zeros((L,))
        # `ni == 1` leaves an empty candidate axis; the guard then selects nothing and `prop` falls
        # back to `cur` -- the same harmless no-op the uniform sweep makes.
        idx = jnp.sum(cw <= env.u_prop[t][:, None] * jnp.maximum(total, 1e-30)[:, None], axis=1)
        idx = jnp.clip(idx, 0, max(ni - 2, 0))
        return jnp.take_along_axis(cand, idx[:, None], axis=1)[:, 0] if ni > 1 else cur

    def _log_hastings(self, env, prep, c, cur, prop):
        """``log q(prop -> cur) - log q(cur -> prop)``.

        With ``q(a -> b) = p_b / (1 - p_a)`` this is ``g(cur) - g(prop)`` for
        ``g = log p + log1p(-p)``. It is identically zero for a binary coordinate
        (``p_b = 1 - p_a``) and for a uniform table, which is why neither case changes.
        """
        _, g, _ = prep
        lo, ni, L = self.lower, self.n_values, env.n_lanes
        if ni <= 1:
            return jnp.zeros((L,))
        lanes = jnp.arange(L)
        return g[lanes, c, cur - lo] - g[lanes, c, prop - lo]

    def step(self, env: SweepEnv, prep, s_idx, c, carry):
        z, x, lp, alpha_sum, moved = carry
        i = self.start + c
        t = self._rng_index(env, s_idx, c)
        cur = z[:, i]                                            # (L,)
        prop = self._propose(env, prep, t, c, cur)

        plan = env.plans.get(self.name)
        if plan is None:
            z_prop = z.at[:, i].set(prop)
            lp_prop = env.sampler._discrete_log_prob(env.state, z_prop.reshape(-1))
            d_density = lp_prop - lp
        else:
            # Restricted: only the components that read this parameter, and for the elementwise
            # ones only this coordinate's term.
            z_prop = None
            d_density = env.sampler._discrete_delta(
                env.state, env.sweep_ctx, z, self.name, c, cur, prop, plan)
        # The proposal is not symmetric, so the ratio needs its Hastings factor.
        delta = d_density + self._log_hastings(env, prep, c, cur, prop)      # (L,)
        # `log(u) < delta` rather than `u < exp(delta)`: exp overflows to inf for a large
        # improvement and underflows to 0 for a large worsening, and log(0) = -inf accepts exactly
        # when it should. A NaN delta compares False, i.e. rejects.
        accept = jnp.log(env.u_acc[t]) < delta                   # (L,), independent per lane
        if z_prop is None:
            z = z.at[:, i].set(jnp.where(accept, prop, cur))
            # `lp` is not maintained here. Accumulating n float32 increments would drift, and the
            # restricted path never forms a total anyway --- only differences drive acceptance.
        else:
            z = jnp.where(accept[:, None], z_prop, z)
            lp = jnp.where(accept, lp_prop, lp)
        return (z, x, lp,
                alpha_sum + jnp.minimum(1.0, jnp.exp(jnp.minimum(delta, 0.0))),
                # A *move*, not an acceptance: a degenerate coordinate (n_i = 1) proposes itself
                # and "accepts", which is not a move. This is the column that catches a frozen
                # label, so it must not be inflated by no-ops.
                moved + (accept & (prop != cur)).astype(jnp.int32))


class ExactGibbsUpdate(DiscreteUpdate):
    """Draw the coordinate from its **exact conditional**, over all ``n_i`` values.

    Better mixed than a Metropolis proposal wherever the proposal wastes attempts on values of
    negligible density, at ``n_i`` conditional evaluations against one --- which is why the factory
    selects it only for a narrow support, or a wide one whose every reading component is
    elementwise in the parameter and so costs ``O(1)`` per candidate.

    **Not for a binary coordinate.** At ``n_i = 2`` the Metropolis arm always proposes the flip, so
    it moves with probability ``min(1, pi_b/pi_a)`` where Gibbs moves with probability ``pi_b``.
    That is Peskun domination: measured asymptotic-variance ratios of 5.0 at ``pi_a = 0.6`` and
    unbounded at 0.5, for *half* the density evaluations. The factory rule never selects this for a
    binary parameter; constructing one directly warns rather than refusing, since it is merely
    worse and not wrong.

    **Built out of differences, which is what makes it free under tempering.** A softmax is
    shift-invariant, so the conditional is recovered from ``_discrete_delta`` evaluated at every
    candidate against the current value --- whose own entry is then identically zero. So this
    method needs no density hook of its own, and
    :class:`~mimcs.pt.ParallelTemperingSampler`'s existing per-rung override of that one hook makes
    the tempered path correct with nothing added. The identity was checked numerically before it
    was written: for random unnormalised conditionals and *every* choice of anchor,
    ``softmax(delta vs cur)`` reproduces the exact conditional to 1e-12.
    """

    kind = "exact"
    uses_proposal_table = False
    #: the conditional is a softmax of *differences*; no total is ever formed, so the sweep must
    #: run on the delta path (``gibbs.restriction_plans(force=True)``)
    forms_running_total = False

    def __init__(self, parameter, start: int):
        super().__init__(parameter, start)
        if self.n_values == 2:
            log.warning(
                "exact conditional Gibbs on binary parameter '%s': the Metropolis sweep's "
                "always-flip proposal Peskun-dominates it (it moves with probability "
                "min(1, pi_b/pi_a) where this moves with probability pi_b) at half the density "
                "evaluations. This is worse, not wrong --- the factory never selects it here.",
                self.name)

    def prepare(self, env: SweepEnv):
        # Cyclic offsets **including 0**, unlike the Metropolis arm: the current value is a
        # candidate of the conditional, and dropping it would make the chain unable to stay.
        return (jnp.arange(0, self.n_values, dtype=jnp.int32),)

    def step(self, env: SweepEnv, prep, s_idx, c, carry):
        (offsets,) = prep
        z, x, lp, alpha_sum, moved = carry
        lo, ni, L = self.lower, self.n_values, env.n_lanes
        i = self.start + c
        # Indexed by the same global step as every other method, and reads only `u_prop`: an
        # exact draw needs one uniform where Metropolis needs two. `u_acc[t]` is deliberately left
        # unread rather than reused, and both draw components stay allocated at unchanged shapes,
        # so the RNG layout does not depend on which methods a model happens to use. Dropping the
        # unused component would renumber every other stream in the library (`RNGBuffer` splits one
        # subkey per component).
        t = self._rng_index(env, s_idx, c)
        cur = z[:, i]                                            # (L,)
        plan = env.plans[self.name]

        cand = lo + jnp.mod((cur[:, None] - lo) + offsets, ni)   # (L, ni); column 0 IS cur
        if ni > 1:
            def delta_at(v):                                     # v: (L,)
                return env.sampler._discrete_delta(
                    env.state, env.sweep_ctx, z, self.name, c, cur, v, plan)
            # vmap, not a Python loop: `ni` reaches 64 on the elementwise path, and unrolling
            # would put that many copies of the density into the `fori_loop` body. vmap traces it
            # once whatever `ni` is --- measured flat, 19 jaxpr equations at both ni=3 and ni=64.
            #
            # It also does *not* cost a second evaluation of the `cur` side of each difference,
            # which was the worry: `cur` is not a batched operand, so vmap leaves that half
            # unbatched and it is computed once. Checked rather than assumed --- the primitive
            # appears exactly twice at every `ni`, once at shape `(ni-1,)` and once scalar --- so
            # a coordinate costs `ni - 1` candidate evaluations plus one, not `2(ni - 1)`.
            others = jax.vmap(delta_at)(jnp.swapaxes(cand[:, 1:], 0, 1))       # (ni-1, L)
            logw = jnp.swapaxes(
                jnp.concatenate([jnp.zeros((1, L), others.dtype), others], axis=0), 0, 1)
        else:
            logw = jnp.zeros((L, 1))
        # A non-finite entry must become *unreachable*, not poison the draw. Under Metropolis a NaN
        # delta compares False and rejects; here it would propagate through the cumulative sum and
        # make every comparison False, selecting index 0 -- a silent stay-put reporting acceptance
        # 1.00. Column 0 is an exact zero, so at least one entry always survives this and the
        # softmax is always well defined.
        logw = jnp.where(jnp.isfinite(logw), logw, -jnp.inf)
        w = jnp.exp(logw - jnp.max(logw, axis=1, keepdims=True))
        cw = jnp.cumsum(w, axis=1)
        idx = jnp.sum(cw <= env.u_prop[t][:, None] * cw[:, -1:], axis=1)
        # `ni - 1`, not `ni - 2`: unlike the Metropolis candidate list this one includes `cur`.
        idx = jnp.clip(idx, 0, ni - 1)
        new = jnp.take_along_axis(cand, idx[:, None], axis=1)[:, 0]

        z = z.at[:, i].set(new)
        return (z, x, lp,
                # A Gibbs draw has acceptance probability 1 by construction, so this is the honest
                # contribution rather than a padded one -- but it does mean `discrete_accept_prob`
                # reads 1.00 for an all-exact model and stops being the informative column.
                # `discrete_moves` is what catches a frozen label.
                alpha_sum + 1.0,
                moved + (new != cur).astype(jnp.int32))


_BY_KIND = {"metropolis": MetropolisUpdate, "exact": ExactGibbsUpdate}



def build_discrete_updaters(model, methods=None) -> list:
    """One :class:`DiscreteUpdate` per discrete parameter, in the model's declaration order.

    ``methods`` maps parameter name to method name; a parameter absent from it, and ``None``
    itself, mean ``"metropolis"`` --- the library's behaviour before per-parameter methods existed,
    so a hand-composed stack that passes nothing is unchanged.

    A parameter carrying a :class:`~mimcs.model.jump.JumpOperator` gets the jump-aware variant of
    whichever method it asked for. The jump does not change the *kind*, so a spec or a factory rule
    still names ``"metropolis"`` or ``"exact"`` and this function does the substitution --- which
    is what keeps the operator a property of the **model** rather than something the sampler has to
    be configured with.

    Module-level rather than a method so the sweep and
    :class:`~mimcs.adaptation.DiscreteMarginalAdaptation` derive the same list from the same
    arguments, with no dependence on which of them the MRO initialises first.
    """
    methods = dict(methods or {})
    jumps = getattr(model, "jump_operators", {})
    known = {p.name for p in model.discrete_parameters}
    unknown = sorted(set(methods) - known)
    if unknown:
        raise ValueError(
            f"discrete_update names {unknown}, which are not discrete parameters of this model; "
            f"its discrete parameters are {sorted(known)}")
    updaters = []
    for p in model.discrete_parameters:
        kind = methods.get(p.name, "metropolis")
        if kind not in _BY_KIND:
            raise ValueError(
                f"unknown discrete update method {kind!r} for parameter {p.name!r} "
                f"(use one of {list(DISCRETE_METHODS)})")
        start, _ = model.discrete_block(p.name)
        op = jumps.get(p.name)
        if op is None:
            updaters.append(_BY_KIND[kind](p, start))
        else:
            updaters.append(_BY_KIND_JUMP[kind](p, start, op, model))
    return updaters


# ------------------------------------------------------------------ custom jump operators


class JumpMap:
    """A :class:`~mimcs.model.jump.JumpOperator` realised on the **coordinate** vector of one lane.

    The operator is written in *sample* space, which is what a DSL author sees; the state holds
    coordinates. This is the adapter, and two details in it are load-bearing.

    **Only the output blocks are written back.** The obvious spelling --- unpack the whole
    coordinate, substitute, repack --- is wrong, because ``to_coordinate(from_coordinate(x))`` is
    not bitwise identity for a nonlinear chart in float32. Repacking would perturb every *other*
    parameter in its last bits, turning "moves only ``eta``" into an unbiased-looking random walk
    on everything and breaking the balance checks for reasons unrelated to the map.

    **The Jacobian is taken in coordinate space**, which is what makes it correct with no separate
    chart-Jacobian term: the target :meth:`~mimcs.model.Model.log_prob_at_coordinate` already
    carries the chart Jacobian, so differentiating the composed coordinate map accounts for it.
    Because no output may be a chart parent (:meth:`~mimcs.model.Model._validate_jumps`), the
    composed map is block **triangular** --- everything outside the output blocks is the identity
    --- so its determinant is that of the output block alone.

    Under tempering this works in the **base** model's layout: a ``ProductModel``'s ``coord_dim``
    is ``K`` times the base's, so unpacking the product vector would be nonsense. Charts are shared
    across rungs and the map depends on no rung's ``beta``, so one implementation serves both.
    """

    def __init__(self, operator, model):
        base = getattr(model, "base", model)
        self.operator = operator
        self.base = base
        by_name = {p.name: (i, p) for i, p in enumerate(base.parameters)}
        self.outputs = []                       # (chart index, parameter, lo, hi)
        for name in operator.outputs:
            i, p = by_name[name]
            lo, hi = base.coord_block(name)
            self.outputs.append((i, p, int(lo), int(hi)))
        self.out_dim = sum(hi - lo for _, _, lo, hi in self.outputs)

    # --- the map -----------------------------------------------------------------------------

    def apply(self, x, z, c, v, hyper, idx):
        """One lane: coordinate ``(coord_dim,)`` and labels ``(lane_dim,)`` -> new coordinate.

        ``c`` is the flat, 0-based offset within the discrete parameter's own block and ``v`` the
        proposed value, which is the contract :class:`~mimcs.model.jump.JumpOperator` states.
        """
        values = self.base.unpack_coordinate(x, hyper, idx, z)
        outs = self.operator.fn(values, c, v)
        if not isinstance(outs, tuple):
            outs = (outs,)
        if len(outs) != len(self.outputs):
            raise ValueError(
                f"jump operator for '{self.operator.parameter}' declares "
                f"{len(self.outputs)} output(s) {list(self.operator.outputs)} but returned "
                f"{len(outs)} value(s)")
        for (i, p, lo, hi), new in zip(self.outputs, outs):
            # Every output's chart parents are untouched by construction -- an output may not *be*
            # a parent -- so the original values are the right ones to read them from.
            parents = {n: values[n] for n in getattr(p, "parents", ())}
            q = p.to_coordinate(new, hyper[i], idx[i], parents)
            x = x.at[lo:hi].set(jnp.reshape(q, (hi - lo,)))
        return x

    def _gather(self, x):
        return jnp.concatenate([x[lo:hi] for _, _, lo, hi in self.outputs])

    def _scatter(self, x, sub):
        k = 0
        for _, _, lo, hi in self.outputs:
            x = x.at[lo:hi].set(sub[k:k + hi - lo])
            k += hi - lo
        return x

    def log_det(self, x, z, c, v, hyper, idx):
        """``log |det dx'/dx|`` over the output block, for one lane.

        Zero and free when the operator is volume preserving, which is what a compensating shift
        is. Otherwise ``jacfwd`` over the output block: ``out_dim`` tangents and an
        ``O(out_dim^3)`` determinant, **per candidate** -- a runtime and memory cost, not a
        compile-size one, since ``jacfwd`` traces one JVP and batches it.
        """
        if self.operator.volume_preserving:
            return jnp.zeros(())

        def f(sub):
            return self._gather(self.apply(self._scatter(x, sub), z, c, v, hyper, idx))

        return jnp.linalg.slogdet(jax.jacfwd(f)(self._gather(x)))[1]


class _JumpUpdate(DiscreteUpdate):
    """Shared machinery for an update method carrying a jump operator.

    A jump is a **modifier, not a kind**: it changes how a candidate is evaluated, not how one is
    chosen, so it composes with both methods below rather than replacing them. That is why this is
    a mixin over the two update classes and not a third ``kind``.

    **The full-density path, always.** :meth:`DiscreteMetropolisWithinGibbs._discrete_delta`
    deliberately omits the chart Jacobian, because it cancels for a move that touches only labels.
    Under a jump it does not cancel, so the restricted path is not merely unhelpful here but
    *unsafe*. Restricted recomputation for jumps is deferred with that as its blocker.

    **No new sampler hook.** Every density is taken through the existing
    :meth:`~DiscreteMetropolisWithinGibbs._discrete_log_prob`, with the moved coordinate
    substituted into the state it is handed. So a tempered sampler's override is picked up through
    the MRO with nothing added --- the same reason :class:`SweepEnv` carries the sampler rather
    than bound copies of its hooks.
    """

    moves_coordinate = True
    #: implied by :attr:`moves_coordinate`: this method needs the density at the *carried*
    #: coordinate, and the running total is seeded to zero on the delta path that a
    #: coordinate-moving method necessarily shares a sweep with.
    forms_running_total = False

    def __init__(self, parameter, start: int, operator, model):
        super().__init__(parameter, start)
        self.operator = operator
        self.map = JumpMap(operator, model)

    def __repr__(self) -> str:
        return (f"{type(self).__name__}({self.name!r}, {self.n_values} values, "
                f"-> {list(self.operator.outputs)})")

    # --- the three things a jump needs, all per lane ---------------------------------------

    def _logp(self, env: SweepEnv, x, z):
        """The target at a given coordinate and labels --- ``(L,)``.

        ``state._replace`` is a Python-level rebuild of a tuple of tracers, so substituting the
        coordinate inside the loop is free; and ``context()`` never reads ``coordinate``, so a
        tempered override resolves correctly against the substituted one.
        """
        st = env.state._replace(coordinate=x.reshape(-1))
        return env.sampler._discrete_log_prob(st, z.reshape(-1))

    def _move(self, env: SweepEnv, x, z, c, v):
        """Apply the map in every lane. ``v`` is ``(L,)`` --- each lane's own proposed value."""
        st = env.state
        return jax.vmap(self.map.apply, in_axes=(0, 0, None, 0, None, None))(
            x, z, c, v, st.chart_hyperparams, st.chart_indices)

    def _log_det(self, env: SweepEnv, x, z, c, v):
        """``log |det|`` in every lane --- ``(L,)``. Identically zero, and free, when the operator
        is volume preserving, which is the common case."""
        if self.operator.volume_preserving:
            return jnp.zeros((env.n_lanes,))
        st = env.state
        return jax.vmap(self.map.log_det, in_axes=(0, 0, None, 0, None, None))(
            x, z, c, v, st.chart_hyperparams, st.chart_indices)


class JumpMetropolisUpdate(_JumpUpdate, MetropolisUpdate):
    """Metropolis-within-Gibbs whose accepted move also carries the continuous parameters.

    The proposal and its Hastings term are **exactly** the ones :class:`MetropolisUpdate` uses, and
    that is a consequence rather than a convenience: the table lookup depends on lane, coordinate
    and label only --- never on the coordinate --- so ``q`` factorises out of the joint proposal
    ``q(a->b) delta(x' - Phi(x))`` unchanged, and a deterministic map contributes its density ratio
    entirely as ``|det|``. It holds *only* because the map is deterministic, which the DSL enforces
    structurally by refusing a sampling statement in the body.

    So one term is added to the ratio::

        log alpha = [log pi(z_b, x') - log pi(z_a, x)] + log|det dPhi/dx| + [g(cur) - g(prop)]

    and the reverse move's Jacobian is the inverse of the forward one, which is what the involution
    buys and why a single term suffices.
    """

    kind = "metropolis"

    def step(self, env: SweepEnv, prep, s_idx, c, carry):
        z, x, lp, alpha_sum, moved = carry
        i = self.start + c
        t = self._rng_index(env, s_idx, c)
        cur = z[:, i]                                            # (L,)
        prop = self._propose(env, prep, t, c, cur)

        z_prop = z.at[:, i].set(prop)
        x_prop = self._move(env, x, z, c, prop)
        # Two full evaluations per coordinate: the running total is unavailable here (see
        # `forms_running_total`), and the current side moves whenever an earlier jump was accepted.
        d_density = self._logp(env, x_prop, z_prop) - self._logp(env, x, z)
        delta = (d_density + self._log_det(env, x, z, c, prop)
                 + self._log_hastings(env, prep, c, cur, prop))
        accept = jnp.log(env.u_acc[t]) < delta                   # independent per lane

        z = jnp.where(accept[:, None], z_prop, z)
        x = jnp.where(accept[:, None], x_prop, x)
        return (z, x, lp,
                alpha_sum + jnp.minimum(1.0, jnp.exp(jnp.minimum(delta, 0.0))),
                moved + (accept & (prop != cur)).astype(jnp.int32))


class JumpExactGibbsUpdate(_JumpUpdate, ExactGibbsUpdate):
    """Exact conditional Gibbs over the **orbit** of a jump operator.

    The candidate set is ``{ (v, Phi_{a->v}(x)) }`` and the draw is from Liu and Sabatti's
    generalized-Gibbs weight, which carries the Jacobian because the orbit is parameterised by the
    representative's coordinate::

        w(v)  proportional to  pi(z_v, Phi_{a->v}(x)) * |det dPhi_{a->v}/dx|

    This needs the **cocycle**, not just the involution --- it is what makes the orbit, and so the
    weight vector up to a common factor, the same seen from any member. The gap is not academic: a
    map that negates a coordinate on a label change satisfies the involution and not the cocycle,
    and measures correct under Metropolis and wrong here.

    **The weights are differences against the current member, not absolute densities.** That is not
    a numerical nicety, it is what keeps :class:`ExactGibbsUpdate`'s ``-inf`` guard working. With
    absolute weights, a current state whose own log-density is non-finite makes *every* column
    non-finite, the cumulative comparison false everywhere and the draw index 0 --- a silent
    stay-put reporting acceptance 1.00. Anchoring the current column to an exact zero after the
    finiteness mask guarantees one surviving entry, and the current side is an evaluation this
    method pays anyway.
    """

    kind = "exact"

    def step(self, env: SweepEnv, prep, s_idx, c, carry):
        (offsets,) = prep
        z, x, lp, alpha_sum, moved = carry
        lo, ni, L = self.lower, self.n_values, env.n_lanes
        i = self.start + c
        t = self._rng_index(env, s_idx, c)
        cur = z[:, i]                                            # (L,)

        cand = lo + jnp.mod((cur[:, None] - lo) + offsets, ni)   # (L, ni); column 0 IS cur
        lp_cur = self._logp(env, x, z)                           # (L,)

        def weight_at(v):                                        # v: (L,)
            xv = self._move(env, x, z, c, v)
            return (self._logp(env, xv, z.at[:, i].set(v)) - lp_cur
                    + self._log_det(env, x, z, c, v))

        # vmapped, not looped: `n_i` copies of a full density in the loop body is the thing to
        # avoid, and the candidate axis is statically sized.
        logw = jnp.swapaxes(jax.vmap(weight_at)(jnp.swapaxes(cand, 0, 1)), 0, 1)   # (L, ni)
        logw = jnp.where(jnp.isfinite(logw), logw, -jnp.inf)
        logw = logw.at[:, 0].set(0.0)                            # the anchor; see the docstring
        w = jnp.exp(logw - jnp.max(logw, axis=1, keepdims=True))
        cw = jnp.cumsum(w, axis=1)
        idx = jnp.clip(jnp.sum(cw <= env.u_prop[t][:, None] * cw[:, -1:], axis=1), 0, ni - 1)
        new = jnp.take_along_axis(cand, idx[:, None], axis=1)[:, 0]

        # Recomputed at the selected value rather than selected out of the vmapped candidates:
        # that would materialise `n_i` whole coordinate vectors per lane, and the map is cheap
        # beside the density. It must run against the **pre-move** labels -- the operator reads the
        # current value out of them, so applying it after `z` is updated would make a
        # difference-form map collapse to the identity and freeze the continuous block silently.
        x_new = self._move(env, x, z, c, new)
        # A draw that lands on the current value must leave the coordinate **exactly** alone.
        # `Phi_{a->a}` is only the identity up to rounding for the recommended difference idiom
        # (`(x + effect(a)) - effect(a)` rounds twice), so applying it would let a stay-put draw
        # random-walk the continuous block by an ulp at a time -- a drift with no acceptance test
        # anywhere to stop it. Skipping it makes a no-op a no-op however the arithmetic rounds.
        x = jnp.where((new == cur)[:, None], x, x_new)
        z = z.at[:, i].set(new)
        return (z, x, lp, alpha_sum + 1.0, moved + (new != cur).astype(jnp.int32))


#: The same two methods, for a parameter carrying a custom jump operator. A jump is a *modifier*
#: rather than a kind --- it changes how a candidate is evaluated, not how one is chosen --- so the
#: ``kind`` a spec or a factory rule names is unchanged and this table is keyed by the same
#: strings. Defined here rather than beside :data:`_BY_KIND` only because the classes it names are
#: below it; :func:`build_discrete_updaters` resolves it at call time.
_BY_KIND_JUMP = {"metropolis": JumpMetropolisUpdate, "exact": JumpExactGibbsUpdate}


# ------------------------------------------------------------- the balance checks


#: Relative tolerance for the balance checks, by working precision. Generous against float32,
#: because the composition runs the map twice through the charts and the failures it must catch are
#: structural --- a map that composes to something else is wrong by ``O(1)``, not by an ulp.
#: Relative tolerance for the balance checks, by working precision. Generous against float32: the
#: composition runs the map twice through the charts, and the failures it must catch are structural
#: --- a map that composes to something else is wrong by ``O(1)``, not by an ulp.
_BALANCE_RTOL = {True: 1e-11, False: 1e-5}          # keyed by "is this float64?"

#: The check is a **probe, not a proof** --- it already samples probe points rather than quantifying
#: over the state space --- so it samples the case space too, under these caps. Without them the
#: cost is ``points x coordinates x n_values^3``, which is not a theoretical worry: measured before
#: the caps existed, a 16-coordinate 4-valued parameter on an exact update took **40 seconds** to
#: construct, scaling linearly in the coordinate count and cubically in the support. A realistic
#: model would have hung for minutes or hours, which is a defect and not a slow check.
BALANCE_MAX_POINTS = 4
BALANCE_MAX_COORDINATES = 4
BALANCE_MAX_CASES = 12


def _spread(n: int, k: int) -> list:
    """At most ``k`` indices from ``range(n)``, evenly spread and always including the ends.

    The ends matter: an off-by-one in a parameter's own block shows at ``0`` or ``size - 1`` and
    nowhere else.
    """
    if n <= k:
        return list(range(n))
    return sorted({0, n - 1, *(int(round(i * (n - 1) / (k - 1))) for i in range(k))})


def _balance_probe_points(model, coordinate, discrete, n_points=BALANCE_MAX_POINTS,
                          seed=0x7B0BA1):
    """``(coordinate, labels)`` pairs to probe --- the state's own, plus perturbations of it.

    The state's own point is not enough on its own: the charts' origin is frequently all-zeros, and
    a multiplicative map is accidentally involutive there. Paired rather than crossed, because the
    product of two lists would multiply the cost for no extra coverage of the *map*.
    """
    rng = np.random.default_rng(seed)
    # The working dtype, not Python float: the identity check compares coordinates, so a probe
    # point widened to float64 would never agree with a float32 result.
    dt = np.asarray(coordinate).dtype
    lo = np.asarray(model.discrete_lower, dtype=np.int64)
    hi = np.asarray(model.discrete_upper, dtype=np.int64)
    points = [(np.asarray(coordinate, dtype=dt), np.asarray(discrete, dtype=np.int32))]
    for _ in range(max(0, n_points - 1)):
        x = (points[0][0] + (rng.normal(size=points[0][0].shape) * 1.7 + 0.3)).astype(dt)
        points.append((x, rng.integers(lo, hi + 1).astype(np.int32)))
    return points


def _balance_cases(u, need_cocycle: bool, seed=0x0CC1C1E) -> list:
    """Bounded, deterministic ``(coordinate, a, b, v)`` cases for one updater.

    ``v`` is ``None`` when only the involution is wanted. Small supports are enumerated exhaustively
    --- which is the common case, since the factory only puts narrow parameters on an exact update
    --- and wider ones are sampled.
    """
    rng = np.random.default_rng(seed)
    values = list(range(u.lower, u.lower + u.n_values))
    coords = _spread(u.size, BALANCE_MAX_COORDINATES)
    if need_cocycle:
        full = [(a, b, v) for a in values for b in values for v in values]
    else:
        full = [(a, b, None) for a in values for b in values]
    if len(full) > BALANCE_MAX_CASES:
        idx = rng.choice(len(full), BALANCE_MAX_CASES, replace=False)
        full = [full[i] for i in sorted(idx)]
    return [(c, *case) for c in coords for case in full]


def check_jump_balance(sampler, state) -> None:
    """Verify every jump operator's balance conditions numerically, once, and raise on failure.

    Violating either condition gives a chain that runs, reports plausible diagnostics and targets
    the **wrong posterior**. Nothing downstream would flag it, which is why this raises rather than
    warns --- the same reasoning as the frozen-coordinate refusal in
    :meth:`~mimcs.samplers.BaseSampler.__init__`.

    Three conditions, and the distinction between the last two is real rather than pedantic. Writing
    ``Phi_{a->v}`` for "set the label to ``v`` and apply the map, from current label ``a``":

    * ``Phi_{a->a}`` is the identity. Checked to *tolerance*, not bitwise: the idiom this library
      recommends, ``x + effect(g) - effect(z[j])``, evaluates as ``(x + effect(a)) - effect(a)`` at
      ``g == a``, which rounds twice and lands ~6e-8 away in float32. Requiring exactness would
      reject the canonical map unless its author happened to parenthesise the difference first.
      Exactness is not needed either, because a drawn value equal to the current one skips the map
      entirely (:meth:`JumpExactGibbsUpdate.step`).
    * The **involution** ``Phi_{b->a} . Phi_{a->b} = id``, which Metropolis needs, because the
      reverse move is this same operator run at the current value.
    * The **cocycle** ``Phi_{b->v} . Phi_{a->b} = Phi_{a->v}``, which exact conditional Gibbs needs
      on top, because it draws from an *orbit* and the orbit must look the same from every member.
      Checked only for a parameter actually on an exact update: a map satisfying the involution and
      not the cocycle is perfectly valid under Metropolis --- measured, not assumed --- so demanding
      it of every operator would reject correct models.

    The reverse leg is evaluated with the **moved label and the moved outputs** substituted, which is
    the whole content of the check: the operator reads the current value out of the values dict, so
    running it against the original dict would compose a different pair of maps.
    """
    updaters = [u for u in sampler.discrete_updaters if u.moves_coordinate]
    if not updaters:
        return
    model = sampler.model
    base = getattr(model, "base", model)
    n_lanes = int(getattr(model, "n_temperatures", 1))
    lane_x, lane_z = int(base.coord_dim), int(model.discrete_dim) // n_lanes
    hyper, idx = state.chart_hyperparams, state.chart_indices
    rtol = _BALANCE_RTOL[jnp.zeros(()).dtype == jnp.float64]

    # One lane's worth: the map is per lane, and the charts are shared across rungs.
    points = _balance_probe_points(
        base, np.asarray(state.coordinate)[:lane_x], np.asarray(state.discrete)[:lane_z])

    def rel(a, b):
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        if a.size == 0:
            return 0.0
        return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1.0))

    n_checked = 0
    for u in updaters:
        jm = u.map
        cases = _balance_cases(u, need_cocycle=(u.kind == "exact"))
        for x0, z0 in points:
            for c, a, b, v in cases:
                i = u.start + c

                def phi(x, z, val, _c=c):
                    """The whole move, per lane: apply the map, **then** set the label.

                    In that order. The operator reads the current value out of the values dict, so
                    setting the label first would make a difference-form map see no difference and
                    collapse to the identity.
                    """
                    x2 = jm.apply(jnp.asarray(x), jnp.asarray(z), _c, jnp.int32(val), hyper, idx)
                    return np.asarray(x2), _set(z, i, val)

                za = _set(z0, i, a)
                n_checked += 1

                xa, _ = phi(x0, za, a)
                if rel(xa, x0) > rtol:
                    raise ValueError(_balance_error(
                        u, "is not the identity at the current value",
                        f"applying it at '{u.name}'[{c}] = {a} with no change of value moved the "
                        f"coordinate by a relative {rel(xa, x0):.3e}"))

                x1, z1 = phi(x0, za, b)
                x2, _ = phi(x1, z1, a)
                if rel(x2, x0) > rtol:
                    raise ValueError(_balance_error(
                        u, "is not an involution",
                        f"'{u.name}'[{c}]: {a} -> {b} -> {a} landed at a different point "
                        f"(relative error {rel(x2, x0):.3e} > {rtol:.0e})"))

                if v is None:
                    continue
                xv, _ = phi(x1, z1, v)
                xd, _ = phi(x0, za, v)
                if rel(xv, xd) > rtol:
                    raise ValueError(_balance_error(
                        u, "does not satisfy the cocycle condition exact conditional Gibbs needs",
                        f"'{u.name}'[{c}]: going {a} -> {b} -> {v} landed somewhere other than "
                        f"{a} -> {v} (relative error {rel(xv, xd):.3e} > {rtol:.0e}). The "
                        f"involution *does* hold, so this operator is valid under a Metropolis "
                        f"update --- set this parameter's method to 'metropolis', or rewrite the "
                        f"map"))

    _check_volume_claim(updaters, points, hyper, idx, rtol)
    log.debug("jump balance verified for %s: %d case(s) over %d probe point(s)",
              [u.name for u in updaters], n_checked, len(points))


#: Above this output dimension a declared volume preservation is not verified: the check is an
#: ``O(m)``-tangent Jacobian and an ``O(m^3)`` determinant, which is the very cost the declaration
#: exists to avoid paying per candidate.
VOLUME_CHECK_MAX_DIM = 64


def _check_volume_claim(updaters, points, hyper, idx, rtol) -> None:
    """A declared ``volume_preserving`` operator really must preserve volume.

    The declaration buys a zero Jacobian term at no runtime cost, and getting it wrong biases the
    posterior with nothing reported --- measured on a scaling map wrongly declared preserving, the
    label marginal came out biased by ~5 standard errors over 6 seeds while every diagnostic looked
    ordinary. A *shift* map cannot detect the fault at all, because dropping the Jacobian leaves a
    volume-preserving map correct, which is why the tests control on a scaling map.

    Verified by computing the determinant *once, here*, affordable precisely because it is not per
    candidate --- and under the same case caps as the conditions above. Past
    :data:`VOLUME_CHECK_MAX_DIM` it is not affordable even once, and the claim is **warned about
    rather than checked**: an unverified claim said out loud beats a silent one.
    """
    for u in updaters:
        if not u.operator.volume_preserving:
            continue
        if u.map.out_dim > VOLUME_CHECK_MAX_DIM:
            log.warning(
                "jump operator for '%s' declares itself volume preserving over %d output "
                "coordinate(s), which is too many to verify here (an O(m^3) determinant). The "
                "claim is taken on trust: if it is wrong the posterior is biased and nothing "
                "downstream reports it.", u.name, u.map.out_dim)
            continue
        worst, worst_at = 0.0, None
        for x0, z0 in points:
            for c, a, b, _v in _balance_cases(u, need_cocycle=False):
                z = _set(z0, u.start + c, a)
                ld = abs(float(_forced_log_det(u.map, jnp.asarray(x0), jnp.asarray(z), c,
                                               jnp.int32(b), hyper, idx)))
                if ld > worst:
                    worst, worst_at = ld, (c, a, b)
        if worst > max(rtol, 1e-4):
            c, a, b = worst_at
            raise ValueError(
                f"the jump operator for '{u.name}' is declared volume preserving, but it is not: "
                f"at '{u.name}'[{c}], {a} -> {b}, |log|det dT/dx|| = {worst:.4g} rather than 0.\n"
                f"Either write a map that preserves volume (a compensating *shift* does), or "
                f"declare the operator as scaling so the Jacobian enters the acceptance ratio. "
                f"Leaving it as is biases the posterior silently.")


def _forced_log_det(jm, x, z, c, v, hyper, idx):
    """``JumpMap.log_det`` with the volume-preserving short circuit bypassed --- testing the claim
    that short circuit rests on is this checker's whole job."""
    def f(sub):
        return jm._gather(jm.apply(jm._scatter(x, sub), z, c, v, hyper, idx))
    return jnp.linalg.slogdet(jax.jacfwd(f)(jm._gather(x)))[1]


def _set(z, i, v):
    z = np.array(z, copy=True)
    z[i] = v
    return z


def _balance_error(u, what, detail) -> str:
    return (
        f"the jump operator for '{u.name}' {what}. {detail}.\n"
        f"Write the map as a *difference against the current value* and both conditions hold by "
        f"construction --- e.g. `eta_new = eta + effect({u.name}[j]) - effect(g)`, whose "
        f"compositions telescope. A map written in terms of the proposed value alone generally "
        f"does not.\n"
        f"This raises rather than warns because neither condition is detectable downstream: the "
        f"chain would run, report plausible diagnostics and sample the wrong posterior.")
