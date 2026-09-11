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
        """One coordinate of this parameter, in every lane. ``carry`` is
        ``(z, lp, alpha_sum, moved)``."""
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

    def step(self, env: SweepEnv, prep, s_idx, c, carry):
        tbl, g, cand_offsets = prep
        z, lp, alpha_sum, moved = carry
        lo, ni, L = self.lower, self.n_values, env.n_lanes
        i = self.start + c
        t = self._rng_index(env, s_idx, c)
        cur = z[:, i]                                            # (L,)

        # Candidates in cyclic order from cur+1, weighted by each lane's own learned marginal.
        cand = lo + jnp.mod((cur[:, None] - lo) + cand_offsets, ni)   # (L, ni-1)
        w = jnp.take_along_axis(tbl[:, c, :], cand - lo, axis=1)      # (L, ni-1)
        cw = jnp.cumsum(w, axis=1)
        total = cw[:, -1] if ni > 1 else jnp.zeros((L,))
        # `ni == 1` leaves an empty candidate axis; the guard then selects nothing and `prop` falls
        # back to `cur` -- the same harmless no-op the uniform sweep makes.
        idx = jnp.sum(cw <= env.u_prop[t][:, None] * jnp.maximum(total, 1e-30)[:, None], axis=1)
        idx = jnp.clip(idx, 0, max(ni - 2, 0))
        prop = jnp.take_along_axis(cand, idx[:, None], axis=1)[:, 0] if ni > 1 else cur

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
        # The proposal is no longer symmetric, so the Metropolis ratio needs its Hastings factor:
        # q(b->a)/q(a->b) = [p_a (1-p_a)] / [p_b (1-p_b)], i.e. g(cur) - g(prop) with
        # g = log p + log1p(-p). It is identically zero for a binary coordinate (p_b = 1 - p_a) and
        # for a uniform table, which is why neither case changes.
        lanes = jnp.arange(L)
        log_hast = (g[lanes, c, cur - lo] - g[lanes, c, prop - lo]) if ni > 1 \
            else jnp.zeros((L,))
        delta = d_density + log_hast                             # (L,)
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
        return (z, lp,
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
        z, lp, alpha_sum, moved = carry
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
        return (z, lp,
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

    Module-level rather than a method so the sweep and
    :class:`~mimcs.adaptation.DiscreteMarginalAdaptation` derive the same list from the same
    arguments, with no dependence on which of them the MRO initialises first.
    """
    methods = dict(methods or {})
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
        updaters.append(_BY_KIND[kind](p, start))
    return updaters
