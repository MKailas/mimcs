"""Metropolis-within-Gibbs over a model's discrete parameters.

Implements the sampler half of ``docs/design/14_discrete_parameters.md``. Two classes:

* :class:`DiscreteMetropolisWithinGibbs` --- a **kernel-composing mixin**. It sweeps the discrete
  coordinates after whatever continuous kernel it is composed over, so
  ``make_sampler_class(RobbinsMonroStepSize, DiscreteMetropolisWithinGibbs, NUTS)`` is a NUTS
  sampler that also moves labels. Composing two ``pi``-invariant kernels leaves ``pi`` invariant,
  which is the whole argument for why this is allowed to be so simple.
* :class:`StaticContinuous` --- a base algorithm that does nothing to the continuous block, so a
  model that is *only* discrete has something to compose the mixin over.

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

log = get_logger(__name__)


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
        return super()._init_hooks(**kwargs)

    # --- RNG ---

    def make_draw_components(self, model, **kwargs):
        components = super().make_draw_components(model, **kwargs)
        n = model.discrete_dim
        if n == 0:
            return components          # stream-neutral: see the class docstring
        sweeps = int(kwargs.get("discrete_sweeps", 1))
        shape = (sweeps * n,)
        return components + [
            DrawComponent("discrete_proposal", shape, generator=jax.random.uniform),
            DrawComponent("discrete_accept", shape, generator=jax.random.uniform),
        ]

    # --- diagnostics ---

    def init_diagnostics(self) -> dict:
        d = super().init_diagnostics()
        if not self.model.discrete_dim:
            return d
        return {**d,
                "discrete_accept_prob": jnp.zeros(()),
                "discrete_moves": jnp.zeros((), jnp.int32)}

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
        lower, upper = model.discrete_lower, model.discrete_upper
        z = jax.random.randint(key, (model.discrete_dim,), lower, upper + 1).astype(jnp.int32)
        state = state._replace(discrete=z)
        return self._after_discrete(state, self._discrete_log_prob(state, z))

    # --- the sweep ---

    def _discrete_log_prob(self, state, discrete):
        """The coordinate-space target at the current position and the given labels."""
        return self.model.log_prob_at_coordinate(
            state.coordinate, state.chart_hyperparams, state.chart_indices, discrete)

    def kernel(self, state):
        """The composed kernel: the continuous algorithm's step, then the discrete sweep."""
        state = super().kernel(state)
        if not self.model.discrete_dim:
            return state
        return self._discrete_sweep(state)

    def _discrete_sweep(self, state):
        """One pass (or ``discrete_sweeps`` passes) over every discrete coordinate.

        Structured as a **Python loop over the parameters** wrapping a ``fori_loop`` over each
        parameter's own coordinates, rather than one flat loop over ``discrete_dim``. The reason is
        the proposal table: it is keyed per parameter, so inside a parameter's loop ``n_i`` is a
        Python int and the candidate axis is statically sized --- no padding to a global maximum
        and no masking. The parameter loop is static and, in practice, one iteration long.
        """
        model = self.model
        n = model.discrete_dim
        tables = state.discrete_proposal_params
        u_prop = state.rng_draw.discrete_proposal
        u_acc = state.rng_draw.discrete_accept

        def logp(z):
            return self._discrete_log_prob(state, z)

        def sweep(s_idx, outer):
            """One full pass over every parameter, in declaration order."""
            z, lp, alpha_sum, moved = outer
            for p in model.discrete_parameters:
                lo = int(p.lower_value)
                ni = int(p.upper_value - p.lower_value + 1)      # Python int: static width
                start, _ = model.discrete_block(p.name)
                tbl = tables[p.name]                             # (size_p, ni)
                # Hoisted out of the coordinate loop: it depends only on the table, which is a
                # per-kernel-call constant. XLA eliminates common subexpressions *within* a loop
                # body but does not lift them out of the loop.
                g = jnp.log(tbl) + jnp.log1p(-tbl)
                cand_offsets = jnp.arange(1, ni, dtype=jnp.int32)   # 1 .. ni-1, compile-time

                def body(c, carry):
                    z, lp, alpha_sum, moved = carry
                    i = start + c
                    # The RNG index is the *global* step, so the draw order matches the flat sweep
                    # this replaced. Getting it wrong shifts every draw and shows up only as a
                    # failed regression test.
                    t = s_idx * n + i
                    cur = z[i]

                    # Candidates in cyclic order from cur+1, weighted by the learned marginal.
                    # At a uniform table this reproduces the unadapted `1 + floor(u*(ni-1))`
                    # offset exactly -- verified over 2.4e6 float32 cases, and asserted in
                    # tests/test_discrete_adaptation.py.
                    cand = lo + jnp.mod((cur - lo) + cand_offsets, ni)
                    w = tbl[c, cand - lo]
                    cw = jnp.cumsum(w)
                    total = cw[-1] if ni > 1 else jnp.zeros(())
                    # `ni == 1` leaves an empty candidate axis; the guard then selects nothing and
                    # `prop` falls back to `cur` -- the same harmless no-op the uniform sweep makes.
                    idx = jnp.sum(cw <= u_prop[t] * jnp.maximum(total, 1e-30))
                    idx = jnp.clip(idx, 0, max(ni - 2, 0))
                    prop = cand[idx] if ni > 1 else cur

                    z_prop = z.at[i].set(prop)
                    lp_prop = logp(z_prop)
                    # The proposal is no longer symmetric, so the Metropolis ratio needs its
                    # Hastings factor:  q(b->a)/q(a->b) = [p_a (1-p_a)] / [p_b (1-p_b)],
                    # i.e. g(cur) - g(prop) with g = log p + log1p(-p). It is identically zero for
                    # a binary coordinate (p_b = 1 - p_a) and for a uniform table, which is why
                    # neither case changes.
                    log_hast = (g[c, cur - lo] - g[c, prop - lo]) if ni > 1 else jnp.zeros(())
                    delta = lp_prop - lp + log_hast
                    # `log(u) < delta` rather than `u < exp(delta)`: exp overflows to inf for a
                    # large improvement and underflows to 0 for a large worsening, and
                    # log(0) = -inf accepts exactly when it should. A NaN delta compares False,
                    # i.e. rejects.
                    accept = jnp.log(u_acc[t]) < delta
                    z = jnp.where(accept, z_prop, z)
                    lp = jnp.where(accept, lp_prop, lp)
                    return (z, lp,
                            alpha_sum + jnp.minimum(1.0, jnp.exp(jnp.minimum(delta, 0.0))),
                            # A *move*, not an acceptance: a degenerate coordinate (n_i = 1)
                            # proposes itself and "accepts", which is not a move. This is the
                            # column that catches a frozen label, so it must not be inflated by
                            # no-ops.
                            moved + (accept & (prop != cur)).astype(jnp.int32))

                z, lp, alpha_sum, moved = jax.lax.fori_loop(
                    0, p.size, body, (z, lp, alpha_sum, moved))
            return (z, lp, alpha_sum, moved)

        z0 = state.discrete
        carry = (z0, logp(z0), jnp.zeros(()), jnp.zeros((), jnp.int32))
        z, lp, alpha_sum, moved = jax.lax.fori_loop(
            0, self._n_discrete_sweeps, sweep, carry)

        state = state._replace(discrete=z)
        state = self._after_discrete(state, lp)
        return state._replace(diagnostics={
            # Merged into the dict the base kernel returned, not into `init_diagnostics()`: a
            # kernel *replaces* the diagnostics dict, so anything not added here is never recorded.
            **state.diagnostics,
            "discrete_accept_prob": alpha_sum / (self._n_discrete_sweeps * n),
            "discrete_moves": moved,
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
