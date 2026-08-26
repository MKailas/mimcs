"""Independent per-temperature trajectory construction and selection for PT-NUTS.

Today's PT-NUTS builds **one** trajectory in the product space and selects **one** leaf index
shared by every temperature. The cold chain's proposal is then chosen partly by energy variation
it does not care about. This module makes each temperature pick its own point.

**Two changes are needed, and only together are they valid.**

1. *Decoupled directions.* Each lane draws its own ``tree_direction``, ``tree_select`` and
   ``leaf_select``, so lane ``k``'s trajectory is an ordinary NUTS orbit for ``pi^beta_k``.
2. *Stopping combines per-lane verdicts, never per-lane quantities.* The trajectory stops when
   **any** lane's own U-turn fires (``min_k`` of the test quantities ``<= 0``) or **any** lane
   diverges (``max_k`` of the per-lane energy ranges over threshold). Every lane therefore stops
   at the same doubling and builds the same number of leaves, which is what keeps the vmapped
   product step efficient --- no lane freezing, no masked accumulators, no wasted work.

**Why that is reversible.** In each lane the doubling construction checks *every canonical
sub-block* of its own ``T_k`` (see ``read_level``'s ``1..ntz(n+1)`` bound in ``mimcs/hmc/nuts.py``).
So a collection ``(T_1..T_K)`` is reachable iff no lane's criterion fires on any *proper* canonical
sub-block of its own tree and some lane fires on the full tree --- both functions of the collection
alone, with no offset in them. For any ``z'`` on ``T``, every level below ``J`` presents lane ``k``
with a proper sub-block, which by validity cannot fire, so construction from ``z'`` cannot stop
early and meets the identical check at ``J``. Both ``z`` and ``z'`` stop at ``J``, each with
direction probability ``2^-JK``; with per-lane multinomial weights ``prod_k pi_k(z'_k)`` the
detailed-balance ratio is symmetric.

**The invariant a reader will otherwise optimize away.** Combining lanes *inside* a test quantity
--- the summed ``sum_k rho_k . v_k`` that today's joint rule uses --- does **not** survive decoupled
directions. That sum pairs the specific per-lane blocks the offsets select, and independent
selection is exactly what makes the offsets differ. Counterexample, K=2, depth 2, four points per
lane: from offsets ``(1,1)`` the level-1 check is ``rho_1{0,1}.v + rho_2{0,1}.v``; from ``(1,2)`` it
is ``rho_1{0,1}.v + rho_2{2,3}.v``. Different quantities, so the stopping time and hence
reachability differ. Measured on a 1-d Gaussian at K=2 (200k draws x 8 seeds) the summed form biases
the cold variance to 0.853 against 1.0 --- a 90-sigma miss --- while this module's rule lands at
0.9985 +- 0.0023. The cold *mean* is clean in both, so a check on means alone would have passed the
invalid rule.

Consequently **anything that lets lane k's step depend on lane j's state destroys the argument**.
Today that is the line-search integrators, whose refinement level comes from the summed Hamiltonian
(``mimcs/hmc/line_search.py``); :func:`~mimcs.pt.sampler.parallel_tempering` refuses them here.

See ``docs/design/13_parallel_tempering.md`` and the writeups in ``tests/experiments/writeups/``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from .._logging import get_logger
from ..hmc.nuts import NUTSTree, _leaf_grad_evals, _leaf_proxy_accept, _ntz, _tree_index, _tree_set
from ..rng import DrawComponent
from .lanes import LaneStateMixin, lanes, spread

log = get_logger(__name__)


def _joint_accept(H0: Array, H: Array) -> Array:
    """Per-leaf acceptance on the **summed** product energy --- the joint path's statistic.

    Deliberately not the per-lane mean. The step size here is a single global number, so it must
    be tuned against a single global statistic, and using the same one the joint path uses is what
    makes the two comparable in an A/B. Adaptation is warmup-only and frozen afterwards, so this
    choice cannot affect what the sampler targets.

    Measured when it *was* the per-lane mean: at the same step size the per-lane statistic reads
    much higher (0.914 vs 0.781 at eps=0.5 on `correlated_gaussian`, K=4) for two compounding
    reasons --- a lane's own dH is a quarter of the summed dH, and the min rule's trees are short
    (2.9 leaves against 11.1) so the mean is taken over leaves near the start where dH is small.
    Against a 0.8 target that drove the step to 1.28, where momentum reverses within one leapfrog
    step, every trajectory U-turned at the first doubling and divergences went 0.2% -> 6.4%. The
    trees were then one leaf, which kept the statistic high: a runaway, not a new equilibrium.
    """
    tot0, tot = jnp.sum(H0), jnp.sum(H)
    return jnp.where(jnp.isfinite(tot), jnp.minimum(1.0, jnp.exp(tot0 - tot)), 0.0)


class PerTemperatureNUTSMixin(LaneStateMixin):
    """Per-lane directions and selection; stopping by combined per-lane verdicts.

    Composed above :class:`~mimcs.hmc.NUTS`. Overrides the doubling loop and the subtree builder
    wholesale --- the base's ``_is_turning`` is not an adequate seam, because ``NUTS._build_subtree``
    inlines the U-turn against stored checkpoint velocities rather than calling it.
    """

    def _init_hooks(self, **kwargs):
        """``per_temperature_step_size``: adapt one step size per rung instead of one global.

        Independent selection is what makes this meaningful -- each rung now has its own
        acceptance signal, which the joint path does not (doc 13). It is off by default because
        the two changes were measured apart: see the module docstring on how a per-lane
        acceptance statistic behaves when the trees are short.
        """
        self._per_temp_step = bool(kwargs.get("per_temperature_step_size", False))
        super()._init_hooks(**kwargs)

    # --- acceptance: one signal per step size ------------------------------------- #

    def _accept_zeros(self) -> Array:
        return jnp.zeros((self._K,)) if self._per_temp_step else jnp.zeros(())

    def _leaf_accept(self, H0: Array, H: Array) -> Array:
        """Per-leaf acceptance, shaped to match the step size it will drive.

        Per-rung: lane ``k``'s own ``min(1, exp(H0_k - H_k))``, the signal for lane ``k``'s step.
        Global: the **summed**-energy acceptance, which is the joint path's statistic -- see
        :func:`_joint_accept` for why a per-lane *mean* must not be used to drive a global step.
        """
        if self._per_temp_step:
            return jnp.where(jnp.isfinite(H), jnp.minimum(1.0, jnp.exp(H0 - H)), 0.0)
        return _joint_accept(H0, H)

    # --- shapes ------------------------------------------------------------------- #

    @property
    def _K(self) -> int:
        return self.model.n_temperatures

    @property
    def _n(self) -> int:
        return self.model.base.coord_dim

    def _lanes(self, x: Array) -> Array:
        return lanes(x, self._K, self._n)

    # --- RNG: one direction / selection stream per temperature ---------------------- #

    def make_draw_components(self, model, **kwargs):
        """Per-lane draws. Terminates the chain (does not call ``super()``), as ``BaseNUTS`` does.

        ``leaf_select`` is stored **flat** as ``(2^J - 1, K)`` rather than the base's rectangular
        ``(J, 2^(J-1))``: only ``2^j`` entries of row ``j`` are ever read, so the rectangular form
        wastes a factor ``J*2^(J-1)/(2^J-1) ~ 5`` of the RNG buffer, which is already the largest
        array a NUTS sampler holds and would otherwise be K times larger again here. The same draws
        are consumed in the same order; only the layout differs.
        """
        comps = []
        for k in self.kinetics:
            comps.extend(k.make_draw_components(model.coord_dim))
        J, K = self.max_tree_depth, model.n_temperatures
        comps.append(DrawComponent("tree_direction", (J, K), jax.random.uniform))
        comps.append(DrawComponent("tree_select", (J, K), jax.random.uniform))
        comps.append(DrawComponent("leaf_select", ((1 << J) - 1, K), jax.random.uniform))
        return comps

    def init_diagnostics(self) -> dict:
        """Per-lane signals ride *alongside* the scalar ones, not in place of them.

        ``accept_prob`` and ``accepted`` keep their existing scalar meanings so that every
        adaptation mixin and reader written against them behaves exactly as it does on the joint
        path. In particular :class:`~mimcs.adaptation.RobbinsMonroStepSize` reads
        ``diagnostics["accept_prob"]`` and would otherwise broadcast a ``(K,)`` signal into a
        ``(K,)`` step size --- turning on per-rung step sizes as a silent side effect, which is
        exactly the second change v1 is meant to hold fixed. (Measured when it did: the step ran
        to [1.23, 1.22, 0.87, 1.19] against the joint path's scalar 0.766, momentum reversed
        within a single leapfrog step, every trajectory U-turned at the first doubling, and the
        divergence rate went 0.2% -> 6.4%.)

        ``accept_prob`` is the mean over lanes and ``accepted`` is the **cold** lane's, matching
        ``get_samples()``. The per-lane vectors are kept under ``*_lanes`` for diagnostics, and
        are what a per-rung step size would read if it is ever turned on.
        """
        d = {**super().init_diagnostics(), "accepted_lanes": jnp.zeros((self._K,), bool)}
        if self._per_temp_step:
            d["accept_prob"] = jnp.zeros((self._K,))     # one signal per rung's step size
        return d

    # --- lane-wise state surgery ---------------------------------------------------- #

    def _mix_lanes(self, a, b, take: Array):
        """Lane ``k`` of the result comes from ``a`` where ``take[k]``, else from ``b``.

        Only ``q``, ``p`` and the potential gradients are lane-structured. ``potential_values`` is
        a **sum over lanes** and so describes neither parent after a mix; it is deliberately left
        as ``a``'s and repaired once at the end of :meth:`_build_nuts`, because nothing in between
        reads it --- energies go through :meth:`per_temperature_energy`, which recomputes from
        ``q``, and the integrator's cached kick reads only the gradients, which mix correctly
        (row ``k`` of a tempered potential's gradient is ``beta_k * grad V(q_k)``, a function of
        lane ``k`` alone).
        """
        m = spread(take, self._n)
        grads = {i: jnp.where(m, g, b.potential_grads[i]) for i, g in a.potential_grads.items()}
        return a._replace(q=jnp.where(m, a.q, b.q), p=jnp.where(m, a.p, b.p),
                          potential_grads=grads)

    def _lane_turn_quantities(self, end_a, end_b, momentum_sum: Array, ctx):
        """Each lane's own generalized U-turn dot products --- two ``(K,)`` vectors."""
        va = self._lanes(self.kinetic_velocity(end_a, ctx))
        vb = self._lanes(self.kinetic_velocity(end_b, ctx))
        rho = self._lanes(momentum_sum)
        return jnp.sum(rho * va, axis=1), jnp.sum(rho * vb, axis=1)

    def _any_lane_turns(self, end_a, end_b, momentum_sum: Array, ctx) -> Array:
        """Scalar verdict: has **any** lane turned? (``min_k`` of the quantities ``<= 0``.)"""
        qa, qb = self._lane_turn_quantities(end_a, end_b, momentum_sum, ctx)
        return jnp.any((qa <= 0.0) | (qb <= 0.0))

    def _diverged(self, h_min: Array, h_max: Array) -> Array:
        """Scalar verdict: has **any** lane's own energy range blown the budget?

        ``max_k (h_max_k - h_min_k)`` is one Hamiltonian's range, so the threshold is the ordinary
        per-chain one --- *not* the ``* K`` the joint path uses to account for a summed Hamiltonian.
        """
        rng_ = h_max - h_min
        return jnp.any(~jnp.isfinite(rng_)) | (jnp.max(rng_) > self.divergence_threshold)

    # --- the doubling loop ----------------------------------------------------------- #

    def _build_nuts(self, state, istate0, ctx):
        K, n = self._K, self._n
        H0 = self.per_temperature_energy(istate0, ctx)              # (K,)
        tree0 = NUTSTree(
            left=istate0, right=istate0, proposal=istate0, momentum_sum=istate0.p,
            log_weight=-H0, h_min=H0, h_max=H0,
            sum_accept=self._accept_zeros(), sum_proxy_accept=jnp.zeros(()),
            sum_grad_evals=jnp.zeros(()),
            n_leaves=jnp.int32(0), depth=jnp.int32(0),
            terminated=jnp.asarray(False), diverging=jnp.asarray(False))

        def cond(c):
            j, tree = c
            return (j < self.max_tree_depth) & (~tree.terminated)

        def body(c):
            j, tree = c
            forward = state.rng_draw.tree_direction[j] >= 0.5        # (K,)
            s = jnp.broadcast_to(jnp.asarray(state.step_size), (K,))
            eps = spread(jnp.where(forward, s, -s), n)               # (K*n,), signed per lane
            frontier = self._mix_lanes(tree.right, tree.left, forward)
            # Zero the cumulative gradient counter so each leaf's delta is its own cost: the
            # counter is a product-level scalar and a lane mix has to take one parent's.
            frontier = frontier._replace(
                integrator_data=self.integrator.init_integrator_data())

            sub = self._build_subtree(
                frontier, eps, j, H0, state.rng_draw.leaf_select, None, ctx)

            far = sub.right
            new_left = self._mix_lanes(tree.left, far, forward)
            new_right = self._mix_lanes(far, tree.right, forward)
            new_psum = tree.momentum_sum + sub.momentum_sum

            # biased progressive multinomial across the merge, independently per lane
            log_accept = jnp.minimum(0.0, sub.log_weight - tree.log_weight)      # (K,)
            take_new = jnp.log(state.rng_draw.tree_select[j]) < log_accept       # (K,)
            new_proposal = self._mix_lanes(sub.proposal, tree.proposal, take_new)
            new_logw = jnp.logaddexp(tree.log_weight, sub.log_weight)            # (K,)
            global_turn = self._any_lane_turns(new_left, new_right, new_psum, ctx)

            # Divergence of the merged trajectory, per lane then combined. As in the base, a
            # merge-divergence stops expansion and flags the transition but does NOT discard the
            # subtree --- only the subtree's own verdict does that, or the discard would be
            # history-dependent and break reversibility.
            cum_h_min = jnp.minimum(tree.h_min, sub.h_min)
            cum_h_max = jnp.maximum(tree.h_max, sub.h_max)
            merged_diverging = self._diverged(cum_h_min, cum_h_max)

            merged = NUTSTree(
                left=new_left, right=new_right, proposal=new_proposal,
                momentum_sum=new_psum, log_weight=new_logw,
                h_min=cum_h_min, h_max=cum_h_max, sum_accept=tree.sum_accept,
                sum_proxy_accept=tree.sum_proxy_accept, sum_grad_evals=tree.sum_grad_evals,
                n_leaves=tree.n_leaves, depth=j + 1,
                terminated=global_turn, diverging=merged_diverging)

            sub_ok = ~sub.terminated        # any lane's subtree verdict discards every lane's
            final = jax.tree.map(lambda m, t: jnp.where(sub_ok, m, t), merged, tree)
            final = final._replace(
                terminated=jnp.where(sub_ok, merged.terminated, True) | merged_diverging,
                diverging=tree.diverging | merged_diverging,
                sum_accept=tree.sum_accept + sub.sum_accept,
                sum_proxy_accept=tree.sum_proxy_accept + sub.sum_proxy_accept,
                sum_grad_evals=tree.sum_grad_evals + sub.sum_grad_evals,
                n_leaves=tree.n_leaves + sub.n_leaves,
                h_min=cum_h_min, h_max=cum_h_max,
                depth=jnp.where(sub_ok, j + 1, tree.depth))
            return (j + 1, final)

        _, tree = jax.lax.while_loop(cond, body, (jnp.int32(0), tree0))

        # Repair the lane-mixed proposal's summed value cache: it rides on into the next state as
        # `log_prob` and as the seed for the next transition's cached kick.
        proposal = tree.proposal._replace(potential_values={
            p.id: jnp.sum(p.per_temperature_values(tree.proposal.q, ctx))
            for p in self.potentials})

        n_leaves = jnp.maximum(tree.n_leaves.astype(float), 1.0)
        accept_prob = tree.sum_accept / n_leaves     # scalar summed energy, or (K,) per rung
        accepted_lanes = jnp.any(
            self._lanes(proposal.q) != self._lanes(istate0.q), axis=1)            # (K,)
        accepted = accepted_lanes[0]                        # scalar: the cold chain's
        proxy_accept_prob = (tree.sum_proxy_accept / n_leaves
                             if self.integrator.emits_step_size_proxy else accept_prob)
        self._lane_diagnostics = {"accepted_lanes": accepted_lanes}
        return (proposal, accept_prob, accepted, tree.diverging, tree.depth,
                proxy_accept_prob, jnp.zeros(()), tree.n_leaves, tree.sum_grad_evals)

    def _nuts_diagnostics(self, built) -> dict:
        """The base's dict, plus the per-lane vectors stashed by :meth:`_build_nuts`.

        The stash is read within the same trace that wrote it (``kernel`` calls ``_build_nuts``
        then this, in order), which is why a plain attribute is safe here.
        """
        return {**super()._nuts_diagnostics(built), **self._lane_diagnostics}

    # --- subtree: the memory-efficient (checkpointed) builder ------------------------ #

    def _build_subtree(self, frontier, eps, depth, H0, leaf_u, leaf_ls, ctx) -> NUTSTree:
        """One size-``2^depth`` subtree, per-lane weights and a combined stopping verdict.

        The subtree's *shape* is combinatorial and shared by every lane: ``n_total``, the leaf
        index ``n``, and the level bounds ``_ntz(n+1)`` / ``_ntz(max(n,1))`` depend only on ``n``.
        So the base's contiguous ``fori_loop`` bounds carry over verbatim; the checkpoint arrays
        simply gain a lane axis and the U-turn result gains a reduction.
        """
        K, n_ = self._K, self._n
        J = self.max_tree_depth
        emits = self.integrator.emits_step_size_proxy
        n_total = jnp.left_shift(jnp.int32(1), depth)
        offset = n_total - 1                       # flat leaf_select row for this depth
        ckpt_velocity = jnp.zeros((J, K, n_))
        ckpt_cumpsum = jnp.zeros((J, K, n_))

        def cond(c):
            n, *_, turning, diverging = c
            return (n < n_total) & (~turning) & (~diverging)

        def body(c):
            (n, frontier, cumpsum, ckpt_velocity, ckpt_cumpsum, leaf0, proposal, sub_logw,
             h_min, h_max, sum_accept, sum_proxy_accept, sum_grad_evals, turning, diverging) = c

            leaf = self.integrator.step(frontier, eps, ctx, None)
            grad_evals_leaf = _leaf_grad_evals(frontier, leaf)
            H = self.per_temperature_energy(leaf, ctx)                # (K,)
            v_leaf = self._lanes(self.kinetic_velocity(leaf, ctx))    # (K, n)
            logw = self._leaf_log_weight(H, H0) + leaf.log_weight     # (K,)
            a_leaf = self._leaf_accept(H0, H)         # scalar, or (K,) per-rung

            cumpsum_before = cumpsum
            cumpsum_after = cumpsum + leaf.p
            leaf0 = jax.tree.map(lambda a, b: jnp.where(n == 0, b, a), leaf0, leaf)

            new_logw = jnp.logaddexp(sub_logw, logw)                  # (K,)
            take = jnp.log(leaf_u[offset + n]) < (logw - new_logw)    # (K,)
            proposal = self._mix_lanes(leaf, proposal, take)
            sub_logw = new_logw

            h_min = jnp.minimum(h_min, H)
            h_max = jnp.maximum(h_max, H)
            diverging = (diverging | jnp.any(~jnp.isfinite(H))
                         | (~jnp.isfinite(leaf.log_weight)) | self._diverged(h_min, h_max))
            sum_accept = sum_accept + a_leaf
            sum_proxy_accept = sum_proxy_accept + _leaf_proxy_accept(emits, leaf)
            sum_grad_evals = sum_grad_evals + grad_evals_leaf

            def read_level(i, turn):
                rho = self._lanes(cumpsum_after) - ckpt_cumpsum[i]                # (K, n)
                t = ((jnp.sum(rho * ckpt_velocity[i], axis=1) <= 0.0)
                     | (jnp.sum(rho * v_leaf, axis=1) <= 0.0))                    # (K,)
                return turn | jnp.any(t)

            turning = jax.lax.fori_loop(1, _ntz(n + 1) + 1, read_level, turning)

            def write_level(i, carry):
                cv, cc = carry
                return cv.at[i].set(v_leaf), cc.at[i].set(self._lanes(cumpsum_before))

            n_write = jnp.where(n == 0, depth, _ntz(jnp.maximum(n, 1)))
            ckpt_velocity, ckpt_cumpsum = jax.lax.fori_loop(
                1, n_write + 1, write_level, (ckpt_velocity, ckpt_cumpsum))

            return (n + 1, leaf, cumpsum_after, ckpt_velocity, ckpt_cumpsum, leaf0, proposal,
                    sub_logw, h_min, h_max, sum_accept, sum_proxy_accept, sum_grad_evals,
                    turning, diverging)

        init = (jnp.int32(0), frontier, jnp.zeros(K * n_), ckpt_velocity, ckpt_cumpsum,
                frontier, frontier, jnp.full((K,), -jnp.inf),
                jnp.full((K,), jnp.inf), jnp.full((K,), -jnp.inf), self._accept_zeros(),
                jnp.zeros(()), jnp.zeros(()), jnp.asarray(False), jnp.asarray(False))
        (n_final, last_leaf, cumpsum_final, _, _, leaf0, proposal, sub_logw,
         h_min, h_max, sum_accept, sum_proxy_accept, sum_grad_evals, turning, diverging) = \
            jax.lax.while_loop(cond, body, init)

        return NUTSTree(
            left=leaf0, right=last_leaf, proposal=proposal, momentum_sum=cumpsum_final,
            log_weight=sub_logw, h_min=h_min, h_max=h_max, sum_accept=sum_accept,
            sum_proxy_accept=sum_proxy_accept, sum_grad_evals=sum_grad_evals,
            n_leaves=n_final, depth=jnp.int32(0),
            terminated=turning | diverging, diverging=diverging)


class PerTemperatureSimpleNUTSMixin(PerTemperatureNUTSMixin):
    """The full-buffer reference builder, for the bit-identity oracle.

    Mirrors :class:`~mimcs.hmc.SimpleNUTS` against :class:`~mimcs.hmc.NUTS`: it stores every leaf
    and does the U-turn checks by indexing them, so the checkpoint bookkeeping in
    :meth:`PerTemperatureNUTSMixin._build_subtree` has something to be checked against. Any change
    to one must be mirrored in the other.
    """

    def _build_subtree(self, frontier, eps, depth, H0, leaf_u, leaf_ls, ctx) -> NUTSTree:
        K, n_ = self._K, self._n
        dim = K * n_
        emits = self.integrator.emits_step_size_proxy
        n_total = jnp.left_shift(jnp.int32(1), depth)
        offset = n_total - 1
        buf = jax.tree.map(
            lambda x: jnp.zeros((self._max_subtree,) + x.shape, x.dtype), frontier)
        psum_prefix = jnp.zeros((self._max_subtree, dim))

        def cond(c):
            n, *_, turning, diverging = c
            return (n < n_total) & (~turning) & (~diverging)

        def body(c):
            (n, frontier, buf, psum_prefix, leaf0, proposal, sub_logw, sub_psum,
             h_min, h_max, sum_accept, sum_proxy_accept, sum_grad_evals, turning, diverging) = c

            leaf = self.integrator.step(frontier, eps, ctx, None)
            grad_evals_leaf = _leaf_grad_evals(frontier, leaf)
            H = self.per_temperature_energy(leaf, ctx)
            logw = self._leaf_log_weight(H, H0) + leaf.log_weight
            a_leaf = self._leaf_accept(H0, H)

            buf = _tree_set(buf, n, leaf)
            prev = jnp.where(n > 0, psum_prefix[jnp.maximum(n - 1, 0)], jnp.zeros(dim))
            cur_psum = prev + leaf.p
            psum_prefix = psum_prefix.at[n].set(cur_psum)
            leaf0 = jax.tree.map(lambda a, b: jnp.where(n == 0, b, a), leaf0, leaf)

            new_logw = jnp.logaddexp(sub_logw, logw)
            take = jnp.log(leaf_u[offset + n]) < (logw - new_logw)
            proposal = self._mix_lanes(leaf, proposal, take)
            sub_logw = new_logw
            sub_psum = sub_psum + leaf.p

            h_min = jnp.minimum(h_min, H)
            h_max = jnp.maximum(h_max, H)
            diverging = (diverging | jnp.any(~jnp.isfinite(H))
                         | (~jnp.isfinite(leaf.log_weight)) | self._diverged(h_min, h_max))
            sum_accept = sum_accept + a_leaf
            sum_proxy_accept = sum_proxy_accept + _leaf_proxy_accept(emits, leaf)
            sum_grad_evals = sum_grad_evals + grad_evals_leaf

            def check(i, turn):
                size = jnp.left_shift(jnp.int32(1), i)
                a = (n + 1) - size
                closes = (jnp.bitwise_and(n + 1, size - 1) == 0) & (a >= 0) & (i <= depth)
                end_a = _tree_index(buf, jnp.maximum(a, 0))
                rho = cur_psum - jnp.where(
                    a > 0, psum_prefix[jnp.maximum(a - 1, 0)], jnp.zeros(dim))
                t = self._any_lane_turns(end_a, leaf, rho, ctx)
                return jnp.where(closes, turn | t, turn)

            turning = jax.lax.fori_loop(1, self.max_tree_depth + 1, check, turning)
            return (n + 1, leaf, buf, psum_prefix, leaf0, proposal, sub_logw, sub_psum,
                    h_min, h_max, sum_accept, sum_proxy_accept, sum_grad_evals,
                    turning, diverging)

        init = (jnp.int32(0), frontier, buf, psum_prefix, frontier, frontier,
                jnp.full((K,), -jnp.inf), jnp.zeros(dim),
                jnp.full((K,), jnp.inf), jnp.full((K,), -jnp.inf), self._accept_zeros(),
                jnp.zeros(()), jnp.zeros(()), jnp.asarray(False), jnp.asarray(False))
        (n_final, last_leaf, _, _, leaf0, proposal, sub_logw, sub_psum,
         h_min, h_max, sum_accept, sum_proxy_accept, sum_grad_evals, turning, diverging) = \
            jax.lax.while_loop(cond, body, init)

        return NUTSTree(
            left=leaf0, right=last_leaf, proposal=proposal, momentum_sum=sub_psum,
            log_weight=sub_logw, h_min=h_min, h_max=h_max, sum_accept=sum_accept,
            sum_proxy_accept=sum_proxy_accept, sum_grad_evals=sum_grad_evals,
            n_leaves=n_final, depth=jnp.int32(0),
            terminated=turning | diverging, diverging=diverging)
