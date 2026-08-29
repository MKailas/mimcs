"""The No-U-Turn Sampler (NUTS): shared machinery and the memory-efficient version.

``BaseNUTS`` holds everything common to the NUTS variants -- the outer tree-doubling
loop, the generalized U-turn test, multinomial selection, the reversible divergence
test, and the kernel/diagnostics plumbing. Subclasses differ only in ``_build_subtree``
(how one subtree of ``2^j`` leaves is built and its internal U-turn checks performed):

* ``NUTS`` (this file) -- the **memory-efficient** version. It keeps only a small
  per-level checkpoint of the velocity and cumulative-momentum at the left endpoint of
  each open subtree (O(max_tree_depth) state), not the whole subtree. Storing the
  *velocity* (the only thing the U-turn needs) also drops the per-leaf velocity
  computations from O(depth) to one. This is what makes it both lighter and faster.
* ``SimpleNUTS`` (``simple_nuts.py``) -- stores every leaf of the subtree and indexes
  them directly. Simpler and obviously correct; kept as a reference oracle.

Both produce the *same transition kernel*; in fact, given the same RNG they trace the
identical trajectory (verified by an exact-match test), since they make the same U-turn
decisions and the same multinomial draws.

The algorithm structure (iterative tree doubling, generalized U-turn, biased progressive
multinomial selection) follows **NumPyro** and **Blackjax**; credit to them. See
``docs/design/06_hamiltonian_monte_carlo.md`` for the design and the two deliberate
choices here: the generalized (metric-aware, momentum-sum) U-turn, and the reversible
``max H - min H`` divergence test.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .._logging import get_logger
from ..rng import DrawComponent
from .state import IntegratorState
from .samplers import BaseHMC

log = get_logger(__name__)

#: Default energy-range budget for the divergence test. Named because it is a *per-Hamiltonian*
#: calibration: a sampler whose Hamiltonian is a sum of several (parallel tempering's product
#: target) must scale it by that many, or ordinary energy ranges read as divergences.
DEFAULT_DIVERGENCE_THRESHOLD = 1000.0


class NUTSTree(NamedTuple):
    """A (sub)trajectory summary. ``left``/``right`` are its two extreme phase points."""

    left: IntegratorState       # trajectory-order leftmost state
    right: IntegratorState      # trajectory-order rightmost state
    proposal: IntegratorState   # multinomially selected state within this (sub)trajectory
    momentum_sum: Array         # sum of momenta over this (sub)trajectory
    log_weight: Array           # logsumexp of leaf log-weights (= -H) over the (sub)trajectory
    h_min: Array                # min H over the (sub)trajectory (for reversible divergence)
    h_max: Array                # max H over the (sub)trajectory
    sum_accept: Array           # sum of per-leaf min(1, e^{-dH}) (step-size adaptation stat)
    sum_proxy_accept: Array     # sum of per-leaf min(1, e^{-proxy_energy}) (line-search step-size stat)
    sum_grad_evals: Array       # sum of per-leaf integrator gradient-evaluation counts (cost diagnostic)
    n_leaves: Array             # number of leaves (leapfrog steps) taken
    depth: Array                # doublings completed (diagnostic)
    terminated: Array           # bool: turned or diverged (stop expansion)
    diverging: Array            # bool: diverged


def _ntz(x: Array) -> Array:
    """Number of trailing zero bits of a positive integer ``x`` (``x = n+1 >= 1``)."""
    lowest = jnp.bitwise_and(x, -x).astype(float)   # isolates lowest set bit (a power of 2)
    return jnp.round(jnp.log2(lowest)).astype(jnp.int32)


def _tree_index(buffer, i):
    return jax.tree.map(lambda b: b[i], buffer)


def _tree_set(buffer, i, leaf):
    return jax.tree.map(lambda b, v: b.at[i].set(v), buffer, leaf)


def _leaf_proxy_accept(emits, leaf):
    """Per-leaf proxy acceptance ``min(1, exp(-proxy_energy))`` for a line-search integrator (the
    coarse-level step-size-adaptation signal). ``0`` when the integrator emits no proxy, so the
    accumulator stays inert and does not read a missing ``integrator_data`` key. ``emits`` is a
    Python bool (static), so both NUTS builders take the identical branch."""
    if not emits:
        return jnp.zeros(())
    e = leaf.integrator_data["proxy_energy"]
    return jnp.where(jnp.isfinite(e), jnp.minimum(1.0, jnp.exp(-e)), 0.0)


def _leaf_grad_evals(frontier, leaf):
    """This leaf's gradient-evaluation cost: the increment of the integrator's cumulative
    ``grad_evals`` from the pre-step frontier to the produced leaf."""
    z = jnp.zeros(())
    return leaf.integrator_data.get("grad_evals", z) - frontier.integrator_data.get("grad_evals", z)


class BaseNUTS(BaseHMC):
    """Shared NUTS machinery; subclasses implement ``_build_subtree``.

    Args:
        max_tree_depth: maximum number of trajectory doublings (static). The trajectory
            holds up to ``2^max_tree_depth - 1`` leapfrog steps.
        divergence_threshold: a trajectory diverges when ``max(H) - min(H)`` exceeds this.
    """

    #: NUTS calls the integrator's ``step`` once per leaf and declares a per-leaf coin array for
    #: it (see ``make_draw_components``), so a randomized integrator really is randomized here.
    supplies_integrator_rng = True

    def __init__(self, *args, max_tree_depth: int = 10,
                 divergence_threshold: float = DEFAULT_DIVERGENCE_THRESHOLD, **kwargs):
        self.max_tree_depth = int(max_tree_depth)
        self.divergence_threshold = float(divergence_threshold)
        self._max_subtree = 1 << (self.max_tree_depth - 1)   # largest subtree size
        super().__init__(*args, **kwargs)
        log.debug("NUTS: max_tree_depth %d (up to %d leapfrog steps per transition), "
                  "divergence threshold %.4g", self.max_tree_depth,
                  (1 << self.max_tree_depth) - 1, self.divergence_threshold)

    def init_diagnostics(self) -> dict:
        z = jnp.zeros(())
        return {**super().init_diagnostics(), "diverging": jnp.asarray(False),
                "tree_depth": jnp.zeros((), jnp.int32), "n_leaves": jnp.zeros((), jnp.int32)}

    # --- RNG: momentum + per-doubling direction/selection + per-leaf selection ---

    def make_draw_components(self, model, **kwargs):
        comps = []
        for k in self.kinetics:
            comps.extend(k.make_draw_components(model.coord_dim))
        J = self.max_tree_depth
        comps.append(DrawComponent("tree_direction", (J,), jax.random.uniform))
        comps.append(DrawComponent("tree_select", (J,), jax.random.uniform))
        # Stored **flat**, one entry per leaf of the whole trajectory, rather than as a
        # rectangular ``(J, 2^(J-1))`` table with a row per doubling. Only ``2^j`` entries of row
        # ``j`` are ever read, so the rectangular form wastes a factor
        # ``J * 2^(J-1) / (2^J - 1) ~ 5`` of the RNG buffer --- and that buffer is the largest
        # array a NUTS sampler holds (21 MB at the defaults, doubled under x64). Depth ``j``'s
        # entries live at ``[2^j - 1 : 2^(j+1) - 1]``, the heap layout, so the offset is
        # ``2^depth - 1``. Same count of draws consumed in the same order; only the layout moves.
        # :mod:`mimcs.pt.nuts` has used this layout since it was written.
        comps.append(DrawComponent("leaf_select", ((1 << J) - 1,), jax.random.uniform))
        # A randomized integrator (MarkovianLineSearchIntegrator) needs per-leaf uniforms for
        # its line-search coins. Declared *only* when the integrator asks for them, so plain
        # NUTS and the deterministic WALNUTS-D integrator keep the identical seed stream.
        n_rng = getattr(self.integrator, "n_rng_per_step", 0)
        if n_rng > 0:
            comps.append(DrawComponent(
                "line_search", ((1 << J) - 1, n_rng), jax.random.uniform))
        return comps

    # --- selection weighting (override point for the slice-sampler variant) ---

    def _leaf_log_weight(self, H, H0) -> Array:
        """Multinomial weight of a leaf in log space (proportional to e^{-H})."""
        return -H

    # --- generalized U-turn predicate (states) ---

    def _is_turning(self, end_a: IntegratorState, end_b: IntegratorState,
                    momentum_sum: Array, ctx) -> Array:
        va = self.kinetic_velocity(end_a, ctx)
        vb = self.kinetic_velocity(end_b, ctx)
        return (jnp.dot(momentum_sum, va) <= 0.0) | (jnp.dot(momentum_sum, vb) <= 0.0)

    # --- subtree builder: subclass duty ---

    def _build_subtree(self, frontier, eps, depth, H0, leaf_u, leaf_ls, ctx) -> NUTSTree:
        """Build one size-``2^depth`` subtree; its ``h_min``/``h_max`` are over its *own* leaves
        (so ``h_max - h_min`` is the subtree's own energy range, used for its own divergence).

        ``leaf_ls`` is the per-leaf line-search draw matrix (shape ``(max_subtree, n_rng)``) for a
        randomized integrator, or ``None`` when the integrator is deterministic."""
        raise NotImplementedError

    # --- build the full trajectory by doubling ---

    def _build_nuts(self, state, istate0, ctx):
        H0 = self.total_energy(istate0, ctx)
        tree0 = NUTSTree(
            left=istate0, right=istate0, proposal=istate0, momentum_sum=istate0.p,
            log_weight=-H0, h_min=H0, h_max=H0, sum_accept=jnp.zeros(()),
            sum_proxy_accept=jnp.zeros(()), sum_grad_evals=jnp.zeros(()),
            n_leaves=jnp.int32(0), depth=jnp.int32(0),
            terminated=jnp.asarray(False), diverging=jnp.asarray(False))

        def cond(c):
            j, tree = c
            return (j < self.max_tree_depth) & (~tree.terminated)

        def body(c):
            j, tree = c
            forward = state.rng_draw.tree_direction[j] >= 0.5
            eps = jnp.where(forward, state.step_size, -state.step_size)
            frontier = jax.tree.map(
                lambda l, r: jnp.where(forward, r, l), tree.left, tree.right)

            # The whole flat draw goes down; ``_build_subtree`` offsets into it by depth.
            leaf_ls = getattr(state.rng_draw, "line_search", None)
            sub = self._build_subtree(
                frontier, eps, j, H0, state.rng_draw.leaf_select, leaf_ls, ctx)

            far = sub.right
            new_left = jax.tree.map(lambda l, f: jnp.where(forward, l, f), tree.left, far)
            new_right = jax.tree.map(lambda r, f: jnp.where(forward, f, r), tree.right, far)
            new_psum = tree.momentum_sum + sub.momentum_sum

            # biased progressive multinomial across the merge (bias toward the new subtree)
            log_accept = jnp.minimum(0.0, sub.log_weight - tree.log_weight)
            take_new = jnp.log(state.rng_draw.tree_select[j]) < log_accept
            new_proposal = jax.tree.map(
                lambda t_, s_: jnp.where(take_new, s_, t_), tree.proposal, sub.proposal)
            new_logw = jnp.logaddexp(tree.log_weight, sub.log_weight)
            global_turn = self._is_turning(new_left, new_right, new_psum, ctx)

            # Divergence of the *merged* trajectory (old tree + new subtree). The ``h_max - h_min``
            # test is over the whole trajectory (``sub`` reports only its own range), so a merge
            # can diverge even when both parts are individually non-divergent -- e.g. a single leaf
            # that drops the energy far below the start. A merge-divergence stops expansion and
            # flags the transition, but it does NOT discard the subtree: the subtree is dropped
            # only if it turns or diverges *on its own* (``sub.terminated``). Discarding on the
            # merged range would be history-dependent and break reversibility.
            cum_h_min = jnp.minimum(tree.h_min, sub.h_min)
            cum_h_max = jnp.maximum(tree.h_max, sub.h_max)
            # A non-finite energy anywhere in the merged trajectory is a divergence too. The
            # bare ``h_max - h_min`` test misses it: a NaN/inf leaf poisons ``cum_h_min``/
            # ``cum_h_max`` and ``NaN > threshold`` is False, so a frozen (overflowing)
            # trajectory would otherwise go unflagged (see also the per-leaf ``isfinite``
            # guard in ``_build_subtree``).
            h_range = cum_h_max - cum_h_min
            merged_diverging = (~jnp.isfinite(h_range)) | (h_range > self.divergence_threshold)

            merged = NUTSTree(
                left=new_left, right=new_right, proposal=new_proposal,
                momentum_sum=new_psum, log_weight=new_logw,
                h_min=cum_h_min, h_max=cum_h_max, sum_accept=tree.sum_accept,
                sum_proxy_accept=tree.sum_proxy_accept, sum_grad_evals=tree.sum_grad_evals,
                n_leaves=tree.n_leaves, depth=j + 1,
                terminated=global_turn, diverging=merged_diverging)

            sub_ok = ~sub.terminated            # discard only on the subtree's own U-turn/divergence
            final = jax.tree.map(lambda m, t: jnp.where(sub_ok, m, t), merged, tree)
            final = final._replace(
                terminated=jnp.where(sub_ok, merged.terminated, True) | merged_diverging,
                diverging=tree.diverging | merged_diverging,
                sum_accept=tree.sum_accept + sub.sum_accept,
                sum_proxy_accept=tree.sum_proxy_accept + sub.sum_proxy_accept,
                sum_grad_evals=tree.sum_grad_evals + sub.sum_grad_evals,
                n_leaves=tree.n_leaves + sub.n_leaves,
                h_min=cum_h_min,
                h_max=cum_h_max,
                depth=jnp.where(sub_ok, j + 1, tree.depth))
            return (j + 1, final)

        _, tree = jax.lax.while_loop(cond, body, (jnp.int32(0), tree0))

        n_leaves = jnp.maximum(tree.n_leaves.astype(float), 1.0)
        accept_prob = tree.sum_accept / n_leaves
        accepted = ~jnp.all(tree.proposal.q == istate0.q)
        # Step-size-adaptation signals for a line-search integrator: proxy acceptance averaged over
        # all leaves (mirrors accept_prob), and the mean refinement level along the accepted path.
        if self.integrator.emits_step_size_proxy:
            proxy_accept_prob = tree.sum_proxy_accept / n_leaves
            pd = tree.proposal.integrator_data
            mean_refine = pd["refine_sum"] / jnp.maximum(pd["n_steps"], 1.0)
        else:
            proxy_accept_prob = accept_prob
            mean_refine = jnp.zeros(())
        return (tree.proposal, accept_prob, accepted, tree.diverging, tree.depth,
                proxy_accept_prob, mean_refine, tree.n_leaves, tree.sum_grad_evals)

    # --- sampler hooks ---

    def build_trajectory_and_select(self, state, istate0, ctx):
        built = self._build_nuts(state, istate0, ctx)
        return built[0], self._nuts_diagnostics(built)

    def _nuts_diagnostics(self, built) -> dict:
        (_, accept_prob, accepted, diverging, depth, proxy_accept_prob, mean_refine,
         n_leaves, grad_evals) = built
        return {"accept_prob": accept_prob, "accepted": accepted, "grad_evals": grad_evals,
                "mean_refine": mean_refine, "proxy_accept_prob": proxy_accept_prob,
                "diverging": diverging, "tree_depth": depth, "n_leaves": n_leaves}

    def kernel(self, state):
        ctx = self.context(state)
        p0 = self.sample_momentum(state.rng_draw, state.coordinate, ctx)
        istate0 = IntegratorState(
            q=state.coordinate, p=p0,
            potential_values=state.potential_values,
            potential_grads=state.potential_grads,
            log_weight=jnp.zeros(()),
            integrator_data=self.integrator.init_integrator_data())
        built = self._build_nuts(state, istate0, ctx)
        chosen = built[0]
        new_coordinate = chosen.q
        new_sample = self.model.coordinate_to_sample(
            new_coordinate, state.chart_hyperparams, state.chart_indices)
        new_log_prob = -sum(chosen.potential_values.values())
        return state._replace(
            coordinate=new_coordinate, sample=new_sample, log_prob=new_log_prob,
            momentum=chosen.p, potential_values=chosen.potential_values,
            potential_grads=chosen.potential_grads,
            diagnostics=self._nuts_diagnostics(built))

    # --- end-of-phase reporting ---

    def _warmup_end_hooks(self, completed: int, stopped: bool) -> None:
        super()._warmup_end_hooks(completed, stopped)
        n_div = self.divergence_count(include_warmup=True, include_sampling=False)
        log.debug("NUTS warmup: %d of %d transition(s) diverged, mean tree depth %.2f",
                  n_div, completed, self.mean_tree_depth())

    def _sample_end_hooks(self, state):
        """Warn about post-warmup divergences: unlike warmup's (common and harmless), a divergence
        with the adaptation frozen biases the draws that were just produced."""
        state = super()._sample_end_hooks(state)
        flags = self._diag_values("diverging")
        n_div = int(flags.sum())
        if n_div:
            log.warning(
                "NUTS: %d of %d post-warmup transition(s) diverged (%.1f%%). The sampler could "
                "not integrate part of the target, so the draws are biased there --- try a "
                "smaller step size (a higher target_accept), a longer warmup, or a "
                "reparametrization.", n_div, flags.size, 100.0 * n_div / max(flags.size, 1))
        else:
            log.debug("NUTS: no post-warmup divergences in %d transition(s)", flags.size)
        return state

    def divergence_count(self, *, include_warmup: bool = False,
                         include_sampling: bool = True) -> int:
        """Number of diverged transitions. Warmup and sampling divergences are counted separately
        (early-warmup divergences are common and harmless), so the default counts **sampling-phase**
        only -- the phase whose draws are the evidence a later run consumes."""
        return int(self._diag_values("diverging", warmup=include_warmup,
                                     sampling=include_sampling).sum())

    def divergence_rate(self, *, include_warmup: bool = False,
                        include_sampling: bool = True) -> float:
        """Fraction of transitions that diverged, over the selected phase(s) (default
        sampling-only, matching :meth:`divergence_count`). The factory reads this to skip
        evidence-based mass-mode selection when the pilot's *sampling* ran too pathologically for
        its scores to be trustworthy (see :func:`mimcs.factory.rules.mass_mode_rule`)."""
        v = self._diag_values("diverging", warmup=include_warmup, sampling=include_sampling)
        return float(v.mean()) if v.size else 0.0

    def mean_tree_depth(self) -> float:
        v = self._diag_values("tree_depth", warmup=True, sampling=True)   # all phases (as before)
        return float(np.mean(v)) if v.size else float("nan")


class NUTS(BaseNUTS):
    """Memory-efficient NUTS: O(max_tree_depth) checkpoints instead of the full subtree.

    For the subtree U-turn checks, only the *velocity* and the cumulative momentum at the
    left endpoint of each open subtree are kept (a per-level checkpoint). The velocity is
    the only thing the U-turn test needs, so this both shrinks the working state from
    O(2^depth) to O(depth) and computes one velocity per leaf instead of one per check.
    """

    def _build_subtree(self, frontier, eps, depth, H0, leaf_u, leaf_ls, ctx):
        dim = frontier.q.shape[0]
        J = self.max_tree_depth
        emits = self.integrator.emits_step_size_proxy
        n_total = jnp.left_shift(jnp.int32(1), depth)
        offset = n_total - 1                      # this depth's slice of the flat leaf draws
        # per-level checkpoints: velocity and cumulative momentum at the left endpoint of
        # the open size-2^i subtree.
        ckpt_velocity = jnp.zeros((J, dim))
        ckpt_cumpsum = jnp.zeros((J, dim))

        def cond(c):
            n, *_, turning, diverging = c
            return (n < n_total) & (~turning) & (~diverging)

        def body(c):
            (n, frontier, cumpsum, ckpt_velocity, ckpt_cumpsum, leaf0, proposal,
             sub_logw, h_min, h_max, sum_accept, sum_proxy_accept, sum_grad_evals,
             turning, diverging) = c

            leaf = self.integrator.step(
                frontier, eps, ctx, None if leaf_ls is None else leaf_ls[offset + n])
            grad_evals_leaf = _leaf_grad_evals(frontier, leaf)
            H = self.total_energy(leaf, ctx)
            v_leaf = self.kinetic_velocity(leaf, ctx)          # one velocity per leaf
            # leaf.log_weight carries an integrator correction (e.g. WALNUTS's reversibility
            # weight; -inf for an invalid step); 0 for ordinary symplectic integrators.
            logw = self._leaf_log_weight(H, H0) + leaf.log_weight
            a_leaf = jnp.where(jnp.isfinite(H), jnp.minimum(1.0, jnp.exp(H0 - H)), 0.0)

            cumpsum_before = cumpsum
            cumpsum_after = cumpsum + leaf.p
            leaf0 = jax.tree.map(lambda a, b: jnp.where(n == 0, b, a), leaf0, leaf)

            new_logw = jnp.logaddexp(sub_logw, logw)
            take = jnp.log(leaf_u[offset + n]) < (logw - new_logw)
            proposal = jax.tree.map(lambda a, b: jnp.where(take, b, a), proposal, leaf)
            sub_logw = new_logw

            h_min = jnp.minimum(h_min, H)
            h_max = jnp.maximum(h_max, H)
            diverging = (diverging | (~jnp.isfinite(H)) | (~jnp.isfinite(leaf.log_weight))
                         | ((h_max - h_min) > self.divergence_threshold))
            sum_accept = sum_accept + a_leaf
            sum_proxy_accept = sum_proxy_accept + _leaf_proxy_accept(emits, leaf)
            sum_grad_evals = sum_grad_evals + grad_evals_leaf

            # The levels closed at leaf n (subtree U-turn checks) are exactly 1..ntz(n+1),
            # and the levels at which leaf n is a new left endpoint are 1..ntz(n) (all
            # levels for n=0). Both are contiguous, so loop only over the real work rather
            # than a masked pass over all max_tree_depth levels.
            def read_level(i, turn):
                rho = cumpsum_after - ckpt_cumpsum[i]
                t = (jnp.dot(rho, ckpt_velocity[i]) <= 0.0) | (jnp.dot(rho, v_leaf) <= 0.0)
                return turn | t

            turning = jax.lax.fori_loop(1, _ntz(n + 1) + 1, read_level, turning)

            def write_level(i, carry):
                cv, cc = carry
                return cv.at[i].set(v_leaf), cc.at[i].set(cumpsum_before)

            n_write = jnp.where(n == 0, depth, _ntz(jnp.maximum(n, 1)))
            ckpt_velocity, ckpt_cumpsum = jax.lax.fori_loop(
                1, n_write + 1, write_level, (ckpt_velocity, ckpt_cumpsum))
            return (n + 1, leaf, cumpsum_after, ckpt_velocity, ckpt_cumpsum, leaf0,
                    proposal, sub_logw, h_min, h_max, sum_accept, sum_proxy_accept,
                    sum_grad_evals, turning, diverging)

        init = (jnp.int32(0), frontier, jnp.zeros(dim), ckpt_velocity, ckpt_cumpsum,
                frontier, frontier, jnp.asarray(-jnp.inf),
                jnp.asarray(jnp.inf), jnp.asarray(-jnp.inf), jnp.zeros(()),   # own range: h_min/h_max
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
