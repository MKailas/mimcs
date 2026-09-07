"""Within-orbit step-size adaptivity: the ``LineSearchIntegrator`` (WALNUTS).

The defining feature of WALNUTS (Bou-Rabee, Carpenter, Kleppe, Liu) --- *within-orbit
adaptivity of leapfrog* --- lives entirely in axis 2 (the integrator). So it is wrapped
here as an integrator: composing it with the fixed-length :class:`HMC` gives WAL-HMC, and
with :class:`NUTS` gives WAL-NUTS, with no sampler changes.

Each macro step refines the discretization until the energy error over the step is within a
budget. A *base* integrator (e.g. leapfrog) is kept as a component and used for both the
forward and the reversibility-preserving backward sub-steps. A **schedule** of
``(h_j, T_j)`` levels (``T_j`` base steps of relative size ``h_j``; level 0 coarsest) and a
**schedule of energy-error thresholds** ``delta_j`` (one per level) drive the refinement.
Classical WALNUTS halves the step and doubles the count (``h_j T_j`` constant), but the
schedule is arbitrary: ``h_j T_j`` need not be fixed, so a refinement may also *shorten* the
integration time --- useful when stiff (funnel) geometry is localized.

Two variants live here:

**The error measure is the energy range** ``max_k H(s_k) - min_k H(s_k)`` over the closed macro
step (:meth:`LineSearchIntegrator._integrate_level`), not a start-relative deviation
``max_k |H(s_k) - H(start)|``. The reason is *direction symmetry*: the level-``j`` segment forward
from ``z`` and the level-``j`` segment backward from its endpoint are the same set of states, so
the range agrees exactly between the two directions, where a deviation does not --- it measures
against ``H(z)`` one way and ``H(z')`` the other. That symmetry is what makes the randomized
variant reversible at all, and it is the same idiom the NUTS divergence test uses.

:class:`LineSearchIntegrator` --- the **deterministic (WALNUTS-D)** scheme. Each macro step
picks the *coarsest* level whose energy range is within budget, so the micro-step-length
distribution is concentrated on that single level. Reversibility: for a macro step from ``z``
(direction ``sign(eps)``), let ``L_fwd(z)`` be that level, giving ``z' = Phi_{L_fwd}(z)``, and
``L_bwd(z')`` the coarsest level within budget going *backward* from ``z'``. The step is **valid**
iff ``L_fwd(z) == L_bwd(z')``; the reverse step ``z' -> z`` is valid under the identical
condition, so valid moves are reversible pairs and the leapfrog map at the chosen level is exactly
reversed. Invalid (or non-finite) steps accumulate ``-inf`` into ``log_weight``, which the
samplers fold into acceptance / selection (so they are rejected / never chosen).

**A symmetric measure does not remove that check**, which is worth stating because the opposite is
the natural guess. Symmetry gives ``E(z', L_fwd) = E(z, L_fwd) <= delta``, so the chosen level is
always valid backward and hence ``L_bwd <= L_fwd`` --- the check reduces to a scan of the
*coarser* levels only (:meth:`_coarser_level_valid`), which is cheaper and never re-integrates
level ``L_fwd``. But *minimality* is a claim about those other levels, and for ``j < L_fwd`` the
level-``j`` segment backward from ``z'`` is a different arc from the level-``j`` segment forward
from ``z`` (a different discretization of the same time interval, started from the opposite end),
so a coarser level can be within budget one way and not the other. Measured on a 21-d Neal funnel
(``scale=3``, ``eps=0.39``, doubling schedule, ``delta=1.0``): the range measure cuts the
disagreement rate from 13.5% to 11.0% --- it does not reach zero.

:class:`MarkovianLineSearchIntegrator` --- a **randomized** alternative that avoids invalidation
entirely (except on a true numerical blow-up), at the cost of a more spread-out micro-step-length
distribution. The level is chosen by a Markov chain from coarse to fine: at level ``j < n``, if
``err_j > thr_j`` the finer level is taken (**forced**); otherwise the finer level is taken with
probability ``p`` (**unforced**) or the chain stops at ``j`` with probability ``1 - p``. The
finest level ``n`` stops automatically (infinite threshold). Stopping at the chosen level ``J``
always has positive reverse probability --- every advance factor below ``J`` is positive (1 when
forced, ``p_j`` when unforced), and the stop factor at ``J`` is positive because the backward range
at ``J`` *equals* the forward one and so is within budget, or ``J = n``. The move is therefore
reversible **in general**: no step needs to be invalidated. That argument is exactly what the range
measure buys and what a start-relative deviation denied it --- under the deviation measure the
reverse chain was *forced* past ``J`` on 0.8% of non-finest steps (9.7% deep in a funnel neck), so
``P_rev(J|z') = 0`` while :meth:`_backward_logp` priced it as positive.

The symmetry is exact in real arithmetic and near-exact in float32: over 3445 non-finest stops on a
21-d funnel the forward and backward ranges were bit-identical 55% of the time and differed by at
most 1.5e-5, against a smallest distance-to-threshold of 2.0e-3. So a float32 tie *at* the
threshold remains conceivable; nothing else does.

Under the codebase's convention (``log_weight`` a reward added to ``-H``; samplers
fold it as ``exp(H0 - H1 + Δlog_weight)``), detailed balance sets the correction to
``log P_rev(J|z') - log P_fwd(J|z)``, i.e. each *unforced forward* move contributes ``-log p_j``
(a boost --- the state was proposed with probability ``~ p``) and each *backward* coarser level
whose backward error is within budget contributes ``+log p_{j'}`` (a penalty). The ``(1 - p)``
stop factors cancel between the two directions. ``-inf`` is reserved for non-finite energy alone.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from .._logging import get_logger
from .hamiltonians import total_energy

log = get_logger(__name__)


def doubling_schedule(n_levels: int = 5) -> list[tuple[float, int]]:
    """Classical WALNUTS levels: ``(h_j, T_j) = (1/2^j, 2^j)`` (fixed total time)."""
    return [(0.5 ** j, 2 ** j) for j in range(n_levels)]


#: The default energy-error budget ``delta_j``, per level, for **one** Hamiltonian. Named because
#: it is not only this class's default: the product space restates it per temperature and scales it
#: by K (:func:`~mimcs.pt.integrators.product_error_thresholds`), which must not re-guess the value.
DEFAULT_ERROR_THRESHOLDS = 1.0


class LineSearchIntegrator:
    """Within-orbit adaptive integrator wrapping a base integrator (axis 2).

    Args:
        base: the base integrator whose ``step`` does one sub-step (e.g. ``leapfrog(...)``).
        potentials, kinetic: the Hamiltonian components, to evaluate the energy error.
        schedule: list of ``(h_j, T_j)`` levels, coarsest first (default doubling).
        error_thresholds: per-level energy-error budget ``delta_j`` (scalar broadcast or a
            list matching ``schedule``).
    """

    n_rng_per_step = 0   # deterministic: the WALNUTS-D line search consumes no randomness
    emits_step_size_proxy = True   # writes a coarse-level proxy energy into integrator_data

    def init_integrator_data(self) -> dict:
        """Schema seeded once per trajectory: a cumulative coarse-level-equivalent ``proxy_energy``
        (drives step-size adaptation), ``refine_sum`` / ``n_steps`` for the mean-refinement
        diagnostic, and the cumulative ``grad_evals`` gradient-evaluation count."""
        z = jnp.zeros(())
        return {"proxy_energy": z, "refine_sum": z, "n_steps": z, "grad_evals": z}

    def _proxy_update(self, data, h_start, h_end_star, level_star, level_actual):
        """Advance the proxy energy by one macro step. The proxy transports the energy change *at the
        coarsest valid (forward) level* ``j*`` to a coarsest-level-equivalent scale, then re-expresses
        it as an energy change over the *actual* level-``j*`` integration time: ``ΔH_{j*} / h_{j*}²``
        (see ``docs/design/06``). [Derivation: the mean energy error at level j is ``~ h² h_j²``, so
        transporting to the coarsest level divides the mean error ``ΔH/(T_j h h_j)`` by ``h_j²``,
        giving ``ΔH/(T_j h h_j³)``; multiplying back by the actual integration time ``T_j h h_j``
        gives ``ΔH/h_j²``.] This coincides with the earlier ``ΔH/(T_j h_j³)`` exactly when the
        integration time is constant (``T_j h_j = 1``, the classical schedule) and differs only for
        schedules that shorten the integration time on finer levels --- fixing the step-size blowup
        those schedules suffered on stiff cold starts, where the old form over-counted the
        integration time by a factor ``1/(T_j h_j)``.

        Using ``j*`` rather than the *realized* level makes the randomized (Markovian) variant match
        the deterministic one: its extra *unforced* refinement no longer makes the proxy optimistic
        and inflate the step size. The ``refine_sum`` diagnostic and ``grad_evals`` cost both track
        the realized level (actual work done). Tolerates an unseeded ``integrator_data``.

        **Finest-level divergence signal.** If forced refinement drives the coarsest *valid* level all
        the way to the finest (``level_star`` at or past the last level --- no coarser level was within
        budget, or nothing was), the coarsest step is simply too large: emit ``+inf`` proxy energy, so
        the proxy acceptance collapses to 0 and the step size is pulled down. This is a *true*-
        divergence signal for step-size adaptation, distinct from the ``-inf`` ``log_weight`` that
        rejects the actual step; it relaxes automatically once ``h`` shrinks enough that a coarser
        level becomes valid again (so it needs no tuned cap)."""
        z = jnp.zeros(())
        raw = (h_end_star - h_start) / (self._h[level_star] ** 2)
        delta = jnp.where(level_star >= self.n_levels - 1, jnp.inf, raw)
        return {"proxy_energy": data.get("proxy_energy", z) + delta,
                "refine_sum": data.get("refine_sum", z) + level_actual.astype(float),
                "n_steps": data.get("n_steps", z) + 1.0,
                "grad_evals": data.get("grad_evals", z) + self._grad_evals_by_level[level_actual]}

    def __init__(self, base, potentials, kinetic, *, schedule=None,
                 error_thresholds=DEFAULT_ERROR_THRESHOLDS):
        self.base = base
        self.potentials = potentials
        self.kinetic = kinetic
        schedule = list(schedule) if schedule is not None else doubling_schedule()
        self.n_levels = len(schedule)
        self._h = jnp.asarray([float(h) for h, _ in schedule], float)
        t_list = [int(T) for _, T in schedule]
        self._T = jnp.asarray(t_list, jnp.int32)
        # Gradient evaluations for a macro step whose finest level used is j: the forward line
        # search integrates levels 0..j, the reversibility check only the *coarser* levels
        # 0..j-1, and a valid step keeps the forward endpoint (no re-integration) --- so
        # ``2·Σ_{i=0}^{j} T_i - T_j`` base leapfrog steps, one gradient each. Exact for a valid
        # step; an invalidated one early-exits its check and costs less. (This restores the
        # values the table held before the reversibility fix, which charged a full backward line
        # search plus a re-integration --- a coincidence of arithmetic, not the same count.)
        self._grad_evals_by_level = jnp.asarray(
            [2 * sum(t_list[:j + 1]) - t_list[j] for j in range(self.n_levels)], float)
        thr = error_thresholds
        thr = [float(thr)] * self.n_levels if np.isscalar(thr) else [float(t) for t in thr]
        if len(thr) != self.n_levels:
            raise ValueError("error_thresholds length must match the schedule")
        self._thresholds = jnp.asarray(thr, float)
        log.debug("%s: %d level(s) with (h, T) %s, error thresholds %s, gradient evaluations "
                  "per macro step by finest level %s", type(self).__name__, self.n_levels,
                  [(float(h), int(t)) for h, t in schedule], thr,
                  [int(g) for g in self._grad_evals_by_level])

    # --- energy + one level's integration ----------------------------------- #

    def _energy(self, istate, ctx):
        return total_energy(istate, self.potentials, self.kinetic, ctx)

    def _integrate_level(self, start, level, eps, ctx):
        """``T_level`` base steps of size ``eps * h_level``; return the endpoint and the energy
        **range** ``max_k H(s_k) - min_k H(s_k)`` over the closed segment.

        Both endpoints are included, and that is the whole point: the level-``j`` segment forward
        from ``z`` and the level-``j`` segment backward from its endpoint are the *same set of
        states* (leapfrog retraces exactly), so the range is identical in the two directions where
        a start-relative deviation ``max_k |H(s_k) - H(start)|`` is not --- the two directions
        measure against different reference energies. Seeding the running extrema with ``H(start)``
        is therefore load-bearing, not tidiness: drop it and the two directions compare different
        point sets, and the symmetry this measure exists to provide is gone.
        """
        sub_eps = eps * self._h[level]
        h0 = self._energy(start, ctx)

        def body(_, carry):
            s, hmax, hmin = carry
            s = self.base.step(s, sub_eps, ctx)
            h = self._energy(s, ctx)
            return s, jnp.maximum(hmax, h), jnp.minimum(hmin, h)

        s, hmax, hmin = jax.lax.fori_loop(0, self._T[level], body, (start, h0, h0))
        return s, hmax - hmin

    def _within_budget(self, err, level):
        """Is this level's energy range inside its budget? Non-finite (a blow-up) never is."""
        return jnp.isfinite(err) & (err <= self._thresholds[level])

    def _line_search(self, start, eps, ctx):
        """Coarsest level whose energy range is within budget; its endpoint; diverged flag."""

        def cond(carry):
            level, accepted, _ = carry
            return (level < self.n_levels) & (~accepted)

        def body(carry):
            level, _, _ = carry
            end, err = self._integrate_level(start, level, eps, ctx)
            ok = self._within_budget(err, level)
            return jnp.where(ok, level, level + 1), ok, end

        level, accepted, end = jax.lax.while_loop(
            cond, body, (jnp.int32(0), jnp.asarray(False), start))
        return level, end, ~accepted

    def _coarser_level_valid(self, start, eps, ctx, level):
        """Is any level *coarser* than ``level`` within budget from ``start``? Early-exits on the
        first one found, and is trivially ``False`` at ``level == 0``.

        This is the whole of the reversibility check (see the class docstring): the chosen level is
        guaranteed valid backward by the symmetry of the range measure, so only *minimality* can
        fail, and only a coarser level can break it.
        """

        def cond(carry):
            j, found = carry
            return (j < level) & (~found)

        def body(carry):
            j, _ = carry
            _, err = self._integrate_level(start, j, eps, ctx)
            return j + 1, self._within_budget(err, j)

        return jax.lax.while_loop(cond, body, (jnp.int32(0), jnp.asarray(False)))[1]

    # --- integrator interface ----------------------------------------------- #

    def step(self, istate, eps, ctx, rng=None):
        # ``rng`` is accepted (and ignored) so the NUTS leaf call site is uniform across the
        # deterministic and randomized line-search integrators.
        # Forward line search from z gives the level ``L_f`` and the endpoint ``z' =
        # Phi_{L_f}(z)``. The step is valid iff the backward search from ``z'`` would pick the
        # same level, and by the symmetry of the range measure that reduces to: no *coarser*
        # level is within budget backward (see the class docstring). A disagreement is
        # invalidated with ``-inf`` rather than reconciled.
        h0 = self._energy(istate, ctx)
        level, z_end, diverged = self._line_search(istate, eps, ctx)
        invalid = diverged | self._coarser_level_valid(z_end, -eps, ctx, level)
        correction = jnp.where(invalid, -jnp.inf, 0.0)
        data = self._proxy_update(istate.integrator_data, h0, self._energy(z_end, ctx),
                                  level, level)
        return z_end._replace(log_weight=z_end.log_weight + correction, integrator_data=data)

    def flow(self, istate, eps, ctx, use_cache=False):
        return self.step(istate, eps, ctx)

    def integrate(self, istate, eps, n_steps, ctx):
        # Seed this integrator's ``integrator_data`` schema so the ``while_loop`` carry structure is
        # stable even from an unseeded state (e.g. the init step-size line search's MALA probe, which
        # builds its start via ``init_integrator_state`` with ``integrator_data={}``).
        istate = istate._replace(
            integrator_data={**self.init_integrator_data(), **istate.integrator_data})

        def cond(carry):
            i, _ = carry
            return i < n_steps

        def body(carry):
            i, s = carry
            return i + 1, self.step(s, eps, ctx)

        _, out = jax.lax.while_loop(cond, body, (jnp.int32(0), istate))
        return out


class MarkovianLineSearchIntegrator(LineSearchIntegrator):
    """Randomized within-orbit adaptive integrator (module docstring, second variant).

    The refinement level is chosen by a coarse-to-fine Markov chain instead of the coarsest
    valid level: below the error threshold the chain advances with probability ``p`` and stops
    with ``1 - p``; above it (or on non-finite energy at a non-finest level) it advances by
    force; the finest level always stops. It is reversible without invalidation. The forward
    pass consumes ``n_levels`` uniform draws (the per-level coins); the backward pass is
    deterministic accounting of ``log P_rev``.

    Args:
        base, potentials, kinetic, schedule, error_thresholds: as
            :class:`LineSearchIntegrator`.
        p: unforced-refinement probability, scalar broadcast to all levels or a per-level
            list. Each entry must lie in the open interval ``(0, 1)``.
    """

    def __init__(self, base, potentials, kinetic, *, schedule=None,
                 error_thresholds=DEFAULT_ERROR_THRESHOLDS, p: float = 0.5):
        super().__init__(base, potentials, kinetic, schedule=schedule,
                         error_thresholds=error_thresholds)
        pv = [float(p)] * self.n_levels if np.isscalar(p) else [float(x) for x in p]
        if len(pv) != self.n_levels:
            raise ValueError("p length must match the schedule")
        if any(not (0.0 < x < 1.0) for x in pv):
            raise ValueError("p must lie in the open interval (0, 1)")
        self._p = jnp.asarray(pv, float)
        self._log_p = jnp.log(self._p)
        self.n_rng_per_step = self.n_levels

    # --- forward Markov selection (consumes the coins) ---------------------- #

    def _markov_forward(self, start, eps, ctx, rng):
        """Coarse-to-fine Markov selection. Returns
        ``(level, endpoint, log_w_fwd, nonfinite, level_star, end_star)`` where ``log_w_fwd`` sums
        ``-log p_j`` over the unforced advances, ``nonfinite`` flags a non-finite energy at the
        finest level (the only divergence), and ``(level_star, end_star)`` is the coarsest valid
        level (first within budget, or the finest) and its endpoint --- the "needed" refinement the
        proxy energy normalizes by (independent of any extra unforced refinement)."""
        last = self.n_levels - 1

        def cond(carry):
            _, stopped, *_ = carry
            return ~stopped

        def body(carry):
            level, _, _, log_w, _, level_star, end_star, captured = carry
            end, err = self._integrate_level(start, level, eps, ctx)
            finite = jnp.isfinite(err)
            within = self._within_budget(err, level)
            is_finest = level >= last
            coin_advance = within & (~is_finest) & (rng[level] < self._p[level])
            forced_advance = (~within) & (~is_finest)          # over budget / non-finite: go finer
            advance = coin_advance | forced_advance
            stop_here = is_finest | (within & (~coin_advance))
            new_log_w = log_w + jnp.where(coin_advance, -self._log_p[level], 0.0)
            nonfinite = is_finest & (~finite)                  # only the finest can diverge
            # capture the coarsest valid level (first within budget, else the finest) exactly once
            capture_now = (within | is_finest) & (~captured)
            level_star = jnp.where(capture_now, level, level_star)
            end_star = jax.tree.map(
                lambda a, b: jnp.where(capture_now, b, a), end_star, end)
            captured = captured | capture_now
            return (jnp.where(advance, level + 1, level), stop_here, end, new_log_w, nonfinite,
                    level_star, end_star, captured)

        init = (jnp.int32(0), jnp.asarray(False), start, jnp.zeros(()), jnp.asarray(False),
                jnp.int32(last), start, jnp.asarray(False))
        (level, _, endpoint, log_w_fwd, nonfinite,
         level_star, end_star, _) = jax.lax.while_loop(cond, body, init)
        return level, endpoint, log_w_fwd, nonfinite, level_star, end_star

    # --- backward accounting (deterministic: log P_rev) -------------------- #

    def _backward_logp(self, endpoint, eps, ctx, level_chosen):
        """``+log p_{j'}`` for each coarser level ``j' < level_chosen`` whose backward error is
        within budget --- the log-probability the reverse chain advances past it."""

        def body(level, log_w):
            _, err = self._integrate_level(endpoint, level, -eps, ctx)
            within = self._within_budget(err, level)
            return log_w + jnp.where(within, self._log_p[level], 0.0)

        return jax.lax.fori_loop(0, level_chosen, body, jnp.zeros(()))

    # --- integrator interface ---------------------------------------------- #

    def step(self, istate, eps, ctx, rng):
        h0 = self._energy(istate, ctx)
        level, endpoint, log_w_fwd, nonfinite, level_star, end_star = \
            self._markov_forward(istate, eps, ctx, rng)
        log_w_bwd = self._backward_logp(endpoint, eps, ctx, level)
        correction = jnp.where(nonfinite, -jnp.inf, log_w_fwd + log_w_bwd)
        # Proxy from the coarsest valid level ``level_star`` / its endpoint (the needed refinement,
        # so unforced extra refinement does not bias it); refine diagnostic tracks realized ``level``.
        data = self._proxy_update(istate.integrator_data, h0, self._energy(end_star, ctx),
                                  level_star, level)
        return endpoint._replace(log_weight=endpoint.log_weight + correction, integrator_data=data)

    def flow(self, istate, eps, ctx, use_cache=False):
        raise NotImplementedError(
            "MarkovianLineSearchIntegrator is randomized and cannot nest as a splitting Op")

    def integrate(self, istate, eps, n_steps, ctx):
        # ``integrate`` (fixed-length HMC, or the init step-size line-search probe) carries no
        # per-step rng, so run the chain *deterministically* at the coarsest valid level --- rng >= p
        # at every level means no unforced refinement, i.e. WALNUTS-D behaviour. Randomized use is
        # via NUTS, which calls ``step`` per leaf with its own coins. (Full randomized fixed-length
        # integration is deferred with WAL-HMC.)
        #
        # This degradation is silent by construction, so it must not be reachable by accident: the
        # factory refuses a randomized integrator under a base whose ``supplies_integrator_rng`` is
        # False. What remains is the deliberate case --- the one-shot MALA probe in
        # ``StepSizeLineSearch``, where a deterministic level choice is what is wanted anyway.
        istate = istate._replace(
            integrator_data={**self.init_integrator_data(), **istate.integrator_data})
        coarsest = jnp.ones((self.n_levels,), float)

        def cond(carry):
            i, _ = carry
            return i < n_steps

        def body(carry):
            i, s = carry
            return i + 1, self.step(s, eps, ctx, coarsest)

        _, out = jax.lax.while_loop(cond, body, (jnp.int32(0), istate))
        return out
