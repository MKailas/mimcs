"""Symplectic integrators (axis 2 of the modular HMC design).

Implements the composable splitting integrators from
``docs/design/06_hamiltonian_monte_carlo.md``. An integrator approximates the flow of
the full Hamiltonian by composing the elementary flows of its components. A step is a
*palindromic* sequence of :class:`Op` operations, which guarantees the reversibility
and volume preservation that make the simple ``min(1, exp(-dH))`` HMC acceptance valid.

This module provides the standard kick--drift--kick :func:`leapfrog` and the generic
:class:`SplittingIntegrator` it is built from. Gradient reuse is requested explicitly
for now via ``Op(..., cached_gradient=True)`` on the leading half-kicks; automatic
inference of where the cache is valid is deferred (see doc 06, "Gradient caching").
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .state import IntegratorState


@dataclass(frozen=True)
class Op:
    """One operation in a splitting: flow ``target`` for ``coeff * eps`` of the step.

    ``cached_gradient`` is meaningful only when ``target`` is a single potential (a
    kick): it reuses the cached gradient instead of recomputing it. Valid only where
    no drift has changed ``q`` since the cache was written (e.g. a leapfrog leading
    half-kick); see doc 06.
    """

    target: object          # a Hamiltonian or a (nested) integrator; must have .flow
    coeff: float
    cached_gradient: bool = False


def _integrate_by_stepping(integrator, istate: IntegratorState, eps, n_steps,
                           ctx) -> IntegratorState:
    """Apply ``integrator.step`` until ``n_steps`` is reached --- the shared ``integrate`` body.

    ``while_loop`` rather than ``fori_loop`` so the stopping condition is a value, not a fixed
    trip count, leaving the door open to randomized or criterion-based integration time.
    """
    def cond(carry):
        i, _ = carry
        return i < n_steps

    def body(carry):
        i, s = carry
        return i + 1, integrator.step(s, eps, ctx)

    _, out = jax.lax.while_loop(cond, body, (jnp.int32(0), istate))
    return out


class SplittingIntegrator:
    """One step = a palindromic sequence of component / sub-integrator flows."""

    n_rng_per_step = 0   # deterministic: consumes no per-step randomness (see line_search.py)
    emits_step_size_proxy = False   # writes no proxy energy into integrator_data (see line_search.py)

    def __init__(self, ops: list[Op]):
        self.ops = list(ops)
        # Gradient evaluations per step: each non-cached kick of a gradient-bearing target (a
        # potential) evaluates one gradient; cached leading kicks and drifts evaluate none. For
        # standard leapfrog this is the number of potentials P (the trailing half-kicks).
        self._grad_evals = float(sum(
            1 for op in self.ops
            if not op.cached_gradient and hasattr(op.target, "value_and_grad")))

    def init_integrator_data(self) -> dict:
        """The integrator's ``IntegratorState.integrator_data`` schema, seeded once per trajectory.
        A plain symplectic integrator reports only its cumulative gradient-evaluation count."""
        return {"grad_evals": jnp.zeros(())}

    def step(self, istate: IntegratorState, eps, ctx, rng=None) -> IntegratorState:
        # ``rng`` is accepted (and ignored) so a single call site can drive both this
        # deterministic integrator and randomized ones (MarkovianLineSearchIntegrator).
        for op in self.ops:
            istate = op.target.flow(istate, op.coeff * eps, ctx,
                                    use_cache=op.cached_gradient)
        # Only accumulate when the trajectory was seeded with the schema (Python-static key check,
        # so the pytree structure is preserved through ``fori_loop``s that thread the state --- e.g.
        # a line-search integrator's per-level integration, or direct standalone use on ``{}``).
        data = istate.integrator_data
        if "grad_evals" in data:
            istate = istate._replace(
                integrator_data={**data, "grad_evals": data["grad_evals"] + self._grad_evals})
        return istate

    def flow(self, istate, eps, ctx, use_cache=False) -> IntegratorState:
        # so integrators nest as Op targets; nested integrators manage their own cache
        return self.step(istate, eps, ctx)

    def integrate(self, istate: IntegratorState, eps, n_steps, ctx) -> IntegratorState:
        """Apply ``step`` until ``n_steps`` is reached.

        Uses ``while_loop`` (not ``fori_loop``) so the stopping condition is a value,
        not a fixed trip count: this leaves the door open to randomized or
        criterion-based integration time (e.g. a per-iteration sampled number of steps,
        or a no-U-turn test) by swapping the condition without changing the structure.
        ``n_steps`` may be a Python int or a traced scalar.
        """
        return _integrate_by_stepping(self, istate, eps, n_steps, ctx)


def leapfrog(potentials, kinetics) -> SplittingIntegrator:
    """Standard Stormer--Verlet leapfrog for ``H = sum_i V_i(q) + sum_j T_j(q, p)``.

    ``kinetics`` is the list of kinetic components (a single kinetic is accepted and treated
    as a one-element list). The step is the palindromic splitting

        ``V_1/2 ... V_p/2  T_1/2 ... T_{k-1}/2  T_k  T_{k-1}/2 ... T_1/2  V_p/2 ... V_1/2``,

    a second-order symmetric (reversible, volume-preserving) integrator. The leading potential
    kicks reuse the cached gradient; the trailing kicks recompute and refresh it --- one
    gradient evaluation per potential per step. With a single kinetic the kinetic part is one
    full-step drift, recovering the ordinary leapfrog; with several it composes their flows
    (each component's ``flow`` may be a separable drift, an explicit cross-block kick, or an
    implicit RMHMC solve --- the integrator does not care)."""
    if not isinstance(kinetics, (list, tuple)):
        kinetics = [kinetics]
    lead_v = [Op(V, 0.5, cached_gradient=True) for V in potentials]
    trail_v = [Op(V, 0.5) for V in reversed(potentials)]
    *head, last = kinetics
    lead_t = [Op(k, 0.5) for k in head]
    mid_t = [Op(last, 1.0)]
    trail_t = [Op(k, 0.5) for k in reversed(head)]
    return SplittingIntegrator(lead_v + lead_t + mid_t + trail_t + trail_v)


class RepeatedIntegrator:
    """Run an inner integrator ``n`` times over one outer time slice (``n`` static).

    The inner loop of a multi-rate (RESPA) splitting: as an ``Op`` target its ``flow`` advances
    ``eps`` of outer time as ``n`` inner steps of ``eps / n`` (``docs/design/06``). ``n`` is a
    Python int, so the loop is a fixed-trip ``fori_loop`` and the op structure stays static under
    JIT. It also satisfies the whole integrator protocol, so it can stand alone or serve as a
    :class:`~mimcs.hmc.LineSearchIntegrator` base (a base is used only as
    ``base.step(istate, eps, ctx)``).

    It carries no gradient cost of its own and deliberately defines **no** ``value_and_grad``, so
    an enclosing :class:`SplittingIntegrator` counts it as zero while the inner steps accumulate
    their own cost into the same ``integrator_data``.
    """

    n_rng_per_step = 0
    emits_step_size_proxy = False

    def __init__(self, inner, n: int = 4):
        if int(n) < 1:
            raise ValueError(f"RepeatedIntegrator needs n >= 1 sub-step(s), got {n}")
        if getattr(inner, "n_rng_per_step", 0):
            raise ValueError(
                f"RepeatedIntegrator needs a deterministic inner integrator, but "
                f"{type(inner).__name__} asks for {inner.n_rng_per_step} random draw(s) per "
                f"step (there is no per-sub-step randomness to give it)")
        self.inner, self.n = inner, int(n)
        self.emits_step_size_proxy = bool(getattr(inner, "emits_step_size_proxy", False))

    def init_integrator_data(self) -> dict:
        return self.inner.init_integrator_data()

    def step(self, istate: IntegratorState, eps, ctx, rng=None) -> IntegratorState:
        sub = eps / self.n
        return jax.lax.fori_loop(0, self.n, lambda _, s: self.inner.step(s, sub, ctx), istate)

    def flow(self, istate, eps, ctx, use_cache=False) -> IntegratorState:
        # nested integrators manage their own caching, so ``use_cache`` is ignored (doc 06)
        return self.step(istate, eps, ctx)

    def integrate(self, istate: IntegratorState, eps, n_steps, ctx) -> IntegratorState:
        return _integrate_by_stepping(self, istate, eps, n_steps, ctx)


def multirate_leapfrog(cheap_potentials, expensive_potentials, kinetics,
                       n: int = 4) -> SplittingIntegrator:
    """Multi-rate (RESPA) leapfrog: one expensive gradient per ``n`` cheap sub-steps.

        ``V_exp/2   [ leapfrog(V_cheap, T) at eps/n ]^n   V_exp/2``

    The expensive potentials are kicked once per outer step of size ``eps``; the cheap ones (a
    prior, the chart Jacobian) and the *whole* kinetic part run ``n`` inner leapfrog steps of
    ``eps/n``. Reversible and second-order: the inner step is palindromic, hence symmetric, hence
    so is its ``n``-fold repetition, and the outer sequence is a palindrome around it.

    Cost per outer step is ``len(expensive) + n * len(cheap)`` gradients, against
    ``len(expensive) + len(cheap)`` for plain leapfrog at the same ``eps`` --- worth it exactly
    when the expensive gradient dominates the wall clock while the cheap part limits the stable
    step size. ``n = 1`` is legal and reproduces ``leapfrog(expensive + cheap, kinetics)`` op for
    op.

    **Cache validity** (the contract of ``docs/design/06``: a ``cached_gradient=True`` kick is
    valid only where no *drift* has changed ``q`` since that cache was written). The only ops
    that move ``q`` are the inner drifts, so:

    * a leading **expensive** kick reads a cache written by ``init_integrator_state`` (first
      step) or by the previous outer step's trailing expensive kick --- only kicks in between;
    * a leading **cheap** kick reads the previous inner iteration's trailing kick (nothing in
      between), and on the first inner iteration the previous outer step's last trailing cheap
      kick --- since when only the two expensive kick groups have run, which move only ``p``.

    Both caches are therefore fresh at every outer-step *endpoint*, which is what NUTS needs to
    resume from either end and what ``total_energy`` needs, since a potential's ``energy`` trusts
    its cached value unconditionally. (Mid-inner-loop the expensive cache is stale; nothing reads
    it there.) **All of this holds only because no kinetic appears in the outer list** --- putting
    one there would separate the trailing and leading expensive kicks by a drift and silently
    invalidate the annotation.
    """
    cheap, expensive = list(cheap_potentials), list(expensive_potentials)
    if int(n) < 1:
        raise ValueError(f"multirate_leapfrog needs n >= 1 sub-step(s), got {n}")
    if not cheap:
        raise ValueError(
            "multirate_leapfrog has no cheap potential to sub-step: the inner loop would repeat "
            "the drift alone. Use leapfrog(potentials, kinetics) instead")
    if not expensive:
        raise ValueError(
            "multirate_leapfrog has no expensive potential: with every potential cheap there is "
            "nothing to sub-step against. Use leapfrog(potentials, kinetics) instead")
    inner = leapfrog(cheap, kinetics)
    return SplittingIntegrator(
        [Op(V, 0.5, cached_gradient=True) for V in expensive]
        + [Op(RepeatedIntegrator(inner, n), 1.0)]
        + [Op(V, 0.5) for V in reversed(expensive)])


def init_integrator_state(potentials, q, p, ctx, log_weight=None) -> IntegratorState:
    """Build an :class:`IntegratorState`, seeding every potential's value/grad cache.

    Pre-populating the caches at ``q`` is what makes the leapfrog leading half-kicks
    (``cached_gradient=True``) valid on the very first step. ``log_weight`` starts at 0
    unless a starting weight is supplied.
    """
    values, grads = {}, {}
    for potential in potentials:
        v, g = potential.value_and_grad(q, ctx)
        values[potential.id] = v
        grads[potential.id] = g
    if log_weight is None:
        log_weight = jnp.zeros(())
    return IntegratorState(q=q, p=p, potential_values=values, potential_grads=grads,
                           log_weight=log_weight)
