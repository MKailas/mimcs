"""Riemannian Manifold HMC, implicit variant (Girolami & Calderhead 2011).

**[experimental]** Correct and useful as the reference implementation, but not reachable from the
sampler factory and not recommended as a default. Two reasons: the factory-facing interface for
supplying ``G(q)`` by hand still needs design (the metric is a bare callable here, with no way to
express it in a spec), and the implicit steps rest on a fixed number of fixed-point iterations
whose residual is not checked (see the note below). The **explicit** block-Riemannian path
(:mod:`mimcs.hmc.block_riemannian` with the :mod:`mimcs.hmc.metric_expr` mini-language) is the
supported one and is fully factory-integrated.

Implements ``docs/design/07_riemannian_hmc.md``, variant 1: a general position-dependent
metric ``G(q)`` with the generalized (implicit) leapfrog integrator.

The augmented target is ``pi(q, p) ~ pi(q) N(p; 0, G(q))``, with non-separable kinetic
energy ``T(q, p) = 1/2 p^T G(q)^{-1} p + 1/2 log|G(q)|``. Rather than hand-derive the
metric-derivative terms of ``d/dq T`` (the trace and ``p^T G^{-1}(dG) G^{-1} p`` terms),
we let ``jax.grad`` differentiate ``T`` through ``G(q)`` directly -- arbitrary metrics
plug in with no manual calculus.

The metric is supplied as a callable ``q -> SPD matrix`` (``AnalyticMetric``), which is
enough to reproduce the classical algorithm and to test against. Learned/adaptive metrics
(the conditional gradient covariance of Kailas-Vihola-Wallin) are a later addition; the
``Metric`` interface leaves room for parameters in ``ham_params``.

NOTE (future work): the implicit steps use a fixed number of *naive* (Picard) fixed-point
iterations. This is the part most worth experimenting on for stability -- Newton iteration
(cheap via autodiff in low dimension) and Anderson acceleration are the natural next steps.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array

from ..rng import DrawComponent
from .hamiltonians import KineticHamiltonian
from .integrators import leapfrog
from .samplers import HMC, default_potentials
from .solvers import FixedPointSolver, PicardSolver


# --- metric representation --------------------------------------------------- #

class Metric:
    """A position-dependent SPD matrix ``G(q)`` (optionally with learnable params)."""

    def init_params(self):
        return None

    def matrix(self, q: Array, params) -> Array:
        raise NotImplementedError


class AnalyticMetric(Metric):
    """Wraps a user-supplied ``fn: q -> SPD matrix`` (e.g. a hand-derived Fisher metric)."""

    def __init__(self, fn: Callable[[Array], Array]):
        self._fn = fn

    def matrix(self, q, params=None):
        return self._fn(q)


# --- non-separable Riemannian kinetic energy --------------------------------- #

class RiemannianKinetic(KineticHamiltonian):
    """``T(q, p) = 1/2 p^T G(q)^{-1} p + 1/2 log|G(q)|`` for a general metric ``G(q)``.

    Non-separable (``separable = False``): ``T``'s flow moves both ``q`` and ``p``, so it
    is not an explicit drift. Instead ``flow`` integrates ``T``'s own (implicit) flow by a
    generalized leapfrog *of T alone* -- the potentials are handled by the surrounding
    ``SplittingIntegrator`` as ordinary explicit kicks. Because ``grad_q V`` is constant
    within the implicit solve (V depends only on q), this V/T splitting yields the *same*
    integrator as the classical monolithic generalized leapfrog, while reusing the
    standard ``leapfrog`` composition. All ``q``-derivatives of ``T`` come from autodiff.
    """

    separable = False

    def __init__(self, metric: Metric, solver: FixedPointSolver | None = None,
                 id: str = "T"):
        self.metric = metric
        self.solver = solver if solver is not None else PicardSolver()
        self.id = id

    def _G(self, q, ctx) -> Array:
        params = ctx.ham_params.get(self.id) if ctx.ham_params else None
        return self.metric.matrix(q, params)

    def energy_qp(self, q, p, ctx) -> Array:
        G = self._G(q, ctx)
        _, logdet = jnp.linalg.slogdet(G)
        return 0.5 * jnp.dot(p, jnp.linalg.solve(G, p)) + 0.5 * logdet

    def velocity_qp(self, q, p, ctx) -> Array:
        return jnp.linalg.solve(self._G(q, ctx), p)        # grad_p T = G(q)^{-1} p

    def grad_q(self, q, p, ctx) -> Array:
        return jax.grad(lambda qq: self.energy_qp(qq, p, ctx))(q)   # grad_q T, via autodiff

    # --- component interface (whole-space; delegates to the q,p forms) ---

    def energy(self, istate, ctx):
        return self.energy_qp(istate.q, istate.p, ctx)

    def velocity_into(self, v, istate, ctx):
        return self._scatter(v, self.velocity_qp(istate.q, istate.p, ctx))

    def flow(self, istate, eps, ctx, use_cache=False):
        """Implicit generalized-leapfrog flow of ``T`` alone (a kick--drift--kick on T).

        Solves the two implicit half-steps with ``self.solver``; the potential kicks are
        composed outside this flow by the surrounding splitting integrator.
        """
        q, p = istate.q, istate.p
        half = 0.5 * eps
        # implicit momentum half-kick (T only)
        p_half = self.solver.solve(lambda p_: p - half * self.grad_q(q, p_, ctx), p)
        # implicit position drift
        vel_q = self.velocity_qp(q, p_half, ctx)
        q_new = self.solver.solve(
            lambda q_: q + half * (vel_q + self.velocity_qp(q_, p_half, ctx)), q)
        # explicit momentum half-kick (T only) at the new position
        p_new = p_half - half * self.grad_q(q_new, p_half, ctx)
        return istate._replace(q=q_new, p=p_new)

    def make_draw_components(self, dim):
        return [DrawComponent(f"{self.id}_momentum", (self._size(dim),), jax.random.normal)]

    def sample_into(self, p, draw, q, ctx):
        # p ~ N(0, G(q)):  p = L z with G = L L^T
        L = jnp.linalg.cholesky(self._G(q, ctx))
        z = getattr(draw, f"{self.id}_momentum")
        return self._scatter(p, L @ z)

    def initial_mass_params(self, dim):
        return self.metric.init_params()


# --- the sampler ------------------------------------------------------------- #

class RMHMC(HMC):
    """Fixed-length Riemannian Manifold HMC with a general (implicit) metric.

    No dedicated integrator is needed: the implicit work lives in
    ``RiemannianKinetic.flow``, and the sampler uses the ordinary ``leapfrog`` splitting
    of the potentials (explicit kicks) and the kinetic (implicit T-flow). This is the
    same integrator as the classical monolithic generalized leapfrog (V's gradient is
    constant inside the implicit solve), now unified with the rest of the HMC machinery.

    Args:
        metric: a :class:`Metric` (e.g. ``AnalyticMetric(lambda q: ...)``).
        n_fixed_point: iterations for the default Picard solver.
        solver: a :class:`FixedPointSolver` for the implicit steps (default Picard).
        n_leapfrog, step_size, ...: as for :class:`HMC`.
    """

    def __init__(self, model, init_position, *, metric: Metric, n_fixed_point: int = 8,
                 solver: FixedPointSolver | None = None, n_leapfrog: int = 20, **kwargs):
        solver_obj = solver if solver is not None else PicardSolver(n_fixed_point)
        kinetic = RiemannianKinetic(metric, solver=solver_obj)
        potentials = default_potentials(model)
        integrator = leapfrog(potentials, kinetic)
        super().__init__(model, init_position, n_leapfrog=n_leapfrog,
                         potentials=potentials, kinetic=kinetic, integrator=integrator,
                         **kwargs)
