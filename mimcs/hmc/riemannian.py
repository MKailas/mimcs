"""Riemannian Manifold HMC with a general (implicit) metric, as a block kinetic.

Implements ``docs/design/07_riemannian_hmc.md``, variant 1 (Girolami & Calderhead 2011): a
position-dependent metric with the generalized (implicit) leapfrog --- restricted to a **block** of
coordinates, so it composes with every other kinetic in ``BaseHMC``'s list exactly as the explicit
block metrics of :mod:`mimcs.hmc.block_riemannian` do.

For a block ``i`` (the kinetic's ``slices``, possibly fused and non-contiguous) the kinetic energy is

    T_i(q, p_i) = 1/2 p_i^T G_i(q)^{-1} p_i + 1/2 log|G_i(q)|,

where ``G_i`` may depend on **all** of ``q`` --- the block's own coordinates included, which is what
the explicit blocks rule out. ``T_i``'s flow therefore moves ``q_i`` and ``p_i`` implicitly, and
kicks every other momentum ``p_{-i}`` explicitly; ``q_{-i}`` never moves, since ``T_i`` does not
depend on ``p_{-i}``. :meth:`RiemannianKinetic.flow` is the generalized leapfrog of ``T_i`` over the
whole ``(q, p)``, with ``h = eps / 2``:

1. implicit block kick: ``p_i' = p_i - h grad_{q_i} T_i(q, p_i')``;
2. explicit dependency kick: ``p_{-i} -= h grad_{q_{-i}} T_i(q, p_i')`` (the same gradient);
3. implicit drift: ``q_i'' = q_i + h [G_i(q)^{-1} + G_i(q'')^{-1}] p_i'`` (``q''`` = ``q`` with
   ``q_i''``);
4. explicit kick of all momenta by ``-h grad_q T_i(q'', p_i')``.

It is symplectic and reversible at the exact fixed points, so it composes in the palindromic
``leapfrog`` like any other kinetic (the potentials stay ordinary explicit kicks, outside). When
``G_i`` does not depend on ``q_i`` both solves are exact after one evaluation, and the map *is* the
explicit block flow of :class:`~mimcs.hmc.block_riemannian.DiagonalBlock` --- the oracle the tests
hold it to.

**The metric** is given by a *pre-image* ``M(q)`` --- ``G`` itself for a given metric, the block
Hessian for :class:`HessianMetric` --- and stable functions of it (energy, velocity, momentum draw,
whitened residual norms). Step 1 holds ``q`` fixed, so the kinetic linearizes ``M`` **once** per
kick (``jax.vjp``) and each fixed-point iteration costs one pullback of ``dT/dM``: the forward
work (``k`` Hessian-vector products for a Hessian metric) is never repeated inside the solve. The
pullback is over the full ``q``, so its block part drives the fixed point and the rest *is* the
dependency kick. No metric-derivative calculus is hand-derived anywhere.

Three metrics:

* :class:`AnalyticMetric` --- ``fn(q) -> G`` over the flat coordinate (the original low-level form);
* :class:`CallableMetric` --- ``fn(coords) -> G`` with ``coords`` keyed by parameter name (what a
  factory ``BlockSpec(kind="riemannian", params={"metric": fn})`` builds);
* :class:`HessianMetric` --- the block Hessian of the target, eigenvalue-clamped
  (:mod:`mimcs.hmc.spectral`), with an adapted softness ``1/b`` in ``ham_params``.

**A solve that does not converge poisons the step with NaN.** The generalized leapfrog is only
reversible at its fixed points, so an unconverged step is not a valid proposal; a NaN phase point
has non-finite energy, which HMC rejects and NUTS flags as a divergence --- the step-size adaptation
then shrinks the step. The per-trajectory iteration and failure counts ride in ``integrator_data``
(``fp_iters`` / ``fp_failures``) into ``state.diagnostics``.
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
from .solvers import FixedPointSolver, SolveResult, resolve_solver
from .spectral import Clamp, resolve_clamp, sym_matfun, sym_tracefun

#: Cost of one Hessian-vector product of the target inside a ``k``-HVP block Hessian, in gradient
#: evaluations (wall clock). Measured on hierarchical logistic regressions (N = 200..20000 rows,
#: ``tests/experiments/rmhmc_cost_ratios.py``): 0.58--1.00 per HVP, the high end where dispatch
#: overhead dominates a small model. The top of the range is used, so the cost is never flattered.
HVP_COST = 1.0
#: Cost of pulling one cotangent back through one of those HVPs (third order), in gradient
#: evaluations. Measured alongside :data:`HVP_COST`: 0.63--1.32.
HVP_PULLBACK_COST = 1.3


# --- metric representation ------------------------------------------------------------------ #

class Metric:
    """A position-dependent SPD metric ``G(q)`` over one block, given through a pre-image ``M(q)``.

    ``pre`` evaluates ``M``; everything else is a function of ``M`` (and the metric's parameters,
    ``params`` = ``ctx.ham_params[kinetic.id]``). ``energy`` must be differentiable in ``M``; the
    others are only ever evaluated. ``prepare`` factors ``M`` once for the evaluated functions.
    """

    #: whether the metric has a softness to adapt (:class:`HessianMetric` only).
    adapts_softness = False

    def init_params(self):
        return None

    def pre(self, q: Array, ctx, block) -> Array:
        raise NotImplementedError

    def prepare(self, M: Array, params):
        raise NotImplementedError

    def energy(self, M: Array, p: Array, params) -> Array:
        """``T = 1/2 p^T G^{-1} p + 1/2 log|G|`` --- differentiable in ``M``."""
        raise NotImplementedError

    def velocity(self, F, p: Array) -> Array:
        """``G^{-1} p`` from the factor ``F``."""
        raise NotImplementedError

    def sample(self, F, z: Array) -> Array:
        """``S z`` with ``S S^T = G``, so ``p = S z ~ N(0, G)`` for ``z ~ N(0, I)``."""
        raise NotImplementedError

    def whiten_p(self, F, r: Array) -> Array:
        """A momentum-space vector in ``G^{-1/2}`` units (a draw of ``p`` becomes ``N(0, I)``)."""
        raise NotImplementedError

    def whiten_q(self, F, r: Array) -> Array:
        """A position-space vector in ``G^{1/2}`` units."""
        raise NotImplementedError

    def eval_cost(self, size: int) -> float:
        """Gradient-evaluation equivalents of one ``pre`` (0 for a metric that reads no model)."""
        return 0.0

    def pullback_cost(self, size: int) -> float:
        """Gradient-evaluation equivalents of one pullback through ``pre``."""
        return 0.0


class _MatrixMetric(Metric):
    """A metric whose pre-image is ``G`` itself: a dense SPD ``(k, k)`` or a positive ``(k,)``
    diagonal (the rank of ``M`` is static, so the branch is too). The dense form is factored by
    Cholesky; the diagonal one works in log space, as the explicit blocks do, so nothing forms
    ``G^{-2}`` under autodiff."""

    def prepare(self, M, params):
        if M.ndim == 1:
            return jnp.log(M)
        return jnp.linalg.cholesky(0.5 * (M + M.T))

    def energy(self, M, p, params):
        if M.ndim == 1:
            log_m = jnp.log(M)
            return 0.5 * jnp.sum((p * jnp.exp(-0.5 * log_m)) ** 2) + 0.5 * jnp.sum(log_m)
        L = jnp.linalg.cholesky(0.5 * (M + M.T))
        w = jax.scipy.linalg.solve_triangular(L, p, lower=True)
        return 0.5 * jnp.dot(w, w) + jnp.sum(jnp.log(jnp.diag(L)))

    def velocity(self, F, p):
        if F.ndim == 1:
            return p * jnp.exp(-F)
        return jax.scipy.linalg.cho_solve((F, True), p)

    def sample(self, F, z):
        if F.ndim == 1:
            return jnp.exp(0.5 * F) * z
        return F @ z

    def whiten_p(self, F, r):
        if F.ndim == 1:
            return r * jnp.exp(-0.5 * F)
        return jax.scipy.linalg.solve_triangular(F, r, lower=True)

    def whiten_q(self, F, r):
        if F.ndim == 1:
            return r * jnp.exp(0.5 * F)
        return F.T @ r


class AnalyticMetric(_MatrixMetric):
    """Wraps ``fn: q -> G`` over the **flat coordinate** --- the whole-space low-level form.

    ``G`` is the block's metric: ``(k, k)`` SPD, or ``(k,)`` positive for a diagonal one, with ``k``
    the block size (the whole coordinate for a whole-space kinetic)."""

    def __init__(self, fn: Callable[[Array], Array]):
        self._fn = fn

    def matrix(self, q, params=None):
        return self._fn(q)

    def pre(self, q, ctx, block):
        return jnp.asarray(self._fn(q), q.dtype)


class CallableMetric(_MatrixMetric):
    """Wraps ``fn(coords) -> G`` with ``coords`` keyed by parameter name --- the factory form.

    ``coords`` holds every continuous parameter's **coordinate-space** slice (flat), and, for a
    model with integer parameters, every discrete parameter's labels (a trajectory constant, read
    from ``ctx.discrete``). ``G`` is ``(k, k)`` SPD or ``(k,)`` positive (a diagonal metric), over
    the block's coordinates in the order of its slices; a scalar broadcasts to a diagonal."""

    def __init__(self, fn: Callable, model, size: int):
        self._fn = fn
        self.model = model
        self.size = int(size)
        self.__name__ = getattr(fn, "__name__", type(fn).__name__)

    def coords(self, q, labels) -> dict:
        m = self.model
        out = {}
        for p in m.parameters:
            s, e = m.coord_block(p.name)
            out[p.name] = q[s:e]
        if labels is not None and getattr(m, "discrete_dim", 0):
            for p in m.discrete_parameters:
                s, e = m.discrete_block(p.name)
                out[p.name] = labels[s:e]
        return out

    def pre(self, q, ctx, block):
        G = jnp.asarray(self._fn(self.coords(q, getattr(ctx, "discrete", None))), q.dtype)
        if G.ndim == 0 or (G.ndim == 1 and G.shape[0] == 1 and self.size != 1):
            G = jnp.broadcast_to(G.reshape(()), (self.size,))
        return G


class HessianMetric(Metric):
    """The eigenvalue-clamped block Hessian of the target: ``G = Q f(Lambda) Q^T``.

    ``H = Q Lambda Q^T`` is the Hessian of ``V = sum(potentials)`` --- the coordinate-space target,
    the chart Jacobian included --- restricted to the block's coordinates, at the full current
    ``q`` (so it depends on every other block too). ``f(lambda) = phi(b lambda) / b`` for a clamp
    ``phi`` (:data:`mimcs.hmc.spectral.CLAMPS`): ``b lambda >> 1`` leaves a positive curvature
    essentially intact, and the band ``b |lambda| <~ 1`` is reshaped into a positive mass. The
    softness ``1/b`` lives in ``ham_params[id]`` as ``{"log_softness": log(1/b)}`` --- traced, so
    adapting it (:class:`~mimcs.adaptation.HessianSoftnessAdaptation`) never retraces the kernel.

    ``H`` costs ``k`` Hessian-vector products (``jacfwd`` over the block of the full gradient), so
    this is for low-dimensional blocks --- a fused block of hyperparameters is the intended use.

    The default clamp is ``softabs``: on targets whose block Hessian goes indefinite (Rosenbrock,
    centered eight schools) it was measured better than ``softplus`` on every paired seed, and equal
    where the curvature stays positive (``docs/design/07``).
    Its derivative runs through :func:`~mimcs.hmc.spectral.sym_matfun` /
    :func:`~mimcs.hmc.spectral.sym_tracefun`, which stay finite at repeated eigenvalues.
    """

    def __init__(self, potentials, clamp="softabs", softness: float = 1.0,
                 adapt_softness: bool = True, softness_quantile: float = 0.1,
                 softness_ratio: float = 3.0):
        if not potentials:
            raise ValueError("HessianMetric needs the target's potentials to differentiate")
        if not softness > 0:
            raise ValueError(f"softness (1/b) must be positive, got {softness}")
        if not 0.0 < softness_quantile < 1.0:
            raise ValueError(f"softness_quantile must be in (0, 1), got {softness_quantile}")
        if not softness_ratio > 0:
            raise ValueError(f"softness_ratio must be positive, got {softness_ratio}")
        self.potentials = list(potentials)
        self.clamp: Clamp = resolve_clamp(clamp)
        self.softness = float(softness)
        self.adapts_softness = bool(adapt_softness)
        self.softness_quantile = float(softness_quantile)
        self.softness_ratio = float(softness_ratio)
        self.__name__ = f"hessian({self.clamp.name})"

    def init_params(self):
        return {"log_softness": jnp.log(jnp.asarray(self.softness, float))}

    def _b(self, params):
        return jnp.exp(-params["log_softness"])

    def potential(self, q, ctx):
        return sum(pot.potential(q, ctx) for pot in self.potentials)

    def pre(self, q, ctx, block):
        grad_v = jax.grad(self.potential)

        def block_grad(xb):
            return block._gather(grad_v(block._scatter(q, xb), ctx))

        H = jax.jacfwd(block_grad)(block._gather(q))
        return 0.5 * (H + H.T)

    def prepare(self, M, params):
        b = self._b(params)
        lam, Q = jnp.linalg.eigh(M)
        return Q, self.clamp.log_phi(b * lam) - jnp.log(b)       # (Q, log f(lambda))

    def energy(self, M, p, params):
        b = self._b(params)
        clamp = self.clamp
        quad = b * jnp.dot(p, sym_matfun(b * M, clamp.inv_phi) @ p)      # p^T G^{-1} p
        logdet = sym_tracefun(b * M, clamp.log_phi) - M.shape[0] * jnp.log(b)
        return 0.5 * (quad + logdet)

    def velocity(self, F, p):
        Q, log_f = F
        return Q @ (jnp.exp(-log_f) * (Q.T @ p))

    def sample(self, F, z):
        Q, log_f = F
        return Q @ (jnp.exp(0.5 * log_f) * z)

    def whiten_p(self, F, r):
        Q, log_f = F
        return jnp.exp(-0.5 * log_f) * (Q.T @ r)

    def whiten_q(self, F, r):
        Q, log_f = F
        return jnp.exp(0.5 * log_f) * (Q.T @ r)

    def spectrum(self, q, ctx, block) -> Array:
        """The block Hessian's eigenvalues at ``q`` (what the softness adaptation reads)."""
        return jnp.linalg.eigvalsh(self.pre(q, ctx, block))

    def eval_cost(self, size):
        return HVP_COST * size

    def pullback_cost(self, size):
        return HVP_PULLBACK_COST * size


# --- the block kinetic ---------------------------------------------------------------------- #

def _max_abs(x):
    return jnp.max(jnp.abs(x))


class RiemannianKinetic(KineticHamiltonian):
    """``T_i = 1/2 p_i^T G_i(q)^{-1} p_i + 1/2 log|G_i(q)|`` over a block, integrated implicitly.

    ``slices`` is the block (``None``: the whole coordinate); ``G_i`` may depend on all of ``q``.
    ``flow`` is the block generalized leapfrog of the module docstring, solved by ``solver`` (a
    :class:`~mimcs.hmc.FixedPointSolver`; default Anderson to ``sqrt(eps)``). The metric's
    parameters (a :class:`HessianMetric`'s softness) live in ``ctx.ham_params[id]``.
    """

    separable = False
    mass_mode = None       # no quadratic-mass adaptation touches it

    def __init__(self, metric: Metric, solver: FixedPointSolver | None = None,
                 id: str = "T", slices=None):
        self.metric = metric
        self.solver = solver if solver is not None else resolve_solver()
        self.id = id
        self.slices = None if slices is None else [tuple(map(int, s)) for s in slices]

    @property
    def adapts_softness(self) -> bool:
        return bool(getattr(self.metric, "adapts_softness", False))

    # --- metric plumbing ---

    def _params(self, ctx):
        return ctx.ham_params.get(self.id) if ctx.ham_params else None

    def _pre_fn(self, ctx):
        return lambda qq: self.metric.pre(qq, ctx, self)

    def _grad_T(self, M, pullback, p_i, params):
        """``grad_q T_i`` over the **full** ``q``, from a linearization ``(M, pullback)`` at ``q``."""
        cot = jax.grad(lambda MM: self.metric.energy(MM, p_i, params))(M)
        return pullback(cot)[0]

    def spectrum(self, q, ctx) -> Array:
        return self.metric.spectrum(q, ctx, self)

    # --- component interface ---

    def energy(self, istate, ctx):
        M = self.metric.pre(istate.q, ctx, self)
        return self.metric.energy(M, self._gather(istate.p), self._params(ctx))

    def velocity_into(self, v, istate, ctx):
        params = self._params(ctx)
        F = self.metric.prepare(self.metric.pre(istate.q, ctx, self), params)
        return self._scatter(v, self.metric.velocity(F, self._gather(istate.p)))

    def make_draw_components(self, dim):
        return [DrawComponent(f"{self.id}_momentum", (self._size(dim),), jax.random.normal)]

    def sample_into(self, p, draw, q, ctx):
        params = self._params(ctx)
        F = self.metric.prepare(self.metric.pre(q, ctx, self), params)
        z = getattr(draw, f"{self.id}_momentum")
        return self._scatter(p, self.metric.sample(F, z))

    def initial_mass_params(self, dim):
        return self.metric.init_params()

    def integrator_data_schema(self) -> dict:
        """The counters this kinetic accumulates into ``integrator_data`` (see
        :meth:`~mimcs.hmc.SplittingIntegrator.init_integrator_data`)."""
        z = jnp.zeros(())
        return {"fp_iters": z, "fp_failures": z}

    def flow(self, istate, eps, ctx, use_cache=False):
        """The block generalized leapfrog of ``T_i`` (module docstring); potentials are kicked
        outside, by the surrounding splitting."""
        metric, params = self.metric, self._params(ctx)
        h = 0.5 * eps
        q, p = istate.q, istate.p
        k = self._size(q.shape[0])
        pre = self._pre_fn(ctx)

        # 1. implicit block kick, at fixed q: linearize M once, one pullback per iteration.
        M0, pull0 = jax.vjp(pre, q)
        F0 = metric.prepare(M0, params)
        p_i = self._gather(p)
        kick = self.solver.solve(
            lambda x: p_i - h * self._gather(self._grad_T(M0, pull0, x, params)), p_i,
            norm=lambda r: _max_abs(metric.whiten_p(F0, r)))
        p_half = kick.x
        # 2. dependency kick with the same gradient; the block keeps its fixed point.
        p = self._scatter(p - h * self._grad_T(M0, pull0, p_half, params), p_half)

        # 3. implicit drift of q_i, from the explicit-Euler guess.
        v0 = metric.velocity(F0, p_half)
        q_i = self._gather(q)

        def drift(x):
            F = metric.prepare(pre(self._scatter(q, x)), params)
            return q_i + h * (v0 + metric.velocity(F, p_half))

        move = self.solver.solve(drift, q_i + 2.0 * h * v0,
                                 norm=lambda r: _max_abs(metric.whiten_q(F0, r)))
        q_new = self._scatter(q, move.x)

        # 4. explicit kick of every momentum at the new position.
        M1, pull1 = jax.vjp(pre, q_new)
        p = p - h * self._grad_T(M1, pull1, p_half, params)

        ok = kick.converged & move.converged
        q_new = jnp.where(ok, q_new, jnp.nan)
        p = jnp.where(ok, p, jnp.nan)
        return istate._replace(q=q_new, p=p,
                               integrator_data=self._count(istate.integrator_data, k, kick, move,
                                                           ok))

    def _count(self, data: dict, k: int, kick: SolveResult, move: SolveResult, ok) -> dict:
        """Accumulate this flow's cost and solver counters, for whichever keys the trajectory was
        seeded with (a Python-static check, so the carry structure never changes)."""
        if not any(key in data for key in ("grad_evals", "fp_iters", "fp_failures")):
            return data
        data = dict(data)
        if "grad_evals" in data:
            m = self.metric
            # two linearizations (+ the drift's evaluations) and kick+2 pullbacks; one more
            # evaluation for the energy the sampler reads at the new point.
            evals = 3.0 + move.n_iter.astype(float)
            pulls = kick.n_iter.astype(float) + 2.0
            data["grad_evals"] = (data["grad_evals"] + evals * m.eval_cost(k)
                                  + pulls * m.pullback_cost(k))
        if "fp_iters" in data:
            data["fp_iters"] = data["fp_iters"] + (kick.n_iter + move.n_iter).astype(float)
        if "fp_failures" in data:
            data["fp_failures"] = data["fp_failures"] + (~ok).astype(float)
        return data


# --- the sampler ------------------------------------------------------------------------------ #

class RMHMC(HMC):
    """Fixed-length Riemannian Manifold HMC with a whole-space general (implicit) metric.

    A convenience: ``HMC`` with one whole-space :class:`RiemannianKinetic` and the ordinary
    ``leapfrog`` (the implicit work lives in the kinetic's ``flow``). Block-restricted metrics are
    built by listing ``RiemannianKinetic(..., slices=...)`` among ``BaseHMC``'s kinetics --- or from
    the factory, ``BlockSpec(kind="riemannian")``.

    Args:
        metric: a :class:`Metric` (e.g. ``AnalyticMetric(lambda q: ...)``).
        solver: a :class:`FixedPointSolver`, a name (``"anderson"`` default / ``"picard"``), or
            ``None``.
        max_iter: the solver's iteration cap when ``solver`` is a name or ``None``.
        n_leapfrog, step_size, ...: as for :class:`HMC`.
    """

    def __init__(self, model, init_position, *, metric: Metric, solver=None,
                 max_iter: int | None = None, n_leapfrog: int = 20, **kwargs):
        opts = {} if max_iter is None else {"max_iter": int(max_iter)}
        kinetic = RiemannianKinetic(metric, solver=resolve_solver(solver, **opts))
        potentials = default_potentials(model)
        integrator = leapfrog(potentials, kinetic)
        super().__init__(model, init_position, n_leapfrog=n_leapfrog,
                         potentials=potentials, kinetic=kinetic, integrator=integrator,
                         **kwargs)
