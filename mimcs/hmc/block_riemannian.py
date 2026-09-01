"""Explicit (block / hierarchical) Riemannian Manifold HMC.

Implements ``docs/design/07_riemannian_hmc.md`` variant 2 (Kleppe; Kailas-Vihola-Wallin).
The coordinates are partitioned into blocks (the ``Model``'s parameters), with a
block-diagonal metric ``G(q) = blockdiag(M_1(q_-1), ..., M_k(q_-k))`` whose block ``i``
depends only on *other* blocks' positions ``q_{-i}`` (declared via ``depends_on``), never
on ``q_i`` itself.

That constraint makes each block term ``T_i = 1/2 p_i^T M_i(q_{-i})^{-1} p_i +
1/2 log|M_i(q_{-i})|`` *explicitly* integrable: its flow holds ``q_{-i}`` and ``p_i``
fixed (so ``M_i`` is constant), drifting ``q_i`` linearly and kicking the dependency
momenta ``p_{-i}`` by a constant force. So no implicit solve is needed: the full
integrator is an ordinary palindromic ``SplittingIntegrator`` of the potential kicks and
the per-block flows. The metric-derivative kick is obtained by autodiff of ``T_i`` (no
hand-derived terms), exactly as in the implicit variant.

Metrics are **diagonal** and come in two flavours:

* :class:`BlockMetric` --- an explicitly given analytic function of the dependency
  coordinates (or a constant identity).
* a *learned* metric given by a :class:`~mimcs.hmc.metric_expr.MetricExpr` from the
  mass-matrix mini-language (``Exp("v") + Exp()``, ``Exp()*Sigmoid("v","x") + Exp()``, ...),
  whose parameters are adapted online by SGD on a KL objective (see
  :mod:`mimcs.adaptation.metric`).

The learned parameters live in ``state.ham_params[kinetic.id]`` as ``{block_name: pytree}``
and are read back through the :class:`~mimcs.hmc.state.HamiltonianContext` during
integration --- so the same machinery serves given and learned metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import jax.scipy.linalg
from jax import Array

from ..rng import DrawComponent
from .hamiltonians import KineticHamiltonian
from .metric_encode import encode_discrete, encoded_width
from .metric_expr import MetricExpr
from . import lowrank


@dataclass(frozen=True)
class BlockMetric:
    """Given diagonal metric for one block.

    Args:
        depends_on: names of the (other) blocks this block's metric depends on.
        fn: ``fn(deps) -> diagonal`` where ``deps`` is ``{name: coordinate slice}`` for the
            ``depends_on`` blocks and the result is the diagonal of ``M_i`` (a vector of the
            block size, or a scalar / length-1 array broadcast to it). ``None`` means a
            constant identity metric (``M_i = I``).
    """

    depends_on: tuple = ()
    fn: Callable | None = None


# --- blocks: one explicit kinetic T_i per coordinate block ------------------- #


class _DiagBlock(KineticHamiltonian):
    """A diagonal (possibly position-dependent) block kinetic ``T_i`` --- a slice-aware
    :class:`~mimcs.hmc.KineticHamiltonian` component in ``BaseHMC``'s kinetics list.

    Subclasses provide ``_mass(q, labels, params) -> diagonal vector``; ``labels`` is the model's
    flat discrete block (``ctx.discrete``, ``None`` for a continuous model) --- a metric may depend
    on integer parameters as well as continuous ones (doc 14), and a label is a **trajectory
    constant**, so it enters the metric exactly as it enters the density. ``params`` is this
    block's
    ``ctx.ham_params[self.id]`` (``None`` for a block with no learned parameters). The block's
    ``id`` is its parameter name (unique across the list). Non-separable (``M_i`` may vary with
    ``q_{-i}``), so it overrides ``flow`` with the explicit block flow (drift ``q_i``, kick the
    dependency momenta).
    """

    separable = False
    mass_mode = None
    depends: bool = False       # whether M_i varies with q (drives the metric-derivative kick)

    def _mass(self, q: Array, labels, params) -> Array:
        raise NotImplementedError

    # params plumbing -------------------------------------------------------- #

    def _params(self, ctx):
        return ctx.ham_params.get(self.id)

    def initial_mass_params(self, dim):
        return None

    def make_draw_components(self, dim):
        return [DrawComponent(f"{self.id}_momentum", (self.size,), jax.random.normal)]

    # energy / velocity / momentum refresh ----------------------------------- #
    #
    # The block metric enters only through three overridable primitives, so the explicit flow
    # (drift + autodiff kick) and the KineticHamiltonian interface are metric-shape agnostic: a
    # diagonal block implements them from ``_mass``; a shaped block (D(x)^1/2 A D(x)^1/2) overrides
    # them with the dense / low-rank algebra. ``_velocity`` is ``M(q)^{-1} p_i``; ``_energy`` is
    # ``T_i = 1/2 p_i^T M^{-1} p_i + 1/2 log det M``; ``_sample_factor`` applies an ``S`` with
    # ``S S^T = M(q)`` to ``z ~ N(0, I)`` so ``p_i = S z`` has covariance ``M``.

    def _velocity(self, q: Array, labels, p_i: Array, params) -> Array:
        return p_i / self._mass(q, labels, params)

    def _energy(self, q: Array, labels, p_i: Array, params) -> Array:
        M = self._mass(q, labels, params)
        return 0.5 * jnp.sum(p_i ** 2 / M) + 0.5 * jnp.sum(jnp.log(M))

    def _sample_factor(self, q: Array, labels, z: Array, params) -> Array:
        return jnp.sqrt(self._mass(q, labels, params)) * z   # p_i = sqrt(M_i) z ~ N(0, M_i)

    def _labels(self, ctx):
        """The model's discrete block from the context --- ``None`` for a continuous model.

        Read here rather than closed over, for the reason ``HamiltonianContext.discrete`` exists
        at all: a jitted reseed would otherwise bake in the first call's labels."""
        return getattr(ctx, "discrete", None)

    def energy(self, istate, ctx) -> Array:
        return self._energy(istate.q, self._labels(ctx), istate.p[self.s:self.e],
                            self._params(ctx))

    def velocity_into(self, v: Array, istate, ctx) -> Array:
        return v.at[self.s:self.e].set(
            self._velocity(istate.q, self._labels(ctx), istate.p[self.s:self.e],
                           self._params(ctx)))

    def sample_into(self, p: Array, draw, q: Array, ctx) -> Array:
        z = getattr(draw, f"{self.id}_momentum")
        return p.at[self.s:self.e].set(
            self._sample_factor(q, self._labels(ctx), z, self._params(ctx)))

    def flow(self, istate, eps, ctx, use_cache=False):
        """Explicit flow of ``T_i``: drift ``q_i``, kick the dependency momenta.

        ``q_{-i}`` and ``p_i`` are constant during this flow (``M_i`` is constant), so both
        updates are closed-form. The kick ``-eps * d/dq_{-i} T_i`` is taken by autodiff.
        """
        params = self._params(ctx)
        labels = self._labels(ctx)
        q, p = istate.q, istate.p
        p_i = p[self.s:self.e]
        q_new = q.at[self.s:self.e].add(
            eps * self._velocity(q, labels, p_i, params))                       # drift q_i
        if self.depends:
            # `labels` is closed over as a constant, which is exactly right: a label does not move
            # along a trajectory, so `M = f(labels) * g(q)` has `dM/dq = f(labels) * dg/dq` --- the
            # discrete factor scales the kick rather than contributing one of its own.
            grad = jax.grad(lambda qq: self._energy(qq, labels, p_i, params))(q)  # d/dq T_i
            p_new = p - eps * grad
        else:
            p_new = p
        return istate._replace(q=q_new, p=p_new)


class DiagonalBlock(_DiagBlock):
    """A block whose diagonal metric is a given function of the dependency coordinates."""

    def __init__(self, name: str, coord_slice: tuple[int, int],
                 depends: list[tuple[str, tuple[int, int]]], fn: Callable | None):
        self.name = name
        self.id = name
        self.s, self.e = coord_slice
        self.size = self.e - self.s
        self.slices = [coord_slice]       # a block RMHMC block is a single (contiguous) parameter
        self._dep_slices = depends        # [(dep_name, (start, stop)), ...]
        self.depends = bool(depends) and fn is not None
        self.fn = fn

    def _mass(self, q: Array, labels, params) -> Array:
        if self.fn is None:
            return jnp.ones(self.size)
        deps = {name: q[s:e] for name, (s, e) in self._dep_slices}
        return jnp.broadcast_to(jnp.asarray(self.fn(deps)), (self.size,))


class LearnedDiagonalBlock(_DiagBlock):
    """A block whose diagonal metric is a learned mini-language expression over other blocks.

    ``M_i = expr.evaluate(params, {dep: q[dep_slices]})`` for a
    :class:`~mimcs.hmc.metric_expr.MetricExpr`. Its parameters live in ``ctx.ham_params[self.id]``
    and are adapted by :class:`mimcs.adaptation.MetricAdaptation` (SGD on :meth:`metric_loss`).
    Dependency blocks may be fused/non-contiguous, so each is gathered from its slice list."""

    is_learned = True

    def __init__(self, name: str, coord_slice: tuple[int, int], expr: MetricExpr,
                 dep_slices: dict[str, list[tuple[int, int]]], init=None, discrete_deps=None):
        self.name = name
        self.id = name
        self.s, self.e = coord_slice
        self.size = self.e - self.s
        self.slices = [coord_slice]       # a block RMHMC block is a single (contiguous) parameter
        self.expr = expr
        self._dep_slices = dep_slices     # {dep_name: [(start, stop), ...]}
        self._discrete_deps = dict(discrete_deps or {})   # {name: (start, stop, kind, lo, hi)}
        # Only a **continuous** dependency makes M vary along a trajectory, and this flag is what
        # gates the metric-derivative kick in `flow`. A metric depending only on labels is constant
        # over the trajectory, so it needs no kick --- skipping it there is an optimisation. Leaving
        # this False while a continuous dependency exists would be the real bug: that kick is a term
        # of the dynamics, not a refinement.
        self.depends = bool(expr.deps())
        self._init = init                 # optional pre-fitted parameters (e.g. factory regression)

    def initial_mass_params(self, dim):
        return self._init if self._init is not None else self.init_params()

    def metric_loss(self, params, q, labels, score):
        """This block's KL objective ``1/2 sum_d (log M_i[d] + g_i[d]^2 / M_i[d])`` (its per-
        sample minimiser is ``M_i = g_i^2``, expectation the conditional gradient 2nd moment).

        ``labels`` conditions the metric on the model's discrete parameters, making the fitted
        quantity ``E[g_i^2 | q_{-i}, z]`` rather than its average over ``z`` (doc 14)."""
        M = self._mass(q, labels, params)
        g = score[self.s:self.e]
        return 0.5 * jnp.sum(jnp.log(M) + g ** 2 / M)

    def _gather(self, q: Array, slices: list[tuple[int, int]]) -> Array:
        return jnp.concatenate([q[s:e] for s, e in slices])

    def _dep_dims(self) -> dict:
        """Each dependency's width as the expression sees it --- for a discrete one, the width of
        its **encoded** design columns, not its label count."""
        dims = {d: sum(e - s for s, e in sl) for d, sl in self._dep_slices.items()}
        for d, (lo_i, hi_i, kind, lo, hi) in self._discrete_deps.items():
            dims[d] = encoded_width(hi_i - lo_i, kind, lo, hi)
        return dims

    def init_params(self) -> dict:
        """Initialise the expression's parameters (weights zero, ``M_i`` ~ ``I`` at init)."""
        return self.expr.init_params(self.size, self._dep_dims())

    def _dep_coords(self, q: Array, labels) -> dict:
        """Every dependency as a real feature vector: continuous coordinates gathered from ``q``,
        discrete labels put through :func:`~mimcs.hmc.metric_encode.encode_discrete`."""
        out = {d: self._gather(q, sl) for d, sl in self._dep_slices.items()}
        for d, (lo_i, hi_i, kind, lo, hi) in self._discrete_deps.items():
            out[d] = encode_discrete(labels[lo_i:hi_i], kind, lo, hi)
        return out

    def _mass(self, q: Array, labels, params) -> Array:
        return self.expr.evaluate(params, self._dep_coords(q, labels))


class ShapedLearnedBlock(_DiagBlock):
    """A position-dependent diagonal times a **constant** shape: ``M(x) = D(x)^{1/2} A D(x)^{1/2}``.

    ``D(x)`` is a learned mini-language expression over other blocks (as in
    :class:`LearnedDiagonalBlock`); ``A`` is a *constant* correlation shape --- **dense**
    ``A = K K^T`` (``K`` lower-Cholesky) or **low-rank** ``A = I + sum_j gamma_j v_j v_j^T``
    (``gamma_j >= 0``, ``v_j`` unit directions). Only the cheap diagonal whitening varies with
    position; the parameter-heavy shape is constant. Whitening by ``D(x)`` makes ``A`` a
    correlation matrix (unit diagonal), so both shape forms stay well conditioned. Adapted by
    :class:`mimcs.adaptation.ShapedMetricAdaptation` --- ``D(x)`` by the diagonal metric KL,
    ``A`` by the existing dense / low-rank score adapters on the ``D(x)^{-1/2}``-whitened score.

    ``ctx.ham_params[self.id]`` is ``{"diag": <expr params>, "shape": <K>  or  <(W, gamma)>}`` with
    ``W`` the ``(size, J)`` eigen-directions and ``gamma`` the ``(J,)`` stiffenings. ``M(x)`` is
    formed as ``L(x) = diag(sqrt D) K`` (dense; ``M = L L^T``) or ``diag(D) + V^T V`` with
    ``V[j] = sqrt(gamma_j) sqrt(D) v_j`` (low-rank; :mod:`mimcs.hmc.lowrank`)."""

    is_shaped = True

    def __init__(self, name: str, coord_slice: tuple[int, int], expr: MetricExpr,
                 dep_slices: dict[str, list[tuple[int, int]]], shape, init=None,
                 discrete_deps=None):
        self.name = name
        self.id = name
        self.s, self.e = coord_slice
        self.size = self.e - self.s
        self.slices = [coord_slice]
        self.expr = expr
        self._dep_slices = dep_slices
        self._discrete_deps = dict(discrete_deps or {})
        self.depends = bool(expr.deps())    # D(x) varies with q_{-i} (drives the metric kick)
        self._init = init
        if shape == "dense":
            self.shape_kind, self.rank = "dense", None
        elif isinstance(shape, (tuple, list)) and len(shape) == 2 and shape[0] == "lowrank":
            self.shape_kind, self.rank = "lowrank", int(shape[1])
        else:
            raise ValueError(f"unknown shape {shape!r} (use 'dense' or ('lowrank', J))")

    # D(x), as LearnedDiagonalBlock -------------------------------------------- #

    def _gather(self, q: Array, slices: list[tuple[int, int]]) -> Array:
        return jnp.concatenate([q[s:e] for s, e in slices])

    _dep_dims = LearnedDiagonalBlock._dep_dims
    _dep_coords = LearnedDiagonalBlock._dep_coords

    def _D(self, q: Array, labels, diag_params) -> Array:
        return self.expr.evaluate(diag_params, self._dep_coords(q, labels))

    def metric_loss(self, params, q, labels, score):
        """Diagonal KL over ``D(x)`` only (the shape ``A`` captures the residual correlation)."""
        M = self._D(q, labels, params["diag"])
        g = score[self.s:self.e]
        return 0.5 * jnp.sum(jnp.log(M) + g ** 2 / M)

    def init_params(self) -> dict:
        diag = self.expr.init_params(self.size, self._dep_dims())
        shape = (jnp.eye(self.size) if self.shape_kind == "dense"
                 else (jnp.zeros((self.size, self.rank)), jnp.zeros((self.rank,))))   # A = I at init
        return {"diag": diag, "shape": shape}

    def initial_mass_params(self, dim):
        if self._init is None:
            return self.init_params()
        if isinstance(self._init, dict) and "diag" in self._init:
            return self._init                          # already a full {"diag", "shape"} pytree
        # a bare D(x) warm-start (the factory's fitted diagonal metric): pair it with A = I.
        return {"diag": self._init, "shape": self.init_params()["shape"]}

    # metric primitives (override _DiagBlock) ---------------------------------- #

    def _lowrank_V(self, D: Array, params) -> Array:
        W, gamma = params["shape"]                       # W: (size, J); gamma: (J,)
        return jnp.sqrt(gamma)[:, None] * (W.T * jnp.sqrt(D)[None, :])    # (J, size)

    def _velocity(self, q, labels, p_i, params):
        D = self._D(q, labels, params["diag"])
        if self.shape_kind == "dense":
            L = jnp.sqrt(D)[:, None] * params["shape"]                    # M = L L^T
            w = jax.scipy.linalg.solve_triangular(L, p_i, lower=True)     # L^{-1} p
            return jax.scipy.linalg.solve_triangular(L.T, w, lower=False)  # L^{-T} w = M^{-1} p
        return lowrank.apply_inv(D, self._lowrank_V(D, params), p_i)

    def _energy(self, q, labels, p_i, params):
        D = self._D(q, labels, params["diag"])
        if self.shape_kind == "dense":
            L = jnp.sqrt(D)[:, None] * params["shape"]
            w = jax.scipy.linalg.solve_triangular(L, p_i, lower=True)
            return 0.5 * jnp.dot(w, w) + jnp.sum(jnp.log(jnp.abs(jnp.diag(L))))
        V = self._lowrank_V(D, params)
        return 0.5 * jnp.dot(p_i, lowrank.apply_inv(D, V, p_i)) + 0.5 * lowrank.log_det(D, V)

    def _sample_factor(self, q, labels, z, params):
        D = self._D(q, labels, params["diag"])
        if self.shape_kind == "dense":
            return jnp.sqrt(D) * (params["shape"] @ z)       # L z, Cov = L L^T = M
        return lowrank.apply_chol(D, self._lowrank_V(D, params), z)   # S z, S S^T = M


# --- build one block kinetic per parameter ---------------------------------- #


def _resolve_discrete_deps(model, spec) -> dict:
    """``{name: (start, stop, kind, lower, upper)}`` for a metric's discrete dependencies.

    The bounds come from the declared support, which is what makes the ordinal transform
    deterministic and evidence-free (:mod:`mimcs.hmc.metric_encode`). Resolved against
    ``discrete_block``, the *parallel* layout --- a discrete parameter is not in ``coord_block``
    at all, which is why ``deps()`` and ``discrete_deps()`` are separate accessors.
    """
    names = getattr(spec, "discrete_deps", lambda: set())()
    if not names:
        return {}
    by_name = {p.name: p for p in getattr(model, "discrete_parameters", ())}
    out = {}
    for d in sorted(names):
        p = by_name.get(d)
        if p is None:
            raise ValueError(
                f"metric depends on {d!r} as a discrete parameter, but this model has no such "
                f"discrete parameter (it has "
                f"{sorted(by_name) or 'none'}). A continuous dependency is named positionally, "
                f"a discrete one through categorical=/ordinal=.")
        lo_i, hi_i = model.discrete_block(d)
        out[d] = (lo_i, hi_i, spec.dep_kind(d), int(p.lower_value), int(p.upper_value))
    return out


def _resolve_dep(model, dep_name: str, block_name: str) -> list[tuple[int, int]]:
    """Coordinate slices for a dependency-block name; a fused name (``x__y``) splits on ``__``."""
    parts = dep_name.split("__")
    if block_name in parts:
        raise ValueError(f"block '{block_name}' metric cannot depend on itself")
    return [model.coord_block(p) for p in parts]


def build_block(model, name: str, spec, init=None, shape=None):
    """Build the block kinetic ``T_i`` for parameter ``name`` from a metric ``spec``.

    ``spec`` is ``None`` (constant identity), a :class:`BlockMetric` (given diagonal), or a
    :class:`~mimcs.hmc.metric_expr.MetricExpr` (mini-language, adapted). ``init`` optionally seeds a
    learned block's parameters (e.g. the factory's fitted regression). ``shape`` upgrades a
    mini-language metric to a **shaped** (nondiagonal) one ``D(x)^{1/2} A D(x)^{1/2}``: ``"dense"``
    for ``A = K K^T`` or ``("lowrank", J)`` for ``A = I + sum_j gamma_j v_j v_j^T`` (see
    :class:`ShapedLearnedBlock`); ``None`` (default) keeps the plain diagonal metric. A block's
    metric must not depend on its own block. Explicit block RMHMC is then just ``BaseHMC`` with the
    list of these block kinetics -- each an ordinary slice-aware
    :class:`~mimcs.hmc.KineticHamiltonian` component, composed by the unified ``leapfrog``.
    """
    if spec is None:
        spec = BlockMetric()
    if isinstance(spec, BlockMetric):
        if shape is not None:
            raise ValueError("shape= applies to a mini-language (MetricExpr) metric, not BlockMetric")
        if name in spec.depends_on:
            raise ValueError(f"block '{name}' metric cannot depend on itself")
        depends = [(d, model.coord_block(d)) for d in spec.depends_on]
        return DiagonalBlock(name, model.coord_block(name), depends, spec.fn)
    if isinstance(spec, MetricExpr):
        dep_slices = {d: _resolve_dep(model, d, name) for d in spec.deps()}
        discrete_deps = _resolve_discrete_deps(model, spec)
        if shape is not None:
            return ShapedLearnedBlock(name, model.coord_block(name), spec, dep_slices, shape,
                                      init=init, discrete_deps=discrete_deps)
        return LearnedDiagonalBlock(name, model.coord_block(name), spec, dep_slices, init=init,
                                    discrete_deps=discrete_deps)
    raise TypeError(f"unknown metric spec for block '{name}': {type(spec)}")


def build_blocks(model, metrics: dict | None = None) -> list:
    """One block kinetic per model parameter (``metrics`` maps names to specs)."""
    metrics = metrics or {}
    return [build_block(model, p.name, metrics.get(p.name)) for p in model.parameters]
