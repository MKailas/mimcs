"""Test problems: a ``Model`` bundled with an exact reference sampler when available.

A :class:`TargetProblem` packages everything a correctness test needs: the
:class:`~mimcs.model.Model` the sampler runs on, an optional exact i.i.d. sampler that
serves as analytic ground truth, optional closed-form moments, and labels for plots.

Three problems are provided:

* :func:`correlated_gaussian` --- exact, easy; full correctness checks expected.
* :func:`rosenbrock` --- the "banana"; factorizes so it can be sampled exactly, and
  its curvature exercises the shape (energy-distance) check.
* :func:`neal_funnel` --- exact reference available but deliberately hard for
  random-walk samplers (the varying-scale neck). Marked ``hard`` so distributional
  assertions are skipped; kept for graphical inspection and later HMC/NUTS tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import jax.numpy as jnp

from ..model import (
    Model, EuclideanParameter, BoundedParameter, PositiveParameter, IntervalParameter,
    UnitVectorParameter, SimplexParameter, OrderedParameter, CovMatrixParameter)

ExactSampler = Callable[[int, np.random.Generator], np.ndarray]


@dataclass
class TargetProblem:
    name: str
    model: Model
    dim: int
    labels: list[str]
    exact_sampler: ExactSampler | None = None
    mean: np.ndarray | None = None
    cov: np.ndarray | None = None
    hard: bool = False

    @property
    def has_reference(self) -> bool:
        return self.exact_sampler is not None

    def exact_sample(self, n: int, seed: int = 0) -> np.ndarray:
        if self.exact_sampler is None:
            raise ValueError(f"problem '{self.name}' has no exact reference sampler")
        return np.asarray(self.exact_sampler(n, np.random.default_rng(seed)), float)


def correlated_gaussian(mean=(1.0, -2.0), cov=((2.0, 1.4), (1.4, 1.5))) -> TargetProblem:
    mean = np.asarray(mean, float)
    cov = np.asarray(cov, float)
    d = mean.shape[0]
    precision = jnp.asarray(np.linalg.inv(cov))
    jmean = jnp.asarray(mean)

    def log_post(params):
        delta = params["x"] - jmean
        return -0.5 * delta @ precision @ delta

    model = Model([EuclideanParameter("x", (d,))], {"log_post": log_post})
    chol = np.linalg.cholesky(cov)

    def sampler(n, rng):
        return mean + rng.standard_normal((n, d)) @ chol.T

    return TargetProblem(
        name="correlated_gaussian", model=model, dim=d,
        labels=[f"x{i}" for i in range(d)],
        exact_sampler=sampler, mean=mean, cov=cov,
    )


def rosenbrock(a: float = 1.0, b: float = 5.0) -> TargetProblem:
    """2D Rosenbrock "banana": ``-((a - x0)^2 + b (x1 - x0^2)^2)``.

    Factorizes as ``x0 ~ N(a, 1/2)`` and ``x1 | x0 ~ N(x0^2, 1/(2b))``, giving an
    exact sampler and closed-form moments.
    """
    a = float(a)
    b = float(b)

    def log_post(params):
        x = params["x"]
        return -((a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2)

    model = Model([EuclideanParameter("x", (2,))], {"log_post": log_post})

    s0 = np.sqrt(0.5)            # sd of x0
    s1 = np.sqrt(1.0 / (2 * b))  # conditional sd of x1

    def sampler(n, rng):
        x0 = a + s0 * rng.standard_normal(n)
        x1 = x0**2 + s1 * rng.standard_normal(n)
        return np.column_stack([x0, x1])

    # Closed-form moments (x0 ~ N(a, 1/2)):
    #   E[x1] = E[x0^2] = 1/2 + a^2
    #   Var[x1] = Var(x0^2) + 1/(2b) = (1/2 + 2 a^2) + 1/(2b)
    #   Cov(x0, x1) = Cov(x0, x0^2) = a
    mean = np.array([a, 0.5 + a**2])
    var_x1 = 0.5 + 2.0 * a**2 + 1.0 / (2.0 * b)
    cov = np.array([[0.5, a], [a, var_x1]])

    return TargetProblem(
        name="rosenbrock", model=model, dim=2, labels=["x0", "x1"],
        exact_sampler=sampler, mean=mean, cov=cov,
    )


def neal_funnel(dim: int = 2, scale: float = 3.0) -> TargetProblem:
    """Neal's funnel: ``v ~ N(0, scale^2)``, ``x_i | v ~ N(0, e^v)``.

    Hard for random-walk samplers because the conditional scale of ``x`` varies by
    orders of magnitude with ``v``. Coordinate 0 is ``v``; the rest are ``x``.
    """
    nx = dim - 1

    def log_post(params):
        z = params["x"]
        v = z[0]
        x = z[1:]
        lp_v = -0.5 * (v / scale) ** 2
        lp_x = -0.5 * jnp.sum(x**2 * jnp.exp(-v)) - 0.5 * nx * v
        return lp_v + lp_x

    model = Model([EuclideanParameter("x", (dim,))], {"log_post": log_post})

    def sampler(n, rng):
        v = scale * rng.standard_normal(n)
        x = rng.standard_normal((n, nx)) * np.exp(v / 2.0)[:, None]
        return np.column_stack([v, x])

    mean = np.zeros(dim)
    cov = np.diag([scale**2] + [float(np.exp(scale**2 / 2.0))] * nx)
    labels = ["v"] + [f"x{i}" for i in range(nx)]

    return TargetProblem(
        name="neal_funnel", model=model, dim=dim, labels=labels,
        exact_sampler=sampler, mean=mean, cov=cov, hard=True,
    )


def neal_funnel_blocks(dim: int = 2, scale: float = 3.0) -> TargetProblem:
    """Neal's funnel with ``v`` and ``x`` as *separate* parameters (blocks).

    Same target as :func:`neal_funnel`, but the model exposes two parameters -- ``v``
    (scalar) and ``x`` (the rest) -- so the explicit block RMHMC can put a metric on ``x``
    that depends on ``v``. The flat sample is ``[v, x_0, ...]``, identical to
    ``neal_funnel``'s, so the same exact reference applies.
    """
    nx = dim - 1

    def log_prior_v(params):
        return -0.5 * (params["v"] / scale) ** 2

    def log_lik_x(params):
        v, x = params["v"], params["x"]
        return -0.5 * jnp.sum(x**2 * jnp.exp(-v)) - 0.5 * nx * v

    model = Model([EuclideanParameter("v", ()), EuclideanParameter("x", (nx,))],
                  {"log_prior_v": log_prior_v, "log_lik_x": log_lik_x})

    def sampler(n, rng):
        v = scale * rng.standard_normal(n)
        x = rng.standard_normal((n, nx)) * np.exp(v / 2.0)[:, None]
        return np.column_stack([v, x])

    mean = np.zeros(dim)
    cov = np.diag([scale**2] + [float(np.exp(scale**2 / 2.0))] * nx)
    labels = ["v"] + [f"x{i}" for i in range(nx)]

    return TargetProblem(
        name="neal_funnel_blocks", model=model, dim=dim, labels=labels,
        exact_sampler=sampler, mean=mean, cov=cov, hard=True,
    )


def neal_funnel_vector(n: int = 30, scale: float = 3.0) -> TargetProblem:
    """A vector funnel: ``n`` independent funnels sharing a per-element scale parameter.

    Parameters ``s`` and ``x`` are both ``n``-dimensional, with ``s_j ~ N(0, scale^2)`` and
    ``x_j | s_j ~ N(0, exp(s_j))`` --- so ``s_j`` is the (log-)variance of ``x_j`` alone, a
    bijective row correspondence like a horseshoe's per-element scale. The ideal conditional
    metric for ``x`` is ``E[g_x,j^2 | s] = exp(-s_j)`` (elementwise), i.e. the *sparse*
    ``SpExp("s")`` with weight ``-1``, bias ``0`` --- the sparse analogue of the funnel's
    ``Exp("v")``. The flat sample is ``[s_0..s_{n-1}, x_0..x_{n-1}]``.
    """
    def log_prior_s(params):
        return -0.5 * jnp.sum((params["s"] / scale) ** 2)

    def log_lik_x(params):
        s, x = params["s"], params["x"]
        return -0.5 * jnp.sum(x**2 * jnp.exp(-s)) - 0.5 * jnp.sum(s)

    model = Model([EuclideanParameter("s", (n,)), EuclideanParameter("x", (n,))],
                  {"log_prior_s": log_prior_s, "log_lik_x": log_lik_x})

    def sampler(m, rng):
        s = scale * rng.standard_normal((m, n))
        x = rng.standard_normal((m, n)) * np.exp(s / 2.0)
        return np.column_stack([s, x])

    mean = np.zeros(2 * n)
    cov = np.diag([scale**2] * n + [float(np.exp(scale**2 / 2.0))] * n)
    labels = [f"s{i}" for i in range(n)] + [f"x{i}" for i in range(n)]

    return TargetProblem(
        name="neal_funnel_vector", model=model, dim=2 * n, labels=labels,
        exact_sampler=sampler, mean=mean, cov=cov, hard=True,
    )


# --- problems that exercise constrained / parent-dependent parameters -------- #


def positive_lognormal(sigma: float = 1.0) -> TargetProblem:
    """``s ~ LogNormal(0, sigma)`` via a :func:`PositiveParameter` (log chart).

    The log chart's Jacobian combines with the log-normal density so the coordinate
    ``q = log s`` is exactly ``N(0, sigma^2)``: a clean correctness check of the
    positive-parameter chart and its Jacobian on a heavy-tailed target.
    """
    sigma = float(sigma)

    def log_post(params):
        ls = jnp.log(params["s"])
        return -0.5 * (ls / sigma) ** 2 - jnp.log(params["s"])  # LogNormal(0, sigma)

    model = Model([PositiveParameter("s")], {"lp": log_post})

    def sampler(n, rng):
        return np.exp(sigma * rng.standard_normal((n, 1)))

    return TargetProblem(name="positive_lognormal", model=model, dim=1, labels=["s"],
                         exact_sampler=sampler)


def uniform_interval(lower: float = -2.0, upper: float = 3.0) -> TargetProblem:
    """``x ~ Uniform(lower, upper)`` via an :func:`IntervalParameter` (logit chart)."""
    lower, upper = float(lower), float(upper)

    def log_post(params):
        return jnp.zeros(())  # uniform: constant density on the interval

    model = Model([IntervalParameter("x", lower, upper)], {"lp": log_post})

    def sampler(n, rng):
        return rng.uniform(lower, upper, (n, 1))

    return TargetProblem(name="uniform_interval", model=model, dim=1, labels=["x"],
                         exact_sampler=sampler)


def nested_uniform() -> TargetProblem:
    """Parent-dependent bound: ``a ~ U(0, 1)``, ``b ~ U(0, a)``.

    ``b``'s upper bound is the value of its parent ``a``. The conditional density of
    ``b`` is ``1/a`` (so ``log_post = -log a``), which exactly cancels the ``log a``
    in ``b``'s chart Jacobian, leaving ``q_a, q_b`` independent standard-logistic ---
    a stringent test that the parent-dependent Jacobian is correct.
    """
    a = BoundedParameter("a", lower=0.0, upper=1.0)
    b = BoundedParameter("b", lower=0.0, upper="a")   # upper bound = parent 'a'

    def log_post(params):
        return -jnp.log(params["a"])   # density of Uniform(0, a) is 1/a

    model = Model([a, b], {"lp": log_post})

    def sampler(n, rng):
        av = rng.uniform(0.0, 1.0, n)
        bv = av * rng.uniform(0.0, 1.0, n)
        return np.column_stack([av, bv])

    return TargetProblem(name="nested_uniform", model=model, dim=2, labels=["a", "b"],
                         exact_sampler=sampler)


def block_gaussian(cov_a=((2.0, 1.5), (1.5, 2.0)), var_b=(0.5, 3.0, 1.2),
                   mean_a=(1.0, -1.0), mean_b=(0.0, 2.0, -3.0)) -> TargetProblem:
    """Two *independent* Gaussian parameter blocks: a correlated block ``a`` and a diagonal
    block ``b``. The ideal mass is block-diagonal --- dense within ``a``, diagonal within
    ``b``. Exercises a block-diagonal constant mass (a list of Diagonal/Dense kinetics)."""
    cov_a = np.asarray(cov_a, float); var_b = np.asarray(var_b, float)
    mean_a = np.asarray(mean_a, float); mean_b = np.asarray(mean_b, float)
    da, db = mean_a.shape[0], mean_b.shape[0]
    Pa = jnp.asarray(np.linalg.inv(cov_a)); Pb = jnp.asarray(1.0 / var_b)
    ma, mb = jnp.asarray(mean_a), jnp.asarray(mean_b)

    def log_post(params):
        ra, rb = params["a"] - ma, params["b"] - mb
        return -0.5 * ra @ Pa @ ra - 0.5 * jnp.sum(Pb * rb ** 2)

    model = Model([EuclideanParameter("a", (da,)), EuclideanParameter("b", (db,))],
                  {"log_post": log_post})

    mean = np.concatenate([mean_a, mean_b])
    cov = np.zeros((da + db, da + db))
    cov[:da, :da] = cov_a
    cov[da:, da:] = np.diag(var_b)
    chol = np.linalg.cholesky(cov)

    def sampler(n, rng):
        return mean + rng.standard_normal((n, da + db)) @ chol.T

    labels = [f"a{i}" for i in range(da)] + [f"b{i}" for i in range(db)]
    return TargetProblem(name="block_gaussian", model=model, dim=da + db, labels=labels,
                         exact_sampler=sampler, mean=mean, cov=cov)


# --- unit vectors on the sphere --------------------------------------------- #


def _frame_to(mu: np.ndarray) -> np.ndarray:
    """An orthogonal matrix whose last column is ``mu`` (a Householder reflection)."""
    d = mu.shape[0]
    e_d = np.zeros(d); e_d[-1] = 1.0
    v = mu - e_d
    nv = np.linalg.norm(v)
    if nv < 1e-12:
        return np.eye(d)
    v = v / nv
    return np.eye(d) - 2.0 * np.outer(v, v)


def _vmf_draw(n, rng, mu, kappa):
    """``n`` exact i.i.d. draws from ``vMF(mu, kappa)`` on **S^2** (shape ``(n, 3)``).

    On S^2 the cosine ``t = <x, mu>`` has density proportional to ``e^{kappa t}`` on
    ``[-1, 1]``, so inverse-CDF sampling is closed form; the written form
    ``t = 1 + log(u + (1-u) e^{-2 kappa}) / kappa`` keeps that stable for large ``kappa``.
    The transverse direction is uniform on the circle.
    """
    u = rng.random(n)
    t = 1.0 + np.log(u + (1.0 - u) * np.exp(-2.0 * kappa)) / kappa
    phi = 2.0 * np.pi * rng.random(n)
    r = np.sqrt(np.maximum(0.0, 1.0 - t ** 2))
    local = np.column_stack([r * np.cos(phi), r * np.sin(phi), t])   # frame with mu as e_3
    return local @ _frame_to(mu).T


def _vmf_mean_resultant(kappa: float) -> float:
    """``A(kappa) = coth(kappa) - 1/kappa``: the mean resultant length of a vMF on S^2."""
    return float(1.0 / np.tanh(kappa) - 1.0 / kappa)


def von_mises_fisher(kappa: float = 5.0, mu=(0.0, 0.0, 1.0), *,
                     adaptive: bool = True) -> TargetProblem:
    """``x ~ vMF(mu, kappa)`` on **S^2** via a :class:`UnitVectorParameter`.

    The canonical directional target: density ``exp(kappa <x, mu>)`` with respect to the
    surface measure, so the chart's Jacobian is what makes the coordinate-space sampler
    correct. Ground truth is analytic --- ``E[x] = A(kappa) mu`` --- and the exact sampler is
    closed form (see :func:`_vmf_draw`).

    ``kappa`` is deliberately moderate by default. A vMF's *ambient* covariance is full rank
    with condition number growing like ``kappa`` (the spread along ``mu`` is O(1/kappa^2)
    against O(1/kappa) transverse), so a very concentrated target would approach
    rank-deficiency in R^3 and strain the whitening in :mod:`mimcs.testing.comparison`.
    """
    mu = np.asarray(mu, float)
    mu = mu / np.linalg.norm(mu)
    kappa = float(kappa)
    jmu = jnp.asarray(mu)

    def log_post(params):
        return kappa * jnp.dot(params["x"], jmu)

    model = Model([UnitVectorParameter("x", 3, adaptive=adaptive)], {"log_post": log_post})

    def sampler(n, rng):
        return _vmf_draw(n, rng, mu, kappa)

    return TargetProblem(
        name="von_mises_fisher", model=model, dim=3,
        labels=["x0", "x1", "x2"], exact_sampler=sampler,
        mean=_vmf_mean_resultant(kappa) * mu)


def uniform_sphere(d: int = 3, *, adaptive: bool = True) -> TargetProblem:
    """``x ~ Uniform(S^(d-1))`` via a :class:`UnitVectorParameter`.

    A constant density on the sphere, so the chart's Jacobian carries the *entire* target:
    if ``log_jacobian_det`` were wrong the draws would not come out uniform. It is also the
    worst case for placing the pole --- no direction is emptier than another, so the mean
    resultant length is ~0 and the adaptation's pole guard should hold the pole put.
    """
    d = int(d)

    def log_post(params):
        return jnp.zeros(())

    model = Model([UnitVectorParameter("x", d, adaptive=adaptive)], {"log_post": log_post})

    def sampler(n, rng):
        x = rng.standard_normal((n, d))
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    return TargetProblem(
        name="uniform_sphere", model=model, dim=d,
        labels=[f"x{i}" for i in range(d)], exact_sampler=sampler,
        mean=np.zeros(d), cov=np.eye(d) / d)


def unit_vector_array(mus=((0.0, 0.0, 1.0), (1.0, 0.0, 0.0)), kappas=(4.0, 6.0), *,
                      adaptive: bool = True) -> TargetProblem:
    """``n`` independent vMFs as a single ``array[n] unit_vector[3]`` parameter.

    One parameter of ambient shape ``(n, 3)`` and coordinate dimension ``n * 2``, so each unit
    vector carries its own chart hyperparameters. Distinct ``mus`` and ``kappas`` mean a single
    shared chart could not fit them all --- the per-vector poles and scales have to be adapted
    independently.
    """
    mus = np.asarray(mus, float)
    mus = mus / np.linalg.norm(mus, axis=1, keepdims=True)
    kappas = np.asarray(kappas, float)
    n = mus.shape[0]
    jmus, jkappas = jnp.asarray(mus), jnp.asarray(kappas)

    def log_post(params):
        return jnp.sum(jkappas * jnp.sum(params["x"] * jmus, axis=-1))

    model = Model([UnitVectorParameter("x", 3, (n,), adaptive=adaptive)], {"log_post": log_post})

    def sampler(n_draws, rng):
        parts = [_vmf_draw(n_draws, rng, mus[i], float(kappas[i])) for i in range(n)]
        return np.stack(parts, axis=1).reshape(n_draws, n * 3)

    mean = np.stack([_vmf_mean_resultant(float(kappas[i])) * mus[i] for i in range(n)])
    labels = [f"x{i}{j}" for i in range(n) for j in range(3)]
    return TargetProblem(
        name="unit_vector_array", model=model, dim=n * 3, labels=labels,
        exact_sampler=sampler, mean=mean.reshape(-1))


# --- simplex and ordered ----------------------------------------------------- #


def dirichlet_simplex(alpha=(2.0, 3.0, 4.0)) -> TargetProblem:
    """``x ~ Dirichlet(alpha)`` via a :class:`SimplexParameter`.

    The conjugate workhorse of the simplex, and exactly samplable. Every ``alpha_k > 1`` keeps
    the density zero on the faces, which is what the Stein diagnostic's boundary assumption
    needs (see :meth:`~mimcs.model.SimplexParameter.stein_terms`); it also keeps the target away
    from the corners, where the stick-breaking chart stretches hardest.
    """
    alpha = np.asarray(alpha, float)
    d = alpha.shape[0]
    jalpha = jnp.asarray(alpha)

    def log_post(params):
        return jnp.sum((jalpha - 1.0) * jnp.log(params["x"]))

    model = Model([SimplexParameter("x", d)], {"log_post": log_post})

    def sampler(n, rng):
        return rng.dirichlet(alpha, size=n)

    a0 = alpha.sum()
    mean = alpha / a0
    cov = (np.diag(mean) - np.outer(mean, mean)) / (a0 + 1.0)
    return TargetProblem(
        name="dirichlet_simplex", model=model, dim=d,
        labels=[f"x{i}" for i in range(d)], exact_sampler=sampler, mean=mean, cov=cov)


def wishart_cov(nu: float = 8.0, scale=None) -> TargetProblem:
    """``Sigma ~ Wishart(nu, V)`` via a :class:`CovMatrixParameter`.

    The conjugate workhorse of covariance matrices, and exactly samplable: Bartlett's
    decomposition draws the *Cholesky factor* directly, which is the chart's own coordinates one
    step on, so the reference costs no MCMC. ``nu > K + 1`` keeps the density vanishing at the
    singular boundary; unlike the simplex the Stein diagnostic needs no such assumption (the
    boundary is at infinite distance in the affine-invariant geometry), but a proper density is
    still what makes the target well behaved for a sampler.

    The exact moments are ``E[Sigma] = nu V`` and
    ``Cov(Sigma_ij, Sigma_kl) = nu (V_ik V_jl + V_il V_jk)``.
    """
    V = np.asarray([[2.0, 0.6, 0.1], [0.6, 1.4, -0.3], [0.1, -0.3, 1.0]]
                   if scale is None else scale, float)
    K = V.shape[0]
    if nu <= K + 1:
        raise ValueError(f"wishart_cov needs nu > K + 1 = {K + 1} for a proper density, got {nu}")
    vinv = jnp.asarray(np.linalg.inv(V))

    def log_post(params):
        sigma = params["S"]
        return (nu - K - 1) / 2 * jnp.linalg.slogdet(sigma)[1] - 0.5 * jnp.trace(vinv @ sigma)

    model = Model([CovMatrixParameter("S", K)], {"log_post": log_post})

    def sampler(n, rng):
        chol_v = np.linalg.cholesky(V)
        A = np.zeros((n, K, K))
        for i in range(K):
            A[:, i, i] = np.sqrt(rng.chisquare(nu - i, size=n))
            for j in range(i):
                A[:, i, j] = rng.standard_normal(n)
        L = chol_v @ A
        return (L @ np.transpose(L, (0, 2, 1))).reshape(n, K * K)

    mean = (nu * V).reshape(K * K)
    cov = nu * (np.einsum("ik,jl->ijkl", V, V) + np.einsum("il,jk->ijkl", V, V))
    return TargetProblem(
        name="wishart_cov", model=model, dim=K * K,
        labels=[f"S[{i},{j}]" for i in range(K) for j in range(K)],
        exact_sampler=sampler, mean=mean, cov=cov.reshape(K * K, K * K))


def ordered_normal(d: int = 4, mu: float = 0.0, sigma: float = 1.0) -> TargetProblem:
    """The order statistics of ``d`` i.i.d. ``N(mu, sigma)``, via an unbounded ordered vector.

    The model's density is just the i.i.d. normal one --- the *ordering* is carried entirely by
    the chart, so the draws come out distributed as the sorted sample if and only if the chart
    and its Jacobian are right. Sorting ``d`` i.i.d. normals gives the exact reference.
    """
    d = int(d)

    def log_post(params):
        z = (params["x"] - mu) / sigma
        return -0.5 * jnp.sum(z ** 2)

    model = Model([OrderedParameter("x", d)], {"log_post": log_post})

    def sampler(n, rng):
        return np.sort(mu + sigma * rng.standard_normal((n, d)), axis=1)

    return TargetProblem(
        name="ordered_normal", model=model, dim=d,
        labels=[f"x{i}" for i in range(d)], exact_sampler=sampler)


def ordered_uniform(d: int = 4, lower: float = 0.0, upper: float = 1.0) -> TargetProblem:
    """The order statistics of ``d`` i.i.d. ``Uniform(lower, upper)``, via a bounded ordered vector.

    The ordered analogue of :func:`uniform_sphere`: a *constant* density, so the chart's
    Jacobian carries the entire target. If the doubly-bounded stick-breaking log-Jacobian were
    wrong, the draws would not come out as sorted uniforms. The ``k``-th order statistic has
    mean ``lower + k (upper-lower) / (d+1)`` --- exactly the evenly spaced point the coordinate
    origin maps to.
    """
    d = int(d)

    def log_post(params):
        return jnp.zeros(())

    model = Model([OrderedParameter("x", d, lower=lower, upper=upper)], {"log_post": log_post})

    def sampler(n, rng):
        return np.sort(rng.uniform(lower, upper, size=(n, d)), axis=1)

    k = np.arange(1, d + 1)
    mean = lower + k * (upper - lower) / (d + 1.0)
    return TargetProblem(
        name="ordered_uniform", model=model, dim=d,
        labels=[f"x{i}" for i in range(d)], exact_sampler=sampler, mean=mean)
