"""Tests for the implicit Riemannian **block** kinetic (:class:`mimcs.hmc.RiemannianKinetic`).

The kinetic integrates ``T_i = 1/2 p_i^T G_i(q)^{-1} p_i + 1/2 log|G_i(q)|`` over a block by the
block generalized leapfrog (``mimcs/hmc/riemannian.py``). Its correctness rests on four properties,
each checked here in x64 with a tight solver tolerance, and each with a control:

* the full leapfrog step is **symplectic** (``J^T Omega J = Omega``) and **reversible**, for a given
  metric that depends on the block's own coordinates and for the clamped Hessian metric over a
  fused, non-contiguous block;
* when ``G_i`` does not depend on ``q_i`` the map **equals the explicit block flow** of
  :class:`~mimcs.hmc.DiagonalBlock` (control: one that does, differs);
* a whole-space kinetic reproduces an independent **monolithic generalized leapfrog** of the full
  Hamiltonian (control: a single fixed-point sweep does not);
* the Hessian metric with a vanishing softness on a Gaussian **equals the dense quadratic kinetic
  with ``M`` = the precision** (control: a large softness does not).

Then the plumbing: an unsolvable step is NaN (a divergence) and counted; the counters reach
``state.diagnostics`` under HMC and NUTS; NUTS and SimpleNUTS stay bit-identical with an implicit
kinetic; discrete labels reach a given metric; and the softness adaptation tracks its quantile.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from mimcs import config
from mimcs.adaptation import HessianSoftnessAdaptation, RobbinsMonroStepSize
from mimcs.hmc import (
    HMC, NUTS, SimpleNUTS, CallableMetric, DenseQuadraticKinetic, DiagonalBlock,
    DiagonalQuadraticKinetic, HamiltonianContext, HessianMetric, AnalyticMetric,
    RiemannianKinetic, default_potentials, init_integrator_state, leapfrog)
from mimcs.hmc.solvers import AndersonSolver, NewtonSolver, PicardSolver
from mimcs.model import EuclideanParameter, IntegerParameter, Model
from mimcs.samplers import make_sampler_class
from mimcs.testing import block_gaussian, neal_funnel, neal_funnel_blocks

TIGHT = dict(tol=1e-13, max_iter=200)


@pytest.fixture
def x64():
    was = config.x64_enabled()
    config.enable_x64(True)
    try:
        yield
    finally:
        config.enable_x64(was)


def _stepper(model, kinetics, eps=0.2):
    """One full leapfrog step ``(q, p) -> (q', p', integrator_data)``."""
    pots = default_potentials(model)
    integ = leapfrog(pots, kinetics)
    ham = {k.id: k.initial_mass_params(model.coord_dim) for k in kinetics}
    ctx = HamiltonianContext(model.init_chart_hyperparams(), model.init_chart_indices(), ham)

    def step(q, p, e=eps):
        st = init_integrator_state(pots, q, p, ctx)._replace(
            integrator_data=integ.init_integrator_data())
        out = integ.step(st, e, ctx)
        return out.q, out.p, out.integrator_data

    return step


def _funnel_kinetics(model, metric_x, solver=None):
    sv, sx = model.coord_block("v"), model.coord_block("x")
    return [DiagonalQuadraticKinetic(id="v", slices=[sv]),
            RiemannianKinetic(CallableMetric(metric_x, model, sx[1] - sx[0]),
                              solver=solver or AndersonSolver(**TIGHT), id="x", slices=[sx])]


def _assert_symplectic_and_reversible(step, q0, p0):
    n = q0.shape[0]
    flat = lambda z: jnp.concatenate(step(z[:n], z[n:])[:2])
    J = jax.jacfwd(flat)(jnp.concatenate([q0, p0]))
    Om = jnp.block([[jnp.zeros((n, n)), jnp.eye(n)], [-jnp.eye(n), jnp.zeros((n, n))]])
    assert float(jnp.max(jnp.abs(J.T @ Om @ J - Om))) < 1e-10
    q1, p1, _ = step(q0, p0)
    assert float(jnp.max(jnp.abs(q1 - q0))) > 1e-2            # it actually moved
    q2, p2, _ = step(q1, -p1)
    assert float(jnp.max(jnp.abs(q2 - q0))) < 1e-10
    assert float(jnp.max(jnp.abs(p2 + p0))) < 1e-10


Q0 = [0.7, 0.3, -0.5, 1.1]
P0 = [0.4, -0.8, 0.6, 0.2]


@pytest.mark.parametrize("solver", [AndersonSolver, NewtonSolver])
def test_block_flow_is_symplectic_and_reversible_with_an_own_block_metric(x64, solver):
    m = neal_funnel_blocks(dim=4).model
    own = lambda c: jnp.exp(-c["v"]) * (1.0 + 0.3 * c["x"] ** 2)      # depends on x itself
    step = _stepper(m, _funnel_kinetics(m, own, solver=solver(**TIGHT)))
    _assert_symplectic_and_reversible(step, jnp.asarray(Q0), jnp.asarray(P0))
    _, _, data = step(jnp.asarray(Q0), jnp.asarray(P0))
    assert float(data["fp_iters"]) > 2                           # genuinely implicit
    assert float(data["fp_failures"]) == 0


@pytest.mark.parametrize("clamp,solver", [("softplus", AndersonSolver), ("softabs", AndersonSolver),
                                          ("softabs", NewtonSolver)])
def test_hessian_block_flow_is_symplectic_and_reversible_on_a_fused_block(x64, clamp, solver):
    """The Hessian metric over the non-contiguous block ``{v, x_1}`` of a funnel, the rest on a
    constant diagonal: position-dependent in the block's own and in the other coordinates."""
    m = neal_funnel_blocks(dim=4).model
    pots = default_potentials(m)
    kh = RiemannianKinetic(HessianMetric(pots, clamp=clamp, softness=0.5),
                           solver=solver(**TIGHT), id="h", slices=[(0, 1), (2, 3)])
    kd = DiagonalQuadraticKinetic(id="d", slices=[(1, 2), (3, 4)])
    _assert_symplectic_and_reversible(_stepper(m, [kd, kh], eps=0.15), jnp.asarray(Q0),
                                      jnp.asarray(P0))


def test_metric_independent_of_the_block_equals_the_explicit_block_flow(x64):
    """``G_x(v)`` only: both implicit solves are exact after one evaluation and the map is the
    explicit :class:`DiagonalBlock` flow, bit for bit. Control: a metric that also depends on ``x``
    differs from the explicit flow with that metric frozen at the start."""
    m = neal_funnel_blocks(dim=4).model
    q0, p0 = jnp.asarray(Q0), jnp.asarray(P0)
    kv = DiagonalQuadraticKinetic(id="v", slices=[m.coord_block("v")])
    explicit = DiagonalBlock("x", m.coord_block("x"), [("v", m.coord_block("v"))],
                             lambda d: jnp.exp(-d["v"]))
    qi, pi_, data = _stepper(m, _funnel_kinetics(m, lambda c: jnp.exp(-c["v"]) * jnp.ones(3)))(
        q0, p0)
    qe, pe, _ = _stepper(m, [kv, explicit])(q0, p0)
    assert np.array_equal(np.asarray(qi), np.asarray(qe))
    assert np.array_equal(np.asarray(pi_), np.asarray(pe))
    assert float(data["fp_iters"]) == 2.0                        # one evaluation per solve

    x0 = q0[1:]
    frozen = DiagonalBlock("x", m.coord_block("x"), [("v", m.coord_block("v"))],
                           lambda d: jnp.exp(-d["v"]) * (1.0 + 0.3 * x0 ** 2))
    q1, p1, _ = _stepper(m, _funnel_kinetics(
        m, lambda c: jnp.exp(-c["v"]) * (1.0 + 0.3 * c["x"] ** 2)))(q0, p0)
    qf, pf, _ = _stepper(m, [kv, frozen])(q0, p0)
    assert float(jnp.max(jnp.abs(p1 - pf))) > 1e-2


def _monolithic_generalized_leapfrog(V, G, q, p, eps, sweeps):
    """The classical Girolami--Calderhead step on ``H = V + 1/2 p^T G^-1 p + 1/2 log|G|``, with
    ``sweeps`` Picard sweeps per implicit solve --- written from the paper, sharing no code."""
    def H(q, p):
        Gq = G(q)
        return V(q) + 0.5 * p @ jnp.linalg.solve(Gq, p) + 0.5 * jnp.linalg.slogdet(Gq)[1]
    dHq = jax.grad(H, 0)
    dHp = jax.grad(H, 1)
    h = 0.5 * eps
    ph = p
    for _ in range(sweeps):
        ph = p - h * dHq(q, ph)
    qn = q
    for _ in range(sweeps):
        qn = q + h * (dHp(q, ph) + dHp(qn, ph))
    return qn, ph - h * dHq(qn, ph)


def test_whole_space_kinetic_matches_the_monolithic_generalized_leapfrog(x64):
    fun = neal_funnel(dim=3).model
    pots = default_potentials(fun)
    ctx = HamiltonianContext(fun.init_chart_hyperparams(), fun.init_chart_indices(), {})
    V = lambda q: sum(pot.potential(q, ctx) for pot in pots)
    G = lambda q: jnp.diag(jnp.concatenate([jnp.ones(1), jnp.exp(-q[0]) * (1 + 0.2 * q[1:] ** 2)]))
    kin = RiemannianKinetic(AnalyticMetric(G), solver=PicardSolver(**TIGHT))
    q0, p0 = jnp.asarray([0.4, 0.8, -0.6]), jnp.asarray([0.5, -0.3, 0.9])
    q1, p1, _ = _stepper(fun, [kin], eps=0.2)(q0, p0)
    qr, pr = _monolithic_generalized_leapfrog(V, G, q0, p0, 0.2, sweeps=200)
    assert float(jnp.max(jnp.abs(q1 - qr))) < 1e-10
    assert float(jnp.max(jnp.abs(p1 - pr))) < 1e-10
    qc, pc = _monolithic_generalized_leapfrog(V, G, q0, p0, 0.2, sweeps=1)     # control
    assert float(jnp.max(jnp.abs(p1 - pc))) > 1e-4


def test_hessian_metric_is_the_precision_on_a_gaussian(x64):
    """Constant Hessian = the precision; at softness 1/b = 1e-3 the clamp leaves it intact, so the
    step equals a dense quadratic kinetic with ``M`` = precision. Control: softness 3 does not."""
    g = block_gaussian()
    m = g.model
    pots = default_potentials(m)
    cov = jnp.asarray(g.cov)
    qg = jnp.asarray(np.random.default_rng(0).normal(size=m.coord_dim))
    pg = jnp.asarray(np.random.default_rng(1).normal(size=m.coord_dim))
    ctx = HamiltonianContext(m.init_chart_hyperparams(), m.init_chart_indices(),
                             {"T": jnp.linalg.cholesky(cov)})             # chol of M^{-1}
    ref = leapfrog(pots, [DenseQuadraticKinetic(id="T")]).step(
        init_integrator_state(pots, qg, pg, ctx), 0.2, ctx)
    for softness, close in ((1e-3, True), (3.0, False)):
        kh = RiemannianKinetic(HessianMetric(pots, softness=softness),
                               solver=AndersonSolver(**TIGHT), id="T")
        q1, p1, _ = _stepper(m, [kh])(qg, pg)
        err = float(jnp.max(jnp.abs(q1 - ref.q)))
        assert (err < 1e-10) if close else (err > 1e-2), (softness, err)


def test_unsolvable_step_is_nan_and_counted():
    """A stiff own-block metric at a large step with a one-evaluation cap cannot converge: the
    step is NaN (so HMC rejects it and NUTS flags a divergence) and the failure is counted."""
    m = neal_funnel_blocks(dim=4).model
    stiff = lambda c: jnp.exp(-3.0 * c["v"]) * (1.0 + c["x"] ** 2)
    step = _stepper(m, _funnel_kinetics(m, stiff, solver=AndersonSolver(max_iter=1)), eps=1.0)
    q1, p1, data = step(jnp.asarray(Q0), jnp.asarray(P0))
    assert not bool(jnp.any(jnp.isfinite(q1)))
    assert not bool(jnp.any(jnp.isfinite(p1)))
    assert float(data["fp_failures"]) >= 1


def _funnel_sampler(base, model, metric_x, **kw):
    kin = _funnel_kinetics(model, metric_x, solver=AndersonSolver())
    Cls = make_sampler_class(RobbinsMonroStepSize, base)
    return Cls(model, init_position=model.default_sample(), kinetics=kin, step_size=0.3,
               seed=3, **kw)


@pytest.mark.parametrize("base", [HMC, NUTS])
def test_counters_reach_the_diagnostics(base):
    m = neal_funnel_blocks(dim=3).model
    s = _funnel_sampler(base, m, lambda c: jnp.exp(-c["v"]) * (1.0 + 0.1 * c["x"] ** 2))
    s.warmup(30)
    s.sample(30)
    d = s.diagnostics()
    assert {"fp_iters", "fp_failures", "grad_evals"} <= set(d)
    assert np.all(d["fp_iters"] > 0)
    assert np.isfinite(s.fixed_point_failure_rate())
    assert s.mean_fixed_point_iterations() > 0
    # a sampler without an implicit kinetic declares none of them
    plain = make_sampler_class(RobbinsMonroStepSize, base)(m, init_position=m.default_sample(),
                                                           seed=3)
    plain.warmup(5)
    assert "fp_iters" not in plain.diagnostics(phase="warmup")
    assert np.isnan(plain.fixed_point_failure_rate(include_warmup=True))


def test_nuts_and_simple_nuts_stay_bit_identical_with_an_implicit_kinetic():
    m = neal_funnel_blocks(dim=3).model
    metric = lambda c: jnp.exp(-c["v"]) * (1.0 + 0.1 * c["x"] ** 2)
    a = _funnel_sampler(NUTS, m, metric, max_tree_depth=5)
    b = _funnel_sampler(SimpleNUTS, m, metric, max_tree_depth=5)
    da, db = a.sample(40), b.sample(40)
    for k in da:
        assert np.array_equal(np.asarray(da[k]), np.asarray(db[k]))
    for k in ("fp_iters", "fp_failures", "grad_evals", "n_leaves"):
        assert np.array_equal(a.diagnostics()[k], b.diagnostics()[k]), k


def test_discrete_labels_reach_a_given_metric():
    def lp(v):
        s = jnp.where(v["z"] == 1, 2.0, 0.5)
        return -0.5 * jnp.sum((v["beta"] / s) ** 2)

    m = Model([EuclideanParameter("beta", (3,))], {"p": lp},
              discrete_parameters=[IntegerParameter("z", (3,), lower=0, upper=1)])
    seen = {}

    def metric(c):
        seen.update(c)
        return 1.0 / jnp.where(c["z"] == 1, 2.0, 0.5) ** 2

    km = CallableMetric(metric, m, 3)
    labels = jnp.asarray([1, 0, 1], jnp.int32)
    ctx = HamiltonianContext(m.init_chart_hyperparams(), m.init_chart_indices(), {},
                             discrete=labels)
    G = km.pre(jnp.zeros(3), ctx, None)
    assert set(seen) == {"beta", "z"}
    assert np.array_equal(np.asarray(seen["z"]), np.asarray(labels))
    assert np.allclose(np.asarray(G), [0.25, 4.0, 0.25])


def test_softness_adaptation_tracks_the_positive_curvature_quantile():
    """On the funnel's ``v`` block the curvature ``1/9 + 1/2 sum x^2 e^{-v}`` varies with the draw.
    After warmup ``1/b`` should be the 10% quantile of the positive curvatures the chain visits,
    divided by 3 --- here held against that quantile of the *sampling* draws' curvatures (a fresh
    sample from the same chain), within a factor of 1.4. Measured over 8 seeds (SoftAbs): 0.85--1.24,
    this seed 1.24. The target needs the 4000 draws: the 10% quantile of 1000 autocorrelated draws
    moves the ratios to 0.81--1.32 on its own. Control: the target is recomputed from the draws
    rather than hardcoded, so a mis-scaled softness (e.g. forgetting the ratio 3) would miss by 3x."""
    m = neal_funnel_blocks(dim=4).model
    pots = default_potentials(m)
    kv = RiemannianKinetic(HessianMetric(pots), id="v", slices=[m.coord_block("v")])
    kx = DiagonalBlock("x", m.coord_block("x"), [("v", m.coord_block("v"))],
                       lambda d: jnp.exp(-d["v"]))
    Cls = make_sampler_class(RobbinsMonroStepSize, HessianSoftnessAdaptation, NUTS)
    # target_accept 0.8 as the factory sets it: the mixin default (0.234, the random-walk optimum)
    # drives NUTS to a step at which ~40% of transitions hit an unsolvable implicit step.
    s = Cls(m, init_position=m.default_sample(), kinetics=[kv, kx], step_size=0.3, seed=1,
            target_accept=0.8)
    s.warmup(1000)
    draws = s.sample(4000)
    v = np.asarray(draws["v"])
    x = np.asarray(draws["x"])
    curv = 1.0 / 9.0 + 0.5 * np.sum(x ** 2, axis=1) * np.exp(-v)
    target = np.quantile(curv, 0.1) / 3.0
    ratio = s.softness("v") / target
    print(f"\n1/b = {s.softness('v'):.4g}, sampling-draw q0.1/3 = {target:.4g}, ratio {ratio:.3f}")
    assert 1 / 1.4 < ratio < 1.4
