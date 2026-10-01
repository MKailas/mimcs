"""Tests for implicit Riemannian blocks under parallel tempering.

The line-search integrators are refused under per-temperature (independent) selection because they
**couple** the lanes: one refinement level is chosen from the summed product Hamiltonian, so lane
k's step depends on lane j's state. An implicit block does not: ``ProductKinetic`` runs its flow
per lane under ``vmap``, and a vmapped ``while_loop`` with per-lane stopping tests runs to the
slowest lane while holding the finished ones --- each lane's result is bitwise its unbatched one.
Different iteration counts cost load imbalance, not validity. These tests pin that and the plumbing
around it, in x64 where they compare maps, each with a control:

* **lane independence**: perturbing lane 2 leaves lanes 0 and 1 bitwise unchanged (control: a
  product line search does change them);
* **each lane is a single chain at its own beta**: lane k's step equals an untempered step on the
  model with its tempered components scaled by beta_k --- fully tempered and as a power posterior
  (control: an unbound metric, the Hessian of the *cold* target, does not);
* a given metric opts into tempering by taking ``beta``; one taking only ``coords`` does not;
* the softness adapts per rung: on a Gaussian ``1/b_k = beta_k / b_0`` exactly (control: unbound);
* the counters are the *batched* work (max over lanes per solve) and reach the diagnostics;
* and the cold chain samples correctly, under independent and joint selection.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from mimcs import config
from mimcs.adaptation import HessianSoftnessAdaptation
from mimcs.factory import BlockSpec, default_spec
from mimcs.hmc import (CallableMetric, DiagonalQuadraticKinetic, HamiltonianContext, Exp,
                       HessianMetric, RiemannianKinetic, default_potentials,
                       init_integrator_state, leapfrog)
from mimcs.hmc.solvers import AndersonSolver
from mimcs.model import EuclideanParameter, Model
from mimcs.pt import (build_product_kinetics, build_tempered_potentials, parallel_tempering,
                      product_line_search)
from mimcs.testing import evaluate, neal_funnel_blocks

TIGHT = dict(tol=1e-13, max_iter=200)
BETAS = [1.0, 0.45, 0.2]


@pytest.fixture
def x64():
    was = config.x64_enabled()
    config.enable_x64(True)
    try:
        yield
    finally:
        config.enable_x64(was)


def _inner(model, tight=True):
    """``x`` on a constant diagonal, ``v`` on the Hessian metric (implicit)."""
    solver = AndersonSolver(**TIGHT) if tight else AndersonSolver()
    return [DiagonalQuadraticKinetic(id="x", slices=[model.coord_block("x")]),
            RiemannianKinetic(HessianMetric(default_potentials(model), softness=0.5),
                              solver=solver, id="v", slices=[model.coord_block("v")])]


def _product(model, inner, betas, tempered=None, bind=True):
    """The product components ``parallel_tempering`` would build, and a one-step function."""
    K = len(betas)
    betas = jnp.asarray(betas, float)
    pots = build_tempered_potentials(model, betas, tempered=tempered)
    if bind:
        flags = {p.inner.id: p.tempered for p in pots}
        for k in inner:
            if hasattr(getattr(k, "metric", None), "bind_tempering"):
                k.metric.bind_tempering(flags)
    pkin = build_product_kinetics(inner, K, model.coord_dim)
    ham = {k.id: k.initial_mass_params(model.coord_dim) for k in pkin}
    ctx = HamiltonianContext(model.init_chart_hyperparams(), model.init_chart_indices(), ham,
                             betas=betas)
    return pots, pkin, ctx


def _stepper(integ, pots, ctx, eps=0.2):
    def step(q, p):
        st = init_integrator_state(pots, q, p, ctx)._replace(
            integrator_data=integ.init_integrator_data())
        out = integ.step(st, eps, ctx)
        return out.q, out.p, out.integrator_data
    return step


def _lane_states(model, K, seed=0):
    rng = np.random.default_rng(seed)
    n = model.coord_dim
    return (jnp.asarray(rng.normal(scale=0.5, size=K * n)),
            jnp.asarray(rng.normal(size=K * n)))


def test_lanes_stay_uncoupled_but_a_line_search_couples_them(x64):
    m = neal_funnel_blocks(dim=3).model
    n, K = m.coord_dim, len(BETAS)
    pots, pkin, ctx = _product(m, _inner(m), BETAS)
    step = _stepper(leapfrog(pots, pkin), pots, ctx)
    q, p = _lane_states(m, K)
    q2 = q.at[2 * n:].add(jnp.asarray([0.9, -0.7, 0.4]))           # move lane 2 only
    p2 = p.at[2 * n:].multiply(3.0)
    a, b = step(q, p), step(q2, p2)
    assert np.array_equal(np.asarray(a[0][:2 * n]), np.asarray(b[0][:2 * n]))
    assert np.array_equal(np.asarray(a[1][:2 * n]), np.asarray(b[1][:2 * n]))
    assert not np.array_equal(np.asarray(a[0][2 * n:]), np.asarray(b[0][2 * n:]))
    assert float(a[2]["fp_iters"]) > 2                             # genuinely implicit

    # Control: the line search picks one refinement level from the summed Hamiltonian, so a lane-2
    # state that forces a finer level moves lanes 0 and 1 too --- the coupling this test detects.
    plain = [DiagonalQuadraticKinetic(id="x", slices=[m.coord_block("x")]),
             DiagonalQuadraticKinetic(id="v", slices=[m.coord_block("v")])]
    pots, pkin, ctx = _product(m, plain, BETAS)
    ls = product_line_search()(pots, pkin, K)
    step = _stepper(ls, pots, ctx, eps=0.8)
    a, b = step(q, p), step(q2, p.at[2 * n:].multiply(40.0))
    assert not np.array_equal(np.asarray(a[0][:2 * n]), np.asarray(b[0][:2 * n]))


def _scaled_model(model, weights):
    """The base model with each log-density component multiplied by ``weights[name]``."""
    fns = {name: (lambda f, w: (lambda prm: w * f(prm)))(fn, weights[name])
           for name, fn in model.log_prob_fns.items()}
    return Model([EuclideanParameter("v", ()), EuclideanParameter("x", (model.coord_dim - 1,))],
                 fns)


@pytest.mark.parametrize("tempered", [None, ("log_lik_x",)])
def test_each_lane_is_the_chain_at_its_own_beta(x64, tempered):
    """Lane k's step equals an *untempered* step of the target with its tempered components scaled
    by beta_k (all of them, or only the likelihood: a power posterior). Control: with the metric
    left unbound --- the Hessian of the cold target at every rung --- a hot lane misses."""
    m = neal_funnel_blocks(dim=3).model
    n, K = m.coord_dim, len(BETAS)
    q, p = _lane_states(m, K, seed=1)
    pots, pkin, ctx = _product(m, _inner(m), BETAS, tempered=tempered)
    qk, pk, _ = _stepper(leapfrog(pots, pkin), pots, ctx)(q, p)
    pots_u, pkin_u, ctx_u = _product(m, _inner(m), BETAS, tempered=tempered, bind=False)
    qu, pu, _ = _stepper(leapfrog(pots_u, pkin_u), pots_u, ctx_u)(q, p)
    for k, beta in enumerate(BETAS):
        w = {c: (beta if tempered is None or c in tempered else 1.0) for c in m.log_prob_fns}
        ref_m = _scaled_model(m, w)
        ref_pots = default_potentials(ref_m)
        ref_kin = _inner(ref_m)
        ham = {kk.id: kk.initial_mass_params(n) for kk in ref_kin}
        ref_ctx = HamiltonianContext(ref_m.init_chart_hyperparams(),
                                     ref_m.init_chart_indices(), ham)
        rq, rp, _ = _stepper(leapfrog(ref_pots, ref_kin), ref_pots, ref_ctx)(
            q[k * n:(k + 1) * n], p[k * n:(k + 1) * n])
        assert float(jnp.max(jnp.abs(qk[k * n:(k + 1) * n] - rq))) < 1e-12, k
        assert float(jnp.max(jnp.abs(pk[k * n:(k + 1) * n] - rp))) < 1e-12, k
        if beta < 1.0 and tempered is None:
            assert float(jnp.max(jnp.abs(pu[k * n:(k + 1) * n] - rp))) > 1e-4, k


def test_a_given_metric_sees_beta_only_if_it_asks():
    m = neal_funnel_blocks(dim=3).model
    seen = []

    def tempered_fn(coords, beta):
        seen.append(beta)
        return beta * jnp.ones(1)

    km2 = CallableMetric(tempered_fn, m, 1)
    km1 = CallableMetric(lambda coords: 2.0 * jnp.ones(1), m, 1)
    assert km2.takes_beta and not km1.takes_beta
    q = jnp.zeros(3)
    for beta in (None, 0.3):
        ctx = HamiltonianContext(m.init_chart_hyperparams(), m.init_chart_indices(), {},
                                 betas=None if beta is None else jnp.asarray(beta))
        assert float(km2.pre(q, ctx, None)[0]) == pytest.approx(1.0 if beta is None else 0.3)
        assert float(km1.pre(q, ctx, None)[0]) == 2.0
    # and through the product kinetic, each lane gets its own rung's beta
    kin = RiemannianKinetic(km2, id="v", slices=[m.coord_block("v")])
    _, pkin, ctx = _product(m, [kin], BETAS)
    pq = jnp.zeros(len(BETAS) * 3)
    per = pkin[0].per_temperature_energy(
        init_integrator_state([], pq, jnp.ones_like(pq), ctx), ctx)
    # T_k = 1/2 p^2 / beta_k + 1/2 log beta_k  (p_v = 1)
    expect = [0.5 / b + 0.5 * np.log(b) for b in BETAS]
    assert np.allclose(np.asarray(per), expect, rtol=1e-6)


def _gaussian(d=10, seed=0):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    P = jnp.asarray(Q @ np.diag(np.linspace(0.5, 5.0, d)) @ Q.T, float)
    return Model([EuclideanParameter("x", (d,))],
                 {"lp": lambda prm: -0.5 * prm["x"] @ P @ prm["x"]}), np.linspace(0.5, 5.0, d)


@pytest.mark.parametrize("bind", [True, False])
def test_softness_adapts_per_rung(x64, bind):
    """The Hessian at rung k of a fully tempered Gaussian is ``beta_k P``: its spectrum is constant,
    so the adapted quantile is exact and ``1/b_k = beta_k (1/b_0)``. Control: an unbound metric (the
    cold Hessian at every rung) adapts every rung to the same softness."""
    m, eig = _gaussian()
    kin = RiemannianKinetic(HessianMetric(default_potentials(m)), id="T")
    s = parallel_tempering(m, betas=BETAS, adapt_ladder=False, kinetics=[kin],
                           adapt_mixins=(HessianSoftnessAdaptation,), step_size=0.2, seed=0)
    if not bind:
        kin.metric.tempered = None
    s.warmup(20)
    soft = np.exp(np.asarray(s.state.ham_params["T"]["log_softness"]))
    base = np.quantile(eig, 0.1) / 3.0
    expect = base * np.asarray(BETAS) if bind else np.full(len(BETAS), base)
    assert np.allclose(soft, expect, rtol=1e-8), (soft, expect)


def test_product_counters_are_the_batched_work(x64):
    m = neal_funnel_blocks(dim=3).model
    n, K = m.coord_dim, len(BETAS)
    inner = _inner(m, tight=False)
    pots, pkin, ctx = _product(m, inner, BETAS)
    q, p = _lane_states(m, K, seed=2)
    integ = leapfrog(pots, pkin)
    st = init_integrator_state(pots, q, p, ctx)._replace(
        integrator_data=integ.init_integrator_data())
    vk = pkin[1]
    out = vk.flow(st, 0.3, ctx)
    # the same lanes, one at a time
    kicks, moves, oks = [], [], []
    for k in range(K):
        ctx_k = HamiltonianContext(ctx.chart_hyperparams, ctx.chart_indices,
                                   {"v": jax.tree.map(lambda a: a[k], ctx.ham_params["v"])},
                                   betas=ctx.betas[k])
        lane = init_integrator_state([], q[k * n:(k + 1) * n], p[k * n:(k + 1) * n], ctx_k)
        _, (ki, mi, ok) = vk.inner.flow_with_stats(lane, 0.3, ctx_k)
        kicks.append(int(ki)), moves.append(int(mi)), oks.append(bool(ok))
    assert len(set(kicks + moves)) > 1                       # the lanes did differ
    assert float(out.integrator_data["fp_iters"]) == max(kicks) + max(moves)
    assert float(out.integrator_data["fp_failures"]) == sum(not o for o in oks)
    assert float(out.integrator_data["fp_lane_iters"]) == sum(kicks) + sum(moves)


def _pt_spec(model, selection=None, v_params=None):
    spec = default_spec(model)
    cb = model.coord_block
    spec.base = "pt_nuts"
    spec.tempering_params = {"n_temperatures": 3, "beta_min": 0.2}
    if selection is not None:
        spec.tempering_params["selection"] = selection
    spec.blocks = [BlockSpec(["v"], [cb("v")], "riemannian", dict(v_params or {})),
                   BlockSpec(["x"], [cb("x")], "learned_metric", {"metric": Exp("v")})]
    return spec


@pytest.mark.parametrize("base", ["pt_nuts", "pt_hmc"])
def test_counters_reach_the_tempered_diagnostics(base):
    m = neal_funnel_blocks(dim=2).model
    spec = _pt_spec(m)
    spec.base = base
    s = spec.build(seed=0)
    s.warmup(20)
    d = s.diagnostics(phase="warmup")
    assert {"fp_iters", "fp_failures", "fp_lane_iters"} <= set(d)
    assert np.all(d["fp_iters"] * 3 >= d["fp_lane_iters"])        # batched work >= mean lane
    assert np.all(d["fp_iters"] > 0)


@pytest.mark.parametrize("selection", ["independent", "joint"])
def test_pt_with_a_hessian_block_samples_the_funnel(artifacts_dir, selection):
    """The cold chain against the exact funnel (scale 2: see ``test_factory_riemannian`` for why not
    3), with the ``v`` Hessian block and the learned ``x`` metric, under both selection modes."""
    problem = neal_funnel_blocks(dim=2, scale=2.0)
    problem.hard = False
    report = evaluate(problem, {selection: lambda model, seed: _pt_spec(model, selection).build(
        seed=seed)}, n_warmup=1000, n_samples=4000, seed=0,
        out_dir=str(artifacts_dir / f"pt_riemannian_{selection}"))
    print("\n" + report.summary())
    report.assert_correct()


def test_pt_with_a_beta_aware_given_metric_samples_the_funnel(artifacts_dir):
    """A given metric that tempers itself: the natural funnel metric, ``beta * (1, e^{-v})`` --- the
    Fisher metric of the fully tempered target."""
    problem = neal_funnel_blocks(dim=2, scale=2.0)
    problem.hard = False

    def make(model, seed):
        spec = default_spec(model)
        spec.base = "pt_nuts"
        spec.tempering_params = {"n_temperatures": 3, "beta_min": 0.2}
        spec.blocks = [BlockSpec(["v", "x"], [(0, 2)], "riemannian", {
            "metric": lambda c, beta: beta * jnp.concatenate([jnp.ones(1), jnp.exp(-c["v"])])})]
        return spec.build(seed=seed)

    report = evaluate(problem, {"given_beta": make}, n_warmup=1000, n_samples=4000, seed=0,
                      out_dir=str(artifacts_dir / "pt_riemannian_given"))
    print("\n" + report.summary())
    report.assert_correct()
