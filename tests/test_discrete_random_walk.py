"""Tests for unbounded and ordinal integer parameters and their random-walk proposal.

The proposal is a fair coin for the direction and a ``Geometric(p)`` step, ``1/p = 1 + exp(rho)``,
clamped at a declared bound. Its algebra was checked on an exactly enumerated kernel before any of
this existed (``tests/experiments/discrete_random_walk_kernel.py``); the checks below keep it. Three
things here are silent when wrong, and each has a control that must fail:

* **Clamping makes the proposal asymmetric.** An endpoint collects the whole geometric tail, so the
  ratio needs ``log p * (1[b at bound] - 1[a at bound])``. Without it the chain still runs and still
  looks healthy; detailed balance is what catches it.
* **A clamped no-op must not feed the adaptation.** Counting its acceptance of 1 drives the scale to
  its cap next to a bound holding most of the mass (IACT 3.2e5 against 2.71 on Poisson(0.1)).
* **An open side uses a sentinel bound**, so the clamp arithmetic must not overflow int32 there.
"""

import logging

import numpy as np
import jax.numpy as jnp
import pytest
from jax.scipy.special import gammaln as jgammaln
from scipy.special import expit, gammaln

from mimcs import compile_model, DslError
from mimcs.adaptation import DiscreteRandomWalkAdaptation, RobbinsMonroStepSize
from mimcs.hmc import NUTS
from mimcs.model import EuclideanParameter, IntegerParameter, Model
from mimcs.model.integer import INT_BOUND
from mimcs.model.jump import JumpOperator
from mimcs.samplers import SystematicScanMetropolisWithinGibbs, StaticContinuous, make_sampler_class
from mimcs.samplers.discrete_updates import (RW_LOG_SCALE_MAX_UNBOUNDED, RW_LOG_SCALE_MIN,
                                             RandomWalkUpdate, SweepEnv, build_discrete_updaters)

GIBBS_ONLY = make_sampler_class(SystematicScanMetropolisWithinGibbs, StaticContinuous)
GIBBS_RW = make_sampler_class(DiscreteRandomWalkAdaptation, SystematicScanMetropolisWithinGibbs,
                              StaticContinuous)
NUTS_GIBBS = make_sampler_class(RobbinsMonroStepSize, DiscreteRandomWalkAdaptation,
                                SystematicScanMetropolisWithinGibbs, NUTS)


# --------------------------------------------------------------------------- #
# the proposal, enumerated                                                     #
# --------------------------------------------------------------------------- #

def _analytic_row(a, lo, hi, rho):
    """``q(a -> .)`` on ``lo..hi`` for the clamped two-sided geometric, by direct summation."""
    p = expit(-rho)
    row = np.zeros(hi - lo + 1)
    for sgn, edge in ((+1, hi - a), (-1, a - lo)):
        if edge == 0:
            row[a - lo] += 0.5                                   # outward at a bound: a no-op
            continue
        for k in range(1, edge):
            row[a + sgn * k - lo] += 0.5 * p * (1 - p) ** (k - 1)
        row[(hi if sgn > 0 else lo) - lo] += 0.5 * (1 - p) ** (edge - 1)
    return row


def _grid_row(u, a, rho, n_u=100001):
    """``q(a -> .)`` from the real ``_propose``, over a fine grid of uniforms (one lane each)."""
    grid = jnp.asarray((np.arange(n_u) + 0.5) / n_u, float)
    env = SweepEnv(sampler=None, state=None, sweep_ctx=None, plans={},
                   tables={u.name: {"log_scale": jnp.full((n_u, 1), rho, float)}},
                   u_prop=grid[None, :], u_acc=grid[None, :], n_lanes=n_u, lane_dim=1)
    prep = u.prepare(env)
    prop = np.asarray(u._propose(env, prep, 0, 0, jnp.full((n_u,), a, jnp.int32)))
    return prop, env, prep


@pytest.mark.parametrize("rho", [-1.0, 1.5])
@pytest.mark.parametrize("a", [0, 4, 12])
def test_the_proposal_is_the_clamped_two_sided_geometric(a, rho):
    u = RandomWalkUpdate(IntegerParameter("z", lower=0, upper=12, ordinal=True), 0)
    prop, _, _ = _grid_row(u, a, rho)
    assert prop.min() >= 0 and prop.max() <= 12
    emp = np.bincount(prop, minlength=13) / len(prop)
    want = _analytic_row(a, 0, 12, rho)
    assert np.max(np.abs(emp - want)) < 2e-4, (emp, want)
    # CONTROL: the law with p and 1-p exchanged is a different distribution, and the grid tells.
    wrong = _analytic_row(a, 0, 12, -rho)
    assert np.max(np.abs(emp - wrong)) > 2e-2


def test_the_hastings_term_restores_detailed_balance():
    """The enumerated single-coordinate kernel, built from the real `_propose` and `_log_hastings`,
    on a skewed pmf over a bounded support where every state is near one bound or the other."""
    lo, hi, rho = 0, 12, 0.7
    w = np.array([5, 1, 1, 8, 2, 1, 1, 1, 6, 1, 1, 3, 4.0])
    lp = np.log(w / w.sum())
    u = RandomWalkUpdate(IntegerParameter("z", lower=lo, upper=hi, ordinal=True), 0)
    n = hi - lo + 1
    Q, H = np.zeros((n, n)), np.zeros((n, n))
    for a in range(n):
        prop, env, prep = _grid_row(u, a, rho)
        Q[a] = np.bincount(prop, minlength=n) / len(prop)
        cur = jnp.full((n,), a, jnp.int32)
        H[a] = np.asarray(u._log_hastings(env, (prep[0][:n], prep[1][:n]), 0, cur,
                                          jnp.arange(n, dtype=jnp.int32)))

    def imbalance(with_h):
        A = np.minimum(1.0, np.exp(lp[None, :] - lp[:, None] + (H if with_h else 0.0)))
        F = np.exp(lp)[:, None] * Q * A
        np.fill_diagonal(F, 0.0)
        return np.max(np.abs(F - F.T))

    good, bad = imbalance(True), imbalance(False)
    assert good < 2e-4, good
    assert bad > 10 * good and bad > 5e-3, (good, bad)          # CONTROL: the term does work


def test_a_clamped_proposal_is_a_no_op_and_is_not_counted():
    u = RandomWalkUpdate(IntegerParameter("z", lower=0), 0)
    prop, _, _ = _grid_row(u, 0, 0.0, n_u=2001)
    grid = (np.arange(2001) + 0.5) / 2001
    assert np.all(prop[grid >= 0.5] == 0), "the downward half must clamp onto the bound"
    assert np.all(prop[grid < 0.5] > 0), "the upward half must move"
    stats = {"z": (jnp.zeros((3, 1)), jnp.zeros((3, 1)))}
    acc, n = u._record(stats, 0, jnp.asarray([1.0, 1.0, 0.4]),
                       jnp.asarray([False, True, True]))["z"]
    assert np.allclose(np.asarray(n)[:, 0], [0, 1, 1])
    assert np.allclose(np.asarray(acc)[:, 0], [0.0, 1.0, 0.4])


def test_an_interior_coordinate_counts_every_sweep():
    """CONTROL for the no-op exclusion: away from any bound, every proposal is genuine."""
    p = IntegerParameter("g")
    m = Model([], {"lp": lambda v: jnp.sum(-0.5 * (v["g"] / 50.0) ** 2)}, discrete_parameters=[p])
    s = GIBBS_ONLY(m, {}, seed=0, discrete_update={"g": "random_walk"}, discrete_sweeps=3)
    s.initialize()
    s.warmup(5)
    assert np.allclose(np.asarray(s.state.discrete_proposal_params["g"]["n_proposed"]), 3.0)


@pytest.mark.parametrize("cur, up", [(INT_BOUND, True), (INT_BOUND, False),
                                     (-INT_BOUND, False), (-INT_BOUND, True), (0, True)])
def test_the_clamp_cannot_overflow_at_the_sentinel(cur, up):
    u = RandomWalkUpdate(IntegerParameter("g"), 0)
    tiny = np.array([1e-7, 0.25, 1e-30]) if up else np.array([0.5 + 1e-7, 0.75, 0.9999999])
    env = SweepEnv(sampler=None, state=None, sweep_ctx=None, plans={},
                   tables={"g": {"log_scale": jnp.full((3, 1), RW_LOG_SCALE_MAX_UNBOUNDED + 30.0)}},
                   u_prop=jnp.asarray(tiny, float)[None, :], u_acc=jnp.zeros((1, 3)),
                   n_lanes=3, lane_dim=1)
    prop = np.asarray(u._propose(env, u.prepare(env), 0, 0, jnp.full((3,), cur, jnp.int32)),
                      dtype=np.int64)
    assert np.all(prop >= -INT_BOUND) and np.all(prop <= INT_BOUND), prop
    if up:
        assert np.all(prop >= cur)
    else:
        assert np.all(prop <= cur)


# --------------------------------------------------------------------------- #
# the sweep samples exact targets                                              #
# --------------------------------------------------------------------------- #

def _sample(p, logpmf, n=40000, seed=0, rho=0.5):
    m = Model([], {"lp": lambda v: jnp.sum(logpmf(v[p.name]))}, discrete_parameters=[p])
    s = GIBBS_ONLY(m, {}, seed=seed, discrete_update={p.name: "random_walk"},
                   discrete_rw_init_log_scale=rho)
    s.initialize()
    s.warmup(200)
    s.sample(n)
    return s, np.asarray(s.get_samples()[p.name]).reshape(-1)


def _tv(z, support, pmf):
    emp = np.array([(z == v).mean() for v in support])
    return 0.5 * np.abs(emp - pmf).sum()


_XP = np.arange(0, 60)
_POIS3 = np.exp(_XP * np.log(3.0) - gammaln(_XP + 1))
_POIS3 /= _POIS3.sum()


def test_a_bounded_ordinal_pmf_is_sampled_exactly():
    w = np.array([5, 1, 1, 8, 2, 1, 1, 1, 6, 1, 1, 3.0])
    lw = jnp.asarray(np.log(w / w.sum()), float)
    _, z = _sample(IntegerParameter("a", lower=1, upper=12, ordinal=True),
                   lambda z: lw[z - 1], rho=1.0)
    assert _tv(z, np.arange(1, 13), w / w.sum()) < 0.025


def test_a_count_with_an_open_upper_side_is_sampled_exactly():
    s, z = _sample(IntegerParameter("n", lower=0), lambda z: z * np.log(3.0) - jgammaln(z + 1.0))
    assert _tv(z, _XP, _POIS3) < 0.015
    assert z.min() == 0, "the chain must reach the bound, which is where the clamp acts"
    assert np.mean(np.asarray(s.diagnostics()["discrete_moves"])) > 0.3


def test_a_perturbed_count_fails_the_same_comparison():
    """CONTROL for the test above: Poisson(3.3) must not pass as Poisson(3)."""
    _, z = _sample(IntegerParameter("n", lower=0), lambda z: z * np.log(3.3) - jgammaln(z + 1.0))
    assert _tv(z, _XP, _POIS3) > 0.04


def test_a_two_sided_unbounded_integer_is_sampled_exactly():
    xg = np.arange(-60, 47)
    pg = np.exp(-0.5 * ((xg + 7) / 6.0) ** 2)
    pg /= pg.sum()
    _, z = _sample(IntegerParameter("g"), lambda z: -0.5 * ((z + 7.0) / 6.0) ** 2, rho=2.0)
    assert _tv(z, xg, pg) < 0.03
    assert abs(z.mean() + 7.0) < 0.35


def test_an_open_side_starts_inside_its_window():
    ps = [IntegerParameter("a", (50,), lower=3), IntegerParameter("b", (50,)),
          IntegerParameter("c", (50,), upper=-10)]
    m = Model([], {"lp": lambda v: jnp.zeros(())}, discrete_parameters=ps)
    s = GIBBS_ONLY(m, {}, seed=0, discrete_update={p.name: "random_walk" for p in ps})
    s.initialize()
    d = {k: np.asarray(v) for k, v in m.unpack_discrete(s.state.discrete).items()}
    assert d["a"].min() >= 3 and d["a"].max() <= 7
    assert d["b"].min() >= -2 and d["b"].max() <= 2
    assert d["c"].min() >= -14 and d["c"].max() <= -10


def test_enumerating_methods_refuse_an_open_side():
    m = Model([], {"lp": lambda v: jnp.zeros(())},
              discrete_parameters=[IntegerParameter("n", lower=0)])
    for kind in ("metropolis", "exact"):
        with pytest.raises(ValueError, match="no enumerable support"):
            build_discrete_updaters(m, {"n": kind})


# --------------------------------------------------------------------------- #
# the adaptation                                                               #
# --------------------------------------------------------------------------- #

def _adapt(p, logpmf, warm=3000, seed=0):
    m = Model([], {"lp": lambda v: jnp.sum(logpmf(v[p.name]))}, discrete_parameters=[p])
    s = GIBBS_RW(m, {}, seed=seed, discrete_update={p.name: "random_walk"})
    s.initialize()
    s.warmup(warm)
    return s, float(np.asarray(s.state.discrete_proposal_params[p.name]["log_scale"]).ravel()[0])


def test_the_scale_converges_to_the_exact_kernel_root_on_a_laplace_target():
    """The exact-kernel scale at which a genuine proposal is accepted 1/3 of the time is rho ~ 4.0
    for a discretised Laplace with b = 15 (also where its IACT is minimal)."""
    s, rho = _adapt(IntegerParameter("a"), lambda z: -jnp.abs(z - 3.0) / 15.0)
    assert 3.4 < rho < 4.6, rho
    assert 0.25 < float(s._rw_pooled["a"]) < 0.42
    before = np.asarray(s.state.discrete_proposal_params["a"]["log_scale"]).copy()
    s.sample(300)
    after = np.asarray(s.state.discrete_proposal_params["a"]["log_scale"])
    assert np.array_equal(before, after), "the scale must be frozen during sampling"


def test_a_mass_at_the_bound_settles_on_the_pm1_walk_rather_than_running_away(caplog):
    """Poisson(0.1): no scale reaches 1/3 over genuine proposals, so rho goes to its floor. Counting
    clamped no-ops instead would have sent it to its *cap* (the exact kernel says IACT 3.2e5)."""
    with caplog.at_level(logging.INFO, logger="mimcs"):
        s, rho = _adapt(IntegerParameter("n", lower=0), lambda z: z * np.log(0.1) - jgammaln(z + 1.0))
    assert rho < -5.0, rho
    assert any("+-1 walk" in r.getMessage() for r in caplog.records)


def test_the_adaptation_is_stream_neutral_on_a_model_without_a_walk():
    p = IntegerParameter("z", (3,), lower=0, upper=3)
    lw = jnp.asarray([0.3, -1.0, 2.0, 0.1], float)
    m = Model([], {"lp": lambda v: jnp.sum(lw[v["z"]])}, discrete_parameters=[p])
    a = GIBBS_ONLY(m, {}, seed=3)
    b = GIBBS_RW(m, {}, seed=3)
    for s in (a, b):
        s.initialize(); s.warmup(50); s.sample(200)
    assert np.array_equal(np.asarray(a.get_samples()["z"]), np.asarray(b.get_samples()["z"]))


# --------------------------------------------------------------------------- #
# jump operators on the walk                                                   #
# --------------------------------------------------------------------------- #

_MU = 2.0


def _count_with_effect(jump):
    """``n ~ Poisson(3)``, ``x | n ~ N(mu n, 1)``: the marginal of ``n`` is Poisson(3) exactly."""
    def lp(v):
        n, x = v["n"], jnp.reshape(v["x"], ())
        return n * np.log(3.0) - jgammaln(n + 1.0) - 0.5 * (x - _MU * n) ** 2
    ops = {"n": JumpOperator("n", ("x",), jump)} if jump is not None else {}
    return Model([EuclideanParameter("x", ())], {"p": lp},
                 discrete_parameters=[IntegerParameter("n", (), lower=0)], jump_operators=ops)


def test_a_compensated_jump_on_an_unbounded_count_samples_the_exact_marginal():
    m = _count_with_effect(lambda values, c, v: (values["x"] + _MU * (v - values["n"]),))
    s = NUTS_GIBBS(m, m.default_sample(), seed=0, discrete_update={"n": "random_walk"})
    s.initialize()
    s.warmup(600)
    s.sample(20000)
    d = s.get_samples()
    z, x = np.asarray(d["n"]).ravel(), np.asarray(d["x"]).ravel()
    assert _tv(z, _XP, _POIS3) < 0.04
    assert abs(x.mean() - _MU * 3.0) < 0.35


def test_a_non_involutive_jump_on_a_walk_is_refused():
    """CONTROL: a map written in the proposed value alone does not telescope."""
    m = _count_with_effect(lambda values, c, v: (values["x"] * (1.0 + 0.3 * v),))
    with pytest.raises(ValueError, match="jump operator for 'n'"):
        NUTS_GIBBS(m, m.default_sample(), seed=0, discrete_update={"n": "random_walk"})


# --------------------------------------------------------------------------- #
# tempering                                                                    #
# --------------------------------------------------------------------------- #

def test_each_rung_adapts_its_own_scale_and_a_hot_rung_steps_further():
    from mimcs.pt import parallel_tempering
    m = Model([], {"lp": lambda v: jnp.sum(-0.5 * (v["g"] / 6.0) ** 2)},
              discrete_parameters=[IntegerParameter("g", (2,))])
    s = parallel_tempering(m, n_temperatures=3, seed=0, adapt_ladder=False,
                           betas=jnp.asarray([1.0, 0.3, 0.05]),
                           extra_mixins=(DiscreteRandomWalkAdaptation,),
                           discrete_update={"g": "random_walk"})
    s.initialize()
    s.warmup(1500)
    rho = np.asarray(s.state.discrete_proposal_params["g"]["log_scale"])
    assert rho.shape == (3, 2)
    assert np.all(rho[2] > rho[0] + 1.0), rho          # a 20x flatter rung: ~4.5x wider target
    s.sample(20000)
    g = np.asarray(s.get_samples()["g"]).reshape(-1)
    xg = np.arange(-40, 41)
    pg = np.exp(-0.5 * (xg / 6.0) ** 2); pg /= pg.sum()
    assert _tv(g, xg, pg) < 0.04


# --------------------------------------------------------------------------- #
# DSL                                                                          #
# --------------------------------------------------------------------------- #

def test_the_dsl_declares_ordinal_and_open_sided_integers():
    m = compile_model("parameters { array[3] ordinal int<lower=1, upper=10> a; int n; "
                      "int<lower=0> c; int<lower=0, upper=1> b; } model { }", data={})
    by = {p.name: p for p in m.discrete_parameters}
    assert by["a"].ordinal and by["a"].bounded and by["a"].n_values == 10
    assert by["n"].ordinal and not by["n"].bounded
    assert (by["c"].lower_value, by["c"].upper_value) == (0, None)
    assert not by["b"].ordinal


@pytest.mark.parametrize("src, fragment", [
    ("parameters { ordinal real x; } model { }", "does not apply to `real`"),
    ("data { ordinal int n; } parameters { real x; } model { }", "only be declared in the `parameters`"),
    ("parameters { real x; } model { ordinal int k = 1; }", "only be declared in the `parameters`"),
    ("functions { real f(ordinal int k) { return 1.0; } } parameters { real x; } model { }",
     "only be declared in the `parameters`"),
    ("parameters { int ordinal x; } model { }", "is a keyword"),
    ("parameters { real x; } model { real ordinal = 1.0; }", "is a keyword"),
])
def test_ordinal_is_refused_where_it_means_nothing(src, fragment):
    with pytest.raises(DslError, match=fragment):
        compile_model(src, data={"n": 1})


# --------------------------------------------------------------------------- #
# factory                                                                      #
# --------------------------------------------------------------------------- #

def _factory_model():
    ps = [IntegerParameter("g", (2,)), IntegerParameter("o3", lower=0, upper=2, ordinal=True),
          IntegerParameter("o2", lower=0, upper=1, ordinal=True),
          IntegerParameter("b3", lower=0, upper=2)]
    return Model([], {"lp": lambda v: (jnp.sum(-0.5 * (v["g"] / 4.0) ** 2) + 0.3 * v["o3"]
                                       + 0.2 * v["o2"] - 0.1 * v["b3"])},
                 discrete_parameters=ps)


def test_the_factory_walks_open_and_ordinal_parameters_but_not_a_binary_one():
    from mimcs import analyze
    spec = analyze(_factory_model())
    kinds = {d.name: d.kind for d in spec.discrete}
    assert kinds == {"g": "random_walk", "o3": "random_walk", "o2": "metropolis",
                     "b3": "metropolis"}
    assert any("vacuous" in r for r in spec.rationale)
    assert "unbounded" in str(spec.discrete[0])
    s = spec.build(seed=0)
    assert any(c.__name__ == "DiscreteRandomWalkAdaptation" for c in type(s).__mro__)


def test_the_factory_refuses_an_enumerating_method_on_an_open_side():
    from mimcs import analyze
    spec = analyze(_factory_model())
    spec.discrete[0].kind = "exact"
    with pytest.raises(ValueError, match="open bound"):
        spec.build(seed=0)


def test_the_default_spec_builds_and_samples_an_unbounded_model():
    from mimcs.factory import default_spec
    m = Model([], {"lp": lambda v: jnp.sum(-0.5 * ((v["g"] - 20.0) / 5.0) ** 2)},
              discrete_parameters=[IntegerParameter("g")])
    spec = default_spec(m)
    spec.base, spec.adapt_step_size, spec.mass_adapt, spec.terminate = "static", False, None, None
    s = spec.build(seed=1)
    s.initialize(); s.warmup(800); s.sample(8000)
    assert abs(np.asarray(s.get_samples()["g"]).mean() - 20.0) < 1.0


def test_evidence_sets_each_walk_coordinates_starting_scale_from_its_iqr():
    """``2/p = IQR`` per coordinate: ``rho = log(IQR/2 - 1)``, floored at the +-1 walk."""
    from mimcs import analyze
    from mimcs.factory.rules import _rw_evidence_log_scale
    m = _factory_model()
    n = 4001
    z = np.zeros((n, m.discrete_dim), np.int64)
    s, _ = m.discrete_block("g")
    z[:, s] = np.arange(n) % 41 - 20          # uniform on -20..20: IQR 20 -> 1/p = 10
    z[:, s + 1] = np.arange(n) % 3            # IQR 2 -> 1/p = 1 -> the floor
    o3 = m.discrete_block("o3")[0]
    z[:, o3] = np.arange(n) % 3
    spec = analyze(m, {"discrete": z})
    rho = spec.discrete[0].params["init_log_scale"]
    np.testing.assert_allclose(rho, [np.log(9.0), RW_LOG_SCALE_MIN])
    assert "scale from evidence" in str(spec.discrete[0])
    assert "init_log_scale" in spec.discrete[1].params
    assert "init_log_scale" not in spec.discrete[2].params     # metropolis slots are untouched
    assert _rw_evidence_log_scale(m, SimpleNamespaceNoLabels(), m.discrete_parameters[0]) is None
    # CONTROL: without evidence the slot carries no starting scale, and the walk starts at rho = 0
    assert "init_log_scale" not in analyze(m).discrete[0].params

    s1 = spec.build(seed=0)
    got = np.asarray(s1.state.discrete_proposal_params["g"]["log_scale"])
    np.testing.assert_allclose(got, [[np.log(9.0), RW_LOG_SCALE_MIN]], rtol=1e-6)
    got0 = np.asarray(analyze(m).build(seed=0).state.discrete_proposal_params["g"]["log_scale"])
    np.testing.assert_allclose(got0, 0.0)


class SimpleNamespaceNoLabels:
    discrete = None


def test_a_hand_set_float_still_starts_the_walks_evidence_did_not_scale():
    from mimcs import analyze
    m = _factory_model()
    spec = analyze(m)
    spec.discrete[0].params["init_log_scale"] = np.array([1.0, 2.0])
    spec.algo_kwargs["discrete_rw_init_log_scale"] = 0.5
    st = spec.build(seed=0).state.discrete_proposal_params
    np.testing.assert_allclose(np.asarray(st["g"]["log_scale"]), [[1.0, 2.0]], rtol=1e-6)
    np.testing.assert_allclose(np.asarray(st["o3"]["log_scale"]), 0.5, rtol=1e-6)
    spec.discrete[0].params["init_log_scale"] = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="one entry per coordinate"):
        spec.build(seed=0)


def test_an_unbounded_parameter_is_never_a_metric_dependency():
    from types import SimpleNamespace
    from mimcs.factory.rules import _discrete_dep_cols
    from mimcs.hmc.block_riemannian import _resolve_discrete_deps
    from mimcs.hmc.metric_expr import Exp
    m = _factory_model()
    cols = _discrete_dep_cols(m, SimpleNamespace(discrete=np.zeros((4, m.discrete_dim), int)))
    assert "g" not in cols and {"o3", "o2", "b3"} <= set(cols)
    _resolve_discrete_deps(m, Exp(ordinal=["o3"]))                # CONTROL: a bounded one resolves
    with pytest.raises(ValueError, match="open bound"):
        _resolve_discrete_deps(m, Exp(ordinal=["g"]))
