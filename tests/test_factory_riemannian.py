"""Tests for the factory's ``BlockSpec(kind="riemannian")``: the implicit general-metric block.

Lowering: the kind builds a :class:`~mimcs.hmc.RiemannianKinetic` over the block's slices (fused and
non-contiguous allowed), a clamped-Hessian metric when ``params`` has no ``"metric"`` (with the
softness adaptation composed in) and a given one otherwise; misspelled or inapplicable options, a
malformed given metric, and a tempered base all raise rather than pass silently.

Sampling (``evaluate`` harness, pinned seeds): a Hessian ``v`` block with a learned-metric ``x`` block
on Neal's funnel, and the ~9x fewer divergences it buys over a constant ``v`` mass; a whole-space
SoftAbs Hessian block on the Rosenbrock banana, whose Hessian is indefinite off the ridge, and the
softplus clamp's failure there; a fused, non-contiguous Hessian block of two funnels' scale
parameters (the "hyperparameter block"); and a given metric through ``params["metric"]``.
"""

import numpy as np
import jax.numpy as jnp
import pytest

from mimcs.adaptation import HessianSoftnessAdaptation
from mimcs.factory import BlockSpec, default_spec
from mimcs.hmc import CallableMetric, Exp, HessianMetric, RiemannianKinetic
from mimcs.model import EuclideanParameter, Model
from mimcs.testing import TargetProblem, evaluate, neal_funnel_blocks, rosenbrock


def _funnel_spec(model, v_params=None, x_kind="learned_metric"):
    spec = default_spec(model)
    sv, sx = model.coord_block("v"), model.coord_block("x")
    x_params = {"metric": Exp("v")} if x_kind == "learned_metric" else {}
    spec.blocks = [BlockSpec(["v"], [sv], "riemannian", dict(v_params or {})),
                   BlockSpec(["x"], [sx], x_kind, x_params)]
    return spec


def _builder(make_spec):
    return lambda model, seed: make_spec(model).build(seed=seed)


# --- lowering ------------------------------------------------------------------------------- #

def test_hessian_block_builds_with_softness_adaptation():
    m = neal_funnel_blocks(dim=3).model
    s = _funnel_spec(m).build()
    kv = next(k for k in s.kinetics if k.id == "v")
    assert isinstance(kv, RiemannianKinetic) and isinstance(kv.metric, HessianMetric)
    assert kv.slices == [m.coord_block("v")]
    assert isinstance(s, HessianSoftnessAdaptation)
    assert "riemannian" in str(_funnel_spec(m)) and "hessian(softabs)" in str(_funnel_spec(m))


def test_given_metric_builds_without_softness_adaptation():
    m = neal_funnel_blocks(dim=3).model
    spec = _funnel_spec(m, {"metric": lambda c: jnp.ones(1) / 9.0, "solver": "picard",
                            "solver_params": {"max_iter": 12}}, x_kind="diagonal")
    s = spec.build()
    kv = next(k for k in s.kinetics if k.id == "v")
    assert isinstance(kv.metric, CallableMetric)
    assert kv.solver.max_iter == 12 and type(kv.solver).__name__ == "PicardSolver"
    assert not isinstance(s, HessianSoftnessAdaptation)


@pytest.mark.parametrize("params, err, match", [
    ({"clamp": "softplus", "metric": lambda c: jnp.ones(1)}, ValueError, "would be ignored"),
    ({"softnes": 1.0}, ValueError, "unknown params"),
    ({"clamp": "relu"}, ValueError, "unknown clamp"),
    ({"solver_params": {"depht": 3}}, ValueError, "unknown solver option"),
    ({"metric": lambda c: jnp.ones(2)}, ValueError, "shape"),
    ({"metric": lambda c: -jnp.ones(1)}, ValueError, "positive"),
    ({"metric": 3.0}, TypeError, "callable"),
])
def test_bad_riemannian_params_raise(params, err, match):
    m = neal_funnel_blocks(dim=3).model
    with pytest.raises(err, match=match):
        _funnel_spec(m, params, x_kind="diagonal").build()


def test_dense_given_metric_must_be_spd():
    m = neal_funnel_blocks(dim=3).model
    spec = default_spec(m)
    spec.blocks = [BlockSpec(["v", "x"], [(0, 3)], "riemannian",
                             {"metric": lambda c: jnp.diag(jnp.asarray([1.0, -1.0, 1.0]))})]
    with pytest.raises(ValueError, match="positive definite"):
        spec.build()


def test_tempered_base_builds_with_per_temperature_softness():
    """Under a ``pt_`` base the block keeps per-temperature (independent) selection --- an implicit
    block does not couple the lanes, unlike a line search (``tests/test_pt_riemannian.py``) --- and
    its softness adapts on every rung's own host, against that rung's tempered target."""
    from mimcs.pt import PerTemperatureNUTSMixin
    m = neal_funnel_blocks(dim=3).model
    spec = _funnel_spec(m)
    spec.base = "pt_nuts"
    spec.tempering_params = {"n_temperatures": 3}
    s = spec.build()
    assert isinstance(s, PerTemperatureNUTSMixin)
    assert not isinstance(s, HessianSoftnessAdaptation)         # not on the product chain...
    hosts = s._adapt_hosts
    assert len(hosts) == 3 and all(isinstance(h, HessianSoftnessAdaptation) for h in hosts)
    kv = next(k.inner for k in s.kinetics if k.id == "v")
    assert kv.metric.tempered == {"V_log_prior_v": True, "V_log_lik_x": True}


# --- sampling ------------------------------------------------------------------------------- #

def test_hessian_v_block_with_learned_x_samples_the_funnel(artifacts_dir):
    """The "hyperparameter block" use: ``v`` on the clamped Hessian, ``x`` on the learned
    ``Exp("v")`` metric, through the factory. Checked against the exact funnel.

    Scale 2, not 3: at scale 3 ``x`` is so heavy-tailed (``E x^2 = E e^v`` is carried by rare
    ``v ~ 9`` draws) that the harness's variance and correlation checks fail the *supported*
    diagonal-``v`` path on 4 of 4 seeds as well --- measured, ``rmhmc_corr_check.py``. Here both
    this arm and the given-metric one pass 11 of 12 seed-runs (the one miss: |corr diff| 0.123)."""
    problem = neal_funnel_blocks(dim=2, scale=2.0)
    problem.hard = False
    report = evaluate(problem, {"hessian_v": _builder(_funnel_spec)},
                      n_warmup=1000, n_samples=6000, seed=0,
                      out_dir=str(artifacts_dir / "riemannian_funnel_hessian_v"))
    print("\n" + report.summary())
    report.assert_correct()


def test_hessian_v_block_cuts_divergences_on_the_funnel():
    """What the Hessian ``v`` block buys on Neal's funnel (scale 3): with ``x`` on the learned metric
    in both arms, a constant diagonal ``v`` diverged on 5.5--17.5% of sampling transitions over 8
    seeds (in the neck: median start ``v`` -0.4), the Hessian ``v`` (SoftAbs) on 0.5--3.3% --- nearly
    all of them unsolvable implicit steps, moved out to the mouth --- while ``P(v > 6)`` and
    ``E x^2`` were indistinguishable between the arms
    (``tests/experiments/writeups/implicit_rmhmc_blocks.md``). Asserted at a factor of 3; the paired
    per-seed ratios were 3.3--22 (11 on this seed)."""
    m = neal_funnel_blocks(dim=2, scale=3.0).model
    rates = {}
    for kind in ("diagonal", "riemannian"):
        spec = default_spec(m)
        cb = m.coord_block
        spec.blocks = [BlockSpec(["v"], [cb("v")], kind, {}),
                       BlockSpec(["x"], [cb("x")], "learned_metric", {"metric": Exp("v")})]
        s = spec.build(seed=0)
        s.warmup(1000)
        s.sample(4000)
        rates[kind] = s.divergence_rate()
        if kind == "riemannian":
            # the remaining divergences are (nearly all) failed implicit solves
            assert s.fixed_point_failure_rate() >= 0.8 * rates[kind]
    print(f"\ndivergence rates: {rates}")
    assert rates["riemannian"] < rates["diagonal"] / 3, rates


def _banana_spec(clamp):
    def make(model):
        spec = default_spec(model)
        spec.blocks = [BlockSpec(["x"], [(0, 2)], "riemannian", {"clamp": clamp})]
        return spec
    return make


def test_whole_space_hessian_block_samples_the_banana(artifacts_dir):
    """Rosenbrock's Hessian is indefinite away from the ridge (10--36% of the draws), so the clamp
    is active on much of the path; SoftAbs, which gives a negative curvature mass ``~|lambda|``.
    Correct --- but a third of the transitions end in an unsolvable implicit step (measured 30--40%
    over 4 seeds): the metric turns fast along a banana, and a whole-space Hessian metric is not an
    efficient kinetic for one. It samples it right, which is what is asserted."""
    problem = rosenbrock(a=1.0, b=5.0)
    report = evaluate(problem, {"softabs": _builder(_banana_spec("softabs"))}, n_warmup=1000,
                      n_samples=4000, seed=0,
                      out_dir=str(artifacts_dir / "riemannian_banana_hessian"))
    print("\n" + report.summary())
    report.assert_correct()


def test_softplus_starves_negative_curvature_on_the_banana():
    """The softplus clamp's hazard on an indefinite target: a negative curvature ``-|lambda|`` keeps
    mass ``~e^{-b|lambda|}/b``, and with ``1/b`` adapted to the *positive* curvatures (~0.14 here)
    that is ~0.006 at ``lambda = -0.8``: a near-singular mass, a huge velocity, and an implicit step
    with no solution at almost any step size. Over 4 seeds softplus failed 47--69% of transitions
    with the step adapted down to 0.03--0.14; SoftAbs failed 30--40% at 0.23--0.36 --- lower on
    every paired seed. Asserted on the step size, the sharper separation (11x on this seed)."""
    m = rosenbrock(a=1.0, b=5.0).model
    out = {}
    for clamp in ("softplus", "softabs"):
        s = _banana_spec(clamp)(m).build(seed=0)
        s.warmup(1000)
        s.sample(1000)
        out[clamp] = (float(s.state.step_size), s.fixed_point_failure_rate())
    print(f"\n(step, failure rate): {out}")
    assert out["softabs"][0] > 2.0 * out["softplus"][0], out
    assert out["softabs"][1] < out["softplus"][1], out


def _two_funnels(scale: float = 1.5) -> TargetProblem:
    """Two independent 3-d funnels declared ``v1, x1, v2, x2`` --- so the scales' block ``{v1, v2}``
    is non-contiguous in the coordinate."""
    def lp(p):
        out = 0.0
        for v, x in ((p["v1"], p["x1"]), (p["v2"], p["x2"])):
            out = out - 0.5 * (v / scale) ** 2 - 0.5 * jnp.sum(x ** 2 * jnp.exp(-v)) - v
        return out

    model = Model([EuclideanParameter("v1", ()), EuclideanParameter("x1", (2,)),
                   EuclideanParameter("v2", ()), EuclideanParameter("x2", (2,))], {"lp": lp})

    def sampler(n, rng):
        cols = []
        for _ in range(2):
            v = scale * rng.standard_normal(n)
            cols += [v[:, None], rng.standard_normal((n, 2)) * np.exp(v / 2.0)[:, None]]
        return np.concatenate(cols, axis=1)

    cov = np.diag([scale ** 2, *[np.exp(scale ** 2 / 2)] * 2] * 2)
    return TargetProblem(name="two_funnels", model=model, dim=6,
                         labels=["v1", "x1_0", "x1_1", "v2", "x2_0", "x2_1"],
                         exact_sampler=sampler, mean=np.zeros(6), cov=cov)


def test_fused_noncontiguous_hessian_block_samples_two_funnels(artifacts_dir):
    problem = _two_funnels()

    def make(model):
        spec = default_spec(model)
        cb = model.coord_block
        spec.blocks = [BlockSpec(["v1", "v2"], [cb("v1"), cb("v2")], "riemannian", {}),
                       BlockSpec(["x1"], [cb("x1")], "learned_metric", {"metric": Exp("v1")}),
                       BlockSpec(["x2"], [cb("x2")], "learned_metric", {"metric": Exp("v2")})]
        return spec

    s = make(problem.model).build()
    assert next(k for k in s.kinetics if k.id == "v1__v2").slices == [(0, 1), (3, 4)]
    report = evaluate(problem, {"fused": _builder(make)}, n_warmup=1000, n_samples=4000, seed=0,
                      out_dir=str(artifacts_dir / "riemannian_two_funnels"))
    print("\n" + report.summary())
    report.assert_correct()


def test_given_metric_samples_the_funnel(artifacts_dir):
    """A given metric through ``params["metric"]``: the natural funnel metric over the whole
    coordinate as one riemannian block (``1`` for ``v``, ``e^{-v}`` for ``x``). Scale 2 for the
    reason given in the Hessian-``v`` test."""
    problem = neal_funnel_blocks(dim=2, scale=2.0)
    problem.hard = False

    def make(model):
        spec = default_spec(model)
        spec.blocks = [BlockSpec(["v", "x"], [(0, 2)], "riemannian", {
            "metric": lambda c: jnp.concatenate([jnp.ones(1), jnp.exp(-c["v"])])})]
        return spec

    report = evaluate(problem, {"given": _builder(make)}, n_warmup=1000, n_samples=6000, seed=0,
                      out_dir=str(artifacts_dir / "riemannian_funnel_given"))
    print("\n" + report.summary())
    report.assert_correct()
