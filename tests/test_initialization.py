"""Tests for the sampler initialization pass (``sampler.initialize()`` and its mixins).

``initialize()`` (explicit, before warmup) runs the ``_initialize_hooks`` chain:
:class:`UniformInit` draws the initial *coordinate* from ``U(-2, 2)`` (retrying to finite
density), and :class:`StepSizeLineSearch` backtracks the step size from ``0.5`` to a
single-leapfrog (MALA) acceptance of ``0.9``. Both are inert unless ``initialize()`` is called.

Seeds are fixed, so pass/fail is deterministic.
"""

import numpy as np
import jax.numpy as jnp
import pytest

from mimcs.model import Model, EuclideanParameter, PositiveParameter
from mimcs.factory import make_sampler
from mimcs.samplers import make_sampler_class
from mimcs.hmc import HMC
from mimcs.hmc.integrators import init_integrator_state
from mimcs.adaptation import (
    UniformInit, StepSizeLineSearch, RobbinsMonroStepSize, ScoreMassAdaptation)
from mimcs.testing import correlated_gaussian, evaluate


def _mala_accept_at(s, eps):
    """The single-leapfrog acceptance at ``eps`` from the sampler's initialized state, using the
    same momentum the line search used (so it reproduces what the search saw)."""
    st = s.state
    ctx = s.context(st)
    p = s.sample_momentum(s._init_momentum_draw(), st.coordinate, ctx)
    i0 = init_integrator_state(s.potentials, st.coordinate, p, ctx)
    log_alpha = s.total_energy(i0, ctx) - s.total_energy(
        s.integrator.integrate(i0, jnp.asarray(eps, jnp.float32), 1, ctx), ctx)
    return float(jnp.minimum(1.0, jnp.exp(log_alpha)))


def test_uniform_init_draws_finite_density_coordinate_in_box():
    """The initial coordinate is in [-2, 2]^d, has finite density, and the state is
    self-consistent (sample maps back from the coordinate; log_prob matches)."""
    prob = correlated_gaussian(mean=[5.0, -3.0, 10.0], cov=np.diag([1.0, 4.0, 0.25]))
    s = make_sampler(prob.model, seed=0)
    assert np.allclose(np.asarray(s.state.coordinate), 0.0)      # zeros before initialize()
    s.initialize()

    u = np.asarray(s.state.coordinate)
    assert np.all(np.abs(u) <= 2.0)
    assert np.isfinite(float(s.state.log_prob))
    h, c = s.state.chart_hyperparams, s.state.chart_indices
    x = np.asarray(prob.model.coordinate_to_sample(s.state.coordinate, h, c))
    assert np.allclose(x, np.asarray(s.state.sample), atol=1e-5)
    assert np.isclose(float(s.state.log_prob),
                      float(prob.model.log_prob_at_coordinate(s.state.coordinate, h, c)), atol=1e-3)


def test_uniform_init_bounded_parameter_is_unconstrained():
    """For a bounded (positive) parameter the U(-2,2) draw is in the *unconstrained* coordinate
    (log link); the mapped sample respects the bound."""
    model = Model([PositiveParameter("sig")],
                  {"lp": lambda p: -0.5 * jnp.sum(jnp.log(p["sig"]) ** 2) - jnp.sum(jnp.log(p["sig"]))})
    s = make_sampler(model, seed=1).initialize()
    assert np.all(np.abs(np.asarray(s.state.coordinate)) <= 2.0)   # unconstrained coordinate
    assert np.all(np.asarray(s.state.sample) > 0.0)                # positive sample


def test_uniform_init_retries_until_finite_density():
    """A target that is -inf outside a small region still yields a finite-density start."""
    model = Model([EuclideanParameter("x", (1,))],
                  {"lp": lambda p: jnp.where(jnp.abs(p["x"][0]) < 0.5, -0.5 * p["x"][0] ** 2,
                                             -jnp.inf)})
    s = make_sampler(model, seed=0).initialize()
    assert np.abs(float(s.state.coordinate[0])) < 0.5              # landed in the finite region
    assert np.isfinite(float(s.state.log_prob))


def test_uniform_init_retries_until_the_gradient_is_finite_too():
    """A finite density is not enough: HMC's first leapfrog reads the gradient. The target here is
    the documented `where` trap --- finite everywhere, but for x > 0 the unselected `sqrt(-x)`
    branch makes the gradient NaN --- so half of U(-2, 2) is a start that used to be accepted.
    (On the cscm Gamma GLM a NaN gradient with a finite density crashed the first warmup step.)

    CONTROL: the gradient really is NaN at a positive start, and over these seeds the old
    density-only check accepts at least one such start (verified before the fix: 3 of 6).
    """
    import jax
    lp = lambda p: -0.5 * p["x"][0] ** 2 + jnp.where(p["x"][0] < 0.0, jnp.sqrt(-p["x"][0]), 0.0)
    model = Model([EuclideanParameter("x", (1,))], {"lp": lp})
    assert np.isfinite(float(lp({"x": jnp.array([1.0])})))
    assert np.isnan(float(jax.grad(lambda x: lp({"x": x}))(jnp.array([1.0]))[0]))
    for seed in range(6):
        s = make_sampler(model, seed=seed).initialize()
        assert float(s.state.coordinate[0]) < 0.0, seed               # landed where grad is finite
        assert all(np.all(np.isfinite(np.asarray(g))) for g in s.state.potential_grads.values())


def test_step_size_line_search_hits_mala_target():
    """A sharply-scaled target shrinks the step below 0.5; an easy Gaussian keeps ~0.5. In both
    the single-leapfrog acceptance at the chosen step clears the 0.9 target."""
    narrow = Model([EuclideanParameter("x", (2,))],
                   {"lp": lambda p: -0.5 * jnp.sum((p["x"] / 0.05) ** 2)})
    s = make_sampler(narrow, seed=0).initialize()
    assert float(s.state.step_size) < 0.5
    assert _mala_accept_at(s, float(s.state.step_size)) >= 0.9

    easy = correlated_gaussian(mean=[0.0, 0.0], cov=np.eye(2))
    s2 = make_sampler(easy.model, seed=0).initialize()
    assert float(s2.state.step_size) == pytest.approx(0.5)        # no shrink needed
    assert _mala_accept_at(s2, float(s2.state.step_size)) >= 0.9


def test_initialize_must_precede_warmup():
    prob = correlated_gaussian(mean=[0.0, 0.0], cov=np.eye(2))
    s = make_sampler(prob.model, seed=0)
    s.warmup(3)
    with pytest.raises(RuntimeError, match="before warmup"):
        s.initialize()


def test_not_initializing_is_a_no_op():
    """Without calling initialize() the start is unchanged (the mixins are inert)."""
    prob = correlated_gaussian(mean=[3.0, 3.0], cov=np.eye(2))
    s = make_sampler(prob.model, seed=0)
    assert np.allclose(np.asarray(s.state.coordinate), 0.0)
    assert float(s.state.step_size) == pytest.approx(0.5)


def test_initialized_factory_samples_correctly(artifacts_dir):
    """make_sampler(gaussian).initialize() then warmup/sample is correct (NUTS + init mixins)."""
    prob = correlated_gaussian(mean=[5.0, -3.0], cov=[[2.0, 1.4], [1.4, 1.5]])
    report = evaluate(prob, {"init": lambda m, seed: make_sampler(m, seed=seed).initialize()},
                      n_warmup=2000, n_samples=8000, seed=0,
                      out_dir=str(artifacts_dir / "initialized_factory"))
    print("\n" + report.summary())
    report.assert_correct()


def test_init_mixins_compose_with_hmc():
    """The init mixins also compose with fixed HMC: initialize() sets a boxed position and a
    line-searched step, and the chain then samples."""
    prob = correlated_gaussian(mean=[1.0, -2.0], cov=[[2.0, 1.4], [1.4, 1.5]])
    Cls = make_sampler_class(
        StepSizeLineSearch, UniformInit, RobbinsMonroStepSize, ScoreMassAdaptation, HMC)
    s = Cls(prob.model, init_position=np.zeros(2), seed=0,
            n_leapfrog=20, step_size=0.5, metric="diagonal", target_accept=0.8)
    s.initialize()
    assert np.all(np.abs(np.asarray(s.state.coordinate)) <= 2.0)
    assert _mala_accept_at(s, float(s.state.step_size)) >= 0.9
    s.warmup(1000)
    x = s.sample(4000)["x"]
    assert np.all(np.abs(x.mean(axis=0) - np.array([1.0, -2.0])) < 0.3)


# --------------------------------------------------------------------------- #
# bit-identity of the default path                                            #
# --------------------------------------------------------------------------- #

def test_default_initialize_is_bit_identical_to_the_golden_capture():
    """Nothing supplied and no evidence: ``initialize()`` starts from exactly the coordinate,
    labels and step size it did before initialization became configurable.

    The golden file was written by ``tests/_golden_init_capture.py`` on the unmodified code (and
    this test passed there too), so a change that shifts the default start --- a re-ordered draw,
    a different substream, an overlay that perturbs an unfixed coordinate --- fails here even
    where no seed-pinned statistical test happens to notice.
    """
    import pathlib, sys
    here = pathlib.Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    from _golden_init_capture import golden_cases, snapshot
    golden = np.load(here / "data" / "golden_init.npz")
    for name, make in golden_cases().items():
        got = snapshot(make())
        keys = sorted(k.split("__", 1)[1] for k in golden.files if k.startswith(name + "__"))
        assert keys == sorted(got), (name, keys, sorted(got))
        for k, v in got.items():
            ref = golden[f"{name}__{k}"]
            assert v.dtype == ref.dtype and np.array_equal(v, ref), (name, k, v, ref)


# --------------------------------------------------------------------------- #
# the fixed set: Model.pin_fixed / complete_sample                            #
# --------------------------------------------------------------------------- #

from mimcs.factory import analyze                                          # noqa: E402
from mimcs.model import (BoundedParameter, IntegerParameter, SimplexParameter,  # noqa: E402
                         UnitVectorParameter)


def _parent_model():
    """``x in (0, sigma)``: a chart that depends on a parent, so fixing either side matters."""
    def lp(p):
        return (-0.5 * jnp.log(p["sigma"]) ** 2 - 3.0 * jnp.log(p["sigma"])
                - 0.5 * jnp.sum(p["m"] ** 2))
    return Model([PositiveParameter("sigma"), BoundedParameter("x", (3,), lower=0.0, upper="sigma"),
                  EuclideanParameter("m", (2,))], {"lp": lp})


def _mixture(k=3, per=6):
    rng = np.random.default_rng(0)
    y = np.concatenate([rng.normal(3.0 * (j - (k - 1) / 2), 1.0, per) for j in range(k)])

    def lp(v):
        return -0.5 * jnp.sum((y - v["mu"][v["z"]]) ** 2) - 0.5 * jnp.sum(v["mu"] ** 2)

    return Model([EuclideanParameter("mu", (k,))], {"p": lp},
                 discrete_parameters=[IntegerParameter("z", (k * per,), lower=0, upper=k - 1)])


def _slice(model, name, flat, coord=False):
    i = [p.name for p in model.parameters].index(name)
    off = model._coord_offsets if coord else model._ambient_offsets
    return np.asarray(flat)[off[i]:off[i + 1]]


@pytest.mark.parametrize("case", ["parent_bounds", "unit_vector", "simplex", "centered"])
def test_pin_fixed_keeps_fixed_samples_exactly_and_free_coordinates_untouched(case):
    """The overlay primitive, checked against the slow reference (``coordinate_to_sample`` of the
    result) on every chart shape that matters: a parent-dependent bound, a unit vector (ambient
    and coordinate dims differ), a simplex and a centred chart.

    CONTROL: a naive coordinate-space mask --- each fixed value's coordinate computed under the
    *original* parents --- is measurably wrong exactly where a free parent moves (it was 2.8 off
    on the bound and 0.38 on the centred chart when this was derived), so the topological walk is
    what makes the fixed child's sample come back right.
    """
    lp = {"lp": lambda v: 0.0}
    model, child = {
        "parent_bounds": (_parent_model(), "x"),
        "unit_vector": (Model([UnitVectorParameter("u", 4), EuclideanParameter("m", (2,))], lp),
                        None),
        "simplex": (Model([SimplexParameter("w", 4), PositiveParameter("s")], lp), None),
        "centered": (Model([PositiveParameter("tau", centered=True),
                            BoundedParameter("y", (2,), lower="tau", centered=True)], lp), "y"),
    }[case]
    import jax
    h, idx = model.init_chart_hyperparams(), model.init_chart_indices()
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    g = jax.random.uniform(k1, (model.coord_dim,), minval=-2, maxval=2)
    u = jax.random.uniform(k2, (model.coord_dim,), minval=-2, maxval=2)
    if case == "parent_bounds":
        # A fixed x must fit under the *new* sigma, or its coordinate is (rightly) NaN: draw the
        # given point under a small sigma (e^-1) and the free one under a larger one (e^1).
        g, u = g.at[0].set(-1.0), u.at[0].set(1.0)
    given = model.coordinate_to_sample(g, h, idx)
    names = [p.name for p in model.parameters]
    for fixed in [{n} for n in names] + [set(names)]:
        coord, sample = model.pin_fixed(u, given, fixed, h, idx)
        assert np.allclose(model.coordinate_to_sample(coord, h, idx), sample, atol=1e-6)
        for n in names:
            if n in fixed:
                assert np.array_equal(_slice(model, n, sample), _slice(model, n, given)), n
            else:
                assert np.array_equal(_slice(model, n, coord, True), _slice(model, n, u, True)), n
    if child is not None:                              # the control: a free parent, fixed child
        full = model.sample_to_coordinate(given, h, idx)
        naive = np.asarray(u).copy()
        i = names.index(child)
        naive[model._coord_offsets[i]:model._coord_offsets[i + 1]] = np.asarray(full)[
            model._coord_offsets[i]:model._coord_offsets[i + 1]]
        wrong = _slice(model, child, model.coordinate_to_sample(jnp.asarray(naive), h, idx))
        assert np.max(np.abs(wrong - _slice(model, child, given))) > 0.1


def test_complete_sample_refuses_values_outside_their_support():
    """The round-trip check: a bound violation, a value on a bound, a child outside its parent's
    bound, a unit vector off the sphere and a simplex not summing to one all raise; valid values
    pass (the control)."""
    m = _parent_model()
    out = m.complete_sample({"sigma": 2.0, "x": [0.5, 1.5, 0.2]})       # valid: no raise
    assert np.array_equal(_slice(m, "x", out), np.asarray([0.5, 1.5, 0.2], np.float32))
    for bad, name in (({"sigma": -1.0}, "sigma"), ({"x": [0.0, 0.5, 0.5]}, "x"),
                      ({"sigma": 1.0, "x": [0.5, 1.5, 0.2]}, "x")):
        with pytest.raises(ValueError, match=f"'{name}'.*outside its support"):
            m.complete_sample(bad)
    with pytest.raises(ValueError, match="given its parent"):
        m.complete_sample({"sigma": 1.0, "x": [0.5, 1.5, 0.2]})
    lp = {"lp": lambda v: 0.0}
    with pytest.raises(ValueError, match="'u'"):
        Model([UnitVectorParameter("u", 3)], lp).complete_sample({"u": [1.0, 1.0, 0.0]})
    with pytest.raises(ValueError, match="'w'"):
        Model([SimplexParameter("w", 3)], lp).complete_sample({"w": [0.5, 0.5, 0.5]})
    with pytest.raises(ValueError, match="not parameters of this model"):
        m.complete_sample({"sgima": 1.0})
    with pytest.raises(ValueError, match="element"):
        m.complete_sample({"x": [0.1, 0.2]})


def test_an_unnamed_child_of_a_named_parent_is_rechartered_into_its_new_bounds():
    """``x`` unnamed, ``sigma`` named: ``x`` keeps the base *coordinate*, so under the new
    ``sigma`` it is inside ``(0, sigma)`` --- not the old value, which may no longer be valid."""
    m = _parent_model()
    base = m.complete_sample({"sigma": 10.0, "x": [9.0, 8.0, 7.0]})
    out = m.complete_sample({"sigma": 0.5}, base_sample=base)
    x = _slice(m, "x", out)
    assert np.all((x > 0) & (x < 0.5)), x


# --------------------------------------------------------------------------- #
# initialize() keeps the fixed set                                            #
# --------------------------------------------------------------------------- #

def test_partial_init_keeps_the_supplied_values_and_draws_exactly_the_rest():
    """``sigma`` supplied: it is bit-exact after ``initialize()``, and every *free* coordinate is
    exactly what an unfixed run with the same seed draws (the overlay comes after the full draw).

    CONTROL: the unfixed run's ``sigma`` is not 3.0, so "kept" is not vacuous.
    """
    m = _parent_model()
    fixed = analyze(m).build(seed=3, init={"sigma": 3.0})
    assert fixed.init_fixed == {"sigma"}
    fixed.initialize()
    free = analyze(m).build(seed=3).initialize()
    assert _slice(m, "sigma", fixed.state.sample)[0] == np.float32(3.0)
    assert _slice(m, "sigma", free.state.sample)[0] != np.float32(3.0)
    for n in ("x", "m"):
        assert np.array_equal(_slice(m, n, fixed.state.coordinate, True),
                              _slice(m, n, free.state.coordinate, True)), n
    x = _slice(m, "x", fixed.state.sample)
    assert np.all((x > 0) & (x < 3.0))
    assert np.isfinite(float(fixed.state.log_prob))


def test_a_fixed_child_of_a_drawn_parent_stays_put_and_in_support():
    """``x`` supplied, ``sigma`` drawn: the retry loop only accepts a ``sigma`` above every ``x``
    (anything else makes ``x``'s coordinate non-finite), and ``x`` is kept exactly. The supplied
    ``x`` must be valid under the *default* ``sigma`` (1) too, since that is where the built
    sampler starts; one that is not raises at build, naming the parent to supply."""
    m = _parent_model()
    with pytest.raises(ValueError, match="supply them in init"):
        analyze(m).build(init={"x": [0.9, 1.2, 0.4]})
    x0 = np.asarray([0.9, 0.6, 0.4], np.float32)
    for seed in range(4):
        s = analyze(m).build(seed=seed, init={"x": x0}).initialize()
        assert np.array_equal(_slice(m, "x", s.state.sample), x0)
        assert _slice(m, "sigma", s.state.sample)[0] > x0.max()
        assert np.isfinite(float(s.state.log_prob))


def test_a_full_array_init_is_kept_and_the_step_size_is_still_searched():
    """The case from the request: a complete start, and ``initialize()`` for the step only."""
    narrow = Model([EuclideanParameter("x", (2,))],
                   {"lp": lambda p: -0.5 * jnp.sum((p["x"] / 0.05) ** 2)})
    x0 = np.asarray([0.03, -0.02], np.float32)
    s = analyze(narrow).build(seed=0, init=x0)
    assert s.init_fixed == {"x"}
    s.initialize()
    assert np.array_equal(np.asarray(s.state.sample), x0)
    assert float(s.state.step_size) < 0.5                       # the search ran
    assert _mala_accept_at(s, float(s.state.step_size)) >= 0.9


def test_the_two_slots_switch_independently():
    """All four combinations of ``position_init`` x ``step_size_init`` on a stiff target."""
    narrow = Model([EuclideanParameter("x", (2,))],
                   {"lp": lambda p: -0.5 * jnp.sum((p["x"] / 0.05) ** 2)})
    for pos in ("uniform", None):
        for step in ("line_search", None):
            spec = analyze(narrow)
            spec.position_init, spec.step_size_init = pos, step
            s = spec.build(seed=0)
            mro = [c.__name__ for c in type(s).__mro__]
            assert ("UniformInit" in mro) == (pos is not None)
            assert ("StepSizeLineSearch" in mro) == (step is not None)
            s.initialize()
            moved = not np.allclose(np.asarray(s.state.coordinate), 0.0)
            assert moved == (pos is not None), (pos, step)
            searched = float(s.state.step_size) < 0.5
            assert searched == (step is not None), (pos, step)


def test_a_spec_pickled_before_the_init_slots_existed_still_builds():
    """An older pickle has no ``*_init`` fields; unpickling gives them their defaults."""
    import pickle
    from mimcs.factory import SamplerSpec
    narrow = Model([EuclideanParameter("x", (2,))],
                   {"lp": lambda p: -0.5 * jnp.sum((p["x"] / 0.05) ** 2)})
    state = dict(analyze(narrow).__dict__, model=None)          # pickled without its model
    added = ("position_init", "position_init_params", "step_size_init", "step_size_init_params",
             "metric_regression")
    for name in added:
        del state[name]
    old = SamplerSpec.__new__(SamplerSpec)                      # what an older pickle held
    old.__dict__.update(state)
    with pytest.raises(AttributeError):                         # non-vacuous: really missing
        old.position_init_params
    spec = pickle.loads(pickle.dumps(old))
    assert (spec.position_init, spec.position_init_params) == ("uniform", {})
    assert (spec.step_size_init, spec.step_size_init_params) == ("line_search", {})
    assert spec.metric_regression == {}
    spec.model = narrow
    assert "uniform + line_search" in str(spec)
    spec.build(seed=0).initialize()


def test_init_slot_options_reach_the_mixins_and_are_validated():
    """``position_init_params`` / ``step_size_init_params`` are translated to the mixin kwargs;
    unknown initializers or options, options for a ``None`` slot, and the old ``algo_kwargs``
    keys all raise."""
    prob = correlated_gaussian(mean=[0.0, 0.0, 0.0], cov=np.eye(3))
    spec = analyze(prob.model)
    spec.position_init_params = {"radius": 0.25}
    spec.step_size_init_params = {"target_accept": 0.99}
    s = spec.build(seed=0)
    assert (s._init_radius, s._init_target_accept) == (0.25, 0.99)
    s.initialize()
    assert np.all(np.abs(np.asarray(s.state.coordinate)) <= 0.25)
    assert "uniform(radius=0.25) + line_search(target_accept=0.99)" in str(spec)

    cases = [
        (dict(position_init="pathfinder"), "unknown position_init"),
        (dict(step_size_init="doubling"), "unknown step_size_init"),
        (dict(position_init_params={"radus": 1.0}), "unknown position_init_params"),
        (dict(step_size_init_params={"target": 0.9}), "unknown step_size_init_params"),
        (dict(position_init=None, position_init_params={"radius": 1.0}), "is None"),
        (dict(position_init_params={"radius": -1.0}), "init_radius must be"),
        (dict(step_size_init_params={"target_accept": 1.5}), "init_target_accept must be"),
        (dict(algo_kwargs={"init_radius": 1.0}), r"position_init_params\['radius'\]"),
        (dict(algo_kwargs={"init_target_accept": 0.95}), "step_size_init_params"),
        (dict(algo_kwargs={"init_fixed": ("x",)}), "build\\(init"),
    ]
    for changes, message in cases:
        spec = analyze(prob.model)
        for k, v in changes.items():
            setattr(spec, k, v)
        with pytest.raises(ValueError, match=message):
            spec.build(seed=0)


def test_build_refuses_a_bad_init_at_build_time():
    m = _parent_model()
    with pytest.raises(ValueError, match="not parameters of this model"):
        analyze(m).build(init={"sgima": 1.0})
    with pytest.raises(ValueError, match="outside its support"):
        analyze(m).build(init={"sigma": -2.0})
    with pytest.raises(ValueError, match="ambient_dim"):
        analyze(m).build(init=np.zeros(4))


def test_a_fully_supplied_point_with_no_finite_density_is_refused_by_initialize():
    """Nothing to draw, and nowhere to start: ``initialize()`` says so rather than letting the
    first leapfrog fail."""
    model = Model([EuclideanParameter("x", (1,))],
                  {"lp": lambda p: jnp.where(jnp.abs(p["x"][0]) < 0.5, -0.5 * p["x"][0] ** 2,
                                             -jnp.inf)})
    s = analyze(model).build(seed=0, init={"x": [2.0]})
    with pytest.raises(RuntimeError, match="every continuous parameter was supplied"):
        s.initialize()
    ok = analyze(model).build(seed=0, init={"x": [0.2]}).initialize()      # control
    assert float(ok.state.sample[0]) == np.float32(0.2)


def test_hand_built_samplers_infer_the_fixed_set_from_a_dict_only():
    """An array is required by the constructor even when the caller has no opinion, so only a
    dict (or an explicit ``init_fixed``) fixes anything; a typo in ``init_fixed`` raises."""
    m = _parent_model()
    Cls = make_sampler_class(StepSizeLineSearch, UniformInit, RobbinsMonroStepSize, HMC)
    kw = dict(seed=0, n_leapfrog=5, step_size=0.5, metric="diagonal")
    assert Cls(m, init_position={"sigma": 2.0}, **kw).init_fixed == {"sigma"}
    assert Cls(m, init_position=m.default_sample(), **kw).init_fixed == frozenset()
    s = Cls(m, init_position=m.default_sample(), init_fixed=["m"], **kw).initialize()
    assert np.array_equal(_slice(m, "m", s.state.sample), np.zeros(2, np.float32))
    with pytest.raises(ValueError, match="init_fixed names"):
        Cls(m, init_position=m.default_sample(), init_fixed=["mm"], **kw)


# --------------------------------------------------------------------------- #
# labels                                                                      #
# --------------------------------------------------------------------------- #

def test_supplied_labels_are_kept_and_the_rest_drawn():
    """``z`` supplied: kept exactly, ``mu`` drawn. ``mu`` supplied: kept, ``z`` drawn (CONTROL:
    the drawn labels are not the supplied pattern). ``position_init=None``: both stay as built."""
    m = _mixture()
    z0 = np.tile([2, 1, 0], 6).astype(np.int32)
    s = analyze(m).build(seed=1, init={"z": z0}).initialize()
    assert np.array_equal(np.asarray(s.state.discrete), z0)
    assert not np.allclose(np.asarray(s.state.sample), 0.0)

    mu0 = np.asarray([-3.0, 0.0, 3.0], np.float32)
    s = analyze(m).build(seed=1, init={"mu": mu0}).initialize()
    assert np.array_equal(np.asarray(s.state.sample), mu0)
    assert not np.array_equal(np.asarray(s.state.discrete), np.zeros_like(z0))

    spec = analyze(m)
    spec.position_init = None
    s = spec.build(seed=1, init={"mu": mu0}).initialize()
    assert np.array_equal(np.asarray(s.state.sample), mu0)
    assert np.array_equal(np.asarray(s.state.discrete), np.zeros_like(z0))   # default, kept


def test_a_static_base_keeps_supplied_labels_and_has_no_step_to_search():
    """Discrete-only: the factory turns ``step_size_init`` off; ``position_init`` still governs
    the label draw, and a partial label init is honoured."""
    m = Model([], {"p": lambda v: jnp.sum(v["a"].astype(float)) - jnp.sum(v["b"].astype(float))},
              discrete_parameters=[IntegerParameter("a", (4,), lower=0, upper=3),
                                   IntegerParameter("b", (4,), lower=0, upper=3)])
    spec = analyze(m)
    assert spec.base == "static" and spec.step_size_init is None
    a0 = np.asarray([3, 3, 3, 3], np.int32)
    s = spec.build(seed=0, init={"a": a0}).initialize()
    z = np.asarray(s.state.discrete)
    assert np.array_equal(z[:4], a0) and not np.array_equal(z[4:], np.zeros(4))
    spec.position_init = None
    s = spec.build(seed=0, init={"a": a0}).initialize()
    assert np.array_equal(np.asarray(s.state.discrete), np.r_[a0, np.zeros(4, np.int32)])


# --------------------------------------------------------------------------- #
# parallel tempering                                                          #
# --------------------------------------------------------------------------- #

def test_a_tempered_build_takes_a_partial_dict_and_every_rung_keeps_it():
    """A dict ``init`` used to crash the tempered path (``np.asarray(dict)``). Now each rung starts
    from it: supplied labels in every lane, the continuous half drawn per rung; and supplied ``mu``
    in every rung with each lane's labels drawn."""
    from mimcs.pt import parallel_tempering
    m, K = _mixture(), 3
    z0 = np.tile([2, 1, 0], 6).astype(np.int32)
    spec = analyze(m)
    spec.base, spec.tempering_params = "pt_nuts", {"n_temperatures": K}
    s = spec.build(seed=2, init={"z": z0}).initialize()
    assert np.array_equal(np.asarray(s.state.discrete), np.tile(z0, K))
    rows = np.asarray(s.state.sample).reshape(K, -1)
    assert not np.allclose(rows[0], rows[1])                   # each rung drew its own mu

    mu0 = np.asarray([-3.0, 0.0, 3.0], np.float32)
    s = spec.build(seed=2, init={"mu": mu0}).initialize()
    assert np.array_equal(np.asarray(s.state.sample), np.tile(mu0, K))
    lanes = np.asarray(s.state.discrete).reshape(K, -1)
    assert not np.array_equal(lanes[0], lanes[1])              # each lane drew its own labels

    hand = parallel_tempering(m, {"mu": mu0, "z": z0}, n_temperatures=K, seed=0)
    assert hand.init_fixed == {"mu", "z"}
    assert np.array_equal(np.asarray(hand.state.discrete), np.tile(z0, K))


# --------------------------------------------------------------------------- #
# the warm start                                                              #
# --------------------------------------------------------------------------- #

def test_evidence_turns_the_position_initializer_off_so_the_warm_start_survives():
    """``analyze(model, run)`` proposes ``position_init=None``: ``initialize()`` then keeps the
    last evidence draw and only searches the step. CONTROL: forcing ``"uniform"`` back redraws."""
    prob = correlated_gaussian(mean=[5.0, -3.0], cov=np.eye(2))
    pilot = make_sampler(prob.model, seed=0)
    pilot.warmup(60)
    pilot.sample(30)
    last = np.asarray(pilot.get_samples_flat()[-1], np.float32)
    spec = analyze(prob.model, pilot)
    assert spec.position_init is None
    assert any("warm_start_init" in r for r in spec.rationale)
    s = spec.build(seed=1)
    assert s.init_fixed == frozenset()
    s.initialize()
    assert np.array_equal(np.asarray(s.state.sample), last)
    spec.position_init = "uniform"
    s = spec.build(seed=1).initialize()
    assert not np.allclose(np.asarray(s.state.sample), last)
