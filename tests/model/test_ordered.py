"""Tests for ``OrderedParameter``: a strictly increasing vector, optionally bounded at the ends.

Unlike the sphere and the simplex, ordering costs no dimension --- an increasing vector is an
open subset of ``R^d`` --- so there are ``d`` coordinates for ``d`` components and the default
features and Stein terms apply. What has to be checked is the four link regimes, which differ by
which bounds are present, and their log-Jacobians.

Two of them are worth singling out. The **upper-only** chart is the reflection of the
lower-only one, mirroring the reflected-log link on a scalar, and the test below pins that
relationship rather than just the result. The **doubly-bounded** chart is stick breaking: the
``d + 1`` gaps of ``L < x_1 < ... < x_d < U`` are positive and sum to ``U - L``, so the object
*is* a scaled simplex. ``ordered_uniform`` is the sharp end-to-end test of it --- a constant
density, so the Jacobian alone has to produce sorted uniforms.

Seeds are fixed, so pass/fail is deterministic.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from mimcs.model import Model, OrderedParameter, PositiveParameter
from mimcs.dsl import DslError
from mimcs import compile_model
from mimcs.testing import ordered_normal, ordered_uniform, evaluate, nuts


#: The four regimes, as ``(lower, upper)``.
BOUNDS = [(None, None), (0.5, None), (None, 2.0), (-1.0, 3.0)]
BOUND_IDS = ["unbounded", "lower", "upper", "both"]


# --- the chart --------------------------------------------------------------- #

def test_dimensions_and_shapes():
    p = OrderedParameter("x", 4)
    assert p.ambient_shape == (4,) and p.coord_dim == 4
    q = OrderedParameter("x", 3, (5, 2))
    assert q.ambient_shape == (5, 2, 3) and q.coord_dim == 30


def test_ordered_needs_at_least_one_component():
    with pytest.raises(ValueError, match="d >= 1"):
        OrderedParameter("x", 0)


@pytest.mark.parametrize("lower, upper", BOUNDS, ids=BOUND_IDS)
@pytest.mark.parametrize("d", [1, 2, 4])
@pytest.mark.parametrize("batch", [(), (3,)])
def test_chart_round_trip_ordering_and_bounds(lower, upper, d, batch):
    p = OrderedParameter("x", d, batch, lower=lower, upper=upper)
    rng = np.random.default_rng(0)
    y = jnp.asarray(rng.normal(size=(p.coord_dim,)), float)
    x = p.from_coordinate(y)
    xn = np.asarray(x)

    assert x.shape == p.ambient_shape
    assert np.all(np.diff(xn, axis=-1) > 0.0), f"not increasing: {xn}"
    if lower is not None:
        assert np.all(xn[..., 0] > lower)
    if upper is not None:
        assert np.all(xn[..., -1] < upper)
    assert np.allclose(np.asarray(p.to_coordinate(x)), np.asarray(y), atol=1e-4)


@pytest.mark.parametrize("lower, upper", BOUNDS, ids=BOUND_IDS)
@pytest.mark.parametrize("d", [1, 2, 4])
def test_log_jacobian_matches_autodiff(lower, upper, d):
    p = OrderedParameter("x", d, lower=lower, upper=upper)
    rng = np.random.default_rng(1)
    y = jnp.asarray(rng.normal(size=(d,)), float)
    dense = jax.jacfwd(p.from_coordinate)(y)
    assert np.allclose(float(p.log_jacobian_det(y)),
                       float(jnp.log(jnp.abs(jnp.linalg.det(dense)))), atol=1e-3)


@pytest.mark.parametrize("d", [1, 3, 5])
def test_coordinate_origin_is_evenly_spaced_when_doubly_bounded(d):
    """``Model.default_sample`` evaluates every chart at ``0``; here that is the even spacing."""
    p = OrderedParameter("x", d, lower=0.0, upper=1.0)
    x0 = np.asarray(p.from_coordinate(jnp.zeros((d,), float)))
    assert np.allclose(x0, np.arange(1, d + 1) / (d + 1.0), atol=1e-6)


def test_coordinate_origin_is_unit_spaced_when_unbounded():
    p = OrderedParameter("x", 4)
    assert np.allclose(np.asarray(p.from_coordinate(jnp.zeros((4,), float))),
                       [0.0, 1.0, 2.0, 3.0], atol=1e-6)


def test_the_upper_only_chart_is_the_reflection_of_the_lower_only_one():
    """``x -> U - x`` maps one to the other, reversed --- the relationship, not just the result.

    This is the same mirror the reflected-log link has to the log link on a scalar bounded
    parameter, which is why the two regimes share an implementation up to a flip.
    """
    d, u = 4, 2.0
    lo = OrderedParameter("x", d, lower=0.0)
    up = OrderedParameter("x", d, upper=u)
    y = jnp.asarray(np.random.default_rng(2).normal(size=(d,)), float)

    x_lo = np.asarray(lo.from_coordinate(y))
    x_up = np.asarray(up.from_coordinate(y))
    assert np.allclose(x_up, u - x_lo[::-1], atol=1e-5)
    # A reflection is an isometry, so the two carry the same log-Jacobian.
    assert np.isclose(float(lo.log_jacobian_det(y)), float(up.log_jacobian_det(y)), atol=1e-4)


def test_a_small_gap_is_still_strictly_ordered():
    """A gap far below the bulk scale must still separate two entries.

    It also shows where the precision goes. ``exp(-12) = 6.1e-6`` sits on top of an entry of
    order one, where float32 resolves ``1.2e-7`` --- so the gap is stored in about 52 ulps, and
    only the leading two digits of it survive. The ordering is intact and ``from_coordinate`` is
    exact, but recovering the *coordinate* means taking a log of those two digits, so ``y`` comes
    back with roughly a percent of quantization error. Nothing is wrong; this is the float32
    budget, and it is why the round trip is checked at ordinary scales elsewhere.
    """
    p = OrderedParameter("x", 3, lower=0.0)
    y = jnp.asarray([0.0, -12.0, 0.0], float)
    x = np.asarray(p.from_coordinate(y))

    assert np.all(np.isfinite(x))
    assert np.all(np.diff(x) > 0.0), f"a representable gap must still separate entries: {x}"
    back = np.asarray(p.to_coordinate(jnp.asarray(x, float)))
    assert np.allclose(back, np.asarray(y), atol=0.05), f"{back} vs {y}"


def test_a_gap_below_the_floating_point_spacing_is_absorbed_but_stays_sane():
    """The limit of the representation, pinned so it is a known property rather than a surprise.

    At ``y_k = -30`` the gap is ``exp(-30) ~ 9e-14``. Added to an entry of order one it simply
    disappears --- float32 resolves about ``1.2e-7`` there --- so the two entries come out equal
    and the ordering is no longer *strict*. No chart can avoid this: the two values are the same
    float. What must not happen is a ``nan``, an ``inf``, or an out-of-order pair, and the
    log-Jacobian must stay exact, which it does because it is computed from the coordinate and
    never from the differences. Enabling x64 pushes the threshold down by ~9 orders of magnitude.
    """
    p = OrderedParameter("x", 3, lower=0.0)
    y = jnp.asarray([0.0, -30.0, 0.0], float)
    x = np.asarray(p.from_coordinate(y))

    assert np.all(np.isfinite(x))
    assert np.all(np.diff(x) >= 0.0), f"absorption must not reorder: {x}"
    assert np.isfinite(float(p.log_jacobian_det(y)))
    # exact, and unaffected by the absorption: sum(y) for a lower-bounded ordered vector
    assert np.isclose(float(p.log_jacobian_det(y)), -30.0, atol=1e-3)


@pytest.mark.parametrize("lower, upper", BOUNDS, ids=BOUND_IDS)
def test_model_gradient_matches_finite_difference(lower, upper):
    """Autodiff through ``log_prob_at_coordinate`` in every regime --- what HMC differentiates."""
    model = Model([OrderedParameter("x", 3, lower=lower, upper=upper)],
                  {"m": lambda p: -0.5 * jnp.sum(p["x"] ** 2)})
    h, c = model.init_chart_hyperparams(), model.init_chart_indices()

    def f(q):
        return model.log_prob_at_coordinate(q, h, c)

    q0 = np.array([0.2, -0.5, 0.35])
    grad = np.asarray(jax.grad(f)(jnp.asarray(q0, float)))
    eps = 1e-3
    fd = np.array([
        (float(f(jnp.asarray(q0 + eps * e, float)))
         - float(f(jnp.asarray(q0 - eps * e, float)))) / (2 * eps)
        for e in np.eye(3)])
    assert np.all(np.isfinite(grad))
    assert np.allclose(grad, fd, atol=2e-3), f"grad {grad} != fd {fd}"


def test_parent_dependent_bounds_thread_through_the_model():
    """A bound may be another parameter's value, exactly as for ``BoundedParameter``."""
    p = OrderedParameter("c", 3, lower=0.0, upper="s")
    assert p.parents == ("s",)

    model = Model([PositiveParameter("s"), p],
                  {"m": lambda v: -v["s"] - 0.5 * jnp.sum(v["c"] ** 2)})
    h, c = model.init_chart_hyperparams(), model.init_chart_indices()
    q = jnp.asarray([0.4, 0.1, -0.3, 0.2], float)
    values = model.unpack_coordinate(q, h, c)
    x, s = np.asarray(values["c"]), float(values["s"])
    assert np.all(np.diff(x) > 0.0) and x[0] > 0.0 and x[-1] < s

    grad = np.asarray(jax.grad(lambda z: model.log_prob_at_coordinate(z, h, c))(q))
    assert np.all(np.isfinite(grad))


# --- features ---------------------------------------------------------------- #

def test_features_are_the_default_two_per_component():
    """Ordering costs no dimension, so nothing has to be dropped for rank."""
    p = OrderedParameter("x", 3)
    assert p.n_features == 6
    assert p.feature_names() == ["x[0]", "x[1]", "x[2]", "x[0]^2", "x[1]^2", "x[2]^2"]
    assert p.ambient_names() == ["x[0]", "x[1]", "x[2]"]


# --- the DSL ----------------------------------------------------------------- #

def test_dsl_ordered_matches_the_hand_built_parameter():
    model = compile_model("parameters { ordered[3] c; }\nmodel { target += 0.0; }", {})
    p = model.parameters[0]
    assert isinstance(p, OrderedParameter)
    assert p.d == 3 and p.ambient_shape == (3,) and p.coord_dim == 3


@pytest.mark.parametrize("decl, has_lower, has_upper", [
    ("ordered[3] c", False, False),
    ("ordered<lower=0>[3] c", True, False),
    ("ordered<upper=10>[3] c", False, True),
    ("ordered<lower=0, upper=1>[3] c", True, True),
])
def test_dsl_bounds_go_before_the_size(decl, has_lower, has_upper):
    """Stan's order: the bounds constrain the type, the brackets size it."""
    model = compile_model(f"parameters {{ {decl}; }}\nmodel {{ target += 0.0; }}", {})
    p = model.parameters[0]
    assert p._has_lower is has_lower and p._has_upper is has_upper


def test_dsl_array_of_ordered_vectors():
    model = compile_model(
        "data { int n; }\nparameters { array[n, 2] ordered<lower=0, upper=1>[3] c; }\n"
        "model { target += 0.0; }", {"n": 4})
    p = model.parameters[0]
    assert p.ambient_shape == (4, 2, 3) and p.coord_dim == 24


def test_dsl_ordered_bound_may_be_a_parent():
    model = compile_model(
        "parameters { real<lower=0> s; ordered<lower=0, upper=s>[3] c; }\n"
        "model { target += -s; }", {})
    assert model.parameters[1].parents == ("s",)


def test_dsl_ordered_density_matches_the_hand_built_model():
    src = ("parameters { ordered<lower=0>[3] c; }\n"
           "model { target += -0.5 * (c[1]*c[1] + c[2]*c[2] + c[3]*c[3]); }")
    model = compile_model(src, {})
    hand = Model([OrderedParameter("c", 3, lower=0.0)],
                 {"m": lambda p: -0.5 * jnp.sum(p["c"] ** 2)})
    q = jnp.asarray([0.3, -0.7, 0.2], float)
    h, c = model.init_chart_hyperparams(), model.init_chart_indices()
    assert np.isclose(float(model.log_prob_at_coordinate(q, h, c)),
                      float(hand.log_prob_at_coordinate(q, h, c)), atol=1e-4)


@pytest.mark.parametrize("src, match", [
    ("parameters { ordered c; }\nmodel { target += 0; }", "needs a size"),
    ("parameters { ordered[0] c; }\nmodel { target += 0; }", "degenerate"),
    ("parameters { ordered[3,2] c; }\nmodel { target += 0; }", "one size"),
    ("parameters { ordered[3]<lower=0> c; }\nmodel { target += 0; }", "before the size"),
    ("data { ordered[3] d; }\nparameters { real x; }\nmodel { target += x; }",
     "`parameters` block"),
    ("functions { real f(ordered[3] c) { return c[1]; } }", "`parameters` block"),
])
def test_dsl_ordered_errors(src, match):
    with pytest.raises(DslError, match=match):
        compile_model(src, {})


def test_existing_bound_syntax_still_parses():
    """Bounds moved into the type; the scalar forms that already used that position must hold."""
    model = compile_model(
        "data { int n; }\n"
        "parameters { real<lower=0> s; array[n] real<lower=0, upper=1> t; }\n"
        "model { target += -s; }", {"n": 3})
    assert [p.name for p in model.parameters] == ["s", "t"]
    assert model.parameters[1].ambient_shape == (3,)


# --- end to end -------------------------------------------------------------- #

def test_ordered_uniform_sampling_is_correct(artifacts_dir):
    """A constant density: the doubly-bounded stick-breaking Jacobian alone has to produce it."""
    problem = ordered_uniform(4)
    report = evaluate(problem, {"nuts": nuts()}, n_warmup=2000, n_samples=20000, seed=0,
                      out_dir=artifacts_dir / "ordered_uniform")
    print("\n" + report.summary())
    report.assert_correct()

    draws = report.outputs["nuts"].samples
    assert np.all(np.diff(draws, axis=1) > 0.0)
    assert np.all(draws > 0.0) and np.all(draws < 1.0)
    # the k-th order statistic of d uniforms has mean k/(d+1)
    assert np.allclose(draws.mean(0), np.arange(1, 5) / 5.0, atol=0.01)


def test_ordered_normal_sampling_is_correct(artifacts_dir):
    """The ordering is carried entirely by the chart: the density is plain i.i.d. normal."""
    problem = ordered_normal(4)
    report = evaluate(problem, {"nuts": nuts()}, n_warmup=2000, n_samples=20000, seed=0,
                      out_dir=artifacts_dir / "ordered_normal")
    print("\n" + report.summary())
    report.assert_correct()
    assert np.all(np.diff(report.outputs["nuts"].samples, axis=1) > 0.0)
