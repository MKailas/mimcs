"""Tests for ``UnitVectorParameter``: a point on ``S^(d-1)`` in an adaptive stereographic chart.

This is the first parameter whose coordinate space has a *lower dimension* than its ambient
(sample) space --- ``d - 1`` against ``d`` --- so several of the suite's usual patterns do not
apply here. Most visibly, ``from_coordinate`` has a **non-square** ``(d, d-1)`` Jacobian, so the
change of variables is the *Gram* determinant ``0.5 log det(J^T J)`` rather than
``log|det J|`` (cf. ``test_centering.py``, where the map is a bijection of ``R^d``).

The chart is ``u = s * (H_v x)[:-1] / (1 - (H_v x)[-1])``, with the reflection ``H_v`` placing
the pole ``p = H_v e_d`` and ``s`` scaling so that the sphere-plane intersection is the unit
circle. The identity ``|u| < 1  <=>  <x, p> < c``, ``c = (1-s^2)/(1+s^2)``, is what makes the
adaptation's quantile target meaningful, so it is tested directly.

The pole is fitted intrinsically, by stepping along the geodesic toward each draw's antipode
(the Frechet mean of the antipodal draws). Two properties are worth testing on their own: that
the step really is a geodesic of the right arc length, and that the great-circle case --- which
an extrinsic ``-normalize(E[x])`` estimator is blind to, since ``E[x] = 0`` there --- puts the
pole on the circle's axis. Note what is *not* asserted: that the pole converges. On a uniform
target it has nothing to converge to, and the guarantee is diminishing adaptation instead.

Seeds are fixed, so pass/fail is deterministic.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from mimcs.model import Model, EuclideanParameter, UnitVectorParameter, SphereChart
from mimcs.adaptation._stochastic import rm_gain
from mimcs.adaptation.unit_vector import _geodesic_step
from mimcs.dsl import DslError
from mimcs import compile_model
from mimcs.testing import (
    von_mises_fisher, uniform_sphere, unit_vector_array, evaluate, nuts)
from mimcs.testing.problems import _vmf_mean_resultant


def _chart(d, batch=(), seed=0, log_scale=0.0):
    """A deliberately non-trivial chart: a random pole and a non-unit scale."""
    v = jax.random.normal(jax.random.PRNGKey(seed), batch + (d,))
    v = v / jnp.linalg.norm(v, axis=-1, keepdims=True)
    return SphereChart(householder=v, log_scale=jnp.full(batch, log_scale))


# --- the chart itself -------------------------------------------------------- #

def test_dimensions_differ_between_sample_and_coordinate():
    """The point of the type: ``coord_dim`` is the *intrinsic* dimension, not the ambient one."""
    p = UnitVectorParameter("x", 4)
    assert p.ambient_shape == (4,) and p.coord_dim == 3
    q = UnitVectorParameter("q", 3, (5,))
    assert q.ambient_shape == (5, 3) and q.coord_dim == 10

    model = Model([EuclideanParameter("a", (2,)), UnitVectorParameter("x", 3)],
                  {"lp": lambda d: jnp.zeros(())})
    assert model.ambient_dim == 5 and model.coord_dim == 4


def test_unit_vector_requires_at_least_two_components():
    with pytest.raises(ValueError, match="d >= 2"):
        UnitVectorParameter("x", 1)


@pytest.mark.parametrize("d", [2, 3, 5])
def test_chart_round_trip_and_unit_norm(d):
    """``to_coordinate`` inverts ``from_coordinate``, and the ambient value is on the sphere."""
    p = UnitVectorParameter("x", d)
    hp = _chart(d, log_scale=0.4)
    u = jax.random.normal(jax.random.PRNGKey(1), (d - 1,))
    x = p.from_coordinate(u, hp)
    assert np.isclose(float(jnp.linalg.norm(x)), 1.0, atol=1e-5)
    assert np.allclose(np.asarray(p.to_coordinate(x, hp)), np.asarray(u), atol=1e-4)


@pytest.mark.parametrize("d, log_scale", [(2, 0.0), (3, 0.0), (3, 0.7), (4, -0.5)])
def test_chart_jacobian_matches_autodiff_gram_determinant(d, log_scale):
    """``log_jacobian_det`` must equal the Gram determinant of the (non-square) Jacobian.

    ``from_coordinate: R^(d-1) -> R^d`` is not a bijection of a single space, so the volume
    element of the chart is ``0.5 log det(J^T J)``, not ``log|det J|``.
    """
    p = UnitVectorParameter("x", d)
    hp = _chart(d, seed=2, log_scale=log_scale)
    u = jax.random.normal(jax.random.PRNGKey(3), (d - 1,))
    J = np.asarray(jax.jacobian(lambda uu: p.from_coordinate(uu, hp))(u))
    assert J.shape == (d, d - 1)
    gram = 0.5 * np.log(np.linalg.det(J.T @ J))
    assert np.isclose(gram, float(p.log_jacobian_det(u, hp)), atol=1e-4)


def test_chart_jacobian_matches_autodiff_batched():
    """Each unit vector in an array contributes its own term; the total is their sum."""
    n, d = 3, 3
    p = UnitVectorParameter("x", d, (n,))
    hp = _chart(d, (n,), seed=4)
    hp = SphereChart(householder=hp.householder, log_scale=jnp.array([0.3, -0.2, 0.6]))
    u = jax.random.normal(jax.random.PRNGKey(5), (p.coord_dim,))
    J = np.asarray(jax.jacobian(lambda uu: p.from_coordinate(uu, hp).ravel())(u))
    assert J.shape == (n * d, n * (d - 1))
    gram = 0.5 * np.log(np.linalg.det(J.T @ J))
    assert np.isclose(gram, float(p.log_jacobian_det(u, hp)), atol=1e-4)


def test_householder_is_an_involutive_isometry():
    """``H_v`` may reposition the pole but must not distort the sphere."""
    d = 4
    p = UnitVectorParameter("x", d)
    v = _chart(d, seed=6).householder
    y = jax.random.normal(jax.random.PRNGKey(7), (d,))
    assert np.isclose(float(jnp.linalg.norm(p._reflect(y, v))),
                      float(jnp.linalg.norm(y)), atol=1e-5)                 # isometry
    assert np.allclose(np.asarray(p._reflect(p._reflect(y, v), v)),
                       np.asarray(y), atol=1e-5)                            # involution


def test_pole_is_the_charts_singularity():
    """``u`` diverges at the pole and the pole is a unit vector; the antipode maps to 0."""
    d = 3
    p = UnitVectorParameter("x", d)
    hp = _chart(d, seed=8)
    pole = p.pole(hp)
    assert np.isclose(float(jnp.linalg.norm(pole)), 1.0, atol=1e-5)
    near = pole * 0.9999 + jnp.array([0.0, 0.0, 1e-4])
    near = near / jnp.linalg.norm(near)
    assert float(jnp.linalg.norm(p.to_coordinate(near, hp))) > 1e2
    # from_coordinate(0) is the antipode of the pole: the point farthest from the singularity.
    assert np.allclose(np.asarray(p.from_coordinate(jnp.zeros(d - 1), hp)),
                       -np.asarray(pole), atol=1e-5)


def test_default_chart_is_the_equatorial_north_pole_projection():
    p = UnitVectorParameter("x", 3)
    hp = p.init_hyperparams()
    assert np.allclose(np.asarray(p.pole(hp)), [0.0, 0.0, 1.0], atol=1e-6)
    assert np.isclose(float(p.plane_offset(hp)), 0.0, atol=1e-6)


@pytest.mark.parametrize("log_scale", [-0.6, 0.0, 0.5])
def test_inside_unit_circle_iff_beyond_the_cutting_plane(log_scale):
    """``|u| < 1  <=>  <x, pole> < c``.

    The identity the scale adaptation rests on: targeting a fraction of draws inside the unit
    circle *is* targeting a quantile of the pole alignment, so ``c`` is that quantile.
    """
    d = 3
    p = UnitVectorParameter("x", d)
    hp = _chart(d, seed=9, log_scale=log_scale)
    pole, c = np.asarray(p.pole(hp)), float(p.plane_offset(hp))

    rng = np.random.default_rng(0)
    xs = rng.standard_normal((2000, d))
    xs /= np.linalg.norm(xs, axis=1, keepdims=True)
    u = jax.vmap(lambda x: p.to_coordinate(x, hp))(jnp.asarray(xs))
    inside = np.asarray(jnp.linalg.norm(u, axis=1)) < 1.0
    beyond = (xs @ pole) < c
    assert np.array_equal(inside, beyond)


def test_default_sample_is_on_the_manifold():
    """A flat ambient zero vector is *not* valid here; the coordinate origin is."""
    model = Model([EuclideanParameter("a", (2,)), UnitVectorParameter("x", 3)],
                  {"lp": lambda d: jnp.zeros(())})
    s = model.default_sample()
    assert s.shape == (5,)
    assert np.allclose(np.asarray(s[:2]), 0.0)                 # Euclidean part unchanged
    assert np.isclose(float(jnp.linalg.norm(s[2:])), 1.0, atol=1e-6)


# --- the adaptation ---------------------------------------------------------- #

def test_adaptation_puts_the_pole_opposite_the_mass_and_hits_the_quantile():
    """On a vMF the pole should converge to ``-mu`` and the target fraction should be met."""
    mu = np.array([0.3, -0.5, 0.81])
    mu /= np.linalg.norm(mu)
    problem = von_mises_fisher(kappa=5.0, mu=mu)
    sampler = nuts(unit_vector_center=True)(problem.model, 0)
    sampler.warmup(3000)

    pole, c = sampler.unit_vector_chart("x")
    cos_angle = float(np.asarray(pole) @ (-mu))
    assert cos_angle > 0.99, f"pole {np.asarray(pole)} is not opposite mu (cos {cos_angle})"

    draws = sampler.sample(4000)["x"]
    p, hp = problem.model.parameters[0], sampler.state.chart_hyperparams[0]
    u = jax.vmap(lambda x: p.to_coordinate(x, hp))(jnp.asarray(draws))
    frac = float(np.mean(np.asarray(jnp.linalg.norm(u, axis=1)) < 1.0))
    assert abs(frac - 0.5) < 0.06, f"inside-fraction {frac} is not near the 0.5 target"
    # c is that same quantile of the pole alignment.
    assert abs(float(np.mean((draws @ np.asarray(pole)) < float(c))) - frac) < 1e-6


def _angle(a, b):
    """Geodesic distance between two unit vectors, via the chord.

    Not ``arccos(<a, b>)``: that is catastrophically ill-conditioned near 0 (its derivative
    blows up), so in float32 it cannot resolve a small angle better than ~4e-4 --- which is
    precisely the regime these tests probe, since the step shrinks with the gain.
    ``2 arcsin(|a - b| / 2)`` is stable there.
    """
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(2.0 * np.arcsin(np.clip(np.linalg.norm(a - b) / 2.0, 0.0, 1.0)))


@pytest.mark.parametrize("frac", [0.0, 0.25, 0.5, 1.0])
def test_geodesic_step_lands_at_the_right_arc_length(frac):
    """``Exp_p(frac Log_p(t))`` must sit at geodesic distance ``frac * d(p, t)`` along the arc."""
    rng = np.random.default_rng(0)
    p = rng.standard_normal(3); p /= np.linalg.norm(p)
    t = rng.standard_normal(3); t /= np.linalg.norm(t)
    theta = _angle(p, t)
    q = np.asarray(_geodesic_step(jnp.asarray(p), jnp.asarray(t), frac))
    assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-6)
    assert np.isclose(_angle(p, q), frac * theta, atol=1e-5)          # moved this far
    assert np.isclose(_angle(q, t), (1.0 - frac) * theta, atol=1e-5)  # ...along the arc, not off it


def test_geodesic_step_handles_coincident_and_antipodal_targets():
    """The geodesic direction is undefined at ``sin(theta) = 0``; both cases must stay put."""
    p = jnp.asarray([0.0, 0.0, 1.0])
    for target in (p, -p):
        q = np.asarray(_geodesic_step(p, target, 0.25))
        assert np.all(np.isfinite(q))
        assert np.allclose(q, np.asarray(p), atol=1e-6)


def test_geodesic_step_is_batched_per_unit_vector():
    rng = np.random.default_rng(1)
    P = rng.standard_normal((4, 3)); P /= np.linalg.norm(P, axis=1, keepdims=True)
    T = rng.standard_normal((4, 3)); T /= np.linalg.norm(T, axis=1, keepdims=True)
    Q = np.asarray(_geodesic_step(jnp.asarray(P), jnp.asarray(T), 0.3))
    for i in range(4):
        assert np.isclose(_angle(P[i], Q[i]), 0.3 * _angle(P[i], T[i]), atol=1e-5)


def test_pole_adaptation_diminishes_with_the_gain():
    """Validity rests on *diminishing adaptation*, not on the pole converging.

    The step is ``Exp_p(gain Log_p(-x))``, so the chart moves by exactly ``gain * theta`` and
    never more than ``gain * pi``. Since the Robbins--Monro gain tends to 0, the per-iteration
    change in the chart does too --- which is the condition adaptive MCMC actually needs, and it
    holds whether or not a Frechet mean exists to converge to.
    """
    rng = np.random.default_rng(2)
    p = rng.standard_normal(3); p /= np.linalg.norm(p)
    t = rng.standard_normal(3); t /= np.linalg.norm(t)
    theta = _angle(p, t)
    for n in (1, 100, 10_000):
        gain = rm_gain(n)
        q = _geodesic_step(jnp.asarray(p), jnp.asarray(t), gain)
        assert np.isclose(_angle(p, q), gain * theta, atol=1e-5)
        assert _angle(p, q) <= gain * np.pi + 1e-6
    assert rm_gain(10_000) < 1e-2                       # ...and the gain really does vanish


def test_frechet_pole_finds_the_axis_of_a_great_circle():
    """The case the extrinsic mean is blind to: mass on a great circle has ``E[x] = 0``.

    The Frechet mean of the antipodal draws is the circle's *axis* (``F = pi^2/4`` there against
    ``pi^2/3`` on the circle), which is exactly where a girdle is emptiest and so where the
    chart's singularity belongs. Both poles of the circle are minima, hence the ``abs``.
    """
    rng = np.random.default_rng(3)
    phi = 2.0 * np.pi * rng.random(4000)
    circle = np.column_stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)])   # axis = e_3
    assert np.linalg.norm(circle.mean(0)) < 0.05, "E[x] should be ~0: nothing for a mean to find"

    sampler = nuts(unit_vector_center=True)(uniform_sphere(3).model, 0)
    start = np.array([0.6, 0.5, 0.62]); start /= np.linalg.norm(start)
    sampler._uv_pole[0] = jnp.asarray(start)
    for n, x in enumerate(circle, start=1):
        sampler._uv_pole_step(0, rm_gain(n), jnp.asarray(x))

    pole = np.asarray(sampler._uv_pole[0])
    assert abs(abs(float(pole[2])) - 1.0) < 0.05, f"pole {pole} did not reach the circle's axis"


def test_adaptation_fits_each_array_element_separately():
    """``array[n] unit_vector[d]``: per-vector poles, so a shared chart could not fit both."""
    problem = unit_vector_array(mus=((0.0, 0.0, 1.0), (1.0, 0.0, 0.0)), kappas=(6.0, 6.0))
    sampler = nuts(unit_vector_center=True)(problem.model, 0)
    sampler.warmup(3000)
    poles, _ = sampler.unit_vector_chart("x")
    poles = np.asarray(poles)
    assert poles.shape == (2, 3)
    assert poles[0] @ np.array([0.0, 0.0, -1.0]) > 0.98
    assert poles[1] @ np.array([-1.0, 0.0, 0.0]) > 0.98


def test_chart_adaptation_relabels_without_moving_the_chain():
    """A chart update must hold the physical point fixed and keep the state self-consistent."""
    problem = von_mises_fisher(kappa=4.0)
    sampler = nuts(unit_vector_center=True)(problem.model, 0)
    sampler.warmup(200)
    state = sampler.state
    model = problem.model
    # the coordinate and the sample still describe the same point under the *live* chart
    rebuilt = model.coordinate_to_sample(
        state.coordinate, state.chart_hyperparams, state.chart_indices)
    assert np.allclose(np.asarray(rebuilt), np.asarray(state.sample), atol=1e-4)
    # and log_prob is the coordinate-space target there
    assert np.isclose(
        float(state.log_prob),
        float(model.log_prob_at_coordinate(
            state.coordinate, state.chart_hyperparams, state.chart_indices)),
        atol=1e-3)


def test_centering_does_not_disturb_an_adaptive_unit_vector_chart():
    """Regression: centering rebuilds the coordinate vector and must touch only its own slices.

    ``_CenteringBase`` computes the link value ``z`` under the *initial* hyperparameters. For a
    unit vector those are the initial pole, so writing ``z`` back wholesale would silently
    relabel --- and physically move --- the unit vector while its chart said otherwise.
    """
    mu = jnp.array([0.0, 0.0, 1.0])

    def log_post(params):
        return 4.0 * jnp.dot(params["x"], mu) - 0.5 * jnp.sum((params["a"] - 3.0) ** 2)

    model = Model([EuclideanParameter("a", (2,), centered=True),
                   UnitVectorParameter("x", 3)], {"lp": log_post})
    sampler = nuts(center=True, unit_vector_center=True)(model, 0)
    sampler.warmup(400)
    state = sampler.state
    # Both charts are live at once; the state must still be internally consistent.
    rebuilt = model.coordinate_to_sample(
        state.coordinate, state.chart_hyperparams, state.chart_indices)
    assert np.allclose(np.asarray(rebuilt), np.asarray(state.sample), atol=1e-4)
    assert np.isclose(float(jnp.linalg.norm(state.sample[2:])), 1.0, atol=1e-4)


# --- the DSL ----------------------------------------------------------------- #

def test_dsl_unit_vector_matches_the_hand_built_parameter():
    model = compile_model("parameters { unit_vector[3] x; }\nmodel { target += 2.0 * x[3]; }",
                          data={})
    p = model.parameters[0]
    assert isinstance(p, UnitVectorParameter)
    assert (p.name, p.d, p.batch_shape) == ("x", 3, ())
    assert p.ambient_shape == (3,) and p.coord_dim == 2


def test_dsl_array_of_unit_vectors():
    model = compile_model(
        "data { int n; }\nparameters { array[n] unit_vector[4] q; }\nmodel { target += 0.0; }",
        data={"n": 5})
    p = model.parameters[0]
    assert (p.d, p.batch_shape, p.ambient_shape, p.coord_dim) == (4, (5,), (5, 4), 15)


def test_dsl_unit_vector_density_matches_the_hand_built_model():
    """The compiled target must agree with the equivalent hand-built model in coordinates."""
    src = "parameters { unit_vector[3] x; }\nmodel { target += 5.0 * x[3]; }"
    compiled = compile_model(src, data={})
    mu = jnp.array([0.0, 0.0, 1.0])
    hand = Model([UnitVectorParameter("x", 3)],
                 {"lp": lambda p: 5.0 * jnp.dot(p["x"], mu)})
    u = jnp.array([0.4, -0.9])
    h, c = compiled.init_chart_hyperparams(), compiled.init_chart_indices()
    assert np.isclose(float(compiled.log_prob_at_coordinate(u, h, c)),
                      float(hand.log_prob_at_coordinate(u, h, c)), atol=1e-5)


@pytest.mark.parametrize("src, match", [
    ("parameters { unit_vector x; }\nmodel { target += 0; }", "needs a size"),
    ("parameters { unit_vector[1] x; }\nmodel { target += 0; }", "degenerate"),
    ("parameters { unit_vector[3,2] x; }\nmodel { target += 0; }", "one size"),
    ("parameters { unit_vector[3]<lower=0> x; }\nmodel { target += 0; }", "cannot carry"),
    ("data { unit_vector[3] d; }\nparameters { real x; }\nmodel { target += x; }",
     "only be declared in the `parameters` block"),
])
def test_dsl_unit_vector_errors(src, match):
    with pytest.raises(DslError, match=match):
        compile_model(src, data={"d": np.zeros(3)})


# --- end to end -------------------------------------------------------------- #

def test_von_mises_fisher_sampling_is_correct(artifacts_dir):
    """The chart's Jacobian is what makes the coordinate-space chain target the vMF."""
    problem = von_mises_fisher(kappa=5.0)
    report = evaluate(
        problem, {"nuts": nuts(unit_vector_center=True)},
        n_warmup=2000, n_samples=20000, seed=0,
        out_dir=artifacts_dir / "von_mises_fisher")
    report.assert_correct()
    # the analytic mean resultant, recovered in sample space
    draws = report.outputs["nuts"].samples
    assert np.isclose(float(np.linalg.norm(draws.mean(0))), _vmf_mean_resultant(5.0), atol=0.02)


def test_uniform_sphere_sampling_is_correct(artifacts_dir):
    """A constant density: the Jacobian alone has to produce a uniform sphere."""
    problem = uniform_sphere(3)
    report = evaluate(
        problem, {"nuts": nuts(unit_vector_center=True)},
        n_warmup=2000, n_samples=20000, seed=0,
        out_dir=artifacts_dir / "uniform_sphere")
    report.assert_correct()


def test_unit_vector_array_sampling_is_correct(artifacts_dir):
    problem = unit_vector_array()
    report = evaluate(
        problem, {"nuts": nuts(unit_vector_center=True)},
        n_warmup=2000, n_samples=20000, seed=0,
        out_dir=artifacts_dir / "unit_vector_array")
    report.assert_correct()
