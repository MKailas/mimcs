"""Tests for ``SimplexParameter``: ``d`` positive components summing to one.

The second parameter whose coordinate space has a *lower dimension* than its ambient space, and
the second whose constraint makes the generic ``[x, x^2]`` features rank-deficient --- but for
the opposite reason to the unit vector. There the constraint is ``sum_k x_k^2 = 1``, so the
*squares* are collinear; here it is ``sum_k x_k = 1``, so the *linear* terms are.

The checks that matter: the chart is a bijection onto the open simplex, its log-Jacobian agrees
with autodiff, and the Langevin--Stein terms really are mean-zero under the target --- verified
against exactly-sampled Dirichlet draws, with a control that fires when the score is wrong, so
the check cannot pass vacuously.

Seeds are fixed, so pass/fail is deterministic.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from mimcs.model import Model, SimplexParameter
from mimcs.dsl import DslError
from mimcs import compile_model
from mimcs.testing import dirichlet_simplex, evaluate, nuts


ALPHA = (2.0, 3.0, 4.0)


def _coords(p, rng, scale=1.5):
    return jnp.asarray(rng.normal(size=(p.coord_dim,)) * scale, float)


# --- the chart --------------------------------------------------------------- #

def test_dimensions_and_shapes():
    p = SimplexParameter("x", 4)
    assert p.ambient_shape == (4,) and p.coord_dim == 3
    q = SimplexParameter("x", 3, (5,))
    assert q.ambient_shape == (5, 3) and q.coord_dim == 10


def test_simplex_requires_at_least_two_components():
    with pytest.raises(ValueError, match="d >= 2"):
        SimplexParameter("x", 1)


@pytest.mark.parametrize("d", [2, 3, 5])
@pytest.mark.parametrize("batch", [(), (3,), (2, 2)])
def test_chart_round_trip_and_simplex_membership(d, batch):
    """``to_coordinate`` inverts ``from_coordinate``, and the value is on the open simplex."""
    p = SimplexParameter("x", d, batch)
    rng = np.random.default_rng(0)
    y = _coords(p, rng)
    x = p.from_coordinate(y)

    assert x.shape == p.ambient_shape
    assert np.all(np.asarray(x) > 0.0)
    assert np.allclose(np.asarray(jnp.sum(x, axis=-1)), 1.0, atol=1e-6)
    assert np.allclose(np.asarray(p.to_coordinate(x)), np.asarray(y), atol=1e-4)


@pytest.mark.parametrize("d", [2, 3, 6])
def test_coordinate_origin_is_the_uniform_point(d):
    """The offsets exist for this: ``Model.default_sample`` evaluates every chart at ``0``."""
    p = SimplexParameter("x", d)
    x0 = np.asarray(p.from_coordinate(jnp.zeros((p.coord_dim,), float)))
    assert np.allclose(x0, 1.0 / d, atol=1e-6)


@pytest.mark.parametrize("d", [2, 3, 5])
def test_log_jacobian_matches_autodiff(d):
    """``log|J|`` of the map onto the ``d-1`` free components, against a dense Jacobian."""
    p = SimplexParameter("x", d)
    rng = np.random.default_rng(1)
    y = _coords(p, rng)

    # The chart's determinant is that of y -> (x_1..x_{d-1}); x_d is then determined.
    free = jax.jacfwd(lambda c: p.from_coordinate(c)[:-1])(y)
    assert np.allclose(float(p.log_jacobian_det(y)),
                       float(jnp.log(jnp.abs(jnp.linalg.det(free)))), atol=1e-3)


def test_a_tiny_component_stays_finite():
    """The chart is computed in logs, so a component far out in the tail must not blow up."""
    p = SimplexParameter("x", 3)
    x = p.from_coordinate(jnp.asarray([-30.0, 0.0], float))
    assert np.all(np.isfinite(np.asarray(x)))
    assert float(x[0]) > 0.0 and float(x[0]) < 1e-10
    assert np.isfinite(float(p.log_jacobian_det(jnp.asarray([-30.0, 0.0], float))))


def test_model_gradient_matches_finite_difference():
    """Autodiff through ``log_prob_at_coordinate`` --- what HMC differentiates."""
    model = dirichlet_simplex(ALPHA).model
    h, c = model.init_chart_hyperparams(), model.init_chart_indices()

    def f(q):
        return model.log_prob_at_coordinate(q, h, c)

    q0 = np.array([0.4, -0.6])
    grad = np.asarray(jax.grad(f)(jnp.asarray(q0, float)))
    eps = 1e-3
    fd = np.array([
        (float(f(jnp.asarray(q0 + eps * e, float)))
         - float(f(jnp.asarray(q0 - eps * e, float)))) / (2 * eps)
        for e in np.eye(2)])
    assert np.all(np.isfinite(grad))
    assert np.allclose(grad, fd, atol=1e-3), f"grad {grad} != fd {fd}"


# --- features and Stein terms ------------------------------------------------ #

def test_features_drop_the_last_component():
    """Two per degree of freedom, and full rank --- see :meth:`SimplexParameter.features`."""
    p = SimplexParameter("x", 4, (2,))
    assert p.n_features == 2 * 2 * 3
    assert p.feature_names()[:3] == ["x[1][1]", "x[1][2]", "x[1][3]"]
    assert p.feature_names()[3:6] == ["x[1][1]^2", "x[1][2]^2", "x[1][3]^2"]
    assert p.ambient_names()[:4] == ["x[1][1]", "x[1][2]", "x[1][3]", "x[1][4]"]

    x = p.from_coordinate(jnp.asarray(np.random.default_rng(2).normal(size=(p.coord_dim,)), float))
    f = np.asarray(p.features(x))
    assert f.shape == (p.n_features,)
    assert np.allclose(f[:3], np.asarray(x)[0, :3], atol=1e-6)


@pytest.mark.parametrize("d", [2, 3])
def test_features_are_full_rank_on_the_simplex(d):
    """The reason a component is dropped: a collinear column breaks the classifier's fit.

    At ``d = 2`` this is what rules out keeping ``x_d^2`` as well --- with no cross terms,
    ``x_2^2 = 1 - 2 x_1 + x_1^2`` lies exactly in the span of the intercept and ``x_1, x_1^2``.
    """
    p = SimplexParameter("x", d)
    rng = np.random.default_rng(3)
    draws = jnp.asarray(rng.dirichlet(np.full(d, 2.0), size=200), float)
    design = np.column_stack(
        [np.ones(200), np.asarray(jax.vmap(p.features)(draws))])
    assert np.linalg.matrix_rank(design, tol=1e-8) == design.shape[1]


def test_stein_terms_are_mean_zero_under_the_target():
    """The target-aware check, against exact draws --- and a control, so it cannot pass vacuously.

    Every ``alpha_k > 1``, so the density vanishes on the faces and the boundary term of the
    integration by parts really does drop (the assumption
    :meth:`SimplexParameter.stein_terms` records).
    """
    alpha = np.asarray(ALPHA)
    n = 200_000
    rng = np.random.default_rng(4)
    x = rng.dirichlet(alpha, size=n)
    p = SimplexParameter("x", alpha.shape[0])

    score = (alpha - 1.0) / x                                   # the Dirichlet's ambient score
    t = np.asarray(jax.vmap(p.stein_terms)(jnp.asarray(x, float), jnp.asarray(score, float)))
    z = t.mean(0) / (t.std(0, ddof=1) / np.sqrt(n))
    assert np.all(np.abs(z) < 4.0), f"Stein z-scores {z} are not consistent with zero"

    # Control: a mis-scaled score is a different target, and must be caught.
    t_bad = np.asarray(jax.vmap(p.stein_terms)(
        jnp.asarray(x, float), jnp.asarray(score * 1.3, float)))
    z_bad = t_bad.mean(0) / (t_bad.std(0, ddof=1) / np.sqrt(n))
    assert np.any(np.abs(z_bad) > 10.0), f"the control did not fire: z {z_bad}"


def test_stein_terms_ignore_the_normal_component_of_the_score():
    """The ambient score is defined only up to the constraint normal ``1``; ``P`` discards it."""
    p = SimplexParameter("x", 4)
    rng = np.random.default_rng(5)
    x = jnp.asarray(rng.dirichlet(np.full(4, 2.0)), float)
    g = jnp.asarray(rng.normal(size=(4,)), float)
    shifted = g + 3.7                                           # move along the normal
    assert np.allclose(np.asarray(p.stein_terms(x, g)),
                       np.asarray(p.stein_terms(x, shifted)), atol=1e-5)


# --- the DSL ----------------------------------------------------------------- #

def test_dsl_simplex_matches_the_hand_built_parameter():
    model = compile_model("parameters { simplex[3] w; }\nmodel { target += 0.0; }", {})
    p = model.parameters[0]
    assert isinstance(p, SimplexParameter)
    assert p.d == 3 and p.ambient_shape == (3,) and p.coord_dim == 2


def test_dsl_array_of_simplices():
    model = compile_model(
        "data { int n; }\nparameters { array[n] simplex[3] w; }\nmodel { target += 0.0; }",
        {"n": 4})
    p = model.parameters[0]
    assert p.ambient_shape == (4, 3) and p.coord_dim == 8


def test_dsl_simplex_density_matches_the_hand_built_model():
    """The compiled program and the hand-built model must agree on the coordinate target."""
    src = ("parameters { simplex[3] w; }\n"
           "model { target += log(w[1]) + 2.0 * log(w[2]) + 3.0 * log(w[3]); }")
    model = compile_model(src, {})
    hand = Model([SimplexParameter("w", 3)],
                 {"m": lambda p: jnp.sum(jnp.asarray([1.0, 2.0, 3.0]) * jnp.log(p["w"]))})
    q = jnp.asarray([0.3, -0.7], float)
    h, c = model.init_chart_hyperparams(), model.init_chart_indices()
    assert np.isclose(float(model.log_prob_at_coordinate(q, h, c)),
                      float(hand.log_prob_at_coordinate(q, h, c)), atol=1e-4)


@pytest.mark.parametrize("src, match", [
    ("parameters { simplex x; }\nmodel { target += 0; }", "needs a size"),
    ("parameters { simplex[1] x; }\nmodel { target += 0; }", "degenerate"),
    ("parameters { simplex[3,2] x; }\nmodel { target += 0; }", "one size"),
    ("parameters { simplex<lower=0>[3] x; }\nmodel { target += 0; }", "cannot carry"),
    ("data { simplex[3] d; }\nparameters { real x; }\nmodel { target += x; }",
     "`parameters` block"),
])
def test_dsl_simplex_errors(src, match):
    with pytest.raises(DslError, match=match):
        compile_model(src, {})


def test_a_simplex_chart_takes_no_options():
    """It has no adaptive hyperparameters, and saying otherwise should say so plainly."""
    factory = compile_model("parameters { simplex[3] w; } model { target += 0; }")
    spec = factory.analyze({})
    spec.parameter("w").centered = True
    with pytest.raises(DslError, match="no chart options"):
        spec.build()


# --- end to end -------------------------------------------------------------- #

def test_dirichlet_sampling_is_correct(artifacts_dir):
    """The stick-breaking Jacobian is what makes the coordinate chain target the Dirichlet."""
    problem = dirichlet_simplex(ALPHA)
    report = evaluate(problem, {"nuts": nuts()}, n_warmup=2000, n_samples=20000, seed=0,
                      out_dir=artifacts_dir / "dirichlet_simplex")
    print("\n" + report.summary())
    report.assert_correct()

    draws = report.outputs["nuts"].samples
    assert np.allclose(draws.sum(axis=1), 1.0, atol=1e-4)
    assert np.all(draws > 0.0)
