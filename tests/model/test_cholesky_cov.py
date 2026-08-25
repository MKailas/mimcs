"""Tests for the covariance types: ``CovMatrixParameter`` and ``CholeskyFactorCovParameter``.

Both are the log-Cholesky chart over ``K(K+1)/2`` coordinates; they differ only in whether the
ambient value is ``Sigma`` or its factor ``L``. Three things carry real risk and get real tests:

* the **Jacobians**, which differ between the two types and are checked against autodiff;
* the **score conversion** for the factor type, which recovers ``Sigma S Sigma`` from a score taken
  with respect to ``L`` by a short piece of triangular algebra --- checked against a direct autodiff
  of the density written in ``Sigma``, the slow obviously-correct reference;
* the **Stein terms**, from the Brownian motion on the SPD symmetric space, checked against exactly
  sampled Wishart draws (Bartlett gives the Cholesky factor directly, so no MCMC is involved) with
  a control that fires when the score is wrong, so the check cannot pass vacuously.

Seeds are fixed, so pass/fail is deterministic.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from mimcs.model import Model, CovMatrixParameter, CholeskyFactorCovParameter
from mimcs.model.cholesky_cov import _sigma_score
from mimcs.dsl import DslError
from mimcs import compile_model, make_sampler
from mimcs.testing import evaluate, nuts, wishart_cov

K, NU = 3, 8.0


def _scale_matrix(K=K, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((K, K))
    return A @ A.T + K * np.eye(K)


V = _scale_matrix()
VINV = np.linalg.inv(V)


def _log_p_sigma(S, K=K, nu=NU, vinv=None):
    """Wishart(nu, V) log-density in ``Sigma``, up to a constant."""
    vinv = jnp.asarray(VINV if vinv is None else vinv)
    return (nu - K - 1) / 2 * jnp.linalg.slogdet(S)[1] - 0.5 * jnp.trace(vinv @ S)


def _log_p_l(L, K=K, nu=NU):
    """The same target as a density on the factor: ``+ log|dSigma/dL|``."""
    i = jnp.arange(1, K + 1)
    return (_log_p_sigma(L @ L.T, K, nu) + K * jnp.log(2.0)
            + jnp.sum((K - i + 1) * jnp.log(jnp.diag(L))))


def bartlett(n, rng, K=K, nu=NU, scale=None):
    """Exact Wishart draws, returned as their Cholesky factors (Bartlett's decomposition)."""
    chol_v = np.linalg.cholesky(V if scale is None else scale)
    A = np.zeros((n, K, K))
    for i in range(K):
        A[:, i, i] = np.sqrt(rng.chisquare(nu - i, size=n))
        for j in range(i):
            A[:, i, j] = rng.standard_normal(n)
    return chol_v @ A


def _coords(p, rng, scale=0.6):
    return jnp.asarray(rng.normal(size=(p.coord_dim,)) * scale, float)


# --- the chart --------------------------------------------------------------- #

def test_dimensions_and_shapes():
    p = CovMatrixParameter("S", 4)
    assert p.ambient_shape == (4, 4) and p.coord_dim == 10
    q = CholeskyFactorCovParameter("L", 3, (5,))
    assert q.ambient_shape == (5, 3, 3) and q.coord_dim == 30


@pytest.mark.parametrize("cls", [CovMatrixParameter, CholeskyFactorCovParameter])
def test_a_covariance_needs_at_least_one_dimension(cls):
    with pytest.raises(ValueError, match="at least one dimension"):
        cls("S", 0)


@pytest.mark.parametrize("cls", [CovMatrixParameter, CholeskyFactorCovParameter])
@pytest.mark.parametrize("k,batch", [(1, ()), (2, ()), (4, ()), (3, (2,))])
def test_chart_round_trip(cls, k, batch):
    p = cls("S", k, batch)
    rng = np.random.default_rng(3)
    z = _coords(p, rng)
    value = p.from_coordinate(z)
    assert value.shape == p.ambient_shape
    assert np.allclose(np.asarray(p.to_coordinate(value)), np.asarray(z), atol=1e-5)


@pytest.mark.parametrize("k", [1, 2, 4])
def test_the_value_is_positive_definite_and_the_factor_is_lower_triangular(k):
    rng = np.random.default_rng(4)
    cov = CovMatrixParameter("S", k)
    sigma = np.asarray(cov.from_coordinate(_coords(cov, rng)))
    assert np.allclose(sigma, sigma.T, atol=1e-6)
    assert np.min(np.linalg.eigvalsh(sigma)) > 0

    chol = CholeskyFactorCovParameter("L", k)
    L = np.asarray(chol.from_coordinate(_coords(chol, rng)))
    assert np.allclose(np.triu(L, 1), 0.0)
    assert np.all(np.diag(L) > 0)


@pytest.mark.parametrize("cls", [CovMatrixParameter, CholeskyFactorCovParameter])
def test_coordinate_origin_is_the_identity(cls):
    p = cls("S", 4)
    at_zero = np.asarray(p.from_coordinate(jnp.zeros((p.coord_dim,), float)))
    assert np.allclose(at_zero, np.eye(4), atol=1e-6)


@pytest.mark.parametrize("cls", [CovMatrixParameter, CholeskyFactorCovParameter])
@pytest.mark.parametrize("k", [1, 2, 3])
def test_log_jacobian_matches_autodiff(cls, k):
    """The two types' Jacobians differ; both are checked against the real thing.

    The map is into the ``(K, K)`` ambient array, whose entries are not independent, so the
    reference determinant is taken over the free entries --- the lower triangle.
    """
    p = cls("S", k)
    rng = np.random.default_rng(5)
    z = _coords(p, rng)
    rows, cols = np.tril_indices(k)

    def free(coord):
        return p.from_coordinate(coord)[rows, cols]

    jac = np.asarray(jax.jacobian(free)(z))
    want = float(np.linalg.slogdet(jac)[1])
    assert np.isclose(float(p.log_jacobian_det(z)), want, atol=1e-4), (
        f"{cls.__name__}(K={k}): {float(p.log_jacobian_det(z))} != {want}")


def test_stan_jacobian_formula_for_cov_matrix():
    """The composed Jacobian is Stan's ``K log 2 + sum_i (K - i + 2) log L_ii``."""
    p = CovMatrixParameter("S", 3)
    rng = np.random.default_rng(6)
    z = _coords(p, rng)
    L = np.linalg.cholesky(np.asarray(p.from_coordinate(z)))
    i = np.arange(1, 4)
    want = 3 * np.log(2.0) + np.sum((3 - i + 2) * np.log(np.diag(L)))
    assert np.isclose(float(p.log_jacobian_det(z)), want, atol=1e-5)


def test_a_tiny_diagonal_stays_finite():
    """A near-singular factor is far away in coordinates, not a numerical cliff."""
    p = CholeskyFactorCovParameter("L", 3)
    z = jnp.asarray(np.full(p.coord_dim, -12.0), float)
    L = np.asarray(p.from_coordinate(z))
    assert np.all(np.isfinite(L)) and np.all(np.diag(L) > 0)
    assert np.isfinite(float(p.log_jacobian_det(z)))


@pytest.mark.parametrize("cls", [CovMatrixParameter, CholeskyFactorCovParameter])
def test_model_gradient_matches_finite_difference(cls):
    p = cls("S", 3)
    fn = _log_p_sigma if cls is CovMatrixParameter else _log_p_l
    model = Model([p], {"target": lambda v: fn(v["S"])})
    rng = np.random.default_rng(7)
    z = _coords(p, rng)
    charts = (model.init_chart_hyperparams(), model.init_chart_indices())
    f = lambda c: float(model.log_prob_at_coordinate(c, *charts))
    grad = np.asarray(jax.grad(model.log_prob_at_coordinate)(z, *charts))
    eps = 1e-3
    for j in range(p.coord_dim):
        step = np.zeros(p.coord_dim)
        step[j] = eps
        fd = (f(z + step) - f(z - step)) / (2 * eps)
        assert np.isclose(grad[j], fd, rtol=2e-2, atol=2e-2), f"coordinate {j}"


# --- the score conversion ---------------------------------------------------- #

@pytest.mark.parametrize("k", [1, 2, 3, 5])
def test_sigma_score_inversion_matches_autodiff(k):
    """``Sigma S Sigma`` recovered from the score wrt ``L``, against the density's own gradient.

    The reference is deliberately dumb: differentiate the log-density written in ``Sigma``,
    symmetrize, and form the product. The implementation instead starts from the score wrt ``L``
    and undoes both the chain rule and the ``L -> Sigma`` Jacobian by triangular algebra.
    """
    rng = np.random.default_rng(8)
    scale = _scale_matrix(k, seed=k)
    vinv = np.linalg.inv(scale)
    nu = k + 5.0
    L = jnp.asarray(bartlett(1, rng, k, nu, scale)[0], float)

    logp_l = lambda M: (_log_p_sigma(M @ M.T, k, nu, vinv) + k * jnp.log(2.0)
                        + jnp.sum((k - jnp.arange(1, k + 1) + 1) * jnp.log(jnp.diag(M))))
    got = np.asarray(_sigma_score(L, jnp.tril(jax.grad(logp_l)(L)), k))

    sigma = np.asarray(L @ L.T)
    G = np.asarray(jax.grad(lambda S: _log_p_sigma(S, k, nu, vinv))(jnp.asarray(sigma)))
    want = sigma @ ((G + G.T) / 2) @ sigma
    rel = np.max(np.abs(got - want)) / np.max(np.abs(want))
    assert rel < 1e-4, f"K={k}: relative error {rel:.2e}"          # float32 floor


def test_cov_matrix_stein_terms_use_only_the_symmetric_score():
    """The ambient array holds a symmetric matrix, so an antisymmetric cotangent moves nothing."""
    p = CovMatrixParameter("S", 3)
    rng = np.random.default_rng(9)
    sigma = jnp.asarray(_scale_matrix(3, seed=2), float)
    g = jnp.asarray(rng.normal(size=(3, 3)), float)
    skew = jnp.asarray(rng.normal(size=(3, 3)), float)
    skew = skew - skew.T
    assert np.allclose(np.asarray(p.stein_terms(sigma, g)),
                       np.asarray(p.stein_terms(sigma, g + skew)), atol=1e-4)


def test_cholesky_stein_terms_ignore_the_structural_upper_triangle():
    """Those entries are held at zero; whatever lands on them is not a direction of the manifold."""
    p = CholeskyFactorCovParameter("L", 3)
    rng = np.random.default_rng(10)
    L = jnp.asarray(bartlett(1, rng)[0], float)
    g = jnp.asarray(rng.normal(size=(3, 3)), float)
    upper = jnp.triu(jnp.asarray(rng.normal(size=(3, 3)), float), 1)
    assert np.allclose(np.asarray(p.stein_terms(L, g)),
                       np.asarray(p.stein_terms(L, g + upper)), atol=1e-4)


# --- the Stein identity ------------------------------------------------------ #

@pytest.mark.parametrize("cls", [CovMatrixParameter, CholeskyFactorCovParameter])
def test_stein_terms_are_mean_zero_under_the_target(cls):
    """Against exact Wishart draws --- and a control, so it cannot pass vacuously.

    ``P(K)`` has no boundary, so unlike the simplex there is no vanishing-flux assumption to
    arrange for: any proper Wishart will do.
    """
    n = 200_000
    rng = np.random.default_rng(11)
    L = bartlett(n, rng)
    p = cls("S", K)

    if cls is CovMatrixParameter:
        value = jnp.asarray(L @ np.transpose(L, (0, 2, 1)), float)
        score = jax.vmap(jax.grad(lambda S: _log_p_sigma(S)))(value)
    else:
        value = jnp.asarray(L, float)
        score = jax.vmap(jax.grad(_log_p_l))(value)

    t = np.asarray(jax.vmap(p.stein_terms)(value, score))
    z = t.mean(0) / (t.std(0, ddof=1) / np.sqrt(n))
    assert np.all(np.abs(z) < 4.0), f"{cls.__name__}: Stein z {np.round(z, 2)} not zero"

    t_bad = np.asarray(jax.vmap(p.stein_terms)(value, score * 1.3))
    z_bad = t_bad.mean(0) / (t_bad.std(0, ddof=1) / np.sqrt(n))
    assert np.any(np.abs(z_bad) > 10.0), f"the control did not fire: z {np.round(z_bad, 2)}"


def test_the_two_types_agree_on_the_stein_terms():
    """Same target, same draws, two ambient values: the operator lives on ``Sigma`` either way."""
    rng = np.random.default_rng(12)
    L = jnp.asarray(bartlett(64, rng), float)
    sigma = jnp.asarray(np.asarray(L) @ np.transpose(np.asarray(L), (0, 2, 1)), float)

    chol = CholeskyFactorCovParameter("L", K)
    cov = CovMatrixParameter("S", K)
    a = np.asarray(jax.vmap(chol.stein_terms)(L, jax.vmap(jax.grad(_log_p_l))(L)))
    b = np.asarray(jax.vmap(cov.stein_terms)(
        sigma, jax.vmap(jax.grad(lambda S: _log_p_sigma(S)))(sigma)))
    assert np.allclose(a, b, rtol=1e-3, atol=1e-3)


# --- features ---------------------------------------------------------------- #

def test_features_are_the_covariance_lower_triangle_and_its_squares():
    rng = np.random.default_rng(13)
    L = np.asarray(bartlett(1, rng)[0])
    rows, cols = np.tril_indices(K)
    want = (L @ L.T)[rows, cols]

    for p, value in [(CholeskyFactorCovParameter("L", K), L),
                     (CovMatrixParameter("S", K), L @ L.T)]:
        got = np.asarray(p.features(jnp.asarray(value, float)))
        assert got.shape == (p.n_features,) == (2 * K * (K + 1) // 2,)
        assert np.allclose(got[:len(want)], want, rtol=1e-5)
        assert np.allclose(got[len(want):], want ** 2, rtol=1e-5)


def test_feature_names_say_which_matrix_they_are_from():
    assert CovMatrixParameter("S", 2).feature_names() == [
        "S[0,0]", "S[1,0]", "S[1,1]", "S[0,0]^2", "S[1,0]^2", "S[1,1]^2"]
    # For the factor type the features are of the *covariance*, and the label says so.
    assert CholeskyFactorCovParameter("L", 2).feature_names()[0] == "cov(L)[0,0]"


def test_features_are_full_rank():
    """No component is dropped here --- unlike the simplex and the unit vector, nothing is tied."""
    rng = np.random.default_rng(14)
    p = CovMatrixParameter("S", 3)
    L = bartlett(400, rng)
    sigma = jnp.asarray(L @ np.transpose(L, (0, 2, 1)), float)
    f = np.asarray(jax.vmap(p.features)(sigma))
    design = np.column_stack([np.ones(len(f)), f])
    assert np.linalg.matrix_rank(design, tol=1e-8) == design.shape[1]


def test_batched_features_and_names():
    p = CovMatrixParameter("S", 2, (3,))
    assert p.n_features == 3 * 2 * 3
    names = p.feature_names()
    assert len(names) == p.n_features
    assert names[0] == "S[0][0,0]" and names[6] == "S[1][0,0]"
    assert p.ambient_names()[:2] == ["S[0,0,0]", "S[0,0,1]"]      # batch index, then the entry


# --- the DSL ----------------------------------------------------------------- #

def test_dsl_builds_the_same_parameters():
    m = compile_model("parameters { cov_matrix[3] S; }\nmodel { target += 0.0; }", {})
    assert isinstance(m.parameters[0], CovMatrixParameter)
    assert m.ambient_dim == 9 and m.coord_dim == 6

    m = compile_model("parameters { cholesky_factor_cov[3] L; }\nmodel { target += 0.0; }", {})
    assert isinstance(m.parameters[0], CholeskyFactorCovParameter)
    assert m.ambient_dim == 9 and m.coord_dim == 6


def test_dsl_array_of_covariances():
    m = compile_model(
        "parameters { array[4] cov_matrix[2] S; }\nmodel { target += 0.0; }", {})
    assert m.parameters[0].ambient_shape == (4, 2, 2)
    assert m.coord_dim == 12


def test_dsl_density_matches_the_hand_built_model():
    """`L * L'` in the DSL is the same target as the hand-built one."""
    # log det(L L') = 2 sum_i log L_ii and tr(A S) = sum(A .* S) for symmetric S, so the whole
    # density is written through the factor -- which is why the factor is worth having as a type.
    src = """
    data { array[3, 3] real vinv; }
    parameters { cholesky_factor_cov[3] L; }
    model {
      target += (8.0 - 3.0 - 1.0) * sum(log(diag(L))) - 0.5 * sum(vinv .* (L * L'));
      target += 3 * log(2.0) + 3 * log(L[1,1]) + 2 * log(L[2,2]) + 1 * log(L[3,3]);
    }
    """
    model = compile_model(src, {"vinv": VINV})
    hand = Model([CholeskyFactorCovParameter("L", 3)], {"target": lambda v: _log_p_l(v["L"])})
    rng = np.random.default_rng(15)
    z = jnp.asarray(rng.normal(size=(6,)) * 0.5, float)
    charts = (model.init_chart_hyperparams(), model.init_chart_indices())
    assert np.isclose(float(model.log_prob_at_coordinate(z, *charts)),
                      float(hand.log_prob_at_coordinate(z, *charts)), atol=1e-4)


@pytest.mark.parametrize("src,match", [
    ("parameters { cov_matrix[0] S; }\nmodel { target += 0.0; }", "at least one dimension"),
    ("parameters { cov_matrix<lower=0>[2] S; }\nmodel { target += 0.0; }", "bound"),
    ("data { cov_matrix[2] S; }\nmodel { target += 0.0; }", "parameters"),
])
def test_dsl_errors(src, match):
    with pytest.raises(DslError, match=match):
        compile_model(src, {})


@pytest.mark.parametrize("kind", ["cov_matrix", "cholesky_factor_cov"])
def test_a_covariance_chart_takes_no_options(kind):
    from mimcs.dsl import compile_model as compile_
    factory = compile_(f"parameters {{ {kind}[2] S; }}\nmodel {{ target += 0.0; }}")
    spec = factory.analyze({})
    with pytest.raises(DslError, match="centered"):
        spec.parameter("S").centered = True
        spec.build()


# --- end to end -------------------------------------------------------------- #

def test_wishart_sampling_is_correct(artifacts_dir):
    """The composed chart Jacobian is what makes the coordinate chain target the Wishart."""
    problem = wishart_cov()
    report = evaluate(problem, {"nuts": nuts()}, n_warmup=2000, n_samples=20000, seed=0,
                      out_dir=artifacts_dir / "wishart_cov")
    print("\n" + report.summary())
    report.assert_correct()

    draws = report.outputs["nuts"].samples.reshape(-1, 3, 3)
    assert np.allclose(draws, np.transpose(draws, (0, 2, 1)), atol=1e-4), "not symmetric"
    assert np.min(np.linalg.eigvalsh(draws)) > 0, "not positive definite"


def test_cholesky_factor_sampling_recovers_the_wishart_mean():
    """The factor type on the same target, checked in the covariance it stands for.

    ``E[Sigma] = nu V``. The comparison is in ``Sigma`` rather than ``L`` because that is what the
    parametrization means -- and it is what the features and the Stein terms are taken from.
    """
    nu = 8.0
    model = Model([CholeskyFactorCovParameter("L", 3)],
                  {"log_post": lambda v: _log_p_l(v["L"], 3, nu)})
    sampler = make_sampler(model, seed=0)
    sampler.initialize()
    sampler.warmup(2000)
    draws = np.asarray(sampler.sample(20000)["L"])

    assert np.allclose(np.triu(draws, 1), 0.0), "the factor must stay lower triangular"
    sigma = draws @ np.transpose(draws, (0, 2, 1))
    want = nu * V
    err = np.abs(sigma.mean(0) - want) / want.diagonal().mean()
    assert np.max(err) < 0.05, f"E[Sigma] off by {np.max(err):.3f} of the scale:\n{sigma.mean(0)}"
