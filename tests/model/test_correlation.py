"""Tests for the correlation types: ``CorrMatrixParameter`` and ``CholeskyFactorCorrParameter``.

The correlation counterparts of ``test_cholesky_cov.py``, and the same three risks, plus one new
one:

* the **chart** --- Stan's ``tanh`` link and stick breaking on sums of squares --- whose two
  log-Jacobians differ between the types and are checked against autodiff;
* the **projection** onto the elliptope's tangent space, checked to be zero-diagonal and metric
  orthogonal to every zero-diagonal direction;
* the **Stein terms**, which unlike the covariance ones have no closed form and are assembled in
  divergence form. Two things stand in for the missing closed form: the same helper applied to the
  SPD geometry must reproduce :class:`CovMatrixParameter`'s closed forms, and the terms must average
  to zero over exactly sampled LKJ draws --- with a control that fires when the score is wrong;
* the **score conversion** for the factor type, where the diagonal is *determined* rather than free.

Exact LKJ draws come from the canonical partial correlations, which are what the chart is written
in: for LKJ(eta) they are independent with ``z_ij ~ 2 Beta(b, b) - 1``, ``b = eta + (K - 2 - j)/2``
on 0-based column ``j``. That law was derived from ``log det Omega = sum log(1 - z^2)`` and verified
to 1e-15 before these tests were written; :func:`test_the_lkj_sampling_law` keeps it honest.

Seeds are fixed, so pass/fail is deterministic.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest
from scipy.stats import beta as beta_dist

from mimcs.model import (Model, CorrMatrixParameter, CholeskyFactorCorrParameter,
                        CovMatrixParameter)
from mimcs.model.correlation import _project_tangent, _strict_tril
from mimcs.dsl import DslError
from mimcs import compile_model, make_sampler

K, ETA = 3, 2.0


def _beta_shape(K, eta):
    """The LKJ law's Beta parameter for each coordinate, by 0-based column."""
    _, cols = np.tril_indices(K, -1)
    return np.array([eta + (K - 2 - j) / 2.0 for j in cols])


def lkj_coordinates(n, K=K, eta=ETA, seed=0):
    """Exact LKJ(eta) draws, as the chart's own coordinates ``y``."""
    b = _beta_shape(K, eta)
    rng = np.random.default_rng(seed)
    # size= is required: scipy squeezes a leading axis of 1, which silently makes n=1 draws
    # come back with the wrong rank.
    z = 2 * beta_dist.rvs(b, b, size=(n, len(b)), random_state=rng) - 1
    return jnp.asarray(np.arctanh(z), float)


def lkj_omega(n, K=K, eta=ETA, seed=0):
    p = CorrMatrixParameter("O", K)
    return jax.vmap(p.from_coordinate)(lkj_coordinates(n, K, eta, seed))


def _log_p_omega(Om, eta=ETA):
    return (eta - 1.0) * jnp.linalg.slogdet(Om)[1]


def _log_p_l_free(u, K=K, eta=ETA):
    """The LKJ density of the *factor*, as a function of its free (strict lower) entries.

    The diagonal is determined by the rows, which is the whole subtlety: differentiating with the
    diagonal held independent gives a different --- and wrong --- score.
    """
    rows, cols = np.tril_indices(K, -1)
    M = jnp.zeros((K, K), float).at[rows, cols].set(u)
    M = M + jnp.diag(jnp.sqrt(jnp.clip(1.0 - jnp.sum(M ** 2, axis=1), 1e-30)))
    i = jnp.arange(K, dtype=float)
    return jnp.sum(((K - 1 - i + 2.0 * eta - 2.0) * jnp.log(jnp.diag(M)))[1:])


def _chol_score(L, K=K, eta=ETA):
    """The score a chart reading only the free entries hands to ``stein_terms``."""
    rows, cols = np.tril_indices(K, -1)
    g = jax.grad(lambda u: _log_p_l_free(u, K, eta))(L[rows, cols])
    return jnp.zeros((K, K), float).at[rows, cols].set(g)


def _coords(p, rng, scale=0.7):
    return jnp.asarray(rng.normal(size=(p.coord_dim,)) * scale, float)


BOTH = [CorrMatrixParameter, CholeskyFactorCorrParameter]


# --- the chart --------------------------------------------------------------- #

def test_dimensions_and_shapes():
    p = CorrMatrixParameter("O", 4)
    assert p.ambient_shape == (4, 4) and p.coord_dim == 6
    q = CholeskyFactorCorrParameter("L", 3, (5,))
    assert q.ambient_shape == (5, 3, 3) and q.coord_dim == 15


@pytest.mark.parametrize("cls", BOTH)
def test_a_correlation_matrix_needs_at_least_two_dimensions(cls):
    with pytest.raises(ValueError, match="at least 2 dimensions"):
        cls("O", 1)


@pytest.mark.parametrize("cls", BOTH)
@pytest.mark.parametrize("k,batch", [(2, ()), (3, ()), (5, ()), (3, (2,))])
def test_chart_round_trip(cls, k, batch):
    p = cls("O", k, batch)
    y = _coords(p, np.random.default_rng(3))
    value = p.from_coordinate(y)
    assert value.shape == p.ambient_shape
    assert np.allclose(np.asarray(p.to_coordinate(value)), np.asarray(y), atol=1e-4)


@pytest.mark.parametrize("k", [2, 3, 5])
def test_the_constraints_hold_exactly(k):
    rng = np.random.default_rng(4)
    Om = np.asarray(CorrMatrixParameter("O", k).from_coordinate(
        _coords(CorrMatrixParameter("O", k), rng)))
    assert np.allclose(np.diag(Om), 1.0, atol=1e-6), "unit diagonal"
    assert np.allclose(Om, Om.T, atol=1e-6)
    assert np.min(np.linalg.eigvalsh(Om)) > 0

    L = np.asarray(CholeskyFactorCorrParameter("L", k).from_coordinate(
        _coords(CholeskyFactorCorrParameter("L", k), rng)))
    assert np.allclose(np.triu(L, 1), 0.0), "lower triangular"
    assert np.allclose((L ** 2).sum(axis=1), 1.0, atol=1e-6), "unit row norms"
    assert np.all(np.diag(L) > 0)


@pytest.mark.parametrize("cls", BOTH)
def test_coordinate_origin_is_the_identity(cls):
    p = cls("O", 4)
    at_zero = np.asarray(p.from_coordinate(jnp.zeros((p.coord_dim,), float)))
    assert np.allclose(at_zero, np.eye(4), atol=1e-6)


@pytest.mark.parametrize("cls", BOTH)
@pytest.mark.parametrize("k", [2, 3, 4])
def test_log_jacobian_matches_autodiff(cls, k):
    """The free entries are the *strict* lower triangle: the diagonal is determined, not free."""
    p = cls("O", k)
    y = _coords(p, np.random.default_rng(5))
    rows, cols = np.tril_indices(k, -1)
    jac = np.asarray(jax.jacobian(lambda c: p.from_coordinate(c)[rows, cols])(y))
    want = float(np.linalg.slogdet(jac)[1])
    assert np.isclose(float(p.log_jacobian_det(y)), want, atol=1e-3), (
        f"{cls.__name__}(K={k}): {float(p.log_jacobian_det(y))} != {want}")


def test_a_near_singular_matrix_stays_finite():
    """Strong correlations are far away in coordinates, not a numerical cliff."""
    p = CorrMatrixParameter("O", 3)
    y = jnp.asarray(np.full(p.coord_dim, 6.0), float)
    Om = np.asarray(p.from_coordinate(y))
    assert np.all(np.isfinite(Om)) and np.isfinite(float(p.log_jacobian_det(y)))
    assert np.min(np.linalg.eigvalsh(Om)) >= 0.0


@pytest.mark.parametrize("cls", BOTH)
def test_model_gradient_matches_finite_difference(cls):
    p = cls("O", 3)
    fn = (_log_p_omega if cls is CorrMatrixParameter
          else (lambda L: _log_p_l_free(L[np.tril_indices(3, -1)])))
    model = Model([p], {"target": lambda v: fn(v["O"])})
    y = _coords(p, np.random.default_rng(7), scale=0.4)
    charts = (model.init_chart_hyperparams(), model.init_chart_indices())
    f = lambda c: float(model.log_prob_at_coordinate(c, *charts))
    grad = np.asarray(jax.grad(model.log_prob_at_coordinate)(y, *charts))
    eps = 1e-3
    for j in range(p.coord_dim):
        step = np.zeros(p.coord_dim); step[j] = eps
        fd = (f(y + step) - f(y - step)) / (2 * eps)
        assert np.isclose(grad[j], fd, rtol=3e-2, atol=3e-2), f"coordinate {j}"


# --- the geometry ------------------------------------------------------------ #

@pytest.mark.parametrize("k", [2, 3, 5])
def test_the_tangent_projection_is_metric_orthogonal(k):
    """``P(V)`` has zero diagonal, and ``V - P(V)`` is orthogonal to every zero-diagonal direction
    in the induced metric ``tr(Om^-1 A Om^-1 B)`` --- which is what makes it *the* projection."""
    rng = np.random.default_rng(8)
    Om = np.asarray(lkj_omega(1, k, 2.0, seed=k)[0], dtype=float)
    V = rng.standard_normal((k, k)); V = (V + V.T) / 2
    P = np.asarray(_project_tangent(jnp.asarray(Om), jnp.asarray(V)))
    assert np.abs(np.diag(P)).max() < 1e-6, "the projection must land in the tangent space"

    inv = np.linalg.inv(Om)
    residual = V - P
    for _ in range(50):
        B = rng.standard_normal((k, k)); B = (B + B.T) / 2
        np.fill_diagonal(B, 0.0)
        ip = np.trace(inv @ residual @ inv @ B) / (np.linalg.norm(B) + 1e-12)
        assert abs(ip) < 1e-4, "the residual must be orthogonal to the tangent space"


def test_the_divergence_form_reproduces_the_covariance_closed_forms():
    """The oracle that stands in for a closed form on the elliptope.

    The correlation Stein terms are assembled as ``div_x(grad phi) + grad phi . s`` rather than as
    ``Delta phi + <grad phi, grad log pi>``, because the elliptope's Laplacian is not available in
    closed form. The two are the same operator, and here that identity is checked where a closed
    form *does* exist: on the SPD matrices, where :class:`CovMatrixParameter` uses
    ``L Sigma_ab = (K+1) Sigma_ab + (Sigma S Sigma)_ab``.
    """
    d = 3
    rows, cols = np.tril_indices(d)
    rng = np.random.default_rng(9)
    A = rng.standard_normal((d, d)); Sig = jnp.asarray(A @ A.T + d * np.eye(d), float)
    vinv = jnp.asarray(np.linalg.inv(np.asarray(Sig) + np.eye(d)), float)

    def unvech(x):
        M = jnp.zeros((d, d), float).at[rows, cols].set(x)
        return M + M.T - jnp.diag(jnp.diag(M))

    logp = lambda x: 2.0 * jnp.linalg.slogdet(unvech(x))[1] - 0.5 * jnp.trace(vinv @ unvech(x))
    x0 = Sig[rows, cols]
    s = jax.grad(logp)(x0)
    G = jax.grad(lambda M: 2.0 * jnp.linalg.slogdet(M)[1] - 0.5 * jnp.trace(vinv @ M))(Sig)
    S = (G + G.T) / 2

    for (a, b) in [(0, 0), (2, 1), (1, 1)]:
        C = jnp.zeros((d, d), float).at[a, b].add(0.5).at[b, a].add(0.5)
        field = lambda x: (unvech(x) @ C @ unvech(x))[rows, cols]
        divergence = jnp.trace(jax.jacfwd(field)(x0))
        got = float(divergence + jnp.dot(field(x0), s))
        want = float((d + 1) * Sig[a, b] + (Sig @ S @ Sig)[a, b])
        assert np.isclose(got, want, rtol=1e-4), f"({a},{b}): {got} != {want}"


def test_the_lkj_sampling_law():
    """The exact-draw machinery the Stein test rests on, checked against ``det(Omega)^(eta-1)``.

    If this law were wrong the Stein test would be comparing against the wrong target and could
    fail --- or pass --- for reasons that have nothing to do with the operator.
    """
    for k, eta in [(3, 2.0), (4, 1.5)]:
        p = CorrMatrixParameter("O", k)
        b = _beta_shape(k, eta)
        rng = np.random.default_rng(0)
        constants = []
        for _ in range(5):
            z = 2 * beta_dist.rvs(b, b, random_state=rng) - 1
            y = jnp.asarray(np.arctanh(z), float)
            log_pz = float(np.sum(beta_dist.logpdf((z + 1) / 2, b, b) - np.log(2.0)))
            log_jac_z = float(p.log_jacobian_det(y)) - float(np.sum(np.log1p(-z ** 2)))
            Om = np.asarray(p.from_coordinate(y), dtype=float)
            constants.append(log_pz - log_jac_z - (eta - 1) * np.log(np.linalg.det(Om)))
        spread = float(np.max(constants) - np.min(constants))
        assert spread < 1e-3, f"K={k}: the vine law is not LKJ({eta}); spread {spread}"


# --- the Stein identity ------------------------------------------------------ #

@pytest.mark.parametrize("cls", BOTH)
def test_stein_terms_are_mean_zero_under_the_target(cls):
    """Against exact LKJ draws --- and a control, so it cannot pass vacuously.

    The elliptope has no boundary in this metric (the singular matrices are infinitely far), so
    unlike the simplex there is no vanishing-flux assumption to arrange for.
    """
    n = 60_000
    Om = lkj_omega(n, K, ETA, seed=11)
    p = cls("O", K)
    if cls is CorrMatrixParameter:
        value, score = Om, jax.vmap(jax.grad(_log_p_omega))(Om)
    else:
        value = jnp.linalg.cholesky(Om)
        score = jax.vmap(_chol_score)(value)

    t = np.asarray(jax.vmap(p.stein_terms)(value, score))
    z = t.mean(0) / (t.std(0, ddof=1) / np.sqrt(n))
    assert np.all(np.abs(z) < 4.0), f"{cls.__name__}: Stein z {np.round(z, 2)} not zero"

    t_bad = np.asarray(jax.vmap(p.stein_terms)(value, score * 1.3))
    z_bad = t_bad.mean(0) / (t_bad.std(0, ddof=1) / np.sqrt(n))
    assert np.any(np.abs(z_bad) > 8.0), f"the control did not fire: z {np.round(z_bad, 2)}"


def test_the_two_types_agree_on_the_stein_terms():
    """Same target, same draws, two ambient values: the operator lives on ``Omega`` either way."""
    Om = lkj_omega(200, K, ETA, seed=12)
    L = jnp.linalg.cholesky(Om)
    a = np.asarray(jax.vmap(CorrMatrixParameter("O", K).stein_terms)(
        Om, jax.vmap(jax.grad(_log_p_omega))(Om)))
    b = np.asarray(jax.vmap(CholeskyFactorCorrParameter("L", K).stein_terms)(
        L, jax.vmap(_chol_score)(L)))
    assert np.allclose(a, b, rtol=1e-3, atol=1e-3)


def test_corr_matrix_stein_terms_use_only_the_symmetric_off_diagonal_score():
    """The diagonal is structurally 1 and the array is symmetric: neither direction is real."""
    p = CorrMatrixParameter("O", K)
    rng = np.random.default_rng(13)
    Om = lkj_omega(1, K, ETA, seed=3)[0]
    g = jnp.asarray(rng.normal(size=(K, K)), float)
    skew = jnp.asarray(rng.normal(size=(K, K)), float)
    skew = skew - skew.T
    diagonal = jnp.diag(jnp.asarray(rng.normal(size=(K,)), float))
    base = np.asarray(p.stein_terms(Om, g))
    assert np.allclose(base, np.asarray(p.stein_terms(Om, g + skew)), atol=1e-3)
    assert np.allclose(base, np.asarray(p.stein_terms(Om, g + diagonal)), atol=1e-3)


def test_cholesky_stein_terms_ignore_the_structural_upper_triangle():
    p = CholeskyFactorCorrParameter("L", K)
    rng = np.random.default_rng(14)
    L = jnp.linalg.cholesky(lkj_omega(1, K, ETA, seed=4)[0])
    g = _chol_score(L)
    upper = jnp.triu(jnp.asarray(rng.normal(size=(K, K)), float), 1)
    assert np.allclose(np.asarray(p.stein_terms(L, g)),
                       np.asarray(p.stein_terms(L, g + upper)), atol=1e-3)


# --- features ---------------------------------------------------------------- #

def test_features_are_the_strict_lower_triangle_and_its_squares():
    """The diagonal is constantly 1 and is excluded: a constant feature is a frozen coordinate."""
    Om = lkj_omega(1, K, ETA, seed=15)[0]
    rows, cols = np.tril_indices(K, -1)
    want = np.asarray(Om)[rows, cols]
    for p, value in [(CorrMatrixParameter("O", K), Om),
                     (CholeskyFactorCorrParameter("L", K), jnp.linalg.cholesky(Om))]:
        got = np.asarray(p.features(value))
        assert got.shape == (p.n_features,) == (K * (K - 1),)
        assert np.allclose(got[:len(want)], want, atol=1e-5)
        assert np.allclose(got[len(want):], want ** 2, atol=1e-5)
        assert not np.any(np.isclose(got, 1.0, atol=1e-9)), "no constant feature"


def test_feature_names_say_which_matrix_they_are_from():
    assert CorrMatrixParameter("O", 3).feature_names()[:3] == ["O[2,1]", "O[3,1]", "O[3,2]"]
    assert CholeskyFactorCorrParameter("L", 3).feature_names()[0] == "corr(L)[2,1]"


def test_features_are_full_rank():
    p = CorrMatrixParameter("O", 3)
    f = np.asarray(jax.vmap(p.features)(lkj_omega(400, 3, 2.0, seed=16)))
    design = np.column_stack([np.ones(len(f)), f])
    assert np.linalg.matrix_rank(design, tol=1e-6) == design.shape[1]


def test_batched_features_and_names():
    p = CorrMatrixParameter("O", 3, (2,))
    assert p.n_features == 2 * 2 * 3
    names = p.feature_names()
    assert len(names) == p.n_features
    assert names[0] == "O[1][2,1]" and names[6] == "O[2][2,1]"


# --- the DSL ----------------------------------------------------------------- #

def test_dsl_builds_the_same_parameters():
    m = compile_model("parameters { corr_matrix[3] O; }\nmodel { target += 0.0; }", {})
    assert isinstance(m.parameters[0], CorrMatrixParameter)
    assert m.ambient_dim == 9 and m.coord_dim == 3

    m = compile_model("parameters { cholesky_factor_corr[4] L; }\nmodel { target += 0.0; }", {})
    assert isinstance(m.parameters[0], CholeskyFactorCorrParameter)
    assert m.ambient_dim == 16 and m.coord_dim == 6


def test_dsl_array_of_correlation_matrices():
    m = compile_model("parameters { array[4] corr_matrix[3] O; }\nmodel { target += 0.0; }", {})
    assert m.parameters[0].ambient_shape == (4, 3, 3)
    assert m.coord_dim == 12


def test_lkj_corr_matches_a_hand_written_density():
    dsl = compile_model("parameters { corr_matrix[3] O; }\nmodel { O ~ lkj_corr(2.0); }", {})
    hand = Model([CorrMatrixParameter("O", 3)], {"target": lambda v: _log_p_omega(v["O"], 2.0)})
    y = jnp.asarray(np.random.default_rng(17).normal(size=(3,)) * 0.6, float)
    charts = (dsl.init_chart_hyperparams(), dsl.init_chart_indices())
    assert np.isclose(float(dsl.log_prob_at_coordinate(y, *charts)),
                      float(hand.log_prob_at_coordinate(y, *charts)), atol=1e-4)


def test_the_two_lkj_forms_are_the_same_target():
    """`lkj_corr` on `Omega` and `lkj_corr_cholesky` on `L` must give the same coordinate density.

    The Cholesky form carries the ``L -> Omega`` Jacobian, and the two charts' own Jacobians differ
    by exactly that, so the two cancel. Getting the Jacobian wrong -- the easy mistake, and the
    reason the distribution is worth naming -- breaks this.
    """
    a = compile_model("parameters { corr_matrix[4] O; }\nmodel { O ~ lkj_corr(2.5); }", {})
    b = compile_model("parameters { cholesky_factor_corr[4] L; }\n"
                      "model { L ~ lkj_corr_cholesky(2.5); }", {})
    rng = np.random.default_rng(18)
    diffs = []
    for _ in range(5):
        y = jnp.asarray(rng.normal(size=(6,)) * 0.8, float)
        diffs.append(float(a.log_prob_at_coordinate(y, a.init_chart_hyperparams(),
                                                    a.init_chart_indices()))
                     - float(b.log_prob_at_coordinate(y, b.init_chart_hyperparams(),
                                                      b.init_chart_indices())))
    assert np.ptp(np.asarray(diffs)) < 1e-3, f"not the same target: {diffs}"


@pytest.mark.parametrize("src,match", [
    ("parameters { corr_matrix[1] O; }\nmodel { target += 0.0; }", "at least 2 dimensions"),
    ("parameters { corr_matrix<lower=0>[2] O; }\nmodel { target += 0.0; }", "bound"),
    ("data { corr_matrix[2] O; }\nmodel { target += 0.0; }", "parameters"),
])
def test_dsl_errors(src, match):
    with pytest.raises(DslError, match=match):
        compile_model(src, {})


# --- end to end -------------------------------------------------------------- #

def test_sampling_an_lkj_target_recovers_it():
    """NUTS on LKJ against exact draws, compared on the free entries.

    The comparison is by hand rather than through ``evaluate``: a correlation matrix's ambient
    array has ``K`` structurally constant entries (the unit diagonal), and the harness's variance
    and correlation checks divide by their zero standard deviation. That is a harness limitation,
    not a property of the draws -- it fails comparing exact LKJ draws to *themselves*.
    """
    eta, k = 2.0, 3
    model = compile_model(f"parameters {{ corr_matrix[{k}] O; }}\n"
                          f"model {{ O ~ lkj_corr({eta}); }}", {})
    sampler = make_sampler(model, seed=0)
    sampler.initialize()
    sampler.warmup(1000)
    draws = np.asarray(sampler.sample(8000)["O"])

    assert np.allclose(np.diagonal(draws, axis1=1, axis2=2), 1.0, atol=1e-5)
    assert np.min(np.linalg.eigvalsh(draws)) > 0

    rows, cols = np.tril_indices(k, -1)
    got = draws[:, rows, cols]
    exact = np.asarray(lkj_omega(50_000, k, eta, seed=99))[:, rows, cols]
    se = got.std(0) / np.sqrt(200)                       # a conservative ESS floor
    assert np.all(np.abs(got.mean(0) - exact.mean(0)) < 4 * se), (
        f"means {got.mean(0)} vs {exact.mean(0)}")
    assert np.all(np.abs(got.std(0) / exact.std(0) - 1.0) < 0.1), (
        f"sds {got.std(0)} vs {exact.std(0)}")
