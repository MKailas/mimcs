"""Gaussian-process covariance builtins: Stan's six kernels and ``add_diag`` (``mimcs/dsl/gp.py``).

Values are checked against a slow per-pair reference written straight from the Stan Functions
Reference formulas, for every overload Stan has (one input / cross-covariance, scalar inputs /
points, scalar / ARD length scale). The gradient tests carry a control --- the same kernel with an
unguarded ``sqrt`` --- because the defect they guard against is a ``NaN`` on the diagonal, and a test
that never sees a diagonal would pass vacuously.
"""

import math

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from mimcs import compile_model
from mimcs.dsl.builtins import BUILTINS
from mimcs.testing import TargetProblem, evaluate, nuts

KERNELS = ["gp_exp_quad_cov", "gp_exponential_cov", "gp_matern32_cov", "gp_matern52_cov",
           "gp_periodic_cov", "gp_dot_prod_cov"]
ARD_KERNELS = KERNELS[:4]
SIGMA, PERIOD = 1.3, 2.1


def _hypers(name, length_scale=0.8):
    if name == "gp_dot_prod_cov":
        return (SIGMA,)
    if name == "gp_periodic_cov":
        return (SIGMA, length_scale, PERIOD)
    return (SIGMA, length_scale)


def _ref_entry(name, a, b, sigma, l=1.0, p=None):
    """One covariance entry, from the Stan formula, in plain Python floats."""
    a, b = np.atleast_1d(a).astype(float), np.atleast_1d(b).astype(float)
    if name == "gp_dot_prod_cov":
        return sigma**2 + float(a @ b)
    if name == "gp_periodic_cov":
        r = math.sqrt(((a - b) ** 2).sum())
        return sigma**2 * math.exp(-2.0 * math.sin(math.pi * r / p) ** 2 / l**2)
    r = math.sqrt((((a - b) / np.broadcast_to(l, a.shape)) ** 2).sum())   # ARD: scale, then norm
    return sigma**2 * {
        "gp_exp_quad_cov": math.exp(-0.5 * r * r),
        "gp_exponential_cov": math.exp(-r),
        "gp_matern32_cov": (1 + math.sqrt(3) * r) * math.exp(-math.sqrt(3) * r),
        "gp_matern52_cov": (1 + math.sqrt(5) * r + 5 * r * r / 3) * math.exp(-math.sqrt(5) * r),
    }[name]


def _ref(name, x1, x2, hypers):
    return np.array([[_ref_entry(name, a, b, *hypers) for b in x2] for a in x1])


def _inputs(dim, seed=0):
    """``dim=None`` gives Stan's ``array[] real`` (1-d); an int gives ``dim``-dimensional points."""
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(n,) if dim is None else (n, dim)) for n in (5, 4)]


def _cases():
    for name in KERNELS:
        for dim in (None, 1) if name == "gp_periodic_cov" else (None, 1, 3):   # periodic: D=1 only
            for cross in (False, True):
                yield name, dim, cross, False
                if name in ARD_KERNELS and dim == 3:
                    yield name, dim, cross, True


def _decl(arr):
    return "array[" + ", ".join(str(n) for n in arr.shape) + "] real"


@pytest.mark.parametrize("name,dim,cross,ard", list(_cases()))
def test_kernel_matches_the_stan_formula_through_the_dsl(name, dim, cross, ard):
    """Every overload, reached through the DSL's own call path (arity picks the overload)."""
    x1, x2 = _inputs(dim)
    ls = np.array([0.6, 1.1, 1.9]) if ard else 0.8
    hypers = _hypers(name, ls)
    other = x2 if cross else x1
    W = np.random.default_rng(1).normal(size=(len(x1), len(other)))
    data = {"x1": x1, "x2": x2, "W": W, "ls": ls}
    hyper_args = ", ".join(["1.3", "ls", "2.1"][:len(hypers)] if name != "gp_dot_prod_cov"
                           else ["1.3"])
    inputs = "x1, x2" if cross else "x1"
    src = (f"data {{ {_decl(x1)} x1; {_decl(x2)} x2; {_decl(W)} W; "
           f"{'array[3] real' if ard else 'real'} ls; }}\n"
           "parameters { real d; }\n"
           f"model {{ target += sum(W .* {name}({inputs}, {hyper_args})) - 0.5 * d * d; }}")
    lp = compile_model(src, data).log_prob_fns["target"]
    got = float(lp({"d": jnp.asarray(0.0)}))
    want = float(np.sum(W * _ref(name, x1, other, hypers)))
    assert np.isclose(got, want, rtol=1e-5, atol=1e-5), (got, want)


@pytest.mark.parametrize("name", KERNELS)
@pytest.mark.parametrize("dim", [None, 2])
def test_kernel_matrix_shape_and_values(name, dim):
    if name == "gp_periodic_cov" and dim == 2:
        dim = 1                                        # periodic refuses D > 1 (tested below)
    x1, x2 = _inputs(dim)
    fn, h = BUILTINS[name], _hypers(name)
    K, C = np.asarray(fn(x1, *h)), np.asarray(fn(x1, x2, *h))
    assert K.shape == (5, 5) and C.shape == (5, 4)
    assert np.allclose(K, _ref(name, x1, x1, h), rtol=1e-5, atol=1e-6)
    assert np.allclose(C, _ref(name, x1, x2, h), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("name", KERNELS[:5])
def test_stationary_kernels_are_exactly_symmetric_psd_with_diagonal_sigma_squared(name):
    """Differences, not the expanded square: exact symmetry and a diagonal of exactly sigma^2."""
    dim = (12,) if name == "gp_periodic_cov" else (12, 2)     # periodic: PSD in 1-d only (below)
    x = np.random.default_rng(2).normal(size=dim)
    K = np.asarray(BUILTINS[name](x, *_hypers(name)))
    assert np.array_equal(K, K.T)
    assert np.array_equal(np.diag(K), np.full(12, np.float32(SIGMA) ** 2, K.dtype))
    assert np.linalg.eigvalsh(K.astype(np.float64)).min() > -1e-5


def _stan_periodic(x, l=0.8, p=PERIOD):
    """Stan's periodic formula on Euclidean distances, any ``D`` --- what the builtin refuses."""
    r = np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(-1))
    return np.exp(-2.0 * np.sin(np.pi * r / p) ** 2 / l**2)


def test_periodic_kernel_on_multidimensional_points_is_not_a_valid_covariance():
    """Why the builtin refuses ``D > 1``: Stan's formula there is not positive definite (the
    control), while the product of one 1-d kernel per dimension --- the remedy the error names ---
    is. The indefiniteness is checked on the formula itself, since the builtin will not build it."""
    x = np.random.default_rng(2).normal(size=(12, 2)) * 2
    assert np.linalg.eigvalsh(_stan_periodic(x)).min() < -0.1
    prod = np.asarray(BUILTINS["gp_periodic_cov"](x[:, 0], 1.0, 0.8, PERIOD)
                      * BUILTINS["gp_periodic_cov"](x[:, 1], 1.0, 0.8, PERIOD), np.float64)
    assert np.linalg.eigvalsh(prod).min() > -1e-5


@pytest.mark.parametrize("shapes", [[(6, 2)], [(6, 3)], [(6, 2), (4, 2)]])
def test_periodic_kernel_refuses_multidimensional_points(shapes):
    xs = [np.ones(s) for s in shapes]
    with pytest.raises(ValueError, match="gp_periodic_cov: inputs must be scalar.*1-d periodic"):
        BUILTINS["gp_periodic_cov"](*xs, 1.0, 0.8, PERIOD)


def test_periodic_kernel_refusal_reaches_the_dsl_user():
    src = ("data { array[6, 2] real X; }\nparameters { real d; }\n"
           "model { target += sum(gp_periodic_cov(X, 1.0, 0.8, 2.1)) - 0.5 * d * d; }")
    with pytest.raises(ValueError, match="inputs must be scalar"):
        compile_model(src, {"X": np.ones((6, 2))}).log_prob_fns["target"]({"d": jnp.asarray(0.)})


def test_periodic_kernel_on_one_column_points_matches_scalar_inputs():
    """``(n, 1)`` points are scalar inputs, and stay accepted."""
    x = np.random.default_rng(3).normal(size=6)
    assert np.allclose(BUILTINS["gp_periodic_cov"](x[:, None], 1.0, 0.8, PERIOD),
                       BUILTINS["gp_periodic_cov"](x, 1.0, 0.8, PERIOD))


def test_dot_product_kernel_is_bias_plus_gram():
    x = np.random.default_rng(3).normal(size=(6, 3))
    K = np.asarray(BUILTINS["gp_dot_prod_cov"](x, 0.5))
    assert np.allclose(K, 0.25 + x @ x.T, rtol=1e-5, atol=1e-6)


def test_ard_with_equal_length_scales_is_the_isotropic_kernel():
    x = np.random.default_rng(4).normal(size=(6, 3))
    for name in ARD_KERNELS:
        iso = BUILTINS[name](x, SIGMA, 0.7)
        ard = BUILTINS[name](x, SIGMA, jnp.full(3, 0.7))
        assert np.allclose(iso, ard, rtol=1e-6), name


# --- gradients ---------------------------------------------------------------- #

def _naive_matern32(x, sigma, l):
    """Control: the Matern 3/2 kernel with an unguarded ``sqrt`` --- NaN gradient on the diagonal."""
    s = jnp.sum(((x[:, None, :] - x[None, :, :]) / l) ** 2, axis=-1)
    r = jnp.sqrt(3.0) * jnp.sqrt(s)
    return sigma**2 * (1.0 + r) * jnp.exp(-r)


def test_unguarded_sqrt_control_does_give_nan_gradients():
    """Pins the hazard the guard exists for, so the test below cannot pass vacuously."""
    x = jnp.asarray(np.random.default_rng(5).normal(size=(6, 2)), float)
    g = jax.grad(lambda l: jnp.sum(_naive_matern32(x, 1.0, l)))(0.9)
    assert not np.isfinite(g)


@pytest.mark.parametrize("name", KERNELS)
def test_gradients_are_finite_on_the_diagonal_and_match_finite_differences(name):
    """Through inputs *and* hyperparameters --- latent GP inputs are a real use."""
    rng = np.random.default_rng(6)
    D = 1 if name == "gp_periodic_cov" else 2               # periodic refuses D > 1
    x0 = jnp.asarray(rng.normal(size=(6, D)), float)
    W = jnp.asarray(rng.normal(size=(6, 6)), float)
    h0 = jnp.asarray(_hypers(name), float)

    def f(v):
        return jnp.sum(W * BUILTINS[name](v[:6 * D].reshape(6, D), *v[6 * D:]))

    v = jnp.concatenate([x0.ravel(), h0])
    g = np.asarray(jax.grad(f)(v))
    assert np.all(np.isfinite(g))
    eps = 1e-2                                   # float32 suite: coarse central differences
    fd = np.array([(f(v.at[i].add(eps)) - f(v.at[i].add(-eps))) / (2 * eps)
                   for i in range(v.size)])
    assert np.allclose(g, fd, rtol=2e-2, atol=2e-2), np.abs(g - fd).max()


# --- errors --------------------------------------------------------------------- #

@pytest.mark.parametrize("call,exc,match", [
    (lambda x: BUILTINS["gp_exp_quad_cov"](x, 1.0), TypeError, r"takes 3 or 4 arguments"),
    (lambda x: BUILTINS["gp_periodic_cov"](x, 1.0, 1.0), TypeError, r"takes 4 or 5 arguments"),
    (lambda x: BUILTINS["gp_dot_prod_cov"](x), TypeError, r"takes 2 or 3 arguments"),
    (lambda x: BUILTINS["gp_matern32_cov"](x, 1.0, jnp.ones(2)), ValueError, "one entry per input"),
    (lambda x: BUILTINS["gp_periodic_cov"](x, 1.0, jnp.ones(3), 1.0), ValueError, "scalar"),
    (lambda x: BUILTINS["gp_exp_quad_cov"](x, jnp.ones(2), 1.0), ValueError, "sigma must be"),
    (lambda x: BUILTINS["gp_exp_quad_cov"](x, x[:, :2], 1.0, 1.0), ValueError, "dimension"),
    (lambda x: BUILTINS["gp_exp_quad_cov"](x[None], 1.0, 1.0), ValueError, "one point per row"),
    (lambda x: BUILTINS["add_diag"](x[0], 1.0), ValueError, "must be a matrix"),
    (lambda x: BUILTINS["add_diag"](x, jnp.ones(5)), ValueError, "3 entries"),
])
def test_misuse_is_reported_with_the_function_name(call, exc, match):
    with pytest.raises(exc, match=match):
        call(jnp.ones((4, 3)))


# --- add_diag and a GP model ---------------------------------------------------- #

def test_add_diag_with_a_scalar_and_with_a_vector():
    m = np.arange(9.0).reshape(3, 3)
    assert np.allclose(BUILTINS["add_diag"](m, 0.5), m + 0.5 * np.eye(3))
    assert np.allclose(BUILTINS["add_diag"](m, jnp.array([1.0, 2.0, 3.0])),
                       m + np.diag([1.0, 2.0, 3.0]))


GP_REGRESSION = """
data { int n; array[n] real x; array[n] real y; }
parameters { real<lower=0> alpha; real<lower=0> rho; real<lower=0> sigma; }
model {
  alpha ~ normal(0, 1); rho ~ normal(0, 1); sigma ~ normal(0, 1);
  y ~ multi_normal(0.0, add_diag(gp_exp_quad_cov(x, alpha, rho), square(sigma) + 1e-6));
}
"""


def test_gp_regression_density_and_gradient():
    """The marginal-likelihood GP --- Stan's canonical example --- with a zero scalar mean."""
    from jax.scipy.stats import multivariate_normal as mvn, norm
    rng = np.random.default_rng(7)
    x = np.linspace(0.0, 5.0, 10)
    y = np.sin(x) + 0.3 * rng.normal(size=10)
    model = compile_model(GP_REGRESSION, {"n": 10, "x": x, "y": y})
    lp = model.log_prob_fns["target"]
    a, r, s = 1.2, 0.9, 0.4
    got = float(lp({"alpha": jnp.asarray(a), "rho": jnp.asarray(r), "sigma": jnp.asarray(s)}))
    K = _ref("gp_exp_quad_cov", x, x, (a, r)) + (s**2 + 1e-6) * np.eye(10)
    want = (float(mvn.logpdf(jnp.asarray(y), jnp.zeros(10), jnp.asarray(K)))
            + float(jnp.sum(norm.logpdf(jnp.asarray([a, r, s])))))
    assert np.isclose(got, want, rtol=1e-5)
    charts = (model.init_chart_hyperparams(), model.init_chart_indices())
    z = jnp.asarray(rng.normal(size=(model.coord_dim,)) * 0.3, float)
    g = np.asarray(jax.grad(model.log_prob_at_coordinate)(z, *charts))
    assert np.all(np.isfinite(g)) and np.all(np.abs(g) > 1e-6)


@pytest.mark.parametrize("prior", [
    "multi_normal(0.0, K)",
    "multi_normal_cholesky(0.0, cholesky(K))",
    "multi_normal_prec(0.0, inverse(K))",
])
def test_latent_gp_posterior_is_recovered_by_nuts(prior):
    """Latent ``f ~ GP``, Gaussian noise, fixed hyperparameters: the posterior is exactly
    Gaussian, ``Cov = (K^-1 + I / s^2)^-1`` and ``mean = Cov y / s^2``, so NUTS is checked against
    exact draws --- once per multivariate-normal parametrisation, which must all agree. Matern 5/2
    with ``rho`` near the input spacing keeps ``K`` well conditioned (and its float32 inverse
    usable)."""
    n, alpha, rho, s = 8, 1.0, 0.8, 0.5
    x = np.linspace(0.0, 7.0, n)
    y = np.cos(x) + s * np.random.default_rng(8).normal(size=n)
    model = compile_model("""
        data { int n; array[n] real x; array[n] real y; real alpha; real rho; real s; }
        parameters { array[n] real f; }
        model {
          array[n, n] real K = add_diag(gp_matern52_cov(x, alpha, rho), 1e-6);
          f ~ %s;
          y ~ normal(f, s);
        }
    """ % prior, {"n": n, "x": x, "y": y, "alpha": alpha, "rho": rho, "s": s})
    K = _ref("gp_matern52_cov", x, x, (alpha, rho)) + 1e-6 * np.eye(n)
    cov = np.linalg.inv(np.linalg.inv(K) + np.eye(n) / s**2)
    mean = cov @ y / s**2
    chol = np.linalg.cholesky(cov)
    problem = TargetProblem(
        name="dsl_latent_gp", model=model, dim=n, labels=[f"f{i}" for i in range(n)],
        exact_sampler=lambda m, rng: mean + rng.standard_normal((m, n)) @ chol.T,
        mean=mean, cov=cov)
    report = evaluate(problem, {"nuts": nuts()}, n_warmup=1000, n_samples=8000, seed=0)
    print("\n" + report.summary())
    report.assert_correct()
