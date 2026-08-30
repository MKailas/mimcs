"""Tests for the model DSL (``docs/design/08_model_dsl.md``).

The core check is *density equivalence*: each of the seven hand-written `problems.py` models,
re-expressed in the DSL, must compile to a `Model` whose coordinate-space log-density (and its
gradient) matches the hand-written one. Comparing `log_prob_at_coordinate` folds in the chart
Jacobian, so this exercises parameter mapping (incl. the parent-dependent `nested_uniform`
bound), the `target`/`~` accumulation, index lowering, distribution adapters, and matmul.

Gradients must match exactly (what HMC uses; additive constants drop out). Densities match
exactly when the source is written with explicit `target += <unnormalized>`, and up to an
additive constant when `~` is used (SciPy logpdfs are normalized). Seeds are fixed.
"""

import numpy as np
import jax
import jax.numpy as jnp

import pytest

from mimcs import compile_model, DslError
from mimcs.dsl import semantics
from mimcs.dsl.parser import parse
from mimcs.testing import problems as P, evaluate, TargetProblem, adaptive_mh, nuts


# --- unit tests of the error-prone pieces ------------------------------------ #

def test_index_lowering():
    assert semantics.lower_scalar_index(1) == 0
    assert semantics.lower_scalar_index(5) == 4
    assert semantics.lower_range(1, 3) == slice(0, 3)      # Stan a[1:3] -> JAX a[0:3]
    assert semantics.lower_range(2, None) == slice(1, None)
    assert semantics.lower_range(None, 3) == slice(None, 3)
    assert semantics.lower_range(None, None) == slice(None, None)


def test_new_axis_indexing_matches_numpy():
    """``None`` in an index is ``jnp.newaxis``: the NumPy way to reshape for broadcasting."""
    e = np.array([0.5, -0.25, 2.0, 1.0])
    src = ("data { int n; array[n] real e; }\nparameters { real m; }\n"
           "model { target += sum(e[:, None] .* e[None, :]) + m; }")
    fn = compile_model(src, {"n": 4, "e": e}).log_prob_fns["target"]
    assert np.isclose(float(fn({"m": jnp.asarray(0.0)})),
                      float(np.sum(e[:, None] * e[None, :])), rtol=1e-5)


def test_new_axis_on_a_matrix_and_mixed_with_ranges():
    A = np.arange(6.0).reshape(2, 3)
    src = ("data { array[2,3] real A; }\nparameters { real m; }\n"
           "model { target += sum(A[:, None, :]) + sum(A[1:2, None, 1]) + m; }")
    fn = compile_model(src, {"A": A}).log_prob_fns["target"]
    expected = float(np.sum(A[:, None, :]) + np.sum(A[0:2, None, 0]))
    assert np.isclose(float(fn({"m": jnp.asarray(0.0)})), expected, rtol=1e-5)


def test_a_new_axis_does_not_disturb_an_eager_constant():
    """``transformed data`` runs eagerly on numpy arrays; a reshape must not force a conversion.

    ``_is_traced_key`` decides whether to hand the base to ``jnp.asarray`` first. ``None`` is
    static (it is a pure reshape), and treating it otherwise would silently convert a float64
    numpy constant to float32 here --- invisible unless asserted.
    """
    e = np.array([0.5, -0.25, 2.0, 1.0])
    src = ("data { int n; array[n] real e; }\n"
           "transformed data { array[n,1] real col = e[:, None]; }\n"
           "parameters { real m; }\nmodel { target += sum(col) + m; }")
    factory = compile_model(src, data=None)
    spec = factory.analyze({"n": 4, "e": e})
    col = spec.constants["col"]
    assert isinstance(col, np.ndarray), f"eager constant became {type(col).__name__}"
    assert col.dtype == e.dtype and col.shape == (4, 1)


def test_none_cannot_be_a_slice_bound():
    with pytest.raises(DslError, match="cannot be a slice bound"):
        parse("data { array[3] real a; } model { target += sum(a[None:2]); }")


def test_const_eval():
    assert semantics.const_eval(parse("model{ target += 0; }").blocks[0].body[0].value, {}) == 0
    # a small arithmetic constant via the parser
    decl = parse("parameters{ array[2 * 3] real x; } model{}").blocks[0].body[0]
    assert semantics.const_eval(decl.shape[0], {}) == 6


def test_parse_error_has_span():
    with pytest.raises(DslError) as e:
        parse("parameters { real x }")               # missing ';'
    assert e.value.span is not None and "expected" in str(e.value)


def _eval0(model):
    return model.log_prob_at_coordinate(
        jnp.zeros(1), model.init_chart_hyperparams(), model.init_chart_indices())


def test_unknown_distribution_and_name():
    # the interpreter builds a closure lazily, so these surface on first evaluation (trace)
    with pytest.raises(DslError, match="distribution"):
        _eval0(compile_model("parameters{ real x; } model{ x ~ wishart(1); }").build())
    with pytest.raises(DslError, match="unknown name"):
        _eval0(compile_model("parameters{ real x; } model{ target += y; }").build())


def test_missing_data_errors():
    with pytest.raises(DslError, match="missing data"):
        compile_model("data{ real s; } parameters{ real x; } model{ x ~ normal(0, s); }").build()


# --- density / gradient equivalence with the hand-written problems ----------- #

def _assert_equivalent(dsl_model, hand_model, dim, *, exact_density, n=50, atol=1e-4):
    rng = np.random.default_rng(0)
    hp, ci = dsl_model.init_chart_hyperparams(), dsl_model.init_chart_indices()
    f_dsl = lambda q: dsl_model.log_prob_at_coordinate(q, hp, ci)
    f_hand = lambda q: hand_model.log_prob_at_coordinate(q, hp, ci)
    diffs = []
    for _ in range(n):
        q = jnp.asarray(rng.standard_normal(dim))
        assert np.allclose(np.asarray(jax.grad(f_dsl)(q)), np.asarray(jax.grad(f_hand)(q)),
                           atol=atol), "gradient mismatch"
        diffs.append(float(f_dsl(q)) - float(f_hand(q)))
    diffs = np.asarray(diffs)
    assert np.std(diffs) < atol, "density differs by more than a constant"
    if exact_density:
        assert np.abs(diffs).max() < atol, "density not exact"


CORRELATED_GAUSSIAN = """
data { int d; array[d] real mu; array[d, d] real precision; }
parameters { array[d] real x; }
model { target += -0.5 * (x - mu) * (precision * (x - mu)); }
"""

ROSENBROCK = """
data { real a; real b; }
parameters { array[2] real x; }
model { target += -((a - x[1])^2 + b * (x[2] - x[1]^2)^2); }
"""

NEAL_FUNNEL = """
data { int nx; real scale; }
parameters { real v; array[nx] real x; }
model {
  v ~ normal(0, scale);
  x ~ normal(0, exp(v / 2));
}
"""

POSITIVE_LOGNORMAL = """
data { real sigma; }
parameters { real<lower=0> s; }
model { target += -0.5 * (log(s) / sigma)^2 - log(s); }
"""

UNIFORM_INTERVAL = """
data { real L; real U; }
parameters { real<lower=L, upper=U> x; }
model { }
"""

NESTED_UNIFORM = """
parameters {
  real<lower=0, upper=1> a;
  real<lower=0, upper=a> b;
}
model { target += -log(a); }
"""


def test_dsl_correlated_gaussian():
    cov = np.array([[2.0, 1.4], [1.4, 1.5]]); mean = np.array([1.0, -2.0])
    dsl = compile_model(CORRELATED_GAUSSIAN,
                        data={"d": 2, "mu": mean, "precision": np.linalg.inv(cov)})
    _assert_equivalent(dsl, P.correlated_gaussian(mean=mean, cov=cov).model, 2, exact_density=True)


def test_dsl_rosenbrock():
    dsl = compile_model(ROSENBROCK, data={"a": 1.0, "b": 5.0})
    _assert_equivalent(dsl, P.rosenbrock(a=1.0, b=5.0).model, 2, exact_density=True)


def test_dsl_neal_funnel():
    dsl = compile_model(NEAL_FUNNEL, data={"nx": 1, "scale": 3.0})
    # '~' uses normalized logpdfs, so the density matches only up to an additive constant.
    _assert_equivalent(dsl, P.neal_funnel_blocks(dim=2, scale=3.0).model, 2, exact_density=False)


def test_dsl_positive_lognormal():
    dsl = compile_model(POSITIVE_LOGNORMAL, data={"sigma": 1.0})
    _assert_equivalent(dsl, P.positive_lognormal(sigma=1.0).model, 1, exact_density=True)


def test_dsl_uniform_interval():
    dsl = compile_model(UNIFORM_INTERVAL, data={"L": -2.0, "U": 3.0})
    _assert_equivalent(dsl, P.uniform_interval(lower=-2.0, upper=3.0).model, 1, exact_density=True)


def test_dsl_nested_uniform():
    dsl = compile_model(NESTED_UNIFORM).build()          # no data -> factory
    _assert_equivalent(dsl, P.nested_uniform().model, 2, exact_density=True)


# --- end-to-end: a DSL-compiled model is sampler-compatible ------------------ #

def test_dsl_model_samples_end_to_end():
    cov = np.array([[2.0, 1.4], [1.4, 1.5]]); mean = np.array([1.0, -2.0])
    chol = np.linalg.cholesky(cov)
    model = compile_model(CORRELATED_GAUSSIAN,
                          data={"d": 2, "mu": mean, "precision": np.linalg.inv(cov)})
    problem = TargetProblem(
        name="dsl_correlated_gaussian", model=model, dim=2, labels=["x0", "x1"],
        exact_sampler=lambda n, rng: mean + rng.standard_normal((n, 2)) @ chol.T,
        mean=mean, cov=cov)
    report = evaluate(problem, {"mh": adaptive_mh(step_size=0.5)},
                      n_warmup=3000, n_samples=15000, seed=0)
    print("\n" + report.summary())
    report.assert_correct()


def _eval(model, q):
    return float(model.log_prob_at_coordinate(
        jnp.asarray(q, float), model.init_chart_hyperparams(), model.init_chart_indices()))


def test_for_loop_indexed_assignment_and_transformed_parameters():
    """An unrolled `for`, indexed assignment into a declared array, and a transformed
    parameter recomputed each evaluation."""
    model = compile_model("""
        data { int n; }
        parameters { array[n] real x; }
        transformed parameters { array[n] real y; for (i in 1:n) { y[i] = x[i] * x[i]; } }
        model { for (i in 1:n) { target += -0.5 * y[i]; } }
    """, data={"n": 3})
    assert np.isclose(_eval(model, [1.0, 2.0, 3.0]), -0.5 * (1 + 4 + 9))


def test_transpose_and_matmul_quadratic_form():
    P = np.array([[2.0, 0.5], [0.5, 1.0]])
    model = compile_model(
        "data{ int d; array[d,d] real P; } parameters{ array[d] real x; }"
        " model{ target += -0.5 * (x' * (P * x)); }", data={"d": 2, "P": P})
    x = np.array([1.0, -1.0])
    assert np.isclose(_eval(model, x), -0.5 * x @ P @ x)


def test_static_if_else_chain():
    src = ("data{ int k; } parameters{ real x; }"
           " model{ if (k == 1) { target += x; } else if (k == 2) { target += 2 * x; }"
           "        else { target += 0; } }")
    assert np.isclose(_eval(compile_model(src, data={"k": 1}), [1.5]), 1.5)
    assert np.isclose(_eval(compile_model(src, data={"k": 2}), [1.5]), 3.0)
    assert np.isclose(_eval(compile_model(src, data={"k": 9}), [1.5]), 0.0)


def test_factory_reuse():
    """A parsed program can be built against different datasets."""
    factory = compile_model(POSITIVE_LOGNORMAL)
    m1, m2 = factory.build({"sigma": 1.0}), factory.build({"sigma": 2.0})
    hp, ci = m1.init_chart_hyperparams(), m1.init_chart_indices()
    q = jnp.array([0.7])
    assert float(m1.log_prob_at_coordinate(q, hp, ci)) != float(m2.log_prob_at_coordinate(q, hp, ci))


# --- discrete distributions (logpmf) in the `~` statement -------------------- #

def _assert_density_matches(model, ref, points, *, atol=1e-4):
    """Compiled-model coordinate density and its gradient match a reference at each point."""
    hp, ci = model.init_chart_hyperparams(), model.init_chart_indices()
    f = lambda q: model.log_prob_at_coordinate(q, hp, ci)
    for q in points:
        q = jnp.asarray(q, float)
        assert np.allclose(float(f(q)), float(ref(q)), atol=atol), "density mismatch"
        assert np.allclose(np.asarray(jax.grad(f)(q)), np.asarray(jax.grad(ref)(q)),
                           atol=atol), "gradient mismatch"


def test_discrete_registry_parameterization_adapters():
    """Lock the Stan->SciPy parameterization adapters against explicit formulas."""
    import jax.scipy.stats as jss
    from jax.scipy.special import gammaln
    from mimcs.dsl.distributions import DISTRIBUTIONS

    k = jnp.array([0.0, 2.0, 5.0, 10.0])

    # neg_binomial(alpha, beta): C(k+alpha-1,k) (beta/(beta+1))^alpha (1/(beta+1))^k
    alpha, beta = 3.5, 2.0
    stan_nb = (gammaln(k + alpha) - gammaln(alpha) - gammaln(k + 1)
               + alpha * jnp.log(beta / (beta + 1)) + k * jnp.log(1 / (beta + 1)))
    assert np.allclose(np.asarray(DISTRIBUTIONS["neg_binomial"](k, alpha, beta)),
                       np.asarray(stan_nb), atol=1e-4)

    # neg_binomial_2(mu, phi): mean mu, var mu + mu^2/phi
    mu, phi = 4.0, 1.5
    stan_nb2 = (gammaln(k + phi) - gammaln(phi) - gammaln(k + 1)
                + phi * jnp.log(phi / (phi + mu)) + k * jnp.log(mu / (mu + phi)))
    assert np.allclose(np.asarray(DISTRIBUTIONS["neg_binomial_2"](k, mu, phi)),
                       np.asarray(stan_nb2), atol=1e-4)

    # poisson(rate): rate is SciPy mu
    assert np.allclose(np.asarray(DISTRIBUTIONS["poisson"](k, 2.5)),
                       np.asarray(jss.poisson.logpmf(k, 2.5)), atol=1e-6)
    # binomial(N, theta) = SciPy binom(n, p)
    assert np.allclose(np.asarray(DISTRIBUTIONS["binomial"](k, 10, 0.3)),
                       np.asarray(jss.binom.logpmf(k, 10, 0.3)), atol=1e-6)


def test_dsl_poisson_likelihood():
    """`y ~ poisson(exp(x))` accumulates sum(logpmf); exact density and gradient in x."""
    import jax.scipy.stats as jss
    y = np.array([0.0, 3.0, 5.0, 2.0])
    model = compile_model(
        "data { int n; array[n] real y; } parameters { real x; }"
        " model { y ~ poisson(exp(x)); }", data={"n": 4, "y": y})
    ref = lambda q: jnp.sum(jss.poisson.logpmf(jnp.asarray(y), jnp.exp(q[0])))
    _assert_density_matches(model, ref, [[0.5], [1.2], [-0.3]])


def test_dsl_bernoulli_and_binomial_with_logit_link():
    import jax.scipy.stats as jss
    yb = np.array([1.0, 0.0, 1.0, 1.0, 0.0])
    mb = compile_model(
        "data { int n; array[n] real y; } parameters { real x; }"
        " model { y ~ bernoulli(sigmoid(x)); }", data={"n": 5, "y": yb})
    sig = lambda z: 1.0 / (1.0 + jnp.exp(-z))
    _assert_density_matches(
        mb, lambda q: jnp.sum(jss.bernoulli.logpmf(jnp.asarray(yb), sig(q[0]))),
        [[0.0], [0.8], [-1.5]])

    yc = np.array([3.0, 7.0, 5.0])
    mc = compile_model(
        "data { int n; real N; array[n] real y; } parameters { real x; }"
        " model { y ~ binomial(N, sigmoid(x)); }", data={"n": 3, "N": 10.0, "y": yc})
    _assert_density_matches(
        mc, lambda q: jnp.sum(jss.binom.logpmf(jnp.asarray(yc), 10.0, sig(q[0]))),
        [[0.2], [1.0], [-0.7]])


def test_dsl_multinomial_with_softmax():
    """Joint (scalar) multinomial over a softmax-constructed simplex; integer counts."""
    import jax.scipy.stats as jss
    y = np.array([2.0, 3.0, 5.0])
    model = compile_model(
        "data { int K; array[K] real y; } parameters { array[K] real z; }"
        " transformed parameters { array[K] real theta; theta = exp(z) / sum(exp(z)); }"
        " model { y ~ multinomial(theta); }", data={"K": 3, "y": y})

    def ref(q):
        theta = jnp.exp(q) / jnp.sum(jnp.exp(q))
        return jss.multinomial.logpmf(jnp.asarray(y).astype(jnp.int32), int(y.sum()), theta)

    _assert_density_matches(model, ref, [[0.0, 0.0, 0.0], [0.5, -0.3, 1.0]])


def test_dsl_unknown_discrete_distribution_still_errors():
    # `hypergeometric` is not registered -> the usual unknown-distribution error (on first eval).
    # (This used to name `categorical`, which is now a real distribution -- see test_discrete.py.)
    with pytest.raises(DslError, match="distribution"):
        _eval0(compile_model("parameters{ real x; } model{ x ~ hypergeometric(1); }").build())


# --- user-defined functions (the `functions` block) -------------------------- #

FUNCTIONS_SRC = """
functions {
  array real add_two(array real a, array real b) { return a + b; }
  real scale(real x, real k) { return x * k; }
  real weighted(array real w, array real z) { return sum(scale(1.0, 2.0) * add_two(w, z)); }
}
data { int n; }
parameters { array[n] real x; }
model { target += -0.5 * weighted(x, x); }
"""


def test_function_definition_and_call():
    """A function calling another function and a builtin, from the model block."""
    model = compile_model(FUNCTIONS_SRC, data={"n": 3})
    x = [1.0, 2.0, 3.0]
    assert np.isclose(_eval(model, x), -0.5 * 2.0 * 2.0 * sum(x))


def test_function_signature_forms_parse():
    """Both array spellings, and the scalar forms, land in the AST as declared."""
    prog = parse("""
        functions {
          real f(real x, int k) { return x; }
          array real g(array real a) { return a; }
          array[] real h(array[] real a, array[,] real m) { return a; }
        }
        parameters { real y; } model { target += y; }
    """)
    defs = {d.name: d for d in prog.blocks[0].body}
    assert set(defs) == {"f", "g", "h"}
    assert [p.type.base for p in defs["f"].params] == ["real", "int"]
    assert defs["f"].return_type.dims == () and defs["f"].params[0].type.dims == ()
    assert defs["g"].return_type.dims is None            # `array real`: rank left unsaid
    assert defs["h"].return_type.dims == (None,)         # `array[] real`
    assert defs["h"].params[1].type.dims == (None, None)  # `array[,] real`


def test_function_equivalent_to_the_inlined_model():
    """The real check: a function is exactly the expression it stands for."""
    inlined = compile_model("""
        data { int n; }
        parameters { array[n] real x; }
        model { target += -0.5 * sum(2.0 * (x + x)); }
    """, data={"n": 3})
    _assert_equivalent(compile_model(FUNCTIONS_SRC, data={"n": 3}), inlined, 3,
                       exact_density=True)


def test_function_locals_loops_and_early_return():
    """A body with a local array, an unrolled loop, indexed assignment --- and a `return`
    from inside the unrolling, which stops it."""
    model = compile_model("""
        functions {
          real cumulate(real x, int n) {
            array[3] real acc;
            for (i in 1:n) { acc[i] = x * i; }
            return sum(acc);
          }
          real first_big(real x, int n) {
            for (i in 1:n) { if (i == 2) { return x * i; } }
            return 0.0;
          }
        }
        parameters { real x; }
        model { target += cumulate(x, 3) + first_big(x, 3); }
    """, data={})
    assert np.isclose(_eval(model, [2.0]), 2.0 * (1 + 2 + 3) + 2.0 * 2)


def test_recursive_function():
    """Recursion resolves (the table is complete before evaluation) with a static base case."""
    model = compile_model("""
        functions { real power(real x, int n) { if (n == 0) { return 1.0; }
                                                return x * power(x, n - 1); } }
        parameters { real x; }
        model { target += power(x, 4); }
    """, data={})
    assert np.isclose(_eval(model, [1.5]), 1.5 ** 4)


def test_runaway_recursion_is_a_dsl_error():
    """A depth cap, so this is a span-carrying DslError rather than a RecursionError."""
    with pytest.raises(DslError, match="call depth"):
        _eval0(compile_model("""
            functions { real f(real x) { return f(x); } }
            parameters { real x; }
            model { target += f(x); }
        """).build())


def test_function_used_in_transformed_data():
    """The eager path gets the same call table."""
    model = compile_model("""
        functions { real twice(real x) { return 2 * x; } }
        data { real a; }
        transformed data { real b = twice(a); }
        parameters { real x; }
        model { target += b * x; }
    """, data={"a": 3.0})
    assert np.isclose(_eval(model, [2.0]), 6.0 * 2.0)


@pytest.mark.parametrize("src, match", [
    # name reservation and uniqueness
    ("functions { real exp(real x) { return x; } }", "builtin function"),
    ("functions { real normal(real x) { return x; } }", "distribution name"),
    ("functions { real for(real x) { return x; } }", "reserved word"),
    ("functions { real f(real x) { return x; } real f(real y) { return y; } }",
     "already defined"),
    ("functions { real f(real a, real a) { return a; } }", "duplicate argument"),
    ("functions { real f(real target) { return target; } }", "reserved word"),
    # purity, return type, returning at all
    ("functions { real f(real x) { x ~ normal(0, 1); return x; } }", "functions are pure"),
    ("functions { real f(real x) { target += x; return x; } }", "functions are pure"),
    ("functions { void f(real x) { return; } }", "no observable effect"),
    ("functions { real f(real x) { real z = x; } }", "never returns"),
    # scope: arguments and locals only
    ("functions { real f(real x) { return x + y; } }", "sees only its arguments"),
    # signatures
    ("functions { real f(array[2] real a) { return a[1]; } }", "no declared size"),
    ("functions { real f(unit_vector[3] u) { return u[1]; } }", "`parameters` block"),
    ("functions { real f(real<lower=0> x) { return x; } }", "cannot carry lower/upper"),
    ("functions { real x; }", "function definitions only"),
    # calls
    ("functions { real f(real x) { return x; } } transformed parameters { real z = f(1, 2); }",
     "takes 1 argument"),
    ("functions { real f(real x) { return g(x); } }", "unknown function"),
])
def test_function_errors(src, match):
    with pytest.raises(DslError, match=match):
        compile_model(src + " parameters { real y; } model { target += y; }", data={})


def test_return_outside_a_function_errors():
    with pytest.raises(DslError, match="only allowed in a function body"):
        compile_model("parameters { real y; } model { return y; }", data={})


# --- multiple named `model` components --------------------------------------- #

SPLIT_FUNNEL = """
data { int nx; real scale; }
parameters { real v; array[nx] real x; }
model prior      { v ~ normal(0, scale); }
model likelihood { x ~ normal(0, exp(v / 2)); }
"""


def test_named_components_are_keyed_by_name():
    model = compile_model(SPLIT_FUNNEL, data={"nx": 2, "scale": 3.0})
    assert list(model.log_prob_fns) == ["prior", "likelihood"]   # source order is preserved


def test_split_components_match_the_fused_model():
    """Splitting a model must be a no-op on the density and its gradient."""
    data = {"nx": 2, "scale": 3.0}
    _assert_equivalent(compile_model(SPLIT_FUNNEL, data=data),
                       compile_model(NEAL_FUNNEL, data=data), 3, exact_density=True)


def test_components_sum_to_the_joint_density():
    model = compile_model(SPLIT_FUNNEL, data={"nx": 2, "scale": 3.0})
    sample = {"v": jnp.array(0.4), "x": jnp.array([1.0, -0.5])}
    parts = model.log_prob_components(sample)
    assert set(parts) == {"prior", "likelihood"}
    assert np.isclose(float(sum(parts.values())), float(model.log_prob(sample)))


def test_bare_and_named_components_coexist():
    model = compile_model("""
        parameters { real v; real x; }
        model { v ~ normal(0, 1); }
        model likelihood { x ~ normal(v, 1); }
    """, data={})
    assert set(model.log_prob_fns) == {"target", "likelihood"}


def test_transformed_parameters_are_visible_in_every_component():
    """TP statements are prepended to each component closure, so both see `s`."""
    model = compile_model("""
        parameters { real log_s; real x; }
        transformed parameters { real s = exp(log_s); }
        model prior      { target += -0.5 * log_s^2; }
        model likelihood { target += -0.5 * (x / s)^2; }
    """, data={})
    log_s, x = 0.3, 1.2
    assert np.isclose(_eval(model, [log_s, x]),
                      -0.5 * log_s ** 2 - 0.5 * (x / np.exp(log_s)) ** 2)


def test_one_model_potential_per_component():
    """The point of the split: the Hamiltonian gets one component per `model` block, which is
    what a multi-rate integrator would give different rates."""
    from mimcs.hmc import default_potentials
    model = compile_model(SPLIT_FUNNEL, data={"nx": 2, "scale": 3.0})
    assert [p.id for p in default_potentials(model)] == ["V_prior", "V_likelihood"]


def test_duplicate_potential_ids_are_rejected():
    """A collision would silently reuse another component's cached gradient."""
    from mimcs.hmc import default_potentials
    from mimcs.model import Model
    from mimcs.model import EuclideanParameter, PositiveParameter

    model = Model([EuclideanParameter("x"), PositiveParameter("s")],
                  {"jacobian": lambda d: jnp.zeros(())})
    with pytest.raises(ValueError, match="duplicate potential id"):
        default_potentials(model)


@pytest.mark.parametrize("src, match", [
    ("parameters { real y; } model { target += y; } model { target += y; }",
     "duplicate `model` block"),
    ("parameters { real y; } model a { target += y; } model a { target += y; }",
     "duplicate model component 'a'"),
    ("parameters { real y; } model { target += y; } model target { target += y; }",
     "duplicate model component 'target'"),
    ("parameters { real y; } model jacobian { target += y; }",
     "reserved model-component name"),
    ("parameters { real y; } transformed parameters { y ~ normal(0, 1); }"
     " model a { target += y; } model b { target += y; }", "ambiguous"),
    ("parameters { real y; } transformed parameters { target += y; }"
     " model a { target += y; } model b { target += y; }", "ambiguous"),
])
def test_component_errors(src, match):
    with pytest.raises(DslError, match=match):
        compile_model(src, data={})


def test_density_statement_in_transformed_parameters_is_fine_with_one_component():
    """The multi-component rule must not regress the single-component (Stan) behaviour."""
    model = compile_model("""
        parameters { real y; }
        transformed parameters { y ~ normal(0, 1); }
        model { target += 0; }
    """, data={})
    assert np.isclose(_eval(model, [1.0]), float(jax.scipy.stats.norm.logpdf(1.0, 0.0, 1.0)))


def test_functions_and_components_together():
    model = compile_model("""
        functions { real half_sq(real x) { return -0.5 * x^2; } }
        parameters { real v; real x; }
        model prior      { target += half_sq(v); }
        model likelihood { target += half_sq(x - v); }
    """, data={})
    assert list(model.log_prob_fns) == ["prior", "likelihood"]
    assert np.isclose(_eval(model, [0.5, 1.5]), -0.5 * 0.25 - 0.5 * 1.0)


def test_split_model_samples_end_to_end():
    """A split model must sample as well as an unsplit one --- and NUTS is the sampler that
    actually builds one potential per component (adaptive_mh never constructs potentials)."""
    cov = np.array([[2.0, 1.4], [1.4, 1.5]]); mean = np.array([1.0, -2.0])
    precision = np.linalg.inv(cov)
    model = compile_model("""
        data { int d; array[d] real mu; array[d, d] real precision; }
        parameters { array[d] real x; }
        model quadratic { target += -0.5 * (x - mu) * (precision * (x - mu)); }
        model constant  { target += 0; }
    """, data={"d": 2, "mu": mean, "precision": precision})
    assert list(model.log_prob_fns) == ["quadratic", "constant"]
    problem = TargetProblem(
        name="dsl_split_gaussian", model=model, dim=2, labels=["x0", "x1"],
        exact_sampler=lambda n, rng: mean + rng.standard_normal((n, 2)) @ np.linalg.cholesky(cov).T,
        mean=mean, cov=cov)
    report = evaluate(problem, {"nuts": nuts()}, n_warmup=1000, n_samples=8000, seed=0)
    print("\n" + report.summary())
    report.assert_correct()


# --- the model spec: component cost and chart options ------------------------ #

SPLIT_REGRESSION = """
data { int n; int k; array[n, k] real X; array[n] real y; real prior_scale; }
parameters { array[k] real beta; }
model prior      { beta ~ normal(0, prior_scale); }
model likelihood { y ~ normal(X * beta, 1.0); }
"""


def _regression_data(n=1000, k=3):
    """``n=1000`` (not a smaller round number) so ``beta`` (size ``k=3``) stays under the cost
    rule's 1/500 threshold even though the threshold pools parameters in with constants --- the
    point of these fixtures is a clean cheap/expensive split, and at the loose 1/500 bar a small
    absolute scale (e.g. ``n=100``) makes even a 3-element parameter count as "large"."""
    rng = np.random.default_rng(0)
    return {"n": n, "k": k, "X": rng.normal(size=(n, k)), "y": rng.normal(size=n),
            "prior_scale": 2.0}


def test_constant_size_measures_what_it_can():
    from mimcs.dsl.cost import constant_size
    assert constant_size(3) == 1 and constant_size(3.0) == 1
    assert constant_size(np.zeros((2, 3))) == 6
    assert constant_size(jnp.zeros(4)) == 4
    assert constant_size([1, 2, 3]) == 3
    assert constant_size(None) is None            # a claim, not a measurement
    assert constant_size([[1, 2], [3]]) is None   # ragged


def test_large_item_rule_thresholds():
    """The rule: the largest is always large, a scalar never is, >= 1/500th of the largest is.

    ``large_items`` is generic over what it is handed --- constants, parameters, or (as
    :func:`classify_components` uses it) both pooled together --- so these are exercised as a
    plain ``{name: size}`` dict.
    """
    from mimcs.dsl.cost import large_items
    assert large_items({}) == frozenset()                       # nothing to compare
    assert large_items({"a": 1, "b": 1}) == frozenset()          # all scalars
    assert large_items({"a": 7}) == {"a"}                        # the only one is the largest
    assert large_items({"a": 100000, "b": 200, "c": 199, "d": 1}) == {"a", "b"}  # 1/500 = 200
    assert large_items({"a": 100000, "b": 199}) == {"a"}         # just under the threshold
    assert large_items({"a": 5, "b": 1}) == {"a"}                # a scalar is never large


def test_component_costs_split_prior_from_likelihood():
    spec = compile_model(SPLIT_REGRESSION).analyze(_regression_data())
    assert spec.constant_sizes == {"n": 1, "k": 1, "X": 3000, "y": 1000, "prior_scale": 1}
    assert spec.parameter_sizes == {"beta": 3}          # small next to X/y, so still cheap
    assert spec.large_constants == {"X", "y"}
    assert spec.large_parameters == frozenset()
    assert [(c.name, c.cost, sorted(c.touches)) for c in spec.components] == [
        ("prior", "cheap", []),
        ("likelihood", "expensive", ["X", "y"])]
    model = spec.build()
    assert model.cheap_components == {"prior"}
    assert model.expensive_components == {"likelihood"} and model.is_cheap("prior")


def test_a_large_parameter_makes_its_component_expensive():
    """The point of pooling parameters into the cost rule: a hierarchical prior over a large
    *parameter* (not a large constant) is exactly the `reg_horseshoe` case, reproduced small."""
    spec = compile_model("""
        data { int d; }
        parameters { array[d] real beta; real<lower=0> tau; }
        model prior      { beta ~ normal(0, tau); tau ~ normal(0, 1); }
        model likelihood { target += -0.5 * sum(beta^2); }
    """).analyze({"d": 2000})
    assert spec.constant_sizes == {"d": 1}
    assert spec.parameter_sizes == {"beta": 2000, "tau": 1}
    assert spec.large_constants == frozenset()
    assert spec.large_parameters == {"beta"}             # tau is a scalar: never large
    assert [(c.name, c.cost) for c in spec.components] == [
        ("prior", "expensive"), ("likelihood", "expensive")]      # both read `beta`


def test_a_model_without_data_has_only_cheap_components():
    """Nothing is large when every constant and parameter is a scalar, so nothing is expensive
    --- which is the right answer: there is nothing for a multi-rate split to buy anything on.
    ``nx=1`` (not 2): at the loose 1/500 threshold even a 2-element parameter is "large" once it
    is the biggest thing around, which is exactly what the next test pins on purpose."""
    spec = compile_model(SPLIT_FUNNEL).analyze({"nx": 1, "scale": 3.0})
    assert spec.large_constants == frozenset()
    assert spec.large_parameters == frozenset()
    assert spec.build().cheap_components == {"prior", "likelihood"}
    assert any("nothing is large" in r for r in spec.rationale)


def test_a_lone_vector_parameter_can_make_itself_large():
    """With no data at all, a large enough vector parameter is still the biggest thing around ---
    and the rule pools parameters in, so it is not exempt just because there is no data."""
    spec = compile_model(SPLIT_FUNNEL).analyze({"nx": 2, "scale": 3.0})
    assert spec.large_constants == frozenset()
    assert spec.large_parameters == {"x"}
    assert [(c.name, c.cost) for c in spec.components] == [
        ("prior", "cheap"), ("likelihood", "expensive")]      # only `likelihood` reads `x`


def test_a_transformed_data_array_can_be_the_largest_constant():
    spec = compile_model("""
        data { int n; array[n] real y; }
        transformed data { array[n, n] real outer; for (i in 1:n) { outer[i] = y * y[i]; } }
        parameters { real mu; }
        model prior { mu ~ normal(0, 1); }
        model likelihood { target += sum(outer) * mu; }
    """).analyze({"n": 20, "y": np.ones(20)})
    assert spec.constant_sizes["outer"] == 400
    assert spec.large_constants == {"outer", "y"}   # 20/400 clears the loose 1/500 bar too
    assert [c.cost for c in spec.components] == ["cheap", "expensive"]   # neither reads `y`
    # directly -- it is folded into `outer` inside `transformed data`, before `c.reads` starts.


def test_a_much_larger_constant_hides_a_smaller_data_array():
    """A known limitation of comparing against the largest item, pinned deliberately.

    A wide design matrix (`n * k` elements) pushes the response vector (`n`) below 1/500th of it
    once `k` is a few hundred, so a component reading only `y` is called cheap although it sweeps
    every observation. The 1/500 threshold (vs. an all-constants version's 1/10) makes this need a
    much wider matrix than before to bite --- on the real `diamonds` program (`X` 125000, `Y`
    5000, a 4% ratio) `Y` now clears the bar and is correctly large; it takes `k` in the hundreds,
    not the tens, to still hide it. Harmless while such programs are single-component, and the
    reason the classifier is swappable (`analyze(data, classify=...)`).
    """
    from mimcs.dsl.cost import large_items
    n, k = 100, 600
    assert large_items({"X": n * k, "y": n, "N": 1}) == {"X"}          # y is 1/600 of X: hidden
    assert large_items({"X": n * 5, "y": n, "N": 1}) == {"X", "y"}     # k = 5: y is 1/5, not hidden


def test_a_leaked_loop_variable_is_not_counted_as_a_constant():
    """`for` leaves its counter bound in the eager environment; it is not a constant."""
    spec = compile_model("""
        data { int n; }
        transformed data { real s = 0; for (i in 1:n) { s = s + i; } }
        parameters { real mu; }
        model { target += s * mu; }
    """).analyze({"n": 3})
    assert "i" not in spec.constant_sizes and set(spec.constant_sizes) == {"n", "s"}


def test_transformed_parameters_reading_data_makes_every_component_expensive():
    """TP runs inside every component's closure, so the cost is real, not an artefact."""
    spec = compile_model("""
        data { int n; array[n] real y; }
        parameters { real mu; }
        transformed parameters { real fit = sum(y) * mu; }
        model prior      { mu ~ normal(0, 1); }
        model likelihood { target += fit; }
    """).analyze({"n": 50, "y": np.ones(50)})
    assert spec.shared_reads == {"y", "mu"}      # `fit` is declared here, so it is not free
    assert [c.cost for c in spec.components] == ["expensive", "expensive"]
    assert any("transformed parameters" in r for r in spec.rationale)


def test_cost_override_survives_the_build():
    spec = compile_model(SPLIT_REGRESSION).analyze(_regression_data())
    spec.component("likelihood").cost = "cheap"
    assert spec.build().cheap_components == {"prior", "likelihood"}


def test_an_injected_classifier_replaces_the_rule():
    def everything_expensive(spec):
        for c in spec.components:
            c.cost = "expensive"
        spec.rationale.append("custom rule: nothing is cheap")

    spec = compile_model(SPLIT_REGRESSION).analyze(_regression_data(),
                                                   classify=everything_expensive)
    assert spec.build().cheap_components == frozenset()
    assert "custom rule: nothing is cheap" in spec.rationale


def test_the_rationale_records_the_classification():
    rationale = "\n".join(compile_model(SPLIT_REGRESSION).analyze(_regression_data()).rationale)
    assert "largest" in rationale and "'X'" in rationale
    assert "components['prior'].cost = 'cheap'" in rationale


def test_spec_build_equals_the_one_liner():
    """The spec path must change no density: it only labels and configures."""
    data = _regression_data()
    factory = compile_model(SPLIT_REGRESSION)
    _assert_equivalent(factory.analyze(data).build(), factory.build(data), 3,
                       exact_density=True)
    assert compile_model(SPLIT_REGRESSION, data=data).cheap_components == {"prior"}


def test_spec_components_cannot_be_added_or_removed():
    spec = compile_model(SPLIT_REGRESSION).analyze(_regression_data())
    spec.components.pop()
    with pytest.raises(ValueError, match="does not match the program"):
        spec.build()


def test_an_unknown_cost_label_is_rejected():
    spec = compile_model(SPLIT_REGRESSION).analyze(_regression_data())
    spec.component("prior").cost = "free"
    with pytest.raises(ValueError, match="unknown component cost"):
        spec.build()


def test_model_rejects_cheap_components_it_does_not_have():
    from mimcs.model import Model
    from mimcs.model import EuclideanParameter
    with pytest.raises(ValueError, match="not log-density components|not log-density"):
        Model([EuclideanParameter("x")], {"lp": lambda d: jnp.zeros(())},
              cheap_components={"nope"})


def test_a_hand_written_model_can_declare_cheap_components():
    from mimcs.model import Model
    from mimcs.model import EuclideanParameter
    model = Model([EuclideanParameter("x")],
                  {"prior": lambda d: jnp.zeros(()), "lik": lambda d: jnp.zeros(())},
                  cheap_components={"prior"})
    assert model.is_cheap("prior") and not model.is_cheap("lik")
    assert Model([EuclideanParameter("x")], {"lp": lambda d: jnp.zeros(())}).cheap_components \
        == frozenset()                       # unlabelled: nothing is known cheap


# --- the model spec: chart options the grammar cannot express ---------------- #

def test_centering_a_parameter_through_the_spec():
    from mimcs.hmc import default_potentials
    factory = compile_model("parameters { real mu; real<lower=0> s; } model { mu ~ normal(0, s); }")
    spec = factory.analyze({})
    assert [(p.name, p.kind, p.centered) for p in spec.parameters] == [
        ("mu", "real", None), ("s", "real", None)]        # unset: the class default (False)
    plain = spec.build()
    assert [p.centered for p in plain.parameters] == [False, False]
    assert [p.id for p in default_potentials(plain)] == ["V_target", "V_jacobian"]

    spec.parameter("mu").centered = True
    centered = spec.build()
    assert centered.parameters[0].centered is True
    assert centered.parameters[0].is_euclidean() is False   # a centered chart has a Jacobian


def test_adaptive_flag_for_a_unit_vector():
    factory = compile_model("parameters { unit_vector[3] u; } model { target += u[1]; }")
    spec = factory.analyze({})
    assert [(p.name, p.kind) for p in spec.parameters] == [("u", "unit_vector")]
    assert spec.build().parameters[0].adaptive is True      # the class default
    spec.parameter("u").adaptive = False
    assert spec.build().parameters[0].adaptive is False


@pytest.mark.parametrize("param, opt, match", [
    ("unit_vector[3] u", "centered", "no `centered` flag"),
    ("real x", "adaptive", "not a chart option"),
])
def test_a_chart_option_must_apply_to_the_parameter_kind(param, opt, match):
    factory = compile_model(f"parameters {{ {param}; }} model {{ target += 0; }}")
    spec = factory.analyze({})
    setattr(spec.parameters[0], opt, True)
    with pytest.raises(DslError, match=match):
        spec.build()


def test_an_unknown_parameter_or_component_lookup_errors():
    spec = compile_model(SPLIT_REGRESSION).analyze(_regression_data())
    with pytest.raises(KeyError, match="no model component"):
        spec.component("nope")
    with pytest.raises(KeyError, match="no parameter"):
        spec.parameter("nope")


def test_bernoulli_logit_and_inv_gamma_adapters():
    """`bernoulli_logit` is not sugar for `bernoulli(sigmoid(.))`: it is the only form that
    survives a large linear predictor, which is why it exists."""
    import jax.scipy.stats as jss
    from jax.scipy.special import gammaln
    from mimcs.dsl.distributions import DISTRIBUTIONS

    k = jnp.array([0.0, 1.0, 1.0, 0.0])
    alpha = jnp.array([-2.0, 0.5, 3.0, 1.0])
    theta = 1.0 / (1.0 + jnp.exp(-alpha))
    assert np.allclose(np.asarray(DISTRIBUTIONS["bernoulli_logit"](k, alpha)),
                       np.asarray(jss.bernoulli.logpmf(k, theta)), atol=1e-6)
    # ... and stays finite where the sigmoid form dies: a saturated probability only kills the
    # density for an outcome on the *other* side (k=0 with p=1, k=1 with p=0), which is exactly
    # what a badly-scaled linear predictor produces
    big = jnp.array([200.0, -200.0])
    kb = jnp.array([0.0, 1.0])
    assert np.all(np.isfinite(np.asarray(DISTRIBUTIONS["bernoulli_logit"](kb, big))))
    assert not np.all(np.isfinite(
        np.asarray(jss.bernoulli.logpmf(kb, 1.0 / (1.0 + jnp.exp(-big))))))

    # inv_gamma(alpha, beta): alpha log beta - lgamma(alpha) - (alpha+1) log x - beta/x
    x, a, b = jnp.array([0.5, 2.0, 7.0]), 2.5, 1.5
    ref = a * jnp.log(b) - gammaln(a) - (a + 1.0) * jnp.log(x) - b / x
    assert np.allclose(np.asarray(DISTRIBUTIONS["inv_gamma"](x, a, b)), np.asarray(ref), atol=1e-6)
    # it is the density of 1/x under gamma(alpha, rate=beta), Jacobian included
    jac = np.asarray(jss.gamma.logpdf(1.0 / x, a, scale=1.0 / b) - 2.0 * jnp.log(x))
    assert np.allclose(np.asarray(DISTRIBUTIONS["inv_gamma"](x, a, b)), jac, atol=1e-6)


# --- linear algebra builtins -------------------------------------------------- #
#
# Names follow JAX/NumPy rather than Stan (`det`, not `determinant`), so that a DSL model reads
# like the same model written as a plain JAX function. Three have sharp edges the tests pin down:
# `slogdet` returns a pair, `solve_triangular` takes `lower` positionally, and `eigvals` is
# complex.

def _linalg_model(body, data, params="parameters { real d; }", extra=" - 0.5 * d * d"):
    src = (f"data {{ array[3,3] real A; array[3] real b; }}\n{params}\n"
           f"model {{ {body}{extra}; }}")
    model = compile_model(src, data)
    charts = (model.init_chart_hyperparams(), model.init_chart_indices())
    z = jnp.zeros((model.coord_dim,), float)
    return float(model.log_prob_at_coordinate(z, *charts)), model, z, charts


@pytest.fixture
def spd_data():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((3, 3))
    return {"A": A @ A.T + 3 * np.eye(3), "b": rng.standard_normal(3)}


@pytest.mark.parametrize("body,ref", [
    ("target += trace(A)", lambda d: np.trace(d["A"])),
    ("target += det(A)", lambda d: np.linalg.det(d["A"])),
    ("target += sum(solve(A, b))", lambda d: np.linalg.solve(d["A"], d["b"]).sum()),
    ("target += sum(cholesky(A))", lambda d: np.linalg.cholesky(d["A"]).sum()),
    ("target += sum(eigvalsh(A))", lambda d: np.linalg.eigvalsh(d["A"]).sum()),
    ("target += max(abs(eigvals(A)))", lambda d: np.abs(np.linalg.eigvals(d["A"])).max()),
])
def test_linalg_builtins_match_numpy(body, ref, spd_data):
    got, *_ = _linalg_model(body, spd_data)
    assert np.isclose(got, ref(spd_data), rtol=1e-4), body


def test_slogdet_returns_a_pair_and_must_be_destructured(spd_data):
    """As in JAX. The DSL cannot index a tuple, so destructuring is the way to use it."""
    got, *_ = _linalg_model("(real s, real ld) = slogdet(A); target += s * ld", spd_data)
    sign, logabsdet = np.linalg.slogdet(spd_data["A"])
    assert np.isclose(got, sign * logabsdet, rtol=1e-5)


def test_slogdet_agrees_with_log_det(spd_data):
    got, *_ = _linalg_model("(real s, real ld) = slogdet(A); target += ld - log(det(A))", spd_data)
    assert np.isclose(got, 0.0, atol=1e-4)


def test_solve_triangular_defaults_to_the_lower_triangle(spd_data):
    """Deliberately NOT JAX's default: `cholesky` and both Cholesky types give lower factors.

    Getting this wrong reads the other triangle and returns a wrong answer rather than raising,
    which is why the common case is the one without an argument.
    """
    L = np.linalg.cholesky(spd_data["A"])
    data = {"A": L, "b": spd_data["b"]}
    default, *_ = _linalg_model("target += sum(solve_triangular(A, b))", data)
    explicit_lower, *_ = _linalg_model("target += sum(solve_triangular(A, b, 1))", data)
    upper, *_ = _linalg_model("target += sum(solve_triangular(A, b, 0))", data)
    assert np.isclose(default, np.linalg.solve(np.tril(L), data["b"]).sum(), rtol=1e-4)
    assert np.isclose(default, explicit_lower, rtol=1e-6)
    assert np.isclose(upper, np.linalg.solve(np.triu(L), data["b"]).sum(), rtol=1e-4)
    assert not np.isclose(default, upper), "the flag must actually select a triangle"


def test_a_multivariate_normal_written_through_the_cholesky_factor(spd_data):
    """The reason these exist: the density a covariance parameter is for, without an inverse."""
    body = ("(real s, real ld) = slogdet(A); "
            "target += -0.5 * ld - 0.5 * sum(square(solve_triangular(cholesky(A), b))) "
            "- 1.5 * log(2.0 * 3.141592653589793)")
    got, *_ = _linalg_model(body, spd_data)
    from jax.scipy.stats import multivariate_normal as mvn
    want = float(mvn.logpdf(jnp.asarray(spd_data["b"]), jnp.zeros(3),
                            jnp.asarray(spd_data["A"])))
    assert np.isclose(got, want, rtol=1e-5)


@pytest.mark.parametrize("body", [
    "target += trace(A * S)",
    "target += log(det(S))",
    "target += sum(log(eigvalsh(S)))",
    "target += sum(square(solve_triangular(cholesky(S), b)))",
    "target += sum(solve(S, b))",
])
def test_linalg_builtins_are_differentiable_through_a_parameter(body, spd_data):
    """A builtin is only useful here if a sampler can take gradients of a target using it."""
    _, model, _, charts = _linalg_model(
        body, spd_data, params="parameters { cov_matrix[3] S; }", extra="")
    z = jnp.asarray(np.random.default_rng(1).normal(size=(model.coord_dim,)) * 0.3, float)
    g = np.asarray(jax.grad(model.log_prob_at_coordinate)(z, *charts))
    assert g.shape == (model.coord_dim,) and np.all(np.isfinite(g))
    assert np.any(np.abs(g) > 1e-6), "gradient is identically zero"


def test_a_builtin_name_cannot_be_redefined(spd_data):
    with pytest.raises(DslError, match="reserved|builtin"):
        compile_model("functions { real det(real x) { return x; } }\n"
                      "parameters { real d; }\nmodel { target += -0.5*d*d; }", {})
