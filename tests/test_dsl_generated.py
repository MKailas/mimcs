"""`transformed parameters` saved with the draws, and the `generated quantities` block.

The DSL half of derived quantities (the `Model` / sampler half is `tests/test_generated.py`):

* the outputs are the **top-level** declarations of both blocks, in order, with the declared
  dtype; a loop-local is not one;
* `generated quantities` sees data, parameters and transformed parameters --- and **no** `model`
  block, a deliberate departure from Stan;
* `<name>_rng` is `jax.random.<name>` with the key supplied and JAX's signature otherwise, one
  fresh key per call (so an unrolled `for` draws anew), and it is refused, statically, everywhere
  but `generated quantities`.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

import mimcs
from mimcs import compile_model
from mimcs.dsl import DslError
from mimcs.dsl.random import RNG_FUNCTIONS

HEAD = """
data { int n; array[n] real y; }
parameters { real mu; real<lower=0> sigma; }
"""
MODEL = "model { mu ~ normal(0, 10); sigma ~ exponential(1); y ~ normal(mu, sigma); }\n"
DATA = {"n": 12, "y": np.linspace(-1.0, 3.0, 12)}
POINT = {"mu": jnp.asarray(0.7), "sigma": jnp.asarray(1.3)}


def _gen(body_tp="", body_gq="", data=DATA):
    src = HEAD + (f"transformed parameters {{ {body_tp} }}\n" if body_tp else "") + MODEL + (
        f"generated quantities {{ {body_gq} }}\n" if body_gq else "")
    return compile_model(src, data)


def _eval(model, key=jnp.zeros(2, jnp.uint32)):
    return model.generated.fn(dict(POINT), key if model.generated.rng else None)


# --- what is saved ----------------------------------------------------------- #

def test_outputs_are_the_top_level_declarations_of_both_blocks_in_order():
    m = _gen("real mu2 = mu * mu; array[n] real r = y - mu;",
             "real total = 0; for (i in 1:n) { real t = r[i] ^ 2; total = total + t; } "
             "int big = sigma > 1;")
    assert m.generated_names == ["mu2", "r", "total", "big"]      # the loop-local `t` is not
    assert [m.generated_kind(n) for n in m.generated_names] == [
        "transformed", "transformed", "generated", "generated"]
    out = _eval(m)
    assert float(out["mu2"]) == pytest.approx(0.49)
    assert float(out["total"]) == pytest.approx(float(np.sum((DATA["y"] - 0.7) ** 2)), rel=1e-5)
    assert out["big"].dtype == jnp.int32 and int(out["big"]) == 1
    assert not m.generated.rng


def test_a_program_without_either_block_has_no_derived_quantities():
    assert not _gen().has_generated


def test_generated_quantities_see_transformed_parameters_but_no_model_locals():
    assert float(_eval(_gen("real m2 = 2 * mu;", "real m3 = m2 + mu;"))["m3"]) == pytest.approx(2.1)
    src = HEAD + "model { real loc = mu; y ~ normal(loc, sigma); }\n" \
                 "generated quantities { real g = loc; }"
    with pytest.raises(DslError, match="unknown name 'loc'"):
        mimcs.make_sampler(compile_model(src, DATA))       # traced at sampler construction


def test_the_transformed_values_in_the_draws_match_a_recomputation():
    m = _gen("array[n] real r = y - mu;", "real y_rep = mu + sigma * normal_rng();")
    s = mimcs.make_sampler(m, seed=1)
    s.initialize().warmup(150)
    d = s.sample(200)
    want = (DATA["y"].astype(np.float32)[None, :] - d["mu"][:, None])
    np.testing.assert_allclose(d["r"], want, atol=1e-6)
    assert d["y_rep"].shape == (200,)


# --- static errors -------------------------------------------------------------- #

@pytest.mark.parametrize("src, match", [
    (HEAD + "transformed parameters { real e = normal_rng(); }\n" + MODEL,
     "only available in `generated quantities`, not in `transformed parameters`"),
    (HEAD + "model { real e = normal_rng(); y ~ normal(mu, sigma); }",
     "not in the `target` model component"),
    (HEAD + "transformed data { real e = normal_rng(); }\n" + MODEL,
     "not in `transformed data`"),
    ("functions { real f(real x) { return x + normal_rng(); } }\n" + HEAD + MODEL,
     "not in the function 'f'"),
    (HEAD + MODEL + "generated quantities { mu ~ normal(0, 1); }",
     "adds nothing to the density"),
    (HEAD + MODEL + "generated quantities { real sigma = 1; }",
     "already declared in `parameters`"),
    (HEAD + "transformed parameters { real a = mu; }\n" + MODEL +
     "generated quantities { real a = 2; }", "already declared in `transformed parameters`"),
    ("functions { real normal_rng(real x) { return x; } }\n" + HEAD + MODEL,
     "random-variate function"),
])
def test_static_errors(src, match):
    with pytest.raises(DslError, match=match):
        compile_model(src)


def test_a_stan_style_call_is_an_arity_error_that_names_the_jax_signature():
    m = _gen(body_gq="real e = normal_rng(mu, sigma);")
    with pytest.raises(DslError, match=r"normal_rng\(shape=\(\)\).*standard normal"):
        _eval(m)


# --- random variates -------------------------------------------------------------- #

def test_each_call_takes_the_next_key_by_sequential_split():
    m = _gen(body_gq="array[3] real a = normal_rng(3); real b = uniform_rng(None, 2, 5);")
    key = jax.random.PRNGKey(7)
    out = _eval(m, key)
    k, s1 = jax.random.split(key)
    _, s2 = jax.random.split(k)
    np.testing.assert_array_equal(out["a"], jax.random.normal(s1, (3,)))
    assert float(out["b"]) == float(jax.random.uniform(s2, (), minval=2, maxval=5))
    # control: the same key reproduces exactly, a different one does not
    assert np.array_equal(_eval(m, key)["a"], out["a"])
    assert not np.array_equal(_eval(m, jax.random.PRNGKey(8))["a"], out["a"])


def test_an_unrolled_loop_draws_fresh_variates_each_pass():
    m = _gen(body_gq="array[4] real z; for (i in 1:4) { z[i] = normal_rng(); }")
    z = np.asarray(_eval(m, jax.random.PRNGKey(0))["z"])
    assert len(set(z.tolist())) == 4


def test_parameter_dependent_variates_and_int_dtypes():
    m = _gen(body_gq="int k = poisson_rng(sigma * 10); array[n] real yr = mu + sigma * "
                     "normal_rng(n); real g = gamma_rng(sigma); array[2, 3] real u = "
                     "uniform_rng((2, 3));")
    out = _eval(m, jax.random.PRNGKey(3))
    assert out["k"].dtype == jnp.int32 and out["yr"].shape == (12,)
    assert out["u"].shape == (2, 3) and float(out["g"]) > 0


def test_a_shape_must_be_a_compile_time_constant():
    m = _gen(body_gq="array[2] real z = normal_rng(sigma);")
    with pytest.raises(DslError, match="compile-time int constant"):
        _eval(m)


def test_every_form_names_a_jax_random_sampler_and_runs_on_its_defaults():
    """The table is hand-written, so check it against JAX: each required argument filled with a
    plausible value, every form draws without error."""
    fill = {"a": 2.0, "b": 2.0, "n": 5, "p": 0.3, "lam": 3.0, "df": 4.0, "alpha": jnp.ones(3),
            "logits": jnp.zeros(4), "loc": 0.0, "scale": 1.0, "dfnum": 3.0, "dfden": 4.0,
            "mean": 1.0, "cov": jnp.eye(2), "concentration": 1.5, "left": 0.0, "mode": 0.5,
            "right": 1.0, "lower": -1.0, "upper": 1.0, "minval": 0, "maxval": 5, "d": 3,
            "x": 5, "shape": (2,)}
    from mimcs.dsl.random import call_rng
    for name, form in RNG_FUNCTIONS.items():
        args = [fill[p] for p, _, d in form.params[:form.n_required]]
        if name == "multinomial_rng":
            args = [5.0, jnp.array([0.2, 0.8])]
        elif name == "multivariate_normal_rng":
            args = [jnp.zeros(2), jnp.eye(2)]
        out = call_rng(form, jax.random.PRNGKey(0), args, None)
        assert jnp.all(jnp.isfinite(jnp.asarray(out, float))), name


# --- end to end ------------------------------------------------------------------ #

def test_posterior_predictive_variance_is_noise_plus_parameter_uncertainty():
    """`y_rep = mu + sigma * normal_rng()`: Var(y_rep) = E[sigma^2] + Var(mu) (law of total
    variance). Predicted before running; checked within 4 standard errors of the variance."""
    m = _gen(body_gq="real y_rep = mu + sigma * normal_rng();")
    s = mimcs.make_sampler(m, seed=4)
    s.initialize().warmup(300)
    d = s.sample(2000)
    want = float(np.mean(d["sigma"] ** 2) + np.var(d["mu"]))
    got = float(np.var(d["y_rep"]))
    se = want * np.sqrt(2.0 / len(d["y_rep"])) * 2      # x2 for the chain's autocorrelation
    assert abs(got - want) < 4 * se, (got, want, se)


DISCRETE = """
data { int n; int k; array[n] real y; array[k] real w; }
parameters { ordered[k] mu; array[n] int<lower=1, upper=k> z; real<lower=0> sigma; }
model {
  mu ~ normal(0, 10); sigma ~ lognormal(0, 1);
  for (i in 1:n) { z[i] ~ categorical(w); y[i] ~ normal(mu[z[i]], sigma); }
}
"""


def test_discrete_labels_are_in_scope_and_the_chain_is_unchanged():
    rng = np.random.default_rng(0)
    data = {"n": 10, "k": 2, "y": np.r_[rng.normal(-2, 1, 5), rng.normal(2, 1, 5)],
            "w": np.full(2, 0.5)}
    gq = ("generated quantities { int n1 = 0; for (i in 1:n) { n1 = n1 + (z[i] == 1); } "
          "real noise = normal_rng(); }")
    runs = []
    for src in (DISCRETE, DISCRETE + gq):
        s = mimcs.make_sampler(compile_model(src, data), seed=2)
        s.initialize().warmup(100)
        s.sample(150)
        runs.append(s)
    a, b = runs
    assert np.array_equal(a.get_samples_flat(), b.get_samples_flat())
    assert np.array_equal(a.get_discrete_flat(), b.get_discrete_flat())
    d = b.get_samples()
    np.testing.assert_array_equal(d["n1"], (d["z"] == 1).sum(axis=1))
    assert d["n1"].dtype == np.int32


def test_x64_outputs_are_float64_and_ints_stay_int32():
    from mimcs import config
    was = config.x64_enabled()
    try:
        config.enable_x64(True)
        m = _gen("real mu2 = mu * mu;", "int k = poisson_rng(3.0); real e = normal_rng();")
        fl, it = m.generated_flat(m.pack_sample(POINT), None, jax.random.PRNGKey(0))
        assert fl.dtype == jnp.float64 and it.dtype == jnp.int32
    finally:
        config.enable_x64(was)
