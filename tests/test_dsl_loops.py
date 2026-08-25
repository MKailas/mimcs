"""Tests for the DSL's non-unrolling loops --- ``scan`` and ``fori_loop`` --- and the tuples
that let ``scan`` keep JAX's signature.

The load-bearing test is not that these run, but that they compute *exactly* what the unrolled
``for`` computes while producing a jaxpr whose size does not grow with the loop. Both halves
matter: agreement alone would be satisfied by an unrolled implementation, and a small jaxpr
alone would be satisfied by one that computes the wrong thing. So each recursion is written
twice --- once with a loop form, once with `for` --- and the two are compared on value, on
gradient, and on jaxpr size at two different lengths.

Seeds are fixed and every model here is deterministic, so pass/fail is reproducible.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from mimcs import compile_model
from mimcs.dsl import DslError
from mimcs.model import Model, EuclideanParameter, PositiveParameter
from mimcs.testing import evaluate, nuts


# --- programs written twice: with a loop form, and unrolled --------------------- #

#: An AR(1)-style recursion whose coefficient is a *parameter*, so the loop cannot be constant
#: folded away and the jaxpr comparison means something.
SCAN_AR = """
functions {
  (real, real) ar_step(real c, real x, real phi) {
    real nxt = phi * c + x;
    return (nxt, nxt);
  }
}
data { int n; array[n] real e; }
parameters { real phi; }
model {
  (real last, array[n] real path) = scan(ar_step, 0.0, e, phi);
  target += last + sum(path .* path);
}
"""

FOR_AR = """
data { int n; array[n] real e; }
parameters { real phi; }
model {
  real c = 0.0;
  real s = 0.0;
  for (i in 1:n) { c = phi * c + e[i]; s = s + c * c; }
  target += c + s;
}
"""

FORI_AR = """
functions { real acc(int i, real a, array real e, real phi) { return phi * a + e[i]; } }
data { int n; array[n] real e; }
parameters { real phi; }
model { target += fori_loop(1, n, acc, 0.0, e, phi); }
"""

FOR_FORI_AR = """
data { int n; array[n] real e; }
parameters { real phi; }
model {
  real a = 0.0;
  for (i in 1:n) { a = phi * a + e[i]; }
  target += a;
}
"""

FUNCS = "functions { (real, real) st(real c, real x) { return (c + x, c); } }\n"
DECLS = "data { int n; array[n] real e; }\nparameters { real phi; }\n"


def _data(n):
    return {"n": n, "e": np.linspace(-1.0, 1.0, n)}


def _probe(src, n, phi=0.7):
    """``(log_prob, d/dphi, jaxpr equation count)`` for one program at one loop length."""
    fn = compile_model(src, _data(n)).log_prob_fns["target"]
    value = jnp.asarray(phi, float)
    return (float(fn({"phi": value})),
            float(jax.grad(lambda p: fn({"phi": p}))(value)),
            len(jax.make_jaxpr(fn)({"phi": value}).eqns))


# --- tuples ------------------------------------------------------------------- #

def test_tuple_literal_and_destructuring():
    model = compile_model(
        "parameters { real m; }\n"
        "model { (real a, real b) = (m + 1.0, m * 2.0); target += a + b; }", {})
    fn = model.log_prob_fns["target"]
    assert np.isclose(float(fn({"m": jnp.asarray(3.0)})), 4.0 + 6.0, atol=1e-5)


def test_a_function_can_return_a_tuple():
    model = compile_model(
        "functions { (real, real) split(real x) { return (x - 1.0, x + 1.0); } }\n"
        "parameters { real m; }\n"
        "model { (real lo, real hi) = split(m); target += hi - lo; }", {})
    assert np.isclose(float(model.log_prob_fns["target"]({"m": jnp.asarray(5.0)})), 2.0, atol=1e-5)


def test_nested_tuples_destructure():
    """Nesting has to work, because a ``scan`` with a tuple carry returns ``((a, b), ys)``."""
    model = compile_model(
        "parameters { real m; }\n"
        "model {\n"
        "  ((real a, real b), real c) = ((m, m + 1.0), m + 2.0);\n"
        "  target += a + b + c;\n"
        "}", {})
    assert np.isclose(float(model.log_prob_fns["target"]({"m": jnp.asarray(1.0)})), 6.0, atol=1e-5)


def test_a_tuple_typed_local_is_not_supported():
    """Destructuring is the only way to bind a tuple, so ``(`` in a target list always nests.

    That is what removes the ambiguity between ``(real a, real b) = ...`` (destructuring) and
    ``(real, real) t = ...`` (a tuple-typed local): only the first exists. A tuple type is still
    legal where it is unambiguous --- a function's argument or return type.
    """
    with pytest.raises(DslError):
        compile_model(
            "parameters { real m; }\n"
            "model { (real, real) t = (m, m); target += m; }", {})


def test_a_parenthesised_expression_is_not_a_tuple():
    """``(x)`` stays grouping --- there is no 1-tuple, so precedence is unaffected."""
    model = compile_model(
        "parameters { real m; }\nmodel { target += (m + 1.0) * 2.0; }", {})
    assert np.isclose(float(model.log_prob_fns["target"]({"m": jnp.asarray(3.0)})), 8.0, atol=1e-5)


@pytest.mark.parametrize("src, match", [
    ("model { (real a, real b, real c) = (m, m); target += a; }", "2-tuple into 3"),
    ("model { (real a, real b) = m; target += a; }", "not a tuple"),
])
def test_destructuring_errors(src, match):
    model = compile_model("parameters { real m; }\n" + src, {})
    with pytest.raises(DslError, match=match):
        model.log_prob_fns["target"]({"m": jnp.asarray(1.0)})


def test_a_destructured_name_is_visible_inside_a_function():
    """`_bound_names` must count destructuring targets, or the scope check rejects the body."""
    model = compile_model(
        "functions {\n"
        "  real f(real x) { (real a, real b) = (x, x + 1.0); return a + b; }\n"
        "}\n"
        "parameters { real m; }\nmodel { target += f(m); }", {})
    assert np.isclose(float(model.log_prob_fns["target"]({"m": jnp.asarray(2.0)})), 5.0, atol=1e-5)


# --- scan and fori_loop equal the unrolled loop -------------------------------- #

@pytest.mark.parametrize("n", [3, 25])
def test_scan_matches_the_unrolled_for(n):
    """Value *and* gradient, at two lengths --- the whole point is that only the graph differs."""
    scanned, unrolled = _probe(SCAN_AR, n), _probe(FOR_AR, n)
    assert np.isclose(scanned[0], unrolled[0], rtol=1e-5), f"{scanned[0]} vs {unrolled[0]}"
    assert np.isclose(scanned[1], unrolled[1], rtol=1e-4), f"{scanned[1]} vs {unrolled[1]}"


@pytest.mark.parametrize("n", [3, 25])
def test_fori_loop_matches_the_unrolled_for(n):
    looped, unrolled = _probe(FORI_AR, n), _probe(FOR_FORI_AR, n)
    assert np.isclose(looped[0], unrolled[0], rtol=1e-5)
    assert np.isclose(looped[1], unrolled[1], rtol=1e-4)


def test_the_loop_forms_do_not_unroll():
    """The reason they exist: the graph must not grow with the loop.

    Checked structurally rather than by timing --- the jaxpr equation count is exactly the thing
    that drives compile time and memory, and it is deterministic.
    """
    small, large = _probe(SCAN_AR, 3), _probe(SCAN_AR, 60)
    assert small[2] == large[2], f"scan grew: {small[2]} -> {large[2]} equations"

    small_f, large_f = _probe(FORI_AR, 3), _probe(FORI_AR, 60)
    assert small_f[2] == large_f[2], f"fori_loop grew: {small_f[2]} -> {large_f[2]}"

    # ... while the unrolled loop it replaces does grow, several times over.
    small_u, large_u = _probe(FOR_AR, 3), _probe(FOR_AR, 60)
    assert large_u[2] > 5 * small_u[2], f"the `for` baseline did not grow: {small_u[2]} -> {large_u[2]}"
    assert large[2] < large_u[2] / 10, f"scan {large[2]} is not much smaller than for {large_u[2]}"


def test_reverse_mode_gradient_matches_finite_differences():
    """Static lengths exist so that the loop is reverse-mode differentiable; check that it is."""
    fn = compile_model(SCAN_AR, _data(12)).log_prob_fns["target"]
    phi0 = 0.6
    grad = float(jax.grad(lambda p: fn({"phi": p}))(jnp.asarray(phi0, float)))
    eps = 1e-3                                   # the float32 sweet spot used across the suite
    fd = (float(fn({"phi": jnp.asarray(phi0 + eps, float)}))
          - float(fn({"phi": jnp.asarray(phi0 - eps, float)}))) / (2 * eps)
    assert np.isclose(grad, fd, rtol=2e-3), f"grad {grad} != fd {fd}"


# --- the pieces that tuples buy ------------------------------------------------ #

def test_scan_carries_a_tuple():
    """``init`` is a pytree, so several values can be carried at once."""
    src = """
    functions {
      ((real, real), real) two(( real, real ) c, real x) {
        (real a, real b) = c;
        return ((a + x, b * 2.0), a + b);
      }
    }
    data { int n; array[n] real e; }
    parameters { real m; }
    model {
      ((real a, real b), array[n] real ys) = scan(two, (0.0, 1.0), e);
      target += a + b + sum(ys) + m;
    }
    """
    fn = compile_model(src, _data(4)).log_prob_fns["target"]
    got = float(fn({"m": jnp.asarray(0.0)}))
    e = _data(4)["e"]
    a, b, total = 0.0, 1.0, 0.0
    for x in e:
        total += a + b
        a, b = a + x, b * 2.0
    assert np.isclose(got, a + b + total, rtol=1e-5), f"{got} vs {a + b + total}"


def test_scan_over_a_tuple_of_arrays():
    """``xs`` is a pytree too, so several arrays can be scanned in step."""
    src = """
    functions {
      (real, real) both((real, real) c, (real, real) xy) {
        (real u, real v) = xy;
        (real s, real ignored) = c;
        return ((s + u * v, 0.0), s + u * v);
      }
    }
    data { int n; array[n] real e; array[n] real f; }
    parameters { real m; }
    model {
      ((real s, real z), array[n] real ys) = scan(both, (0.0, 0.0), (e, f));
      target += s + m;
    }
    """
    data = _data(5)
    data["f"] = np.linspace(2.0, 3.0, 5)
    fn = compile_model(src, data).log_prob_fns["target"]
    expected = float(np.sum(data["e"] * data["f"]))
    assert np.isclose(float(fn({"m": jnp.asarray(0.0)})), expected, rtol=1e-5)


def test_extra_arguments_are_forwarded_to_every_call():
    """The DSL's stand-in for a closure: a body sees only its arguments, so data is passed in."""
    src = """
    functions { real step(int i, real a, array real e, real k) { return a + k * e[i]; } }
    data { int n; array[n] real e; }
    parameters { real m; }
    model { target += fori_loop(1, n, step, 0.0, e, 3.0) + m; }
    """
    fn = compile_model(src, _data(4)).log_prob_fns["target"]
    expected = 3.0 * float(np.sum(_data(4)["e"]))
    assert np.isclose(float(fn({"m": jnp.asarray(0.0)})), expected, atol=1e-5)


# --- fori_loop's bounds -------------------------------------------------------- #

def test_fori_loop_bounds_are_inclusive():
    """A deliberate departure from ``jax.lax.fori_loop``'s half-open range --- pinned here.

    ``fori_loop(1, n, ...)`` runs ``n`` times, matching this language's ``for (i in 1:n)`` and
    its 1-based indexing. A later "fix" toward JAX's ``[lower, upper)`` would change every
    result by one iteration, so it must not be able to pass silently.
    """
    src = """
    functions { real count(int i, real a) { return a + 1.0; } }
    parameters { real m; }
    model { target += fori_loop(1, 5, count, 0.0) + m; }
    """
    fn = compile_model(src, {}).log_prob_fns["target"]
    assert float(fn({"m": jnp.asarray(0.0)})) == 5.0


def test_an_empty_fori_range_returns_the_initial_value():
    src = """
    functions { real count(int i, real a) { return a + 1.0; } }
    parameters { real m; }
    model { target += fori_loop(3, 1, count, 7.0) + m; }
    """
    fn = compile_model(src, {}).log_prob_fns["target"]
    assert float(fn({"m": jnp.asarray(0.0)})) == 7.0


def test_the_loop_index_indexes_one_based():
    """``i`` runs 1..n and `a[i]` is 1-based, so the two line up without an adjustment."""
    src = """
    functions { real first(int i, real a, array real e) { return a + e[i]; } }
    data { int n; array[n] real e; }
    parameters { real m; }
    model { target += fori_loop(1, 1, first, 0.0, e) + m; }
    """
    data = _data(4)
    fn = compile_model(src, data).log_prob_fns["target"]
    assert np.isclose(float(fn({"m": jnp.asarray(0.0)})), data["e"][0], atol=1e-6)


# --- errors -------------------------------------------------------------------- #

@pytest.mark.parametrize("src, match", [
    (FUNCS + DECLS + "model { (real a, array[n] real b) = scan(st, 0.0); target += a; }",
     "takes at least 3 arguments"),
    (FUNCS + DECLS + "model { (real a, array[n] real b) = scan(st, 0.0, e, phi); target += a; }",
     "declares 2"),
    (DECLS + "model { (real a, array[n] real b) = scan(exp, 0.0, e); target += a; }",
     "builtin cannot be a loop body"),
    (FUNCS + DECLS + "model { (real a, array[n] real b) = scan(1.0, 0.0, e); target += a; }",
     "must name a function"),
    (DECLS + "model { (real a, array[n] real b) = scan(nope, 0.0, e); target += a; }",
     "not a user-defined function"),
    ("functions { real scan(real x) { return x; } }" + DECLS + "model { target += phi; }",
     "is a loop form"),
])
def test_loop_form_errors_are_caught_at_compile_time(src, match):
    with pytest.raises(DslError, match=match):
        compile_model(src, _data(4))


def test_a_loop_body_may_not_touch_the_density():
    """A body is an ordinary function, so the purity rule still applies inside a loop."""
    src = ("functions { real b(real c, real x) { target += x; return c; } }\n"
           + DECLS + "model { target += phi; }")
    with pytest.raises(DslError, match="functions are pure"):
        compile_model(src, _data(4))


@pytest.mark.parametrize("bound, match", [
    ("phi", "compile-time constant integer"),
    ("2.7", "whole number"),
])
def test_a_loop_bound_must_be_a_constant_whole_number(bound, match):
    """Constant keeps it differentiable; whole stops ``int()`` silently truncating it."""
    src = ("functions { real g(int i, real a, array real e) { return a + e[i]; } }\n"
           + DECLS + f"model {{ target += fori_loop(1, {bound}, g, 0.0, e); }}")
    fn = compile_model(src, _data(4)).log_prob_fns["target"]
    with pytest.raises(DslError, match=match):
        jax.jit(fn)({"phi": jnp.asarray(3.0)})


def test_a_scan_body_must_return_a_pair():
    src = ("functions { real bad(real c, real x) { return c + x; } }\n"
           + DECLS + "model { (real a, array[n] real b) = scan(bad, 0.0, e); target += a; }")
    fn = compile_model(src, _data(4)).log_prob_fns["target"]
    with pytest.raises(DslError, match="must return a `\\(carry, y\\)` pair"):
        fn({"phi": jnp.asarray(0.5)})


def test_a_carry_that_changes_shape_is_reported_as_such():
    """The translated message, not JAX's pytree diff --- and only for a genuine carry mismatch."""
    src = ("functions { (real, real) bad(real c, real x, array real e) { return (e, c); } }\n"
           + DECLS
           + "model { (real a, array[n] real b) = scan(bad, 0.0, e, e); target += a; }")
    fn = compile_model(src, _data(4)).log_prob_fns["target"]
    with pytest.raises(DslError, match="carry changes type or shape"):
        fn({"phi": jnp.asarray(0.5)})


def test_an_unrelated_body_error_is_not_relabelled_as_a_carry_problem():
    """The translation must stay narrow: everything a body does passes through ``lax.scan``.

    Here the body calls a builtin with the wrong number of arguments, which JAX raises as a
    ``TypeError`` --- the same exception class a carry mismatch uses, so only the message tells
    them apart. Reporting this as "your carry changed type" would replace a true error with a
    confident false one, so the original must survive untouched.

    (An out-of-range index would *not* work as the probe here: JAX clamps out-of-bounds gathers
    rather than raising, so there would be no error to preserve.)
    """
    src = ("functions { real g(int i, real a, array real e) { return a + dot(e); } }\n"
           + DECLS + "model { target += fori_loop(1, n, g, 0.0, e); }")
    fn = compile_model(src, _data(4)).log_prob_fns["target"]
    with pytest.raises(TypeError) as excinfo:
        jax.jit(fn)({"phi": jnp.asarray(0.5)})
    assert "carry" not in str(excinfo.value).lower(), str(excinfo.value)


# --- end to end ---------------------------------------------------------------- #

def test_a_scan_model_samples_like_its_hand_built_twin(artifacts_dir):
    """An AR(1) posterior written with `scan`, against the same model built directly in JAX."""
    rng = np.random.default_rng(0)
    n = 60
    phi_true, sigma = 0.6, 0.5
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = phi_true * y[i - 1] + sigma * rng.standard_normal()

    src = """
    functions { (real, real) pred(real prev, real yi, real phi) { return (yi, phi * prev); } }
    data { int n; array[n] real y; }
    parameters { real phi; real<lower=0> sigma; }
    model {
      (real last, array[n] real mu) = scan(pred, 0.0, y, phi);
      target += -0.5 * sum(((y .- mu) ./ sigma) .* ((y .- mu) ./ sigma)) - n * log(sigma);
      target += -0.5 * phi * phi - sigma;
    }
    """
    dsl_model = compile_model(src, {"n": n, "y": y})

    jy = jnp.asarray(y, float)

    def log_post(p):
        mu = jnp.concatenate([jnp.zeros((1,), float), p["phi"] * jy[:-1]])
        z = (jy - mu) / p["sigma"]
        return (-0.5 * jnp.sum(z ** 2) - n * jnp.log(p["sigma"])
                - 0.5 * p["phi"] ** 2 - p["sigma"])

    hand = Model([EuclideanParameter("phi"), PositiveParameter("sigma")], {"log_post": log_post})

    # Same density, evaluated the same way: the coordinate targets must agree pointwise.
    q = jnp.asarray([0.4, -0.3], float)
    h, c = dsl_model.init_chart_hyperparams(), dsl_model.init_chart_indices()
    assert np.isclose(float(dsl_model.log_prob_at_coordinate(q, h, c)),
                      float(hand.log_prob_at_coordinate(q, h, c)), rtol=1e-4)

    # ... and the posterior it induces is sampled correctly.
    from mimcs.testing import TargetProblem
    problem = TargetProblem(name="ar1_scan", model=dsl_model, dim=2, labels=["phi", "sigma"])
    report = evaluate(problem, {"nuts": nuts()}, n_warmup=1000, n_samples=4000, seed=0,
                      out_dir=artifacts_dir / "ar1_scan")
    draws = report.outputs["nuts"].samples
    assert np.all(draws[:, 1] > 0.0)
    assert abs(draws[:, 0].mean() - phi_true) < 0.2, draws[:, 0].mean()


# --- `None`: the empty carry and the empty input ------------------------------- #

#: Asymmetric on purpose --- a symmetric array sums to zero, which would let a wrong answer pass.
E_ASYM = np.array([0.5, -0.25, 2.0, 1.0])

MAP_NONE = """
functions {
  (None, real) scale(None c, real x, real phi) { return (None, phi * x); }
}
data { int n; array[n] real e; }
parameters { real phi; }
model {
  (None ignored, array[n] real ys) = scan(scale, None, e, phi);
  target += sum(ys);
}
"""

#: The same thing with the name dropped from both the parameter and the target.
MAP_NAMELESS = (MAP_NONE.replace("(None c, real x, real phi)", "(None, real x, real phi)")
                        .replace("(None ignored, array[n] real ys)", "(None, array[n] real ys)"))

#: A carry that is a real value, doing the same work --- the reference for the two above.
MAP_REAL_CARRY = """
functions {
  (real, real) scale(real c, real x, real phi) { return (c, phi * x); }
}
data { int n; array[n] real e; }
parameters { real phi; }
model {
  (real ignored, array[n] real ys) = scan(scale, 0.0, e, phi);
  target += sum(ys);
}
"""


def _run(src, data, phi=0.7):
    fn = compile_model(src, data).log_prob_fns["target"]
    value = jnp.asarray(phi, float)
    return (float(fn({"phi": value})),
            float(jax.grad(lambda p: fn({"phi": p}))(value)),
            len(jax.make_jaxpr(fn)({"phi": value}).eqns))


@pytest.mark.parametrize("src", [MAP_NONE, MAP_NAMELESS], ids=["named", "nameless"])
def test_an_empty_carry_turns_scan_into_a_map(src):
    """``None`` is an empty pytree, so it threads through untouched and the scan becomes a map."""
    data = {"n": len(E_ASYM), "e": E_ASYM}
    got, grad, _ = _run(src, data)
    assert np.isclose(got, 0.7 * E_ASYM.sum(), rtol=1e-5)
    assert np.isclose(grad, E_ASYM.sum(), rtol=1e-4)
    # ... and it agrees with the same computation carrying a real value.
    assert np.isclose(got, _run(MAP_REAL_CARRY, data)[0], rtol=1e-5)


def test_an_empty_carry_still_does_not_unroll():
    """The empty carry changes the pytree `lax.scan` threads, so re-check the graph stays flat."""
    small = _run(MAP_NONE, {"n": 4, "e": np.linspace(0.1, 1.0, 4)})
    large = _run(MAP_NONE, {"n": 60, "e": np.linspace(0.1, 1.0, 60)})
    assert small[2] == large[2], f"scan grew: {small[2]} -> {large[2]} equations"


NO_INPUTS = """
functions { (real, real) tick(real c, None, real phi) { return (c + phi, c + phi); } }
parameters { real phi; }
model {
  (real last, array[6] real path) = scan(tick, 0.0, None, 6, phi);
  target += last + sum(path);
}
"""


def test_scan_with_no_inputs_runs_the_given_length():
    """`xs = None` has no array to take a length from, so the length is written next."""
    fn = compile_model(NO_INPUTS, {}).log_prob_fns["target"]
    got = float(fn({"phi": jnp.asarray(0.7, float)}))
    traj = np.cumsum(np.full(6, 0.7))
    assert np.isclose(got, traj[-1] + traj.sum(), rtol=1e-5)


def test_scan_with_no_inputs_agrees_with_fori_loop():
    """The same recursion, one collecting its trajectory and one keeping only the final value."""
    src = ("functions { real tick(int i, real c, real phi) { return c + phi; } }\n"
           "parameters { real phi; }\nmodel { target += fori_loop(1, 6, tick, 0.0, phi); }")
    final = float(compile_model(src, {}).log_prob_fns["target"]({"phi": jnp.asarray(0.7, float)}))
    assert np.isclose(final, np.cumsum(np.full(6, 0.7))[-1], rtol=1e-5)


def test_a_none_typed_local_binds_the_empty_value():
    model = compile_model(
        "functions { (None, real) f(None c, real x) { return (c, x); } }\n"
        "data { int n; array[n] real e; }\nparameters { real m; }\n"
        "model {\n"
        "  None nothing;\n"
        "  (None, array[n] real ys) = scan(f, nothing, e);\n"
        "  target += sum(ys) + m;\n"
        "}", {"n": 4, "e": E_ASYM})
    got = float(model.log_prob_fns["target"]({"m": jnp.asarray(0.0)}))
    assert np.isclose(got, E_ASYM.sum(), rtol=1e-5)


@pytest.mark.parametrize("src, match", [
    # no length at all after the `None`
    ("functions { (real, real) st(real c, None) { return (c + 1.0, c); } }\n"
     "parameters { real phi; }\n"
     "model { (real a, array[3] real b) = scan(st, 0.0, None); target += a + phi; }",
     "needs a length"),
    # a length that is really an extra: the arity error must point at the rule
    ("functions { (real, real) st(real c, None, real k) { return (c + k, c); } }\n"
     "parameters { real phi; }\n"
     "model { (real a, array[3] real b) = scan(st, 0.0, None, phi); target += a; }",
     "the next argument is the loop length"),
    # `None` is reserved
    ("functions { real None(real x) { return x; } }\nparameters { real m; }\n"
     "model { target += m; }", "reserved"),
    # ... and is not something you can sample
    ("parameters { None x; }\nmodel { target += 0; }", "not a parameter type"),
])
def test_none_errors_are_caught_at_compile_time(src, match):
    with pytest.raises(DslError, match=match):
        compile_model(src, {"n": 4, "e": E_ASYM})


@pytest.mark.parametrize("length, match", [
    ("phi", "compile-time constant integer"),
    ("6.5", "whole number"),
])
def test_a_scan_length_must_be_a_constant_whole_number(length, match):
    """The length is what keeps the loop static, so it gets the same treatment as a bound."""
    src = ("functions { (real, real) tick(real c, None, real phi) { return (c + phi, c); } }\n"
           "parameters { real phi; }\n"
           f"model {{ (real a, array[6] real p) = scan(tick, 0.0, None, {length}, phi);"
           " target += a; }")
    fn = compile_model(src, {}).log_prob_fns["target"]
    with pytest.raises(DslError, match=match):
        jax.jit(fn)({"phi": jnp.asarray(3.0)})
