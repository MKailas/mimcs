"""Conditional values and control flow in the DSL: `where`, `cond`, and the predicate builtins.

The one numerical claim in this arc is the reason `cond` exists at all, and it is pinned here
rather than described in prose:

**`where` evaluates both branches, so a NaN in the branch it does not pick still poisons the
gradient.** `where(x > 0, sqrt(x), 0.0)` at `x = -1` returns the right *value* and a NaN
*derivative*. `cond` returns a finite one. The two halves are asserted in the same test, so if JAX
ever stops producing that NaN the test says so rather than silently losing its subject.

It matters here specifically because this library **vmaps densities** — over draws
(`mimcs/_chunked.py`) and over discrete candidates (`mimcs/samplers/discrete_updates.py`) — and the
NaN survives batching, poisoning entries that were never anywhere near the bad branch.

Two further things are easy to get wrong and have controls:

* the translation of JAX's `cond` complaints must stay **narrow**, exactly as the `scan` carry
  translation does: every error a branch raises passes through `lax.cond`, so a loose match would
  relabel a true error as a confident false one;
* `cond` has **two** function slots, and the code that handled one did not merely give a worse
  message for the second — it never looked at it, so a bad `false_fn` compiled and failed inside
  JAX. That asymmetry is the single most important test below.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from mimcs import compile_model, DslError
from mimcs.dsl.loops import LOOP_FORMS

DATA = {"n": 3, "v": np.array([-1.0, 2.0, -3.0]), "flag": 1}
DECL = "data { int n; array[n] real v; int flag; }\nparameters { real x; }\n"
BRANCHES = ("functions { real on(real e, real b) { return e - b; } "
            "real off(real e, real b) { return e + b; } }\n")


def _fn(src, data=None):
    return compile_model(src, data=data or DATA).log_prob_fns["target"]


def _value_and_grad(src, x=0.5, data=None):
    f = _fn(src, data)
    v = jnp.asarray(x)
    return float(f({"x": v})), float(jax.grad(lambda z: f({"x": z}))(v))


# =========================================================== 1. the NaN pair

SQRT_WHERE = (DECL + "model { target += where(x > 0.0, sqrt(x), 0.0); }")
SQRT_COND = ("functions { real root(real z) { return sqrt(z); } real zero(real z) "
             "{ return 0.0 * z; } }\n" + DECL
             + "model { target += cond(x > 0.0, root, zero, x); }")


def test_where_poisons_the_gradient_where_cond_does_not():
    """The whole reason `cond` is here, asserted from both sides.

    At x = -1 both forms give the right VALUE. Only `where` gives a NaN DERIVATIVE, because JAX
    differentiates `sqrt` at a negative argument and the NaN survives the multiply by zero.

    The `where` half is its own control: if it ever stops being NaN, the motivation for `cond` is
    gone and this test is what says so.
    """
    w_val, w_grad = _value_and_grad(SQRT_WHERE, x=-1.0)
    c_val, c_grad = _value_and_grad(SQRT_COND, x=-1.0)
    assert w_val == 0.0 and c_val == 0.0, (w_val, c_val)      # values agree
    assert np.isnan(w_grad), f"`where` no longer produces the NaN this test exists for: {w_grad}"
    assert np.isfinite(c_grad), f"`cond` gradient should be finite, got {c_grad}"

    # ...and where the branch IS taken, the two agree on the gradient too.
    assert np.isclose(_value_and_grad(SQRT_WHERE, x=4.0)[1],
                      _value_and_grad(SQRT_COND, x=4.0)[1], rtol=1e-5)


def test_the_nan_survives_batching_but_cond_stays_finite():
    """Why it is not a curiosity: this library vmaps densities over draws (`_chunked.py`) and over
    discrete candidates (`discrete_updates.py`), so a predicate on a parameter WILL be batched."""
    xs = jnp.asarray([-1.0, 4.0])
    w, c = _fn(SQRT_WHERE), _fn(SQRT_COND)
    gw = jax.vmap(jax.grad(lambda z: w({"x": z})))(xs)
    gc = jax.vmap(jax.grad(lambda z: c({"x": z})))(xs)
    assert np.isnan(np.asarray(gw)).any(), f"the batched `where` control is vacuous: {gw}"
    assert np.all(np.isfinite(np.asarray(gc))), f"batched `cond` went non-finite: {gc}"


# =================================================== 2. cond: value, gradient, structure

COND_SRC = BRANCHES + DECL + "model { target += cond(flag == 1, on, off, x, 2.0); }"


@pytest.mark.parametrize("flag, want", [(1, -2.0), (0, 2.0)])
def test_cond_runs_the_branch_the_predicate_selects(flag, want):
    v, g = _value_and_grad(COND_SRC, x=0.0, data={**DATA, "flag": flag})
    assert v == want, v
    assert g == 1.0, g                       # d/dx of both branches is +1


def test_cond_traces_to_one_equation_whichever_branch_is_live():
    """Both branches are traced into the graph, so the jaxpr must not depend on which one runs.

    CONTROL: the equation counts for the two flags must be EQUAL. An implementation that traced
    only the selected branch would give different counts, and one that inlined both as an unrolled
    `if` would not contain a `cond` primitive at all.
    """
    counts = []
    for flag in (0, 1):
        f = _fn(COND_SRC, data={**DATA, "flag": flag})
        jaxpr = jax.make_jaxpr(f)({"x": jnp.asarray(0.5)})
        counts.append(len(jaxpr.eqns))
        assert "cond" in str(jaxpr), "no cond primitive: the branch was inlined"
    assert counts[0] == counts[1], f"graph depends on the live branch: {counts}"


def test_cond_returns_a_tuple_and_destructures():
    src = ("functions { (real, real) two(real z) { return (z, -z); } "
           "(real, real) neg(real z) { return (-z, z); } }\n" + DECL
           + "model { (real a, real b) = cond(flag == 1, two, neg, x); target += a - b; }")
    assert _value_and_grad(src, x=1.0)[0] == 2.0


def test_a_function_serves_as_both_a_scan_body_and_a_cond_branch():
    """Arity is checked per call site, so one function can play both roles."""
    src = ("functions { (real, real) st(real c, real e) { return (c + e, c); } }\n"
           "data { int n; array[n] real e; }\nparameters { real x; }\n"
           "model {\n"
           "  (real a, array[n] real ys) = scan(st, 0.0, e);\n"
           "  (real p, real q) = cond(n > 1, st, st, x, 1.0);\n"
           "  target += a + p + q;\n}")
    m = compile_model(src, data={"n": 3, "e": np.array([1.0, 2.0, 3.0])})
    assert np.isfinite(float(m.log_prob_fns["target"]({"x": jnp.asarray(0.5)})))


# =========================================== 3. cond: compile-time errors, both slots

@pytest.mark.parametrize("src, match", [
    (BRANCHES + DECL + "model { target += cond(flag == 1, on); }", "takes at least 3 arguments"),
    # slot 1 and slot 2 alike -- the second is the one the old single-slot code never examined
    (BRANCHES + DECL + "model { target += cond(flag == 1, 1.0, off, x, 2.0); }",
     r"argument 2 of `cond` \(the `true_fn` slot\)"),
    (BRANCHES + DECL + "model { target += cond(flag == 1, on, 1.0, x, 2.0); }",
     r"argument 3 of `cond` \(the `false_fn` slot\)"),
    (BRANCHES + DECL + "model { target += cond(flag == 1, on, nope, x, 2.0); }",
     "not a user-defined function"),
    (BRANCHES + DECL + "model { target += cond(flag == 1, on, exp, x, 2.0); }",
     "builtin cannot be a"),
    (BRANCHES + DECL + "model { target += cond(flag == 1, on, off, x); }", "declares 2"),
    ("functions { real cond(real z) { return z; } }" + DECL + "model { target += x; }",
     "is a loop form"),
])
def test_cond_errors_are_caught_at_compile_time(src, match):
    with pytest.raises(DslError, match=match):
        compile_model(src, data=DATA)


def test_a_bad_false_branch_is_caught_and_not_left_to_jax():
    """The single most important new test. The pre-`cond` code resolved only ONE function slot, so
    a bad `false_fn` compiled cleanly and failed inside a JAX trace.

    CONTROL: the same program with a bad `true_fn`, which was always caught — both must now raise,
    and at the slot's own span.
    """
    for slot in ("on, nope", "nope, off"):
        with pytest.raises(DslError, match="not a user-defined function") as e:
            compile_model(BRANCHES + DECL
                          + f"model {{ target += cond(flag == 1, {slot}, x, 2.0); }}", data=DATA)
        assert "'nope'" in str(e.value)


def test_branches_that_disagree_with_each_other_say_so():
    """The likelier slip is forgetting an operand in one signature, and 'one of these is wrong'
    does not say which or why."""
    src = ("functions { real a(real p, real q) { return p; } real b(real p) { return p; } }\n"
           + DECL + "model { target += cond(flag == 1, a, b, x, 2.0); }")
    with pytest.raises(DslError, match="every function slot must take the same arguments") as e:
        compile_model(src, data=DATA)
    assert "true_fn" in str(e.value) and "false_fn" in str(e.value)


# ============================================ 4. cond: trace-time errors and narrowness

def _trace(src):
    _fn(src)({"x": jnp.asarray(0.5)})


def test_an_array_predicate_points_at_where():
    """The mistake someone arriving from `where` will make: `cond` picks one branch for the whole
    computation, so it cannot select elementwise."""
    src = ("functions { real p(real z) { return z; } real q(real z) { return -z; } }\n" + DECL
           + "model { target += cond(v < 0.0, p, q, x); }")
    with pytest.raises(DslError, match="elementwise choice use"):
        _trace(src)


def test_a_float_predicate_is_refused_rather_than_silently_branching():
    """`lax.cond` accepts a float and branches on `pred != 0`, so in a language with no boolean
    type `cond(x, ...)` — a plausible slip for `cond(x > 0, ...)` — is a wrong answer with nothing
    reported. CONTROL: the same program with the comparison restored must work.
    """
    body = ("functions { real p(real z) { return z; } real q(real z) { return -z; } }\n" + DECL
            + "model { target += cond(%s, p, q, x); }")
    with pytest.raises(DslError, match="must be a condition, not a number"):
        _trace(body % "x")
    _trace(body % "x > 0.0")


@pytest.mark.parametrize("branches, match", [
    # a dtype clash and a pytree clash are DIFFERENT JAX messages; a single substring misses one
    ("real p(real z) { return z; } real q(real z) { return 0; }", "same shape, same dtype"),
    ("(real, real) p(real z) { return (z, z); } real q(real z) { return z; }",
     "same tuple structure"),
])
def test_branch_output_mismatches_are_translated(branches, match):
    src = ("functions { " + branches + " }\n" + DECL
           + "model { target += 0.0 * sum(v); target += cond(flag == 1, p, q, x); }")
    with pytest.raises(DslError, match=match):
        _trace(src)


def test_an_unrelated_branch_error_is_not_relabelled():
    """The narrowness rule, mirroring `test_an_unrelated_body_error_is_not_relabelled_as_a_carry
    _problem`. Every error a branch raises passes through `lax.cond`, so only the message tells a
    branch-type clash apart from an ordinary mistake — and relabelling the latter would replace a
    true error with a confident false one.
    """
    src = ("functions { real p(real z) { return exp(z, z); } real q(real z) { return z; } }\n"
           + DECL + "model { target += cond(flag == 1, p, q, x); }")
    with pytest.raises(TypeError) as e:
        _trace(src)
    msg = str(e.value)
    assert "branches" not in msg and "predicate" not in msg, msg


def test_jax_cond_wording_is_still_what_we_match():
    """A canary. The translations match JAX's own prose, so a JAX upgrade that rewords it should
    fail loudly here rather than silently reverting users to raw JAX text."""
    with pytest.raises(TypeError, match="Pred must be a scalar"):
        jax.lax.cond(jnp.asarray([True, False]), lambda v: v, lambda v: -v, jnp.zeros(2))
    with pytest.raises(TypeError, match="branches must have equal output types"):
        jax.lax.cond(True, lambda v: v, lambda v: jnp.sum(v), jnp.zeros(2))
    with pytest.raises(TypeError, match="branch outputs must have the same pytree structure"):
        jax.lax.cond(True, lambda v: (v, v), lambda v: v, jnp.zeros(2))


# ==================================================== 5. the registry generalization

def test_loop_form_registry_invariants():
    """`check_loop_forms` counts everything past `n_fixed` as a forwarded operand, so a function
    slot among the extras would be miscounted as one."""
    for name, form in LOOP_FORMS.items():
        assert list(form.fn_args) == sorted(set(form.fn_args)), f"{name}: fn_args not sorted/unique"
        assert all(i < form.n_fixed for i in form.fn_args), f"{name}: a function slot past n_fixed"
        assert len(form.slot_names) == len(form.fn_args), f"{name}: slot_names/fn_args mismatch"


@pytest.mark.parametrize("fn_args, position, n_args, expected", [
    ((0,), 1, 3, 2),          # scan: `xs` is source index 2
    ((0,), 1, 4, 2),          # ...unchanged by a forwarded extra
    ((2,), 0, 4, 0),          # a slot after the position shifts nothing
    ((0, 2), 1, 4, 3),        # TWO slots: the shift from the first carries past the second
])
def test_a_non_function_position_maps_to_the_right_source_index(fn_args, position, n_args,
                                                                expected):
    """The `(0, 2)` row is the one that matters and no shipped form exercises it.

    CONTROL: the obvious arithmetic shortcut — "add one per function slot at or below" — gives 2
    there, which is itself a function slot. That is why the implementation builds the surviving
    index list instead.
    """
    from mimcs.dsl.loops import LoopForm
    form = LoopForm(name="t", fn_args=fn_args, n_fixed=99, body_arity=0, static_args=(),
                    impl=None, signature="t")
    assert form.source_index(position, n_args) == expected
    naive = position + sum(1 for f in fn_args if f <= position)
    if fn_args == (0, 2):
        assert naive != expected, "the control is vacuous: the naive formula agrees here"
