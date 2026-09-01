"""`model <name> scan(a, b) { ... }` --- a component that is a **sum over elements**.

An ordinary component is an opaque scalar: nothing in it says whether it is a sum of
per-observation terms or one irreducible quantity. That distinction is what a single-site discrete
sweep needs, and it cannot be recovered from a JAX closure, so the DSL declares it.

The checks here are about the *construct*: that the sum is the same density as the `for` loop it
replaces, that one element is one element, and that the header's restrictions are enforced. What
the sweep then does with it is `tests/test_discrete_restricted.py`.

One property is deliberately **not** tested, because it is structural rather than behavioural:
that the body cannot reach the whole scanned array. Inside the body the name *is* the element, so
the array is not in scope, and a user function cannot capture it either (a function sees only its
arguments). There is no code path to assert against --- which is the point, since the soundness of
the coordinate-restricted sweep rests on it.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from mimcs import compile_model
from mimcs.dsl import DslError

HEAD = """
data { int n; int k; array[n] real y; array[k] real w; }
parameters { ordered[k] mu; array[n] int<lower=1, upper=k> z; real<lower=0> sigma; }
"""
LOOP = HEAD + """
model {
  mu ~ normal(0, 10); sigma ~ lognormal(0, 1);
  for (i in 1:n) { z[i] ~ categorical(w); y[i] ~ normal(mu[z[i]], sigma); }
}
"""
SCAN = HEAD + """
model prior { mu ~ normal(0, 10); sigma ~ lognormal(0, 1); }
model lik scan(z, y) { z ~ categorical(w); y ~ normal(mu[z], sigma); }
"""


def _data(n=25, k=3, seed=0):
    rng = np.random.default_rng(seed)
    true_mu = 4.0 * (np.arange(k) - (k - 1) / 2.0)
    y = true_mu[rng.integers(0, k, n)] + rng.standard_normal(n)
    return {"n": n, "k": k, "y": y, "w": np.full(k, 1.0 / k)}


def _values(n=25, k=3, seed=1):
    rng = np.random.default_rng(seed)
    return {"mu": jnp.asarray(np.sort(rng.normal(size=k)) * 3),
            "z": jnp.asarray(rng.integers(1, k + 1, n), jnp.int32),
            "sigma": jnp.asarray(float(np.exp(rng.normal() * 0.3)))}


# --------------------------------------------------------------------------- #
# 1. the sum is the same density                                              #
# --------------------------------------------------------------------------- #

def test_a_scan_component_is_the_same_density_as_the_for_loop_it_replaces():
    """The claim the whole construct rests on. Compared at several random points rather than one,
    since a single point can agree by accident on a density this structured."""
    data = _data()
    loop, scan = compile_model(LOOP, data=data), compile_model(SCAN, data=data)
    for seed in range(4):
        v = _values(seed=seed)
        a, b = float(loop.log_prob(v)), float(scan.log_prob(v))
        assert abs(a - b) < 1e-3 * max(1.0, abs(a)), (a, b)


def test_the_gradient_matches_too():
    """A density can agree pointwise and still be a different function of the parameters --- the
    gradient is what HMC actually integrates."""
    data = _data()
    loop, scan = compile_model(LOOP, data=data), compile_model(SCAN, data=data)
    v = _values()
    for name in ("mu", "sigma"):
        ga = jax.grad(lambda x: loop.log_prob({**v, name: x}))(v[name])
        gb = jax.grad(lambda x: scan.log_prob({**v, name: x}))(v[name])
        assert np.allclose(np.asarray(ga), np.asarray(gb), rtol=1e-3, atol=1e-4), name


def test_the_elements_sum_to_the_component():
    data = _data()
    m = compile_model(SCAN, data=data)
    sc = m.scan_components["lik"]
    assert sc.scanned == ("z", "y") and sc.length == data["n"]
    v = _values()
    total = sum(float(sc.element_fn(v, i)) for i in range(sc.length))
    assert abs(total - float(m.log_prob_fns["lik"](v))) < 1e-2


def test_an_override_equals_writing_the_value_into_the_array():
    """`overrides` is what lets the sweep propose without building a modified array per
    coordinate; it must be exactly equivalent to having built one."""
    data = _data()
    m = compile_model(SCAN, data=data)
    sc, v = m.scan_components["lik"], _values()
    for i, new in ((0, 2), (7, 1), (24, 3)):
        via_override = float(sc.element_fn(v, i, {"z": jnp.asarray(new, jnp.int32)}))
        via_array = float(sc.element_fn({**v, "z": v["z"].at[i].set(new)}, i))
        assert via_override == via_array, (i, new)


def test_only_the_named_element_matters_to_its_own_term():
    """The soundness property, stated as a measurement: changing `z[j]` moves element `j` and
    leaves every other element exactly where it was."""
    m = compile_model(SCAN, data=_data())
    sc, v = m.scan_components["lik"], _values()
    j, new = 5, 1 + (int(v["z"][5]) % 3)
    moved = {**v, "z": v["z"].at[j].set(new)}
    before = np.array([float(sc.element_fn(v, i)) for i in range(sc.length)])
    after = np.array([float(sc.element_fn(moved, i)) for i in range(sc.length)])
    assert before[j] != after[j]                       # the one that should move, did
    assert np.array_equal(np.delete(before, j), np.delete(after, j))


def test_the_graph_does_not_grow_with_the_number_of_elements():
    """The unrolled `for` puts one copy of the body in the graph per observation; the scan
    component puts one, at any n. Asserted because it is the compile-time half of the win --- and
    because a `for` that quietly crept back in would still give the right answer."""
    def eqns(src, n):
        m = compile_model(src, data=_data(n=n))
        v = _values(n=n)
        f = lambda mu: m.log_prob({**v, "mu": mu})
        return len(jax.make_jaxpr(jax.grad(f))(v["mu"]).jaxpr.eqns)

    small, large = eqns(SCAN, 20), eqns(SCAN, 200)
    assert small == large, (small, large)
    assert eqns(LOOP, 200) > 10 * large                 # the control: the loop does grow


# --------------------------------------------------------------------------- #
# 2. what the header requires                                                  #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("src, match", [
    ("model scan(y) { }", "needs a name"),
    ("model lik scan() { }", "at least one array"),
    ("model lik scan(y, y) { }", "twice"),
    ("model lik scan(y[1]) { }", "plain array names"),
])
def test_the_header_is_checked(src, match):
    with pytest.raises(DslError, match=match):
        compile_model(src)


def test_a_scanned_name_must_be_declared():
    with pytest.raises(DslError, match="not declared"):
        compile_model("data { int n; } parameters { real mu; }\n"
                      "model lik scan(nope) { target += mu; }")


def test_a_scanned_scalar_is_refused():
    """The leading dimension *is* the element count, so there has to be one."""
    with pytest.raises(DslError, match="scalar"):
        compile_model("data { real c; } parameters { real mu; }\n"
                      "model lik scan(c) { target += mu * c; }", data={"c": 1.0})


def test_scanned_arrays_must_agree_on_their_length():
    """They are stepped through together, so two lengths would make 'element i' mean two things."""
    src = ("data { int n; int m; array[n] real y; array[m] real x; } parameters { real mu; }\n"
           "model lik scan(y, x) { y ~ normal(mu + x, 1.0); }")
    with pytest.raises(DslError, match="different lengths"):
        compile_model(src, data={"n": 4, "m": 5, "y": np.zeros(4), "x": np.zeros(5)})


def test_a_component_may_only_add_to_its_own_density():
    with pytest.raises(DslError, match="may only add to its own density"):
        compile_model("data { int n; array[n] real y; } parameters { real mu; }\n"
                      "model lik scan(y) { other += mu; }")


def test_the_component_name_is_an_accepted_increment():
    """`lik += ...` reads as an increment to *this element's* contribution, and is the spelling the
    construct is documented with; `target +=` means the same thing."""
    data = {"n": 3, "y": np.zeros(3)}
    src = ("data { int n; array[n] real y; } parameters { real mu; }\n"
           "model lik scan(y) { %s += -0.5 * (y - mu) * (y - mu); }")
    a = compile_model(src % "lik", data=data)
    b = compile_model(src % "target", data=data)
    v = {"mu": jnp.asarray(0.5)}
    assert float(a.log_prob(v)) == float(b.log_prob(v))


def test_a_density_statement_in_transformed_parameters_is_refused():
    """`transformed parameters` runs **once**, outside the scan, while the component's own
    statements run per element. Counting one thing once and its neighbour n times is not something
    a reader should have to know, so it is an error."""
    src = ("data { int n; array[n] real y; } parameters { real mu; }\n"
           "transformed parameters { real s = mu * mu; s ~ normal(0, 1); }\n"
           "model lik scan(y) { y ~ normal(s, 1.0); }")
    with pytest.raises(DslError, match="transformed parameters"):
        compile_model(src, data={"n": 3, "y": np.zeros(3)})


def test_transformed_parameters_still_reach_the_body():
    """Rejecting *density* statements there must not stop the deterministic ones from working ---
    they run once, outside the scan, and the body sees their values."""
    src = ("data { int n; array[n] real y; } parameters { real mu; }\n"
           "transformed parameters { real s = mu * mu; }\n"
           "model lik scan(y) { y ~ normal(s, 1.0); }")
    m = compile_model(src, data={"n": 3, "y": np.zeros(3)})
    got = float(m.log_prob({"mu": jnp.asarray(2.0)}))
    want = float(jnp.sum(-0.5 * (jnp.zeros(3) - 4.0) ** 2 - 0.5 * jnp.log(2 * jnp.pi)))
    assert abs(got - want) < 1e-3


# --------------------------------------------------------------------------- #
# 3. what the rest of the model sees                                           #
# --------------------------------------------------------------------------- #

def test_the_cost_rule_sees_the_scanned_arrays():
    """The scanned arrays appear only in the *header*, which `read_names` (a walk over statements)
    cannot see. Left out, a per-observation likelihood over a large data array reads as touching
    nothing large and is labelled **cheap** --- the cost rule wrong in the direction that matters."""
    factory = compile_model(SCAN)
    spec = factory.analyze(_data(n=200))
    lik = spec.component("lik")
    assert "y" in lik.reads and "z" in lik.reads
    assert lik.cost == "expensive", spec.rationale


def test_component_reads_reach_the_model():
    m = compile_model(SCAN, data=_data())
    assert m.component_reads["prior"] == frozenset({"mu", "sigma"})
    assert {"z", "y", "mu", "sigma", "w"} <= m.component_reads["lik"]
    assert "z" not in m.component_reads["prior"]        # what lets the sweep skip it entirely


def test_an_ordinary_component_registers_no_scan_entry():
    m = compile_model(LOOP, data=_data())
    assert m.scan_components == {}
