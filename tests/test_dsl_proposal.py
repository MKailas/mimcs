"""The DSL's ``proposal`` block: custom jump operators written in source.

The runtime is tested in ``tests/test_jump_operators.py``; this file is about the surface. Three
things it has to get right, each with a control:

* **The index is shaped, not flat.** The sampler works in a flat 0-based offset within the
  parameter's own block; the DSL works in the declared shape with Stan's 1-based indexing, because
  that is what ``gamma[j, k]`` in a body has to mean. The conversion lives in exactly one place,
  and the row-major agreement between ``jnp.unravel_index`` and ``Model.unpack_discrete``'s reshape
  is an assumption rather than a guarantee --- so it is asserted against ``np.unravel_index`` over
  the whole flat range, not spot-checked.
* **``at`` / ``to`` / ``scales`` are contextual, not reserved.** They are matched by text in fixed
  header positions, so a model already using them as ordinary names keeps working. That is
  asserted, because adding them to ``KEYWORDS`` would have been the easy way and would break
  programs silently at the next release.
* **A body is pure.** It has a ``functions`` body's shape and a ``model`` body's scope; the purity
  is what keeps the acceptance ratio's Hastings term the ordinary one, so it is a correctness
  property rather than a style rule.
"""

import numpy as np
import pytest
import jax.numpy as jnp

import mimcs
from mimcs import DslError, compile_model
from mimcs.adaptation import RobbinsMonroStepSize
from mimcs.hmc import NUTS
from mimcs.samplers import DiscreteMetropolisWithinGibbs, make_sampler_class

NUTS_GIBBS = make_sampler_class(RobbinsMonroStepSize, DiscreteMetropolisWithinGibbs, NUTS)

HEAD = """
data { int n; int m; }
parameters {
  array[2] real eta;
  array[n, m] int<lower=1, upper=4> g2;
  int<lower=0, upper=1> inc;
}
model lik { target += sum(eta); }
"""
DATA = {"n": 3, "m": 4}


def _compile(proposal, head=HEAD, data=None):
    return compile_model(head + proposal, data=DATA if data is None else data)


# ------------------------------------------------------------------ 1. parsing

def test_the_three_header_forms_parse():
    from mimcs.dsl.parser import parse
    prog = parse("""
    proposal {
      gamma at j to g -> (eta) { return eta; }
      w at (j, k) to v scales -> (a, b) { return (a, b); }
      inc to g -> (tau) { return tau; }
    }
    parameters { real y; } model { target += y; }
    """)
    defs = prog.blocks[0].body
    assert [d.parameter for d in defs] == ["gamma", "w", "inc"]
    assert [d.index_names for d in defs] == [("j",), ("j", "k"), ()]
    assert [d.volume_preserving for d in defs] == [True, False, True]
    assert [d.outputs for d in defs] == [("eta",), ("a", "b"), ("tau",)]


def test_at_and_to_and_scales_are_not_reserved():
    """They are contextual keywords. Reserving them would break existing programs silently, and
    the header shape makes reserving unnecessary."""
    m = compile_model("""
    functions { real to(real at) { return at * 2.0; } }
    parameters { real scales; }
    model { target += to(scales); }
    """, data={})
    assert np.isclose(float(m.log_prob_fns["target"]({"scales": jnp.asarray(1.5)})), 3.0)


def test_an_arrow_does_not_disturb_existing_operators():
    """`->` is new, but `-` and `>` are not. A `-` immediately followed by `>` parses in no
    existing program, so nothing that compiled before can change meaning."""
    from mimcs.dsl.lexer import tokenize
    kinds = [t.kind.name for t in tokenize("a -> b; c - d; e >= f; g >-1;")]
    assert "ARROW" in kinds
    assert kinds[4:7] == ["IDENT", "MINUS", "IDENT"]       # `c - d` untouched
    assert "GE" in kinds                                    # `>=` untouched


# --------------------------------------------------- 2. the shaped index

def test_the_index_is_shaped_and_one_based():
    """Asserted over the WHOLE flat range against `np.unravel_index`, because the row-major
    agreement with `Model.unpack_discrete`'s reshape is the correctness assumption here.

    The body encodes its binders into a number, so each flat offset's (j, k) is readable back out.
    """
    m = _compile("""
    proposal {
      g2 at (j, k) to g -> (eta) {
        array[2] real e2 = eta;
        e2[1] = 100.0 * j + 10.0 * k + 1.0 * g;
        return e2;
      }
    }
    """)
    p = m.discrete_parameters[0]
    assert p.ambient_shape == (3, 4)
    op = m.jump_operators["g2"]
    vals = {"eta": jnp.zeros(2), "g2": jnp.ones((3, 4), jnp.int32),
            "inc": jnp.int32(0), "n": 3, "m": 4}
    for c in range(p.size):
        j, k = np.unravel_index(c, p.ambient_shape)        # row major, as the model packs
        got = float(op.fn(vals, c, jnp.int32(2))[0][0])
        assert got == 100 * (j + 1) + 10 * (k + 1) + 2, f"offset {c} bound ({j+1}, {k+1}) wrongly"


def test_the_body_reads_the_current_value_at_the_bound_index():
    """`gamma[j]` must be the label the sweep is about to move, which is what makes the difference
    form --- and so the balance conditions --- expressible at all."""
    m = _compile("""
    proposal {
      g2 at (j, k) to g -> (eta) {
        array[2] real e2 = eta;
        e2[1] = 1.0 * g2[j, k];
        return e2;
      }
    }
    """)
    z = (jnp.arange(12, dtype=jnp.int32).reshape(3, 4) % 4) + 1
    op = m.jump_operators["g2"]
    vals = {"eta": jnp.zeros(2), "g2": z, "inc": jnp.int32(0), "n": 3, "m": 4}
    for c in range(12):
        got = float(op.fn(vals, c, jnp.int32(1))[0][0])
        assert got == float(z.reshape(-1)[c]), f"offset {c} read the wrong element"


@pytest.mark.parametrize("proposal, match", [
    ("proposal { g2 at j to v -> (eta) { return eta; } }", "binds 1 index name"),
    ("proposal { g2 to v -> (eta) { return eta; } }", "binds 0 index name"),
    ("proposal { inc at j to v -> (eta) { return eta; } }", "binds 1 index name"),
])
def test_the_binder_count_must_match_the_parameter_s_rank(proposal, match):
    with pytest.raises(DslError, match=match):
        _compile(proposal)


# ------------------------------------------------------------- 3. the static checks

@pytest.mark.parametrize("proposal, match", [
    # what the block itself can settle
    ("proposal { eta at j to v -> (eta) { return eta; } }", "not a discrete"),
    ("proposal { g2 at (j,k) to v -> (nope) { return nope; } }", "not a declared parameter"),
    ("proposal { g2 at (j,k) to v -> (g2) { return g2; } }", "lists 'g2' among"),
    ("proposal { g2 at (j,k) to v -> (eta, eta) { return (eta, eta); } }", "more than once"),
    ("proposal { g2 at (j,k) to v -> (eta) { eta ~ normal(0,1); return eta; } }", "not allowed"),
    ("proposal { g2 at (j,k) to v -> (eta) { target += 1.0; return eta; } }", "not allowed"),
    ("proposal { g2 at (j,k) to v -> (eta) { real q = 1.0; } }", "never returns"),
    ("proposal { g2 at (j,k) to n -> (eta) { return eta; } }", "already a declared name"),
    ("proposal { g2 at (j,j) to v -> (eta) { return eta; } }", "binds a name twice"),
    ("proposal { g2 at (j,k) to v -> (eta) { return eta; } "
     "         g2 at (a,b) to c -> (eta) { return eta; } }", "duplicate proposal"),
    # arity of the return, which only shows once the closure runs
    ("proposal { inc to v -> (eta) { return (eta, eta); } }", "returned 2 value"),
])
def test_proposal_errors(proposal, match):
    with pytest.raises(DslError, match=match):
        m = _compile(proposal)
        # the return-arity check fires when the closure runs, as JAX shape errors do
        op = next(iter(m.jump_operators.values()))
        op.fn({"eta": jnp.zeros(2), "g2": jnp.ones((3, 4), jnp.int32),
               "inc": jnp.int32(0), "n": 3, "m": 4}, 0, jnp.int32(1))


def test_a_proposal_sees_transformed_parameters():
    """A body has a model component's *scope*, which means the `transformed parameters` statements
    are prepended exactly as they are for a component. Nothing else would put them in reach."""
    m = compile_model("""
    data { int n; }
    parameters { array[2] real eta; array[n] int<lower=1, upper=3> z; }
    transformed parameters { real doubled = 2.0 * eta[1]; }
    model lik { target += sum(eta); }
    proposal {
      z at j to g -> (eta) {
        array[2] real e2 = eta;
        e2[1] = doubled;
        return e2;
      }
    }
    """, data={"n": 4})
    op = m.jump_operators["z"]
    out = op.fn({"eta": jnp.asarray([1.5, 0.0]), "z": jnp.ones(4, jnp.int32), "n": 4},
                0, jnp.int32(2))
    assert float(out[0][0]) == 3.0


# ------------------------------------------------------- 4. end to end, through the DSL

def test_a_dsl_proposal_samples_the_right_posterior():
    """The whole path: source -> Model -> sweep. A three-component mixture of known weights whose
    jump shifts the location to compensate the label change.

    The check is against the closed-form label marginal, and the CONTROL is the identical program
    with the `proposal` block removed --- which must also be correct, since a jump changes how the
    chain moves and not what it targets. So the real content is that adding the block does not
    *break* it, plus the move counter showing the jump is not inert.
    """
    body = """
    data { int n; array[3] real w; array[3] real mu; }
    parameters { real x; array[n] int<lower=1, upper=3> z; }
    model lik scan(z) {
      target += log(w[z]) - 0.5 * (x - mu[z]) * (x - mu[z]);
    }
    """
    prop = """
    proposal {
      z at j to g -> (x) { return x + mu[g] - mu[z[j]]; }
    }
    """
    data = {"n": 1, "w": np.array([0.2, 0.5, 0.3]), "mu": np.array([-1.0, 0.4, 2.0])}
    out = {}
    for label, src in (("jump", body + prop), ("control", body)):
        m = compile_model(src, data=data)
        s = NUTS_GIBBS(m, m.default_sample(), seed=0)
        s.initialize()
        s.warmup(600)
        s.sample(20000)
        z = np.asarray(s.get_samples()["z"]).ravel()
        out[label] = np.bincount(z - 1, minlength=3) / len(z)
        assert len(np.unique(z)) == 3, f"{label}: a label froze"
        assert float(np.sum(s.diagnostics()["discrete_moves"])) > 0, f"{label}: nothing moved"

    # The exact marginal: w_k * integral N-ish, so compare the two arms against each other AND
    # each against the prior weights, which the symmetric-sigma construction makes the answer.
    for label, emp in out.items():
        assert np.abs(emp - data["w"]).max() < 0.03, f"{label}: marginal {emp} off"
