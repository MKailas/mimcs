"""Learned metrics that condition on **discrete** parameters.

A metric is fitted to the conditional score covariance ``E[g_i g_i^T | q_{-i}]``. With integer
parameters in the model the right conditioning includes them --- ``E[... | q_{-i}, z]`` --- because
labels change a mode's scale and one mass averaged over them is wrong under each (doc 14). The
expected form is ``(Exp(discrete) + Exp()) * (Exp(continuous) + Exp())``: labels modulate the
continuous metric multiplicatively.

The tests lean on a target whose ideal metric is known **in closed form** rather than on a
reference run: with ``beta_j | z_j ~ N(0, s_j^2)``, the score is ``g_j = -beta_j / s_j^2`` and so
``E[g_j^2 | z_j] = 1 / s_j^2`` exactly. Under the ordinal coding a binary label standardizes to
``x = 2z - 1``, which pins both fitted parameters:

    W = log(eps / tau),    b = (log(1/tau^2) + log(1/eps^2)) / 2

so a fit can be checked against numbers, not against another fit.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from mimcs.adaptation import MetricAdaptation, RobbinsMonroStepSize
from mimcs.factory.regression import select_metric, typed_discrete, typed_dims
from mimcs.hmc import (NUTS, Exp, Sigmoid, SpExp, build_block, default_potentials, leapfrog)
from mimcs.hmc.metric_encode import encode_discrete, encoded_width, support_moments
from mimcs.model import EuclideanParameter, IntegerParameter, Model
from mimcs.samplers import DiscreteMetropolisWithinGibbs, make_sampler_class

TAU, EPS = 1.0, 0.2


def _evidence(n=3000, b=6, tau=TAU, eps=EPS, seed=0, coupled=True):
    """Draws from the closed-form target; ``coupled=False`` is the control where `z` is inert."""
    rng = np.random.default_rng(seed)
    z = rng.integers(0, 2, size=(n, b))
    s = np.where(z == 1, tau, eps) if coupled else np.full((n, b), tau)
    beta = rng.normal(scale=s)
    return beta, -beta / s**2, z


def _conditional_model(b=5, tau=TAU, eps=EPS):
    def lp(v):
        s = jnp.where(v["z"] == 1, tau, eps)
        return -0.5 * jnp.sum((v["beta"] / s) ** 2) - jnp.sum(jnp.log(s)) + 0.0 * jnp.sum(v["z"])

    return Model([EuclideanParameter("beta", (b,))], {"p": lp},
                 discrete_parameters=[IntegerParameter("z", (b,), lower=0, upper=1)])


# --------------------------------------------------------------------------- #
# 1. the encoding                                                              #
# --------------------------------------------------------------------------- #

def test_ordinal_uses_the_declared_support_not_the_data():
    """Support-based, so the transform is the same with a pilot and without one --- which it has
    to be, since `MetricAdaptation` starts cold and a hand-written metric never sees evidence."""
    assert support_moments(1, 5) == (3.0, pytest.approx(np.sqrt((25 - 1) / 12)))
    got = np.asarray(encode_discrete(jnp.asarray([1, 3, 5]), "ordinal", 1, 5))
    assert got == pytest.approx([-1.4142, 0.0, 1.4142], abs=1e-3)


def test_categorical_is_reference_coded():
    """`k-1` columns, the lowest value being the reference. Full one-hot would duplicate the
    additive `+ Exp()` constant and leave a direction the optimizer cannot resolve."""
    assert encoded_width(2, "categorical", 1, 3) == 4          # 2 coordinates x (3-1) levels
    got = np.asarray(encode_discrete(jnp.asarray([1, 3]), "categorical", 1, 3))
    assert got.tolist() == [0.0, 0.0, 0.0, 1.0]                # value 1 is the reference


def test_the_two_codings_coincide_for_a_binary_parameter():
    """The enumeration offers only one coding for a binary label, and this is why: the reference
    indicator is an affine function of the standardized value, so they span the same models."""
    z = jnp.asarray([0, 1, 1, 0, 1])
    o = np.asarray(encode_discrete(z, "ordinal", 0, 1))
    c = np.asarray(encode_discrete(z, "categorical", 0, 1))
    assert np.allclose(o, 2.0 * c - 1.0)


def test_a_degenerate_support_encodes_to_nothing_rather_than_dividing_by_zero():
    assert encoded_width(3, "categorical", 2, 2) == 0
    assert support_moments(2, 2)[1] == 1.0                     # floored, not zero


def test_an_unknown_coding_is_refused():
    with pytest.raises(ValueError, match="unknown discrete metric coding"):
        encode_discrete(jnp.asarray([0]), "nominal", 0, 1)


# --------------------------------------------------------------------------- #
# 2. the declaration                                                           #
# --------------------------------------------------------------------------- #

def test_discrete_deps_are_separate_from_continuous_ones():
    """`deps()` keeps meaning *continuous* so every existing caller --- all of which resolve a name
    against the coordinate layout a label is not in --- stays correct untouched."""
    e = Exp("sigma", categorical=["z"], ordinal="k")
    assert e.deps() == {"sigma"}
    assert e.discrete_deps() == {"z", "k"}
    assert (e.dep_kind("z"), e.dep_kind("k"), e.dep_kind("sigma")) == \
        ("categorical", "ordinal", None)


def test_a_bare_string_is_one_name():
    assert Exp(ordinal="z").discrete_deps() == {"z"}


def test_a_name_may_not_be_two_kinds_at_once():
    with pytest.raises(ValueError, match="more than once"):
        Exp("z", ordinal=["z"])


def test_the_repr_round_trips_the_declaration():
    """Reprs are asserted in tests and printed in `spec.rationale`, so a lossy one hides which
    coding was selected --- the single most interesting thing about a discrete candidate."""
    assert repr(Exp("s", categorical=["z"])) == "Exp('s', categorical=['z'])"
    assert repr(SpExp(ordinal=["z"])) == "SpExp(ordinal=['z'])"


def test_composites_collect_both_kinds():
    e = (Exp(ordinal=["z"]) + Exp()) * (Exp("x") + Exp())
    assert e.deps() == {"x"} and e.discrete_deps() == {"z"}
    assert e.dep_kind("z") == "ordinal"


def test_the_product_of_sums_initialises_at_its_target():
    """The target form is a product of sums --- supported by the algebra before this work but never
    built by anything, so it had no test. `Product` gives the whole target to its left factor and
    1.0 to the right, so the composite must land exactly on the target."""
    e = (Exp(ordinal=["z"]) + Exp()) * (Exp("x") + Exp())
    dims = {"z": 3, "x": 4}
    p = e.init_params(3, dims, target=jnp.asarray([5.0, 5.0, 5.0]))
    v = e.evaluate(p, {"z": jnp.zeros(3), "x": jnp.zeros(4)})
    assert np.asarray(v) == pytest.approx([5.0, 5.0, 5.0], rel=2e-2)
    # 12 + 3 + 15 + 3: each atom is block_dim * dep_width + block_dim.
    assert e.n_params(3, dims) == 33
    g = jax.grad(lambda pp: jnp.sum(e.evaluate(pp, {"z": jnp.zeros(3), "x": jnp.zeros(4)})))(p)
    assert all(np.all(np.isfinite(l)) for l in jax.tree_util.tree_leaves(g))


# --------------------------------------------------------------------------- #
# 3. the offline fit recovers a known metric                                   #
# --------------------------------------------------------------------------- #

def test_the_regression_recovers_the_closed_form_metric():
    beta, g, z = _evidence()
    b = beta.shape[1]
    ranked = select_metric(list(range(b)), {}, beta, g,
                           discrete_cols={"z": (list(range(b)), 0, 1)}, discrete=z)
    best = ranked[0]
    assert best.expr.discrete_deps() == {"z"}, repr(best.expr)
    p = best.params[0] if isinstance(best.params, list) else best.params
    W = np.asarray(p["W"][0]).ravel()
    assert W.mean() == pytest.approx(np.log(EPS / TAU), rel=0.05)


def test_labels_that_do_not_matter_keep_the_constant_baseline():
    """The control for the test above, and for the second selection pass: a discrete factor is
    multiplied onto the best continuous candidate, so noise could ride along. It must not."""
    beta, g, z = _evidence(coupled=False)
    b = beta.shape[1]
    ranked = select_metric(list(range(b)), {}, beta, g,
                           discrete_cols={"z": (list(range(b)), 0, 1)}, discrete=z)
    assert ranked[0].expr.discrete_deps() == set(), repr(ranked[0].expr)


def test_the_pool_keeps_the_two_namespaces_apart():
    """`dep_dims` carries both namespaces so `n_params` stays one call; the continuous enumeration
    must subtract the discrete names or a label is enumerated a second time as a coordinate."""
    from mimcs.factory.regression import enumerate_candidates
    pool = enumerate_candidates(4, {"s": 4}, param_budget=400, max_candidates=50,
                                discrete_cols={"z": ([0, 1, 2, 3], 0, 1)})
    assert {d for e in pool for d in e.deps()} == {"s"}
    assert {d for e in pool for d in e.discrete_deps()} == {"z"}


def test_both_codings_are_offered_only_above_two_values():
    from mimcs.factory.regression import discrete_factors
    binary = discrete_factors(3, {"z": ([0, 1, 2], 0, 1)})
    three = discrete_factors(3, {"z": ([0, 1, 2], 0, 2)})
    assert {e.dep_kind("z") for e in binary} == {"ordinal"}
    assert {e.dep_kind("z") for e in three} == {"ordinal", "categorical"}


def test_the_sparse_gate_reads_the_encoded_width():
    """A binary indicator of the block's own length is the elementwise case that matters
    (spike-and-slab, `z_j` for `beta_j`); a 3-level categorical over the same coordinates encodes
    to `2n` and is not an elementwise correspondence."""
    from mimcs.factory.regression import discrete_factors
    binary = discrete_factors(3, {"z": ([0, 1, 2], 0, 1)})
    three = discrete_factors(3, {"z": ([0, 1, 2], 0, 2)})
    assert any(isinstance(e, SpExp) for e in binary)
    assert not any(isinstance(e, SpExp) and e.dep_kind("z") == "categorical" for e in three)


def test_widths_are_per_candidate_not_shared():
    """The same label is one column as an ordinal and `k-1` as a categorical, so the width --- and
    therefore the parameter count AIC is judged on --- cannot be computed once for the pool."""
    cols = {"z": ([0, 1, 2], 0, 2)}
    o, c = Exp(ordinal=["z"]), Exp(categorical=["z"])
    assert typed_dims({}, typed_discrete(cols, o))["z"] == 3
    assert typed_dims({}, typed_discrete(cols, c))["z"] == 6


# --------------------------------------------------------------------------- #
# 4. the runtime path                                                          #
# --------------------------------------------------------------------------- #

def test_the_runtime_and_offline_encoders_agree():
    """One definition, two callers. A metric fitted offline that meant something else when the
    sampler evaluated it would produce a plausible number either way, so this is asserted rather
    than trusted."""
    m = _conditional_model()
    blk = build_block(m, "beta", SpExp(ordinal=["z"]))
    labels = jnp.asarray([0, 1, 1, 0, 1], jnp.int32)
    runtime = np.asarray(blk._dep_coords(jnp.zeros(5), labels)["z"])
    offline = np.asarray(encode_discrete(labels, "ordinal", 0, 1))
    assert np.array_equal(runtime, offline)


def test_a_purely_discrete_metric_takes_no_metric_derivative_kick():
    """`depends` gates the kick in `flow`. Labels do not move along a trajectory, so a metric over
    them alone is constant and needs none --- and a metric with a continuous dependency still
    does, which is the direction that would be a bug."""
    m = _conditional_model()
    assert build_block(m, "beta", SpExp(ordinal=["z"])).depends is False
    m2 = Model([EuclideanParameter("beta", (5,)), EuclideanParameter("v", ())],
               {"p": lambda val: -0.5 * jnp.sum(val["beta"] ** 2) - 0.5 * val["v"] ** 2},
               discrete_parameters=[IntegerParameter("z", (5,), lower=0, upper=1)])
    assert build_block(m2, "beta", Exp("v", ordinal=["z"])).depends is True


def test_an_unknown_discrete_dependency_is_named():
    m = _conditional_model()
    with pytest.raises(ValueError, match="no such discrete parameter"):
        build_block(m, "beta", SpExp(ordinal=["nope"]))


def test_the_online_adaptation_recovers_the_closed_form_metric():
    """The same numbers as the offline fit, reached by SGD during warmup --- which is what proves
    `state.discrete` actually reaches `metric_loss`."""
    m = _conditional_model()
    blk = build_block(m, "beta", SpExp(ordinal=["z"]))
    pot = default_potentials(m)
    cls = make_sampler_class(RobbinsMonroStepSize, MetricAdaptation,
                             DiscreteMetropolisWithinGibbs, NUTS)
    s = cls(m, m.default_sample(), seed=0, kinetics=[blk], potentials=pot,
            integrator=leapfrog(pot, [blk]), step_size=0.3)
    s.initialize()
    s.warmup(1500)
    d = s.sample(1500)

    W = np.asarray(s.state.ham_params["beta"]["W"][0]).ravel()
    assert W.mean() == pytest.approx(np.log(EPS / TAU), rel=0.1), W

    beta, z = np.asarray(d["beta"]), np.asarray(d["z"])
    assert beta[z == 1].std() == pytest.approx(TAU, rel=0.15)
    assert beta[z == 0].std() == pytest.approx(EPS, rel=0.15)


# --------------------------------------------------------------------------- #
# 5. under tempering                                                           #
# --------------------------------------------------------------------------- #

def test_each_rung_learns_the_metric_at_its_own_labels():
    """The adaptation half of PT x discrete, and it has a structural oracle rather than a
    tolerance.

    Tempering a ``N(0, s^2)`` block to ``pi^beta`` gives ``N(0, s^2/beta)`` with score
    ``-beta x / s^2``, so ``M = E[g^2] = beta / s^2``. With the ordinal coding that means the
    **contrast** ``W = log(eps/tau)`` is temperature-*invariant* while the level ``b`` shifts by
    ``log beta`` --- so ``b - log beta`` must be flat across rungs. A rung fed the wrong labels, or
    none, cannot produce that pattern; it is a far sharper check than "W is roughly right".

    The integration path already sliced labels per lane; this is the adaptation path, which needed
    ``AdaptState.discrete`` to exist at all.
    """
    from mimcs.pt import parallel_tempering
    m = _conditional_model()
    blk = build_block(m, "beta", SpExp(ordinal=["z"]))
    s = parallel_tempering(m, n_temperatures=3, seed=0, kinetics=[blk],
                           adapt_mixins=(MetricAdaptation,), step_size=0.3)
    s.warmup(1200)
    s.sample(200)

    W = np.asarray(s.state.ham_params["beta"]["W"][0])          # (K, block, 1)
    b = np.asarray(s.state.ham_params["beta"]["b"])             # (K, block)
    betas = np.asarray(s.state.ham_params["__betas"])
    assert W.shape[0] == b.shape[0] == 3                        # one metric per rung

    w_per_rung = W.reshape(3, -1).mean(axis=1)
    assert w_per_rung == pytest.approx(np.log(EPS / TAU), rel=0.15), w_per_rung
    level = b.mean(axis=1) - np.log(betas)                      # the beta shift removed
    assert level == pytest.approx(level[0], rel=0.15), level


# --------------------------------------------------------------------------- #
# 6. through the factory rule                                                  #
# --------------------------------------------------------------------------- #

def _rule_evidence(n=1500, b=6, seed=1):
    """A two-block model plus row-aligned evidence for it: ``beta`` (the learned block, whose
    scale is set by ``z``) and a 2-d ``s`` block for the metric to have a continuous dependency."""
    from mimcs.factory.evidence import Evidence
    from mimcs.factory.spec import BlockSpec, default_spec

    rng = np.random.default_rng(seed)
    z = rng.integers(0, 2, size=(n, b))
    s = rng.normal(size=(n, 2))
    scale = np.where(z == 1, TAU, EPS)
    beta = rng.normal(scale=scale)
    coords = np.concatenate([beta, s], axis=1)
    grads = np.concatenate([-beta / scale**2, -s], axis=1)

    model = Model([EuclideanParameter("beta", (b,)), EuclideanParameter("s", (2,))],
                  {"p": lambda v: -0.5 * jnp.sum(v["beta"] ** 2) - 0.5 * jnp.sum(v["s"] ** 2)},
                  discrete_parameters=[IntegerParameter("z", (b,), lower=0, upper=1)])
    spec = default_spec(model)
    spec.blocks = [BlockSpec(names=["beta"], coord_slices=[(0, b)], kind="diagonal"),
                   BlockSpec(names=["s"], coord_slices=[(b, b + 2)], kind="dense")]
    return model, spec, Evidence(coordinates=coords, gradients=grads, discrete=z)


def test_the_rule_proposes_a_label_only_metric():
    """The seam the unit tests above cannot see. They all call ``select_metric``, which types its
    own discrete map and reads both namespaces; ``learned_metric_rule`` is the *caller*, and it
    had to do neither correctly. Two defects, both found 2026-09-02 on a spike-and-slab logistic
    regression, where a second ``analyze`` round is exactly the intended workflow:

    * it passed the kind-free ``(cols, lo, hi)`` map straight into ``whitened_scores`` (which it
      calls itself, to pick the metric's constant shape), raising ``ValueError: not enough values
      to unpack (expected 4, got 3)`` in ``_dep_data`` --- and only after the whole candidate pool
      had been fitted;
    * it spelled both "is this constant?" tests with ``deps()``, the **continuous-only** accessor,
      so a label-dependent candidate read as constant. Since ``ranked`` is AIC-sorted the baseline
      could then *be* the winner, silently declining every metric on a model whose ideal metric is
      a function of the labels alone.

    The evidence here makes the second the live one: the continuous block is pure noise, so the
    honest winner is label-only. It is also checkable --- ``SpExp(ordinal=['z'])`` should recover
    ``W = log(eps/tau)``, the same closed form the unit tests use.
    """
    from mimcs.factory.rules import learned_metric_rule

    model, spec, ev = _rule_evidence()
    proposals = learned_metric_rule(spec, ev, model)
    kinds = [p for p in proposals if p.slot == "blocks[0].kind"]
    assert kinds and kinds[0].value == "learned_metric", proposals
    params = next(p for p in proposals if p.slot == "blocks[0].params").value
    chosen = params["metric"]
    assert chosen.discrete_deps() == {"z"} and chosen.deps() == set(), repr(chosen)
    W = np.asarray(params["metric_init"]["W"][0]).ravel()
    assert W.mean() == pytest.approx(np.log(EPS / TAU), rel=0.05), W
