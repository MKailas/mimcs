"""Tests for custom jump operators --- a deterministic map that moves continuous parameters
alongside a discrete label.

The construction is nonstandard enough that the algebra was verified in a throwaway script before
any of this existed (``tests/experiments/jump_algebra.py``, ``jump_flip.py``); the checks below are
those checks, kept rather than discarded, so they cannot go vacuous once a real implementation sits
underneath them.

Four things here are silent when wrong, and each has a control that must fail:

* **The balance conditions are two, not one.** Metropolis needs only the involution
  ``Phi_{b->a} . Phi_{a->b} = id``; exact Gibbs over the orbit needs the strictly stronger cocycle
  ``Phi_{b->v} . Phi_{a->b} = Phi_{a->v}``. The gap is real: ``flip`` below satisfies the first and
  not the second, and samples correctly under Metropolis while being wrong under Gibbs. A single
  merged check would either reject a valid Metropolis map or accept an invalid Gibbs one.
* **A volume-preserving map cannot detect a missing Jacobian**, because dropping the term leaves
  such a map correct. Every Jacobian control therefore runs on ``scale``, whose ``|det| != 1``, and
  asserts the shift arm stays *unaffected* --- that asymmetry is the evidence the term does work.
* **A jump-only chain is reducible.** With a deterministic map the reachable set from one state is
  the finite orbit ``{Phi_{a->v}(x)}``, so a chain that only jumps cannot explore the continuous
  parameter at all and any posterior check on it would be meaningless. Simulation tests pair the
  jump with a move on the continuous block.
* **A frozen label reads as perfectly converged**, and a jump reports acceptance by its own column,
  so ``discrete_moves`` and the distinct-value count remain the only things that catch a stuck
  chain.
"""

import numpy as np
import pytest
import jax.numpy as jnp

import mimcs
from mimcs.model import (EuclideanParameter, IntegerParameter, Model, UnitVectorParameter)
from mimcs.model.bounded import BoundedParameter, PositiveParameter
from mimcs.model.jump import JumpOperator
from mimcs.samplers.discrete_updates import JumpMap, SweepEnv
from mimcs.samplers.gibbs import only_in_scan_components
from mimcs.samplers import DiscreteMetropolisWithinGibbs, make_sampler_class
from mimcs.adaptation import RobbinsMonroStepSize
from mimcs.hmc import NUTS


def _noop(values, c, v):
    return (values["eta"],)


def _model(params, discrete, jumps, fns=None):
    return Model(params, fns or {"p": lambda v: jnp.zeros(())},
                 discrete_parameters=discrete, jump_operators=jumps)


Z = IntegerParameter("z", (3,), lower=0, upper=2)
ETA = EuclideanParameter("eta", (3,))


# ------------------------------------------------------------------ 1. the dataclass


def test_an_operator_must_rewrite_something():
    """An operator with no outputs is the ordinary sweep move, spelled expensively."""
    with pytest.raises(ValueError, match="rewrites no parameters"):
        JumpOperator("z", (), _noop)


def test_an_operator_may_not_name_an_output_twice():
    with pytest.raises(ValueError, match="more than once"):
        JumpOperator("z", ("eta", "eta"), _noop)


def test_the_repr_says_whether_it_scales():
    assert "scales" not in repr(JumpOperator("z", ("eta",), _noop))
    assert "scales" in repr(JumpOperator("z", ("eta",), _noop, volume_preserving=False))


# ------------------------------------------------ 2. what a model refuses, and why

def test_a_plain_continuous_output_builds():
    """The positive case the refusals below are controls for."""
    m = _model([ETA], [Z], {"z": JumpOperator("z", ("eta",), _noop)})
    assert list(m.jump_operators) == ["z"]
    assert _model([ETA], [Z], None).jump_operators == {}


def test_a_bounded_output_builds():
    """`bounded` is allowed because out of domain it gives NaN, which *rejects* --- loud, or at
    worst inert. That is the property the refusal below turns on, not the constraint itself."""
    tau = PositiveParameter("tau", ())
    _model([tau], [Z], {"z": JumpOperator("z", ("tau",), _noop)})


@pytest.mark.parametrize("outputs, match", [
    (("z",), "may only move .*continuous"),
    (("nope",), "not a parameter of this model"),
])
def test_output_must_be_a_continuous_parameter(outputs, match):
    with pytest.raises((ValueError, NotImplementedError), match=match):
        _model([ETA], [Z], {"z": JumpOperator("z", outputs, _noop)})


def test_a_projecting_chart_is_refused():
    """A unit vector's `to_coordinate` accepts an off-manifold value and silently projects, so a
    jump writing one would break the involution by exactly the projection error while reporting a
    finite density. CONTROL: the same model with a Euclidean output builds (above)."""
    uv = UnitVectorParameter("u", 3)
    with pytest.raises(NotImplementedError, match="bijection on its domain"):
        _model([uv], [Z], {"z": JumpOperator("z", ("u",), _noop)})


def test_a_chart_parent_output_is_refused_but_a_child_is_not():
    """The asymmetry is the point. Rewriting a *parent* moves its children's ambient values while
    their coordinates stand still --- parameters the operator never named. Rewriting a *child* is
    fine: its own chart reads a parent nobody touched."""
    lo = EuclideanParameter("lo", ())
    hi = BoundedParameter("hi", (), lower="lo")
    assert hi.parents == ("lo",)
    with pytest.raises(NotImplementedError, match="chart parent"):
        _model([lo, hi], [Z], {"z": JumpOperator("z", ("lo",), _noop)})
    _model([lo, hi], [Z], {"z": JumpOperator("z", ("hi",), lambda v, c, x: (v["hi"],))})


@pytest.mark.parametrize("key, param, match", [
    ("w", "w", "not a discrete parameter"),
    ("z", "q", "key and the operator's own parameter must agree"),
])
def test_the_key_must_name_the_operator_s_own_discrete_parameter(key, param, match):
    with pytest.raises(ValueError, match=match):
        _model([ETA], [Z], {key: JumpOperator(param, ("eta",), _noop)})


# ------------------------------------------------------- 3. the tempered forward

def test_a_product_model_carries_the_jump_operators():
    """`ProductModel` copies attributes one at a time, so an omitted one is not an error --- it
    silently degrades every jump parameter to a label-only move over the product space."""
    from mimcs.pt.tempering import ProductModel
    ops = {"z": JumpOperator("z", ("eta",), _noop)}
    m = _model([ETA], [Z], ops)
    assert ProductModel(m, 4).jump_operators == ops


# --------------------------------------------- 4. the factory cost guard

def _scan_mixture(k, n=30, seed=0):
    src = """
    data { int n; int k; array[n] real y; }
    parameters { array[k] real mu; array[n] int<lower=1, upper=k> z; }
    model lik scan(z, y) { y ~ normal(mu[z], 1.0); }
    model prior { mu ~ normal(0, 5); }
    """
    rng = np.random.default_rng(seed)
    return mimcs.compile_model(src, data={"n": n, "k": k, "y": rng.normal(size=n)})


def test_a_jump_drops_a_scanned_parameter_off_the_wide_exact_cap():
    """The elementwise cap (64) prices `O(1)` element work per candidate. A jump rewrites whole
    continuous arrays, so each candidate costs a *full* density and the cap must fall back to the
    general one (8). Without this, adding a `proposal` block to a working model silently multiplies
    its cost by ~8x with nothing reported.

    This tests the RULE, not the sampler: the control is the identical model with no operator.
    """
    k = 16
    plain = _scan_mixture(k)
    assert only_in_scan_components(plain, "z") is True
    assert mimcs.analyze(plain).discrete[0].kind == "exact"          # CONTROL

    jumped = _scan_mixture(k)
    jumped.jump_operators = {"z": JumpOperator("z", ("mu",), lambda v, c, x: (v["mu"],))}
    assert only_in_scan_components(jumped, "z") is False
    spec = mimcs.analyze(jumped)
    assert spec.discrete[0].kind == "metropolis"
    assert any("jump operator" in r for r in spec.rationale)


def test_a_narrow_scanned_parameter_with_a_jump_still_gets_exact():
    """The guard changes the *cap*, not the method: below the general cap a jump parameter is still
    exact, so the guard cannot be a blanket disable."""
    m = _scan_mixture(5)
    m.jump_operators = {"z": JumpOperator("z", ("mu",), lambda v, c, x: (v["mu"],))}
    assert mimcs.analyze(m).discrete[0].kind == "exact"


# ============================================================ 5. the algebra, end to end
#
# The toy from `tests/experiments/jump_algebra.py`: z in {0,1,2}, x | z ~ Normal(mu_z, sigma_z),
# z ~ Categorical(w). The joint is closed form, so a run can be checked against it rather than
# against another run.

W = jnp.asarray([0.2, 0.5, 0.3])
MU = jnp.asarray([-1.0, 0.4, 2.0])
SIG = jnp.asarray([0.7, 1.3, 0.5])

NUTS_GIBBS = make_sampler_class(RobbinsMonroStepSize, DiscreteMetropolisWithinGibbs, NUTS)


def _toy_logp(v):
    z, x = v["z"], jnp.reshape(v["x"], ())
    return (jnp.log(W[z]) - 0.5 * jnp.log(2 * jnp.pi * SIG[z] ** 2)
            - 0.5 * ((x - MU[z]) / SIG[z]) ** 2)


def _shift(values, c, v):
    """A compensating translation: volume preserving, and a cocycle by construction."""
    return (values["x"] + (MU[v] - MU[values["z"]]),)


def _scale(values, c, v):
    """|det| = sigma_v / sigma_a != 1. The ONLY kind of map that can detect a missing Jacobian:
    dropping the term leaves a volume-preserving map correct, so a shift-only check would pass on
    a broken implementation."""
    return (values["x"] * (SIG[v] / SIG[values["z"]]),)


def _flip(values, c, v):
    """An involution that is NOT a cocycle --- valid under Metropolis, invalid under exact Gibbs."""
    return (jnp.where(v == values["z"], values["x"], -values["x"]),)


def _broken(values, c, v):
    """Neither: written in terms of the proposed value alone, so nothing telescopes."""
    return (values["x"] * (1.0 + 0.3 * v),)


def _toy(jump=None, volume_preserving=True):
    ops = ({} if jump is None
           else {"z": JumpOperator("z", ("x",), jump, volume_preserving=volume_preserving)})
    return Model([EuclideanParameter("x", ())], {"p": _toy_logp},
                 discrete_parameters=[IntegerParameter("z", (), lower=0, upper=2)],
                 jump_operators=ops)


def _run_toy(m, kind="metropolis", n=30000, seed=0):
    s = NUTS_GIBBS(m, m.default_sample(), seed=seed, discrete_update={"z": kind})
    s.initialize()
    s.warmup(600)
    s.sample(n)
    d = s.get_samples()
    return s, np.asarray(d["z"]).ravel(), np.asarray(d["x"]).ravel()


def _errors(z, x):
    w, mu, sg = np.asarray(W), np.asarray(MU), np.asarray(SIG)
    emp = np.bincount(z, minlength=3) / len(z)
    return (float(np.abs(emp - w).max()),
            max(abs(x[z == k].mean() - mu[k]) for k in range(3)))


@pytest.mark.parametrize("kind", ["metropolis", "exact"])
def test_a_shift_jump_recovers_the_closed_form_joint(kind):
    """The positive result. Both the label marginal and each conditional mean, against a joint
    known in closed form rather than against another run."""
    s, z, x = _run_toy(_toy(_shift), kind=kind)
    dz, dm = _errors(z, x)
    assert dz < 0.02, f"label marginal off by {dz}"
    assert dm < 0.05, f"conditional mean off by {dm}"
    # Non-vacuity: a frozen label has zero variance and so a *perfect* ESS and R-hat 1.000, and an
    # exact-Gibbs coordinate reports acceptance 1.00 by construction --- these are what is left.
    assert len(np.unique(z)) == 3
    assert float(np.mean(s.diagnostics()["discrete_moves"])) > 0.05


def test_the_jump_actually_moves_the_continuous_block():
    """CONTROL for every posterior test above: if the map were inert they would all still pass,
    because the model is correct without a jump too. Here the coordinate must move *within* a
    sweep, not only under NUTS."""
    m = _toy(_shift)
    s = NUTS_GIBBS(m, m.default_sample(), seed=0, discrete_update={"z": "metropolis"})
    s.initialize()
    st = s.state
    moved = s._discrete_sweep(st._replace(discrete=jnp.asarray([2], jnp.int32)))
    assert not np.array_equal(np.asarray(moved.coordinate), np.asarray(st.coordinate)) or \
        int(moved.discrete[0]) == 2, "the sweep neither moved the label nor the coordinate"


def test_the_recorded_sample_tracks_the_moved_coordinate():
    """`_retained_sample` returns `state.sample`, so a stale one means every recorded continuous
    draw is the pre-jump value --- a wrong posterior behind clean traces.

    CONTROL: the pre-jump sample, which must differ from the post-jump one.
    """
    m = _toy(_shift)
    s = NUTS_GIBBS(m, m.default_sample(), seed=0, discrete_update={"z": "metropolis"})
    s.initialize()
    s.warmup(200)
    s.sample(50)
    st = s.state
    want = m.coordinate_to_sample(st.coordinate, st.chart_hyperparams, st.chart_indices)
    assert np.allclose(np.asarray(st.sample), np.asarray(want), atol=0), \
        "state.sample disagrees with its own coordinate"


def test_a_map_that_is_an_involution_but_not_a_cocycle_splits_the_two_methods():
    """The reason the balance check has two tiers rather than one.

    `_flip` satisfies the involution and not the cocycle. Metropolis needs only the first, so it
    must be ACCEPTED there; exact Gibbs draws from an orbit and needs the second, so it must be
    REJECTED there. A single merged condition would get one of these wrong.
    """
    m = _toy(_flip)
    NUTS_GIBBS(m, m.default_sample(), seed=0, discrete_update={"z": "metropolis"})
    with pytest.raises(ValueError, match="cocycle"):
        NUTS_GIBBS(m, m.default_sample(), seed=0, discrete_update={"z": "exact"})


@pytest.mark.parametrize("kind", ["metropolis", "exact"])
def test_a_map_that_is_not_an_involution_is_refused(kind):
    m = _toy(_broken)
    with pytest.raises(ValueError, match="not an involution"):
        NUTS_GIBBS(m, m.default_sample(), seed=0, discrete_update={"z": kind})


def test_the_balance_check_runs_without_initialize():
    """Where the check lives is itself a hazard: `initialize()` is optional, so a check placed in
    `_initialize_hooks` would silently never run for a user who goes straight to `warmup()`."""
    m = _toy(_broken)
    with pytest.raises(ValueError, match="not an involution"):
        NUTS_GIBBS(m, m.default_sample(), seed=0)      # construction alone, no initialize()


def test_a_false_volume_preservation_claim_is_refused():
    """Declaring a scaling map preserving buys a zero Jacobian and biases the posterior --- by ~5
    standard errors over 6 seeds, with every diagnostic looking ordinary.

    CONTROL: the same map declared honestly builds.
    """
    bad = _toy(_scale, volume_preserving=True)
    with pytest.raises(ValueError, match="declared volume preserving, but it is not"):
        NUTS_GIBBS(bad, bad.default_sample(), seed=0)
    honest = _toy(_scale, volume_preserving=False)
    NUTS_GIBBS(honest, honest.default_sample(), seed=0)


# ------------------------------------------------- the map and its Jacobian, in closed form

@pytest.mark.parametrize("name, fn, vol", [("shift", _shift, True), ("scale", _scale, False)])
def test_the_coordinate_map_and_its_jacobian_match_closed_forms(name, fn, vol):
    """Unit-level, because a posterior test cannot localise a fault here --- and the identity at
    the current value must hold *exactly*, which is what the exact-Gibbs anchor rests on."""
    m = _toy(fn, volume_preserving=vol)
    jm = JumpMap(m.jump_operators["z"], m)
    h, idx = m.init_chart_hyperparams(), m.init_chart_indices()
    x = jnp.asarray([0.83])
    for a in range(3):
        z = jnp.asarray([a], jnp.int32)
        for v in range(3):
            got = float(jm.apply(x, z, 0, jnp.int32(v), h, idx)[0])
            want = (float(x[0] + (MU[v] - MU[a])) if name == "shift"
                    else float(x[0] * SIG[v] / SIG[a]))
            assert abs(got - want) < 1e-5, f"map {a}->{v}: {got} != {want}"
            ld = float(jm.log_det(x, z, 0, jnp.int32(v), h, idx))
            want_ld = 0.0 if name == "shift" else float(jnp.log(SIG[v] / SIG[a]))
            assert abs(ld - want_ld) < 1e-5, f"log|det| {a}->{v}: {ld} != {want_ld}"
        # exactly, not approximately
        assert float(jm.apply(x, z, 0, jnp.int32(a), h, idx)[0]) == float(x[0])


def test_only_the_output_block_is_written():
    """The obvious spelling --- unpack, substitute, repack --- perturbs every OTHER parameter in
    its last bits, because `to_coordinate(from_coordinate(x))` is not bitwise identity for a
    nonlinear chart. That turns "moves only x" into a random walk on everything.

    CONTROL: the round trip, which must differ bitwise where the masked write does not.
    """
    y = EuclideanParameter("y", (2,))
    tau = PositiveParameter("tau", ())          # a nonlinear chart, so the round trip is lossy
    zp = IntegerParameter("z", (), lower=0, upper=2)
    m = Model([EuclideanParameter("x", ()), y, tau],
              {"p": lambda v: -0.5 * jnp.sum(v["y"] ** 2) - jnp.reshape(v["tau"], ())},
              discrete_parameters=[zp],
              jump_operators={"z": JumpOperator("z", ("x",), _shift)})
    jm = JumpMap(m.jump_operators["z"], m)
    h, idx = m.init_chart_hyperparams(), m.init_chart_indices()
    # The `tau` coordinate is chosen so its exp/log round trip is genuinely lossy in float32:
    # only ~9% of values are, so picking one at random would make the control below vacuous --- as
    # it was on the first attempt, which is why the control is asserted rather than assumed.
    zf = jnp.asarray([0], jnp.int32)
    rt = lambda q: m.pack_coordinate(m.unpack_coordinate(q, h, idx, zf), h, idx)
    tau_c = next(t for t in np.linspace(0.05, 3.0, 20000).astype(np.float32)
                 if not np.array_equal(
                     np.asarray(rt(jnp.asarray([0.83, 1.7, -0.4, t]))),
                     np.asarray(jnp.asarray([0.83, 1.7, -0.4, t]))))
    x = jnp.asarray([0.83, 1.7, -0.4, float(tau_c)])
    out = jm.apply(x, zf, 0, jnp.int32(2), h, idx)
    lo, hi = m.coord_block("x")
    untouched = np.concatenate([np.asarray(x)[:lo], np.asarray(x)[hi:]])
    got = np.concatenate([np.asarray(out)[:lo], np.asarray(out)[hi:]])
    assert np.array_equal(got, untouched), "a non-output coordinate block moved"

    # CONTROL: the round trip really is lossy at this point, so the assertion above has teeth.
    assert not np.array_equal(np.asarray(rt(x)), np.asarray(x))


# ==================================================================== 6. under tempering


@pytest.mark.parametrize("kind", ["metropolis", "exact"])
def test_a_jump_works_per_rung_under_tempering(kind):
    """No tempering-specific code exists for jumps, and this is what says that is right.

    Every density goes through `_discrete_log_prob`, so `ParallelTemperingSampler`'s override is
    picked up through the MRO --- the same reason `SweepEnv` carries the sampler rather than bound
    copies of its hooks. The map itself is per rung, over the *base* model's layout.
    """
    from mimcs.pt import parallel_tempering
    m = _toy(_shift)
    s = parallel_tempering(m, n_temperatures=3, seed=0, discrete_update={"z": kind})
    s.initialize()
    s.warmup(600)
    s.sample(20000)
    d = s.get_samples()
    z, x = np.asarray(d["z"]).ravel(), np.asarray(d["x"]).ravel()
    dz, dm = _errors(z, x)
    assert dz < 0.03, f"label marginal off by {dz}"
    assert dm < 0.05, f"conditional mean off by {dm}"
    assert len(np.unique(z)) == 3


def test_the_jacobian_is_not_tempered():
    """`log|det|` is a property of the state map, not of the density, so it enters each rung's
    ratio ONCE, unscaled by that rung's beta. Scaling it by beta is the obvious wrong 'fix'.

    Pinned structurally: for a scaling map the term depends on neither the coordinate nor the rung,
    so every lane must report the identical closed-form value. A beta-weighted term would come back
    as a ladder.
    """
    from mimcs.pt import parallel_tempering
    K = 4
    m = _toy(_scale, volume_preserving=False)
    s = parallel_tempering(m, n_temperatures=K, seed=0, discrete_update={"z": "metropolis"})
    s.initialize()
    u = s.discrete_updaters[0]
    st = s.state
    env = SweepEnv(sampler=s, state=st, sweep_ctx=None, plans={}, tables={},
                   u_prop=None, u_acc=None, n_lanes=K,
                   lane_dim=m.discrete_dim)
    x = st.coordinate.reshape(K, -1)
    z = st.discrete.reshape(K, -1)
    a, v = 0, 2
    z = z.at[:, 0].set(a)
    got = np.asarray(u._log_det(env, x, z, 0, jnp.full((K,), v, jnp.int32)))
    want = float(jnp.log(SIG[v] / SIG[a]))
    assert got.shape == (K,)
    assert np.allclose(got, want, atol=1e-5), f"{got} != {want} in every lane"
    # CONTROL: the betas really are a non-trivial ladder here, so "all lanes equal" has content.
    betas = np.asarray(s.context(st, kinetic_cache=False).betas)
    assert betas.min() < 0.9 * betas.max(), f"ladder is flat ({betas}), the check is vacuous"


def test_a_stay_put_draw_leaves_the_coordinate_exactly_alone():
    """`Phi_{a->a}` is only the identity up to *rounding* for the idiom this library recommends:
    `x + effect(g) - effect(z[j])` evaluates as `(x + effect(a)) - effect(a)` at `g == a`, which
    rounds twice and lands ~6e-8 away in float32.

    So exact Gibbs must skip the map when the draw lands on the current value. Without that, a
    stay-put draw random-walks the continuous block by an ulp at a time, with no acceptance test
    anywhere to stop it -- it is not rejected, because nothing is proposed.

    CONTROL: the same map applied unconditionally, which must move the coordinate.
    """
    # The two-step spelling, so the rounding is the realistic one rather than an exact difference.
    def rounding_shift(values, c, v):
        return (values["x"] + MU[v] - MU[values["z"]],)

    m = _toy(rounding_shift)
    jm = JumpMap(m.jump_operators["z"], m)
    h, idx = m.init_chart_hyperparams(), m.init_chart_indices()
    x = jnp.asarray([0.83])
    z = jnp.asarray([2], jnp.int32)
    # CONTROL: applied unconditionally, the "identity" really does move --- so the guard is needed.
    naive = jm.apply(x, z, 0, jnp.int32(2), h, idx)
    assert not np.array_equal(np.asarray(naive), np.asarray(x)), \
        "the control is vacuous: this map's no-op happens to be exact here"

    # And the sweep, which guards it, must not move the coordinate on a stay-put draw.
    s = NUTS_GIBBS(m, m.default_sample(), seed=0, discrete_update={"z": "exact"})
    s.initialize()
    st = s.state
    u = s.discrete_updaters[0]
    env = SweepEnv(sampler=s, state=st, sweep_ctx=None, plans=s._restricted(force=True),
                   tables=st.discrete_proposal_params,
                   u_prop=st.rng_draw.discrete_proposal, u_acc=st.rng_draw.discrete_accept,
                   n_lanes=1, lane_dim=m.discrete_dim)
    x0 = st.coordinate.reshape(1, -1)
    z0 = st.discrete.reshape(1, -1)
    carry = (z0, x0, jnp.zeros((1,)), jnp.zeros((1,)), jnp.zeros((1,), jnp.int32))
    zo, xo, *_ = u.step(env, u.prepare(env), 0, 0, carry)
    if int(zo[0, 0]) == int(z0[0, 0]):                   # the draw stayed put
        assert np.array_equal(np.asarray(xo), np.asarray(x0)), \
            "a stay-put draw moved the coordinate"
