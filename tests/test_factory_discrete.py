"""Factory wiring for discrete parameters --- ``analyze`` / ``make_sampler`` on an ``int`` model.

The factory used to **refuse** such a model, and the reason it refused is the reason these tests
are shaped the way they are: discrete parameters are kept out of ``model.parameters`` entirely, so
every rule would partition the continuous half perfectly well and hand back a sampler that never
moves a label --- and a frozen coordinate has zero variance, so it reports ESS = n and split R-hat
1.000. Nothing the factory or the summary prints would show it. The checks therefore lean on
statements that *can* fail:

* the built stack is compared **bit-for-bit** against a hand-composed one, with a sampler missing
  the marginal adaptation as the control that must differ (at ``k = 3``; at ``k = 2`` the two are
  bit-identical by the Hastings identity of doc 14, so a binary model cannot discriminate);
* the discrete-only path is checked against an **exactly enumerated** pmf, not against another
  sampler;
* the support-width gate is asserted on the composed MRO, not on the log line alone.

One finding worth stating because it makes an obvious test vacuous: the relative MRO order of
``DiscreteMarginalAdaptation`` and ``DiscreteMetropolisWithinGibbs`` **cannot** change the draws.
They touch disjoint hooks --- the sweep composes on ``kernel``, the adaptation writes tables in
``_postprocess_hooks`` --- so a test that "verifies" the documented ordering passes whichever way
round they go. The ordering is a readability convention; what actually constrains the draws is the
sweep being left of the *base algorithm*, and (under tempering) inside the replica exchange.
"""

import itertools

import numpy as np
import jax.numpy as jnp
import pytest

from mimcs.adaptation import (ClassifierTermination, DiscreteMarginalAdaptation,
                              RobbinsMonroStepSize, ScoreMassAdaptation, StepSizeLineSearch,
                              UniformInit)
from mimcs.adaptation.discrete_marginal import WIDE_SUPPORT
from mimcs.factory import analyze, make_sampler
from mimcs.factory.evidence import normalize
from mimcs.hmc import NUTS, DenseQuadraticKinetic, default_potentials, leapfrog
from mimcs.model import EuclideanParameter, IntegerParameter, Model
from mimcs.samplers import (DiscreteMetropolisWithinGibbs, StaticContinuous, make_sampler_class)


def _mro(sampler_or_cls) -> list[str]:
    cls = sampler_or_cls if isinstance(sampler_or_cls, type) else type(sampler_or_cls)
    return [c.__name__ for c in cls.__mro__]


def _mixture(k=3, per=8, seed=0):
    """A `k`-component mixture: `k` means and one label per observation."""
    rng = np.random.default_rng(seed)
    y = np.concatenate([rng.normal(3.0 * (j - (k - 1) / 2), 1.0, per) for j in range(k)])

    def lp(v):
        return -0.5 * jnp.sum((y - v["mu"][v["z"]]) ** 2) - 0.5 * jnp.sum(v["mu"] ** 2)

    return Model([EuclideanParameter("mu", (k,))], {"p": lp},
                 discrete_parameters=[IntegerParameter("z", (k * per,), lower=0, upper=k - 1)]), y


def _binary_only(n=3, w=(1.3, -0.7, 2.1)):
    """A discrete-only Ising-ish target: no continuous parameters, exactly enumerable."""
    wv = jnp.asarray(w[:n], float)
    J = jnp.asarray([[0.0, 1.5, -0.9], [1.5, 0.0, 0.6], [-0.9, 0.6, 0.0]])[:n, :n]

    def lp(v):
        z = v["z"].astype(float)
        return jnp.dot(wv, z) + 0.5 * z @ J @ z

    return Model([], {"p": lp},
                 discrete_parameters=[IntegerParameter("z", (n,), lower=0, upper=1)]), lp


def _exact_pmf(lp, n):
    states = np.array(list(itertools.product((0, 1), repeat=n)))
    logp = np.array([float(lp({"z": jnp.asarray(s)})) for s in states])
    w = np.exp(logp - logp.max())
    return states, w / w.sum()


def _wide(ni, extra_narrow=False):
    """A model with one wide discrete parameter, optionally alongside a narrow one."""
    disc = [IntegerParameter("w", (3,), lower=0, upper=ni - 1)]
    if extra_narrow:
        disc.append(IntegerParameter("z", (2,), lower=0, upper=1))

    def lp(v):
        return -0.5 * jnp.sum(v["mu"] ** 2) + 0.0 * jnp.sum(v["w"])

    return Model([EuclideanParameter("mu", (2,))], {"p": lp}, discrete_parameters=disc)


# --------------------------------------------------------------------------- #
# 1. the sweep is built at all                                                 #
# --------------------------------------------------------------------------- #

def test_the_factory_builds_a_sampler_that_moves_the_labels():
    """The refusal is gone, and what replaced it actually sweeps.

    ``handles_discrete`` is asserted alongside the MRO because it is the flag
    ``BaseSampler.__init__`` checks: a stack that failed to compose the sweep would have raised
    there rather than reaching this line, so the two together say the guard was satisfied for the
    right reason and not merely bypassed.
    """
    m, _ = _mixture()
    s = make_sampler(m, seed=0)
    assert "DiscreteMetropolisWithinGibbs" in _mro(s)
    assert type(s).handles_discrete
    s.warmup(50)
    s.sample(50)
    z = np.asarray(s.get_discrete_flat())
    assert z.shape == (50, m.discrete_dim)
    assert len(np.unique(z, axis=0)) > 1              # the labels actually moved


def test_the_sweep_is_composed_whatever_the_proposal_is():
    """The *proposal* is a choice; the sweep is not. Holding the labels frozen is the quiet wrong
    answer the whole guard exists to prevent, so no spec setting may produce it."""
    m, _ = _mixture()
    for proposal in ("marginal", None):
        spec = analyze(m)
        spec.discrete_proposal = proposal
        mro = _mro(spec.build(seed=0))
        assert "DiscreteMetropolisWithinGibbs" in mro
        assert ("DiscreteMarginalAdaptation" in mro) is (proposal == "marginal")


def test_an_unknown_discrete_proposal_raises():
    m, _ = _mixture()
    spec = analyze(m)
    spec.discrete_proposal = "marginals"              # a plausible typo
    with pytest.raises(ValueError, match="unknown discrete_proposal"):
        spec.build(seed=0)


def test_an_unknown_discrete_proposal_raises_on_a_continuous_model_too():
    """Validated on every model, not only the ones where the field bites --- otherwise a typo sits
    silently in a spec until someone adds an integer parameter."""
    m = Model([EuclideanParameter("x")], {"p": lambda v: -0.5 * v["x"] ** 2})
    spec = analyze(m)
    spec.discrete_proposal = "nonsense"
    with pytest.raises(ValueError, match="unknown discrete_proposal"):
        spec.build(seed=0)


# --------------------------------------------------------------------------- #
# 2. the built stack, bit-for-bit                                              #
# --------------------------------------------------------------------------- #

def test_the_built_stack_matches_a_hand_composed_one_bit_for_bit():
    """The sharpest single check on the wiring: any divergence in mixin set, mixin order relative
    to the base, or constructor kwargs moves the draws.

    The control is a stack **without** the marginal adaptation, and it is run at ``k = 3`` on
    purpose: at ``k = 2`` the Hastings term is identically zero and the adapted and unadapted
    samplers are bit-identical (doc 14), so a binary model would make the control pass vacuously.
    """
    m, _ = _mixture(k=3)
    kin = [DenseQuadraticKinetic(id="mu", slices=[(0, 3)])]
    pot = default_potentials(m)
    head = (ClassifierTermination, RobbinsMonroStepSize, ScoreMassAdaptation,
            StepSizeLineSearch, UniformInit)

    def hand(*tail):
        cls = make_sampler_class(*head, *tail, NUTS)
        return cls(m, np.asarray(m.default_sample(), float), seed=0, kinetics=kin, potentials=pot,
                   integrator=leapfrog(pot, kin), step_size=0.5,
                   target_accept=0.8, mass_min_samples=50)

    def run(s):
        s.warmup(80)
        s.sample(80)
        return np.asarray(s.get_samples_flat()), np.asarray(s.get_discrete_flat())

    built = run(make_sampler(m, seed=0))
    matched = run(hand(DiscreteMarginalAdaptation, DiscreteMetropolisWithinGibbs))
    control = run(hand(DiscreteMetropolisWithinGibbs))          # no learned marginal

    assert np.array_equal(built[0], matched[0])
    assert np.array_equal(built[1], matched[1])
    assert not np.array_equal(built[1], control[1])             # the control must differ


def test_the_factory_sampler_recovers_the_marginal_label_probability():
    """End to end, against a **quadrature** oracle rather than another sampler.

    The bit-for-bit test above pins the factory's stack to a hand-composed one, but that only
    transfers correctness if the hand-composed one is itself covered --- and this exact stack
    (with the termination and initialization mixins) is not the one `tests/test_discrete.py`
    exercises. So the composed sampler is checked directly: `P(z = 1)` by integrating the joint
    over `mu` on a grid, against the empirical label mean.
    """
    def lp(v):
        z = v["z"].astype(float)
        return -0.5 * (1.0 - 2.0 * z) ** 2 - 0.5 * v["mu"] ** 2 + 0.4 * z + 0.3 * v["mu"] * z

    m = Model([EuclideanParameter("mu")], {"p": lp},
              discrete_parameters=[IntegerParameter("z", (), lower=0, upper=1)])
    s = make_sampler(m, seed=0)
    s.initialize()
    s.warmup(1000)
    s.sample(12000)

    grid = np.linspace(-8, 8, 4001)
    mass = {z: np.trapezoid(
        np.exp([float(lp({"mu": jnp.asarray(g), "z": jnp.asarray(z)})) for g in grid]), grid)
        for z in (0, 1)}
    exact = mass[1] / (mass[0] + mass[1])

    z = np.asarray(s.get_samples()["z"]).ravel()
    assert len(np.unique(z)) == 2                          # it moved at all
    assert abs(z.mean() - exact) < 0.03
    # ... and the oracle is not trivially satisfied: the label is far from a coin flip
    assert abs(exact - 0.5) > 0.05


# --------------------------------------------------------------------------- #
# 3. the support-width gate                                                    #
# --------------------------------------------------------------------------- #

def test_a_narrow_support_gets_the_learned_marginal():
    m, _ = _mixture(k=3)
    spec = analyze(m)
    assert spec.discrete_proposal == "marginal"
    assert "DiscreteMarginalAdaptation" in _mro(spec.build(seed=0))


def test_a_wide_support_keeps_the_uniform_placeholder_and_says_so(caplog):
    """Above ``WIDE_SUPPORT`` each value would collect only ~1/n_i of the draws, so the table is
    mostly memory. The uniform proposal that stands instead is itself poor on a wide support ---
    which is why the omission warns rather than passing silently."""
    m = _wide(WIDE_SUPPORT + 1)
    with caplog.at_level("WARNING", logger="mimcs.factory.rules"):
        spec = analyze(m)
    assert spec.discrete_proposal is None
    assert "DiscreteMarginalAdaptation" not in _mro(spec.build(seed=0))
    # `getMessage()`, not `record.message % record.args`: the latter is only populated after a
    # handler formats the record, so it reads empty under caplog.
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "'w'" in text and str(WIDE_SUPPORT + 1) in text and "uniform" in text


def test_exactly_at_the_threshold_is_still_narrow(caplog):
    """The comparison is ``> WIDE_SUPPORT``, so 64 values adapt and 65 do not. Pinned because an
    off-by-one here is invisible: both arms produce a working sampler."""
    with caplog.at_level("WARNING", logger="mimcs.factory.rules"):
        assert analyze(_wide(WIDE_SUPPORT)).discrete_proposal == "marginal"
        assert not caplog.records
        assert analyze(_wide(WIDE_SUPPORT + 1)).discrete_proposal is None
        assert caplog.records


def test_the_widest_parameter_decides_for_the_whole_model(caplog):
    """The mixin allocates and updates every parameter's table in one hook, so it cannot adapt the
    narrow parameter and skip the wide one. Given the choice, the factory declines rather than
    building a table it has just called too wide --- and the warning names only the wide one."""
    m = _wide(200, extra_narrow=True)
    with caplog.at_level("WARNING", logger="mimcs.factory.rules"):
        spec = analyze(m)
    assert spec.discrete_proposal is None
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "'w'" in text and "'z'" not in text


def test_the_user_can_override_the_gate():
    """The threshold is a heuristic, and the spec is the override seam --- so asking for the
    learned marginal on a wide support must actually get it."""
    m = _wide(200)
    spec = analyze(m)
    spec.discrete_proposal = "marginal"
    assert "DiscreteMarginalAdaptation" in _mro(spec.build(seed=0))


# --------------------------------------------------------------------------- #
# 4. the discrete-only (static) base                                           #
# --------------------------------------------------------------------------- #

def test_a_discrete_only_model_gets_the_static_base_and_samples_exactly():
    """No continuous coordinate means nothing for NUTS to integrate, so the base is
    ``StaticContinuous`` and the sweep does all the moving. Checked against the **enumerated**
    posterior rather than another sampler."""
    m, lp = _binary_only()
    spec = analyze(m)
    assert spec.base == "static"
    assert spec.blocks == [] and spec.mass_adapt is None and not spec.adapt_step_size

    s = spec.build(seed=0)
    assert "StaticContinuous" in _mro(s) and "NUTS" not in _mro(s)
    s.initialize()
    s.warmup(400)
    s.sample(40000)

    draws = np.asarray(s.get_discrete_flat())
    emp = np.bincount(draws @ np.array([4, 2, 1]), minlength=8) / len(draws)
    _, exact = _exact_pmf(lp, 3)
    assert exact.max() / exact.min() > 20                   # the target is far from uniform
    assert np.max(np.abs(emp - exact)) < 0.012, (emp, exact)
    # ... and the oracle is not vacuous: a mis-scaled one must fail the same threshold
    assert np.max(np.abs(emp - np.roll(exact, 1))) > 0.012


def test_the_static_base_refuses_what_it_cannot_do():
    """Refused, not silently ignored: a user who asked for an adapted step size and got a sampler
    with no step size at all has been handed a different algorithm from the one they asked for."""
    m, _ = _binary_only()
    cont = Model([EuclideanParameter("x")], {"p": lambda v: -0.5 * v["x"] ** 2})

    spec = analyze(cont)
    spec.base = "static"
    with pytest.raises(ValueError, match="no discrete parameters"):
        spec.build(seed=0)

    for field, value, message in (("adapt_step_size", True, "no step to adapt|takes no step"),
                                  ("mass_adapt", "score", "no kinetics")):
        spec = analyze(m)
        setattr(spec, field, value)
        with pytest.raises(ValueError, match=message):
            spec.build(seed=0)

    spec = analyze(m)
    spec.base = "pt_static"
    spec.tempering_params = {"n_temperatures": 2}
    with pytest.raises(ValueError, match="pt_static"):
        spec.build(seed=0)


def test_the_static_base_refuses_a_leftover_block():
    """A block *is* a kinetic over a coordinate slice, and the static base has no kinetics."""
    from mimcs.factory import BlockSpec
    m, _ = _binary_only()
    spec = analyze(m)
    spec.blocks = [BlockSpec(names=["ghost"], coord_slices=[(0, 1)], kind="diagonal")]
    with pytest.raises(ValueError, match="no kinetics"):
        spec.build(seed=0)


# --------------------------------------------------------------------------- #
# 5. the tempered path                                                         #
# --------------------------------------------------------------------------- #

def test_the_tempered_path_composes_the_sweep_exactly_once():
    """``parallel_tempering`` injects the sweep itself, at a position ``build`` cannot express
    (inside the replica exchange, left of the selection mixins). So ``build`` must add only the
    *adaptation* --- adding the sweep too would both misplace it and duplicate it."""
    m, _ = _mixture(k=2, per=4)
    spec = analyze(m)
    spec.base = "pt_nuts"
    spec.tempering_params = {"n_temperatures": 3}
    s = spec.build(seed=0)

    mro = _mro(s)
    assert mro.count("DiscreteMetropolisWithinGibbs") == 1
    assert mro.index("ReplicaExchangeMixin") < mro.index("DiscreteMetropolisWithinGibbs")
    assert mro.index("DiscreteMarginalAdaptation") < mro.index("ReplicaExchangeMixin")
    # one proposal table per rung: a hot rung learns its own, flatter marginal
    assert s.state.discrete_proposal_params["z"].shape[0] == 3


def test_the_tempered_path_matches_parallel_tempering_directly():
    """The factory is a way of *deciding* the blanks, not a second implementation of them."""
    from mimcs.pt import parallel_tempering
    m, _ = _mixture(k=2, per=4)
    kin = [DenseQuadraticKinetic(id="mu", slices=[(0, 2)])]

    spec = analyze(m)
    spec.base = "pt_nuts"
    spec.tempering_params = {"n_temperatures": 3}
    spec.terminate = None
    spec.mass_adapt = None
    built = spec.build(seed=0)

    direct = parallel_tempering(
        m, np.asarray(m.default_sample(), float), n_temperatures=3, seed=0, kinetics=kin,
        step_size=0.5, extra_mixins=(RobbinsMonroStepSize, StepSizeLineSearch, UniformInit,
                                     DiscreteMarginalAdaptation),
        target_accept=0.8, mass_min_samples=50)

    for s in (built, direct):
        s.warmup(40)
        s.sample(40)
    assert np.array_equal(np.asarray(built.get_samples_flat()),
                          np.asarray(direct.get_samples_flat()))
    assert np.array_equal(np.asarray(built.get_discrete_flat()),
                          np.asarray(direct.get_discrete_flat()))


# --------------------------------------------------------------------------- #
# 6. evidence                                                                  #
# --------------------------------------------------------------------------- #

def test_evidence_carries_the_labels():
    m, _ = _mixture(k=2, per=8)
    s = make_sampler(m, seed=0)
    s.warmup(40)
    s.sample(60)
    ev = normalize(m, s)
    assert ev.discrete is not None
    assert ev.discrete.shape == (60, m.discrete_dim)
    assert ev.discrete.dtype == np.int32              # labels, not floats round-tripped


def test_a_recomputed_score_uses_each_row_s_own_labels():
    """A discrete model's coordinate-space density is *conditional* on the labels, so there is no
    score to take without them. Omitting them used to raise inside ``Model._require_discrete``,
    which the caller swallowed --- leaving ``gradients=None`` and silently disabling the mass-mode
    and metric-regression rules.

    The control is the omission itself: the same call without labels must still raise, so this is
    not passing because the density stopped needing them.
    """
    m, _ = _mixture(k=2, per=8)
    s = make_sampler(m, seed=0)
    s._save_gradients = False                          # force the recompute path
    s.warmup(40)
    s.sample(60)
    assert s.get_gradients() is None

    ev = normalize(m, s)
    assert ev.gradients is not None and ev.gradients.shape == (60, m.coord_dim)

    from mimcs.factory.evidence import _recomputed_scores
    with pytest.raises(ValueError, match="discrete parameter"):
        _recomputed_scores(m, s, ev.coordinates)       # the control: no labels, no score


def test_the_warm_start_carries_the_labels():
    """Warm-starting the position to a fitted configuration while resetting the labels would pair
    a position with the wrong assignment --- for a mixture, coordinates fitted under one clustering
    and every label saying 'cluster 0'."""
    m, _ = _mixture(k=2, per=8)
    pilot = make_sampler(m, seed=0)
    pilot.warmup(40)
    pilot.sample(60)

    warm = analyze(m, pilot).build(seed=1)
    assert np.array_equal(np.asarray(warm.state.discrete),
                          np.asarray(pilot.get_discrete_flat()[-1]))
    assert np.allclose(np.asarray(warm.state.sample), np.asarray(pilot.get_samples_flat()[-1]))

    cold = analyze(m).build(seed=1)                    # the control: no evidence, no warm start
    assert np.array_equal(np.asarray(cold.state.discrete), np.asarray(m.default_discrete()))


def test_a_tempered_build_warm_starts_its_position_but_not_its_labels():
    """``parallel_tempering`` takes one rung's position and tiles it ``K``-fold, so the init must
    stay a flat array there --- the dict form the untempered path uses to carry both halves cannot
    survive ``np.asarray(init, float)``, and it used to raise a bare ``TypeError`` from inside the
    tiling. Pinned because the failure is a build-time crash on a combination (tempering + discrete
    + evidence) that no other test exercises.
    """
    m, _ = _mixture(k=2, per=4)
    pilot = make_sampler(m, seed=0)
    pilot.warmup(20)
    pilot.sample(20)

    spec = analyze(m, pilot)
    spec.base = "pt_nuts"
    spec.tempering_params = {"n_temperatures": 2}
    s = spec.build(seed=1)                                  # must not raise

    assert np.array_equal(np.asarray(s.state.discrete),
                          np.asarray(s.model.default_discrete()))
    one = s.model.base.ambient_dim
    assert np.allclose(np.asarray(s.state.sample)[:one],
                       np.asarray(pilot.get_samples_flat()[-1]))


def test_the_dimension_guard_notices_a_different_discrete_width():
    """Two models can agree on every continuous dimension and still be different targets."""
    m8, _ = _mixture(k=2, per=4)
    m12, _ = _mixture(k=2, per=6)
    assert m8.coord_dim == m12.coord_dim and m8.discrete_dim != m12.discrete_dim
    s = make_sampler(m8, seed=0)
    s.warmup(10)
    s.sample(10)
    with pytest.raises(ValueError, match="different"):
        normalize(m12, s)


def test_a_bundle_may_carry_the_labels():
    m, _ = _mixture(k=2, per=4)
    z = np.zeros((5, m.discrete_dim), dtype=np.int32)
    x = np.zeros((5, m.ambient_dim))
    assert normalize(m, {"samples": x, "discrete": z}).discrete.shape == z.shape
    assert normalize(m, (x, None, None, z)).discrete.shape == z.shape
    assert normalize(m, (x, None, None)).discrete is None      # a 3-tuple still means what it did


# --------------------------------------------------------------------------- #
# 7. a discrete name is not a block                                            #
# --------------------------------------------------------------------------- #

def test_blocks_naming_a_discrete_parameter_says_which_mistake_it_is():
    """The name *is* a parameter of the model, so the generic "not a parameter" message would read
    as a typo when it is a category error."""
    m, _ = _mixture(k=2, per=4)
    with pytest.raises(ValueError, match="discrete parameter 'z'"):
        analyze(m, blocks=["z"])


# --------------------------------------------------------------------------- #
# 8. a continuous model is untouched                                           #
# --------------------------------------------------------------------------- #

def test_a_continuous_model_is_unchanged():
    """Both new rules are inert, the new spec field carries its default, and no discrete mixin is
    composed --- so nothing about an existing factory call moves."""
    m = Model([EuclideanParameter("x", (3,))], {"p": lambda v: -0.5 * jnp.sum(v["x"] ** 2)})
    spec = analyze(m)
    assert spec.base == "nuts"
    assert spec.discrete_proposal == "marginal"        # inert: the model has no labels
    assert spec.evidence.discrete is None
    mro = _mro(spec.build(seed=0))
    assert "DiscreteMetropolisWithinGibbs" not in mro
    assert "DiscreteMarginalAdaptation" not in mro
    assert "discrete" not in str(spec)                 # nor does the summary mention it
