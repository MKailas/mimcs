"""Tests for the sampler factory (``docs/design/09_sampler_factory.md``).

Fast unit tests of the interface (Evidence normalization, the default spec, the
block-partition rule, the weighted arbiter, and the spec->sampler lowering onto a list of
kinetics), plus end-to-end runs proving ``make_sampler`` produces samplers that sample
correctly (a single dense block, and a mixed multi-block partition).
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from mimcs import Model
from mimcs.model import EuclideanParameter
from mimcs.factory import normalize, analyze, make_sampler, default_spec, BlockSpec
from mimcs.factory.rules import (
    Proposal, arbitrate, DENSE_MAX_DIM, FUSE_MIN_DIM, LOWRANK_MAX_DIM, LOWRANK_DEFAULT_RANK)
from mimcs.adaptation import (
    RobbinsMonroStepSize, ScoreMassAdaptation, MassMatrixAdaptation, RobustCenteringAdaptation,
    LowRankAdaptation, UnitVectorCenteringAdaptation)
from mimcs.hmc import NUTS, HMC, LowRankQuadraticKinetic
from mimcs.hmc.metric_expr import Exp
from mimcs.testing import problems as P, evaluate, block_gaussian, TargetProblem


def _gaussian(mean, cov):
    return P.correlated_gaussian(mean=mean, cov=cov)


def _model(*dims):
    """A model with one Euclidean parameter ``p{i}`` of each given coordinate dimension."""
    names = [f"p{i}" for i in range(len(dims))]
    params = [EuclideanParameter(n, (d,)) for n, d in zip(names, dims)]
    return Model(params, {"lp": lambda p: -0.5 * sum(jnp.sum(p[n] ** 2) for n in names)})


def _partition(model, **kwargs):
    return [(tuple(b.names), b.coord_slices, b.kind) for b in analyze(model, **kwargs).blocks]


# --- Evidence normalization -------------------------------------------------- #

def test_normalize_array_tuple_dict():
    model = _gaussian([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]]).model
    x = np.zeros((10, 2))

    assert normalize(model, x).samples.shape == (10, 2)
    assert normalize(model).samples is None                       # no results -> empty

    ev = normalize(model, (x, x, x))                              # (samples, coords, grads)
    assert ev.samples.shape == ev.coordinates.shape == ev.gradients.shape == (10, 2)

    ev = normalize(model, {"samples": x, "gradients": x})
    assert ev.samples.shape == (10, 2) and ev.coordinates is None
    assert ev.gradients.shape == (10, 2)

    with pytest.raises(TypeError, match="cannot interpret"):
        normalize(model, "not a result")


def test_normalize_sampler_output_duck():
    model = _gaussian([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]]).model
    from mimcs.testing.runner import SamplerOutput
    out = SamplerOutput(name="x", samples=np.zeros((7, 2)), accept_rate=0.9,
                        ess=np.array([5.0, 6.0]), warmup_step_sizes=np.array([0.5]))
    ev = normalize(model, out)
    assert ev.samples.shape == (7, 2)
    assert ev.diagnostics.accept_rate == 0.9
    assert ev.diagnostics.ess.tolist() == [5.0, 6.0]


def test_normalize_live_sampler_and_model_mismatch():
    model = _gaussian([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]]).model
    sampler = make_sampler(model)
    sampler.warmup(5)
    sampler.sample(5)

    ev = normalize(model, sampler)
    assert ev.samples.shape == (5, 2)
    assert ev.diagnostics is not None
    assert isinstance(ev.diagnostics.divergence_count, int)      # NUTS exposes this

    other = _gaussian([0.0, 0.0, 0.0], np.eye(3)).model          # different dimension
    with pytest.raises(ValueError, match="different"):
        normalize(other, sampler)


def test_normalize_sampler_uses_saved_gradients():
    """A sampler saves its gradients by default, so normalize reuses them (no recompute) and
    always recomputes the cheap coordinates."""
    model = _gaussian([0.5, -1.0], [[1.5, 0.4], [0.4, 0.8]]).model
    sampler = make_sampler(model, seed=0)
    sampler.warmup(50)
    sampler.sample(80)

    ev = normalize(model, sampler)
    assert ev.coordinates.shape == (80, 2)                       # coordinates always recomputed
    assert ev.gradients is not None
    assert np.array_equal(ev.gradients, sampler.get_gradients())  # the saved gradients, verbatim


def test_normalize_sampler_recompute_and_skip():
    """Without saved gradients, normalize recomputes them by default, or skips them (leaving the
    coordinates) when ``recompute_gradients=False`` --- for an expensive model."""
    model = _gaussian([0.5, -1.0], [[1.5, 0.4], [0.4, 0.8]]).model
    from mimcs.testing import hmc
    sampler = hmc(n_leapfrog=8, step_size=0.3, save_gradients=False)(model, 0)
    sampler.warmup(50)
    sampler.sample(80)
    assert sampler.get_gradients() is None                       # nothing saved

    assert normalize(model, sampler).gradients is not None       # recomputed (default)
    skipped = normalize(model, sampler, recompute_gradients=False)
    assert skipped.gradients is None and skipped.coordinates.shape == (80, 2)


# --- default spec ------------------------------------------------------------ #

def test_default_spec_is_the_baseline():
    model = _gaussian([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]]).model
    spec = default_spec(model)
    assert spec.base == "nuts"
    assert len(spec.blocks) == 1                     # one whole-space block, before rules run
    assert spec.blocks[0].kind == "diagonal"
    assert spec.blocks[0].coord_slices == [(0, model.coord_dim)]
    assert spec.centering is False and spec.adapt_step_size is True   # centering opt-in, off
    assert spec.terminate == "classifier"            # warmup termination on by default


# --- the RNG buffer ----------------------------------------------------------- #

def test_buffer_size_reaches_the_rng_buffer_through_every_entry_point():
    """`buffer_size` sizes the RNG buffer, which for NUTS is ~21 MB at the defaults almost
    regardless of the model's dimension (it is dominated by `leaf_select`, not by `d`). It has
    always been reachable via `algo_kwargs`; these are the explicit routes."""
    model = _gaussian([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]]).model

    assert make_sampler(model, seed=0)._rng_buffer.buffer_size == 1024          # the default
    assert make_sampler(model, seed=0, buffer_size=64)._rng_buffer.buffer_size == 64
    assert analyze(model).build(seed=0, buffer_size=32)._rng_buffer.buffer_size == 32

    spec = analyze(model)                                                       # tempered path
    spec.base = "pt_nuts"
    spec.tempering_params = {"n_temperatures": 2, "beta_min": 0.5}
    assert spec.build(seed=0, buffer_size=48)._rng_buffer.buffer_size == 48


def test_an_explicit_buffer_size_overrides_the_spec_and_none_does_not():
    """`None` means "unspecified", so it must leave `algo_kwargs` in force --- the distinction
    that lets the default be a genuine no-op rather than an implicit 1024 that overwrites."""
    spec = analyze(_gaussian([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]]).model)
    spec.algo_kwargs = {"buffer_size": 128}
    assert spec.build(seed=0)._rng_buffer.buffer_size == 128
    assert spec.build(seed=0, buffer_size=None)._rng_buffer.buffer_size == 128
    assert spec.build(seed=0, buffer_size=16)._rng_buffer.buffer_size == 16


def test_a_smaller_buffer_really_allocates_less():
    """The point of the knob. Asserted on the buffered arrays themselves rather than on a proxy,
    since the whole motivation is memory."""
    model = _gaussian([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]]).model
    sizes = {}
    for b in (16, 256):
        s = make_sampler(model, seed=0, buffer_size=b)
        s.warmup(1)                                       # force the (lazy) first fill
        sizes[b] = sum(int(np.asarray(v).size) for v in s._rng_buffer._buffers.values())
    assert sizes[16] * 16 == sizes[256], sizes           # exactly linear in buffer_size


def test_the_default_buffer_size_leaves_the_stream_untouched():
    """Adding the keyword must not perturb the default path: `buffer_size` is *not* stream-neutral
    (see tests/test_rng.py), so an accidental change of the default would silently move every
    seed-pinned result in the suite."""
    model = _gaussian([1.0, -2.0], [[2.0, 1.4], [1.4, 1.5]]).model
    a, b = make_sampler(model, seed=0), make_sampler(model, seed=0, buffer_size=None)
    for s in (a, b):
        s.warmup(30)
    assert np.array_equal(np.asarray(a.sample(30)["x"]), np.asarray(b.sample(30)["x"]))


# --- randomized integrator vs. the base's per-step RNG ------------------------ #

@pytest.mark.parametrize("base", ["hmc", "randomized_hmc", "pt_hmc", "pt_randomized_hmc"])
def test_a_randomized_integrator_is_refused_by_a_base_that_cannot_randomize_it(base):
    """`MarkovianLineSearchIntegrator` asks for per-step coins. A base that integrates a whole
    trajectory in one `integrate` call has nowhere to put them, so the integrator runs as its
    deterministic variant (WALNUTS-D) and says nothing --- the user gets a different algorithm
    from the one they asked for. Refused instead.

    `randomized_hmc` is in the list deliberately: its `traj_uniform` randomizes the trajectory
    *length*, not the integrator's per-step refinement, so it is no better off than fixed-length
    HMC despite the name."""
    spec = analyze(_gaussian([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]]).model)
    spec.base = base
    spec.integrator, spec.integrator_params = "markovian_line_search", {"error_thresholds": 0.8}
    if base.startswith("pt_"):
        spec.tempering_params = {"n_temperatures": 2, "beta_min": 0.5}
    with pytest.raises(ValueError, match="needs per-step randomness"):
        spec.build(seed=0)


@pytest.mark.parametrize("base", ["nuts", "pt_nuts"])
def test_a_nuts_base_really_randomizes_the_markovian_integrator(base):
    """The positive case, asserted on the draw stream rather than on the type: NUTS declares a
    per-leaf coin array keyed on the integrator's `n_rng_per_step`, which is what makes the
    refinement genuinely randomized."""
    spec = analyze(_gaussian([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]]).model)
    spec.base = base
    spec.integrator, spec.integrator_params = "markovian_line_search", {"error_thresholds": 0.8}
    if base.startswith("pt_"):
        spec.tempering_params = {"n_temperatures": 2, "beta_min": 0.5}
    s = spec.build(seed=0)
    assert s.integrator.n_rng_per_step > 0
    assert "line_search" in [c.name for c in s._draw_components]


def test_the_deterministic_line_search_is_accepted_by_every_base():
    """`line_search` consumes no randomness, so it is not affected by the refusal above --- the
    check must be about the randomized variant only, not about line searches in general."""
    for base in ("nuts", "hmc", "randomized_hmc"):
        spec = analyze(_gaussian([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]]).model)
        spec.base = base
        spec.integrator, spec.integrator_params = "line_search", {"error_thresholds": 0.8}
        s = spec.build(seed=0)
        assert s.integrator.n_rng_per_step == 0


# --- mass adaptation --------------------------------------------------------- #

def test_factory_wires_the_mass_adaptation():
    """`mass_adapt` selects the estimator for the *quadratic* blocks. The two candidates share a
    block filter (`mass_mode in ("diagonal", "dense")`), so this is a swap and never a stack ---
    which is what the `not in` assertion pins."""
    model = _gaussian([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]]).model

    default = make_sampler(model, seed=0)              # spec.mass_adapt defaults to "score"
    assert ScoreMassAdaptation in type(default).__mro__

    spec = analyze(model)
    spec.mass_adapt = "covariance"
    cov = spec.build(seed=0)
    assert MassMatrixAdaptation in type(cov).__mro__
    assert ScoreMassAdaptation not in type(cov).__mro__

    spec.mass_adapt = None
    off = spec.build(seed=0)
    assert not (set(type(off).__mro__) & {ScoreMassAdaptation, MassMatrixAdaptation})
    # ... and the sampler is still usable: the kinetic's own identity mass is always present.
    assert np.allclose(np.asarray(off.state.ham_params["x"]), np.eye(2))

    spec.mass_adapt = "bogus"
    with pytest.raises(ValueError, match="unknown mass_adapt"):
        spec.build(seed=0)


def test_the_default_mass_adaptation_is_the_same_path():
    """Adding the field must not perturb the default sampler. Membership tests cannot see a
    misplaced `append`, so this pins the composed order *and* checks the draws bitwise."""
    model = _gaussian([1.0, -2.0], [[2.0, 1.4], [1.4, 1.5]]).model

    implicit = make_sampler(model, seed=0)
    spec = analyze(model)
    spec.mass_adapt = "score"                          # the default, stated explicitly
    explicit = spec.build(seed=0)

    assert [c.__name__ for c in type(implicit).__mro__] == \
           [c.__name__ for c in type(explicit).__mro__]
    assert [c.__name__ for c in type(implicit).__mro__[1:6]] == [
        "ClassifierTermination", "_WarmupTermination", "RobbinsMonroStepSize",
        "ScoreMassAdaptation", "StepSizeLineSearch"]

    for s in (implicit, explicit):
        s.warmup(30)
    assert np.array_equal(np.asarray(implicit.sample(30)["x"]),
                          np.asarray(explicit.sample(30)["x"]))


def test_covariance_mass_waits_for_mass_min_samples():
    """`MassMatrixAdaptation` writes nothing for the first `mass_min_samples` (50) draws, so a
    short warmup silently leaves the mass at identity. Safe at the default `min_warmup` of 500,
    but worth pinning because nothing warns."""
    model = _gaussian([1.0, -2.0], [[2.0, 1.4], [1.4, 1.5]]).model
    spec = analyze(model)
    spec.mass_adapt = "covariance"

    early = spec.build(seed=0)
    early.warmup(30)
    assert np.allclose(np.asarray(early.state.ham_params["x"]), np.eye(2))

    late = spec.build(seed=0)
    late.warmup(300)
    assert not np.allclose(np.asarray(late.state.ham_params["x"]), np.eye(2))


def test_covariance_mass_is_a_partial_swap():
    """`mass_adapt` governs only the quadratic blocks: a low-rank block keeps its own (score-
    driven) adaptation, so a mixed partition runs both estimators side by side."""
    spec = analyze(_model(2, 60))
    spec.mass_adapt = "covariance"
    s = spec.build(seed=0)
    assert MassMatrixAdaptation in type(s).__mro__ and LowRankAdaptation in type(s).__mro__
    s.warmup(120)                                      # > mass_min_samples
    assert set(s._mm_stat) == {"p0"}                   # the dense block, by covariance
    assert set(s._lr_blocks) == {"p1"}                 # the low-rank one, still by score
    assert set(s.state.ham_params) == {"p0", "p1"}


def test_covariance_mass_samples_a_correlated_gaussian():
    """End to end: the empirical-covariance mass must actually sample correctly through the
    factory, and land the adapted mass near the target covariance."""
    cov = np.array([[2.0, 1.4], [1.4, 1.5]])
    spec = analyze(_gaussian([1.0, -2.0], cov).model)
    spec.mass_adapt = "covariance"
    s = spec.build(seed=0)
    s.warmup(600)
    draws = np.asarray(s.sample(2000)["x"])
    assert np.all(np.abs(draws.mean(axis=0) - np.array([1.0, -2.0])) < 0.2)
    assert np.all(np.abs(np.cov(draws.T) / cov - 1.0) < 0.25), np.cov(draws.T)
    L = np.asarray(s.state.ham_params["x"])            # M^-1 = L L^T, i.e. the fitted covariance
    assert np.all(np.abs(L @ L.T / cov - 1.0) < 0.4), L @ L.T


# --- unit-vector chart adaptation ------------------------------------------- #

def test_factory_composes_unit_vector_centering_and_the_chart_moves():
    """`UnitVectorParameter.adaptive` defaults to True and its docstring promises the mixin will
    fit the chart, but the factory never composed it --- so every factory-built sphere model ran
    with its chart frozen at the initial `pole = e_d`. That default is actively bad here: this
    problem's mass sits at `mu = e_3`, i.e. exactly where the unfitted pole is.

    Asserted end to end rather than by MRO membership alone, because membership is what the
    hand-built tests already cover; what was missing is that the factory path reaches them."""
    mu = np.array([0.0, 0.0, 1.0])
    s = make_sampler(P.von_mises_fisher(kappa=5.0, mu=tuple(mu)).model, seed=0)
    assert UnitVectorCenteringAdaptation in type(s).__mro__

    pole_before, _ = (np.asarray(v).copy() for v in s.unit_vector_chart("x"))
    assert np.allclose(pole_before, [0.0, 0.0, 1.0])       # the unfitted chart, in the bulk
    s.warmup(800)
    pole_after, offset_after = (np.asarray(v) for v in s.unit_vector_chart("x"))
    assert not np.allclose(pole_before, pole_after), "the chart never moved"
    assert float(pole_after @ -mu) > 0.9, f"pole {pole_after} not opposite the mass"
    assert offset_after != 0.0                             # the plane offset was fitted too


@pytest.mark.parametrize("problem, expected", [("euclidean", False), ("unit_vector", True)])
def test_unit_vector_centering_is_gated_on_the_model(problem, expected):
    """A model-derived gate, not a spec field: the parameter carries the switch."""
    model = (_gaussian([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]]).model if problem == "euclidean"
             else P.von_mises_fisher(kappa=5.0).model)
    s = make_sampler(model, seed=0)
    assert (UnitVectorCenteringAdaptation in type(s).__mro__) is expected


def test_adaptive_false_opts_out_of_unit_vector_centering():
    """`adaptive=False` on the parameter is how a user declines it --- the factory must honour
    that rather than deciding for itself."""
    s = make_sampler(P.von_mises_fisher(kappa=5.0, adaptive=False).model, seed=0)
    assert UnitVectorCenteringAdaptation not in type(s).__mro__


def test_a_tempered_base_refuses_an_adaptive_unit_vector():
    """A pole is not a per-rung quantity, and `ProductModel.parameters` is the *base* model's
    list --- so the mixin would find the parameter and rechart with base-model offsets against a
    K*n product coordinate. Wrong with nothing raising, hence the explicit refusal."""
    spec = analyze(P.von_mises_fisher(kappa=5.0).model)
    spec.base = "pt_nuts"
    with pytest.raises(NotImplementedError, match="adaptive unit vector"):
        spec.build(seed=0)


def test_factory_wires_the_termination_criterion():
    from mimcs.adaptation import ClassifierTermination, GelmanRubinTermination
    model = _gaussian([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]]).model

    default = make_sampler(model, seed=0)            # spec.terminate defaults to "classifier"
    assert ClassifierTermination in type(default).__mro__

    spec = analyze(model)
    spec.terminate = "rhat"
    rhat = spec.build(seed=0)
    assert GelmanRubinTermination in type(rhat).__mro__
    assert ClassifierTermination not in type(rhat).__mro__

    spec.terminate = None
    off = spec.build(seed=0)
    assert not (set(type(off).__mro__) & {ClassifierTermination, GelmanRubinTermination})

    spec.terminate = "bogus"
    with pytest.raises(ValueError, match="unknown terminate"):
        spec.build(seed=0)


def test_factory_default_terminates_warmup_early_on_an_easy_target():
    """The default classifier ends warmup well short of the cap on a 2-D Gaussian."""
    model = _gaussian([1.0, -2.0], [[2.0, 1.4], [1.4, 1.5]]).model
    sampler = make_sampler(model, seed=0)
    sampler.initialize()
    sampler.warmup(8000)                             # n is now an upper bound
    assert sampler.warmup_terminated_early()
    assert sampler._iteration < 2000
    sampler.sample(4000)
    draws = sampler.get_samples_flat()
    assert draws.shape == (4000, model.ambient_dim) and np.all(np.isfinite(draws))


# --- the block-partition rule ------------------------------------------------ #

def test_partition_lone_scalar_is_diagonal():
    assert _partition(_model(1)) == [(("p0",), [(0, 1)], "diagonal")]


def test_partition_low_dim_param_is_dense():
    assert _partition(_model(5)) == [(("p0",), [(0, 5)], "dense")]


def test_partition_large_param_is_lowrank():
    d = DENSE_MAX_DIM + 10                                   # 51..1000 -> low-rank mass
    assert _partition(_model(d)) == [(("p0",), [(0, d)], "lowrank")]
    block = analyze(_model(d)).blocks[0]
    assert block.params == {"rank": LOWRANK_DEFAULT_RANK}


def test_partition_very_large_param_is_diagonal():
    d = LOWRANK_MAX_DIM + 10                                 # above LOWRANK_MAX_DIM -> diagonal
    assert _partition(_model(d)) == [(("p0",), [(0, d)], "diagonal")]


def test_partition_medium_param_is_its_own_dense_block():
    # a small param then a medium (21..50) one: the medium does NOT fuse
    assert _partition(_model(3, 40)) == [
        (("p0",), [(0, 3)], "dense"), (("p1",), [(3, 43)], "dense")]


def test_partition_fuses_adjacent_low_dim_into_one_slice():
    # adjacent low-dim params fuse and their slices coalesce to a single contiguous slice
    assert _partition(_model(1, 1, 1)) == [(("p0", "p1", "p2"), [(0, 3)], "dense")]


def test_partition_fuses_non_adjacent_low_dim():
    # low-dim params separated by a large block fuse into ONE dense block with two slices
    part = _partition(_model(1, 60, 1))
    assert (("p1",), [(1, 61)], "lowrank") in part                  # dim-60 -> low-rank mass
    assert (("p0", "p2"), [(0, 1), (61, 62)], "dense") in part      # non-contiguous fused block


def test_partition_fusion_caps_at_the_limit():
    # 25 scalars: fusion is capped at FUSE_MIN_DIM (20), never grown past it, so the first block is
    # exactly 20 (not 21) and the remainder splits off.
    blocks = analyze(_model(*([1] * 25))).blocks
    assert [(b.coord_slices, b.kind) for b in blocks] == [
        ([(0, 20)], "dense"), ([(20, 25)], "dense")]


def test_partition_cap_keeps_a_full_vector_as_its_own_block():
    # a scalar then a cap-sized (20) vector: the vector must NOT be fused onto the scalar (which
    # would make a 21-dim block ineligible for a learned metric); it stays its own single-param
    # block, and the lone leading scalar falls back to diagonal.
    assert _partition(_model(1, 20)) == [
        (("p0",), [(0, 1)], "diagonal"), (("p1",), [(1, 21)], "dense")]


def test_partition_hierarchical_mix():
    # mu(1)+sigma(1) fuse to a dense block; theta(30) its own dense; x(100) low-rank
    assert _partition(_model(1, 1, 30, 100)) == [
        (("p0", "p1"), [(0, 2)], "dense"),
        (("p2",), [(2, 32)], "dense"),
        (("p3",), [(32, 132)], "lowrank")]
    assert any("block_partition" in r for r in analyze(_model(1, 1, 30, 100)).rationale)


# --- overriding the partition ------------------------------------------------ #

def test_override_forces_a_grouping_the_rule_would_not_choose():
    """Two scalars either side of a large parameter, fused on request into one scattered block."""
    assert _partition(_model(1, 60, 1), blocks=[("p0", "p2")]) == [
        (("p0", "p2"), [(0, 1), (61, 62)], "dense"),
        (("p1",), [(1, 61)], "lowrank")]


def test_override_unfuses_what_the_rule_would_fuse():
    """The motivating case: the size heuristic fuses these three, the caller wants them apart."""
    assert _partition(_model(1, 1, 1)) == [(("p0", "p1", "p2"), [(0, 3)], "dense")]
    assert _partition(_model(1, 1, 1), blocks=["p0", "p1", "p2"]) == [
        (("p0",), [(0, 1)], "diagonal"),
        (("p1",), [(1, 2)], "diagonal"),
        (("p2",), [(2, 3)], "diagonal")]


def test_override_is_partial_and_the_rest_follows_the_rule():
    """Unnamed parameters are partitioned exactly as if the override were not there."""
    model = _model(1, 1, 30, 100)
    forced = _partition(model, blocks=[("p0",)])
    assert (("p0",), [(0, 1)], "diagonal") in forced
    # p1 is now alone, but p2 and p3 are untouched by the override
    default = dict(((tuple(n), tuple(map(tuple, sl))), k) for n, sl, k in _partition(model))
    for names, slices, kind in forced:
        if names in (("p2",), ("p3",)):
            assert default[(names, tuple(map(tuple, slices)))] == kind


def test_override_accepts_a_bare_string_for_a_single_parameter():
    assert _partition(_model(3, 3), blocks=["p0"]) == _partition(_model(3, 3), blocks=[("p0",)])


def test_override_puts_each_group_in_declaration_order():
    """``block.names`` drives the coordinate order and the kinetic id, so it is not just a label."""
    part = _partition(_model(1, 1, 1), blocks=[("p2", "p0")])
    assert part[0][0] == ("p0", "p2")


def test_override_leaves_the_kind_to_the_refinement_rules():
    """Only the grouping is fixed --- a forced big block is still upgraded to low-rank."""
    d = LOWRANK_MAX_DIM - 10
    assert _partition(_model(d), blocks=["p0"]) == [(("p0",), [(0, d)], "lowrank")]


@pytest.mark.parametrize("blocks, match", [
    ([("p0", "nope")], "not a parameter of this model"),
    ([("p0", "p1"), ("p1",)], "already in blocks\\[0\\]"),
    ([()], "is empty"),
])
def test_override_rejects_a_bad_partition(blocks, match):
    with pytest.raises(ValueError, match=match):
        analyze(_model(1, 1), blocks=blocks)


def test_override_records_itself_in_the_rationale():
    spec = analyze(_model(1, 1, 1), blocks=[("p0", "p2")])
    assert any("fixed by the caller" in r for r in spec.rationale)


def test_override_makes_a_fused_parameter_eligible_for_a_learned_metric():
    """The whole point of the override, on the case that motivated it.

    ``learned_metric_rule`` skips any block that is not a single contiguous parameter, so a
    fusion decided by size alone can silently rule out the funnel-whitening metric ``x`` needs.
    Splitting the block is what puts it back on the table --- and nothing else about the analysis
    changes.
    """
    from mimcs.testing import neal_funnel_blocks, nuts
    problem = neal_funnel_blocks(dim=6, scale=3.0)
    model = problem.model

    # by default v(1) and x(5) fuse into one dense block -- not a learned-metric candidate
    assert _partition(model) == [(("v", "x"), [(0, 6)], "dense")]

    pilot = nuts()(model, 0)
    pilot.warmup(500)
    pilot.sample(600)

    fused = analyze(model, pilot)
    assert len(fused.blocks) == 1                                # still one fused block ...
    assert not any(b.kind == "learned_metric" for b in fused.blocks)   # ... so nothing learned

    split = analyze(model, pilot, blocks=["v", "x"])
    assert [tuple(b.names) for b in split.blocks] == [("v",), ("x",)]
    assert any(b.kind == "learned_metric" for b in split.blocks), \
        [b.kind for b in split.blocks]


def test_override_samples_correctly_end_to_end():
    """A forced scattered block must lower to a working kinetic, not just a plausible spec."""
    problem = _gaussian([1.0, -2.0], [[2.0, 1.4], [1.4, 1.5]])
    sampler = make_sampler(problem.model, blocks=["x"], seed=0)
    sampler.warmup(1500)
    draws = sampler.sample(4000)["x"]
    assert np.all(np.abs(draws.mean(axis=0) - np.array([1.0, -2.0])) < 0.2)


# --- the weighted arbiter ---------------------------------------------------- #

def test_arbiter_highest_weight_wins_and_records_losers():
    spec = default_spec(_gaussian([0.0, 0.0], np.eye(2)).model)
    spec.rationale.clear()
    arbitrate(spec, [
        Proposal("base", "hmc", 0.5, "weaker", "ra"),
        Proposal("base", "randomized_hmc", 0.8, "stronger", "rb")])
    assert spec.base == "randomized_hmc"
    line = spec.rationale[-1]
    assert "randomized_hmc" in line and "over" in line and "hmc" in line


def test_arbiter_ties_break_by_registration_order():
    spec = default_spec(_gaussian([0.0, 0.0], np.eye(2)).model)
    arbitrate(spec, [
        Proposal("base", "hmc", 0.5, "first", "ra"),
        Proposal("base", "randomized_hmc", 0.5, "second", "rb")])
    assert spec.base == "hmc"                                     # first max wins


# --- spec -> sampler lowering (a list of kinetics) --------------------------- #

def test_build_multi_block_kinetics_and_mixins():
    s = make_sampler(_model(2, 60))                              # p0 dense-2, p1 low-rank-60
    assert [(k.id, type(k).__name__, k.slices) for k in s.kinetics] == [
        ("p0", "DenseQuadraticKinetic", [(0, 2)]),
        ("p1", "LowRankQuadraticKinetic", [(2, 62)])]
    mro = type(s).__mro__
    assert NUTS in mro and RobbinsMonroStepSize in mro
    assert ScoreMassAdaptation in mro and RobustCenteringAdaptation not in mro   # centering opt-in
    assert LowRankAdaptation in mro                              # pulled in by the low-rank block
    assert set(s.state.ham_params) == {"p0", "p1"}


def test_factory_score_mass_and_lowrank_adapt_their_blocks():
    """The factory partitions a small param to a dense score-mass block and a mid-sized one to a
    low-rank block; ScoreMassAdaptation and LowRankAdaptation each adapt only their own blocks
    (partitioned by mass_mode), and every block's mass is written to the state."""
    s = make_sampler(_model(2, 60))                            # p0 dense-2, p1 low-rank-60
    s.warmup(120)
    assert set(s._sm_blocks) == {"p0"} and s._sm_blocks["p0"].mode == "dense"
    assert set(s._lr_blocks) == {"p1"}                         # the low-rank block is LowRank's
    assert set(s.state.ham_params) == {"p0", "p1"}            # each block's mass is written


def test_build_fused_block_kinetic_id():
    s = make_sampler(_model(1, 1))                              # p0+p1 fused (size 2) -> dense
    assert len(s.kinetics) == 1
    assert s.kinetics[0].id == "p0__p1" and s.kinetics[0].mass_mode == "dense"


def test_override_spec_before_build():
    spec = analyze(_gaussian([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]]).model)
    assert spec.blocks[0].kind == "dense"                        # low-dim -> dense by partition
    spec.blocks[0].kind = "diagonal"                             # advanced override
    spec.base = "hmc"
    s = spec.build()
    assert isinstance(s, HMC) and s.kinetics[0].mass_mode == "diagonal"


def test_relativistic_kind_not_implemented():
    spec = analyze(_gaussian([0.0, 0.0], np.eye(2)).model)
    spec.blocks[0].kind = "relativistic"
    with pytest.raises(NotImplementedError, match="stage 2"):
        spec.build()


# --- end-to-end: the produced samplers sample correctly ---------------------- #

def test_default_samples_correctly():
    """make_sampler(model) with no results -> a single dense block -> correct draws."""
    prob = _gaussian([0.5, -1.0, 2.0], np.diag([1.5, 0.7, 2.0]))   # 3-D -> one dense block

    def builder(model, seed):
        return make_sampler(model, seed=seed)

    report = evaluate(prob, {"factory": builder},
                      n_warmup=2000, n_samples=8000, seed=0, make_plots=False)
    print("\n" + report.summary())
    report.assert_correct()


def test_multi_block_partition_samples_correctly():
    """A hierarchical shape: a small correlated block 'a' (dense) beside a mid-sized block 'b'
    (low-rank, dim 55) -> the partition yields two kinetics, and the sampler is correct (the
    low-rank mass handles the diagonal 'b' too -- its rank part goes inert)."""
    prob = block_gaussian(var_b=np.linspace(0.5, 2.0, 55), mean_b=np.zeros(55))

    def builder(model, seed):
        return make_sampler(model, seed=seed)

    s = builder(prob.model, 0)
    assert [(type(k).__name__, k.slices) for k in s.kinetics] == [
        ("DenseQuadraticKinetic", [(0, 2)]), ("LowRankQuadraticKinetic", [(2, 57)])]

    report = evaluate(prob, {"factory": builder},
                      n_warmup=2000, n_samples=8000, seed=0, make_plots=False)
    print("\n" + report.summary())
    report.assert_correct()


def test_noncontiguous_fused_block_samples_correctly():
    """Two scalar parameters (correlated with each other) sit on either side of a large
    diagonal block in the coordinate vector. The factory fuses them into ONE dense block with
    non-contiguous slices, and the sampler is correct."""
    cov_a = np.array([[1.0, 0.8], [0.8, 1.0]])
    Pa = jnp.asarray(np.linalg.inv(cov_a))
    var_b = np.linspace(0.5, 2.0, 55)
    Pb = jnp.asarray(1.0 / var_b)

    def lp(p):
        a = jnp.array([p["a0"][0], p["a1"][0]])
        return -0.5 * a @ Pa @ a - 0.5 * jnp.sum(Pb * p["b"] ** 2)

    model = Model([EuclideanParameter("a0", (1,)), EuclideanParameter("b", (55,)),
                   EuclideanParameter("a1", (1,))], {"lp": lp})
    chol_a = np.linalg.cholesky(cov_a)

    def exact(n, rng):
        a = rng.standard_normal((n, 2)) @ chol_a.T
        b = rng.standard_normal((n, 55)) * np.sqrt(var_b)
        return np.column_stack([a[:, 0], b, a[:, 1]])            # flat order [a0, b(55), a1]

    prob = TargetProblem(name="noncontig", model=model, dim=57,
                         labels=["a0"] + [f"b{i}" for i in range(55)] + ["a1"],
                         exact_sampler=exact)

    s = make_sampler(model, seed=0)
    fused = next(k for k in s.kinetics if k.id == "a0__a1")
    assert fused.slices == [(0, 1), (56, 57)] and fused.mass_mode == "dense"    # non-contiguous

    report = evaluate(prob, {"factory": lambda m, seed: make_sampler(m, seed=seed)},
                      n_warmup=2000, n_samples=8000, seed=0, make_plots=False)
    print("\n" + report.summary())
    report.assert_correct()


# --- the integrator slot ----------------------------------------------------- #

def _split_model(cheap=("prior",)):
    """Two components, one of them declared cheap --- what the multi-rate rule wants."""
    return Model([EuclideanParameter("x", (2,))],
                 {"prior": lambda p: -0.5 * jnp.sum(p["x"] ** 2),
                  "lik": lambda p: -0.25 * jnp.sum(p["x"] ** 2)},
                 cheap_components=cheap)


def _repeated(integrator):
    from mimcs.hmc import RepeatedIntegrator
    return [op.target for op in integrator.ops if isinstance(op.target, RepeatedIntegrator)]


def test_default_integrator_is_plain_leapfrog():
    from mimcs.model import PositiveParameter
    spec = default_spec(_model(2))
    assert spec.integrator == "leapfrog" and spec.integrator_params == {}
    assert _repeated(make_sampler(_model(2)).integrator) == []
    # ... and a *constrained* model, whose chart Jacobian is cheap by construction, still gets
    # plain leapfrog: a lone Jacobian is not a reason to change every constrained model's dynamics
    constrained = Model([EuclideanParameter("x", (2,)), PositiveParameter("s")],
                        {"lp": lambda p: -0.5 * jnp.sum(p["x"] ** 2) - jnp.log(p["s"])})
    assert analyze(constrained).integrator == "leapfrog"
    assert _repeated(make_sampler(constrained).integrator) == []


def test_multirate_rule_fires_on_a_declared_split():
    from mimcs.factory.rules import MULTIRATE_DEFAULT_N
    spec = analyze(_split_model())
    assert spec.integrator == "multirate"
    assert spec.integrator_params == {"n": MULTIRATE_DEFAULT_N}
    assert any("multirate_integrator" in r for r in spec.rationale)

    sampler = spec.build()
    outer = [getattr(op.target, "id", type(op.target).__name__) for op in sampler.integrator.ops]
    assert outer == ["V_lik", "RepeatedIntegrator", "V_lik"]     # expensive kicks outside
    repeated, = _repeated(sampler.integrator)
    assert repeated.n == MULTIRATE_DEFAULT_N
    # the inner loop: the cheap kicks around the kinetic (whose id is the block's parameter name)
    assert [op.target.id for op in repeated.inner.ops] == ["V_prior", "x", "V_prior"]


def test_multirate_rule_is_inert_without_both_sides():
    assert analyze(_split_model(cheap=())).integrator == "leapfrog"            # nothing cheap
    assert analyze(_split_model(cheap=("prior", "lik"))).integrator == "leapfrog"  # nothing else


@pytest.mark.parametrize("integrator, params, match", [
    ("bogus", {}, "unknown integrator"),
    ("multirate", {"bogus": 1}, "unknown integrator_params"),
    ("line_search", {"base": "bogus"}, "unknown line-search base integrator"),
])
def test_unknown_integrator_settings_raise(integrator, params, match):
    spec = analyze(_split_model())
    spec.integrator, spec.integrator_params = integrator, params
    with pytest.raises(ValueError, match=match):
        spec.build()


def test_line_search_integrator_swaps_the_step_size_mixin():
    from mimcs.adaptation import LineSearchStepSizeAdaptation
    from mimcs.hmc import LineSearchIntegrator

    plain = make_sampler(_model(2))
    assert RobbinsMonroStepSize in type(plain).__mro__
    assert LineSearchStepSizeAdaptation not in type(plain).__mro__

    spec = analyze(_model(2))
    spec.integrator = "line_search"
    sampler = spec.build()
    assert isinstance(sampler.integrator, LineSearchIntegrator)
    # the proxy-driven mixin *replaces* the acceptance-driven one (it subclasses it, so listing
    # both would be an MRO error --- and asserting its absence here would be wrong for that reason)
    assert LineSearchStepSizeAdaptation in type(sampler).__mro__


def test_line_search_can_refine_a_multirate_step():
    from mimcs.hmc import LineSearchIntegrator, MarkovianLineSearchIntegrator, doubling_schedule
    spec = analyze(_split_model())
    spec.integrator = "line_search"
    spec.integrator_params = {"base": "multirate", "base_params": {"n": 2},
                              "schedule": doubling_schedule(3)}
    sampler = spec.build()
    assert isinstance(sampler.integrator, LineSearchIntegrator)
    repeated, = _repeated(sampler.integrator.base)
    assert repeated.n == 2

    spec.integrator = "markovian_line_search"
    spec.integrator_params["p"] = 0.5
    markovian = spec.build()
    assert isinstance(markovian.integrator, MarkovianLineSearchIntegrator)
    assert markovian.integrator.n_rng_per_step == 3          # NUTS declares the per-leaf draw
    markovian.warmup(30)
    markovian.sample(30)
    assert np.all(np.isfinite(markovian.get_samples_flat()))


# --- parallel-tempered bases (doc 13) ----------------------------------------- #

@pytest.mark.parametrize("base", ["pt_nuts", "pt_hmc", "pt_randomized_hmc"])
@pytest.mark.parametrize("integrator", ["leapfrog", "line_search"])
def test_every_base_has_a_parallel_tempered_counterpart(base, integrator):
    """A `pt_` base runs the same algorithm over the product space and keeps the cold chain, so
    every base x integrator combination the factory allows must build and sample."""
    model = P.correlated_gaussian(mean=[1.0, -2.0], cov=[[2.0, 1.4], [1.4, 1.5]]).model
    spec = analyze(model)
    spec.base, spec.integrator = base, integrator
    if integrator == "line_search":
        spec.integrator_params = {"error_thresholds": 0.8}
    spec.tempering_params = {"n_temperatures": 3, "beta_min": 0.1}
    s = spec.build(seed=0)
    # The full user-facing lifecycle, `initialize()` included: the factory composes `UniformInit`
    # and `StepSizeLineSearch` into every sampler it builds, and both are inert unless it is
    # called --- which is why a tempered sampler could be built, warmed up and sampled while
    # `initialize()` still raised.
    s.initialize()
    s.warmup(150)
    draws = np.asarray(s.sample(150)["x"])
    assert draws.shape == (150, 2)                       # the *cold* chain, not the product
    assert np.all(np.isfinite(draws))
    assert np.asarray(s.betas).shape == (3,)
    summary = s.summary()                                # evaluated against the cold chain
    assert len(summary.mean) == model.ambient_dim


def test_the_tempered_factory_path_splits_the_adaptations():
    """The mass adaptations belong to each temperature (a hot rung is wider); the step size,
    termination and initialization are global. That split is why PT keeps K adaptation hosts."""
    model = P.correlated_gaussian().model
    spec = analyze(model)
    spec.base = "pt_nuts"
    spec.tempering_params = {"n_temperatures": 3, "beta_min": 0.1}
    s = spec.build(seed=0)

    assert "ScoreMassAdaptation" in {c.__name__ for c in s._adapt_hosts[0].__class__.__mro__}
    assert len(s._adapt_hosts) == 3                       # one host per temperature
    # ... and the global ones are on the product chain itself, not in the hosts.
    chain = {c.__name__ for c in type(s).__mro__}
    assert "RobbinsMonroStepSize" in chain and "ClassifierTermination" in chain
    assert "ScoreMassAdaptation" not in chain


def test_a_tempered_base_runs_the_covariance_mass_per_temperature():
    """`MassMatrixAdaptation` must be classified per-temperature, not global.

    Left out of `_PER_TEMPERATURE_ADAPTATIONS` it lands on the product chain instead: a
    `ProductKinetic` copies `mass_mode` and `slices` from the block it wraps, so the mixin still
    passes its own filter there and gathers rung 0's coordinates. This model's dense block then
    raises a bare `Incompatible shapes for broadcasting` naming nothing about tempering; a
    diagonal block's `(n,)` update can instead broadcast against the `(K, n)` parameters with no
    error at all, every rung quietly getting the cold rung's mass. The `not in chain` assertion
    catches both before either can happen."""
    spec = analyze(P.correlated_gaussian().model)
    spec.base, spec.mass_adapt = "pt_nuts", "covariance"
    spec.tempering_params = {"n_temperatures": 3, "beta_min": 0.1}
    s = spec.build(seed=0)

    assert "MassMatrixAdaptation" in {c.__name__ for c in s._adapt_hosts[0].__class__.__mro__}
    assert "MassMatrixAdaptation" not in {c.__name__ for c in type(s).__mro__}

    s.warmup(200)                                         # > mass_min_samples
    params = np.asarray(jax.tree.leaves(s.state.ham_params[s.kinetics[0].id])[0])
    assert params.shape[0] == 3                           # one mass per rung ...
    assert not np.allclose(params[0], params[-1])         # ... fitted to its own width


def test_the_tempered_factory_path_scales_both_k_dependent_budgets():
    """Two budgets are per-Hamiltonian and must scale with K: the line search's energy-error
    threshold and NUTS's divergence threshold (doc 13)."""
    from mimcs.hmc.nuts import DEFAULT_DIVERGENCE_THRESHOLD
    model = P.correlated_gaussian().model
    spec = analyze(model)
    spec.base, spec.integrator = "pt_nuts", "line_search"
    spec.integrator_params = {"error_thresholds": 0.7}
    spec.tempering_params = {"n_temperatures": 4, "beta_min": 0.1}
    s = spec.build(seed=0)
    assert np.allclose(np.asarray(s.integrator._thresholds), 0.7 * 4)
    assert s.divergence_threshold == DEFAULT_DIVERGENCE_THRESHOLD * 4


def test_the_two_tempered_line_search_paths_share_one_default_budget():
    """`product_line_search` and the factory's `pt_` bases are two entry points to the same
    integrator, and the K-scaled budget is stated in neither: both defer to
    `mimcs.pt.product_error_thresholds`. Left duplicated, the default is exactly what would drift
    --- a caller passing no `error_thresholds` would get a different budget from each path."""
    from mimcs.hmc import DEFAULT_ERROR_THRESHOLDS
    from mimcs.pt import parallel_tempering, product_line_search

    model = P.correlated_gaussian().model
    spec = analyze(model)
    spec.base, spec.integrator = "pt_nuts", "line_search"
    spec.tempering_params = {"n_temperatures": 3, "beta_min": 0.1}       # no error_thresholds
    from_factory = np.asarray(spec.build(seed=0).integrator._thresholds)

    direct = parallel_tempering(model, n_temperatures=3, beta_min=0.1, seed=0,
                                integrator=product_line_search())        # no error_thresholds
    assert np.allclose(from_factory, np.asarray(direct.integrator._thresholds))
    assert np.allclose(from_factory, DEFAULT_ERROR_THRESHOLDS * 3)


def test_tempering_params_are_validated():
    model = P.correlated_gaussian().model
    spec = analyze(model)
    spec.base = "pt_nuts"
    spec.tempering_params = {"n_temperatues": 3}          # typo
    with pytest.raises(ValueError, match="unknown tempering_params"):
        spec.build(seed=0)

    spec2 = analyze(model)                                 # tempering options on a plain base
    spec2.tempering_params = {"n_temperatures": 3}
    with pytest.raises(ValueError, match="not tempered"):
        spec2.build(seed=0)

    spec3 = analyze(model)
    spec3.base = "pt_nonsense"
    with pytest.raises(ValueError, match="unknown base"):
        spec3.build(seed=0)


def test_a_tempered_base_rejects_what_it_cannot_do():
    """The refusal is at build time, not at the first transition.

    Centering is the one remaining combination: a chart's (mu, sigma) is shared by every
    temperature, so it is neither a per-rung quantity nor well defined from the product
    coordinate. Learned metrics *are* supported --- see the test below.
    """
    model = P.correlated_gaussian().model
    spec = analyze(model)
    spec.base, spec.centering = "pt_nuts", True
    with pytest.raises(NotImplementedError, match="centering is not supported"):
        spec.build(seed=0)


def test_a_tempered_base_takes_a_learned_metric_block():
    """Learned metrics are part of the factory's standard repertoire, so tempering must not cost
    them. Each rung carries its own metric parameters (a leading K axis) and adapts them in its
    own host --- a hot rung's funnel is shallower, so it wants a different metric."""
    funnel = P.neal_funnel_blocks(dim=3, scale=3.0)
    spec = analyze(funnel.model, blocks=["v", "x"])
    spec.base = "pt_nuts"
    for b in spec.blocks:
        if b.names == ["x"]:
            b.kind, b.params = "learned_metric", {"metric": Exp("v")}
    assert any(b.kind == "learned_metric" for b in spec.blocks)   # else the test is vacuous
    spec.tempering_params = {"n_temperatures": 3, "beta_min": 0.5}

    s = spec.build(seed=0)
    assert len(s._adapt_hosts) == 3
    assert "MetricAdaptation" in {c.__name__ for c in s._adapt_hosts[0].__class__.__mro__}
    s.warmup(300)
    draws = np.asarray(s.sample(300)["v"]).ravel()
    assert np.all(np.isfinite(draws)) and len(np.unique(draws)) > 100
    # every metric parameter carries the temperature axis
    leaves = jax.tree.leaves(s.state.ham_params["x"])
    assert leaves and all(np.shape(leaf)[0] == 3 for leaf in leaves), [
        np.shape(leaf) for leaf in leaves]
