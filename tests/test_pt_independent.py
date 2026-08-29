"""Independent per-temperature selection in PT-NUTS (``selection="independent"``).

The failure this path can produce quietly is a **biased beta=1 marginal**, and no convergence
diagnostic would reveal it. Worse, the bias is symmetric in the sample, so it moves the *variance*
and leaves the *mean* clean: on the 1-d reference study the invalid summed rule showed z_mean
+0.14 against z_var -90 (``tests/experiments/writeups/pt_independent_bias.md``). So the
load-bearing tests here compare against a known answer, and on second moments.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from mimcs.adaptation import MassMatrixAdaptation, RobbinsMonroStepSize
from mimcs.hmc import NUTS, SimpleNUTS
from mimcs.hmc.nuts import DEFAULT_DIVERGENCE_THRESHOLD
from mimcs.pt import parallel_tempering
from mimcs.pt.integrators import product_line_search
from mimcs.pt.nuts import PerTemperatureNUTSMixin, PerTemperatureSimpleNUTSMixin
from mimcs.testing import block_gaussian, correlated_gaussian, evaluate, positive_lognormal


def _pt(sel, **kw):
    def build(model, seed):
        return parallel_tempering(model, seed=seed, selection=sel,
                                  extra_mixins=(RobbinsMonroStepSize,),
                                  adapt_mixins=(MassMatrixAdaptation,), **kw)
    return build


# --- oracles ------------------------------------------------------------------ #

def test_the_checkpointed_builder_matches_the_full_buffer_oracle():
    """``NUTS`` vs ``SimpleNUTS`` for the per-lane path, the analogue of `test_nuts.py`'s oracle.

    This is what catches a mistake in the lane-axis checkpoint bookkeeping (`ckpt_velocity` /
    `ckpt_cumpsum` and the `_ntz` level bounds), which nothing statistical would localize.
    """
    draws = {}
    for base in (NUTS, SimpleNUTS):
        s = parallel_tempering(correlated_gaussian().model, n_temperatures=3, seed=0, base=base,
                               selection="independent", max_tree_depth=6, step_size=0.4,
                               metric="dense")
        s.warmup(200)
        draws[base.__name__] = np.asarray(s.sample(600)["x"])
    assert np.array_equal(draws["NUTS"], draws["SimpleNUTS"]), (
        "the checkpointed per-lane subtree builder diverged from the full-buffer oracle")


def test_the_right_builder_is_chosen_for_each_base():
    for base, mixin in ((NUTS, PerTemperatureNUTSMixin),
                        (SimpleNUTS, PerTemperatureSimpleNUTSMixin)):
        s = parallel_tempering(correlated_gaussian().model, n_temperatures=2, seed=0, base=base,
                               selection="independent")
        assert isinstance(s, mixin), (base, type(s).__mro__)


def test_one_temperature_reduces_to_plain_nuts():
    """K=1 leaves one lane, so "any lane turns" is just that lane's own U-turn.

    Statistical rather than draw-for-draw: ``leaf_select`` is stored flat here (``(2^J-1, K)``
    rather than the base's ``(J, 2^(J-1))``) to keep the RNG buffer small, so the seed stream
    differs from plain NUTS by construction.
    """
    model = correlated_gaussian().model
    pt = parallel_tempering(model, n_temperatures=1, seed=0, selection="independent")
    pt.warmup(400)
    a = np.asarray(pt.sample(2000)["x"])
    plain = NUTS(model, np.asarray(model.default_sample()), seed=0)
    plain.warmup(400)
    b = np.asarray(plain.sample(2000)["x"])

    assert np.abs(a.mean(0) - b.mean(0)).max() < 0.15, (a.mean(0), b.mean(0))
    assert np.allclose(a.std(0), b.std(0), rtol=0.10), (a.std(0), b.std(0))
    d_pt = np.mean(pt.diagnostics("sampling")["tree_depth"])
    d_pl = np.mean(plain.diagnostics("sampling")["tree_depth"])
    assert abs(d_pt - d_pl) < 0.3, f"K=1 tree depth {d_pt:.2f} vs plain NUTS {d_pl:.2f}"


def test_any_lane_turning_stops_the_trajectory():
    """The combiner is over per-lane *verdicts*, never over a summed quantity.

    Summing across lanes is what breaks reversibility once the directions are decoupled (see
    :mod:`mimcs.pt.nuts`), so this pins the semantics directly rather than through a moment.
    """
    import jax.numpy as jnp
    from mimcs.hmc.state import IntegratorState

    s = parallel_tempering(correlated_gaussian().model, n_temperatures=3, seed=0,
                           selection="independent")
    ctx = s.context(s.state)
    n, K = s.model.base.coord_dim, s.model.n_temperatures

    def state(p):
        return IntegratorState(q=jnp.zeros(K * n), p=jnp.asarray(p, float),
                               potential_values={}, potential_grads={},
                               log_weight=jnp.zeros(()), integrator_data={})

    # unit mass: velocity == momentum, so rho . v = |p|^2 > 0 with every lane aligned
    aligned = np.ones(K * n)
    assert not bool(s._any_lane_turns(state(aligned), state(aligned), jnp.asarray(aligned), ctx))

    # flip only lane 1's momentum against the accumulated rho: that lane alone turns
    one_bad = np.ones(K * n)
    one_bad[n:2 * n] = -1.0
    assert bool(s._any_lane_turns(state(one_bad), state(one_bad), jnp.asarray(aligned), ctx)), (
        "a single turning lane must stop the whole trajectory")


# --- construction guards ------------------------------------------------------- #

def test_auto_prefers_per_lane_selection_but_yields_to_a_coupled_integrator():
    """``"auto"`` is independent for NUTS -- it wins on every target measured (doc 13) -- except
    where the integrator couples the temperatures, which independent selection cannot use."""
    model = correlated_gaussian().model
    plain = parallel_tempering(model, n_temperatures=3, seed=0)
    assert isinstance(plain, PerTemperatureNUTSMixin), "auto should pick independent by default"

    with_ls = parallel_tempering(model, n_temperatures=3, seed=0,
                                 integrator=product_line_search())
    assert not isinstance(with_ls, PerTemperatureNUTSMixin), (
        "auto must fall back to joint when a line search couples the temperatures")
    # and the fallback is a fallback, not a silent downgrade of an explicit request
    assert parallel_tempering(model, n_temperatures=3, seed=0,
                              selection="joint").divergence_threshold == \
        DEFAULT_DIVERGENCE_THRESHOLD * 3


def test_a_coupled_integrator_is_refused():
    """A line search picks one refinement level from the *summed* Hamiltonian, so a lane's step
    would depend on the other lanes -- the coupling per-lane selection exists to remove."""
    with pytest.raises(ValueError, match="summed product Hamiltonian"):
        parallel_tempering(correlated_gaussian().model, n_temperatures=3, seed=0,
                           selection="independent", integrator=product_line_search())
    # the same integrator is fine on the joint path
    parallel_tempering(correlated_gaussian().model, n_temperatures=3, seed=0,
                       selection="joint", integrator=product_line_search())


def test_the_divergence_threshold_drops_its_k_scaling_for_per_lane_selection():
    """Joint tests the range of a sum of K Hamiltonians; per-lane tests one Hamiltonian's."""
    model = correlated_gaussian().model
    joint = parallel_tempering(model, n_temperatures=6, seed=0, selection="joint")
    indep = parallel_tempering(model, n_temperatures=6, seed=0, selection="independent")
    assert joint.divergence_threshold == DEFAULT_DIVERGENCE_THRESHOLD * 6
    assert indep.divergence_threshold == DEFAULT_DIVERGENCE_THRESHOLD
    explicit = parallel_tempering(model, n_temperatures=6, seed=0, selection="independent",
                                  divergence_threshold=12.0)
    assert explicit.divergence_threshold == 12.0


def test_an_unknown_selection_is_rejected():
    with pytest.raises(ValueError, match="selection must be"):
        parallel_tempering(correlated_gaussian().model, n_temperatures=2, selection="per-lane")


def test_the_step_size_stays_a_single_global_number():
    """v1 holds the step size global, so an A/B against joint isolates the selection rule.

    ``accept_prob`` must therefore stay a **scalar**: it is what
    :class:`~mimcs.adaptation.RobbinsMonroStepSize` reads, and a ``(K,)`` signal would broadcast
    into a ``(K,)`` step size and switch on per-rung steps as a silent side effect. When it did,
    the step ran to ~1.28 against the joint path's 0.765 and every trajectory U-turned at the
    first doubling.
    """
    s = parallel_tempering(correlated_gaussian().model, n_temperatures=4, seed=0,
                           selection="independent", extra_mixins=(RobbinsMonroStepSize,),
                           adapt_mixins=(MassMatrixAdaptation,))
    s.initialize().warmup(300)
    s.sample(200)
    assert np.ndim(s.state.step_size) == 0, f"step size became {np.shape(s.state.step_size)}"
    d = s.diagnostics("sampling")
    assert np.asarray(d["accept_prob"]).ndim == 1, "accept_prob must stay per-iteration scalar"
    assert np.asarray(d["accepted_lanes"]).shape[1] == 4, "per-lane movement is kept as a diagnostic"
    assert np.asarray(d["tree_depth"]).ndim == 1, "one trajectory shape for every lane"


# --- the load-bearing exactness checks ----------------------------------------- #

def test_the_cold_chain_samples_the_target_gaussian(artifacts_dir):
    problem = correlated_gaussian(mean=[1.0, -2.0], cov=[[2.0, 1.4], [1.4, 1.5]])
    report = evaluate(problem, {"independent": _pt("independent", n_temperatures=4, beta_min=0.05)},
                      n_warmup=2000, n_samples=20000, seed=0,
                      out_dir=artifacts_dir / "pt_independent_gaussian")
    print("\n" + report.summary())
    report.assert_correct()


def test_the_cold_chain_samples_a_constrained_target(artifacts_dir):
    """A bounded parameter, so the chart Jacobian is in play -- and must not be tempered."""
    problem = positive_lognormal(sigma=1.0)
    report = evaluate(problem, {"independent": _pt("independent", n_temperatures=4, beta_min=0.05)},
                      n_warmup=2000, n_samples=20000, seed=0,
                      out_dir=artifacts_dir / "pt_independent_lognormal")
    print("\n" + report.summary())
    report.assert_correct()


def test_a_wide_ladder_still_samples_the_target(artifacts_dir):
    """K=8 over a wide beta range maximizes lane heterogeneity, where a combiner error would bite."""
    problem = correlated_gaussian(mean=[1.0, -2.0], cov=[[2.0, 1.4], [1.4, 1.5]])
    report = evaluate(problem, {"independent": _pt("independent", n_temperatures=8, beta_min=0.005)},
                      n_warmup=2000, n_samples=20000, seed=0,
                      out_dir=artifacts_dir / "pt_independent_wide")
    print("\n" + report.summary())
    report.assert_correct()

# --- the matched-draw oracles -------------------------------------------------- #

def _matched_draws(n, J, K, n_iters, seed=0):
    """Identical direction/selection draws for every lane, and their plain-NUTS counterparts.

    ``leaf_select`` is flat in both --- ``(2^J-1, K)`` here and ``(2^J-1,)`` for the base, row
    ``2^j-1+i`` for leaf ``i`` of a depth-``j`` subtree. It used to need translating, because the
    base stored the same numbers rectangularly as ``(J, 2^(J-1))``; the base now uses the flat
    layout this file has always used, so the lanes and the base index it identically.
    """
    rng = np.random.default_rng(seed)
    prod, plain = [], []
    for _ in range(n_iters + 5):
        mom, direction = rng.normal(size=n), rng.random(J)
        select, flat = rng.random(J), rng.random((1 << J) - 1)
        prod.append({"T_momentum": np.stack([mom] * K),
                     "tree_direction": np.stack([direction] * K, axis=1),
                     "tree_select": np.stack([select] * K, axis=1),
                     "leaf_select": np.stack([flat] * K, axis=1)})
        plain.append({"T_momentum": mom, "tree_direction": direction,
                      "tree_select": select, "leaf_select": flat})
    return prod, plain


class _StubBuffer:
    def __init__(self, seq):
        self.seq, self.i = seq, 0

    def next(self):
        d = self.seq[self.i]
        self.i += 1
        return {k: jnp.asarray(v) for k, v in d.items()}


def _product_sampler(model, betas, J, eps):
    """A per-lane PT-NUTS composed directly, so ``betas`` need not be strictly decreasing."""
    from mimcs.hmc.samplers import make_kinetic
    from mimcs.pt.kinetics import build_product_kinetics
    from mimcs.pt.product import ProductSpaceMixin
    from mimcs.pt.tempering import ProductModel, build_tempered_potentials
    from mimcs.samplers.base import make_sampler_class

    K = len(betas)
    n = model.coord_dim
    Cls = make_sampler_class(PerTemperatureSimpleNUTSMixin, ProductSpaceMixin, SimpleNUTS,
                             name="MatchedPT")
    return Cls(ProductModel(model, K),
               np.tile(np.asarray(model.default_sample(), float), K),
               potentials=build_tempered_potentials(model, jnp.asarray(betas)),
               kinetics=build_product_kinetics([make_kinetic("diagonal")], K, n),
               step_size=eps, seed=0, max_tree_depth=J)


def _step_both(prod, plain, n, n_iters):
    """Drive both samplers off matched draws; return lane-0 and standalone coordinate histories."""
    a, b, depths, leaves = [], [], [], []
    for _ in range(n_iters):
        prod.state = prod.postprocess(prod.kernel(prod.preprocess(prod.state)))
        plain.state = plain.postprocess(plain.kernel(plain.preprocess(plain.state)))
        coord = np.asarray(prod.state.coordinate)
        assert np.array_equal(coord[:n], coord[n:2 * n]), "identical lanes drifted apart"
        a.append(coord[:n])
        b.append(np.asarray(plain.state.coordinate))
        depths.append((int(prod.state.diagnostics["tree_depth"]),
                       int(plain.state.diagnostics["tree_depth"])))
        leaves.append((int(prod.state.diagnostics["n_leaves"]),
                       int(plain.state.diagnostics["n_leaves"])))
    return np.array(a), np.array(b), depths, leaves


def test_identical_lanes_reproduce_plain_nuts_draw_for_draw():
    """K=2 at ``betas=[1,1]`` with matched draws: every lane *is* ordinary NUTS on the model.

    The single strongest implementation check --- it exercises per-lane direction handling, the
    combined verdict, per-lane selection and the flat ``leaf_select`` re-indexing at once, and the
    discrete ``tree_depth`` / ``n_leaves`` sequences cannot be fudged by float noise.
    """
    J, K, n_iters, eps = 5, 2, 60, 0.4
    model = correlated_gaussian().model
    n = model.coord_dim
    prod_seq, plain_seq = _matched_draws(n, J, K, n_iters)

    prod = _product_sampler(model, [1.0, 1.0], J, eps)
    plain = SimpleNUTS(model, np.asarray(model.default_sample(), float),
                       step_size=eps, seed=0, max_tree_depth=J)
    prod._rng_buffer, plain._rng_buffer = _StubBuffer(prod_seq), _StubBuffer(plain_seq)

    a, b, depths, leaves = _step_both(prod, plain, n, n_iters)
    assert np.array_equal(a, b), f"lane 0 diverged from plain NUTS (max {np.abs(a - b).max():.2e})"
    assert all(x == y for x, y in depths), depths
    assert all(x == y for x, y in leaves), leaves
    assert len({x for x, _ in depths}) > 1, "every trajectory was the same depth --- vacuous"


def test_the_lane_weighting_is_the_tempered_density():
    """The same oracle at ``betas=[0.5, 0.5]``, against a model whose log-density *is* halved.

    Catches a beta-weighting error --- applied twice, or not at all --- that the ``betas=[1,1]``
    case cannot see, since there ``beta`` is the identity.
    """
    from mimcs.model import EuclideanParameter, Model

    J, K, n_iters, eps = 5, 2, 60, 0.4
    prec = np.array([[1.3, -0.4], [-0.4, 0.9]])

    def make(scale):
        return Model([EuclideanParameter("x", (2,))],
                     {"V": lambda v, _s=scale, **kw: -0.5 * _s * (v["x"] @ prec @ v["x"])})

    prod_seq, plain_seq = _matched_draws(2, J, K, n_iters)
    prod = _product_sampler(make(1.0), [0.5, 0.5], J, eps)
    plain = SimpleNUTS(make(0.5), np.asarray(make(0.5).default_sample(), float),
                       step_size=eps, seed=0, max_tree_depth=J)
    prod._rng_buffer, plain._rng_buffer = _StubBuffer(prod_seq), _StubBuffer(plain_seq)

    a, b, depths, _ = _step_both(prod, plain, 2, n_iters)
    assert np.allclose(a, b, atol=1e-5), (
        f"a lane at beta=0.5 is not the halved target (max {np.abs(a - b).max():.2e})")
    assert len({x for x, _ in depths}) > 1, "every trajectory was the same depth --- vacuous"


# --- the remaining product-space paths ------------------------------------------ #

def test_a_learned_metric_block_still_samples_the_target(artifacts_dir):
    """The non-separable ``ProductKinetic.flow`` branch, the one place the signed per-lane eps is
    consumed through ``_lane_eps`` rather than elementwise."""
    from mimcs.factory import analyze
    from mimcs.hmc.metric_expr import Exp

    def build(model, seed):
        spec = analyze(model, blocks=["a", "b"])
        spec.base = "pt_nuts"
        for blk in spec.blocks:
            if blk.names == ["b"]:
                blk.kind, blk.params = "learned_metric", {"metric": Exp("a")}
        assert any(b.kind == "learned_metric" for b in spec.blocks)   # else vacuous
        spec.tempering_params = {"n_temperatures": 3, "beta_min": 0.1,
                                 "selection": "independent"}
        return spec.build(seed=seed)

    report = evaluate(block_gaussian(), {"pt_lm_indep": build}, n_warmup=1500, n_samples=8000,
                      seed=0, out_dir=str(artifacts_dir / "pt_independent_learned_metric"))
    print("\n" + report.summary())
    report.assert_correct()


def test_it_still_initializes_and_terminates_warmup(artifacts_dir):
    """Two lifecycle paths doc 13 records as having been silently broken once before."""
    from mimcs.adaptation import ClassifierTermination, UniformInit

    model = correlated_gaussian().model
    s = parallel_tempering(model, n_temperatures=3, beta_min=0.1, seed=0, selection="independent",
                           extra_mixins=(UniformInit, RobbinsMonroStepSize, ClassifierTermination),
                           adapt_mixins=(MassMatrixAdaptation,), min_warmup=200, max_warmup=1200)
    s.initialize()
    assert np.isfinite(float(s.state.log_prob))
    s.warmup()
    done = len(np.asarray(s.diagnostics("warmup")["tree_depth"]))
    assert 200 <= done <= 1200, done
    draws = np.asarray(s.sample(500)["x"])
    assert np.isfinite(draws).all()


# --- per-temperature step size --------------------------------------------------- #

def test_a_per_rung_step_size_needs_a_per_rung_acceptance_signal():
    with pytest.raises(ValueError, match="per-rung acceptance signal"):
        parallel_tempering(correlated_gaussian().model, n_temperatures=3, selection="joint",
                           per_temperature_step_size=True)


def test_the_per_rung_step_size_adapts_a_vector():
    """Independent selection gives each rung its own acceptance, which drives its own step.

    The steps come out *larger* than the single global one, and that is the point: a global step
    is tuned so the **summed** K-fold energy error meets a target calibrated for one chain, which
    demands each rung be ``target^(1/K)`` accurate. Measured on this model at K=4: global 0.434
    against per-rung 0.56-0.67, with per-rung acceptance landing on the 0.8 target.
    """
    model = correlated_gaussian(mean=[1.0, -2.0], cov=[[2.0, 1.4], [1.4, 1.5]]).model
    out = {}
    for pts in (False, True):
        s = parallel_tempering(model, n_temperatures=4, seed=0, selection="independent",
                               per_temperature_step_size=pts, target_accept=0.8,
                               extra_mixins=(RobbinsMonroStepSize,),
                               adapt_mixins=(MassMatrixAdaptation,))
        s.initialize().warmup(1000)
        s.sample(500)
        out[pts] = (np.asarray(s.state.step_size),
                    np.asarray(s.diagnostics("sampling")["accept_prob"]))
    assert out[False][0].ndim == 0, "the default must stay a single global step"
    assert out[True][0].shape == (4,), out[True][0]
    assert out[False][1].ndim == 1, "global mode: one scalar acceptance per iteration"
    assert out[True][1].shape[1] == 4, "per-rung mode: one acceptance per rung per iteration"
    assert np.all(out[True][0] > out[False][0]), (
        f"per-rung steps {out[True][0]} should exceed the global {out[False][0]}")
    assert np.abs(out[True][1].mean(axis=0) - 0.8).max() < 0.15, out[True][1].mean(axis=0)


def test_the_per_rung_step_size_still_samples_the_target(artifacts_dir):
    problem = correlated_gaussian(mean=[1.0, -2.0], cov=[[2.0, 1.4], [1.4, 1.5]])
    report = evaluate(problem, {"pts": _pt("independent", n_temperatures=4, beta_min=0.05,
                                          per_temperature_step_size=True, target_accept=0.8)},
                      n_warmup=2000, n_samples=20000, seed=0,
                      out_dir=artifacts_dir / "pt_independent_pts")
    print("\n" + report.summary())
    report.assert_correct()
