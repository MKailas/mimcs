"""Tests for dynamic adaptation: ending warmup when the chain looks to be mixing well.

Three layers, each tested on its own terms:

* **Features** --- the observables a diagnostic reads. The interesting case is the sphere, where
  the generic ``[x, x^2]`` is rank-deficient (``sum_j x_j^2 = 1``) and the override is not.
* **The statistics** --- ``split_rhat`` against a hand-computed value and its known blind spots,
  and ``fit_logistic``, whose ridge is load-bearing rather than decorative.
* **The burn-in search** --- that it recovers a planted transient, sits at its lower bound when
  there is none, and never violates its bounds.
* **The mixins** --- that they keep warming up while the chain is still travelling, stop when it
  has settled, and leave ``warmup(n)`` alone when absent.

Seeds are fixed, so pass/fail is deterministic.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from mimcs.model import Model, EuclideanParameter, PositiveParameter, UnitVectorParameter
from mimcs.adaptation import _burnin
from mimcs.adaptation._logistic import (_fit_jit, accuracy, balanced_accuracy, class_weights,
                                       fit_logistic, log_score, row_buffer)
from mimcs.diagnostics import split_rhat
from mimcs.adaptation import ClassifierTermination, GelmanRubinTermination
from mimcs.samplers import make_sampler_class
from mimcs.testing import correlated_gaussian, von_mises_fisher, evaluate, nuts


# --- features ---------------------------------------------------------------- #

def test_features_are_the_per_coordinate_mean_and_spread():
    p = EuclideanParameter("x", (3,))
    assert p.n_features == 6
    assert p.feature_names() == ["x[0]", "x[1]", "x[2]", "x[0]^2", "x[1]^2", "x[2]^2"]
    assert np.allclose(np.asarray(p.features(jnp.array([1.0, 2.0, 3.0]))),
                       [1.0, 2.0, 3.0, 1.0, 4.0, 9.0])


def test_model_features_concatenate_and_vmap():
    model = Model([EuclideanParameter("a", (2,)), PositiveParameter("s"),
                   UnitVectorParameter("x", 3)], {"lp": lambda d: jnp.zeros(())})
    assert model.n_features == 4 + 2 + 5
    assert len(model.feature_names) == model.n_features
    draws = jnp.asarray(np.random.default_rng(0).standard_normal((7, model.ambient_dim)))
    assert jax.vmap(model.features)(draws).shape == (7, model.n_features)


def test_unit_vector_features_are_full_rank_where_the_generic_ones_are_not():
    """``sum_j x_j^2 = 1`` makes the generic ``[x, x^2]`` collinear with a fitted intercept."""
    rng = np.random.default_rng(0)
    xs = rng.standard_normal((400, 3))
    xs /= np.linalg.norm(xs, axis=1, keepdims=True)

    generic = np.column_stack([np.ones(len(xs)), xs, xs ** 2])          # 1 + 3 + 3 columns
    assert np.linalg.matrix_rank(generic) == 6 < generic.shape[1]        # rank-deficient

    p = UnitVectorParameter("q", 3)
    ours = np.asarray(jax.vmap(p.features)(jnp.asarray(xs)))
    with_intercept = np.column_stack([np.ones(len(xs)), ours])           # 1 + 3 + 2 columns
    assert p.n_features == 5
    assert np.linalg.matrix_rank(with_intercept) == with_intercept.shape[1] == 6

    # ...and nothing was lost: the dropped x_d^2 is a linear function of what we kept.
    coef, *_ = np.linalg.lstsq(with_intercept, xs[:, 2] ** 2, rcond=None)
    assert np.abs(with_intercept @ coef - xs[:, 2] ** 2).max() < 1e-5


# --- split R-hat ------------------------------------------------------------- #

def test_split_rhat_matches_a_hand_computed_value():
    a = np.array([[1.0], [2.0], [3.0], [4.0]])
    b = np.array([[3.0], [4.0], [5.0], [6.0]])
    n, w = 4, np.mean([np.var(a, ddof=1), np.var(b, ddof=1)])
    b_over_n = ((np.array([2.5, 4.5]) - 3.5) ** 2).sum()
    want = np.sqrt(((n - 1) / n * w + b_over_n) / w)
    assert np.isclose(float(split_rhat(a, b)[0]), float(want), atol=1e-9)


def test_split_rhat_is_one_under_the_null():
    """Two halves of one sample agree, and the (n-1)/n term is what makes that come out at 1."""
    rng = np.random.default_rng(0)
    r = split_rhat(rng.standard_normal((2000, 3)), rng.standard_normal((2000, 3)))
    assert np.allclose(r, 1.0, atol=5e-3)


@pytest.mark.parametrize("delta, want", [(0.1, 1.0025), (1.0, 1.2247), (2.0, 1.7321)])
def test_split_rhat_tracks_the_mean_gap(delta, want):
    """``R-hat = sqrt(1 + delta^2/2)`` for a gap of ``delta`` sd -- so 1.1 tolerates 0.65 sd."""
    rng = np.random.default_rng(1)
    r = float(split_rhat(rng.standard_normal((4000, 1)), rng.standard_normal((4000, 1)) + delta)[0])
    assert np.isclose(r, want, atol=0.02)


def test_split_rhat_needs_the_squared_feature_to_see_a_scale_change():
    """R-hat compares means, so on a raw draw it is blind to scale -- the features fix that."""
    rng = np.random.default_rng(2)
    a = rng.standard_normal((4000, 1))
    b = rng.standard_normal((4000, 1)) * 2.0
    assert float(split_rhat(a, b)[0]) < 1.01                       # blind on x alone
    fa = np.column_stack([a, a ** 2])
    fb = np.column_stack([b, b ** 2])
    assert float(np.max(split_rhat(fa, fb))) > 1.1                 # the x^2 column sees it


def test_split_rhat_catches_a_drifting_chain():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((4000, 2)) + 2.0 * np.linspace(0, 1, 4000)[:, None]
    assert float(np.max(split_rhat(x[:2000], x[2000:]))) > 1.1


# --- logistic regression ------------------------------------------------------ #

def test_fit_logistic_recovers_known_coefficients():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((4000, 3))
    w_true = np.array([1.5, -0.8, 0.4])
    y = (rng.random(4000) < 1 / (1 + np.exp(-(X @ w_true + 0.3)))).astype(float)
    fit = fit_logistic(X, y, l2=1e-6, max_iter=200)
    assert fit.converged
    assert np.allclose(np.asarray(fit.w), w_true, atol=0.15)
    assert np.isclose(float(fit.b), 0.3, atol=0.15)


def test_unregularized_fit_on_separable_data_is_an_artefact_of_the_tolerance():
    """The case the ridge exists for -- and it fails *silently*, reporting convergence.

    Separable data has no maximum-likelihood optimum. L-BFGS does not say so: the gradient
    underflows as the coefficients grow, ``gtol`` is met, and ``converged`` comes back true at
    whatever point the tolerance happened to bite. Tightening it moves the answer.
    """
    rng = np.random.default_rng(0)
    X = np.concatenate([rng.standard_normal((200, 2)) + 4, rng.standard_normal((200, 2)) - 4])
    y = np.concatenate([np.ones(200), np.zeros(200)])

    loose = fit_logistic(X, y, l2=0.0, gtol=1e-4, max_iter=500)
    tight = fit_logistic(X, y, l2=0.0, gtol=1e-10, max_iter=500)
    assert loose.converged and tight.converged                      # both claim success...
    assert float(jnp.linalg.norm(tight.w)) > 2 * float(jnp.linalg.norm(loose.w))   # ...and differ

    a = fit_logistic(X, y, l2=1e-2, gtol=1e-6, max_iter=500)
    b = fit_logistic(X, y, l2=1e-2, gtol=1e-10, max_iter=500)
    assert np.allclose(np.asarray(a.w), np.asarray(b.w), atol=1e-5)  # a real optimum
    assert accuracy(a, X, y) == 1.0                                  # still separates them


def test_fit_logistic_warm_start_reaches_the_same_optimum():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((1000, 4))
    y = (rng.random(1000) < 1 / (1 + np.exp(-X[:, 0]))).astype(float)
    cold = fit_logistic(X, y, l2=1e-2)
    warm = fit_logistic(X, y, l2=1e-2, init=(cold.w, cold.b))
    assert np.isclose(cold.loss, warm.loss, atol=1e-8)


# --- row buffering: one compilation per power of two, not one per check --------- #

def test_row_buffer_rounds_up_to_a_power_of_two():
    assert row_buffer(65) == 128 and row_buffer(360) == 512 and row_buffer(513) == 1024
    assert row_buffer(64) == 64 and row_buffer(512) == 512     # exact at a power of two
    assert row_buffer(1) == 64 and row_buffer(63) == 64        # floored: tiny fits are cheap


def test_buffering_does_not_move_the_fit():
    """Padding is only allowed to cost arithmetic, never to move the answer.

    The padded rows carry zero weight and ``sum(wt * l) / sum(wt)`` drops them from numerator and
    denominator alike, so the only difference from an unbuffered fit is the order XLA reduces in.
    ``360`` and ``512`` rows land in the same buffer, so the two fits below are the *same*
    computation on the same data with a different amount of padding.
    """
    rng = np.random.default_rng(3)
    X = rng.standard_normal((360, 4))
    y = (X[:, 0] + 0.5 * rng.standard_normal(360) > 0).astype(float)
    fit = fit_logistic(X, y, l2=1e-2, max_iter=300)
    # the same data padded by hand to the buffer size, at zero weight: must be the same fit
    Xp = np.concatenate([X, np.zeros((152, 4))])
    yp = np.concatenate([y, np.zeros(152)])
    wt = np.concatenate([np.ones(360), np.zeros(152)])
    padded = fit_logistic(Xp, yp, l2=1e-2, max_iter=300, wt=wt)
    assert np.allclose(np.asarray(fit.w), np.asarray(padded.w), atol=1e-6)
    assert np.isclose(fit.loss, padded.loss, atol=1e-7)


def test_zero_weighted_padding_rows_are_inert():
    """The property the whole scheme rests on --- and the one that fails *silently* if the
    padding ever leaks into ``sum(wt)``: junk rows at zero weight must not move the fit.

    The junk is finite on purpose. Padded rows still flow through ``X @ w``, and ``0 * nan`` is
    ``nan``, so a non-finite pad would poison the loss however little weight it carried.
    """
    rng = np.random.default_rng(4)
    X = rng.standard_normal((200, 3))
    y = (X[:, 0] > 0).astype(float)
    clean = fit_logistic(X, y, l2=1e-2, wt=np.ones(200))
    junk = 50.0 * rng.standard_normal((100, 3))            # large, but finite
    dirty = fit_logistic(np.concatenate([X, junk]), np.concatenate([y, np.ones(100)]),
                         l2=1e-2, wt=np.concatenate([np.ones(200), np.zeros(100)]))
    assert np.allclose(np.asarray(clean.w), np.asarray(dirty.w), atol=1e-6)
    assert np.isclose(clean.loss, dirty.loss, atol=1e-7)


@pytest.mark.skipif(not hasattr(_fit_jit, "_cache_size"), reason="jax jit cache introspection")
def test_fits_inside_one_buffer_share_a_compilation():
    """The regression guard for the point of the change.

    Buffering and the cached ``jit`` are load-bearing *together*: ``minimize`` is a bare
    ``lax.while_loop``, so calling it outside ``jit`` rebuilds and re-dispatches the loop on every
    call regardless of shape. Either half alone leaves this test failing.
    """
    rng = np.random.default_rng(5)

    def fit(n):
        X = rng.standard_normal((n, 3))
        fit_logistic(X, (X[:, 0] > 0).astype(float), l2=1e-2, max_iter=20)

    fit(360)                                       # buffer 512
    before = _fit_jit._cache_size()
    fit(432)                                       # same buffer -> no new compilation
    fit(504)
    assert _fit_jit._cache_size() == before
    fit(576)                                       # buffer 1024 -> exactly one more
    assert _fit_jit._cache_size() == before + 1


def test_fit_logistic_rejects_an_unknown_keyword():
    """The optimiser hyperparameters are static to the cached fit, so they are named explicitly
    rather than swallowed by ``**opt`` --- where a typo would silently take the defaults."""
    X = np.zeros((80, 2))
    with pytest.raises(TypeError, match="max_itr"):
        fit_logistic(X, np.zeros(80), max_itr=10)


# --- a mixin with no sampler under it ------------------------------------------ #

class _HookTerminal:
    """Stands in for ``BaseSampler``'s terminal hooks, so the mixin can be tested on its own."""

    def _init_hooks(self, **kwargs):
        return None

    def should_stop(self) -> bool:
        return False


def _classifier(**kwargs):
    """A ``ClassifierTermination`` carrying only its own state (no model, no chain)."""
    cls = make_sampler_class(ClassifierTermination, _HookTerminal)
    obj = object.__new__(cls)                              # skip BaseSampler.__init__
    obj._init_hooks(**kwargs)
    return obj


# --- weights and the smooth scorers -------------------------------------------- #

def test_constant_sample_weights_leave_the_fit_alone():
    """The weighted path must be a strict generalization --- nothing calibrated may shift."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 3))
    y = (X[:, 0] + 0.5 * rng.standard_normal(200) > 0).astype(float)
    plain = fit_logistic(X, y)
    weighted = fit_logistic(X, y, wt=np.full(200, 7.0))
    assert np.isclose(plain.loss, weighted.loss, atol=1e-7)
    # Not bit-identical: sum(w*l)/sum(w) rounds differently from mean(l) in float32, which walks
    # L-BFGS along a slightly different path to the same optimum.
    assert np.allclose(plain.w, weighted.w, atol=1e-4)


def test_class_weights_zero_out_the_rows_they_are_told_to_drop():
    """How the search excludes rows without changing the design matrix's shape."""
    y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    keep = np.array([True, True, False, True, False, False])
    wt = class_weights(y, keep=keep)
    assert wt[2] == 0.0 and wt[4] == 0.0 and wt[5] == 0.0
    assert wt[[0, 1]].sum() == pytest.approx(0.5)         # each class carries half the weight
    assert wt[3] == pytest.approx(0.5)                    # ... however few rows it has left


def test_class_weights_rescue_a_badly_unbalanced_fit():
    """The burn-in search fits prefixes as short as a fortieth of the history."""
    rng = np.random.default_rng(1)
    n, n_pos = 1000, 25
    X = rng.standard_normal((n, 1))
    X[:n_pos] += 1.5
    y = np.zeros(n)
    y[:n_pos] = 1.0
    plain = fit_logistic(X, y)
    weighted = fit_logistic(X, y, wt=class_weights(y))
    # Unweighted, predicting "majority" already scores 97.5%, and that is what it does.
    assert accuracy(plain, X, y) > 0.97
    assert balanced_accuracy(plain, X, y) < 0.55
    assert balanced_accuracy(weighted, X, y) > 0.7


def test_log_score_is_zero_for_an_uninformative_fit_and_positive_for_a_separating_one():
    rng = np.random.default_rng(2)
    X = np.concatenate([rng.standard_normal((100, 1)), rng.standard_normal((100, 1)) + 3.0])
    y = np.concatenate([np.zeros(100), np.ones(100)])
    flat = fit_logistic(X, y)._replace(w=jnp.zeros(1), b=jnp.zeros(()))
    assert log_score(flat, X, y) == 0.0                      # log 2 + log(1/2) = 0
    assert log_score(fit_logistic(X, y), X, y) > 0.3
    # And it is clipped, not negative, when an over-fitted direction transfers badly.
    assert log_score(fit_logistic(X, y), X, 1.0 - y) == 0.0


# --- the burn-in search --------------------------------------------------------- #

def _planted(length, *, n=2000, shift=3.0, dim=4, seed=0):
    """``[x, x^2]`` features for a chain whose first ``length`` draws are shifted."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, dim))
    x[:length] += shift
    return np.concatenate([x, x ** 2], axis=1)


def _search(features, mode, *, null="none", **kwargs):
    """Run the search over ``features`` through the production mixin, sampler and all absent."""
    limits = _burnin.bounds(len(features), min_abs=50, min_frac=0.0, max_frac=0.5, min_tail=40)
    assert limits is not None
    lo, hi = limits
    obj = _classifier(burn_in=mode, burn_in_null=null)
    sd = features.std(0)
    obj._burn_matrix = (features - features.mean(0)) / np.where(sd > 1e-12, sd, 1.0)
    kwargs.setdefault("init", None)
    found = _burnin.estimate_burn_in(len(features), obj._burn_fit, mode=mode, lo=lo, hi=hi,
                                     null=null, **kwargs)
    return found, lo, hi


def test_changepoint_finds_a_planted_step():
    rng = np.random.default_rng(0)
    s = np.concatenate([rng.normal(3.0, 1.0, 200), rng.normal(0.0, 1.0, 800)])
    assert _burnin.changepoint(s, 50, 500) == 200


def test_changepoint_respects_its_bounds_and_survives_a_constant_series():
    rng = np.random.default_rng(0)
    s = np.concatenate([rng.normal(3.0, 1.0, 200), rng.normal(0.0, 1.0, 800)])
    assert _burnin.changepoint(s, 400, 600) == 400           # clamped, not the true 200
    assert 50 <= _burnin.changepoint(np.ones(1000), 50, 500) <= 500      # no NaN, no crash


def test_bounds_collapse_when_the_history_is_too_short():
    assert _burnin.bounds(2000, min_abs=50, min_frac=0.0, max_frac=0.5, min_tail=40) == (50, 1000)
    # min_tail binds before max_frac once the history is short enough for the two to cross.
    assert _burnin.bounds(200, min_abs=50, min_frac=0.0, max_frac=0.9, min_tail=40) == (50, 160)
    assert _burnin.bounds(100, min_abs=50, min_frac=0.0, max_frac=0.5, min_tail=40) is None


def test_ladder_is_geometric_and_ends_on_the_upper_bound():
    assert _burnin.ladder(50, 500) == [50, 100, 200, 400, 500]
    assert _burnin.ladder(50, 60) == [50, 60]


def test_null_terms_shrink_with_the_block_they_describe():
    """The ``1/k`` law: a longer tail has a smaller null, which is the whole point."""
    a_small, b_small = _burnin.null_terms(1.0, 2000, 100)      # tail 1900
    a_big, b_big = _burnin.null_terms(1.0, 2000, 900)          # tail 1100
    assert b_small < b_big                                     # longer tail => smaller null
    assert a_small > a_big                                     # ... but a shorter prefix limits A
    assert _burnin.null_terms(0.0, 2000, 100) == (0.0, 0.0)    # no reference signal, no correction


def test_earliest_below_null_falls_back_to_the_lower_bound():
    """The branch that fixes the funnel runaway: nothing qualifies => discard as little as we can.

    Under the old relative rule this case was forced to nominate a split, and it nominated the cap.
    """
    floors = dict.fromkeys([50, 100, 200, 400], 0.0)
    never = {50: 0.5, 100: 0.4, 200: 0.3, 400: 0.2}            # sliding down, never stationary
    assert _burnin._earliest_below_null(never, floors, 0.5, lo=50) == 50
    settles = {50: 0.5, 100: 0.4, 200: 0.001, 400: 0.0}
    assert _burnin._earliest_below_null(settles, floors, 0.5, lo=50) == 200


def test_the_null_floor_is_what_admits_a_stationary_tail():
    """``log_score`` clips at zero on two thirds of stationary tails, so an exact-zero gate would
    be a knife edge; the floor is calibrated above the measured stationary maximum (0.0117)."""
    floors = dict.fromkeys([50, 100], 0.0)
    tiny = {50: 0.005, 100: 0.0}                               # inside the stationary range
    assert _burnin._earliest_below_null(tiny, floors, 0.5, lo=50, null_floor=0.02) == 50
    assert _burnin._earliest_below_null(tiny, floors, 0.5, lo=50, null_floor=0.0) == 100
    assert _burnin.NULL_FLOOR > 0.0117                         # the measured stationary maximum


def test_unknown_null_is_rejected():
    with pytest.raises(ValueError, match="unknown burn_in_null"):
        _classifier(burn_in="min_discard", burn_in_null="bootstrap")


def test_earliest_flat_takes_the_first_qualifying_split_not_the_best_one():
    """The point of the rule: once the curve has flattened, discarding more buys nothing."""
    scored = {50: 1.0, 100: 0.5, 200: 0.02, 400: 0.01, 800: 0.0}
    assert _burnin._earliest_flat(scored, 0.1) == 200
    assert _burnin._earliest_flat(dict.fromkeys(scored, 0.3), 0.1) == 50     # flat => lower bound


@pytest.mark.parametrize("null", ["none", "scaled"])
@pytest.mark.parametrize("mode", ["changepoint", "min_discard"])
@pytest.mark.parametrize("length", [200, 400])
def test_burn_in_search_recovers_a_planted_transient(mode, length, null):
    found, lo, hi = _search(_planted(length), mode, null=null)
    assert lo <= found.n <= hi
    # min_discard undershoots by design: it stops as soon as what is left looks stationary, and
    # the last few dozen transient rows are too few for the tail's halves to notice.
    assert 0.6 * length <= found.n <= 1.25 * length


def test_the_null_floor_costs_sensitivity_to_a_short_transient():
    """The price of :data:`NULL_FLOOR`, measured and bounded rather than hidden.

    A transient of 100 rows in 2000 shifts the tail's halves by only ~0.004 nats at ``m = lo`` --
    under the floor -- so the corrected rule accepts ``lo`` and does not excise it, where the
    uncorrected rule finds it. The loss is confined to transients below ~10% of the history, which
    is exactly the regime the shipped fixed 10% discard already handles; above it (200 of 2000, in
    the test above) both rules agree.
    """
    loose, lo, _ = _search(_planted(100), "min_discard", null="none")
    tight, _, _ = _search(_planted(100), "min_discard", null="scaled")
    assert 60 <= loose.n <= 125, "the uncorrected rule resolves a 100-row transient"
    assert tight.n == lo, "the corrected rule does not, and settles for discarding the minimum"


@pytest.mark.parametrize("null", ["none", "scaled"])
@pytest.mark.parametrize("mode", ["changepoint", "min_discard"])
def test_burn_in_search_sits_at_the_lower_bound_with_nothing_to_find(mode, null):
    found, lo, _ = _search(_planted(0), mode, null=null)
    assert found.n == lo
    assert found.path[0] == lo and len(found.path) >= 1


def test_null_correction_refuses_to_discard_when_no_split_helps():
    """A history with no stationary tail at any split --- a drift that never settles.

    Uncorrected, ``min_discard`` is forced to nominate a rung and takes the largest; corrected, it
    recognizes that nothing qualifies and keeps the data. This is the funnel failure in miniature.
    """
    rng = np.random.default_rng(3)
    n = 2000
    x = rng.standard_normal((n, 3)) + np.linspace(0.0, 6.0, n)[:, None]     # never settles
    features = np.concatenate([x, x ** 2], axis=1)
    loose, lo, hi = _search(features, "min_discard", null="none")
    tight, _, _ = _search(features, "min_discard", null="scaled")
    assert loose.n > 0.5 * hi, "the uncorrected rule should run toward the cap here"
    assert tight.n == lo, "the corrected rule should decline to discard"


# --- the split and the mixins ------------------------------------------------- #

def test_split_discards_burn_in_and_halves_the_rest():
    obj = _classifier(burn_in_frac=0.1)
    obj._term_features = [np.full(2, float(i)) for i in range(100)]
    early, late = obj._term_split()
    assert len(early) == len(late) == 45                    # 100 - 10 burn-in, halved
    assert early[0][0] == 10.0 and late[0][0] == 55.0


def test_unknown_burn_in_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown burn_in"):
        _classifier(burn_in="elbow")


def test_dynamic_split_excises_a_transient_the_fixed_fraction_would_keep():
    """The defect this exists to fix: at 2000 draws, 10% leaves 300 transient rows in place."""
    features = _planted(500)
    fixed, dynamic = _classifier(), _classifier(burn_in="min_discard")
    for obj in (fixed, dynamic):
        obj._term_features = list(features)

    fixed._term_split()
    assert fixed._term_last_burn == 200                     # a tenth of 2000, transient and all
    dynamic._term_split()
    assert 300 <= dynamic._term_last_burn <= 625            # the transient, within the bounds


def test_dynamic_burn_in_is_cached_per_history_length():
    """``_term_split()`` is called from outside too; the search must not run again for free."""
    obj = _classifier(burn_in="min_discard")
    obj._term_features = list(_planted(500))
    obj._term_split()
    fits = len(obj._burn_last.path)
    obj._term_split()
    assert len(obj._burn_last.path) == fits                 # the same BurnIn, not a fresh search


def test_train_val_split_is_balanced_and_disjoint():
    obj = _classifier(val_every=5)
    block = np.arange(90, dtype=float)[:, None]
    x_tr, y_tr, x_va, y_va = obj._train_val(block, block + 1000.0)
    assert len(x_va) == 2 * 18                              # a fifth of each half
    assert len(x_tr) + len(x_va) == 180
    assert y_tr.mean() == 0.5 and y_va.mean() == 0.5        # balanced labels
    assert not (set(x_tr.ravel()) & set(x_va.ravel()))      # disjoint


def test_warmup_is_unchanged_without_a_termination_mixin():
    """The base-class change must be inert unless a mixin opts in."""
    sampler = nuts()(correlated_gaussian().model, 0)
    sampler.warmup(200)
    assert sampler._iteration == 200
    assert sampler.should_stop() is False


def test_warmup_without_n_needs_a_termination_mixin():
    sampler = nuts()(correlated_gaussian().model, 0)
    with pytest.raises(RuntimeError, match="termination mixin"):
        sampler.warmup()


@pytest.mark.parametrize("terminate", ["rhat", "classifier"])
def test_warmup_stops_early_on_an_easy_target(terminate):
    problem = correlated_gaussian()
    sampler = nuts(terminate=terminate, max_warmup=8000)(problem.model, 0)
    sampler.warmup()
    assert sampler.warmup_terminated_early()
    assert sampler._iteration < 2000, "an easy 2-D Gaussian should not need this long"
    assert len(sampler.warmup_mixing_stats()) >= 1


def test_classifier_keeps_warming_up_while_the_chain_is_still_travelling():
    """From far outside the typical set the periods genuinely differ, and it must say so."""
    problem = correlated_gaussian()
    sampler = nuts(terminate="classifier", init=np.array([200.0, 200.0]),
                   min_warmup=200, check_every=50)(problem.model, 0)
    sampler.warmup()
    stats = sampler.warmup_mixing_stats()
    assert stats[0, 1] > 0.55, "the first check should still separate the periods"
    assert sampler._iteration > 200, "it should not have stopped at the first opportunity"


@pytest.mark.parametrize("burn_in", ["fixed", "changepoint", "min_discard"])
def test_dynamically_warmed_chain_is_still_correct(artifacts_dir, burn_in):
    """Stopping early must not be bought with correctness --- under any burn-in rule."""
    problem = correlated_gaussian()
    report = evaluate(
        problem, {"nuts": nuts(terminate="classifier", burn_in=burn_in, max_warmup=8000)},
        n_warmup=8000, n_samples=20000, seed=0,
        out_dir=artifacts_dir / f"termination_classifier_{burn_in}")
    report.assert_correct()


@pytest.mark.parametrize("burn_in", ["changepoint", "min_discard"])
def test_dynamic_burn_in_is_recorded_and_stays_inside_its_bounds(burn_in):
    problem = correlated_gaussian()
    sampler = nuts(terminate="classifier", burn_in=burn_in, init=np.array([200.0, 200.0]),
                   max_warmup=8000)(problem.model, 0)
    sampler.warmup()
    burn = sampler.warmup_burn_in_estimates()
    assert burn.shape == (len(sampler.warmup_mixing_stats()), 2)
    assert np.all(burn[:, 1] >= 50)                          # the default lower bound
    assert np.all(burn[:, 1] <= 0.5 * burn[:, 0])            # never past ``burn_in_max_frac``


def test_features_of_a_unit_vector_model_flow_through_the_mixin():
    """The whole point of the feature layer: one mixin, any parameter type."""
    problem = von_mises_fisher(kappa=5.0)
    sampler = nuts(terminate="classifier", unit_vector_center=True,
                   max_warmup=4000)(problem.model, 0)
    sampler.warmup()
    assert sampler.warmup_terminated_early()
    assert np.asarray(sampler._term_features).shape[1] == problem.model.n_features == 5
