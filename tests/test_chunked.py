"""Row-chunked passes: `mimcs._chunked`.

Two different promises are pinned here, and they are not the same promise:

* :func:`map_rows` is **bit-identical** to one whole-array ``jax.vmap``. Rows are independent, so
  this is true of the mathematics --- but it is not something XLA guarantees (it is free to lower a
  batched computation differently at different batch sizes, and the functions mapped here contain
  intra-row reductions), so it is checked rather than assumed. Every assertion below is
  ``array_equal``; ``allclose`` would hide exactly the regression this file exists to catch.
* :func:`sum_rows` is **not** bit-identical --- it reorders the accumulation --- and its contract is
  that the difference stays far below the decision margins that read it. A test asserting equality
  there would be asserting the wrong thing.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from mimcs._chunked import CHUNK_BYTES, rows_per_chunk, map_rows, sum_rows
from mimcs.model import (Model, EuclideanParameter, PositiveParameter, UnitVectorParameter)


# --- the budget ------------------------------------------------------------- #

def test_rows_per_chunk_divides_the_budget():
    assert rows_per_chunk(8, budget=800) == 100
    assert rows_per_chunk(300, budget=800) == 2


def test_rows_per_chunk_floors_at_one_row():
    """A row wider than the whole budget must still make progress, not stall on a zero-length
    chunk."""
    assert rows_per_chunk(10 ** 9, budget=1) == 1
    assert rows_per_chunk(0, budget=0) == 1


# --- map_rows: bit-identity ------------------------------------------------- #

def _f(x):
    """A row function with an intra-row reduction, which is the case at risk."""
    return jnp.array([jnp.sum(x ** 2), jnp.prod(jnp.cos(x)), jnp.sum(jnp.exp(-x))])


@pytest.mark.parametrize("n", [1, 2, 63, 997])
@pytest.mark.parametrize("budget", [1, 1 << 6, 1 << 10, 1 << 30])
def test_map_rows_is_bit_identical_to_one_vmap(n, budget):
    """Every shape/chunk combination, including chunk=1, n=1, and an uneven final chunk."""
    x = jnp.asarray(np.random.default_rng(0).normal(size=(n, 11)))
    whole = np.asarray(jax.vmap(_f)(x))
    assert np.array_equal(map_rows(_f, x, budget=budget), whole)


def test_the_bit_identity_check_can_fail():
    """Control: the comparison above is not vacuous."""
    x = jnp.asarray(np.random.default_rng(0).normal(size=(97, 11)))
    got = map_rows(_f, x, budget=1 << 6)
    perturbed = got.copy()
    perturbed[5, 0] = np.nextafter(perturbed[5, 0], np.inf)      # one ulp, one element
    assert not np.array_equal(perturbed, np.asarray(jax.vmap(_f)(x)))


def test_map_rows_takes_several_arguments_and_a_pytree():
    """`_dep_data`'s `{name: (N, dep_dim)}` dict is a row-tree, not an array."""
    rng = np.random.default_rng(1)
    a = jnp.asarray(rng.normal(size=(50, 4)))
    tree = {"u": jnp.asarray(rng.normal(size=(50, 3))), "v": jnp.asarray(rng.normal(size=(50, 2)))}
    fn = lambda row, d: jnp.concatenate([row, d["u"], d["v"]]) * 2.0
    whole = np.asarray(jax.vmap(fn)(a, tree))
    assert np.array_equal(map_rows(fn, a, tree, budget=1 << 5), whole)


@pytest.mark.parametrize("budget", [1 << 5, 1 << 30])       # chunked, and the single-call path
def test_map_rows_owns_its_data(budget):
    """`np.asarray` of a JAX array is zero-copy on CPU and pins the whole device buffer, so the
    result must be a real copy --- on *both* paths, or they differ in the property that matters."""
    x = jnp.asarray(np.random.default_rng(0).normal(size=(64, 8)))
    assert map_rows(_f, x, budget=budget).flags.owndata


def test_map_rows_survives_an_empty_input():
    x = jnp.zeros((0, 11))
    assert map_rows(_f, x).shape == (0, 3)


# --- map_rows on the model functions summarize maps ------------------------- #

def _mixed_model():
    """Euclidean + bounded + manifold: the manifold parameter is the one whose per-row work is
    not elementwise."""
    return Model([EuclideanParameter("mu", (3,)), PositiveParameter("sigma", (2,)),
                  UnitVectorParameter("u", 4)],
                 {"lp": lambda d: (-0.5 * jnp.sum(d["mu"] ** 2) - jnp.sum(d["sigma"])
                                   + 5.0 * jnp.sum(d["u"]))})


def _draws(model, n, seed=0):
    rng = np.random.default_rng(seed)
    s = np.zeros((n, model.ambient_dim))
    s[:, 0:3] = rng.normal(size=(n, 3))
    s[:, 3:5] = np.abs(rng.normal(size=(n, 2))) + 0.1
    u = rng.normal(size=(n, 4))
    s[:, 5:9] = u / np.linalg.norm(u, axis=1, keepdims=True)
    return jnp.asarray(s, float)


@pytest.mark.parametrize("budget", [1 << 6, 1 << 9, 1 << 30])
def test_the_summarize_row_maps_are_bit_identical_when_chunked(budget):
    """The premise the `summarize` chunking rests on, on the four functions it maps."""
    m = _mixed_model()
    n = 397                                       # not a multiple of any chunk the budgets give
    x = _draws(m, n)
    cs = jnp.asarray(np.random.default_rng(3).normal(size=(n, m.coord_dim)), float)
    chp, ci = m.init_chart_hyperparams(), m.init_chart_indices()
    score = jax.vmap(lambda a, g: m.ambient_score(a, g, chp, ci))(x, cs)

    cases = {
        "features": (lambda a: m.features(a), (x,)),
        "sample_to_coordinate": (lambda a: m.sample_to_coordinate(a, chp, ci), (x,)),
        "ambient_score/recompute": (lambda a: m.ambient_score(a), (x,)),
        "ambient_score/pullback": (lambda a, g: m.ambient_score(a, g, chp, ci), (x, cs)),
        "stein_terms": (lambda a, s: m.stein_terms(a, s), (x, score)),
    }
    for name, (fn, args) in cases.items():
        whole = np.asarray(jax.vmap(fn)(*args))
        assert np.array_equal(map_rows(fn, *args, budget=budget), whole), name


@pytest.mark.parametrize("budget", [1 << 6, 1 << 30])
def test_the_fused_score_and_stein_pass_matches_the_two_stage_one(budget):
    """`summarize` computes the ambient score and the Stein terms in **one** mapped body, so the
    `(n, ambient_dim)` score matrix never exists whole.

    That is a different claim from the one above: chunking maps a fixed function over fewer rows,
    while fusion composes two functions into a body XLA may fuse into kernels it would not
    otherwise emit. Checked separately for that reason, on both score branches.
    """
    m = _mixed_model()
    n = 397
    x = _draws(m, n)
    cs = jnp.asarray(np.random.default_rng(3).normal(size=(n, m.coord_dim)), float)
    chp, ci = m.init_chart_hyperparams(), m.init_chart_indices()

    two_stage = np.asarray(jax.vmap(m.stein_terms)(
        x, jax.vmap(lambda a, g: m.ambient_score(a, g, chp, ci))(x, cs)))
    fused = map_rows(lambda a, g: m.stein_terms(a, m.ambient_score(a, g, chp, ci)),
                     x, cs, budget=budget)
    assert np.array_equal(fused, two_stage)

    two_stage_r = np.asarray(jax.vmap(m.stein_terms)(x, jax.vmap(lambda a: m.ambient_score(a))(x)))
    fused_r = map_rows(lambda a: m.stein_terms(a, m.ambient_score(a)), x, budget=budget)
    assert np.array_equal(fused_r, two_stage_r)


# --- sum_rows --------------------------------------------------------------- #

def _tol(scale=50.0):
    """Tolerance in units of the working precision's epsilon.

    The suite runs float32 by default, where a single ulp is ~1.2e-7 --- a fixed `rel=1e-11` would
    be asserting x64 behaviour and would fail for the right reason on the wrong grounds. Expressed
    against `eps` it means the same thing at both precisions: "a few rounding steps, not a
    different answer".
    """
    return scale * float(np.finfo(jnp.zeros(()).dtype).eps)


def _loss_row(params, g_row, dep_row):
    """The shape of the metric-regression objective: a log plus a ratio, summed over the row."""
    M = jnp.exp(params["b"] + dep_row @ params["W"])
    return 0.5 * jnp.sum(jnp.log(M) + g_row ** 2 / M)


def _loss_fixture(n=977, d=16, k=3, seed=0):
    rng = np.random.default_rng(seed)
    g = jnp.asarray(rng.normal(size=(n, d)) * 10.0)
    dep = jnp.asarray(rng.normal(size=(n, k)))
    params = {"b": jnp.asarray(rng.normal(size=d)), "W": jnp.asarray(rng.normal(size=(k, d)) * 0.1)}
    whole = lambda p: jnp.sum(jax.vmap(lambda a, b: _loss_row(p, a, b))(g, dep))
    return g, dep, params, whole


@pytest.mark.parametrize("budget", [1 << 8, 1 << 12, 1 << 30])
def test_sum_rows_matches_the_whole_array_sum(budget):
    g, dep, params, whole = _loss_fixture()
    got = float(sum_rows(lambda r: _loss_row(params, r[0], r[1]), (g, dep), budget=budget))
    assert got == pytest.approx(float(whole(params)), rel=_tol())


def test_sum_rows_actually_chunks():
    """Control for the test above: at a small budget the sum must be built from many chunks, or it
    would be passing by silently taking the whole-array path."""
    g, dep, _, _ = _loss_fixture()
    row_bytes = (g.shape[1] + dep.shape[1]) * g.dtype.itemsize
    assert rows_per_chunk(row_bytes, budget=1 << 8) < len(g) // 10


def test_sum_rows_matches_the_whole_array_gradient():
    """The reason `sum_rows` exists is to be differentiated; the value agreeing is not enough."""
    g, dep, params, whole = _loss_fixture()
    chunked = lambda p: sum_rows(lambda r: _loss_row(p, r[0], r[1]), (g, dep), budget=1 << 10)
    a, b = jax.grad(whole)(params), jax.grad(chunked)(params)
    for k in params:
        ref = np.asarray(a[k])
        assert np.allclose(ref, np.asarray(b[k]),
                           rtol=_tol(), atol=_tol() * float(np.abs(ref).max()))


def test_sum_rows_pads_with_a_real_row_and_not_with_zeros():
    """The padding rows carry zero weight, but `0 * inf` is `nan`, so a pad that is singular under
    the row function would poison the whole sum. Padding repeats row 0 --- a real row --- for
    exactly that reason, and this pins it against a zero-fill that would look equivalent.

    The control is the second assertion: the same function zero-filled *does* produce a nan, so
    the first assertion is not passing because the row function happens to tolerate zeros.
    """
    n, chunk_forcing_budget = 101, 1 << 6          # 101 rows never divides evenly
    x = jnp.asarray(np.abs(np.random.default_rng(0).normal(size=(n, 4))) + 0.5)
    singular_at_zero = lambda row: jnp.sum(1.0 / jnp.sum(row ** 2))

    got = float(sum_rows(lambda r: singular_at_zero(r[0]), (x,), budget=chunk_forcing_budget))
    assert np.isfinite(got)
    assert got == pytest.approx(float(jnp.sum(jax.vmap(singular_at_zero)(x))), rel=_tol())

    zero_padded = jnp.concatenate([x, jnp.zeros((3, 4))])
    weights = jnp.concatenate([jnp.ones(n), jnp.zeros(3)])
    poisoned = jnp.sum(weights * jax.vmap(singular_at_zero)(zero_padded))
    assert not np.isfinite(float(poisoned))        # what a zero-fill would have done


def test_sum_rows_needs_no_padding_when_the_rows_divide_evenly():
    g, dep, params, whole = _loss_fixture(n=1024, d=8, k=2)
    got = float(sum_rows(lambda r: _loss_row(params, r[0], r[1]), (g, dep), budget=1 << 9))
    assert got == pytest.approx(float(whole(params)), rel=_tol())


def test_the_budget_is_a_chunk_size_and_the_loss_gate_is_a_separate_threshold():
    """The two numbers do different jobs and must not drift together.

    `CHUNK_BYTES` sizes a chunk once chunking is on, so it has to be *small* to help --- big enough
    that an ordinary model never chunks, small enough that a 2000-coordinate one gets chunks worth
    having. `CHUNK_LOSS_BYTES` decides *whether* the non-bit-identical `sum_rows` path is taken at
    all, so it has to be *large*: above the biggest fit the suite performs.
    """
    from mimcs.factory.regression import CHUNK_LOSS_BYTES
    assert CHUNK_LOSS_BYTES > CHUNK_BYTES
    # a 2000-coordinate row (scores + an equal-width dependency) must give chunks, not one block
    assert 10 <= rows_per_chunk(2 * 2000 * 8, budget=CHUNK_BYTES) <= 2000
    # a 2-coordinate row must not chunk at any realistic draw count
    assert rows_per_chunk(2 * 2 * 8, budget=CHUNK_BYTES) > 100_000
