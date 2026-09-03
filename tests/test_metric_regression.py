"""Tests for the factory's metric regression + candidate enumeration.

Fit mass-matrix expressions to synthetic evidence whose conditional score covariance has a
known form, and check the AIC ranking selects the right structure: a position-dependent
metric when the variance depends on another coordinate, the constant baseline when it does
not, and the joint log-linear form over the separable one when the data is genuinely joint.
"""

import numpy as np
import jax.numpy as jnp
import pytest

from mimcs.hmc.metric_expr import Exp, Sigmoid, SpExp
from mimcs.factory import regression
from mimcs.factory.regression import (
    fit_metric_expr, aic, enumerate_candidates, select_metric)


def _evidence_conditional_var(rng, N, var_fn):
    """Row-aligned (coords, grads), columns ``[x, y, block]``. The block's score (grad col 2)
    has conditional variance ``var_fn(x, y)``; x, y are the candidate dependency coordinates."""
    x = rng.normal(0.0, 1.2, size=N)
    y = rng.normal(0.0, 1.2, size=N)
    z = rng.normal(0.0, 1.0, size=N)                 # the block's own position (unused metric-wise)
    g_block = rng.normal(0.0, 1.0, size=N) * np.sqrt(var_fn(x, y))
    coords = np.column_stack([x, y, z])              # cols: 0=x, 1=y, 2=block
    grads = np.column_stack([np.zeros(N), np.zeros(N), g_block])
    return coords, grads, 2                           # block score column = 2


def _evidence_elementwise(rng, N, B, var_fn):
    """A B-dimensional block whose score covariance is *elementwise*: the score of block
    coordinate j has conditional variance ``var_fn(s_j)`` for a same-dimensional dependency
    ``s``. Columns are ``[s_0..s_{B-1}, block_0..block_{B-1}]``; returns (coords, grads,
    block_cols, dep_cols)."""
    s = rng.normal(0.0, 1.0, size=(N, B))
    g_block = rng.normal(0.0, 1.0, size=(N, B)) * np.sqrt(var_fn(s))
    coords = np.column_stack([s, np.zeros((N, B))])          # block's own coords unused
    grads = np.column_stack([np.zeros((N, B)), g_block])
    return coords, grads, list(range(B, 2 * B)), {"s": list(range(B))}


# --- enumeration ------------------------------------------------------------- #


def test_enumeration_baseline_budget_and_cap():
    dep_dims = {"x": 1, "y": 1}
    cands = enumerate_candidates(4, dep_dims, param_budget=1000, max_candidates=50)
    # constant baseline is always first
    assert cands[0].deps() == set()
    reprs = [repr(c) for c in cands]
    assert "Exp('x') + Exp()" in reprs and "Exp('x', 'y') + Exp()" in reprs
    # every position-dependent form is offered bare as well, and bare comes first of the pair
    assert "Exp('x')" in reprs and "Exp('x', 'y')" in reprs
    assert reprs.index("Exp('x')") < reprs.index("Exp('x') + Exp()")

    # budget filters out expensive candidates (only the free baseline fits budget 0)
    tiny = enumerate_candidates(4, dep_dims, param_budget=0, max_candidates=50)
    assert len(tiny) == 1 and tiny[0].deps() == set()

    # hard cap is honoured even with many dependencies
    many = {f"p{i}": 1 for i in range(40)}
    capped = enumerate_candidates(1, many, param_budget=10_000, max_candidates=50)
    assert len(capped) == 50


def test_enumeration_without_bare_is_the_floor_only_pool():
    """``include_bare=False`` restores the pre-2026-08 pool: every position-dependent candidate
    carries the additive ``+ Exp()`` floor, and only the constant baseline is floor-free."""
    dep_dims = {"x": 1, "s": 4}
    pool = enumerate_candidates(4, dep_dims, param_budget=1000, max_candidates=50,
                                include_bare=False)
    assert pool[0].deps() == set()
    assert all(repr(c).endswith("+ Exp()") for c in pool[1:]), [repr(c) for c in pool]
    # the bare pool is a strict superset, and keeps the floored candidates in the same order
    both = enumerate_candidates(4, dep_dims, param_budget=1000, max_candidates=50,
                                include_bare=True)
    assert [repr(c) for c in both if repr(c).endswith("+ Exp()")] == \
           [repr(c) for c in pool[1:]]


def test_aic_formula():
    assert aic(loss=0.5, n_params=3, n_rows=100) == 2.0 * 3 + 2.0 * 100 * 0.5


# --- fitting ----------------------------------------------------------------- #


def test_fit_recovers_log_linear_metric():
    """M(x) = e^{-x}: fitting Exp("x") recovers W -> -1, b -> 0 (block score var e^{-x})."""
    rng = np.random.default_rng(0)
    N = 5000
    coords, grads, bcol = _evidence_conditional_var(rng, N, lambda x, y: np.exp(-x))
    loss, params = fit_metric_expr(Exp("x"), [bcol], {"x": [0]}, coords, grads, max_iter=200)
    W = float(np.asarray(params["W"][0]).ravel()[0])
    b = float(np.asarray(params["b"]).ravel()[0])
    assert abs(W + 1.0) < 0.1 and abs(b) < 0.1


# --- selection --------------------------------------------------------------- #


def test_selects_position_dependent_over_constant():
    """When the score variance depends on x, the winner must be position-dependent (uses x),
    beating the constant baseline."""
    rng = np.random.default_rng(1)
    N = 5000
    coords, grads, bcol = _evidence_conditional_var(rng, N, lambda x, y: np.exp(-x))
    ranked = select_metric([bcol], {"x": [0], "y": [1]}, coords, grads, max_iter=150)
    best = ranked[0]
    assert "x" in best.expr.deps(), f"winner {best.expr!r} ignores x"
    baseline = next(r for r in ranked if r.expr.deps() == set())
    assert best.aic < baseline.aic


def test_selects_constant_when_homoscedastic():
    """Constant variance -> no position-dependent candidate improves the fit enough to pay its
    AIC penalty, so the constant baseline wins."""
    rng = np.random.default_rng(2)
    N = 5000
    coords, grads, bcol = _evidence_conditional_var(rng, N, lambda x, y: np.full_like(x, 2.0))
    ranked = select_metric([bcol], {"x": [0], "y": [1]}, coords, grads, max_iter=150)
    assert ranked[0].expr.deps() == set(), f"winner {ranked[0].expr!r} should be constant"


def test_prefers_joint_over_separable_on_joint_data():
    """Variance e^{-x-y} is a single log-linear form: the joint Exp("x","y") fits it exactly
    and must beat the separable Exp("x")+Exp("y") by AIC."""
    rng = np.random.default_rng(3)
    N = 6000
    coords, grads, bcol = _evidence_conditional_var(rng, N, lambda x, y: np.exp(-x - y))
    dep_cols = {"x": [0], "y": [1]}
    joint = Exp("x", "y") + Exp()
    separable = Exp("x") + Exp("y") + Exp()
    lj, _ = fit_metric_expr(joint, [bcol], dep_cols, coords, grads, max_iter=200)
    ls, _ = fit_metric_expr(separable, [bcol], dep_cols, coords, grads, max_iter=200)
    aic_j = aic(lj, joint.n_params(1, {"x": 1, "y": 1}), N)
    aic_s = aic(ls, separable.n_params(1, {"x": 1, "y": 1}), N)
    assert aic_j < aic_s, f"joint AIC {aic_j} !< separable AIC {aic_s}"


# --- sparse (elementwise) dependences --------------------------------------- #


def test_enumeration_sparse_only_when_dims_match():
    """Sparse candidates appear only for a dependency sharing the block's dimension."""
    match = enumerate_candidates(8, {"s": 8}, param_budget=10_000, max_candidates=50)
    assert any(isinstance(c, type(SpExp("s") + Exp())) and "SpExp" in repr(c) for c in match)
    assert "SpExp('s') + Exp()" in [repr(c) for c in match]
    # a dependency of a different dimension -> no sparse candidate
    mismatch = enumerate_candidates(8, {"s": 5}, param_budget=10_000, max_candidates=50)
    assert all("SpExp" not in repr(c) for c in mismatch)


def test_sparse_is_only_viable_form_for_equal_large_dims():
    """For equal *large* dims the dense Exp('s') blows the 20*block_dim budget while the sparse
    SpExp('s') (~2*block_dim params) fits -- so a position-dependent form is only reachable
    sparsely."""
    B = 40
    cands = enumerate_candidates(B, {"s": B}, param_budget=20 * B, max_candidates=50)
    reprs = [repr(c) for c in cands]
    assert "SpExp('s') + Exp()" in reprs                 # sparse fits the budget
    assert "Exp('s') + Exp()" not in reprs               # dense (B^2+2B params) filtered out


def test_selects_sparse_on_elementwise_variance():
    """Block score variance e^{-s_j} (elementwise) -> the AIC winner is sparse, recovering the
    per-coordinate weight -1, and beats both the constant baseline and the dense Exp('s')."""
    rng = np.random.default_rng(4)
    B, N = 8, 4000
    coords, grads, bcols, dep_cols = _evidence_elementwise(
        rng, N, B, lambda s: np.exp(-s))
    ranked = select_metric(bcols, dep_cols, coords, grads, max_iter=150)
    best = ranked[0]
    # the truth has no additive floor, so the *bare* sparse form must win outright
    assert repr(best.expr) == "SpExp('s')", f"winner {best.expr!r} is not the bare sparse form"
    floored = next(r for r in ranked if repr(r.expr) == "SpExp('s') + Exp()")
    assert best.aic < floored.aic
    # the fitted per-coordinate weight is ~ -1 across the block
    W = np.asarray(best.params["W"][0]).ravel()
    assert np.allclose(W, -1.0, atol=0.15), W
    # sparse beats the dense log-linear form (which can also fit, but pays a far larger AIC)
    dense = Exp("s") + Exp()
    ld, _ = fit_metric_expr(dense, bcols, dep_cols, coords, grads, max_iter=200)
    aic_dense = aic(ld, dense.n_params(B, {"s": B}), N)
    assert best.aic < aic_dense, f"sparse AIC {best.aic} !< dense AIC {aic_dense}"
    baseline = next(r for r in ranked if r.expr.deps() == set())
    assert best.aic < baseline.aic


# --- chunked accumulation: the gate, and what it protects -------------------- #

def test_the_loss_gate_keeps_a_small_fit_on_the_whole_array_path():
    """Below `CHUNK_LOSS_BYTES` the loss is the whole-array expression it always was, bit for bit.

    That is the gate's entire purpose: chunked accumulation reorders the sum, and the suite's
    seed-pinned metric selections (bare-vs-floored candidates are decided by a `1/N`-nats margin)
    must not shift underneath it. Every fit in the suite is orders of magnitude below the gate, so
    this asserts the property the rest of the suite silently relies on.
    """
    rng = np.random.default_rng(0)
    coords, grads, block = _evidence_conditional_var(rng, 400, lambda x, y: np.exp(-x))
    args = (Exp("x"), [block], {"x": [0]}, coords, grads)
    a = fit_metric_expr(*args, max_iter=25)
    b = fit_metric_expr(*args, max_iter=25)
    assert a[0] == b[0] and np.array_equal(np.asarray(a[1]["b"]), np.asarray(b[1]["b"]))
    assert 400 * 1 * 8 < regression.CHUNK_LOSS_BYTES      # this fit is nowhere near the gate


def test_the_chunked_loss_agrees_with_the_whole_array_one(monkeypatch):
    """Forced onto the chunked path, the same fit must land in the same place.

    Not bit-identical --- `sum_rows` reorders the accumulation --- so this is a tolerance test, and
    the tolerance is the point: the gap has to be far below the margins that read the loss.
    """
    rng = np.random.default_rng(1)
    coords, grads, block = _evidence_conditional_var(rng, 500, lambda x, y: np.exp(-x))
    args = (Exp("x"), [block], {"x": [0]}, coords, grads)
    whole_loss, whole_params = fit_metric_expr(*args, max_iter=30)

    monkeypatch.setattr(regression, "CHUNK_LOSS_BYTES", 0)          # gate open
    monkeypatch.setattr("mimcs._chunked.CHUNK_BYTES", 64)           # and chunks of a few rows
    chunk_loss, chunk_params = fit_metric_expr(*args, max_iter=30)

    eps = float(np.finfo(jnp.zeros(()).dtype).eps)
    assert chunk_loss == pytest.approx(whole_loss, rel=200 * eps)
    assert np.allclose(np.asarray(chunk_params["W"]), np.asarray(whole_params["W"]),
                       rtol=1e-3, atol=1e-3)


def test_the_gate_test_really_switches_paths(monkeypatch):
    """Control for the test above: with the budget lowered the fit must actually be built from many
    chunks, or it would be comparing the whole-array path against itself.

    `rows_per_chunk` is called with `budget=None` here on purpose --- that is the path the library
    takes, and it is what makes the monkeypatch effective. Binding `CHUNK_BYTES` as a default
    argument would leave every "chunked" arm in this file silently unchunked.
    """
    from mimcs._chunked import rows_per_chunk
    assert rows_per_chunk(64) > 10_000                    # unpatched: nothing this small chunks
    monkeypatch.setattr("mimcs._chunked.CHUNK_BYTES", 64)
    assert rows_per_chunk(64) == 1


def test_fit_is_usable_catches_a_bad_metric_in_a_later_chunk(monkeypatch):
    """The finite/positive check is now a running reduction over chunks, so a row that goes bad
    *after* the first chunk is exactly what a broken short-circuit would miss."""
    rng = np.random.default_rng(2)
    N = 400
    coords, grads, block = _evidence_conditional_var(rng, N, lambda x, y: np.ones_like(x))
    monkeypatch.setattr("mimcs._chunked.CHUNK_BYTES", 64)

    from mimcs._chunked import rows_per_chunk
    chunk = rows_per_chunk(coords.dtype.itemsize)            # one dependency column per row
    # `Exp('x')` is exp(b + W x): with a huge W it overflows only where x is large. The control
    # that makes this test mean anything is that such a row falls in a *later* chunk.
    big = int(np.argmax(coords[:, 0]))
    assert chunk < N and big >= chunk, (
        f"the offending row (index {big}) must not be in the first chunk of {chunk}")
    params = {"W": jnp.asarray([[1.0e4]]), "b": jnp.asarray([0.0])}
    assert not regression.fit_is_usable(Exp("x"), params, {"x": [0]}, coords, 1.0)

    ok = {"W": jnp.asarray([[0.0]]), "b": jnp.asarray([0.0])}
    assert regression.fit_is_usable(Exp("x"), ok, {"x": [0]}, coords, 1.0)


def test_whitened_scores_are_unchanged_by_chunking(monkeypatch):
    """Elementwise throughout, so chunking must be bit-identical here."""
    rng = np.random.default_rng(3)
    coords, grads, block_cols, dep_cols = _evidence_elementwise(
        rng, 300, 4, lambda s: np.exp(-s))
    params = {"W": jnp.asarray(-np.ones((4, 1))), "b": jnp.zeros(4)}
    whole = regression.whitened_scores(SpExp("s"), params, block_cols, dep_cols, coords, grads)
    monkeypatch.setattr("mimcs._chunked.CHUNK_BYTES", 64)
    from mimcs._chunked import rows_per_chunk
    assert rows_per_chunk(8 * coords.dtype.itemsize) < 300        # control: it really chunks
    chunked = regression.whitened_scores(SpExp("s"), params, block_cols, dep_cols, coords, grads)
    assert np.array_equal(whole, chunked)
