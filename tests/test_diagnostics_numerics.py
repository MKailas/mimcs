"""Bit-identity guards for the ESS / MCSE / R-hat estimators.

These are computed several times per :func:`mimcs.summary.summarize` call on the full
``(n_draws, n_features)`` matrix, so they are where a memory optimization is tempting --- and where
one can silently move a reported number. ``ess`` may convert column by column because nothing in it
reduces across columns; ``mcse_mean`` may **not**, because a whole-matrix ``std(axis=0)``
accumulates across all lanes at once while a per-column ``std`` is pairwise down one lane, and the
two differ in the last ulp on most columns. This file pins that asymmetry so neither half is
"tidied" into the other.

Seeds are fixed, so pass/fail is deterministic.
"""

import numpy as np
import pytest

from mimcs.diagnostics import ess, ess_1d, mcse_mean, split_rhat


def _data(n, p, seed, dtype=np.float32):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.standard_normal((n, p)), axis=0) * 0.05 + rng.standard_normal((n, p))
    return x.astype(dtype)


# --- ess: column-wise conversion is exact ------------------------------------------ #

def test_ess_is_unchanged_by_converting_column_by_column():
    """The saving: ``ess`` no longer materializes a float64 copy of the whole matrix."""
    for dtype in (np.float32, np.float64):
        x = _data(997, 23, seed=0, dtype=dtype)
        whole = np.asarray(x, dtype=float)                     # what it used to do
        reference = np.array([ess_1d(whole[:, j]) for j in range(x.shape[1])])
        assert np.array_equal(ess(x), reference), dtype


def test_ess_handles_a_constant_column_and_a_non_contiguous_input():
    x = _data(512, 5, seed=1)
    x[:, 0] = 3.0
    assert np.all(np.isfinite(ess(x)))
    strided = _data(1024, 4, seed=2)[::2]                      # not C-contiguous
    whole = np.asarray(strided, dtype=float)
    assert np.array_equal(
        ess(strided), np.array([ess_1d(whole[:, j]) for j in range(strided.shape[1])]))


# --- mcse_mean: the one that must NOT be column-chunked ---------------------------- #

def test_mcse_mean_keeps_its_whole_matrix_standard_deviation():
    """A regression guard with a reason.

    ``mcse_mean`` is ``sd / sqrt(ess)``, and it is tempting to convert per column as ``ess`` does.
    But ``sd`` comes from ``std(axis=0)`` over the whole matrix, which is **not** the same in the
    last ulp as a column-at-a-time ``std`` --- measured, 88% of columns differ at p=200. ``sd``
    reaches ``Summary.mcse`` and ``stein_mcse``, and ``stein_mcse`` decides ``stein_z`` and
    ``stein_boundary``, so a "tidy-up" here silently moves reported diagnostics.
    """
    x = _data(997, 40, seed=3)
    whole = np.asarray(x, dtype=float)
    assert np.array_equal(mcse_mean(x), whole.std(axis=0, ddof=1) / np.sqrt(ess(x)))
    # and the demonstration that the column-wise route would have been different
    per_col = np.array([np.asarray(x[:, j], dtype=float).std(ddof=1) for j in range(x.shape[1])])
    assert not np.array_equal(whole.std(axis=0, ddof=1), per_col), \
        "column-wise std matched here, so this guard is not testing anything"


# --- split_rhat: computing from the segments equals stacking them ------------------ #

def test_split_rhat_matches_the_stacked_formulation():
    """The stack was a third full copy of both segments; the arithmetic is unchanged without it."""
    for dtype in (np.float32, np.float64):
        for (n, p) in [(2000, 64), (7, 3), (1000, 1)]:
            a, b = _data(n, p, seed=4, dtype=dtype), _data(n, p, seed=5, dtype=dtype)
            fa = np.atleast_2d(np.asarray(a, dtype=float))
            fb = np.atleast_2d(np.asarray(b, dtype=float))
            chains = np.stack([fa, fb])                        # what it used to do
            chain_means = chains.mean(axis=1)
            grand = chain_means.mean(axis=0)
            b_over_n = ((chain_means - grand) ** 2).sum(axis=0) / (chains.shape[0] - 1)
            w = chains.var(axis=1, ddof=1).mean(axis=0)
            var_plus = (n - 1) / n * w + b_over_n
            want = np.where(w > 0, np.sqrt(np.divide(var_plus, w, out=np.ones_like(w),
                                                     where=w > 0)), 1.0)
            assert np.array_equal(split_rhat(a, b), want), (dtype, n, p)


def test_split_rhat_on_a_constant_column_is_one():
    a = np.full((50, 3), 2.0)
    assert np.array_equal(split_rhat(a, a.copy()), np.ones(3))


# --- the 1-D behaviour, which is odd but load-bearing ----------------------------- #

def test_a_one_dimensional_input_is_treated_as_one_row_of_columns():
    """``np.atleast_2d`` turns a ``(n,)`` input into ``(1, n)``, so these return one value *per
    element*, not a scalar. Surprising, but it is the behaviour, and a refactor of the conversion
    could quietly "fix" it into something else."""
    v = np.arange(50.0)
    assert ess(v).shape == (50,)
    assert mcse_mean(v).shape == (50,)
    assert split_rhat(v, v).shape == (50,)
