"""Unit tests for Polyak--Ruppert averaging of mass parameters (``mimcs.adaptation._polyak``).

Averaging is in log space (diagonal) / log-Cholesky space (dense), so a diagonal average is a
geometric mean and a dense factor keeps a positive diagonal (stays a valid Cholesky factor).
The integration tests (Polyak on by default, toggleable, and its noise-reduction effect on the
adapted mass) live in ``tests/test_mass_adaptation.py``.
"""

import numpy as np

from mimcs.adaptation._polyak import PolyakLog


def test_polyak_diagonal_is_geometric_mean():
    p = PolyakLog("diagonal")
    p.update(np.array([2.0, 8.0]))
    assert np.allclose(p.value(), [2.0, 8.0])                        # first = itself
    p.update(np.array([8.0, 2.0]))
    assert np.allclose(p.value(), [4.0, 4.0])                        # geometric mean (log space)


def test_polyak_dense_averages_log_cholesky_and_stays_a_valid_factor():
    p = PolyakLog("dense")
    L1 = np.array([[2.0, 0.0], [1.0, 3.0]])
    L2 = np.array([[8.0, 0.0], [3.0, 12.0]])
    p.update(L1)
    p.update(L2)
    out = p.value()
    assert np.allclose(np.triu(out, 1), 0.0)            # lower-triangular
    assert np.all(np.diag(out) > 0)                     # positive diagonal -> valid Cholesky
    assert np.allclose(np.diag(out), [4.0, 6.0])        # diagonal: geometric mean
    assert np.isclose(out[1, 0], 2.0)                   # off-diagonal: arithmetic mean


def test_polyak_suffix_average_rejects_early_transient():
    """A suffix average ignores the early ramp: 30 steps at 1.0 then 70 at 5.0 -> ~5.0, not the
    uniform geometric mean (~3.1). The tail window here excludes all of the transient."""
    xs = np.array([1.0] * 30 + [5.0] * 70).reshape(-1, 1)
    p = PolyakLog("diagonal")
    for x in xs:
        p.update(x)
    out = p.value()
    assert np.isclose(out[0], 5.0)                              # tail is all 5.0
    uniform = np.exp(np.log(xs).mean())                        # ~3.09
    assert out[0] > uniform + 1.0                              # clearly not the uniform mean


def test_update_does_not_compute_the_average_it_used_to_throw_away():
    """``update`` returned ``value()``, which every caller discarded --- an ``exp`` over the whole
    parameter, computed and dropped on every warmup iteration. It must not be computed now, and
    the average itself must be unchanged."""
    pl = PolyakLog("diagonal")
    for i in range(1, 9):
        assert pl.update(np.full(4, float(i))) is None
    # the fold is unaffected: the tail average is still the geometric mean of the tail window
    got = pl.value()
    assert got is not None and got.shape == (4,)
    assert np.allclose(got, np.exp(np.mean(np.log(np.arange(5, 9, dtype=float)))))
