"""MCMC convergence diagnostics: autocorrelation, effective sample size, MCSE, split-R-hat.

The library's one runtime home for the numeric diagnostic primitives (pure numpy, no other mimcs
dependency), used by warmup termination (:mod:`mimcs.adaptation.termination`), sample evaluation
(:mod:`mimcs.summary`), and the test harness.

MCMC draws are autocorrelated, so the Monte Carlo standard error of an estimate is
``sd / sqrt(ESS)``, not ``sd / sqrt(N)``. The ESS estimator is Geyer's initial monotone positive
sequence estimator (the family Stan uses), applied per coordinate on a single chain.

R-hat compares the variance *between* chains with the variance *within* them: while the chains
have not forgotten where they started, the between-chain spread carries the initial conditions
rather than the target's, and the ratio exceeds 1. **Split** R-hat (Gelman et al.; what Stan
reports) cuts one chain into segments and treats those as the chains, which also makes it
sensitive to a drifting chain. Not implemented: the rank-normalized / folded variants (Vehtari
et al. 2021).
"""

from __future__ import annotations

import numpy as np


def autocorrelation(x: np.ndarray) -> np.ndarray:
    """Normalized autocorrelation function of a 1D series, via FFT.

    Returns ``rho[0..n-1]`` with ``rho[0] == 1``. Zero-variance input returns a
    delta at lag 0.
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    x = x - x.mean()
    var = np.dot(x, x) / n
    if var == 0.0:
        out = np.zeros(n)
        out[0] = 1.0
        return out
    # zero-pad to >= 2n for a linear (non-circular) autocovariance
    m = 1
    while m < 2 * n:
        m *= 2
    f = np.fft.rfft(x, n=m)
    acov = np.fft.irfft(f * np.conj(f), n=m)[:n].real / n
    return acov / acov[0]


def ess_1d(x: np.ndarray) -> float:
    """Effective sample size of a 1D chain (Geyer initial monotone sequence)."""
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    if n < 4 or np.var(x) == 0.0:
        return float(n)

    rho = autocorrelation(x)

    # Pair successive lags: Gamma_k = rho_{2k} + rho_{2k+1}. The pair sums are
    # theoretically positive; truncate at the first non-positive pair.
    gammas = []
    k = 0
    while 2 * k + 1 < n:
        g = rho[2 * k] + rho[2 * k + 1]
        if g <= 0.0:
            break
        gammas.append(g)
        k += 1
    if not gammas:
        return float(n)

    # Initial monotone sequence: enforce non-increasing pair sums.
    gammas = np.minimum.accumulate(np.asarray(gammas))

    tau = 2.0 * gammas.sum() - 1.0   # integrated autocorrelation time
    tau = max(tau, 1.0)
    return float(min(n / tau, n))


def ess(samples: np.ndarray) -> np.ndarray:
    """Per-coordinate effective sample size of a ``(n, d)`` chain.

    Deliberately does **not** cast the whole matrix: :func:`ess_1d` casts each column it is given,
    nothing here reduces across columns, and a float64 copy of an ``(n, d)`` feature matrix is the
    largest transient ``summarize`` allocates. Column-wise conversion is bit-identical because the
    conversion is elementwise --- unlike a column-wise ``std``, which is **not** (see
    :func:`mcse_mean`).
    """
    samples = np.atleast_2d(samples)
    if samples.ndim == 1:
        samples = samples[:, None]
    return np.array([ess_1d(samples[:, j]) for j in range(samples.shape[1])])


def mcse_mean(samples: np.ndarray) -> np.ndarray:
    """Monte Carlo standard error of the per-coordinate mean: ``sd / sqrt(ESS)``."""
    # The whole-matrix float64 cast stays. ``std(axis=0)`` over an ``(n, p)`` array accumulates
    # across all p lanes at once, where a column-at-a-time ``std`` is pairwise down one lane: the
    # two differ in the last ulp on most columns (measured: 88% at p=200). ``sd`` reaches
    # ``Summary.mcse`` and ``stein_mcse``, and ``stein_mcse`` decides ``stein_z`` and
    # ``stein_boundary``, so this one cannot be column-chunked the way :func:`ess` can.
    samples = np.atleast_2d(np.asarray(samples, dtype=float))
    if samples.ndim == 1:
        samples = samples[:, None]
    sd = samples.std(axis=0, ddof=1)
    return sd / np.sqrt(ess(samples))


def split_rhat(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Gelman--Rubin R-hat treating ``a`` and ``b`` as two chains; one value per column.

    Args:
        a, b: ``(n, p)`` segments of a chain (equal length), each column an observable.

    Returns:
        ``(p,)`` R-hat per column. 1 means the two segments agree on both location and spread;
        larger means they do not. A constant column yields 1 (nothing to disagree about).
    """
    a = np.atleast_2d(np.asarray(a, dtype=float))
    b = np.atleast_2d(np.asarray(b, dtype=float))
    if a.shape != b.shape:
        raise ValueError(f"segments must have the same shape, got {a.shape} and {b.shape}")
    n = a.shape[0]
    if n < 2:
        return np.full(a.shape[1], np.inf)

    # Computed from ``a`` and ``b`` directly rather than from a stacked ``(2, n, p)`` copy of
    # both, which was a third full materialization of the segments for no gain: reducing axis 1
    # of a C-contiguous stack visits each slab exactly as reducing axis 0 of each segment, so the
    # values are bit-identical (verified across shapes, dtypes, and a non-contiguous input).
    mean_a, mean_b = a.mean(axis=0), b.mean(axis=0)          # each (p,)
    grand_mean = (mean_a + mean_b) / 2.0

    # B/n is the variance of the chain means; W the mean of the within-chain variances.
    b_over_n = ((mean_a - grand_mean) ** 2 + (mean_b - grand_mean) ** 2) / (2 - 1)
    w = (a.var(axis=0, ddof=1) + b.var(axis=0, ddof=1)) / 2.0

    # var+ overestimates the target variance while the chains disagree, and W underestimates it;
    # their ratio is the diagnostic.
    var_plus = (n - 1) / n * w + b_over_n
    return np.where(w > 0, np.sqrt(np.divide(var_plus, w, out=np.ones_like(w), where=w > 0)), 1.0)
