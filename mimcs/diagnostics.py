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
    return float(min(n / _geyer_tau(lambda t: rho[t], n), n))


def _geyer_tau(rho, n: int) -> float:
    """Integrated autocorrelation time from ``rho(t)`` (a callable, ``rho(0) == 1``) by Geyer's
    initial monotone sequence: sum the lag pairs ``rho(2k) + rho(2k+1)`` up to the first
    non-positive one, made non-increasing. Lags are requested in order and only as far as the
    truncation, so a caller whose lags are expensive computes no more of them than it needs.
    Returns ``1`` when not even the first pair is positive."""
    gammas = []
    k = 0
    while 2 * k + 1 < n:
        g = rho(2 * k) + rho(2 * k + 1)
        if g <= 0.0:
            break
        gammas.append(g)
        k += 1
    if not gammas:
        return 1.0
    gammas = np.minimum.accumulate(np.asarray(gammas))
    return max(2.0 * gammas.sum() - 1.0, 1.0)


def _standardized(H: np.ndarray) -> np.ndarray:
    H = np.asarray(H, dtype=float)
    g = H - H.mean(axis=0)
    return g / np.sqrt(np.maximum((g * g).mean(axis=0), 1e-300))


def pooled_ess(H: np.ndarray, col_chunk: int = 256) -> float:
    """Effective sample size of a ``(n, d)`` chain's columns *pooled*: Geyer on the average of the
    standardized columns' autocorrelations.

    The per-column ESS is noisy, so its minimum over many columns is biased low even on
    independent rows (0.74 n over 100 iid columns at n = 1000). Averaging the autocorrelations
    before truncating has no such extreme-value bias and still reads a common autocorrelation
    exactly. Columns are transformed in chunks of ``col_chunk`` to bound the FFT's memory.
    """
    Z = _standardized(H)
    n, d = Z.shape
    if n < 4:
        return float(n)
    m = 1
    while m < 2 * n:
        m *= 2
    acov = np.zeros(n)
    for j in range(0, d, col_chunk):
        f = np.fft.rfft(Z[:, j:j + col_chunk], n=m, axis=0)
        acov += np.fft.irfft(f * np.conj(f), n=m, axis=0)[:n].real.sum(axis=1)
    if acov[0] <= 0.0:
        return float(n)
    rho = acov / acov[0]
    return float(min(n / _geyer_tau(lambda t: rho[t], n), n))


def second_moment_ess(H: np.ndarray) -> float:
    """Effective sample size of a ``(n, d)`` chain's **second moments**, pooled over every entry of
    ``h h^T`` (squares and cross products alike).

    A sample covariance or correlation matrix is an average of the products ``h_j h_k``, so its
    noise --- the width of its Marchenko-Pastur bulk --- is set by *their* autocorrelation, which
    can be far slower than the columns' own (a variance that follows a slowly mixing parameter).
    Summed over all ``d^2`` entries, the lag-``t`` autocovariance of the products has a closed form
    in the rows' lagged inner products,

        C(t) = mean_s (h_s . h_{s+t})^2 - ||R||_F^2,    R = mean_s h_s h_s^T,

    so every mixed term is covered at ``O(n d)`` per lag, with no ``d^2`` product matrix. Geyer's
    truncation stops the lags early. Columns are standardized first, so each entry is weighted by
    its own variance, as the bulk's width weights it.
    """
    Z = _standardized(H)
    n, d = Z.shape
    if n < 4:
        return float(n)
    small = Z @ Z.T if n < d else Z.T @ Z                 # same Frobenius norm either way
    frob2 = float(np.sum(small * small)) / (n * n)
    cache = {}

    def c(t):
        if t not in cache:
            ip = np.einsum("ij,ij->i", Z[:n - t], Z[t:])
            cache[t] = float(np.mean(ip * ip)) - frob2
        return cache[t]

    c0 = c(0)
    if c0 <= 0.0:
        return float(n)
    return float(min(n / _geyer_tau(lambda t: c(t) / c0, n), n))


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
    # ``stein_boundary``, so this one cannot be column-chunked the way :func:`ess` can. Nor
    # row-chunked: :func:`mimcs.summary.summarize` chunks the *production* of the matrices it hands
    # to these functions, never a reduction over their rows. ``docs/design/11`` tabulates which
    # quantity may be split which way, and why the boundary sits where it does.
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
        larger means they do not. A constant column yields 1 (nothing to disagree about); a
        column holding a non-finite value (or whose variance overflows) yields NaN.
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
    rhat = np.where(w > 0, np.sqrt(np.divide(var_plus, w, out=np.ones_like(w), where=w > 0)), 1.0)
    # A non-finite column is not a constant one: ``nan > 0`` is False, so without this it fell
    # into the guard above and read as R-hat 1 --- perfectly converged --- when an overflowing
    # draw had in fact made the column meaningless.
    return np.where(np.isfinite(w), rhat, np.nan)
