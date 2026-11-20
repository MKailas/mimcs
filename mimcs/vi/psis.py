"""Pareto-smoothed importance sampling (PSIS) and importance resampling, in NumPy on the host.

PSIS (Vehtari, Simpson, Gelman, Yao & Gabry, "Pareto smoothed importance sampling", 2015--2024)
stabilises importance weights by replacing the largest ones with expected order statistics of a
generalized Pareto distribution (GPD) fitted to the tail. The fitted shape ``k`` is the
diagnostic: below 0.5 the raw weights have finite variance; above 0.7 the importance-sampling
estimate is unreliable --- for Pathfinder, that the normal approximation is a poor one.

This follows the reference implementation (the ``loo`` R package and ``arviz``) step for step: a
right tail of ``ceil(min(S/5, 3 sqrt(S)))`` draws, the Zhang & Stephens (2009) empirical-Bayes GPD
fit with the weakly informative prior on ``k`` (centred at 0.5, weight 10), tail weights replaced by
GPD quantiles at ``(i - 1/2)/M`` and truncated at the largest raw weight. ``tests/test_psis.py``
holds it to ``arviz_stats`` where that is installed.

It runs on the host: it sorts, fits a one-dimensional distribution and is called once per
Pathfinder run, on at most a few thousand weights.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["psis", "resample"]

_PRIOR_BS = 3          # Zhang & Stephens grid scale
_PRIOR_K = 10          # weight of the k prior (pseudo-observations at k = 0.5)


def _gpdfit(x: np.ndarray) -> tuple[float, float]:
    """Empirical-Bayes ``(k, sigma)`` of a GPD for sorted, non-negative exceedances ``x``."""
    n = len(x)
    m_est = 30 + int(n ** 0.5)
    xstar = x[int(n / 4 + 0.5) - 1]                        # first quartile
    if xstar <= x[0]:
        return math.nan, math.nan                          # far from any GPD
    b = 1.0 - np.sqrt(m_est / (np.arange(1, m_est + 1, dtype=float) - 0.5))
    b = b / (_PRIOR_BS * xstar) + 1.0 / x[-1]
    k = np.mean(np.log1p(-b[:, None] * x), axis=1)
    len_scale = n * (np.log(-b / k) - k - 1.0)
    w = np.exp(len_scale - np.max(len_scale))
    w /= w.sum()
    keep = w >= 10 * np.finfo(float).eps
    b, w = b[keep], w[keep] / w[keep].sum()
    b_post = float(np.sum(b * w))
    k_hat = float(np.mean(np.log1p(-b_post * x)))
    sigma = -k_hat / b_post
    k_hat = (n * k_hat + _PRIOR_K * 0.5) / (n + _PRIOR_K)
    if math.isnan(k_hat):
        return math.inf, math.nan
    return k_hat, sigma


def _gpinv(p: np.ndarray, k: float, sigma: float, mu: float) -> np.ndarray:
    """GPD quantile function."""
    if not sigma > 0:
        return np.full_like(p, np.nan)
    if k == 0:
        return mu - sigma * np.log1p(-p)
    return mu + sigma * np.expm1(-k * np.log1p(-p)) / k


def psis(log_w) -> tuple[np.ndarray, float]:
    """Pareto-smooth log importance weights: ``(smoothed log weights, k_hat)``.

    ``log_w`` may hold ``-inf`` (a draw with zero weight --- a failed Pathfinder path reports
    ``log q = +inf``); those stay ``-inf`` and take no part in the fit. NaN counts as ``-inf``.
    The smoothed weights are unnormalised, on the scale of the input shifted by its maximum.
    ``k_hat`` is NaN when there are too few finite weights to fit a tail (fewer than five in it) or
    the tail is flat, in which case the weights come back unsmoothed.
    """
    log_w = np.asarray(log_w, dtype=float).ravel()
    out = np.where(np.isfinite(log_w), log_w, -np.inf)
    finite = np.flatnonzero(np.isfinite(out))
    if finite.size == 0:
        return out, math.nan
    out[finite] -= out[finite].max()
    s = finite.size
    m = int(math.floor(3 * s ** 0.5)) if s > 225 else int(math.floor(s / 5))
    if m < 5:
        return out, math.nan
    vals = out[finite]
    order = np.argsort(vals)
    tail_ids = order[s - m:]
    cutoff = vals[order[s - m - 1]]
    tail = vals[tail_ids]
    if tail.max() - tail.min() < np.finfo(float).tiny:
        return out, math.nan
    k_hat, sigma = _gpdfit(np.exp(tail) - np.exp(cutoff))
    if np.isfinite(k_hat):
        p = np.arange(0.5, m) / m
        smoothed = np.log(_gpinv(p, k_hat, sigma, np.exp(cutoff)))
        smoothed = np.minimum(smoothed, tail.max())
        vals[tail_ids] = smoothed
        out[finite] = vals
    return out, k_hat


def resample(rng: np.random.Generator, log_w, size: int) -> np.ndarray:
    """``size`` indices drawn **with replacement**, with probability proportional to ``exp(log_w)``.

    ``-inf`` weights are never drawn. Raises if no weight is positive.
    """
    log_w = np.asarray(log_w, dtype=float)
    finite = np.isfinite(log_w)
    if not finite.any():
        raise ValueError("resample: every importance weight is zero")
    w = np.where(finite, np.exp(log_w - log_w[finite].max()), 0.0)
    return rng.choice(len(w), size=size, replace=True, p=w / w.sum())
