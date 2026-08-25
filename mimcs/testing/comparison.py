"""Two-sample comparison of MCMC outputs.

Everything reduces to comparing two sample sets ``A`` and ``B``, where ``B`` is a
*reference*: either exact i.i.d. draws (comparison against an analytic solution) or
another sampler's output (algorithm-vs-algorithm). The same statistics apply to both;
only the standard-error model differs (i.i.d. references use ``sd/sqrt(N)``, MCMC
references use ``sd/sqrt(ESS)``).

Two complementary families of checks:

* **Moment effect sizes** --- differences in mean (scaled by combined Monte Carlo
  standard error), per-coordinate standard-deviation ratios, and correlation
  differences. These are interpretable and stable, and catch location / scale /
  linear-dependence bugs.
* **Whitened energy distance** --- a parameter-free two-sample statistic with a
  permutation p-value. The samples are first whitened by the pooled covariance, so
  the test is insensitive to scale and sensitive to *shape* (e.g. the curvature of a
  Rosenbrock banana, which moment checks cannot see).

Correctness assertions are driven primarily by the moment effect sizes against
conservative thresholds (default 5 standard errors), with the energy-distance
permutation p-value as a shape check at a deliberately small significance level to
keep the suite non-flaky under fixed seeds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist

from ..diagnostics import ess, mcse_mean


@dataclass
class Thresholds:
    """Pass/fail thresholds for :meth:`ComparisonResult.checks`.

    Both the mean and variance checks are *effect sizes scaled by Monte Carlo standard
    error*, so they automatically loosen when the effective sample size is small or the
    marginal is heavy-tailed (where moment estimates are genuinely uncertain) and
    tighten when ESS is large. This keeps the checks statistically honest rather than
    flagging sampling noise as a bug.
    """

    mean_z: float = 5.0          # max |mean diff| in combined standard errors
    var_z: float = 5.0           # max |log-variance diff| in combined standard errors
    corr_tol: float = 0.12       # max |corr_A - corr_B| over off-diagonal entries
    energy_p_min: float = 1e-3   # fail if energy-distance permutation p-value below


@dataclass
class Check:
    name: str
    passed: bool
    value: float
    threshold: float
    detail: str = ""


@dataclass
class ComparisonResult:
    label: str                       # e.g. "rwmh vs analytic"
    a_name: str
    b_name: str
    dim: int
    mean_a: np.ndarray
    mean_b: np.ndarray
    mean_z: np.ndarray               # per-coordinate |mean diff| / combined SE
    std_a: np.ndarray
    std_b: np.ndarray
    std_ratio: np.ndarray            # std_a / std_b
    var_z: np.ndarray                # per-coordinate |log var diff| / combined SE
    corr_max_diff: float
    energy_stat: float
    energy_pvalue: float
    ks_stat: np.ndarray              # per-coordinate KS statistic (reported)
    ks_pvalue: np.ndarray
    ess_a: np.ndarray
    ess_b: np.ndarray | None
    checks: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.passed for c in self.checks)

    def summary(self) -> str:
        lines = [f"[{self.label}]  ({self.a_name} vs {self.b_name}), dim={self.dim}"]
        lines.append(f"  mean {self.a_name:>8}: {np.array2string(self.mean_a, precision=3)}")
        lines.append(f"  mean {self.b_name:>8}: {np.array2string(self.mean_b, precision=3)}")
        lines.append(f"  mean z-scores     : {np.array2string(self.mean_z, precision=2)}"
                     f"  (max {self.mean_z.max():.2f})")
        lines.append(f"  std ratio (a/b)   : {np.array2string(self.std_ratio, precision=3)}")
        lines.append(f"  var z-scores      : {np.array2string(self.var_z, precision=2)}"
                     f"  (max {self.var_z.max():.2f})")
        lines.append(f"  max |corr| diff   : {self.corr_max_diff:.3f}")
        lines.append(f"  energy distance   : {self.energy_stat:.4f}  (perm p={self.energy_pvalue:.3f})")
        lines.append(f"  KS stat (max)     : {self.ks_stat.max():.3f}")
        lines.append(f"  ESS {self.a_name:>8}  : {np.array2string(self.ess_a.astype(int))}")
        for c in self.checks:
            mark = "PASS" if c.passed else "FAIL"
            lines.append(f"    [{mark}] {c.name}: {c.detail}")
        return "\n".join(lines)


def _correlation(x: np.ndarray) -> np.ndarray:
    if x.shape[1] < 2:
        return np.zeros((1, 1))
    return np.corrcoef(x, rowvar=False)


def _whiten(A: np.ndarray, B: np.ndarray):
    pooled = np.vstack([A, B])
    mu = pooled.mean(axis=0)
    cov = np.atleast_2d(np.cov(pooled, rowvar=False))
    cov = cov + 1e-9 * np.eye(cov.shape[0])
    L = np.linalg.cholesky(cov)
    Winv = np.linalg.inv(L)
    return (A - mu) @ Winv.T, (B - mu) @ Winv.T


def _log_var_se(samples: np.ndarray, n_eff: np.ndarray) -> np.ndarray:
    """Standard error of ``log(variance)`` per coordinate.

    By the delta method, ``SE[log s^2] = sqrt((kappa - 1) / N_eff)`` where ``kappa`` is
    the (non-excess) kurtosis. For a Gaussian (``kappa = 3``) this is ``sqrt(2/N_eff)``;
    heavy-tailed marginals have larger ``kappa`` and correspondingly larger SE.
    """
    mu = samples.mean(0)
    centered = samples - mu
    var = (centered**2).mean(0)
    var = np.where(var > 0, var, np.inf)
    kappa = (centered**4).mean(0) / var**2
    return np.sqrt(np.maximum(kappa - 1.0, 1e-6) / n_eff)


def _subsample(X: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if len(X) <= n:
        return X
    return X[rng.choice(len(X), n, replace=False)]


def _energy_distance(A: np.ndarray, B: np.ndarray) -> float:
    """Energy distance ``2 E||A-B|| - E||A-A'|| - E||B-B'||`` (>= 0, 0 iff equal)."""
    d_ab = cdist(A, B).mean()
    d_aa = cdist(A, A).mean()
    d_bb = cdist(B, B).mean()
    return float(2.0 * d_ab - d_aa - d_bb)


def energy_distance_test(A, B, n_sub: int = 500, n_perm: int = 200, seed: int = 0):
    """Whitened energy-distance two-sample statistic with a permutation p-value."""
    rng = np.random.default_rng(seed)
    A = _subsample(np.asarray(A, float), n_sub, rng)
    B = _subsample(np.asarray(B, float), n_sub, rng)
    Aw, Bw = _whiten(A, B)
    obs = _energy_distance(Aw, Bw)

    pooled = np.vstack([Aw, Bw])
    nA = len(Aw)
    count = 0
    for _ in range(n_perm):
        idx = rng.permutation(len(pooled))
        if _energy_distance(pooled[idx[:nA]], pooled[idx[nA:]]) >= obs:
            count += 1
    pvalue = (count + 1) / (n_perm + 1)
    return obs, pvalue


def compare(
    A: np.ndarray,
    B: np.ndarray,
    *,
    a_name: str = "A",
    b_name: str = "B",
    label: str | None = None,
    a_exact: bool = False,
    b_exact: bool = False,
    thresholds: Thresholds | None = None,
    energy_seed: int = 0,
) -> ComparisonResult:
    """Compare sample set ``A`` against reference ``B``.

    Set ``b_exact=True`` when ``B`` is exact i.i.d. (analytic reference) so its
    standard error uses ``sqrt(N)`` rather than ``sqrt(ESS)``.
    """
    A = np.atleast_2d(np.asarray(A, float))
    B = np.atleast_2d(np.asarray(B, float))
    if A.ndim == 1:
        A = A[:, None]
    if B.ndim == 1:
        B = B[:, None]
    d = A.shape[1]
    thresholds = thresholds or Thresholds()
    label = label or f"{a_name} vs {b_name}"

    mean_a, mean_b = A.mean(0), B.mean(0)
    std_a = A.std(0, ddof=1)
    std_b = B.std(0, ddof=1)

    ess_a = ess(A)
    ess_b = None if b_exact else ess(B)

    se_a = std_a / np.sqrt(len(A)) if a_exact else mcse_mean(A)
    se_b = std_b / np.sqrt(len(B)) if b_exact else mcse_mean(B)
    combined_se = np.sqrt(se_a**2 + se_b**2)
    mean_z = np.abs(mean_a - mean_b) / np.where(combined_se > 0, combined_se, np.inf)

    std_ratio = std_a / np.where(std_b > 0, std_b, np.inf)

    # Variance agreement as an uncertainty-aware z-score on the log scale.
    neff_a = np.full(d, len(A)) if a_exact else ess_a
    neff_b = np.full(d, len(B)) if b_exact else ess(B)
    se_logvar = np.sqrt(_log_var_se(A, neff_a) ** 2 + _log_var_se(B, neff_b) ** 2)
    var_z = np.abs(np.log(std_a**2) - np.log(std_b**2)) / np.where(
        se_logvar > 0, se_logvar, np.inf)

    corr_max_diff = 0.0
    if d >= 2:
        corr_max_diff = float(np.max(np.abs(_correlation(A) - _correlation(B))))

    energy_stat, energy_p = energy_distance_test(A, B, seed=energy_seed)

    ks_stat = np.zeros(d)
    ks_p = np.zeros(d)
    for j in range(d):
        s, p = stats.ks_2samp(A[:, j], B[:, j])
        ks_stat[j], ks_p[j] = s, p

    checks = [
        Check("mean", bool(mean_z.max() <= thresholds.mean_z), float(mean_z.max()),
              thresholds.mean_z,
              f"max mean z-score {mean_z.max():.2f} <= {thresholds.mean_z}"),
        Check("variance", bool(var_z.max() <= thresholds.var_z), float(var_z.max()),
              thresholds.var_z,
              f"max var z-score {var_z.max():.2f} <= {thresholds.var_z} "
              f"(std ratio {np.array2string(std_ratio, precision=3)})"),
        Check("energy_shape", bool(energy_p >= thresholds.energy_p_min), energy_p,
              thresholds.energy_p_min,
              f"energy permutation p {energy_p:.3f} >= {thresholds.energy_p_min}"),
    ]
    if d >= 2:
        checks.append(
            Check("correlation", bool(corr_max_diff <= thresholds.corr_tol),
                  corr_max_diff, thresholds.corr_tol,
                  f"max |corr diff| {corr_max_diff:.3f} <= {thresholds.corr_tol}"))

    return ComparisonResult(
        label=label, a_name=a_name, b_name=b_name, dim=d,
        mean_a=mean_a, mean_b=mean_b, mean_z=mean_z,
        std_a=std_a, std_b=std_b, std_ratio=std_ratio, var_z=var_z,
        corr_max_diff=corr_max_diff,
        energy_stat=energy_stat, energy_pvalue=energy_p,
        ks_stat=ks_stat, ks_pvalue=ks_p,
        ess_a=ess_a, ess_b=ess_b, checks=checks,
    )
