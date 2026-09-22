"""Evidence-informed mass-matrix mode selection: diagonal / low-rank(J) / dense.

A block's per-iteration scores (potential gradients) have covariance ``E[g g^T]`` = the target
*precision*, i.e. the ideal mass. This module reads the block's structure off that covariance and
chooses the mass form. The decision is a property of the **whitened** score covariance

    R = D^{-1/2} S D^{-1/2},   S = cov(g),  D = diag(S)

a sample *correlation* matrix (unit diagonal, trace ``d``). Its spectrum tells us:

* a bulk with nothing isolated above it -> **diagonal** suffices;
* a few eigenvalues isolated above the Marchenko-Pastur bulk -> **low-rank(J)**, ``J`` the spike
  count (a true spike pops above the MP edge ``(1+sqrt(d/n))^2`` past the BBP threshold);
* structure richer than the low-rank cap -> **dense**, where it is estimable.

Three candidate decision rules are provided for comparison (see the study
``tests/experiments/mode_select_study.py``); :func:`select_mass_mode` dispatches and then applies
the boundary gates. All are ``(d, n)``-aware.

**Detection is not the binding constraint --- estimation is.** With ``n`` draws in ``d``
dimensions, the number of directions the spectrum lets you *detect* far exceeds the number you can
*aim* well enough to invert into a mass matrix. Measured on synthetic rank-20 structure at
``d = 400``, the sine of the largest principal angle between the fitted and the true subspace is
0.60 at ``n = 0.75 d`` and still 0.57 at ``n = 2 d``, reaching 0.29 only by ``n = 10 d``. A mass
built from misaimed directions is wrong precisely in the stiff directions it claims to fix, which
is how a 500-draw pilot on a 2000-coordinate block earned a correctly-detected ``lowrank(52)`` that
drove NUTS's step size to 1e-24 at 100% divergences. Hence :data:`RANK_GUARD_FRAC`: below it, no
non-diagonal mode is considered at all, whatever the spectrum says.

**Everything is measured against the bulk, never against an absolute number.** The spread statistic
is ``max(eig) / bulk_scale``, and the bulk scale is estimated as ``median(eig) / mp_median(gamma)``
--- the sample median divided by the median of the Marchenko-Pastur law it is drawn from. Without
that ``mp_median`` correction the statistic is *not* calibrated: the MP median is 0.65 at
``gamma = 1`` and 0.93 at ``gamma = 0.2``, so a raw ``max/median`` inflates by exactly that factor
and a fixed threshold means something different at every ``(n, d)``. Measured on pure isotropic
noise at ``d = 200``, the raw ratio clears the bulk edge in 100% of seeds for every
``0.9 d <= n <= 3 d`` --- the whole band production pilots live in --- while the corrected one
falls below it in 100%.

**Three gates make the test honest about a chain and about value**
(``tests/experiments/writeups/mode_select_gates.md``). The rows are a chain's draws, and the MP
edge is a *significance* test, so two questions were missing:

* *How many independent rows are they?* :func:`effective_sample_size`, by default the pooled ESS of
  the scores' second moments, replaces ``n`` in the rank guard, the bulk scale, the edge, AIC and
  the dense row gate. On isotropic scores at IACT 10 the row count calls structure 100% of the
  time.
* *Is a direction worth a rank-one term?* Each eigenvalue is mapped back to its population
  stiffness (:func:`_bbp_inverse`) and must reach :data:`MIN_STIFFNESS` (2). The floor *truncates*
  ``J``; it never lowers ``jmax``, which would turn "fewer directions suffice" into a dense
  verdict. A single direction pays only from ``~sqrt(d)`` on, but a per-direction floor there
  strips broad structure that pays collectively, so the default stays at 2.
* *Is it there throughout the pilot?* Each half's top directions must clear the other half's
  noise out of sample (:func:`_replicated`), which rejects structure an unsettled pilot shows in
  one part only.
"""

from __future__ import annotations

import math

import numpy as np

from .._logging import get_logger

log = get_logger(__name__)

# --- dimension gates (see the module docstring / docs/design/09) -------------- #
SMALL_DENSE_DIM = 10          # d <= this: dense unless the spectrum is within its own bulk
RANK_GUARD_FRAC = 0.75        # n >= this * d before ANY non-diagonal mode is considered
J_MAX_FRAC = 0.2              # J <= floor(J_MAX_FRAC * d)  (user ceiling: d/5)
DENSE_HARD_MAX = 1000         # dense numerically out of the question above this d
DENSE_HEUR_MAX = 200          # dense is only proposed up to this d ...
DENSE_MIN_ROWS_MULT = 10      # ... and only with n >= this * d
BBP_MARGIN = 0.05             # keep eigenvalues > (1 + BBP_MARGIN) * MP edge
#: the *diagonal gate* needs more headroom than spike detection does. Measured on pure isotropic
#: noise, the 95th-percentile spread sits only 3-7% below the BBP edge at every ``(n, d)``, so
#: `BBP_MARGIN` alone leaves nothing for real data: scores whitened in-sample by a fitted metric
#: carry residual structure and tip over it. This margin governs "is anything above the bulk at
#: all"; `BBP_MARGIN` still governs "which eigenvalues are spikes", and being the looser of the two
#: it keeps the spike count >= 1 whenever this gate is passed.
DIAGONAL_MARGIN = 0.25
AIC_PENALTY = 2.0             # per-parameter penalty (2 = AIC; log n = BIC)
#: How the selector counts its rows (see :func:`effective_sample_size`). ``None`` takes the row
#: count ``n`` at face value, i.e. assumes independent rows; the others estimate an effective
#: sample size of the scores' second moments, which is what the Marchenko-Pastur bulk's width
#: depends on. Read at call time. Default ``"second_moment"``: on isotropic scores at IACT 10 the
#: row count picks a non-diagonal mode 100% of the time, the second-moment ESS 0% once
#: ``n_eff >= 2 d`` (``tests/experiments/writeups/mode_select_gates.md``).
MODE_SELECT_ESS = "second_moment"
ESS_KINDS = (None, "pooled", "second_moment", "min_coord", "min_coord_products")
#: ``"min_coord_products"`` includes the cross products ``h_j h_k`` only up to this dimension.
PRODUCT_ESS_MAX_DIM = 100
#: The usefulness floor: a direction counts toward a low-rank mode only if its de-biased
#: stiffness (population eigenvalue of the whitened score covariance) is at least this. A number
#: is an absolute floor; ``"sqrt_d"`` is ``max(STIFFNESS_FLOOR_MIN, sqrt(d))``; ``None`` turns the
#: floor off, leaving only the significance tests. Default ``2.0``. ``"sqrt_d"`` is where a *single*
#: direction starts to pay on a Gaussian (NUTS's step size is set by the bulk's total energy error,
#: ``eps ~ d^(-1/4)``, so one direction of stiffness ``l`` binds only once ``l ~ sqrt(d)``: at d = 100
#: nothing up to ``l ~ 4.5``, 2.2x at ``l ~ 17``; at d = 500 nothing up to ``l ~ 19``, even with the
#: exact mass). It is not the default because broad structure pays *collectively*: on a correlated
#: horseshoe regression it removed a dense shape worth 1.12x, costing 0.89x.
MIN_STIFFNESS = 2.0
STIFFNESS_FLOOR_MIN = 2.0
#: Split replication: a direction must reappear out of sample between the two halves of the rows.
#: On by default: it is the only gate that rejects structure present in part of a pilot (a strong
#: first-half-only spike passes the floor on 100% of seeds and replication on 0%), and it kept
#: every stationary spike the floor keeps.
REPLICATE = True
REPLICATE_ALPHA = 0.05
#: a bulk scale at or below this means the spectrum is degenerate (a pilot whose rows repeat), and
#: no ratio against it is meaningful. Refused rather than floored --- see :func:`_bulk_scale`.
MIN_BULK_SCALE = 1e-8


# --- whitened spectrum ------------------------------------------------------- #

def _eigs_desc(H: np.ndarray, n: int, d: int) -> np.ndarray:
    """Descending eigenvalues of ``R = (1/n) H^T H`` (H whitened, ``(n, d)``), zero-padded to ``d``.
    Uses the ``n x n`` Gram when ``n < d`` (same nonzero spectrum, far cheaper)."""
    gram = (H @ H.T) if n < d else (H.T @ H)
    w = np.clip(np.linalg.eigvalsh(gram / n), 0.0, None)[::-1]
    eig = np.zeros(d)
    eig[: w.shape[0]] = w
    return eig


def whitened_spectrum(scores: np.ndarray):
    """``(D, eigenvalues, n, d)`` for a block's ``(n, d)`` scores: ``D = diag(cov)`` and the
    descending eigenvalues of the whitened sample covariance ``R`` (a correlation matrix)."""
    scores = np.asarray(scores, dtype=float)
    n, d = scores.shape
    g = scores - scores.mean(axis=0)
    D = np.maximum((g * g).mean(axis=0), 1e-300)      # centred sample variance per coordinate
    H = g / np.sqrt(D)                                 # unit-variance columns; R = (1/n) H^T H
    return D, _eigs_desc(H, n, d), n, d


def _n_computable(n: int, d: int) -> int:
    """How many eigenvalues carry information: ``min(n - 1, d)``.

    ``_eigs_desc`` zero-pads to ``d``, and those trailing zeros are an artifact of having fewer rows
    than dimensions rather than flat directions of the target --- a median taken over them is
    exactly 0 for ``n <= d/2``. The ``n - 1`` (not ``n``) is because :func:`whitened_spectrum`
    *centres* the scores, which costs one rank, so ``eig[n-1]`` is structurally zero too.
    """
    return max(1, min(n - 1, d))


def _effective_rows(eig: np.ndarray, n: int, d: int) -> int:
    """Rows the spectrum can actually speak for: ``min(n, numerical_rank + 1)``.

    The rank guard asks whether there is enough evidence to aim a direction, and a *row count* is
    the wrong measure of that when the rows repeat. A chain that rejects heavily stores the same
    state many times, so its score matrix is rank-deficient in a way ``n`` does not reveal: 500
    draws from 120 distinct states at ``d = 200`` look like ``n/d = 2.5`` and are really
    ``n/d = 0.6``. The numerical rank is free here --- it is the count of nonzero eigenvalues,
    already computed --- and the ``+ 1`` restores the rank that centring cost, so a clean pilot
    gets exactly ``n`` back and the guard is unchanged for it.
    """
    if eig[0] <= 0.0:
        return 0
    return int(min(n, np.sum(eig > 1e-10 * eig[0]) + 1))


def _mp_median(gamma: float, _cache: dict = {}) -> float:
    """Median of the Marchenko-Pastur law with ratio ``gamma = d/n`` (continuous part only).

    The bulk of a correlation matrix is MP-distributed, not concentrated at 1, and its median moves
    a long way with ``gamma``: 0.65 at ``gamma = 1``, 0.83 at 0.5, 0.97 at 0.1. Dividing the sample
    median by this is what turns it into an estimate of the bulk *scale*, so that
    ``max(eig) / scale`` can be compared against the MP edge on the same footing at every
    ``(n, d)``. This is the Gavish-Donoho bulk-scale fit that :func:`_mp_spikes` used to note as
    possible-but-unnecessary; it is necessary.

    Integrated in ``t = sqrt(x)``, which cancels the ``x^{-1/2}`` singularity the density has at
    the left edge when ``gamma -> 1``. Memoised on ``gamma`` rounded to 6 places: it is a pure
    function of the shape, and a block's ``(n, d)`` repeats across a run.
    """
    key = round(float(gamma), 6)
    if key in _cache:
        return _cache[key]
    m = 40000
    a, b = (1.0 - math.sqrt(gamma)) ** 2, (1.0 + math.sqrt(gamma)) ** 2
    ta, tb = math.sqrt(max(a, 0.0)), math.sqrt(b)
    t = np.linspace(ta, tb, m + 1)
    t = 0.5 * (t[:-1] + t[1:])                         # midpoints
    x = t * t
    dens = np.sqrt(np.clip((b - x) * (x - a), 0.0, None)) / (2.0 * np.pi * gamma * x)
    w = dens * 2.0 * t * (tb - ta) / m                 # dx = 2 t dt
    total = w.sum()
    val = 1.0 if total <= 0 else float(x[np.searchsorted(np.cumsum(w) / total, 0.5)])
    _cache[key] = val
    return val


def effective_sample_size(scores: np.ndarray, kind) -> float:
    """The number of rows the spectrum's noise corresponds to.

    * ``None``: ``n``, the rows taken as independent (the behaviour before ESS was introduced).
    * ``"pooled"``: the smaller of the pooled column ESS and the pooled second-moment ESS
      (:func:`mimcs.diagnostics.pooled_ess`, :func:`~mimcs.diagnostics.second_moment_ess`).
    * ``"second_moment"``: the pooled second-moment ESS alone. A correlation matrix is an average
      of the products ``h_j h_k``, so its noise is theirs; the columns' own ESS governs the mean.
    * ``"min_coord"``: the smallest per-coordinate ESS of ``h_j`` and ``h_j^2``, the convention
      the sample summaries use for a Euclidean parameter.
    * ``"min_coord_products"``: that, plus every product ``h_j h_k`` when ``d`` is at most
      :data:`PRODUCT_ESS_MAX_DIM`.

    The minimum over many noisy per-coordinate estimates is biased low even on independent rows,
    by more the more coordinates there are; pooling is not.
    """
    from ..diagnostics import ess, pooled_ess, second_moment_ess
    scores = np.asarray(scores, dtype=float)
    n, d = scores.shape
    if kind is None or n < 4:
        return float(n)
    g = scores - scores.mean(axis=0)
    H = g / np.sqrt(np.maximum((g * g).mean(axis=0), 1e-300))
    if kind == "pooled":
        return min(pooled_ess(H), second_moment_ess(H))
    if kind == "second_moment":
        return second_moment_ess(H)
    if kind in ("min_coord", "min_coord_products"):
        m = min(float(ess(H).min()), float(ess(H * H).min()))
        if kind == "min_coord_products" and 1 < d <= PRODUCT_ESS_MAX_DIM:
            j, k = np.triu_indices(d, 1)
            m = min(m, float(ess(H[:, j] * H[:, k]).min()))
        return m
    raise ValueError(f"unknown ESS kind {kind!r}; expected one of {ESS_KINDS}")


def _bbp_inverse(x: np.ndarray, c: float) -> np.ndarray:
    """Population spike ``l`` behind a sample eigenvalue ``x`` (in bulk units) of a correlation
    matrix at aspect ratio ``c = d/n``: the inverse of the BBP map ``x = l + c l/(l - 1)``, valid
    above the bulk edge ``(1 + sqrt(c))^2``. Below the edge the spike is not identifiable and 1 is
    returned. This is the right map for *in-sample* eigenvalues, the batch case here."""
    x = np.asarray(x, dtype=float)
    b = x + 1.0 - c
    disc = b * b - 4.0 * x
    ell = 0.5 * (b + np.sqrt(np.clip(disc, 0.0, None)))
    return np.where(x > (1.0 + math.sqrt(c)) ** 2, ell, 1.0)


def _replicated(scores: np.ndarray, K: int, n_eff: float, alpha: float) -> int:
    """How many of the top-``K`` directions replicate out of sample between the rows' halves.

    Each half's top eigenvectors are scored by their Rayleigh quotient in the *other* half's
    correlation matrix. A fixed direction's out-of-sample quotient carries no Marchenko-Pastur
    inflation: with no structure along it, it is ``1`` with sd ``sqrt(2/n_half)``, ``n_half`` the
    half's effective size. Direction ``j`` replicates when both cross-fitted quotients clear
    ``1 + z sqrt(2/n_half)``, ``z`` one-sided and Bonferroni-corrected over the ``K`` tested. The
    halves are contiguous, so structure present in only one part of the pilot (a non-stationary
    one) fails. Returns the number of *leading* directions that replicate.
    """
    from statistics import NormalDist
    n = scores.shape[0]
    half = n // 2

    def corr(x):
        g = x - x.mean(axis=0)
        H = g / np.sqrt(np.maximum((g * g).mean(axis=0), 1e-300))
        return H, H.T @ H / H.shape[0]

    HA, RA = corr(scores[:half])
    HB, RB = corr(scores[half:])
    VA = np.linalg.eigh(RA)[1][:, ::-1][:, :K]
    VB = np.linalg.eigh(RB)[1][:, ::-1][:, :K]
    qAB = np.sum((HB @ VA) ** 2, axis=0) / HB.shape[0]
    qBA = np.sum((HA @ VB) ** 2, axis=0) / HA.shape[0]
    z = NormalDist().inv_cdf(1.0 - alpha / max(K, 1))
    thr = 1.0 + z * math.sqrt(2.0 / max(n_eff / 2.0, 1.0))
    ok = np.minimum(qAB, qBA) > thr
    return int(np.argmin(ok)) if not ok.all() else K


def _bulk_scale(eig: np.ndarray, n: int, d: int, n_eff: float | None = None) -> float:
    """The bulk's scale: the sample median of the computable eigenvalues, de-biased by
    :func:`_mp_median` at the effective ratio ``d / n_eff`` (``n_eff`` defaults to ``n``). ``~1``
    when ``R ~ I`` at any ``(n, d)``.

    Returns ``0.0`` for a **degenerate** spectrum, which the caller refuses outright rather than
    flooring --- flooring would turn a detectable pathology into a plausible-looking rank.

    This is belt-and-braces, not load-bearing: the median over ``eig[:min(n-1,d)]`` can only
    collapse when the numerical rank is below half of that window, and :func:`_effective_rows`
    already refuses anything under ``RANK_GUARD_FRAC * d``, which is the stricter bound. Verified
    by mutation --- deleting this guard alone changes no test. It is kept because the cost is two
    lines and the failure it prevents (dividing by ~1e-15) is silent and catastrophic, but it must
    not be mistaken for the thing that handles a rejecting pilot. That is the rank guard.
    """
    med = float(np.median(eig[: _n_computable(n, d)]))
    if not math.isfinite(med) or med <= MIN_BULK_SCALE:
        return 0.0
    return med / _mp_median(d / (n if n_eff is None else n_eff))


def _bulk_edge(n: int, d: int, margin: float = BBP_MARGIN) -> float:
    """The Marchenko-Pastur right edge with a margin, in units of the bulk scale."""
    return (1.0 + math.sqrt(d / n)) ** 2 * (1.0 + margin)


# --- rule backends ----------------------------------------------------------- #

def _mp_spikes(eig: np.ndarray, n: int, d: int, scale: float = 1.0) -> int:
    """Rule A -- Marchenko-Pastur / BBP: count eigenvalues above the bulk right edge.

    ``scale`` is the bulk scale the edge is expressed in. The default of 1 is right for *this*
    rule and is not an oversight: ``trace(R) = d`` exactly, zero padding included, which is the
    normalization in which the edge is ``(1 + sqrt(d/n))^2``. Measured, it returns 0 on pure noise
    at every ``n/d`` from 0.75 to 5 and exactly 3 on three true spikes --- whereas passing the
    *estimated* scale over-counts (5 against a true 3 at ``d = 80``), because strong spikes deflate
    the bulk and a deflated correlation bulk is not a scaled MP law. The estimated scale is used
    for the ``jmax`` cap instead, where over-counting only loosens a ceiling.
    """
    return int(np.sum(eig >= _bulk_edge(n, d) * scale))


def _kl_benefit(eig: np.ndarray) -> np.ndarray:
    """Per-direction KL benefit of capturing eigenvalue ``lambda``: ``(lambda - 1 - log lambda)/2``
    (>= 0, zero at ``lambda = 1``; the drop in the batch-KL mass loss from fitting that direction)."""
    lam = np.clip(eig, 1e-12, None)
    return 0.5 * (lam - 1.0 - np.log(lam))


def _aic_mode(eig: np.ndarray, n: int, d: int, jmax: int, dense_ok: bool,
              penalty: float = AIC_PENALTY):
    """Rule B -- AIC-penalised KL over the low-rank order J. ``AIC(J) - AIC(0) = penalty *
    (params_lr(J) - d) - 2 n * cum_benefit(J)`` with the KL loss decomposed over the whitened
    eigenvalues (low-rank captures only ``lambda > 1``: the shape stiffens only). Its argmin J* is
    the number of directions worth a rank-one term -- the bulk directions individually fail the
    marginal penalty, so J* is *not* inflated by the Marchenko-Pastur bulk. Dense is the fallback
    when the structure is richer than the low-rank cap (J* > jmax), NOT a raw aggregate comparison
    against identity (which would overfit the bulk: an explicit dense-vs-identity AIC term scores
    -44 on pure isotropic noise at d=20, n=1.05d, i.e. it picks dense exactly where dense is least
    estimable --- which is why the fallback is gated on row count instead). Returns ``(kind, J)``."""
    best_J = _aic_order(eig, n, d, penalty)
    if best_J == 0:
        return ("diagonal", 0)
    if best_J <= jmax:
        return ("lowrank", best_J)
    return ("dense", 0) if dense_ok else ("lowrank", jmax)


def _aic_order(eig: np.ndarray, n: float, d: int, penalty: float = AIC_PENALTY) -> int:
    """The AIC-best low-rank order J* of :func:`_aic_mode`, before any cap. ``n`` may be an
    effective sample size: the log-likelihood of dependent rows counts as ``n_eff`` independent
    ones (a quasi-likelihood), which keeps the per-direction test equal to the MP edge at
    ``n_eff``."""
    ben = _kl_benefit(eig)
    n_pos = int(np.sum(eig > 1.0))                        # only lambda > 1 directions are low-rank-able
    best_J, best_score, cum = 0, 0.0, 0.0                 # score = AIC(J) - AIC(0); diagonal is 0
    for J in range(1, min(n_pos, d - 1) + 1):
        cum += ben[J - 1]                                 # eig descending, so top-J
        sc = penalty * (J + (J * d - J * (J + 1) // 2)) - 2.0 * n * cum
        if sc < best_score:
            best_score, best_J = sc, J
    return best_J


def _parallel_spikes(scores: np.ndarray, n: int, d: int, *, n_perm: int = 20,
                     alpha: float = 0.05, seed: int = 0) -> int:
    """Rule C -- Horn's parallel analysis: keep eigenvalues above the per-rank ``(1-alpha)`` quantile
    of a column-permutation null (permuting each column independently kills cross-correlation).

    Needs no bulk-scale argument: the null is built from the same ``H``, so the scale cancels."""
    rng = np.random.default_rng(seed)
    g = scores - scores.mean(axis=0)
    H = g / np.sqrt(np.maximum((g * g).mean(axis=0), 1e-300))
    obs = _eigs_desc(H, n, d)
    null = np.empty((n_perm, d))
    for p in range(n_perm):
        Hp = np.column_stack([rng.permutation(H[:, j]) for j in range(d)])
        null[p] = _eigs_desc(Hp, n, d)
    return int(np.sum(obs > np.quantile(null, 1.0 - alpha, axis=0)))


# --- the selector ------------------------------------------------------------ #

_DEFAULT = object()


def select_mass_mode(scores: np.ndarray, *, rule: str = "aic", penalty: float = AIC_PENALTY,
                     n_perm: int = 20, seed: int = 0, ess=_DEFAULT, min_stiffness=_DEFAULT,
                     replicate=_DEFAULT, info: dict | None = None):
    """Choose a block's mass mode from its ``(n, d)`` scores. ``rule`` in {"aic", "mp", "parallel"}.
    Returns ``(kind, rank)``: ``("diagonal", 0)`` / ``("lowrank", J)`` / ``("dense", 0)``.

    Diagonal is the answer whenever the evidence cannot support anything else --- too few rows for
    the block's width, a degenerate spectrum, or nothing isolated above the bulk. That is a real
    verdict, not a failure: a diagonal mass is always usable, while a nondiagonal one fitted from
    insufficient evidence is worse than none (see the module docstring).

    The rows are a chain's draws in order. Three gates read them as such, each defaulting to its
    module constant, read at call time:

    * ``ess`` (:data:`MODE_SELECT_ESS`): how many independent rows they are worth. That effective
      count replaces ``n`` in the rank guard, the bulk scale, the bulk edge, AIC and the dense
      row gate; the numerical quantities (the Gram path, ``jmax``'s ``n - 1``) keep ``n``.
    * ``min_stiffness`` (:data:`MIN_STIFFNESS`): the usefulness floor. Directions whose de-biased
      stiffness falls below it are not counted, which *truncates* the low-rank order rather than
      capping it, so a spectrum with a few useful directions over many weak ones stays low-rank.
    * ``replicate`` (:data:`REPLICATE`): keep only directions that replicate out of sample between
      the two halves of the rows.

    ``info``, when given, is filled with the quantities behind the decision."""
    if ess is _DEFAULT:
        ess = MODE_SELECT_ESS
    if min_stiffness is _DEFAULT:
        min_stiffness = MIN_STIFFNESS
    if replicate is _DEFAULT:
        replicate = REPLICATE
    scores = np.asarray(scores, dtype=float)
    D, eig, n, d = whitened_spectrum(scores)
    if min_stiffness == "sqrt_d":
        min_stiffness = max(STIFFNESS_FLOOR_MIN, math.sqrt(d))
    n_eff = effective_sample_size(scores, ess) if n >= 2 else float(n)
    dense_ok = (n > d) and (d <= DENSE_HARD_MAX)
    scale = _bulk_scale(eig, n, d, n_eff) if n >= 2 else 0.0
    spread = float(eig[0] / scale) if scale > 0.0 else float("inf")
    edge = _bulk_edge(n_eff, d, DIAGONAL_MARGIN) if n >= 1 else float("inf")
    rec = info if info is not None else {}
    rec.update(n=n, d=d, n_eff=n_eff, scale=scale, spread=spread, edge=edge)

    def decided(kind, J, why):
        """Report the decision and what drove it, then hand it back."""
        rec.update(kind=kind, J=J, why=why)
        log.debug("mass mode for a (%d, %d) score block (n_eff %.0f): spread %.3g vs bulk edge "
                  "%.3g (bulk scale %.3g), rule %r -> %s%s (%s)", n, d, n_eff, spread, edge, scale,
                  rule, kind, f" J={J}" if kind == "lowrank" else "", why)
        return (kind, J)

    # --- gates that do not look at the spectrum's shape at all ---------------- #
    if n < 2:
        return decided("diagonal", 0, "fewer than 2 rows: nothing is estimable")
    # The rank guard. Detection is cheap and estimation is not: below this the directions cannot be
    # aimed well enough to invert into a mass, however clearly the spectrum shows them.
    rows = min(_effective_rows(eig, n, d), n_eff)
    if rows < RANK_GUARD_FRAC * d:
        return decided("diagonal", 0,
                       f"{rows:.0f} effective row(s) < {RANK_GUARD_FRAC} * d: too few to estimate "
                       f"a direction")
    if scale <= 0.0:
        return decided("diagonal", 0, "degenerate spectrum (a pilot whose rows repeat)")

    # --- one statistic, one calibration -------------------------------------- #
    # Everything below compares against `edge * scale`, the MP right edge in units of the estimated
    # bulk scale. Nothing is compared against an absolute constant.
    if spread < edge:
        return decided("diagonal", 0, "nothing isolated above the bulk")
    # De-biased stiffness of every direction, for the usefulness floor.
    ell = _bbp_inverse(eig / scale, d / n_eff)
    rec["ell"] = ell[: min(d, 20)].tolist()
    if min_stiffness is not None and ell[0] < min_stiffness:
        return decided("diagonal", 0, f"top stiffness {ell[0]:.3g} < the floor {min_stiffness}")
    if d <= SMALL_DENSE_DIM:
        if not dense_ok:
            return decided("diagonal", 0, f"d <= {SMALL_DENSE_DIM} but n <= d")
        if replicate and _replicated(scores, 1, n_eff, REPLICATE_ALPHA) == 0:
            return decided("diagonal", 0, "the top direction does not replicate across halves")
        return decided("dense", 0, f"d <= {SMALL_DENSE_DIM} and structured")

    # `jmax` counts what is actually above the bulk, not the bulk's own upper half. A fixed
    # multiple of the median does the latter: `2 * median` counts ~40 of 200 eigenvalues on pure
    # noise at n = 0.75 d -- the same count it gives with 8 real spikes -- because at that gamma
    # the bulk spans [0.02, 4.64] and 2 * median sits inside it.
    n_above = _mp_spikes(eig, n_eff, d, scale)  # BBP edge, in units of the bulk scale
    # >= 1 by construction: passing the gate means eig[0] >= (1+DIAGONAL_MARGIN) * bulk * scale,
    # and this count is taken against the looser (1+BBP_MARGIN) * bulk * scale, so the top
    # eigenvalue clears it too. Both comparisons are `>=`, so no boundary case slips between them.
    jmax = min(int(math.floor(J_MAX_FRAC * d)), n - 1, d - 1, n_above)
    # "dense is estimable" means one thing in this module: enough effective rows for d(d+1)/2.
    dense_heur = dense_ok and (d <= DENSE_HEUR_MAX) and (n_eff >= DENSE_MIN_ROWS_MULT * d)

    if rule == "aic":
        # The candidate order: AIC's J*. Dense is reachable only when it exceeds jmax.
        J_star = _aic_order(eig, n_eff, d, penalty)
        why = f"AIC-penalised KL over the low-rank order (penalty {penalty})"
        if J_star == 0:
            return decided("diagonal", 0, "AIC found no direction worth a rank-one term")
    else:
        # spike-count rules (mp / parallel): count spikes, then map to a mode.
        J_star = (_mp_spikes(eig, n_eff, d) if rule == "mp"
                  else _parallel_spikes(scores, n, d, n_perm=n_perm, seed=seed))
        why = f"{J_star} spike(s) above the null bulk"
        if J_star == 0:
            return decided("diagonal", 0, "no eigenvalue above the null bulk")
    rec["J_star"] = J_star
    # The usefulness floor truncates the order (it never lowers jmax, which would turn a spectrum
    # with a few useful directions into a dense verdict).
    J_use = J_star
    if min_stiffness is not None:
        J_use = int(np.sum(ell[:J_star] >= min_stiffness))
        rec["J_use"] = J_use
        if J_use == 0:
            return decided("diagonal", 0, f"no direction reaches the stiffness floor {min_stiffness}")
    if J_use <= jmax:
        kind, J = "lowrank", J_use
    elif dense_heur:
        kind, J = "dense", 0
        why += f"; {J_use} directions > jmax, and dense is estimable here"
    elif jmax > 0:
        # jmax can be 0 when `parallel` counts a spike the MP edge does not; a rank-zero low-rank
        # mass is not a thing, and `lowrank.inv_factors` would silently drop the write rather than
        # raise on a (0, d) factor array.
        kind, J = "lowrank", jmax
        why += f"; {J_use} directions > jmax, dense not estimable"
    else:
        return decided("diagonal", 0, "no eigenvalue above the bulk edge")
    if replicate:
        K = J if kind == "lowrank" else min(jmax + 1, d - 1)
        J_rep = _replicated(scores, K, n_eff, REPLICATE_ALPHA)
        rec["J_rep"] = J_rep
        if J_rep == 0:
            return decided("diagonal", 0, "no direction replicates across the halves")
        if J_rep < K:
            kind, J = "lowrank", J_rep
            why += f"; {J_rep} of {K} directions replicate"
    return decided(kind, J, why)
