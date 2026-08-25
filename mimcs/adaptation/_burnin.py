"""Estimating how much of a warmup history is burn-in, instead of assuming a tenth of it.

:class:`mimcs.adaptation.ClassifierTermination` asks whether the two halves of the retained warmup
history look alike. Both criteria first drop a burn-in prefix, and that prefix used to be
``0.1 * len(history)`` --- a constant nobody measured. It has a specific failure: a chain whose
transient lasts ``T`` iterations cannot pass until the history reaches ``~10 T``, because until
then transient draws sit inside the "early" half and the classifier is right to separate them.
The stopping time is then set by the discard rule rather than by when the chain mixed.

**The three-way framing.** A good split point ``m`` is one where a classifier separates
``[:m]`` from ``[m:]`` *well* and separates the two halves of ``[m:]`` *badly*. Write
``A(m)`` for the first separation and ``B(m)`` for the second. Two searches over ``m`` are
provided, and they want different things from those two terms:

* ``"changepoint"`` --- maximize an objective in both, alternating a fit with a split-point
  update: fit ``A``, project the history onto the fitted discriminant, and move ``m`` to the
  projection's strongest changepoint. Typically converges in two or three rounds.
* ``"min_discard"`` --- ignore ``A``; take the *smallest* ``m`` whose ``B`` is within ``tail_tol``
  of the best any ``m`` achieves. Reads as "the shortest discard that makes the rest look
  stationary", needs no absolute threshold on ``B`` (only that its curve has flattened), and
  returns the lower bound exactly when there is no transient to find.

**What the separation statistic should be.** Something smooth --- see
:func:`mimcs.adaptation._logistic.log_score`. The number here only *guides* a search over ``m``;
the stop/continue decision is taken afterwards by the criterion's own held-out accuracy at the
chosen ``m``, whose calibration (``docs/design/10_warmup_termination.md``) is thereby untouched.
Held-out 0/1 accuracy is a poor search statistic anyway: a short candidate prefix yields a handful
of validation rows, and a statistic granular in steps of 1/12 has no gradient to follow.

**The trap.** ``A`` is large at *any* big excursion, not only at the initial transient. On a
funnel, a late excursion into the neck looks like a changepoint; ``m`` then jumps past it, the
remaining tail looks homogeneous because it is one excursion, and warmup ends early. That is
worse than the rule this replaces and it is silent. Hence the upper bound on ``m``, and hence
``min_discard`` taking the earliest qualifying split rather than the most separating one.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import numpy as np

from .._logging import get_logger

log = get_logger(__name__)

#: ``_term_burn_count`` dispatches on these. ``"fixed"`` is the historical constant fraction.
MODES = ("fixed", "changepoint", "min_discard")

#: How ``B``'s (and ``A``'s) null is estimated. See :func:`null_constant`.
NULLS = ("none", "scaled", "permutation")


class BurnIn(NamedTuple):
    """The chosen split and the numbers behind it.

    ``sep_prefix`` (the ``A`` term) is ``nan`` for ``min_discard``, which never fits it.
    ``path`` is every candidate the search evaluated, in order --- its length is the fit count.
    """

    n: int
    objective: float
    sep_prefix: float
    sep_tail: float
    path: tuple[int, ...]


#: ``fit(idx_a, idx_b, role) -> (separation, projection)``. The two arguments are *index arrays*
#: into the history, not row blocks: the implementation is expected to keep the design matrix at
#: one fixed shape and select rows by weight, because the optimizer recompiles per shape (see
#: :func:`mimcs.adaptation._logistic.class_weights`). ``role`` is ``"prefix"`` for the ``A`` term
#: and ``"tail"`` for the ``B`` term, so the caller can keep the two warm starts apart --- the two
#: discriminants point in unrelated directions. ``projection`` is the fitted discriminant score
#: for the *whole* history, ``(n,)``; only differences along it are read, so an offset is free.
FitFn = Callable[[np.ndarray, np.ndarray, str], "tuple[float, np.ndarray]"]


# --- objectives ----------------------------------------------------------------- #

def additive(sep_prefix: float, sep_tail: float) -> float:
    """``A - B`` --- the proposal's ``A + (1 - B)``, whose ``+1`` is a constant in ``m``.

    Equal weight on "the prefix is distinguishable" and "the tail is not". Both terms carry a
    null that drifts with ``m`` (``A``'s rises as the prefix class shrinks, ``B``'s as the tail
    does), so the sum has a systematic tilt; :func:`weighted` and a null correction are the
    handles for that.
    """
    return float(sep_prefix - sep_tail)


def weighted(lam: float) -> Callable[[float, float], float]:
    """``A - lam * B``. ``lam -> 0`` is a pure changepoint search; large ``lam`` approaches the
    minimum-discard reading, which ``mode="min_discard"`` implements directly and more cheaply."""
    def objective(sep_prefix: float, sep_tail: float) -> float:
        return float(sep_prefix - lam * sep_tail)
    return objective


# --- the pieces ----------------------------------------------------------------- #

def bounds(n: int, *, min_abs: int, min_frac: float, max_frac: float,
           min_tail: int) -> tuple[int, int] | None:
    """``(lo, hi)`` for the candidate burn-in, or ``None`` if the history is too short to search.

    ``lo`` is not a claim that burn-in exists: it is the shortest prefix the prefix-classifier can
    be fitted on. ``hi`` is a safety cap rather than a statistical statement --- see the trap in
    the module docstring --- and ``min_tail`` keeps enough rows behind it for the tail's own
    two-half comparison to mean anything.
    """
    lo = max(int(min_abs), int(min_frac * n))
    hi = min(int(max_frac * n), n - int(min_tail))
    return (lo, hi) if hi > lo else None


def changepoint(s, lo: int, hi: int) -> int:
    """``argmax`` over ``m`` in ``[lo, hi]`` of the two-sample t-statistic of a scalar series.

    ``O(n)`` via prefix sums of ``s`` and ``s**2``. The statistic's *value* is meaningless here:
    the draws are autocorrelated, so it is inflated by roughly the integrated autocorrelation time
    and no tabulated null applies. Only the location of the maximum is used, as a proposal for the
    next round of the alternating search.
    """
    s = np.asarray(s, dtype=np.float64).ravel()
    n = s.shape[0]
    c1 = np.concatenate([[0.0], np.cumsum(s)])
    c2 = np.concatenate([[0.0], np.cumsum(s * s)])
    m = np.arange(lo, hi + 1, dtype=np.int64)
    n2 = n - m
    sum1, sum2 = c1[m], c1[n] - c1[m]
    sq1, sq2 = c2[m], c2[n] - c2[m]
    mean1, mean2 = sum1 / m, sum2 / n2
    # Within-segment sums of squares: sum(x^2) - mean * sum(x) for each side.
    within = (sq1 - sum1 * mean1) + (sq2 - sum2 * mean2)
    var = np.maximum(within / max(n - 2, 1), 1e-300)      # a constant series has no changepoint
    t = np.abs(mean1 - mean2) / np.sqrt(var * (1.0 / m + 1.0 / n2))
    return int(m[np.argmax(t)])


def tail_halves(n: int, m: int):
    """Indices of the two equal halves of ``[m:n]`` --- the same cut the criterion will make."""
    h = (n - m) // 2
    return np.arange(m, m + h), np.arange(m + h, m + 2 * h)


# --- the null ------------------------------------------------------------------- #
#
# Why this exists: the first version of ``min_discard`` compared ``B(m)`` against its own minimum
# over ``m``, and that is not a fair comparison, because **a shorter tail always looks more
# homogeneous**. The classifier has fewer draws at large ``m``, so it separates less well for
# reasons that have nothing to do with stationarity: ``B`` decreases with ``m`` under the null too.
# On Neal's funnel it therefore never flattened and the search ran to the upper bound on 6 of 8
# seeds (``docs/design/10``). What the rule needs is ``B(m)`` against **its own null at that tail
# length** --- an absolute reference, which then also makes "nothing qualifies" a meaningful and
# actionable answer rather than a forced choice.

def null_constant(n: int, hi: int, fit: FitFn) -> float:
    """``c`` in ``null(k) = c / k`` for a comparison of two blocks of ``k`` rows each.

    **One extra fit per check, not per candidate.** Block permutation would give a per-``m`` null
    directly, at ``n_perm`` refits for every rung of every check --- an order of magnitude more
    arithmetic than the search it corrects. Instead one *within-regime* reference is fitted and
    extrapolated by the ``1/k`` law that a ridged fit's held-out excess obeys (the usual
    overfitting term: effective parameters over sample size).

    The reference is the **last** ``n - hi`` rows split into halves --- the most-converged block in
    the history, and the shortest tail the search will ever score, so every extrapolation runs from
    the well-measured short end toward longer tails where the null is smaller.

    **If that block is not itself stationary the reference is inflated, and an inflated null is
    dangerous in the loosening direction**: the gate it defines rises, more rungs clear it, and the
    search discards *more*, not less. (This was written the other way round at first and measured
    to be wrong: on a never-settling linear drift the uncorrected rule chose the cap at 1000 and
    the corrected one still chose 800, because the reference block was drifting too.) There is no
    fully general repair --- a chain that is nowhere stationary offers no stationary reference --- so
    :func:`clamp_null` bounds how far the correction is allowed to be talked upward.
    """
    r = n - hi
    h = r // 2
    if h < 2:
        return 0.0
    sep, _ = fit(np.arange(n - 2 * h, n - h), np.arange(n - h, n), "null")
    return float(sep) * h


#: Absolute floor for "the tail looks stationary", in nats of held-out log-score.
#:
#: Measured, not chosen: over 240 stationary cells (5 problems x 8 seeds x 6 tail lengths, chains
#: started from an exact draw) the tail log-score is **exactly 0 in 67%** of them, p99 0.0052 and
#: max 0.0117. It does not need the ``1/k`` inflation correction that the *accuracy* statistic
#: needs (whose null sits at 0.52, not 0.50) --- :func:`mimcs.adaptation._logistic.log_score` is a
#: proper scoring rule clipped at zero, so a spurious direction scores no better than uninformative
#: out of sample and clips. 0.02 sits above the observed maximum with margin.
NULL_FLOOR = 0.02

#: How far above :data:`NULL_FLOOR` a *measured* null is still believed. See :func:`clamp_null`.
NULL_CAP_MULT = 3.0


def clamp_null(b0: float, null_floor: float = NULL_FLOOR,
               cap_mult: float = NULL_CAP_MULT) -> float:
    """Bound a measured null from above, at ``cap_mult * null_floor``.

    A stationary tail's separation is small --- measured, 67% of the time exactly zero and never
    above 0.0117. So a *large* measured null is not evidence that this tail length tolerates a
    large separation; it is evidence that the reference block used to measure it was **not
    stationary**, which is precisely the case where believing it does damage: the gate rises, every
    rung clears it, and the search discards to the cap. Clamping keeps the reference useful where
    it is informative (mild autocorrelation inflating the null a little) and inert where it is
    lying.
    """
    return min(float(b0), cap_mult * null_floor)


def null_terms(c: float, n: int, m: int) -> tuple[float, float]:
    """``(A0, B0)`` at split ``m`` from the constant ``c``, by the ``1/k`` law.

    ``B`` compares two blocks of ``(n-m)/2``. ``A`` compares ``m`` against ``n-m`` with the classes
    weighted equally, so its effective per-class size is their harmonic mean, ``2m(n-m)/n`` --- a
    short prefix is what limits it, which is why ``A``'s null climbs steeply as ``m`` falls.
    """
    tail = max(n - m, 1)
    return c / max(2.0 * m * tail / n, 1.0), c / max(tail / 2.0, 1.0)


def ladder(lo: int, hi: int, *, factor: float = 2.0, max_rungs: int = 8) -> list[int]:
    """Geometric rungs from ``lo`` to ``hi``. Geometric because burn-in length is a scale, not an
    offset: the interesting resolution near ``lo`` is a few dozen draws and near ``hi`` a few
    hundred, and a linear grid spends its budget in the wrong place."""
    rungs, m = [], float(lo)
    while m < hi and len(rungs) < max_rungs - 1:
        rungs.append(int(m))
        m *= factor
    rungs.append(int(hi))
    return sorted(set(rungs))


# --- the searches --------------------------------------------------------------- #

def estimate_burn_in(n: int, fit: FitFn, *, mode: str, lo: int, hi: int, init: int | None = None,
                     max_iter: int = 4, objective=additive, tail_tol: float = 0.1,
                     null: str = "none", null_slack: float = 0.5, n_perm: int = 20,
                     perm_block: int = 100, seed: int = 0) -> BurnIn:
    """Choose a burn-in length in ``[lo, hi]`` for a history of ``n`` rows.

    Only the *length* is passed: everything here works in indices, and the feature matrix itself
    lives behind ``fit``. See the module docstring for the two modes and the block above
    :func:`null_constant` for what ``null`` buys.
    """
    n = int(n)
    if null not in NULLS:
        raise ValueError(f"unknown null {null!r} (use one of {NULLS})")
    nulls = _null_fn(n, lo, hi, fit, null, n_perm=n_perm, perm_block=perm_block, seed=seed)
    if mode == "changepoint":
        return _alternating(n, fit, lo=lo, hi=hi, init=init, max_iter=max_iter,
                            objective=objective, nulls=nulls)
    if mode == "min_discard":
        return _min_discard(n, fit, lo=lo, hi=hi, tail_tol=tail_tol, nulls=nulls,
                            null_slack=null_slack)
    raise ValueError(f"unknown burn-in mode {mode!r} (use one of {MODES})")


def _null_fn(n, lo, hi, fit, null, *, n_perm, perm_block, seed):
    """``m -> (A0, B0)``, or ``None`` for no correction."""
    if null == "none":
        return None
    if null == "scaled":
        c = null_constant(n, hi, fit)
        return lambda m: null_terms(c, n, m)
    rng = np.random.default_rng(seed + n)

    def permuted(m):
        """The gold standard, kept for validating ``"scaled"``: relabel contiguous blocks of the
        tail at random and refit, which respects the autocorrelation that inflates the statistic.
        ``n_perm`` refits per candidate, so this is for study runs, not production."""
        early, late = tail_halves(n, m)
        idx = np.concatenate([early, late])
        blocks = np.array_split(idx, max(2, len(idx) // max(1, perm_block)))
        draws = []
        for _ in range(n_perm):
            order = rng.permutation(len(blocks))
            half = len(blocks) // 2
            a = np.concatenate([blocks[i] for i in order[:half]])
            b = np.concatenate([blocks[i] for i in order[half:2 * half]])
            k = min(len(a), len(b))
            draws.append(fit(a[:k], b[:k], "null")[0])
        b0 = float(np.mean(draws))
        # A's null is not measured separately -- rescale B's by the 1/k law, as "scaled" does.
        tail = max(n - m, 1)
        return b0 * (tail / 2.0) / max(2.0 * m * tail / n, 1.0), b0

    return permuted


def _alternating(n: int, fit: FitFn, *, lo, hi, init, max_iter, objective, nulls) -> BurnIn:
    """Alternate "fit at the current split" with "re-split under the current fit"."""
    m = int(np.clip(lo if init is None else init, lo, hi))
    path: list[int] = []
    best: BurnIn | None = None
    for _ in range(max(1, int(max_iter))):
        path.append(m)
        early, late = tail_halves(n, m)
        sep_a, projection = fit(np.arange(0, m), np.arange(m, n), "prefix")
        sep_b, _ = fit(early, late, "tail")
        a0, b0 = nulls(m) if nulls is not None else (0.0, 0.0)
        candidate = BurnIn(m, objective(sep_a - a0, sep_b - b0), sep_a, sep_b, ())
        if best is None or candidate.objective > best.objective:
            best = candidate
        proposal = changepoint(projection, lo, hi)
        # A repeat means the iteration has reached a fixed point (or is cycling between two
        # splits it has already scored); either way there is nothing new left to evaluate.
        if proposal in path:
            break
        m = proposal
    assert best is not None
    return best._replace(path=tuple(path))


def _min_discard(n: int, fit: FitFn, *, lo, hi, tail_tol, nulls, null_slack,
                 null_floor: float = NULL_FLOOR) -> BurnIn:
    """The earliest split at which the tail stops looking non-stationary.

    Two readings of "stops", and the difference is the whole finding above:

    * **Without a null** (``nulls is None``): the earliest rung within ``tail_tol`` of the *range's
      own floor*. Purely relative, and therefore forced to nominate something even when ``B`` is
      merely sliding downward with tail length. This is the rule that ran away on the funnel.
    * **With a null**: the earliest rung whose ``B`` is within ``null_slack`` of ``B0`` at that
      tail length. Absolute, so "no rung qualifies" is a real answer --- and it means no prefix
      discard makes the rest look stationary, which is exactly the funnel's situation. The
      fallback is then ``lo``: **discard as little as possible**, because there is no evidence
      that throwing draws away helps. That is what stops the runaway.
    """
    path: list[int] = []
    scored: dict[int, float] = {}
    floors: dict[int, float] = {}

    def sep(m: int) -> float:
        if m not in scored:
            early, late = tail_halves(n, m)
            scored[m] = fit(early, late, "tail")[0]
            if nulls is not None:
                floors[m] = nulls(m)[1]
            path.append(m)
        return scored[m]

    def choose() -> int:
        if nulls is None:
            return _earliest_flat(scored, tail_tol)
        return _earliest_below_null(scored, floors, null_slack, lo, null_floor)

    rungs = ladder(lo, hi)
    for m in rungs:
        sep(m)
    chosen = choose()

    # One refinement between the chosen rung and the one before it: the ladder's steps double, so
    # a coarse hit can overshoot the true changepoint by up to a factor of two.
    below = [m for m in rungs if m < chosen]
    if below:
        midpoint = (below[-1] + chosen) // 2
        if midpoint not in scored:
            sep(midpoint)
            chosen = choose()

    excess = scored[chosen] - floors.get(chosen, 0.0)
    return BurnIn(chosen, -excess, float("nan"), scored[chosen], tuple(path))


def _earliest_flat(scored: dict[int, float], tail_tol: float) -> int:
    """The smallest ``m`` whose separation is within ``tail_tol`` of the range's bottom."""
    ms = sorted(scored)
    values = np.array([scored[m] for m in ms])
    floor, ceiling = values.min(), values.max()
    cutoff = floor + tail_tol * (ceiling - floor)
    return int(ms[int(np.argmax(values <= cutoff))])


def _earliest_below_null(scored, floors, null_slack: float, lo: int,
                         null_floor: float = NULL_FLOOR) -> int:
    """The smallest ``m`` whose tail is indistinguishable from stationary; ``lo`` if there is none.

    The gate is ``max((1 + null_slack) * B0(m), null_floor)``. Both halves earn their place: the
    measured ``B0`` carries the tail-length dependence when the reference fit finds any signal at
    all, and :data:`NULL_FLOOR` covers the common case where it does not --- the reference clips to
    exactly zero two thirds of the time, and a gate of "``B`` must be exactly 0" is a knife edge.
    """
    for m in sorted(scored):
        gate = max((1.0 + null_slack) * clamp_null(floors[m], null_floor), null_floor)
        if scored[m] <= gate:
            return int(m)
    return int(lo)
