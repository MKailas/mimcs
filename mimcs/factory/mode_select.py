"""Evidence-informed mass-matrix mode selection: diagonal / low-rank(J) / dense.

A block's per-iteration scores (potential gradients) have covariance ``E[g g^T]`` = the target
*precision*, i.e. the ideal mass. This module reads the block's structure off that covariance and
chooses the mass form. The decision is a property of the **whitened** score covariance

    R = D^{-1/2} S D^{-1/2},   S = cov(g),  D = diag(S)

a sample *correlation* matrix (unit diagonal, mean eigenvalue 1). Its spectrum tells us:

* ``R ~ I`` (spread within ~1 order of magnitude) -> **diagonal** suffices;
* a few eigenvalues isolated above the Marchenko-Pastur bulk -> **low-rank(J)**, ``J`` the spike
  count (a true spike pops above the MP edge ``(1+sqrt(d/n))^2`` past the BBP threshold);
* a wide / broadly-structured bulk -> **dense** (only feasible in low dimension).

Three candidate decision rules are provided for comparison (see the study
``tests/experiments/mode_select_study.py``); :func:`select_mass_mode` dispatches and then applies
dimension-aware boundary gates. All are ``(d, n)``-aware and handle ``d > n`` (dense excluded).
"""

from __future__ import annotations

import math

import numpy as np

from .._logging import get_logger

log = get_logger(__name__)

# --- dimension gates (see the module docstring / plan) ----------------------- #
SMALL_DENSE_DIM = 10          # d <= this: dense unless essentially isotropic
COND_DIAGONAL = 10.0          # cond(R) < this (spectrum < 1 order of magnitude): diagonal
J_MAX_FRAC = 0.2              # J <= floor(J_MAX_FRAC * d)  (user ceiling: d/5)
DENSE_HARD_MAX = 1000         # dense numerically out of the question above this d
DENSE_HEUR_MAX = 200          # spike-count rules only propose dense up to this d ...
DENSE_MIN_ROWS_MULT = 10      # ... and only with n >= this * d (the AIC rule needs no such gate:
                              #     its d(d+1)/2 penalty already suppresses dense unless n >> d)
BBP_MARGIN = 0.05             # keep eigenvalues > (1 + BBP_MARGIN) * MP edge
AIC_PENALTY = 2.0             # per-parameter penalty (2 = AIC; log n = BIC)


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


def _cond(eig: np.ndarray) -> float:
    pos = eig[eig > 1e-12]
    return float(eig[0] / pos.min()) if pos.size else 1.0


# --- rule backends ----------------------------------------------------------- #

def _mp_spikes(eig: np.ndarray, n: int, d: int) -> int:
    """Rule A -- Marchenko-Pastur / BBP: count eigenvalues above the bulk right edge.

    Bulk scale ~1 (a correlation matrix); edge ``(1 + sqrt(d/n))^2``. A refined bulk-scale fit
    (Gavish-Donoho median) is possible but this suffices for the near-unit correlation bulk."""
    lam_plus = (1.0 + math.sqrt(d / n)) ** 2 * (1.0 + BBP_MARGIN)
    return int(np.sum(eig > lam_plus))


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
    against identity (which would overfit the bulk). Returns ``(kind, J)``."""
    ben = _kl_benefit(eig)
    n_pos = int(np.sum(eig > 1.0))                        # only lambda > 1 directions are low-rank-able
    best_J, best_score, cum = 0, 0.0, 0.0                 # score = AIC(J) - AIC(0); diagonal is 0
    for J in range(1, min(n_pos, d - 1) + 1):
        cum += ben[J - 1]                                 # eig descending, so top-J
        s = penalty * (J + (J * d - J * (J + 1) // 2)) - 2.0 * n * cum
        if s < best_score:
            best_score, best_J = s, J
    if best_J == 0:
        return ("diagonal", 0)
    if best_J <= jmax:
        return ("lowrank", best_J)
    return ("dense", 0) if dense_ok else ("lowrank", jmax)


def _parallel_spikes(scores: np.ndarray, n: int, d: int, *, n_perm: int = 20,
                     alpha: float = 0.05, seed: int = 0) -> int:
    """Rule C -- Horn's parallel analysis: keep eigenvalues above the per-rank ``(1-alpha)`` quantile
    of a column-permutation null (permuting each column independently kills cross-correlation)."""
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

def select_mass_mode(scores: np.ndarray, *, rule: str = "aic", penalty: float = AIC_PENALTY,
                     n_perm: int = 20, seed: int = 0):
    """Choose a block's mass mode from its ``(n, d)`` scores. ``rule`` in {"aic", "mp", "parallel"}.
    Returns ``(kind, rank)``: ``("diagonal", 0)`` / ``("lowrank", J)`` / ``("dense", 0)``."""
    D, eig, n, d = whitened_spectrum(scores)
    jmax = min(max(1, int(math.floor(J_MAX_FRAC * d))), n - 1, d - 1)
    cond = _cond(eig)
    dense_ok = (n > d) and (d <= DENSE_HARD_MAX)

    def decided(kind, J, why):
        """Report the decision and what drove it, then hand it back."""
        log.debug("mass mode for a (%d, %d) score block: cond(R) %.3g, jmax %d, rule %r -> %s%s "
                  "(%s)", n, d, cond, jmax, rule, kind, f" J={J}" if kind == "lowrank" else "",
                  why)
        return (kind, J)

    # small d: dense is cheap and best -- take it unless the block is essentially isotropic.
    if d <= SMALL_DENSE_DIM:
        if dense_ok and cond >= 2.0:
            return decided("dense", 0, f"d <= {SMALL_DENSE_DIM} and not isotropic")
        return decided("diagonal", 0,
                       f"d <= {SMALL_DENSE_DIM} but {'isotropic' if dense_ok else 'n <= d'}")
    # spectrum within one order of magnitude: a diagonal mass is good enough.
    if cond < COND_DIAGONAL:
        return decided("diagonal", 0, f"cond(R) < {COND_DIAGONAL} (under one order of magnitude)")

    if rule == "aic":
        kind, J = _aic_mode(eig, n, d, jmax, dense_ok, penalty)
        if kind != "lowrank" or J > 0:
            return decided(kind, J, f"AIC-penalised KL over the low-rank order (penalty {penalty})")
        return decided("diagonal", 0, "AIC found no direction worth a rank-one term")

    # spike-count rules (mp / parallel): count spikes, then map to a mode. Dense only heuristically
    # (many spikes -> not low-rank), and only where estimating it is reliable.
    n_spikes = (_mp_spikes(eig, n, d) if rule == "mp"
                else _parallel_spikes(scores, n, d, n_perm=n_perm, seed=seed))
    dense_heur = dense_ok and (d <= DENSE_HEUR_MAX) and (n >= DENSE_MIN_ROWS_MULT * d)
    if n_spikes == 0:
        return decided("diagonal", 0, "no eigenvalue above the null bulk")
    if n_spikes > jmax:
        if dense_heur:
            return decided("dense", 0, f"{n_spikes} spikes > jmax, and dense is estimable here")
        return decided("lowrank", jmax, f"{n_spikes} spikes > jmax, dense not estimable")
    return decided("lowrank", n_spikes, f"{n_spikes} spike(s) above the null bulk")
