"""Options and the kernel's move for the geometric-mean mass target, shared by
:class:`~mimcs.adaptation.ScoreMassAdaptation` and :class:`~mimcs.adaptation.LowRankAdaptation`.

``mass_target="geometric"`` replaces the score mass's log-det barrier by a *spread* term, so that
the mass is fitted to the geometric mean ``M = S^{-1} # F`` of the score covariance ``F`` and the
inverse of a spread ``S`` (``score_mass.py`` derives it). ``geometric_spread`` picks ``S``:

* ``"difference"`` (default) --- ``S = E[Delta Delta^T] / 2`` with ``Delta`` the kernel's own move,
  ``x_t - x_{t-1}``. A stationary reversible chain has ``E[Delta Delta^T] / 2 = Sigma - Gamma_1``
  (the covariance minus the lag-1 autocovariance), so ``S = Sigma`` for independent draws and the
  within-mode covariance when the chain jumps between modes only rarely: a *conditional* spread.
* ``"marginal"`` --- ``S = Cov(x)``, from ``x`` minus its running mean (Seyboldt's ``nutpie``
  target). It sees the distance between modes; it is kept as the control.

``geometric_lag`` (default 1) lengthens the difference spread to ``k`` steps: ``u`` is the sum of
the last ``k`` lag-1 moves. On one chain that is ``x_t - x_{t-k}``, and
``E[(x_t - x_{t-k})(...)^T] / 2 = Sigma - Gamma_k``, so a coordinate with lag-1 correlation ``rho``
is stiffened by ``sqrt(1 - rho^k)`` rather than ``sqrt(1 - rho)`` --- the lag-1 spread conflates
"slow" with "between modes". Under parallel tempering each rung sums its *own pre-swap* moves, and **restarts the sum
whenever a swap hands it a different replica** (``AdaptState.replaced``): the sum is then always the
current replica's own displacement over ``min(k, steps since it arrived)``, a genuine single-chain
lag difference. Replica exchanges stay excluded whatever ``k``; only a mode switch the kernel itself
makes within ``k`` steps enters ``S``. (Without the restart the sum spans several replicas, whose
moves are independent across a break rather than anticorrelated, so every break adds a full
``2 Sigma``: a lag-16 sum widened even non-switching coordinates past the marginal covariance.)
The ``k`` moves are kept per block (``k`` vectors of the coordinate's size); ``k = 1`` is the plain
lag-1 move, unchanged.

The move is read as the coordinate the kernel was handed (captured in ``_preprocess_hooks``,
which no other mixin overrides) against the one it returned. A parallel-tempering host never
runs ``_preprocess_hooks``: :class:`~mimcs.pt.adaptation.PerTemperatureAdaptation` hands it the
rung's own move (the swap undone) as ``state.move`` instead.
"""

from __future__ import annotations

import numpy as np

TARGETS = ("score", "geometric")
SPREADS = ("difference", "marginal")

#: ``c`` in ``S = E[u u^T] / c``: a difference of two independent draws has twice their covariance.
SPREAD_CONSTANT = {"difference": 2.0, "marginal": 1.0}

#: Trust region: no step may change the mass by more than this many nats in any direction (per
#: coordinate on a diagonal block, per eigen-direction of the dense step). The adaptive clip alone
#: does not bound the step: through a far-from-mode transient nearly every step is clipped, so the
#: threshold climbs (to e^8.8 on kilpisjarvi), and the geometric gradient, unlike the score's
#: (bounded above, and shrinking as the dense factor grows), has no brake of its own --- single
#: steps of +-30 nats oscillated the dense factor into singularity. At stationarity it does not
#: bind.
MAX_LOG_STEP = 1.0


def geometric_options(kwargs) -> tuple[str, str]:
    """``(mass_target, geometric_spread)`` from ``algo_kwargs``, validated."""
    target = kwargs.get("mass_target", "score")
    spread = kwargs.get("geometric_spread", "difference")
    if target not in TARGETS:
        raise ValueError(f"algo_kwargs mass_target={target!r}; expected one of {TARGETS}")
    if spread not in SPREADS:
        raise ValueError(f"algo_kwargs geometric_spread={spread!r}; expected one of {SPREADS}")
    return target, spread


def geometric_lag(kwargs) -> int:
    """``geometric_lag`` from ``algo_kwargs``: a positive integer (default 1)."""
    k = kwargs.get("geometric_lag", 1)
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError(f"algo_kwargs geometric_lag={k!r}; expected a positive integer")
    return int(k)


class LagSum:
    """The sum of the last ``k`` vectors fed to it (fewer until ``k`` have arrived).

    With ``k = 1`` it returns its input unchanged (the same array), so the lag-1 spread is
    bitwise what it was before lags existed."""

    def __init__(self, k: int):
        self.k = int(k)
        self._buf: list = []

    def __call__(self, move):
        if self.k == 1:
            return move
        self._buf.append(np.asarray(move, dtype=float))
        if len(self._buf) > self.k:
            self._buf.pop(0)
        return np.sum(self._buf, axis=0)

    def reset(self):
        """Start a new sum (the replica changed): the next move is the first of a new segment."""
        self._buf = []


def spread_source(state, spread: str, pre_coordinate):
    """The raw spread vector over the whole coordinate: the kernel's move for ``"difference"``,
    the position itself for ``"marginal"`` (centred by the block's running mean later).

    Raises if a difference spread has no move to read --- neither a captured pre-kernel coordinate
    nor a ``state.move`` --- rather than silently fitting the score mass instead."""
    if spread == "marginal":
        return np.asarray(state.coordinate, dtype=float)
    move = getattr(state, "move", None)
    if move is not None:
        return np.asarray(move, dtype=float)
    if pre_coordinate is None:
        raise RuntimeError(
            "geometric mass target: no move to read. The pre-kernel coordinate is captured in "
            "_preprocess_hooks; a host that never runs it must pass state.move")
    return np.asarray(state.coordinate, dtype=float) - np.asarray(pre_coordinate, dtype=float)
