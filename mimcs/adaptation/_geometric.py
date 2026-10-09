"""Options, the kernel's moves and the spread window for the geometric-mean mass target, shared by
:class:`~mimcs.adaptation.ScoreMassAdaptation`, :class:`~mimcs.adaptation.LowRankAdaptation`,
:class:`~mimcs.adaptation.MetricAdaptation` and :class:`~mimcs.adaptation.ShapedMetricAdaptation`.

``mass_target="geometric"`` replaces the score loss's log-det barrier by a *spread* term, so that
the mass is fitted to the geometric mean ``M = S^{-1} # F`` of the score covariance ``F`` and the
inverse of a spread ``S`` (``score_mass.py`` derives it). ``geometric_spread`` picks ``S``:

* ``"difference"`` (default) --- ``S = E[u u^T] / 2`` with ``u = x_t - x_{t-k}`` the kernel's own
  displacement over the last ``k`` steps --- ``geometric_lag`` for a constant mass (default 4),
  ``metric_geometric_lag`` for a learned metric (default 1; :data:`CONSTANT_LAG`). A stationary reversible chain
  has ``E[u u^T] / 2 = Sigma - Gamma_k`` (the covariance minus the lag-``k`` autocovariance), so
  ``S = Sigma`` for independent draws and the within-mode covariance when the chain jumps between
  modes only rarely: a *conditional* spread. A coordinate with lag-1 correlation ``rho`` is
  stiffened by ``sqrt(1 - rho^k)``.
* ``"marginal"`` --- ``S = Cov(x)``, from ``x`` minus its running mean (Seyboldt's ``nutpie``
  target). It sees the distance between modes; it is kept as the control.

**Endpoints, not just the difference.** :class:`SpreadTracker` keeps the window's *start
coordinates* (one coordinate vector per step, ``k`` of them) rather than the moves, so the
difference is formed from them and both endpoints remain available: a learned metric ``M(q)``
evaluates its spread term half at the newer endpoint and half at the older one. For a constant mass
only ``u`` matters, and with ``k = 1`` it is formed exactly as before (``end - start``).

**Where a learned metric evaluates the spread** (``geometric_metric_point``). ``"new"``
(default): at the newest coordinate, where the score term also sits (on a plain sampler; under PT
the score is at the post-swap state). ``"split"``: half at each end of the window. On a stationary
reversible chain the two have the same expected loss; in practice ``"new"`` was level on the funnel
and 1.1--1.2x better on irt_2pl stage 2 (16 seeds). Either way the fixed point is the conditional
geometric mean only while the metric's dependency coordinates move little within the window; when
they move, the fitted dependence flattens (funnel, split: the slope of ``log M(v)`` goes
-0.97 / -0.86 / -0.73 at k = 1 / 4 / 16) --- hence a learned metric's lag of 1. Constant masses
ignore it.

**Parallel tempering.** A host never runs ``_preprocess_hooks``:
:class:`~mimcs.pt.adaptation.PerTemperatureAdaptation` hands it the rung's own ``start`` (the
coordinate its trajectory began from) and pre-swap ``end`` instead, and ``replaced`` when the
iteration's swap gave the rung a different replica. The window then **restarts**, so ``u`` is always
the current replica's own displacement over ``min(k, steps since it arrived)``: replica exchanges
stay excluded whatever ``k``. (Without the restart a window spans several replicas, whose moves are
independent across a break rather than anticorrelated, so every break adds a full ``2 Sigma``: a
lag-16 window widened even non-switching coordinates past the marginal covariance.)

On a plain sampler the start is the coordinate the kernel was handed, captured in
``_preprocess_hooks`` (which no other mixin overrides), and the end is the one it returned, read
before any other mixin's ``_postprocess_hooks`` can re-express it.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

TARGETS = ("score", "geometric")
SPREADS = ("difference", "marginal")
#: Where a position-dependent (learned) metric evaluates the spread term: at the newest coordinate
#: only (``"new"``, the default) or half at each end of the window (``"split"``).
METRIC_POINTS = ("split", "new")

#: The default window lengths. A constant mass takes ``k = 4``, a compromise: lag 1 stiffens a
#: slowly mixing coordinate by ``sqrt(1 - rho)`` (ss_logit lost 1.15--1.7x against lag 4), while a
#: long lag drifts toward the marginal spread and with it toward the between-mode distance (lag 4
#: kept 2.1x over the marginal spread on the PT spike-and-slab regression). A learned metric takes
#: ``k = 1``: over a longer window its dependency coordinates move and the fitted dependence
#: flattens (funnel slope -0.97 / -0.86 / -0.73 at k = 1 / 4 / 16), for no gain in ESS.
CONSTANT_LAG = 4
METRIC_LAG = 1

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


def geometric_lag(kwargs, learned: bool = False) -> int:
    """The window length from ``algo_kwargs``, validated: ``geometric_lag`` for a constant mass
    (default :data:`CONSTANT_LAG`), ``metric_geometric_lag`` for a learned metric (default
    :data:`METRIC_LAG`)."""
    key, default = (("metric_geometric_lag", METRIC_LAG) if learned
                    else ("geometric_lag", CONSTANT_LAG))
    k = kwargs.get(key, default)
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError(f"algo_kwargs {key}={k!r}; expected a positive integer")
    return int(k)


class Spread(NamedTuple):
    """One step's spread over the whole coordinate: ``u`` (the window's displacement, or the
    position itself for the marginal spread) and the points a position-dependent metric is
    evaluated at for it --- the newer and older endpoints (both the position, for the marginal
    spread)."""

    u: np.ndarray
    q_new: np.ndarray
    q_old: np.ndarray


class SpreadTracker:
    """The geometric target's options plus its per-mixin spread state: the captured pre-kernel
    coordinate and the window of the last ``k`` start coordinates (module docstring)."""

    def __init__(self, kwargs, learned: bool = False):
        self.target, self.spread = geometric_options(kwargs)
        self.k = geometric_lag(kwargs, learned)
        self.point = kwargs.get("geometric_metric_point", "new")
        if self.point not in METRIC_POINTS:
            raise ValueError(f"algo_kwargs geometric_metric_point={self.point!r}; expected one "
                             f"of {METRIC_POINTS}")
        self.c = SPREAD_CONSTANT[self.spread]
        self.pre = None              # the coordinate the kernel was handed (plain samplers)
        self._starts: list = []      # the window's start coordinates, oldest first

    @property
    def active(self) -> bool:
        return self.target == "geometric"

    def capture(self, state):
        """``_preprocess_hooks``: remember the kernel's input (difference spread only)."""
        if self.active and self.spread == "difference":
            self.pre = state.coordinate

    def observe(self, state) -> Spread:
        """``_postprocess_hooks``, before ``super()``: this step's spread. Raises if a difference
        spread has no start to read --- neither a captured pre-kernel coordinate nor a
        ``state.start`` --- rather than silently fitting the score mass instead."""
        if self.spread == "marginal":
            x = np.asarray(state.coordinate, dtype=float)
            return Spread(x, x, x)
        start, end = getattr(state, "start", None), getattr(state, "end", None)
        if start is None or end is None:
            if self.pre is None:
                raise RuntimeError(
                    "geometric mass target: no move to read. The pre-kernel coordinate is "
                    "captured in _preprocess_hooks; a host that never runs it must pass "
                    "state.start and state.end")
            start, end = self.pre, state.coordinate
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        self._starts.append(start)
        if len(self._starts) > self.k:
            self._starts.pop(0)
        old = self._starts[0]
        u = end - old
        if getattr(state, "replaced", False):          # PT: a new replica starts a new window
            self._starts = []
        # ``q_old`` is where a learned metric evaluates the spread's older half; under
        # ``geometric_metric_point="new"`` both halves sit at the newest coordinate.
        return Spread(u, end, end if self.point == "new" else old)
