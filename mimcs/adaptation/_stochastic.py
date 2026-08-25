"""Shared schedule for stochastic-approximation adaptation.

Every decreasing-gain adaptation that fits to *empirical moments* --- the mass matrix and
proposal covariance (running covariance) and the learned/score-based metrics (SGD on a KL
objective) --- uses the same Robbins--Monro gain ``(n + n0)^{-kappa}`` with ``kappa = 0.75``
and ``n0 = 5``. The offset ``n0`` damps the noisiest first updates; ``kappa in (0.5, 1]``
gives a diminishing, convergent adaptation that weights recent (closer-to-stationary)
draws more than the equal-weight ``1/n`` average. Centralised here so these stay consistent.

(The step-size adaptation has its own ``kappa`` default --- it targets a scalar acceptance
rate rather than a covariance --- and is configured separately in ``step_size.py``.)
"""

from __future__ import annotations

DEFAULT_KAPPA = 0.75
DEFAULT_N0 = 5.0


def rm_gain(count: int, n0: float = DEFAULT_N0, kappa: float = DEFAULT_KAPPA) -> float:
    """Robbins--Monro gain ``(count + n0)^{-kappa}`` (``count`` the 1-based step index)."""
    return (count + n0) ** (-kappa)
