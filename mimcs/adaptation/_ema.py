"""Exponential moving averages of adapted mass parameters --- one form for every mass adaptation.

Adaptive mass estimates are noisy stochastic-approximation / SGD iterates. Every mass adaptation
(:class:`~mimcs.adaptation.ScoreMassAdaptation`, :class:`~mimcs.adaptation.MassMatrixAdaptation`,
:class:`~mimcs.adaptation.LowRankAdaptation`, :class:`~mimcs.adaptation.MetricAdaptation`,
:class:`~mimcs.adaptation.ShapedMetricAdaptation` and
:class:`~mimcs.adaptation.RelativisticMassAdaptation`) can smooth its estimate the same way, under
the same two ``algo_kwargs`` keys, read identically by all of them:

* ``mass_ema``: keep an EMA of the adapted estimate during warmup and freeze it into the state for
  sampling (at the sampler's ``_finalize_hooks``). Warmup is still driven by the raw iterate.
  Default ``False``, except for the learned metrics (:class:`~mimcs.adaptation.MetricAdaptation`,
  :class:`~mimcs.adaptation.ShapedMetricAdaptation`), where it defaults to ``True``: there the
  last raw SGD iterate is measured to be unsafe to sample with (parallel tempering on Neal's
  funnel: 2 of 6 seeds diverge on every transition without it, none with it).
* ``mass_ema_warmup`` (default ``False``, implies ``mass_ema``): the EMA also **drives warmup** ---
  it is what the chain is simulated with each warmup step, and, for a shaped or low-rank mass, what
  whitens the score the shape tracker is fitted to. The SGD / RM update still advances the raw
  iterate.

With both off no average is computed at all, and warmup and sampling use the raw iterate.
The learned metrics are the one place ``mass_ema`` defaults on (``ema_options``'s
``sample_default``).

The EMA is the Kailas--Vihola--Wallin smoother: ``e_1 = x_1``,
``e_n = e_{n-1} + eta_n (x_n - e_{n-1})``, with ``eta_n`` the mixin's own Robbins--Monro gain
``(n + n0)^{-kappa}`` at that update. It is taken in **log space** for a diagonal mass (so it stays
positive, a running geometric mean), in **log-Cholesky** space for a dense factor (the
strict-lower entries plus the log of the diagonal, so it stays a valid Cholesky factor), and
directly on a learned metric's parameters, which are already log-linear.

These keys replace ``mass_polyak``, ``score_mass_polyak_warmup`` and ``metric_ema_warmup``, which
meant different things to different mixins (a mass-space EMA, a suffix average, a uniform mean
from the first step). Passing one of them raises rather than being silently ignored.
"""

from __future__ import annotations

import numpy as np
import jax

# Removed keys -> their replacement. Unknown ``algo_kwargs`` are otherwise silently ignored, so a
# stale key would quietly stop doing anything.
_REMOVED = {
    "mass_polyak": "mass_ema",
    "score_mass_polyak_warmup": "mass_ema_warmup",
    "metric_ema_warmup": "mass_ema_warmup",
}


def ema_options(kwargs, sample_default: bool = False) -> tuple[bool, bool]:
    """``(sample, warmup)``: keep an EMA and freeze it for sampling; let it drive warmup too.

    ``sample_default`` is the mixin's default for ``mass_ema`` (``True`` only for the learned
    metrics). ``mass_ema_warmup`` implies ``mass_ema``. Raises ``ValueError`` on a removed key."""
    for old, new in _REMOVED.items():
        if old in kwargs:
            raise ValueError(
                f"algo_kwargs key {old!r} was removed; use {new!r}. Every mass adaptation now "
                f"reads the same two keys, 'mass_ema' and 'mass_ema_warmup' "
                f"(docs/reference/algo_kwargs.md, 'Mass averaging').")
    warmup = bool(kwargs.get("mass_ema_warmup", False))
    return bool(kwargs.get("mass_ema", sample_default)) or warmup, warmup


def _to_log(param, mode: str):
    param = np.asarray(param, dtype=float)
    if mode == "dense":                        # log-Cholesky: replace the diagonal by its log
        out = np.tril(param, -1).copy()
        d = np.diag_indices_from(out)
        out[d] = np.log(np.diag(param))
        return out
    return np.log(param)                        # diagonal / per-particle vector


def _from_log(logparam, mode: str):
    if mode == "dense":
        out = np.tril(logparam, -1).copy()
        d = np.diag_indices_from(out)
        out[d] = np.exp(np.diag(logparam))
        return out
    return np.exp(logparam)


class LogEMA:
    """EMA of a positive mass parameter in log (``mode="diagonal"``) or log-Cholesky
    (``mode="dense"``, a lower factor with positive diagonal) space.

    ``update(param, gain)`` folds the output-space parameter in with gain ``gain`` (the first
    update sets the average to it); ``value()`` returns the average in output space, or ``None``
    before the first update."""

    def __init__(self, mode: str):
        self.mode = mode
        self._avg = None            # the average, in log / log-Cholesky space

    def update(self, param, gain: float):
        x = _to_log(param, self.mode)
        self._avg = x if self._avg is None else self._avg + gain * (x - self._avg)

    def value(self):
        return None if self._avg is None else _from_log(self._avg, self.mode)


def tree_ema(avg, params, gain: float):
    """Fold a learned metric's (log-linear) parameter pytree into its EMA with gain ``gain``."""
    return jax.tree_util.tree_map(lambda a, x: a + gain * (x - a), avg, params)
