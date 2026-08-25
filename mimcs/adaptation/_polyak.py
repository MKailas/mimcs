"""Polyak--Ruppert averaging of mass parameters, in log / log-Cholesky space.

Adaptive mass-matrix estimates are noisy stochastic-approximation / SGD iterates. Reporting a
*tail average* of the iterate rather than its last value (Polyak--Ruppert averaging, in its
suffix-averaging form) sharply reduces that noise --- most valuably for dense masses, whose
many entries are each individually noisy --- while a *suffix* (rather than from-the-start)
window keeps the early ramp-from-identity transient out of the estimate. The averaging is done
in **log space** for a diagonal mass (the log of the
inverse-mass vector) and in **log-Cholesky** space for a dense mass (the strict-lower entries
of the Cholesky factor of ``M^{-1}`` together with the *log* of its diagonal), which keeps a
diagonal / a Cholesky diagonal positive under the linear average.

The raw iterate keeps driving the warmup dynamics (so step-size adaptation and the sampling
path are unperturbed); the average is folded in each warmup step and only *frozen into the
state when sampling begins* (the sampler's ``_finalize_hooks``), so it is what the frozen
sampler uses --- Polyak--Ruppert averaging proper: the iterate runs the algorithm, the average
is the reported estimate.
"""

from __future__ import annotations

import numpy as np


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


class PolyakLog:
    """Suffix (tail) average of a mass parameter in log / log-Cholesky space.

    ``update(param)`` folds the current output-space parameter --- an ``M^{-1}`` variance
    vector, a per-particle mass vector (``mode="diagonal"``), or the lower Cholesky ``L`` of
    ``M^{-1}`` (``mode="dense"``) --- into the average; ``value()`` returns it in the same space.

    A *suffix* average (rather than a uniform one from the first step) so the early warmup
    transient --- where the estimate is still ramping away from the identity --- does not bias
    the result. The averaging window is the most recent ~50--75% of the iterates, tracked in
    O(1) via a doubling checkpoint: a prefix sum ``S_n`` plus a snapshot ``(S_m, m)`` taken at
    the power of two below ``n`` gives the tail mean ``(S_n - S_m) / (n - m)``. Length-agnostic:
    the window scales with however many warmup steps run.
    """

    def __init__(self, mode: str):
        self.mode = mode
        self._sum = None            # prefix sum of the log-space iterates
        self._n = 0
        self._c_sum, self._c_n = None, 0    # current checkpoint (last power of two)
        self._p_sum, self._p_n = None, 0    # previous checkpoint (=> a >=50% tail window)

    def update(self, param):
        """Fold ``param`` into the tail average; returns the current tail average."""
        x = _to_log(param, self.mode)
        self._sum = x if self._sum is None else self._sum + x
        self._n += 1
        if self._n >= 2 * max(self._c_n, 1):        # crossed the next power of two
            self._p_sum, self._p_n = self._c_sum, self._c_n
            self._c_sum, self._c_n = self._sum, self._n
        return self.value()

    def value(self):
        """The current tail average (output space), or ``None`` if nothing folded in yet."""
        if self._sum is None:
            return None
        if self._p_sum is None or self._n - self._p_n <= 0:
            avg = self._sum / self._n                        # full average early on
        else:
            avg = (self._sum - self._p_sum) / (self._n - self._p_n)
        return _from_log(avg, self.mode)
