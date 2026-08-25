"""Mass-matrix (metric) adaptation for HMC.

A mixin (``docs/design/02_sampler_classes.md``) that adapts the kinetic Hamiltonian's
mass matrix to the target's covariance. In HMC the well-conditioned choice is
``M^{-1} = Cov(target)`` (so the momentum's marginal covariance ``M`` is the target
precision); the mixin tracks the target covariance from the warmup chain and writes the
result into ``state.ham_params[kinetic.id]``.

The covariance is a *stochastic-approximation* running estimate with the shared
Robbins--Monro gain ``(n + n0)^{-kappa}`` (``kappa = 0.75``, ``n0 = 5``; see
:mod:`mimcs.adaptation._stochastic`), ``Cov <- (1 - gain) Cov + gain * delta delta^T`` with
``delta = x - mean``, weighting recent (closer-to-stationary) draws more than an
equal-weight average.

The mixin is **mode-aware** so it never materialises the full ``d x d`` covariance: it
maintains exactly the running statistic the kinetic stores, queried via
``kinetic.mass_mode``:

* ``"diagonal"`` --- the per-coordinate variance vector ``M^{-1}`` (``O(d)`` memory):
  ``var <- (1 - gain) var + gain * delta^2``;
* ``"dense"`` --- the lower Cholesky factor ``L`` of ``M^{-1}``, kept current by a rank-one
  Cholesky update each step (``O(d^2)``) instead of refactorising:
  ``L <- sqrt(1 - gain) * cholupdate(L, sqrt(gain / (1 - gain)) * delta)``.

The running mean and statistic live on the Python object; only the resulting mass
parameters cross into the JAX state, and only once enough samples have accumulated.

Polyak--Ruppert averaging of the frozen mass is available here (``mass_polyak=True``) but is
**off by default** for this scheme, unlike the SGD-based mass adaptations
(:class:`~mimcs.adaptation.ScoreMassAdaptation` et al.) where it is on. The reason: those SGD
iterates oscillate around their target so a suffix average cleanly denoises them, whereas this
Robbins--Monro covariance estimate is a slow *oscillating transient* at typical warmup lengths
(it can swing well past and below the target before settling), so its raw endpoint is already a
good low-variance estimate and a suffix average tends to bias it. When enabled, the mechanism is
the same as elsewhere (raw statistic drives warmup; the suffix average is frozen in for
sampling; :mod:`mimcs.adaptation._polyak`).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .._logging import get_logger
from ..samplers.base import Phase
from ._stochastic import rm_gain, DEFAULT_KAPPA, DEFAULT_N0
from ._cholesky import chol_update
from ._polyak import PolyakLog

log = get_logger(__name__)


@jax.jit
def _dense_update(chol, delta, gain):
    """Rank-one update of ``L`` for ``Cov <- (1 - gain) Cov + gain * delta delta^T``."""
    w = jnp.sqrt(gain / (1.0 - gain)) * delta
    return jnp.sqrt(1.0 - gain) * chol_update(chol, w)


class MassMatrixAdaptation:
    """Mixin: adapt ``state.ham_params[kinetic.id]`` to the target covariance."""

    def _init_hooks(self, **kwargs):
        self._mass_min_samples = int(kwargs.get("mass_min_samples", 50))
        self._mass_n0 = float(kwargs.get("mass_adapt_n0", DEFAULT_N0))
        self._mass_kappa = float(kwargs.get("mass_adapt_kappa", DEFAULT_KAPPA))
        self._mm_polyak = bool(kwargs.get("mass_polyak", False))   # opt-in for this scheme
        self._mm_count = 0
        self._mm_mean = None    # {kinetic.id: running mean vector over its block}
        self._mm_stat = None    # {kinetic.id: variance vector or Cholesky factor}
        self._mm_polyak_avg = {}  # {kinetic.id: PolyakLog} running average of the written mass
        super()._init_hooks(**kwargs)

    def _postprocess_hooks(self, state):
        state = super()._postprocess_hooks(state)
        if self._phase is not Phase.WARMUP:
            return state

        # Adapt every quadratic (diagonal/dense) kinetic block over its own coordinate slice.
        adapted = [k for k in self.kinetics if k.mass_mode in ("diagonal", "dense")]
        if not adapted:
            return state

        x = state.coordinate
        if self._mm_mean is None:
            dim = x.shape[0]
            log.debug("covariance mass adaptation started on block(s) %s; the mass is written "
                      "from iteration %d on", [f"{k.id}[{k.mass_mode}]" for k in adapted],
                      self._mass_min_samples + 1)
            self._mm_mean = {k.id: jnp.zeros((k._size(dim),)) for k in adapted}
            # Each statistic starts at the kinetic's identity mass (var = 1 / L = I), which
            # the decreasing gain washes out -- and keeps a dense factor non-singular.
            self._mm_stat = {k.id: k.initial_mass_params(dim) for k in adapted}

        self._mm_count += 1
        gain = rm_gain(self._mm_count, self._mass_n0, self._mass_kappa)
        for k in adapted:
            delta = k._gather(x) - self._mm_mean[k.id]
            self._mm_mean[k.id] = self._mm_mean[k.id] + gain * delta
            if k.mass_mode == "dense":
                self._mm_stat[k.id] = _dense_update(self._mm_stat[k.id], delta, gain)
            else:  # "diagonal"
                stat = self._mm_stat[k.id]
                self._mm_stat[k.id] = stat + gain * (delta ** 2 - stat)

        if self._mm_count > self._mass_min_samples:
            for k in adapted:                       # fold each written iterate into its average
                if self._mm_polyak:
                    self._mm_polyak_avg.setdefault(k.id, PolyakLog(k.mass_mode)).update(
                        self._mm_stat[k.id])
            state = state._replace(ham_params={           # warmup uses the raw statistic
                **state.ham_params, **{k.id: self._mm_stat[k.id] for k in adapted}})
        return state

    def _finalize_hooks(self, state):
        """Freeze the Polyak--Ruppert average of the mass for sampling (see :mod:`._polyak`)."""
        state = super()._finalize_hooks(state)
        if self._mm_polyak and self._mm_polyak_avg:
            avg = {kid: jnp.asarray(p.value()) for kid, p in self._mm_polyak_avg.items()
                   if p.value() is not None}
            state = state._replace(ham_params={**state.ham_params, **avg})
            log.debug("froze the Polyak-averaged mass of block(s) %s after %d update(s)",
                      sorted(avg), self._mm_count)
        return state
