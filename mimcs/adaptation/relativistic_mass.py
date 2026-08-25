"""Score-covariance adaptation of the relativistic per-particle mass.

**[experimental]**, with :mod:`mimcs.hmc.relativistic`: this adapts the rest mass ``m_i`` but not
the light speed ``c``, and the argument that a fixed ``c`` suffices assumes centered coordinates
--- which is no longer a default. See that module's docstring.

A mixin (``docs/design/02_sampler_classes.md``) that adapts each particle's rest mass
``m_i`` of a :class:`~mimcs.hmc.RelativisticKinetic` by SGD on the KL objective

    L(m_i) = 1/2 ( nu_i log m_i + |g_i|^2 / m_i ),

the per-particle generalization of :class:`ScoreMassAdaptation`: ``g_i`` is the score
(potential gradient) restricted to particle ``i`` and ``|g_i|^2`` its squared norm summed
over the particle's ``nu_i = d_i`` inner components. The minimiser is
``m_i = E[|g_i|^2] / d_i``, i.e. the adaptation drives ``|g_i|^2 / m_i`` to a ``chi^2`` with
``d_i`` degrees of freedom (mean ``d_i``). For a 1-D particle this is exactly
``ScoreMassAdaptation`` (``m = E[g^2]``); pair it with a centering reparametrization (which
standardizes the coordinates) and a fixed light speed ``c``. Nothing enforces that pairing, and
centering is opt-in and off by default --- which is why this is experimental.

(The "d-1" that arises in relativistic dynamics is the *mode* of the radial momentum density,
``|p|^2 = (d-1) T`` at the mode --- a different quantity. The score-covariance / Gaussian-
limit degrees of freedom is ``d``, which also keeps the 1-D case non-degenerate.)

SGD uses the shared schedule ``(n + n0)^{-kappa}`` (kappa=0.75, n0=5) and the same adaptive
gradient clipping as ``ScoreMassAdaptation`` (dimension-aware init, converging gain). The
log-masses live on the Python object; only ``m`` (in ``ham_params[kinetic.id]``) crosses
into the JAX state. Adaptation runs during warmup only. By default (``mass_polyak=True``) the
mass written is the Polyak--Ruppert running average of the SGD iterate, in log space
(:mod:`mimcs.adaptation._polyak`), while the raw iterate is unaffected.
"""

from __future__ import annotations

import math

import numpy as np
import jax.numpy as jnp

from .._logging import get_logger
from ..samplers.base import Phase
from ._stochastic import rm_gain, DEFAULT_KAPPA, DEFAULT_N0
from ._polyak import PolyakLog

log = get_logger(__name__)


class RelativisticMassAdaptation:
    """Mixin: SGD-adapt the relativistic per-particle mass to the score covariance.

    **[experimental]** --- see the module docstring."""

    def _init_hooks(self, **kwargs):
        self._rm_n0 = float(kwargs.get("rel_mass_n0", DEFAULT_N0))
        self._rm_kappa = float(kwargs.get("rel_mass_kappa", DEFAULT_KAPPA))
        self._rm_clip_frac = float(kwargs.get("rel_mass_clip_frac", 0.1))
        self._rm_polyak = bool(kwargs.get("mass_polyak", True))
        self._rm_polyak_avg = PolyakLog("diagonal")   # per-particle mass is a vector
        self._rm_count = 0
        self._rm_log_mass = None        # theta = log m (per particle)
        self._rm_log_clip = None
        super()._init_hooks(**kwargs)

    def _rel_kinetic(self):
        """The relativistic kinetic block this mixin adapts."""
        for k in self.kinetics:
            if hasattr(k, "particle_grad_sq"):
                return k
        raise ValueError("RelativisticMassAdaptation requires a relativistic kinetic")

    def _postprocess_hooks(self, state):
        state = super()._postprocess_hooks(state)
        if self._phase is not Phase.WARMUP:
            return state
        kinetic = self._rel_kinetic()
        nu = kinetic.particle_dim
        score = kinetic._gather(sum(state.potential_grads.values()))
        s = np.asarray(kinetic.particle_grad_sq(score), float)      # |g_i|^2 per particle

        if self._rm_log_mass is None:
            self._rm_log_mass = np.log(np.asarray(state.ham_params[kinetic.id], float))
            self._rm_log_clip = math.log(s.shape[0])            # clip threshold init: log d

        self._rm_count += 1
        m = np.exp(self._rm_log_mass)
        grad = 0.5 * (nu - s / m)                                   # dL/dtheta per particle
        gnorm = float(np.sqrt(np.sum(grad ** 2)))

        thr = math.exp(self._rm_log_clip)
        scale = min(1.0, thr / (gnorm + 1e-12))
        exceeded = 1.0 if gnorm > thr else 0.0
        self._rm_log_clip += (rm_gain(self._rm_count, self._rm_n0, self._rm_kappa)
                              * (exceeded - self._rm_clip_frac))

        lr = rm_gain(self._rm_count, self._rm_n0, self._rm_kappa)
        self._rm_log_mass -= lr * scale * grad

        m = np.exp(self._rm_log_mass)
        if self._rm_polyak:                 # fold the raw iterate into the running average
            self._rm_polyak_avg.update(m)
        return state._replace(ham_params={**state.ham_params, kinetic.id: jnp.asarray(m)})

    def _finalize_hooks(self, state):
        """Freeze the Polyak--Ruppert average of the mass for sampling (see :mod:`._polyak`)."""
        state = super()._finalize_hooks(state)
        if self._rm_polyak and self._rm_polyak_avg.value() is not None:
            kinetic = self._rel_kinetic()
            state = state._replace(ham_params={
                **state.ham_params, kinetic.id: jnp.asarray(self._rm_polyak_avg.value())})
            log.debug("froze the Polyak-averaged relativistic mass of block %r after %d update(s)",
                      kinetic.id, self._rm_count)
        return state
