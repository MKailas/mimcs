"""The product-space sampler: what makes an ordinary sampler run over K temperatures at once.

Implements the product-space half of ``docs/design/13_parallel_tempering.md``. The mixin here is
deliberately thin, because most of the work is done by *pointing an existing sampler at product
components* rather than by rewriting it:

* the potentials are :class:`~mimcs.pt.tempering.TemperedProductPotential`, one per model
  component, each a single vmapped evaluation over the temperature axis;
* the kinetics are :class:`~mimcs.pt.kinetics.ProductKinetic`, the model's own block structure
  applied per temperature;
* the model is a :class:`~mimcs.pt.tempering.ProductModel`, which vmaps the coordinate<->sample
  maps.

With those in place :class:`~mimcs.hmc.NUTS` runs over the product space **unmodified**: its
generalized U-turn ``dot(momentum_sum, velocity)`` over the stacked coordinate *is* the
product-space criterion ``sum_k dot(p_sum_k, v_k)``, and its multinomial selection over the total
energy is ordinary NUTS on the product target. What the mixin adds is only the bookkeeping that
the *kept* draws are the cold chain's.
"""

from __future__ import annotations

import numpy as np

from .._logging import get_logger

log = get_logger(__name__)


class ProductSpaceMixin:
    """Present a product-space chain as if it were a chain on the base model.

    ``state.sample`` carries all K temperatures (that is what the kernel produces), so this
    narrows the *retained* draws and the saved scores to the ``beta = 1`` chain. Everything
    downstream --- :meth:`~mimcs.samplers.BaseSampler.summary`, the evaluation harness, the
    factory's evidence reader --- then sees exactly what it would see from an ordinary run.
    """

    #: index of the cold chain in the ladder; ``beta_1 = 1`` by construction.
    cold_index = 0

    @property
    def base_model(self):
        return self.model.base

    @property
    def summary_model(self):
        """The base model: :meth:`get_samples_flat` and :meth:`_current_score` have already
        narrowed the draws and the scores to the cold chain, so the model they are evaluated
        against must be that chain's too --- the product model's ``ambient_dim`` is ``K`` times
        too wide, and its per-parameter machinery (``ambient_score``, ``stein_terms``) is the
        base model's by construction."""
        return self.base_model

    def get_samples_flat(self) -> np.ndarray:
        """The **cold chain's** draws, ``(n_draws, ambient_dim)``."""
        flat = super().get_samples_flat()
        amb = self.base_model.ambient_dim
        lo = self.cold_index * amb
        return flat[:, lo:lo + amb]

    def get_samples_all(self) -> np.ndarray:
        """Every temperature's draws, ``(n_draws, K, ambient_dim)`` --- for diagnosing the ladder."""
        flat = super().get_samples_flat()
        return flat.reshape(len(flat), self.model.n_temperatures, self.base_model.ambient_dim)

    def _current_score(self, state):
        """The cold chain's score, so the Stein diagnostic sees the target's own gradient.

        At ``beta = 1`` the tempered potential's gradient block *is* the model's gradient, so no
        rescaling is needed --- but only because the ladder pins ``beta_1 = 1``.
        """
        score = super()._current_score(state)
        if score is None:
            return None
        n = self.base_model.coord_dim
        lo = self.cold_index * n
        return score[lo:lo + n]
