"""Per-temperature adaptation: the K adaptation-only samplers.

Implements the adaptation section of ``docs/design/13_parallel_tempering.md``. The existing
adaptation mixins cannot run on the product state directly --- ``MassMatrixAdaptation`` gathers
``state.coordinate`` through each kinetic's slices and writes a mass of that shape, but a product
kinetic's parameters carry a leading temperature axis, so the shapes disagree. Rather than
reimplement adaptation over the product layout, PT keeps **one host per temperature**: each holds
that temperature's own coordinate and its own mass, runs the ordinary mixins on it, and PT stacks
the results back into the product state.

**Only the mass is adapted per temperature; the step size stays global.** That is a consequence of
joint selection (doc 13): the product chain accepts or rejects as one, so there is a single
acceptance signal and no per-temperature one to drive K step sizes. It is also the right split.
A hot chain's target is wider, and *width is exactly what a mass matrix absorbs* --- temperature
``k``'s adapted mass captures its own scale, after which one shared step size suits them all. The
machinery for a per-temperature step-size vector exists (doc 13) and is simply not needed here.

The host is deliberately not a full sampler: the mixins are only ever asked for their
``_postprocess_hooks`` / ``_finalize_hooks``, never for a kernel, RNG or draws.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from .._logging import get_logger
from ..samplers.base import Phase

log = get_logger(__name__)


class AdaptState(NamedTuple):
    """The slice of state one temperature's adaptation sees.

    ``potential_grads`` is what lets the *score*-based adaptations run per temperature ---
    ``ScoreMassAdaptation`` (the factory's default), ``LowRankAdaptation`` and the metric
    adaptations all read it. Row ``k`` of a tempered potential's gradient is
    ``beta_k * grad(V_k)``, i.e. the score of that rung's own target ``pi^beta_k``, which is
    exactly what its mass should be fitted to.
    """

    coordinate: Array
    ham_params: dict
    step_size: Array
    potential_grads: dict = {}
    diagnostics: dict = {}


class _AdaptationHost:
    """Terminal of the mixin chain: just enough surface for the adaptation mixins to run."""

    def __init__(self, model, kinetics, owner, **kwargs):
        self.model = model
        self.kinetics = kinetics
        self._owner = owner            # the PT sampler, for the current phase
        self._kwargs = dict(kwargs)
        self._iteration = 0
        self._init_hooks(**kwargs)

    @property
    def _phase(self):
        """Adaptation runs during warmup only, and the phase is the PT sampler's."""
        return self._owner._phase

    def _init_hooks(self, **kwargs):
        return None

    def _postprocess_hooks(self, state):
        return state

    def _finalize_hooks(self, state):
        return state


class PerTemperatureAdaptation:
    """Mixin: run ``adapt_mixins`` once per temperature and stack the results back.

    Composed *below* the replica-exchange mixin, so it sees the state after a step and a swap.
    """

    def _init_hooks(self, **kwargs):
        super()._init_hooks(**kwargs)
        mixins = tuple(kwargs.get("adapt_mixins", ()) or ())
        self._adapt_hosts = []
        if not mixins:
            log.debug("parallel tempering: no per-temperature adaptation requested")
            return
        base_model = self.model.base
        inner = [k.inner for k in self.kinetics]
        host_cls = type("PTAdaptationHost", (*mixins, _AdaptationHost), {})
        for k in range(self.n_temperatures):
            self._adapt_hosts.append(host_cls(base_model, inner, self, **kwargs))
        log.info("parallel tempering: per-temperature adaptation %s on %d temperature(s)",
                 [m.__name__ for m in mixins], self.n_temperatures)

    # --- state <-> per-temperature slices ---

    def _temperature_params(self, ham_params: dict, k: int) -> dict:
        """Temperature ``k``'s row of every block's parameters."""
        from .ladder import BETAS_KEY
        return {kid: jax.tree.map(lambda x: x[k], v) for kid, v in ham_params.items()
                if kid != BETAS_KEY}

    def _temperature_grads(self, potential_grads: dict, k: int, n: int) -> dict:
        """Temperature ``k``'s block of every potential's gradient, ``(K*n,) -> (n,)``."""
        return {pid: g.reshape(self.n_temperatures, n)[k] for pid, g in potential_grads.items()}

    def _stack_params(self, per_temperature: list) -> dict:
        return {kid: jax.tree.map(lambda *xs: jnp.stack(xs), *[p[kid] for p in per_temperature])
                for kid in per_temperature[0]}

    def _postprocess_hooks(self, state):
        state = super()._postprocess_hooks(state)
        if not self._adapt_hosts or self._phase is not Phase.WARMUP:
            return state

        n = self.model.base.coord_dim
        coords = state.coordinate.reshape(self.n_temperatures, n)
        out = []
        for k, host in enumerate(self._adapt_hosts):
            sub = AdaptState(coordinate=coords[k],
                             ham_params=self._temperature_params(state.ham_params, k),
                             step_size=state.step_size,
                             potential_grads=self._temperature_grads(state.potential_grads, k, n),
                             diagnostics=state.diagnostics)
            out.append(host._postprocess_hooks(sub).ham_params)
        return state._replace(
            ham_params={**state.ham_params, **self._stack_params(out)})

    def _finalize_hooks(self, state):
        """Let each temperature's mixins freeze whatever they average (e.g. a Polyak mass)."""
        state = super()._finalize_hooks(state)
        if not self._adapt_hosts:
            return state
        n = self.model.base.coord_dim
        out = []
        for k, host in enumerate(self._adapt_hosts):
            sub = AdaptState(coordinate=jnp.zeros((n,)),
                             ham_params=self._temperature_params(state.ham_params, k),
                             step_size=state.step_size,
                             potential_grads=self._temperature_grads(state.potential_grads, k, n),
                             diagnostics=state.diagnostics)
            out.append(host._finalize_hooks(sub).ham_params)
        return state._replace(
            ham_params={**state.ham_params, **self._stack_params(out)})
