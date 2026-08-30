"""Random-walk Metropolis--Hastings.

A base sampler class (``docs/design/02_sampler_classes.md``) implementing a Gaussian
random-walk proposal in coordinate space. The proposal covariance is factored into a
scalar ``step_size`` (a global scale, adapted to hit a target acceptance rate) and a
per-coordinate ``proposal_scale`` (a diagonal shape, adapted from the running sample
covariance). Both live in the state so the pure kernel can read them; the adaptation
that updates them lives in mixins (see ``mimcs.adaptation``).

Proposal:  ``q' = q + step_size * proposal_scale * z``,  ``z ~ N(0, I)``.
Because the proposal is symmetric and we work in coordinate space against the
coordinate-space target ``log_prob_at_coordinate`` (which includes the chart Jacobian),
acceptance is simply ``min(1, exp(log pi(q') - log pi(q)))``.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from ..rng import DrawComponent, zero_draw
from .base import BaseSampler


class MHState(NamedTuple):
    """State for random-walk Metropolis--Hastings."""

    coordinate: Array          # position in coordinate (chart) space, flat (n,)
    sample: Array              # position in ambient space, flat
    discrete: Array            # the model's discrete parameters, flat int32 (shape (0,) if none)
    log_prob: Array            # scalar: coordinate-space target log-density at `coordinate`
    rng_draw: Any              # typed RngDraw NamedTuple (proposal_noise, accept_threshold)
    chart_hyperparams: tuple   # per-parameter chart hyperparameters
    chart_indices: tuple       # per-parameter active chart index
    step_size: Array           # scalar global proposal scale
    proposal_scale: Array      # (n,) per-coordinate proposal standard deviations
    diagnostics: dict = {}     # per-transition diagnostics (accept_prob, accepted); filled by kernel


class RandomWalkMH(BaseSampler):
    """Gaussian random-walk Metropolis--Hastings on unconstrained coordinates."""

    state_class = MHState

    def make_draw_components(self, model, **kwargs):
        d = model.coord_dim
        return [
            DrawComponent("proposal_noise", (d,), generator=jax.random.normal),
            DrawComponent("accept_threshold", (), generator=jax.random.uniform),
        ]

    def init_diagnostics(self) -> dict:
        z = jnp.zeros(())
        return {**super().init_diagnostics(), "accept_prob": z, "accepted": jnp.asarray(False)}

    def make_initial_state(self, init_position) -> MHState:
        model = self.model
        chart_hyperparams = model.init_chart_hyperparams()
        chart_indices = model.init_chart_indices()

        sample = _as_sample_flat(model, init_position)
        discrete = _as_discrete_flat(model, init_position)
        coordinate = model.sample_to_coordinate(sample, chart_hyperparams, chart_indices)
        log_prob = model.log_prob_at_coordinate(coordinate, chart_hyperparams, chart_indices,
                                                discrete)

        step_size = jnp.asarray(self._kwargs.get("step_size", 1.0), float)
        proposal_scale = jnp.ones((model.coord_dim,), float)

        return MHState(
            coordinate=coordinate,
            sample=sample,
            discrete=discrete,
            log_prob=log_prob,
            rng_draw=zero_draw(self._rng_draw_class, self._draw_components),
            chart_hyperparams=chart_hyperparams,
            chart_indices=chart_indices,
            step_size=step_size,
            proposal_scale=proposal_scale,
            diagnostics=self.init_diagnostics(),
        )

    def kernel(self, state: MHState) -> MHState:
        model = self.model
        z = state.rng_draw.proposal_noise
        proposal_coord = state.coordinate + state.step_size * state.proposal_scale * z
        proposal_sample = model.coordinate_to_sample(
            proposal_coord, state.chart_hyperparams, state.chart_indices)
        proposal_logp = model.log_prob_at_coordinate(
            proposal_coord, state.chart_hyperparams, state.chart_indices, state.discrete)

        log_alpha = proposal_logp - state.log_prob
        accept_prob = jnp.minimum(1.0, jnp.exp(log_alpha))
        accepted = state.rng_draw.accept_threshold < accept_prob

        new_coordinate = jnp.where(accepted, proposal_coord, state.coordinate)
        new_sample = jnp.where(accepted, proposal_sample, state.sample)
        new_log_prob = jnp.where(accepted, proposal_logp, state.log_prob)

        return state._replace(
            coordinate=new_coordinate,
            sample=new_sample,
            log_prob=new_log_prob,
            diagnostics={"accept_prob": accept_prob, "accepted": accepted},
        )


def _as_sample_flat(model, init_position) -> Array:
    """Accept an init position as a flat array or a ``{param_name: value}`` dict.

    A dict may name discrete parameters too; ``pack_sample`` iterates the *continuous* parameters
    and ignores the rest, and :func:`_as_discrete_flat` takes the other half.
    """
    if isinstance(init_position, dict):
        # `getattr`, not the attribute: `model` may be a wrapper (a PT `ProductModel`) that has no
        # discrete parameters and none of the bookkeeping for them.
        discrete = {p.name for p in getattr(model, "discrete_parameters", ())}
        return model.pack_sample({k: jnp.asarray(v, float)
                                  for k, v in init_position.items() if k not in discrete})
    return jnp.asarray(init_position, float).reshape((model.ambient_dim,))


def _as_discrete_flat(model, init_position) -> Array:
    """The discrete half of an init position: the named values, or the model's default.

    A flat-array ``init_position`` sizes the *continuous* block only --- it is what
    ``model.default_sample()`` returns and what every existing caller passes --- so the discrete
    block falls back to :meth:`~mimcs.model.Model.default_discrete`. To set labels explicitly,
    pass a dict.
    """
    if not model.discrete_parameters:
        return jnp.zeros((0,), jnp.int32)
    if isinstance(init_position, dict):
        given = {p.name: init_position[p.name] for p in model.discrete_parameters
                 if p.name in init_position}
        if given:
            missing = [p.name for p in model.discrete_parameters if p.name not in given]
            if missing:
                raise ValueError(
                    f"init_position names some discrete parameters but not {missing}; give all "
                    f"of them or none (the rest default to their lower bound)")
            for p in model.discrete_parameters:
                p.validate(given[p.name])
            return model.pack_discrete(given)
    return model.default_discrete()
