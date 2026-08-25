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
        coordinate = model.sample_to_coordinate(sample, chart_hyperparams, chart_indices)
        log_prob = model.log_prob_at_coordinate(coordinate, chart_hyperparams, chart_indices)

        step_size = jnp.asarray(self._kwargs.get("step_size", 1.0), float)
        proposal_scale = jnp.ones((model.coord_dim,), float)

        return MHState(
            coordinate=coordinate,
            sample=sample,
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
            proposal_coord, state.chart_hyperparams, state.chart_indices)

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
    """Accept an init position as a flat array or a ``{param_name: value}`` dict."""
    if isinstance(init_position, dict):
        return model.pack_sample({k: jnp.asarray(v, float)
                                  for k, v in init_position.items()})
    return jnp.asarray(init_position, float).reshape((model.ambient_dim,))
