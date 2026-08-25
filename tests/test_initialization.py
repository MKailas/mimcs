"""Tests for the sampler initialization pass (``sampler.initialize()`` and its mixins).

``initialize()`` (explicit, before warmup) runs the ``_initialize_hooks`` chain:
:class:`UniformInit` draws the initial *coordinate* from ``U(-2, 2)`` (retrying to finite
density), and :class:`StepSizeLineSearch` backtracks the step size from ``0.5`` to a
single-leapfrog (MALA) acceptance of ``0.9``. Both are inert unless ``initialize()`` is called.

Seeds are fixed, so pass/fail is deterministic.
"""

import numpy as np
import jax.numpy as jnp
import pytest

from mimcs.model import Model, EuclideanParameter, PositiveParameter
from mimcs.factory import make_sampler
from mimcs.samplers import make_sampler_class
from mimcs.hmc import HMC
from mimcs.hmc.integrators import init_integrator_state
from mimcs.adaptation import (
    UniformInit, StepSizeLineSearch, RobbinsMonroStepSize, ScoreMassAdaptation)
from mimcs.testing import correlated_gaussian, evaluate


def _mala_accept_at(s, eps):
    """The single-leapfrog acceptance at ``eps`` from the sampler's initialized state, using the
    same momentum the line search used (so it reproduces what the search saw)."""
    st = s.state
    ctx = s.context(st)
    p = s.sample_momentum(s._init_momentum_draw(), st.coordinate, ctx)
    i0 = init_integrator_state(s.potentials, st.coordinate, p, ctx)
    log_alpha = s.total_energy(i0, ctx) - s.total_energy(
        s.integrator.integrate(i0, jnp.asarray(eps, jnp.float32), 1, ctx), ctx)
    return float(jnp.minimum(1.0, jnp.exp(log_alpha)))


def test_uniform_init_draws_finite_density_coordinate_in_box():
    """The initial coordinate is in [-2, 2]^d, has finite density, and the state is
    self-consistent (sample maps back from the coordinate; log_prob matches)."""
    prob = correlated_gaussian(mean=[5.0, -3.0, 10.0], cov=np.diag([1.0, 4.0, 0.25]))
    s = make_sampler(prob.model, seed=0)
    assert np.allclose(np.asarray(s.state.coordinate), 0.0)      # zeros before initialize()
    s.initialize()

    u = np.asarray(s.state.coordinate)
    assert np.all(np.abs(u) <= 2.0)
    assert np.isfinite(float(s.state.log_prob))
    h, c = s.state.chart_hyperparams, s.state.chart_indices
    x = np.asarray(prob.model.coordinate_to_sample(s.state.coordinate, h, c))
    assert np.allclose(x, np.asarray(s.state.sample), atol=1e-5)
    assert np.isclose(float(s.state.log_prob),
                      float(prob.model.log_prob_at_coordinate(s.state.coordinate, h, c)), atol=1e-3)


def test_uniform_init_bounded_parameter_is_unconstrained():
    """For a bounded (positive) parameter the U(-2,2) draw is in the *unconstrained* coordinate
    (log link); the mapped sample respects the bound."""
    model = Model([PositiveParameter("sig")],
                  {"lp": lambda p: -0.5 * jnp.sum(jnp.log(p["sig"]) ** 2) - jnp.sum(jnp.log(p["sig"]))})
    s = make_sampler(model, seed=1).initialize()
    assert np.all(np.abs(np.asarray(s.state.coordinate)) <= 2.0)   # unconstrained coordinate
    assert np.all(np.asarray(s.state.sample) > 0.0)                # positive sample


def test_uniform_init_retries_until_finite_density():
    """A target that is -inf outside a small region still yields a finite-density start."""
    model = Model([EuclideanParameter("x", (1,))],
                  {"lp": lambda p: jnp.where(jnp.abs(p["x"][0]) < 0.5, -0.5 * p["x"][0] ** 2,
                                             -jnp.inf)})
    s = make_sampler(model, seed=0).initialize()
    assert np.abs(float(s.state.coordinate[0])) < 0.5              # landed in the finite region
    assert np.isfinite(float(s.state.log_prob))


def test_step_size_line_search_hits_mala_target():
    """A sharply-scaled target shrinks the step below 0.5; an easy Gaussian keeps ~0.5. In both
    the single-leapfrog acceptance at the chosen step clears the 0.9 target."""
    narrow = Model([EuclideanParameter("x", (2,))],
                   {"lp": lambda p: -0.5 * jnp.sum((p["x"] / 0.05) ** 2)})
    s = make_sampler(narrow, seed=0).initialize()
    assert float(s.state.step_size) < 0.5
    assert _mala_accept_at(s, float(s.state.step_size)) >= 0.9

    easy = correlated_gaussian(mean=[0.0, 0.0], cov=np.eye(2))
    s2 = make_sampler(easy.model, seed=0).initialize()
    assert float(s2.state.step_size) == pytest.approx(0.5)        # no shrink needed
    assert _mala_accept_at(s2, float(s2.state.step_size)) >= 0.9


def test_initialize_must_precede_warmup():
    prob = correlated_gaussian(mean=[0.0, 0.0], cov=np.eye(2))
    s = make_sampler(prob.model, seed=0)
    s.warmup(3)
    with pytest.raises(RuntimeError, match="before warmup"):
        s.initialize()


def test_not_initializing_is_a_no_op():
    """Without calling initialize() the start is unchanged (the mixins are inert)."""
    prob = correlated_gaussian(mean=[3.0, 3.0], cov=np.eye(2))
    s = make_sampler(prob.model, seed=0)
    assert np.allclose(np.asarray(s.state.coordinate), 0.0)
    assert float(s.state.step_size) == pytest.approx(0.5)


def test_initialized_factory_samples_correctly(artifacts_dir):
    """make_sampler(gaussian).initialize() then warmup/sample is correct (NUTS + init mixins)."""
    prob = correlated_gaussian(mean=[5.0, -3.0], cov=[[2.0, 1.4], [1.4, 1.5]])
    report = evaluate(prob, {"init": lambda m, seed: make_sampler(m, seed=seed).initialize()},
                      n_warmup=2000, n_samples=8000, seed=0,
                      out_dir=str(artifacts_dir / "initialized_factory"))
    print("\n" + report.summary())
    report.assert_correct()


def test_init_mixins_compose_with_hmc():
    """The init mixins also compose with fixed HMC: initialize() sets a boxed position and a
    line-searched step, and the chain then samples."""
    prob = correlated_gaussian(mean=[1.0, -2.0], cov=[[2.0, 1.4], [1.4, 1.5]])
    Cls = make_sampler_class(
        StepSizeLineSearch, UniformInit, RobbinsMonroStepSize, ScoreMassAdaptation, HMC)
    s = Cls(prob.model, init_position=np.zeros(2), seed=0,
            n_leapfrog=20, step_size=0.5, metric="diagonal", target_accept=0.8)
    s.initialize()
    assert np.all(np.abs(np.asarray(s.state.coordinate)) <= 2.0)
    assert _mala_accept_at(s, float(s.state.step_size)) >= 0.9
    s.warmup(1000)
    x = s.sample(4000)["x"]
    assert np.all(np.abs(x.mean(axis=0) - np.array([1.0, -2.0])) < 0.3)
