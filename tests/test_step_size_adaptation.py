"""Tests for the Robbins--Monro step-size adaptation tweaks.

Three behaviours: the ``n0`` offset in the gain schedule, the default rate
``1/sqrt(a0(1-a0))`` derived from the target acceptance, and Kesten's acceleration
(advance the step counter only when the error changes sign). The counter logic is unit
tested with a synthetic state harness; convergence is checked end-to-end on a Gaussian.
"""

import math
from collections import namedtuple

import numpy as np
import jax.numpy as jnp

from mimcs.adaptation import RobbinsMonroStepSize
from mimcs.samplers.base import Phase
from mimcs.testing import correlated_gaussian, hmc, evaluate


# --- synthetic harness: drive the mixin with hand-chosen acceptance probs ---- #

_FakeState = namedtuple("_FakeState", ["diagnostics", "step_size"])


class _ChainEnd:
    """Terminates the cooperative ``super()`` hook chain with no-ops."""

    def _init_hooks(self, **kwargs):
        pass

    def _postprocess_hooks(self, state):
        return state


class _Harness(RobbinsMonroStepSize, _ChainEnd):
    """Minimal host exposing the mixin's hooks over a no-op chain end."""

    def __init__(self, **kwargs):
        self._phase = Phase.WARMUP
        self._init_hooks(**kwargs)

    def feed(self, accept_prob):
        state = _FakeState({"accept_prob": jnp.asarray(accept_prob)}, jnp.asarray(1.0))
        return self._postprocess_hooks(state)


def test_default_rate_from_target_accept():
    """rate defaults to 1/sqrt(a0(1-a0)); an explicit rate overrides it."""
    for target in (0.234, 0.8, 0.9):
        h = _Harness(target_accept=target)
        assert math.isclose(h._ss_rate, 1.0 / math.sqrt(target * (1.0 - target)), rel_tol=1e-6)
    assert _Harness(target_accept=0.8, step_size_adapt_rate=3.0)._ss_rate == 3.0


def test_n0_offset_in_gain_schedule():
    """The (unaccelerated) counter advances every step and the gain uses (n + n0)."""
    h = _Harness(target_accept=0.8, step_size_adapt_rate=1.0,
                 step_size_adapt_kappa=0.6, step_size_adapt_n0=5.0)
    # one warmup step: count 0 -> 1, gain = 1*(1+5)^-0.6 applied to error (1.0 - 0.8)
    out = h.feed(1.0)
    assert h._ss_count == 1
    expected_gain = (1 + 5.0) ** (-0.6)
    assert math.isclose(float(jnp.log(out.step_size)), expected_gain * (1.0 - 0.8), rel_tol=1e-5)


def test_accelerated_counter_advances_only_on_sign_flips():
    """With acceleration the counter advances only when the error sign flips."""
    h = _Harness(target_accept=0.5, accelerated=True)
    # errors: +, +, + (same sign, no advance), - (flip), - (no advance), + (flip)
    for ap in (0.9, 0.8, 0.7):
        h.feed(ap)
    assert h._ss_count == 0          # monotone same-sign run: gain held constant
    h.feed(0.1)                      # error flips negative
    assert h._ss_count == 1
    h.feed(0.2)
    assert h._ss_count == 1          # still negative
    h.feed(0.9)                      # flips positive
    assert h._ss_count == 2


def test_unaccelerated_counter_advances_every_step():
    h = _Harness(target_accept=0.5)
    for ap in (0.9, 0.8, 0.1, 0.2):
        h.feed(ap)
    assert h._ss_count == 4


# --- end-to-end: both modes converge to the target acceptance ---------------- #


def test_adaptation_converges_to_target_accept():
    """Over seeds, not on one.

    A single chain's realized acceptance scatters with sd ~0.10 around the target here, so a
    one-seed ``|accept - 0.8| < 0.1`` is a coin flip dressed as a threshold: measured over 8 seeds
    it fails on **4 of them** for both variants, and passed only because seed 0 happened to land
    inside the band. What the adaptation actually promises is that the acceptance is centred on the
    target, so that is what is asserted -- the mean over 6 seeds, at a tolerance the spread of the
    mean supports (sd/sqrt(6) ~ 0.04).
    """
    prob = correlated_gaussian(mean=[1.0, -2.0], cov=[[2.0, 1.4], [1.4, 1.5]])
    for accelerated in (False, True):
        accepts = []
        for seed in range(6):
            s = hmc(n_leapfrog=20, step_size=0.5, target_accept=0.8,
                    accelerated=accelerated)(prob.model, seed=seed)
            s.warmup(2000)
            s.sample(2000)
            accepts.append(float(s.acceptance_rate()))
        mean_accept = float(np.mean(accepts))
        assert abs(mean_accept - 0.8) < 0.05, \
            f"accelerated={accelerated}: mean accept {mean_accept:.4f} over {accepts} " \
            f"is not centred on the target 0.8"


# --- warmup step-size trajectory recording + plotting ------------------------ #


def test_warmup_step_sizes_recorded():
    """The step size is saved once per warmup iteration, not during sampling."""
    prob = correlated_gaussian(mean=[1.0, -2.0], cov=[[2.0, 1.4], [1.4, 1.5]])
    s = hmc(n_leapfrog=20, step_size=0.5, target_accept=0.8)(prob.model, seed=0)
    s.warmup(500)
    traj = s.warmup_step_sizes()
    assert traj.shape == (500,)
    assert np.all(traj > 0.0)
    assert traj[0] != traj[-1]              # adaptation actually moved the step size
    s.sample(200)
    assert s.warmup_step_sizes().shape == (500,)   # sampling does not extend it


def test_evaluate_writes_step_size_plot(artifacts_dir):
    """The harness records the warmup step sizes and writes a step-size plot."""
    import os
    prob = correlated_gaussian(mean=[1.0, -2.0], cov=[[2.0, 1.4], [1.4, 1.5]])
    out_dir = str(artifacts_dir / "step_size_plot")
    report = evaluate(prob, {"hmc": hmc(n_leapfrog=20, step_size=0.5)},
                      n_warmup=800, n_samples=2000, seed=0, out_dir=out_dir)
    assert report.outputs["hmc"].warmup_step_sizes.shape == (800,)
    assert os.path.exists(os.path.join(out_dir, "step_size_hmc.png"))
