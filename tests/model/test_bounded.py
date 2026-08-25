"""Correctness tests for ``mimcs/model/bounded.py``: constrained and parent-dependent parameters.

These exercise the chart machinery: a positive parameter (log chart), a bounded
interval (logit chart), and a parent-dependent bound (``b ~ U(0, a)``). Each is
sampled with adaptive MH and compared against an exact reference. A finite-difference
check confirms the coordinate-space gradient (with parent threading) is correct, which
is what HMC will rely on.

Sampling in coordinate space turns these constrained targets into easy unconstrained
ones, so they mix well and the checks are tight.
"""

import numpy as np
import jax
import jax.numpy as jnp

from mimcs.testing import (
    positive_lognormal, uniform_interval, nested_uniform, evaluate, adaptive_mh)


def test_positive_parameter_lognormal(artifacts_dir):
    problem = positive_lognormal(sigma=1.0)
    report = evaluate(
        problem,
        {"rwmh": adaptive_mh(init=np.array([1.0]), step_size=0.8, target_accept=0.35)},
        n_warmup=3000, n_samples=15000, seed=0,
        out_dir=str(artifacts_dir / "positive_lognormal"),
    )
    print("\n" + report.summary())
    report.assert_correct()


def test_interval_parameter_uniform(artifacts_dir):
    problem = uniform_interval(lower=-2.0, upper=3.0)
    report = evaluate(
        problem,
        {"rwmh": adaptive_mh(init=np.array([0.5]), step_size=0.8, target_accept=0.35)},
        n_warmup=3000, n_samples=15000, seed=1,
        out_dir=str(artifacts_dir / "uniform_interval"),
    )
    print("\n" + report.summary())
    report.assert_correct()


def test_parent_dependent_bound(artifacts_dir):
    """b ~ Uniform(0, a): b's upper bound is the value of parent a."""
    problem = nested_uniform()
    report = evaluate(
        problem,
        {"rwmh": adaptive_mh(init=np.array([0.5, 0.25]), step_size=0.8,
                             target_accept=0.35)},
        n_warmup=3000, n_samples=15000, seed=2,
        out_dir=str(artifacts_dir / "nested_uniform"),
    )
    print("\n" + report.summary())
    report.assert_correct()


def test_parent_model_gradient_matches_finite_difference():
    """The coordinate-space gradient must thread through the parent-dependent chart.

    Validates that autodiff over ``log_prob_at_coordinate`` is correct when a child's
    Jacobian depends on a parent's value (the triangular-map case HMC will use)."""
    model = nested_uniform().model
    h = model.init_chart_hyperparams()
    c = model.init_chart_indices()

    def f(q):
        return model.log_prob_at_coordinate(q, h, c)

    q0 = np.array([0.3, -0.4])
    grad = np.asarray(jax.grad(lambda q: f(q))(jnp.array(q0)))

    # Central differences. eps ~ 1e-3 is the sweet spot for float32 (smaller eps is
    # dominated by rounding error, larger by truncation error).
    eps = 1e-3
    fd = np.array([
        (float(f(jnp.array(q0 + eps * e))) - float(f(jnp.array(q0 - eps * e)))) / (2 * eps)
        for e in np.eye(2)
    ])

    assert grad.shape == (2,)
    assert np.all(np.isfinite(grad))
    assert np.allclose(grad, fd, atol=1e-3), f"grad {grad} != fd {fd}"
