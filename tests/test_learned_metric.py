"""Tests for learned diagonal block metrics in explicit RMHMC.

A learned metric is a mass-matrix mini-language expression (``mimcs.hmc.metric_expr``:
``Exp("v")``, ``Exp() + Exp("v")``, ...) adapted online by SGD on the KL objective
(``mimcs.adaptation.MetricAdaptation``). Its expected-loss minimiser is the conditional
gradient second moment ``E[g_i^2 | q_{-i}]``, so on Neal's funnel (``x | v ~ N(0, e^v)``,
hence ``E[g_x^2 | v] = e^{-v}``) the learned ``M_x(v) = exp(W v + b)`` must approach the
*ideal* ``e^{-v}`` --- i.e. weight ``-1`` on ``v`` and bias ``0``. We check both that the
parameters converge to the ideal and that the resulting sampler explores the funnel
correctly, validating the adapted metric against the known optimum. A single-atom
expression stores its parameters directly as ``{"W": [W], "b": b}`` in ``ham_params``.

Seeds are fixed, so pass/fail is deterministic.
"""

import numpy as np

from mimcs.testing import (
    neal_funnel_blocks, correlated_gaussian, evaluate, explicit_rmhmc)
from mimcs.hmc import Exp


def _warmup(builder, model, seed, n):
    sampler = builder(model, seed)
    sampler.warmup(n)
    return sampler.state.ham_params


def test_learned_metric_converges_to_ideal_funnel_metric():
    """On the funnel the learned ``M_x(v) = exp(W v + b)`` must approach the KL-optimal
    ``e^{-v}``: weight ``W -> -1`` on ``v`` and bias ``b -> 0``."""
    prob = neal_funnel_blocks(dim=2, scale=3.0)
    builder = explicit_rmhmc(
        metrics={"x": Exp("v")},
        n_leapfrog=25, step_size=0.25, target_accept=0.9)
    params = _warmup(builder, prob.model, seed=0, n=8000)["x"]
    W = float(np.asarray(params["W"][0]).ravel()[0])
    b = float(np.asarray(params["b"]).ravel()[0])
    assert abs(W - (-1.0)) < 0.15, f"weight {W} far from ideal -1"
    assert abs(b) < 0.3, f"bias {b} far from ideal 0"


def test_learned_metric_quadratic_features_stay_linear():
    """With the quadratic feature map ``feat(v) = [v, v^2]`` the learned metric must still
    recover the linear ideal: linear weight ``-> -1``, quadratic weight ``-> 0``."""
    prob = neal_funnel_blocks(dim=2, scale=3.0)
    builder = explicit_rmhmc(
        metrics={"x": Exp("v", features="quadratic")},
        n_leapfrog=25, step_size=0.25, target_accept=0.9)
    W = np.asarray(_warmup(builder, prob.model, seed=0, n=8000)["x"]["W"][0]).ravel()
    assert abs(W[0] - (-1.0)) < 0.2, f"linear weight {W[0]} far from -1"
    assert abs(W[1]) < 0.25, f"quadratic weight {W[1]} should be ~0"


def test_learned_constant_metric_recovers_gradient_covariance():
    """A constant (dep-less) learned block ``Exp()`` learns a position-independent diagonal
    mass. On a Gaussian its KL optimum is the gradient covariance ``E[g g^T] = precision``,
    so ``M = exp(b)`` must approach ``diag(precision)``."""
    cov = np.array([[2.0, 1.4], [1.4, 1.5]])
    precision = np.linalg.inv(cov)
    prob = correlated_gaussian(mean=[1.0, -2.0], cov=cov)
    builder = explicit_rmhmc(
        metrics={"x": Exp()},
        n_leapfrog=20, step_size=0.2, target_accept=0.8)
    b0 = np.asarray(_warmup(builder, prob.model, seed=0, n=8000)["x"]["b"]).ravel()
    ratio = np.exp(b0) / np.diag(precision)
    assert np.all((ratio > 0.6) & (ratio < 1.8)), \
        f"learned mass {np.exp(b0)} not near diag(precision) {np.diag(precision)}"


def test_metric_centering_opt_in_on_constant_block():
    """``metric_center_grad`` (off by default) may be enabled for a *constant* (marginal)
    block, where subtracting a marginal running mean of the score is appropriate. On a
    Gaussian E[score] -> 0, so the centring is inert at stationarity: the mass still lands at
    the score covariance (precision) and the running ``mean_grad`` is tracked. (Left off by
    default because a conditional block's fit is distorted by a marginal mean.)"""
    cov = np.array([[2.0, 1.4], [1.4, 1.5]])
    precision = np.linalg.inv(cov)
    prob = correlated_gaussian(mean=[1.0, -2.0], cov=cov)
    builder = explicit_rmhmc(
        metrics={"x": Exp()},
        n_leapfrog=20, step_size=0.2, target_accept=0.8, metric_center_grad=True)
    sampler = builder(prob.model, 0)
    sampler.warmup(8000)
    b0 = np.asarray(sampler.state.ham_params["x"]["b"]).ravel()
    ratio = np.exp(b0) / np.diag(precision)
    assert np.all((ratio > 0.5) & (ratio < 2.0)), \
        f"centred constant-block mass {np.exp(b0)} not near diag(precision)"
    assert sampler._metric_mean_grad is not None      # centring active


def test_a_sum_of_two_atoms_builds_and_adapts():
    """A two-term expression --- a log-linear term plus a learned constant baseline --- through
    the ``metrics=`` entry point. Every other test here passes a bare atom, so this is what
    covers a ``Sum`` reaching ``build_block`` that way, and it pins the pytree shape
    ``MetricAdaptation`` and ``ham_params`` indexing expect: one entry per additive term, a
    list rather than the single atom's ``{"W": ..., "b": ...}`` dict."""
    prob = neal_funnel_blocks(dim=2, scale=3.0)
    builder = explicit_rmhmc(
        metrics={"x": Exp("v") + Exp()},
        n_leapfrog=25, step_size=0.25, target_accept=0.9)
    params = _warmup(builder, prob.model, seed=0, n=3000)["x"]
    assert isinstance(params, list) and len(params) == 2      # Sum(Exp("v"), Exp()) pytree


def test_learned_metric_samples_funnel(artifacts_dir):
    """The learned metric -- with no metric given -- must sample the funnel correctly,
    matching what the ideal given metric achieves. Uses a milder ``scale=2.5`` funnel
    (tamer x-tail) for a strict distributional assertion, as the deep funnel's heavy tail
    is intrinsically high-variance to estimate (see test_explicit_rmhmc)."""
    prob = neal_funnel_blocks(dim=2, scale=2.5)
    prob.hard = False
    report = evaluate(
        prob,
        {"learned": explicit_rmhmc(
            metrics={"x": Exp("v")},
            n_leapfrog=20, step_size=0.3, target_accept=0.9)},
        n_warmup=5000, n_samples=15000, seed=0,
        out_dir=str(artifacts_dir / "learned_metric_funnel"))
    print("\n" + report.summary())
    report.assert_correct()
