"""End-to-end: the factory recommends a *sparse* learned metric after a first run.

A horseshoe regression puts a per-coefficient local scale on the coefficients:
``beta_j ~ N(0, lambda_j)``, ``lambda_j ~ half-Cauchy(0, tau0)``, ``y ~ N(X beta, sigma)``.
The coefficient block ``beta`` then has a conditional score covariance that depends
*elementwise* on the local scales --- ``M_j = 1/sigma^2 + 1/lambda_j^2`` --- a bijective row
correspondence between the equal-dimension ``beta`` and ``lambda`` arrays. This is exactly the
structure the sparse mini-language atoms (:class:`mimcs.hmc.metric_expr.SpExp`) capture and the
dense atoms cannot afford (a dense ``Exp("lambda")`` would need ``P*P`` parameters, over the
factory's ``20*P`` budget).

The test exercises the factory's refinement workflow: a **first run with the default sampler**
produces evidence (coordinates + scores, recomputed from its draws), and feeding that run back
to the factory turns the ``beta`` block into a ``learned_metric`` carrying the sparse
``SpExp("lambda")`` form. Seed-pinned: this is a statistical test on a hard shrinkage funnel
(stable across run-1 lengths at this seed), as with the rest of the suite.
"""

import numpy as np
import pytest

from mimcs.dsl import compile_model
from mimcs.factory import make_sampler, analyze, normalize
from mimcs.hmc import LearnedDiagonalBlock
from mimcs.adaptation import MetricAdaptation

_SRC = """
data { int N; int P; array[N, P] real X; array[N] real y; real sigma; real tau0; }
parameters { array[P] real beta; array[P] real<lower=0> lambda; }
model {
    lambda ~ cauchy(0, tau0);        // half-Cauchy local scales (horseshoe)
    beta ~ normal(0, lambda);        // local-scale shrinkage (the funnel geometry)
    y ~ normal(X * beta, sigma);     // orthonormal design -> per-coef likelihood precision 1/sigma^2
}
"""


def _horseshoe_model(seed=0, N=100, P=24, sigma=0.3, tau0=0.5):
    """A horseshoe regression on synthetic data: 5 signal coefficients, the rest ~0, with an
    orthonormal design (so the likelihood precision of ``beta`` is ``1/sigma^2 * I``)."""
    rng = np.random.default_rng(seed)
    X, _ = np.linalg.qr(rng.standard_normal((N, P)))          # orthonormal columns
    beta_true = np.zeros(P)
    beta_true[:5] = [2.0, -2.0, 1.5, -1.5, 2.5]
    y = X @ beta_true + sigma * rng.standard_normal(N)
    return compile_model(_SRC, data={"N": N, "P": P, "X": X, "y": y,
                                     "sigma": sigma, "tau0": tau0})


@pytest.fixture(scope="module")
def first_run():
    """One run of the factory *default* sampler on the horseshoe model (shared by the tests)."""
    model = _horseshoe_model(seed=0)
    sampler = make_sampler(model, seed=0)
    sampler.initialize()
    sampler.warmup(2000)
    sampler.sample(3000)
    return model, sampler


def test_first_run_yields_coordinate_and_gradient_evidence(first_run):
    """A live sampler now carries the coordinates and scores of its draws (recomputed from the
    ambient samples) --- the inputs the metric-regression rule needs."""
    model, sampler = first_run
    ev = normalize(model, sampler)
    assert ev.coordinates is not None and ev.gradients is not None
    assert ev.coordinates.shape == ev.gradients.shape == (3000, model.coord_dim)
    assert np.all(np.isfinite(ev.gradients))


def test_factory_recommends_sparse_metric_after_first_run(first_run):
    """Feeding the first run back to the factory turns the coefficient block into a
    ``learned_metric`` carrying the sparse ``SpExp("lambda")`` form."""
    model, sampler = first_run
    spec = analyze(model, sampler)

    beta = next(b for b in spec.blocks if b.names == ["beta"])
    assert beta.kind == "learned_metric", [(b.names, b.kind) for b in spec.blocks]
    metric = beta.params["metric"]
    assert "SpExp" in repr(metric), f"expected a sparse metric, got {metric!r}"
    assert metric.deps() == {"lambda"}
    assert any("learned_metric" in r for r in spec.rationale)


def test_recommended_sampler_builds_and_samples(first_run):
    """The recommended spec lowers to a working sampler: a learned (sparse) metric block plus
    ``MetricAdaptation``, running to produce finite draws of the right shape.

    (We assert the lowering and liveness, not a sampling-quality threshold: this centered
    horseshoe is a knife-edge funnel, so the learned metric's fitted parameters -- and hence the
    chain's mixing -- shift with float32-level noise in the fitted metric, e.g. between the
    sampler's *saved* gradients and *recomputed* ones. The metric *recommendation* is robust to
    that, as ``test_factory_recommends_sparse_metric_after_first_run`` checks.)"""
    model, sampler = first_run
    s2 = analyze(model, sampler).build(seed=1)

    assert MetricAdaptation in type(s2).__mro__
    assert any(isinstance(k, LearnedDiagonalBlock) and k.id == "beta" for k in s2.kinetics)
    s2.initialize()
    s2.warmup(500)
    s2.sample(1000)
    draws = s2.get_samples_flat()
    assert draws.shape == (1000, model.ambient_dim) and np.all(np.isfinite(draws))
