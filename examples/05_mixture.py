"""Discrete parameters: a Gaussian mixture with the cluster labels sampled.

Run:  python examples/05_mixture.py

Most libraries cannot sample a discrete parameter, so a mixture model is written by
**marginalizing** the labels out with `log_sum_exp` — which works, costs K likelihood evaluations
per observation, and stops working as soon as the labels are coupled to one another.

mimcs samples them. `array[N] int<lower=1, upper=K> z;` declares one label per observation, and a
Metropolis-within-Gibbs sweep composed over NUTS moves them:

    make_sampler_class(RobbinsMonroStepSize, MassMatrixAdaptation,
                       DiscreteMetropolisWithinGibbs, NUTS)

Two modelling points worth copying:

* `ordered[K] mu` rather than `array[K] real mu`. A mixture is invariant under relabelling its
  components, so an unconstrained `mu` has K! equivalent modes, every component has the same
  posterior mean, and R-hat is meaningless. Ordering the means picks one labelling.
* The sampler factory **refuses** a model with `int` parameters — no rule proposes the Gibbs
  sweep yet — so the sampler is composed by hand here. See
  `docs/design/14_discrete_parameters.md`.
"""

import numpy as np

from mimcs import compile_model
from mimcs.adaptation import MassMatrixAdaptation, RobbinsMonroStepSize
from mimcs.hmc import NUTS
from mimcs.samplers import DiscreteMetropolisWithinGibbs, make_sampler_class

SOURCE = """
data {
  int n;
  int k;
  array[n] real y;
  array[k] real w;             // mixing weights (a simplex, given as data here)
}
parameters {
  ordered[k] mu;               // ordered: otherwise the labelling is not identified
  array[n] int<lower=1, upper=k> z;
  real<lower=0> sigma;
}
model {
  mu ~ normal(0, 10);
  sigma ~ lognormal(0, 1);
  for (i in 1:n) {
    z[i] ~ categorical(w);
    y[i] ~ normal(mu[z[i]], sigma);
  }
}
"""


def main() -> None:
    rng = np.random.default_rng(0)
    n, k, sep, sigma_true = 150, 3, 4.0, 1.0
    true_mu = sep * (np.arange(k) - (k - 1) / 2.0)
    true_z = rng.integers(0, k, size=n)
    y = true_mu[true_z] + sigma_true * rng.standard_normal(n)

    model = compile_model(SOURCE, data={"n": n, "k": k, "y": y, "w": np.full(k, 1.0 / k)})
    print(f"model: coord_dim {model.coord_dim} continuous, "
          f"discrete_dim {model.discrete_dim} labels")

    Sampler = make_sampler_class(RobbinsMonroStepSize, MassMatrixAdaptation,
                                 DiscreteMetropolisWithinGibbs, NUTS)
    sampler = Sampler(model, model.default_sample(), seed=0, target_accept=0.9)
    sampler.initialize()
    sampler.warmup(2000)
    draws = sampler.sample(4000)

    mu, z, s = np.asarray(draws["mu"]), np.asarray(draws["z"]), np.asarray(draws["sigma"])
    print(f"\nmu   posterior mean {np.round(mu.mean(axis=0), 2)}   true {true_mu}")
    print(f"sigma posterior mean {s.mean():.3f}   true {sigma_true}")

    # The labels come back as an integer array, one column per observation.
    assert z.dtype.kind == "i", z.dtype
    modal = np.round(np.median(z, axis=0)).astype(int)
    print(f"label agreement with the generating assignment: "
          f"{np.mean(modal == true_z + 1):.1%}")

    moves = sampler.diagnostics()["discrete_moves"]
    print(f"label moves per iteration: {moves.mean():.1f} of {n} "
          f"(0 would mean the sweep is stuck --- a frozen label reports a *perfect* ESS)")

    print()
    print(sampler.summary())


if __name__ == "__main__":
    main()
