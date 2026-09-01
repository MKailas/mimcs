"""Discrete parameters: a Gaussian mixture with the cluster labels sampled.

Run:  python examples/05_mixture.py

Most libraries cannot sample a discrete parameter, so a mixture model is written by
**marginalizing** the labels out with `log_sum_exp` — which works, costs K likelihood evaluations
per observation, and stops working as soon as the labels are coupled to one another.

mimcs samples them. `array[N] int<lower=1, upper=K> z;` declares one label per observation, and
the factory builds a Metropolis-within-Gibbs sweep composed over NUTS to move them — you do not
have to assemble it. `analyze(model)` is used here rather than the `make_sampler(model)`
one-liner only because this example wants to *show* what was decided and to raise
`target_accept`; `make_sampler(model)` alone would sample this model correctly.

Two modelling points worth copying:

* `ordered[K] mu` rather than `array[K] real mu`. A mixture is invariant under relabelling its
  components, so an unconstrained `mu` has K! equivalent modes, every component has the same
  posterior mean, and R-hat is meaningless. Ordering the means picks one labelling.
* The sweep is **not optional** and has no off switch: a sampler that held the labels frozen
  would still print plausible means, and a frozen coordinate reports a *perfect* ESS and
  R-hat 1.000. That is why the `label moves per iteration` line below is the one to read first.
  See `docs/design/14_discrete_parameters.md`.
"""

import numpy as np

from mimcs import analyze, compile_model

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

    # The factory decides the sampler; `spec` is the prototype, and every choice on it can be
    # inspected and overridden before anything is built. (`analyze` logs the whole spec at INFO,
    # so only the discrete decision is echoed here.)
    spec = analyze(model)
    print(f"factory: {spec.base} + a Gibbs sweep using the "
          f"{spec.discrete_proposal or 'uniform'} proposal for {model.discrete_dim} labels")
    spec.algo_kwargs["target_accept"] = 0.9

    sampler = spec.build(seed=0)
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
