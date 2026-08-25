"""Quickstart: a model in the DSL, a sampler from the factory, draws, diagnostics.

This is the shortest complete session with the library, and the one to read first. A linear
regression on simulated data:

    y_i ~ Normal(alpha + x_i . beta, sigma)

The whole API used here is four calls: ``compile_model`` to turn the model source into a
:class:`mimcs.Model`, ``make_sampler`` to choose an algorithm for it, ``warmup``/``sample`` to run
it, and ``summary`` to decide whether to believe the result.

Run it with the `jax` conda environment active:

    python examples/01_quickstart.py
"""

import numpy as np

from mimcs import compile_model, make_sampler

# The model, in the DSL (docs/reference/model_dsl.md).
#
# Two things worth noticing. `sigma` is declared `<lower=0>` and that is all you do about it: you
# write the density in terms of sigma itself, and the compiled model supplies the log-Jacobian for
# the log chart the sampler works in. And the density is split into two *named* components, which
# does not change the model -- their sum is the joint log-density -- but keeps the cheap prior and
# the expensive likelihood as separate computations, which is what lets a sampler treat them
# differently.
SOURCE = """
data {
  int n;                  // observations
  int d;                  // predictors
  array[n, d] real X;
  array[n]    real y;
}

parameters {
  real            alpha;  // intercept
  array[d] real   beta;   // coefficients
  real<lower=0>   sigma;  // noise scale
}

model prior {
  alpha ~ normal(0, 5);
  beta  ~ normal(0, 5);
  sigma ~ lognormal(0, 1);
}

model likelihood {
  y ~ normal(alpha + X * beta, sigma);    // `*` between two arrays is matrix multiplication
}
"""

TRUE_ALPHA = 0.5
TRUE_BETA = np.array([1.5, -0.8, 0.3])
TRUE_SIGMA = 0.4


def simulate(n=100, seed=0):
    """Draw a dataset from the model above, so we know what the answer should be."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, len(TRUE_BETA)))
    y = TRUE_ALPHA + X @ TRUE_BETA + TRUE_SIGMA * rng.standard_normal(n)
    return X, y


def main():
    X, y = simulate()
    n, d = X.shape

    # 1. Compile. Passing `data` gives a finished Model; omit it and you get a ModelFactory you
    #    can `build(data)` repeatedly across datasets.
    model = compile_model(SOURCE, data={"n": n, "d": d, "X": X, "y": y})
    print(f"model: {len(model.parameters)} parameters, "
          f"{model.ambient_dim} ambient / {model.coord_dim} coordinate dimensions\n")

    # 2. Build a sampler. The factory picks the algorithm, the coordinate blocking, the mass
    #    adaptation and the warmup-termination criterion; `spec.rationale` (example 03) explains
    #    every choice. The defaults are meant to be a good answer, not a placeholder.
    sampler = make_sampler(model, seed=0)

    # 3. Run it. `initialize()` is optional: it picks a starting point and a starting step size
    #    instead of beginning at the charts' origin.
    sampler.initialize()

    # `warmup(n)`'s n is an *upper bound*. The default sampler carries a termination criterion, so
    # warmup ends as soon as the chain looks to be mixing -- watch the log line report where it
    # actually stopped.
    sampler.warmup(1000)

    # `sample` returns the draws keyed by parameter name; each is (n_draws, *shape).
    draws = sampler.sample(1000)
    print(f"\ndraws: {', '.join(f'{k}{v.shape}' for k, v in draws.items())}\n")

    # 4. Evaluate. Two tables: a posterior summary per coordinate, and per-*feature* diagnostics.
    #
    #    - ess     how much the autocorrelation costs; ess/n near 1 is as good as independent.
    #    - R-hat   split-R-hat over the run's two halves, a mixing check. The summary warns
    #              above 1.1; 1.01 is the stricter modern recommendation, and what the R-hat
    #              warmup-termination criterion stops on.
    #    - stein-z target-aware. ESS and R-hat only ask whether the draws look like a well-mixed
    #              sample from *some* distribution; the Langevin-Stein z asks whether it is a
    #              sample from *this* one, which is the question you actually have. Note that with
    #              m features about 0.05*m of them exceed the 95% band by chance, so one `*` in a
    #              table of ten is expected noise, not a verdict -- the summary prints how many to
    #              expect for exactly this reason.
    print(sampler.summary())

    # 5. Did it work? We simulated the data, so the truth is known.
    #
    #    Do not expect the posterior mean to *equal* the truth: with n=100 observations the
    #    posterior is a distribution of respectable width, and the truth should sit inside it, not
    #    on top of it. The honest check is the distance in posterior standard deviations, which is
    #    what the last column reports -- a couple of them landing near 1.5 is what a correct
    #    sampler on 100 data points looks like.
    print("\nRecovered vs truth")
    print(f"  {'':>6} {'post. mean':>11} {'truth':>8} {'sd':>7}   distance")
    truth = {"alpha": TRUE_ALPHA, "sigma": TRUE_SIGMA,
             **{f"beta{j}": b for j, b in enumerate(TRUE_BETA)}}
    got = {"alpha": draws["alpha"], "sigma": draws["sigma"],
           **{f"beta{j}": draws["beta"][:, j] for j in range(len(TRUE_BETA))}}
    for name, true_value in truth.items():
        chain = np.asarray(got[name])
        sd = chain.std(ddof=1)
        print(f"  {name:>6} {chain.mean():>11.3f} {true_value:>8.3f} {sd:>7.3f}   "
              f"{abs(chain.mean() - true_value) / sd:>4.1f} sd")


if __name__ == "__main__":
    main()
