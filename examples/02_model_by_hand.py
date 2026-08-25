"""Building a ``Model`` directly: parameter objects plus JAX log-density functions.

The DSL is a front end. Underneath, a :class:`mimcs.Model` is nothing more than a list of typed
parameters and a dict of named log-density functions, and you can write both yourself -- which is
what you do when the density is easier to express in JAX than in the DSL, or when it calls code the
DSL has no syntax for.

This builds the *same* regression as example 01 by hand, and checks the two agree.

Run it with the `jax` conda environment active:

    python examples/02_model_by_hand.py
"""

import numpy as np
import jax.numpy as jnp
from jax.scipy.stats import norm

from mimcs import Model, analyze, compile_model, make_sampler
from mimcs.model import EuclideanParameter, PositiveParameter

TRUE_ALPHA = 0.5
TRUE_BETA = np.array([1.5, -0.8, 0.3])
TRUE_SIGMA = 0.4


def simulate(n=100, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, len(TRUE_BETA)))
    y = TRUE_ALPHA + X @ TRUE_BETA + TRUE_SIGMA * rng.standard_normal(n)
    return X, y


def build_model(X, y):
    """The regression of example 01, assembled from parts."""
    X, y = jnp.asarray(X), jnp.asarray(y)
    d = X.shape[1]

    # The parameters. Each one owns the *chart* that maps its natural (ambient) value to the
    # unconstrained coordinate the sampler moves in: identity for a Euclidean parameter, log for a
    # positive one. `mimcs.model` also has BoundedParameter/IntervalParameter, UnitVectorParameter,
    # SimplexParameter and OrderedParameter.
    parameters = [
        EuclideanParameter("alpha"),          # scalar: shape ()
        EuclideanParameter("beta", (d,)),
        PositiveParameter("sigma"),           # sigma > 0, log chart
    ]

    # The log-density components. Each is a pure function of a dict {name: ambient array} and
    # returns a scalar; their sum is the joint log-density. Write them in terms of the parameters'
    # natural values -- sigma here is sigma, not log sigma. You never add a Jacobian: the chart
    # correction is the Model's job, and doing it yourself would double-count it.
    def prior(p):
        return (norm.logpdf(p["alpha"], 0.0, 5.0).sum()
                + norm.logpdf(p["beta"], 0.0, 5.0).sum()
                + norm.logpdf(jnp.log(p["sigma"]), 0.0, 1.0).sum() - jnp.log(p["sigma"]))

    def likelihood(p):
        return norm.logpdf(y, p["alpha"] + X @ p["beta"], p["sigma"]).sum()

    # `cheap_components` names the components whose *gradient* is cheap -- metadata about cost that
    # a multi-rate integrator can use, and which never changes what the model is. The DSL infers it
    # from data sizes; by hand you say it. (The two do differ here: with only 100 rows the DSL's
    # size rule calls both components cheap, which is why the check below compares densities rather
    # than this label.)
    return Model(parameters, {"prior": prior, "likelihood": likelihood},
                 cheap_components=("prior",))


# The same model in the DSL, to check the hand-built one against.
SOURCE = """
data { int n; int d; array[n, d] real X; array[n] real y; }
parameters { real alpha; array[d] real beta; real<lower=0> sigma; }
model prior      { alpha ~ normal(0, 5); beta ~ normal(0, 5); sigma ~ lognormal(0, 1); }
model likelihood { y ~ normal(alpha + X * beta, sigma); }
"""


def log_density_at(model, sample_flat):
    """The coordinate-space target at an ambient point: what the sampler sees, Jacobian included.

    The chart arguments are the *state* of the charts -- the adapted hyperparameters (a centering
    location and scale, a unit vector's stereographic pole) and which chart of an atlas is in use.
    A sampler threads its own; these are the initial ones.
    """
    charts = (model.init_chart_hyperparams(), model.init_chart_indices())
    coordinate = model.sample_to_coordinate(sample_flat, *charts)
    return float(model.log_prob_at_coordinate(coordinate, *charts))


def main():
    X, y = simulate()
    n, d = X.shape

    model = build_model(X, y)
    reference = compile_model(SOURCE, data={"n": n, "d": d, "X": X, "y": y})

    # The three spaces (docs/design/10). A draw is stored in *sample* space -- the parameters'
    # natural values, which is what get_samples() hands back and what you report. The sampler works
    # in *coordinate* space, where sigma is log sigma and unbounded. Here the two have the same
    # dimension, as they do for every Euclidean and bounded parameter; a unit vector is the case
    # where they differ (d ambient components, d-1 coordinates).
    print(f"ambient (sample) dim {model.ambient_dim}, coordinate dim {model.coord_dim}")
    print(f"default_sample()     {np.asarray(model.default_sample()).round(3)}   "
          "<- the charts' origin: a valid point whatever the parameter types\n")

    # Same density? Compare both models at a point neither was built around. The quantity to
    # compare is the *coordinate-space* target -- the ambient density pulled through the charts,
    # plus the log-Jacobian -- because that is what a sampler actually differentiates.
    point = model.default_sample() + 0.3
    mine, theirs = log_density_at(model, point), log_density_at(reference, point)
    print(f"log density at a test point: by hand {mine:.6f}, DSL {theirs:.6f}  "
          f"(difference {abs(mine - theirs):.2e})\n")

    # The same density, but not necessarily the same *sampler*. `cheap_components` is a claim
    # about gradient cost, and the factory acts on it: ours declares the prior cheap, so a
    # multi-rate integrator can kick that cheap gradient several times per expensive one. The DSL's
    # size rule saw only 100 rows of data, called both components cheap, and so found nothing to
    # split. Cost metadata changes how the model is sampled; it never changes what the model is.
    print(f"integrator for the hand-built model: {analyze(model).integrator}")
    print(f"integrator for the DSL model:        {analyze(reference).integrator}\n")

    # From here everything is identical to example 01: the factory does not care where the Model
    # came from.
    sampler = make_sampler(model, seed=0)
    sampler.initialize()
    sampler.warmup(1000)
    draws = sampler.sample(1000)
    print()
    print(sampler.summary())

    print("\nRecovered vs truth")
    print(f"  alpha  {draws['alpha'].mean():+.3f}  (true {TRUE_ALPHA:+.3f})")
    for j, true_b in enumerate(TRUE_BETA):
        print(f"  beta{j}  {draws['beta'][:, j].mean():+.3f}  (true {true_b:+.3f})")
    print(f"  sigma  {draws['sigma'].mean():+.3f}  (true {TRUE_SIGMA:+.3f})")


if __name__ == "__main__":
    main()
