"""Assembling a sampler by hand with ``make_sampler_class`` (advanced).

The factory (examples 01 and 03) exists so you do not have to do this. Reach for it when you want a
combination the factory will not choose, when you are writing a new adaptation mixin, or when you
want to see exactly what a sampler is made of.

A sampler in this library is a **base algorithm plus adaptation mixins**, composed into one class::

    Sampler = make_sampler_class(UniformInit, RobbinsMonroStepSize, MassMatrixAdaptation, NUTS)

The mixins cooperate through ``super()`` chains on six hooks -- ``_init_hooks``,
``_init_state_hooks``, ``_preprocess_hooks``, ``_postprocess_hooks`` (where adaptation runs, during
warmup only), ``_finalize_hooks`` (once, at the warmup/sampling boundary, e.g. to freeze an averaged
mass) and ``_initialize_hooks`` (once, on ``initialize()``) -- plus ``should_stop()`` for warmup
termination. Adding a mixin never requires editing an existing one.

**The one rule that is not guessable: mixin order is hook order, and the mixin whose work must
happen first goes last**, closest to the base algorithm. See ``docs/design/02_sampler_classes.md``.

The target is a correlated 2-D Gaussian, chosen because it makes the mass matrix's job visible: its
covariance has a condition number near 20, so an identity mass forces the integrator into steps
sized for the narrow direction while it crawls along the wide one.

Run it with the `jax` conda environment active:

    python examples/04_sampler_by_hand.py
"""

import numpy as np
import jax.numpy as jnp

from mimcs import Model, make_sampler_class
from mimcs.model import EuclideanParameter
from mimcs.hmc import NUTS
from mimcs.adaptation import (UniformInit, StepSizeLineSearch, RobbinsMonroStepSize,
                             MassMatrixAdaptation)

MEAN = np.array([1.0, -2.0])
COV = np.array([[4.0, 3.6], [3.6, 4.0]])


def build_model():
    precision = jnp.asarray(np.linalg.inv(COV))
    mean = jnp.asarray(MEAN)

    def log_post(params):
        delta = params["x"] - mean
        return -0.5 * delta @ precision @ delta

    return Model([EuclideanParameter("x", (2,))], {"log_post": log_post})


def main():
    model = build_model()
    print(f"target: 2-D Gaussian, condition number {np.linalg.cond(COV):.1f}\n")

    # Compose the class. Read the list right to left, in the order the work happens:
    #
    #   NUTS                  the base algorithm -- the trajectory and the acceptance
    #   MassMatrixAdaptation  fits the mass to the empirical covariance of the draws
    #   RobbinsMonroStepSize  drives the acceptance rate to `target_accept`
    #   StepSizeLineSearch    one-off: a MALA line search for a starting step size
    #   UniformInit           one-off: a U(-2, 2) draw in coordinate space as the starting point
    #
    # The two initialization mixins act only inside `initialize()`, the two adaptation mixins only
    # during warmup. Swapping in ScoreMassAdaptation, MetricAdaptation, a centering mixin or a
    # termination criterion is a change to this list and nothing else.
    Sampler = make_sampler_class(
        UniformInit, StepSizeLineSearch, RobbinsMonroStepSize, MassMatrixAdaptation, NUTS)
    print(f"composed class: {Sampler.__name__}")
    print("      MRO: " + " -> ".join(c.__name__ for c in Sampler.__mro__[:6]) + " -> ...\n")

    # Constructor arguments go to whichever mixin reads them (docs/reference/algo_kwargs.md).
    # `metric="dense"` asks for a full mass matrix rather than a diagonal one -- the right choice
    # here, since the target's correlation is exactly what a diagonal mass cannot represent.
    sampler = Sampler(
        model,
        init_position=model.default_sample(),
        seed=0,
        step_size=0.5,
        metric="dense",
        target_accept=0.8,
        mass_min_samples=50,      # no mass is written before this many warmup draws
    )

    sampler.initialize()
    sampler.warmup(1000)
    draws = sampler.sample(2000)
    print()
    print(sampler.summary())

    # Did the adaptation do its job? The fitted mass should look like the target's covariance:
    # that is precisely the estimate MassMatrixAdaptation accumulates.
    x = np.asarray(draws["x"])
    print("\nRecovered vs truth")
    print(f"  mean        {x.mean(0).round(3)}   (true {MEAN})")
    print(f"  covariance  {np.cov(x.T).round(2).tolist()}")
    print(f"              (true {COV.tolist()})")
    print(f"  step size   {float(sampler.state.step_size):.3f}, "
          f"acceptance {sampler.acceptance_rate():.3f}, "
          f"mean trajectory {sampler.mean_n_leaves():.1f} leaves")


if __name__ == "__main__":
    main()
