"""Capture golden ``initialize()`` outcomes from the CURRENT code, for the bit-identity guard.

Run on ``dev`` *before* the configurable-initialization change (fixed sets, spec slots), and never
again: the resulting ``tests/data/golden_init.npz`` is what lets ``tests/test_initialization.py``
assert that the default path --- nothing supplied, no evidence --- still starts from exactly the
same coordinate, labels and step size after the old code is gone.

    python tests/_golden_init_capture.py

``initialize()`` draws from dedicated substreams of the seed, not from the per-step RNG buffer, so
``buffer_size`` does not affect these values; the default is used throughout.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

OUT = pathlib.Path(__file__).resolve().parent / "data" / "golden_init.npz"


def golden_cases():
    """``{name: () -> initialized sampler}`` --- shared with the test, so both build the same."""
    import jax.numpy as jnp
    from mimcs.factory import analyze, make_sampler
    from mimcs.model import (BoundedParameter, EuclideanParameter, IntegerParameter, Model,
                             PositiveParameter)
    from mimcs.pt import parallel_tempering
    from mimcs.samplers import StaticContinuous, SystematicScanMetropolisWithinGibbs, \
        make_sampler_class

    def gaussian():
        mean = jnp.asarray([5.0, -3.0, 10.0])
        return Model([EuclideanParameter("x", (3,))],
                     {"lp": lambda p: -0.5 * jnp.sum((p["x"] - mean) ** 2 / jnp.asarray(
                         [1.0, 4.0, 0.25]))})

    def narrow():
        return Model([EuclideanParameter("x", (2,))],
                     {"lp": lambda p: -0.5 * jnp.sum((p["x"] / 0.05) ** 2)})

    def bounded():
        return Model([PositiveParameter("sig"), BoundedParameter("r", (2,), lower=0.0, upper=1.0),
                      EuclideanParameter("mu", (2,))],
                     {"lp": lambda p: (-0.5 * jnp.log(p["sig"]) ** 2
                                       + jnp.sum(jnp.log(p["r"]) + jnp.log1p(-p["r"]))
                                       - 0.5 * jnp.sum(p["mu"] ** 2) / p["sig"] ** 2
                                       - 2 * jnp.log(p["sig"]))})

    def mixture(k=3, per=6):
        rng = np.random.default_rng(0)
        y = np.concatenate([rng.normal(3.0 * (j - (k - 1) / 2), 1.0, per) for j in range(k)])

        def lp(v):
            return -0.5 * jnp.sum((y - v["mu"][v["z"]]) ** 2) - 0.5 * jnp.sum(v["mu"] ** 2)

        return Model([EuclideanParameter("mu", (k,))], {"p": lp},
                     discrete_parameters=[IntegerParameter("z", (k * per,), lower=0,
                                                           upper=k - 1)])

    def binary_only():
        w = jnp.asarray([1.3, -0.7, 2.1])
        return Model([], {"p": lambda v: jnp.dot(w, v["z"].astype(float))},
                     discrete_parameters=[IntegerParameter("z", (3,), lower=0, upper=1)])

    def pt_factory():
        spec = analyze(mixture())
        spec.base = "pt_nuts"
        spec.tempering_params = {"n_temperatures": 3}
        return spec.build(seed=4)

    gibbs_only = make_sampler_class(SystematicScanMetropolisWithinGibbs, StaticContinuous)
    return {
        "gaussian": lambda: make_sampler(gaussian(), seed=0).initialize(),
        "gaussian_s7": lambda: make_sampler(gaussian(), seed=7).initialize(),
        "narrow": lambda: make_sampler(narrow(), seed=0).initialize(),
        "bounded": lambda: make_sampler(bounded(), seed=3).initialize(),
        "mixture": lambda: make_sampler(mixture(), seed=1).initialize(),
        "binary_factory": lambda: make_sampler(binary_only(), seed=2).initialize(),
        "binary_hand": lambda: gibbs_only(binary_only(), binary_only().default_sample(),
                                          seed=5).initialize(),
        "pt_hand": lambda: parallel_tempering(mixture(), n_temperatures=3, seed=6).initialize(),
        "pt_factory": lambda: pt_factory().initialize(),
    }


def snapshot(sampler) -> dict:
    st = sampler.state
    out = {"sample": np.asarray(st.sample), "discrete": np.asarray(st.discrete)}
    if hasattr(st, "coordinate"):
        out["coordinate"] = np.asarray(st.coordinate)
    if getattr(st, "step_size", None) is not None:
        out["step_size"] = np.asarray(st.step_size)
    return out


def main():
    out = {}
    for name, make in golden_cases().items():
        for k, v in snapshot(make()).items():
            out[f"{name}__{k}"] = v
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, **out)
    print(f"wrote {OUT}")
    for k, v in out.items():
        print(f"  {k:28s} {str(v.shape):10s} {v.dtype}  {np.asarray(v).ravel()[:3]}")


if __name__ == "__main__":
    main()
