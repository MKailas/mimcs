"""Capture golden discrete draws from the CURRENT code, for the bit-identity guard.

Run on `dev` *before* the per-parameter-updater refactor, and never again: the resulting
`tests/data/golden_discrete.npz` is what lets `tests/test_discrete_exact.py` assert that a
Metropolis-only model is unchanged **after** the old code path has been deleted. The seed-pinned
suites catch a shifted stream too, but only while the old path still exists to compare against;
this file outlives it.

    python tests/_golden_discrete_capture.py

Three arrangements, chosen to span the two sweep paths and the composed stack:

* ``binary``  -- a hand-written model under ``StaticContinuous``: no scan component, so the
                 **verbatim full-density path**, and a discrete-only sampler.
* ``loop``    -- the DSL mixture written with a ``for``: NUTS + Gibbs, still the full path.
* ``scan``    -- the same posterior written with ``model lik scan(z, y)``: the **delta path**.

`buffer_size` is quoted with the seed because it is *not* stream-neutral (draws agree only up to
the first refill), so a golden file that omitted it would be unreproducible.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import mimcs                                                          # noqa: E402
from mimcs.adaptation import RobbinsMonroStepSize                     # noqa: E402
from mimcs.hmc import NUTS                                            # noqa: E402
from mimcs.samplers import (DiscreteMetropolisWithinGibbs,            # noqa: E402
                            StaticContinuous, make_sampler_class)

from test_discrete import _binary_model                               # noqa: E402
from test_discrete_restricted import LOOP, SCAN, _data                # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "data" / "golden_discrete.npz"

SEED = 0
BUFFER = 1024          # the default, quoted because it is not stream-neutral
GIBBS_ONLY = make_sampler_class(DiscreteMetropolisWithinGibbs, StaticContinuous)
NUTS_GIBBS = make_sampler_class(RobbinsMonroStepSize, DiscreteMetropolisWithinGibbs, NUTS)


def _capture(sampler, n_warmup, n_draws):
    sampler.initialize()
    sampler.warmup(n_warmup)
    sampler.sample(n_draws)
    z = np.asarray(sampler.get_discrete_flat())
    x = np.asarray(sampler.get_samples_flat())
    return z[:20], x[-1]


def main():
    out = {}

    m, _ = _binary_model()
    z, x = _capture(GIBBS_ONLY(m, m.default_sample(), seed=SEED, buffer_size=BUFFER), 200, 500)
    out["binary_z"], out["binary_x"] = z, x

    for name, src in (("loop", LOOP), ("scan", SCAN)):
        model = mimcs.compile_model(src, data=_data())
        z, x = _capture(
            NUTS_GIBBS(model, model.default_sample(), seed=SEED, buffer_size=BUFFER), 200, 500)
        out[f"{name}_z"], out[f"{name}_x"] = z, x

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, **out)
    print(f"wrote {OUT} (seed {SEED}, buffer_size {BUFFER})")
    for k, v in out.items():
        print(f"  {k:10s} {str(v.shape):10s} {v.dtype}  {np.asarray(v).ravel()[:4]}")


if __name__ == "__main__":
    main()
