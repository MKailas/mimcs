"""mimcs: general-purpose Bayesian inference with adaptive MCMC on JAX.

Write a model, hand it to the factory, sample::

    import mimcs

    model = mimcs.compile_model(source, data)      # a Stan-like DSL, or build a Model by hand
    sampler = mimcs.make_sampler(model)            # picks the algorithm, blocks and adaptations
    sampler.warmup(2000)                          # adapts; an upper bound, not a count
    draws = sampler.sample(4000)                  # {parameter name: draws}
    print(sampler.summary())                      # per-feature ESS, split-R-hat, Stein z

**Start with** ``docs/reference/sampler_factory.md`` and ``docs/reference/model_dsl.md``;
``docs/design/`` is the architecture, in numbered order from ``00_overview.md``.

What is here:

* **Samplers** --- HMC, randomized HMC, NUTS, WALNUTS (within-orbit adaptive step size),
  explicit block-Riemannian HMC with learned position-dependent metrics, parallel tempering over
  any of them, and random-walk Metropolis--Hastings. Relativistic HMC and implicit RMHMC are
  implemented but **experimental**.
* **Adaptation** --- step size, mass (score-covariance or empirical, diagonal/dense/low-rank),
  learned metrics, chart centering, and warmup termination on a mixing criterion. Adaptations are
  mixins composed onto a base algorithm, so adding one never edits existing code.
* **Parameters** --- Euclidean, bounded/positive/interval, simplex, ordered, and unit vectors on
  ``S^(d-1)``. A parameter is defined by its *charts*, which map the ambient sample to the
  unconstrained coordinate the sampler works in; the Jacobian correction is applied for you.
* **Evaluation** --- :func:`summarize` / ``sampler.summary()``: per-feature ESS, split-R-hat, and
  a target-aware Langevin--Stein diagnostic.

**This namespace is deliberately small**: a model entry point, the factory, the composer for
hand-built samplers, the evaluation function, and logging control. Everything else lives in the
subpackage that owns it --- ``mimcs.model`` for the parameter types, ``mimcs.hmc`` and ``mimcs.pt``
for the samplers, ``mimcs.adaptation`` for the mixins, ``mimcs.testing`` for the harness. Those are
imported for you, so ``mimcs.hmc.NUTS`` works after a bare ``import mimcs`` (except ``mimcs.testing``,
which pulls in matplotlib and is not a runtime dependency).

Importing ``mimcs`` calls :func:`configure_logging`, which attaches one handler to the ``mimcs``
logger at INFO with ``propagate=False``. Set ``MIMCS_LOG_LEVEL``, or call :func:`set_log_level` or
:func:`log_level`, to change it.
"""

#: The release version. This literal is the single source of truth: ``pyproject.toml``
#: declares the version ``dynamic`` and reads it back from here.
__version__ = "0.1.7"

from ._logging import (configure_logging, get_logger, log_level, set_log_level)

# Attach the library's stream handler (INFO by default; MIMCS_LOG_LEVEL overrides) before any
# submodule can log, so importing mimcs is all it takes to see its messages.
configure_logging()

# Deliberately small. Everything else lives in the subpackage that owns it --- the parameter
# types in :mod:`mimcs.model`, the samplers in :mod:`mimcs.hmc` and :mod:`mimcs.pt`, the mixins in
# :mod:`mimcs.adaptation` --- because re-exporting all of them here would make the top-level
# namespace both large and a thing to remember to update every time the library grows. What stays
# is the handful of names a user needs to build a model, build a sampler, evaluate the draws, and
# control the logging.
from .model import Model
from .dsl import compile_model, DslError
from .samplers import make_sampler_class
from .factory import make_sampler, analyze, SamplerSpec, BlockSpec
from .summary import Summary, summarize

# The subpackages themselves, so ``import mimcs`` makes ``mimcs.hmc.NUTS`` reachable --- the access
# path this namespace deliberately pushes people towards. ``mimcs.testing`` is *not* imported: it is
# the test harness rather than a runtime dependency, and it pulls in matplotlib.
from . import adaptation, diagnostics, dsl, factory, hmc, model, optim, pt, rng, samplers, summary

__all__ = [
    "__version__",
    # models
    "Model",
    "compile_model",
    "DslError",
    # samplers
    "make_sampler",
    "analyze",
    "SamplerSpec",
    "BlockSpec",
    "make_sampler_class",
    # evaluation
    "summarize",
    "Summary",
    # logging
    "configure_logging",
    "get_logger",
    "set_log_level",
    "log_level",
]
