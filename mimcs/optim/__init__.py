"""General-purpose optimisation utilities (JAX).

A limited-memory BFGS minimiser (:func:`minimize`). Used by the sampler factory's metric
regression offline, and *online* by :class:`~mimcs.adaptation.ClassifierTermination`, which fits a
logistic regression during the warmup of every default sampler.
"""

from .lbfgs import minimize, OptimizeResult

__all__ = ["minimize", "OptimizeResult"]
