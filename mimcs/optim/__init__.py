"""General-purpose optimisation utilities (JAX).

Minimisers over pytree parameters, sharing one :class:`OptimizeResult`:

* :func:`minimize` --- limited-memory BFGS for a scalar objective. Used *online* by
  :class:`~mimcs.adaptation.ClassifierTermination`, which fits a logistic regression during the
  warmup of every default sampler, and offline as the control arm of the metric regression.
* :func:`separable_newton` --- a modified Newton for an objective that is a **sum of independent
  low-dimensional problems** sharing one array of data, each solved with its own step, step length
  and convergence test. This is what the sampler factory's metric regression uses.
  :func:`newton_minimize` is its one-problem case, a drop-in alternative to :func:`minimize` when
  a full Hessian is affordable.
* :func:`lbfgs_newton` --- the same lane-separable problems with a **large** shared block: L-BFGS on
  the shared parameters, separable Newton on the lanes (interleaved or profiled). Not used by
  default; the metric regression routes to it only when ``regression.SHARED_SOLVER`` is set.
"""

from ._common import OptimizeResult
from .lbfgs import minimize
from .newton import separable_newton, newton_minimize
from .shared_block import lbfgs_newton

__all__ = ["minimize", "separable_newton", "newton_minimize", "lbfgs_newton", "OptimizeResult"]
