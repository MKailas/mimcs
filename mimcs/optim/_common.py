"""Pieces shared by the minimisers in this package.

The result type and the "how did it end" reporting are the same question whichever solver asked
it, so they live here rather than in whichever module happened to need them first
(:mod:`mimcs.optim.lbfgs` did). Everything is re-exported from :mod:`mimcs.optim`.
"""

from __future__ import annotations

import logging
import math
from typing import NamedTuple

from jax import Array

from .._logging import get_logger

_log = get_logger(__name__)


class OptimizeResult(NamedTuple):
    """Outcome of a minimisation.

    Attributes:
        x: the minimiser, in the same pytree structure as ``x0``.
        fun: objective value at ``x``.
        grad_norm: max-norm of the gradient at ``x`` (over *every* parameter, so for a
            lane-separable solve this is the worst lane's).
        n_iter: number of outer iterations taken.
        converged: whether ``grad_norm`` fell below the tolerance --- for a lane-separable solve,
            whether **every** lane did.
        lane_converged: per-lane convergence flags for a separable solve (see
            :func:`mimcs.optim.separable_newton`), else ``None``. A trailing field with a default
            so every existing caller and every positional construction stays valid.
    """

    x: object
    fun: Array
    grad_norm: Array
    n_iter: Array
    converged: Array
    lane_converged: object = None


def host_float(x):
    """``float(x)``, or ``None`` when ``x`` is a tracer (the minimiser was called under ``jit``).

    Every minimiser here is a ``lax.while_loop``, so its outcome is only inspectable on the host
    when the caller is not itself tracing; there is nothing to report in the traced case.
    """
    try:
        return float(x)
    except Exception:                       # TracerArrayConversionError / ConcretizationTypeError
        return None


def log_outcome(res: OptimizeResult, max_iter: int, gtol: float, warn: bool, *,
                solver: str = "L-BFGS", detail: str = "", logger=None) -> None:
    """Report how a minimisation ended: DEBUG always, WARNING when it ran out of iterations.

    ``detail`` is appended to the max-iter message by solvers with more to say about *what* failed
    to converge (a separable solve names the lanes). ``logger`` is the *caller's* logger, so a
    record keeps the name of the module that actually ran the solve (``mimcs.optim.lbfgs`` /
    ``mimcs.optim.newton``) rather than this shared one.
    """
    log = logger if logger is not None else _log
    n_iter, gnorm, f = host_float(res.n_iter), host_float(res.grad_norm), host_float(res.fun)
    if n_iter is None:                      # traced: no concrete outcome to report
        log.debug("%s traced under jit; termination not reported", solver)
        return
    converged = gnorm < gtol
    if not math.isfinite(f):
        log.warning("%s stopped on a non-finite objective (f=%g) after %d iteration(s); "
                    "the returned point is not a minimiser", solver, f, int(n_iter))
    elif int(n_iter) >= max_iter and not converged:
        log.log(logging.WARNING if warn else logging.DEBUG,
                "%s hit max_iter=%d without converging: gradient max-norm %.3g still "
                "above gtol=%.3g (f=%.6g).%s The fit is the last iterate, not a minimiser.",
                solver, max_iter, gnorm, gtol, f, f" {detail}" if detail else "")
    log.debug("%s terminated after %d/%d iteration(s): f=%.6g, grad max-norm=%.3g, "
              "converged=%s", solver, int(n_iter), max_iter, f, gnorm, converged)
