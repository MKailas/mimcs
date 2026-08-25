"""Integrators over the product space, and the energy-error budget tempering costs them.

Implements the integrator section of ``docs/design/13_parallel_tempering.md``.

Nothing about within-orbit adaptivity is specific to one target, so **WALNUTS needs no product
variant**: :class:`~mimcs.hmc.line_search.LineSearchIntegrator` refines against
``total_energy(istate, potentials, kinetics, ctx)``, which over the product space is already the
sum over temperatures --- every tempered potential and every :class:`~mimcs.pt.ProductKinetic`
returns its own sum across rungs. So the line search sees one scalar Hamiltonian, exactly as it
does for an ordinary target, and refines the *whole product step* until that scalar is within
budget. This mirrors NUTS, whose U-turn becomes the product-space one for the same reason (doc 13).

**The budget must scale with K, and that is the one thing that does not come for free.** A
threshold ``delta`` calibrated for a single chain is a statement about one Hamiltonian's
discretization error. The product Hamiltonian is a sum of K of them, and K independent errors of
typical size ``delta`` sum to something of order ``K * delta``, so holding the sum to ``delta``
silently demands each rung be ~K times more accurate than intended --- the line search would
refine to a needlessly fine level on every step and pay for it in gradients. Thresholds here are
therefore stated **per temperature** and scaled by K, by :func:`product_error_thresholds` --- the
one place that rule and the default it falls back to are written down. The factory's ``pt_`` bases
build the same integrator by a different route and call the same function, so the two cannot
disagree about what a tempered budget means.

The alternative reading --- refine until the *worst* rung is within ``delta``, i.e. compare
``max_k |dH_k|`` against an unscaled threshold --- is the better criterion in principle, since a
sum lets a badly-behaved hot rung hide behind K-1 well-behaved ones. It needs per-temperature
energies threaded through the line search rather than the scalar the integrator interface passes
around, so it is deliberately not what this does; the sum-vs-K*delta form reuses the existing
integrator unchanged, which is what makes PT-WALNUTS free.
"""

from __future__ import annotations

from .._logging import get_logger
from ..hmc.integrators import leapfrog
from ..hmc.line_search import (
    DEFAULT_ERROR_THRESHOLDS, LineSearchIntegrator, MarkovianLineSearchIntegrator)

log = get_logger(__name__)

#: Integrators whose ``error_thresholds`` is a per-Hamiltonian accuracy budget, and so must be
#: restated over the product space. Keyed by the names ``SamplerSpec.integrator`` uses, so the
#: factory's tempered path can ask this module rather than keep its own copy of the rule.
BUDGETED_INTEGRATORS = frozenset({"line_search", "markovian_line_search"})


def product_error_thresholds(error_thresholds=None, n_temperatures: int = 1):
    """The product-space energy-error budget for a **per-temperature** threshold schedule.

    ``error_thresholds`` is a scalar or one value per level, stated for a single rung, and
    ``None`` means the untempered default. The result is that schedule scaled by K --- the one
    place the sum-vs-``K * delta`` reading of the budget (see the module docstring) is written
    down, for both :func:`product_line_search` and the factory's ``pt_`` bases.
    """
    thr = DEFAULT_ERROR_THRESHOLDS if error_thresholds is None else error_thresholds
    K = int(n_temperatures)
    try:
        return [float(t) * K for t in thr]
    except TypeError:
        return float(thr) * K


def product_leapfrog(potentials, kinetics, n_temperatures: int):
    """The default: ordinary leapfrog over the product components."""
    return leapfrog(potentials, kinetics)


def product_line_search(*, schedule=None, error_thresholds=None, markovian: bool = False,
                        base=None, **kwargs):
    """A WALNUTS line search over the product space, with the budget scaled to K rungs.

    Args:
        schedule / error_thresholds: as :class:`~mimcs.hmc.line_search.LineSearchIntegrator`, except
            that ``error_thresholds`` is stated **per temperature** and scaled by ``K`` here, by
            :func:`product_error_thresholds` (see the module docstring). A scalar, one value per
            level, or ``None`` for the untempered default.
        markovian: use the randomized :class:`~mimcs.hmc.line_search.MarkovianLineSearchIntegrator`.
        base: builder ``(potentials, kinetics, K) -> integrator`` for the integrator the line search
            refines; defaults to product leapfrog.

    Returns the builder ``(potentials, kinetics, K) -> integrator`` that
    :func:`~mimcs.pt.parallel_tempering` expects for its ``integrator`` argument.
    """
    base_builder = base if base is not None else product_leapfrog

    def build(potentials, kinetics, n_temperatures: int):
        K = int(n_temperatures)
        scaled = product_error_thresholds(error_thresholds, K)
        cls = MarkovianLineSearchIntegrator if markovian else LineSearchIntegrator
        log.info("product line search over %d temperature(s): energy-error budget %s "
                 "(per temperature x K)", K, scaled)
        return cls(base_builder(potentials, kinetics, K), potentials, kinetics,
                   schedule=schedule, error_thresholds=scaled, **kwargs)

    return build
