"""Parallel tempering: a generic answer to multimodality (``docs/design/13_parallel_tempering.md``).

Everything else here improves how well a chain explores *one* mode. Parallel tempering runs a
ladder of flattened copies of the target, lets the hot ones cross the barriers, and swaps that
mobility down to the cold chain whose draws are kept::

    from mimcs.pt import parallel_tempering

    sampler = parallel_tempering(model, n_temperatures=6, beta_min=0.01)
    sampler.warmup(2000)
    draws = sampler.sample(4000)        # the beta = 1 chain, keyed by parameter name

`beta_min` is a starting point, not a setting: **the ladder adapts by default**
(:class:`LadderAdaptation`, ``adapt_ladder=True``), driving each adjacent pair's swap acceptance
toward 0.234. A hand-set ladder cannot be right across models --- the usable temperature range
narrows as the data grows --- which :mod:`mimcs.pt.ladder` documents at length. Reachable from the
sampler factory as a ``pt_`` base (``spec.base = "pt_nuts"``), and WALNUTS works over the product
space unchanged via :func:`product_line_search`.

The K log-densities and gradients of a step are independent, so they are one `vmap` over a
leading temperature axis --- which is the whole cost of a PT step, and why it is worth running on
a GPU.
"""

from .tempering import (
    geometric_ladder, ProductModel, TemperedProductPotential, build_tempered_potentials)
from .kinetics import ProductKinetic, build_product_kinetics
from .product import ProductSpaceMixin
from .adaptation import PerTemperatureAdaptation
from .hmc import IndependentAcceptanceMixin
from .nuts import PerTemperatureNUTSMixin, PerTemperatureSimpleNUTSMixin
from .ladder import LadderAdaptation, betas_from_rho, rho_from_betas
from .swaps import swap_step, swap_log_ratios, apply_swaps
from .integrators import (
    BUDGETED_INTEGRATORS, product_error_thresholds, product_leapfrog, product_line_search)
from .sampler import parallel_tempering, ReplicaExchangeMixin

__all__ = [
    "parallel_tempering", "ReplicaExchangeMixin",
    "geometric_ladder", "ProductModel", "TemperedProductPotential",
    "build_tempered_potentials",
    "ProductKinetic", "build_product_kinetics",
    "ProductSpaceMixin", "PerTemperatureAdaptation", "IndependentAcceptanceMixin",
    "PerTemperatureNUTSMixin", "PerTemperatureSimpleNUTSMixin",
    "swap_step", "swap_log_ratios", "apply_swaps",
    "product_leapfrog", "product_line_search", "product_error_thresholds",
    "BUDGETED_INTEGRATORS",
    "LadderAdaptation", "betas_from_rho", "rho_from_betas",
]
