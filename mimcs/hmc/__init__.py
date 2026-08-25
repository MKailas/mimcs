"""Hamiltonian Monte Carlo: components, integrators and samplers.

Three orthogonal axes (``docs/design/06_hamiltonian_monte_carlo.md``), each varying independently
of the others:

1. **Hamiltonian components** --- potentials (:class:`ModelPotential` per log-density component,
   :class:`JacobianPotential` for the chart correction) and kinetics
   (:class:`DiagonalQuadraticKinetic`, :class:`DenseQuadraticKinetic`,
   :class:`LowRankQuadraticKinetic`, the position-dependent block metrics of
   :mod:`mimcs.hmc.block_riemannian`, and the experimental
   :class:`~mimcs.hmc.RelativisticKinetic` / :class:`~mimcs.hmc.RiemannianKinetic`).
2. **Integrators** --- :func:`leapfrog`, the RESPA :func:`multirate_leapfrog`, and the WALNUTS
   line searches (:class:`LineSearchIntegrator`, :class:`MarkovianLineSearchIntegrator`), which
   wrap *any* base integrator.
3. **Large-scale structure** --- :class:`HMC`, :class:`RandomizedHMC`, :class:`NUTS` (with
   :class:`SimpleNUTS` as the reference oracle), all subclassing :class:`BaseHMC`.

Any combination of the three works: a sampler consumes only ``integrator.step``, ``total_energy``,
``velocity``, ``refresh_momentum`` and ``init_integrator_state``.
"""

from .state import IntegratorState, HamiltonianContext
from .hamiltonians import (
    Hamiltonian, PotentialHamiltonian, ModelPotential, JacobianPotential,
    KineticHamiltonian, DiagonalQuadraticKinetic, DenseQuadraticKinetic,
    LowRankQuadraticKinetic, total_energy)
from .integrators import (
    Op, SplittingIntegrator, RepeatedIntegrator, leapfrog, multirate_leapfrog,
    init_integrator_state)
from .line_search import (
    LineSearchIntegrator, MarkovianLineSearchIntegrator, doubling_schedule,
    DEFAULT_ERROR_THRESHOLDS)
from .samplers import (
    HMCState, BaseHMC, HMC, RandomizedHMC, default_potentials, split_potentials, make_kinetic)
from .nuts import BaseNUTS, NUTS, NUTSTree
from .simple_nuts import SimpleNUTS
from .solvers import FixedPointSolver, PicardSolver, AndersonSolver
from .riemannian import (
    Metric, AnalyticMetric, RiemannianKinetic, RMHMC)
from .relativistic import RelativisticKinetic
from .block_riemannian import (
    BlockMetric, DiagonalBlock, LearnedDiagonalBlock,
    ShapedLearnedBlock, build_block, build_blocks)
from .metric_expr import MetricExpr, Exp, Sigmoid, SpExp, SpSigmoid, Sum, Product

__all__ = [
    "IntegratorState", "HamiltonianContext",
    "Hamiltonian", "PotentialHamiltonian", "ModelPotential", "JacobianPotential",
    "KineticHamiltonian", "DiagonalQuadraticKinetic", "DenseQuadraticKinetic",
    "LowRankQuadraticKinetic", "total_energy",
    "Op", "SplittingIntegrator", "RepeatedIntegrator", "leapfrog",
    "multirate_leapfrog", "init_integrator_state",
    "LineSearchIntegrator", "MarkovianLineSearchIntegrator", "doubling_schedule",
    "DEFAULT_ERROR_THRESHOLDS",
    "HMCState", "BaseHMC", "HMC", "RandomizedHMC",
    "default_potentials", "split_potentials", "make_kinetic",
    "BaseNUTS", "NUTS", "SimpleNUTS", "NUTSTree",
    "Metric", "AnalyticMetric", "RiemannianKinetic", "RMHMC", "RelativisticKinetic",
    "FixedPointSolver", "PicardSolver", "AndersonSolver",
    "BlockMetric", "DiagonalBlock", "LearnedDiagonalBlock",
    "ShapedLearnedBlock", "build_block", "build_blocks",
    "MetricExpr", "Exp", "Sigmoid", "SpExp", "SpSigmoid", "Sum", "Product",
]
