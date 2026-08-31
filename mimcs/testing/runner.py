"""High-level test harness: run samplers on a problem, compare, plot, report.

:func:`evaluate` is the main entry point. Given a :class:`TargetProblem` and one or
more sampler builders, it:

1. runs each sampler (warmup + sampling) and records draws, acceptance, and ESS;
2. draws exact reference samples if the problem provides them;
3. compares every sampler to the reference (analytic check) and all sampler pairs to
   each other (algorithm-vs-algorithm check);
4. writes trace and pair plots (with the reference overlaid) for manual inspection;
5. returns a :class:`Report` whose :meth:`~Report.assert_correct` raises a readable
   ``AssertionError`` if any non-skipped check fails.

A sampler *builder* is a callable ``build(model, seed) -> sampler`` returning an
object with ``.warmup(n)``, ``.sample(n)``, and ``.acceptance_rate()``.
There is one for every sampler in the library --- :func:`adaptive_mh`, :func:`hmc`,
:func:`randomized_hmc`, :func:`nuts`, :func:`simple_nuts`, the Riemannian and relativistic
variants, the WALNUTS ones, :func:`multirate_nuts` and the block samplers --- each a closure over
its configuration that takes ``seed`` at call time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .comparison import ComparisonResult, Thresholds, compare
from ..diagnostics import ess
from .problems import TargetProblem
from . import plots

Builder = Callable[[object, int], object]


@dataclass
class SamplerOutput:
    name: str
    samples: np.ndarray
    accept_rate: float
    ess: np.ndarray
    warmup_step_sizes: np.ndarray | None = None
    warmup_log_clip: np.ndarray | None = None
    grad_evals: float | None = None       # total gradient evaluations over sampling (cost proxy)
    mean_n_leaves: float | None = None    # mean trajectory size (NUTS)


@dataclass
class Report:
    problem: TargetProblem
    outputs: dict
    reference: np.ndarray | None
    comparisons: list = field(default_factory=list)
    out_dir: str | None = None

    def summary(self) -> str:
        lines = [f"=== Report: {self.problem.name} (dim={self.problem.dim},"
                 f" hard={self.problem.hard}) ==="]
        for name, out in self.outputs.items():
            line = (f"  sampler '{name}': accept={out.accept_rate:.3f}  "
                    f"ESS={np.array2string(out.ess.astype(int))}  n={len(out.samples)}")
            if out.grad_evals:   # efficiency: min-ESS per 1k gradient evaluations (the cost proxy)
                eff = 1000.0 * float(np.min(out.ess)) / out.grad_evals
                line += f"  grad_evals={int(out.grad_evals)}  ESS/1k-grad={eff:.2f}"
            lines.append(line)
        if self.out_dir:
            lines.append(f"  plots: {self.out_dir}")
        lines.append("")
        for c in self.comparisons:
            lines.append(c.summary())
            lines.append("")
        return "\n".join(lines)

    def assert_correct(self, *, skip_hard: bool = True):
        """Raise ``AssertionError`` with the full report if any check failed.

        Comparisons on ``hard`` problems are skipped by default (random-walk samplers
        are not expected to pass them); the run and its plots are still produced.
        """
        if skip_hard and self.problem.hard:
            return self
        failures = [c for c in self.comparisons if not c.ok]
        if failures:
            failed_names = ", ".join(c.label for c in failures)
            raise AssertionError(
                f"MCMC correctness checks FAILED for [{failed_names}]\n\n"
                + self.summary())
        return self


def draw_samples(sampler, n_warmup: int, n_samples: int) -> np.ndarray:
    """Warm up, sample, and return one flat ``(n_samples, ambient_dim)`` array.

    The harness compares draws against a reference sampler's, so it works in the flat
    layout throughout --- this is the single place that converts.
    """
    sampler.warmup(n_warmup)
    sampler.sample(n_samples)
    return sampler.get_samples_flat()


def evaluate(
    problem: TargetProblem,
    samplers: dict[str, Builder],
    *,
    n_warmup: int = 4000,
    n_samples: int = 20000,
    seed: int = 0,
    n_reference: int | None = None,
    out_dir: str | None = None,
    thresholds: Thresholds | None = None,
    make_plots: bool = True,
) -> Report:
    rng_seeds = {name: seed + 1 + i for i, name in enumerate(samplers)}

    outputs: dict[str, SamplerOutput] = {}
    for name, build in samplers.items():
        sampler = build(problem.model, rng_seeds[name])
        samples = draw_samples(sampler, n_warmup, n_samples)
        log_clip = (sampler.warmup_log_clip()
                    if hasattr(sampler, "warmup_log_clip") else None)
        ge = sampler.total_grad_evals() if hasattr(sampler, "total_grad_evals") else None
        nl = sampler.mean_n_leaves() if hasattr(sampler, "mean_n_leaves") else None
        outputs[name] = SamplerOutput(
            name=name, samples=samples,
            accept_rate=float(sampler.acceptance_rate()), ess=ess(samples),
            warmup_step_sizes=sampler.warmup_step_sizes(), warmup_log_clip=log_clip,
            grad_evals=ge, mean_n_leaves=(nl if nl is None or np.isfinite(nl) else None))

    reference = None
    if problem.has_reference:
        n_ref = n_reference if n_reference is not None else max(n_samples, 50_000)
        reference = problem.exact_sample(n_ref, seed=seed + 999)

    comparisons: list[ComparisonResult] = []
    # Each sampler vs the analytic reference.
    if reference is not None:
        for name, out in outputs.items():
            comparisons.append(compare(
                out.samples, reference, a_name=name, b_name="analytic",
                label=f"{name} vs analytic", b_exact=True,
                thresholds=thresholds, energy_seed=seed))
    # All sampler pairs vs each other.
    names = list(outputs)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            comparisons.append(compare(
                outputs[a].samples, outputs[b].samples, a_name=a, b_name=b,
                label=f"{a} vs {b}", thresholds=thresholds, energy_seed=seed))

    if make_plots and out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        for name, out in outputs.items():
            plots.trace_plot(out.samples, problem.labels,
                             os.path.join(out_dir, f"trace_{name}.png"),
                             title=f"{problem.name}: {name} trace")
            plots.pair_plot(out.samples, problem.labels,
                            os.path.join(out_dir, f"pairs_{name}.png"),
                            reference=reference,
                            title=f"{problem.name}: {name} vs reference")
            plots.step_size_plot(out.warmup_step_sizes,
                                 os.path.join(out_dir, f"step_size_{name}.png"),
                                 title=f"{problem.name}: {name} warmup step size")
            if out.warmup_log_clip is not None and len(out.warmup_log_clip):
                plots.clip_threshold_plot(
                    out.warmup_log_clip,
                    os.path.join(out_dir, f"clip_threshold_{name}.png"),
                    title=f"{problem.name}: {name} score-mass clip threshold")

    return Report(problem=problem, outputs=outputs, reference=reference,
                  comparisons=comparisons, out_dir=out_dir)


def adaptive_mh(*, init=None, step_size: float = 0.5, target_accept: float = 0.3,
                cov_min_samples: int = 50, **kwargs) -> Builder:
    """Convenience builder for adaptive random-walk Metropolis--Hastings."""
    from ..samplers import RandomWalkMH, make_sampler_class
    from ..adaptation import RobbinsMonroStepSize, DiagonalCovarianceAdaptation

    Cls = make_sampler_class(
        RobbinsMonroStepSize, DiagonalCovarianceAdaptation, RandomWalkMH)

    def build(model, seed):
        init_position = init if init is not None else model.default_sample()
        return Cls(model, init_position=init_position, seed=seed,
                   step_size=step_size, target_accept=target_accept,
                   cov_min_samples=cov_min_samples, **kwargs)

    return build


def _adaptation_mixins(mass_adapt: str, metric: str, center,
                       unit_vector_center: bool = False, terminate=False) -> tuple:
    """Assemble the adaptation mixins (highest MRO priority first). The recharting mixins
    sit just above the base so they rechart *before* the mass adaptation reads the score.

    ``center``: ``False`` (none), ``True`` / ``"mean"`` (mean-std ``CenteringAdaptation``), or
    ``"robust"`` (median-MAD ``RobustCenteringAdaptation``).
    ``unit_vector_center``: fit the stereographic charts of ``adaptive=True`` unit vectors.
    ``terminate``: end warmup on a mixing criterion --- ``"rhat"`` or ``"classifier"``. Goes
    first: it only observes, so it reads the draw after every adaptation has had its say."""
    from ..adaptation import (
        RobbinsMonroStepSize, MassMatrixAdaptation, ScoreMassAdaptation,
        CenteringAdaptation, RobustCenteringAdaptation, UnitVectorCenteringAdaptation,
        GelmanRubinTermination, ClassifierTermination)
    if mass_adapt == "covariance":
        mass = MassMatrixAdaptation
    elif mass_adapt == "score":
        mass = ScoreMassAdaptation           # handles both diagonal and dense masses
    else:
        raise ValueError(f"unknown mass_adapt {mass_adapt!r} (use 'covariance' or 'score')")
    mixins = []
    if terminate == "rhat":
        mixins.append(GelmanRubinTermination)
    elif terminate == "classifier":
        mixins.append(ClassifierTermination)
    elif terminate:
        raise ValueError(f"unknown terminate {terminate!r} (use 'rhat' or 'classifier')")
    mixins += [RobbinsMonroStepSize, mass]
    if center == "robust":
        mixins.append(RobustCenteringAdaptation)
    elif center:                             # True or "mean"
        mixins.append(CenteringAdaptation)
    if unit_vector_center:
        mixins.append(UnitVectorCenteringAdaptation)
    return tuple(mixins)


def hmc(*, init=None, n_leapfrog: int = 20, step_size: float = 0.5,
        metric: str = "diagonal", target_accept: float = 0.8,
        mass_min_samples: int = 50, mass_adapt: str = "covariance",
        center=False, unit_vector_center=False, terminate=False, **kwargs) -> Builder:
    """Convenience builder for adaptive HMC (Robbins--Monro step size + mass matrix).

    ``mass_adapt`` selects the mass adaptation: ``"covariance"`` (empirical target
    covariance, default) or ``"score"`` (mass fit to the score covariance by KL-SGD, either
    ``metric``). ``center`` adds a centering reparametrization for ``centered=True`` parameters
    (pairs naturally with ``mass_adapt='score'``): ``True`` uses the mean/std
    :class:`mimcs.adaptation.CenteringAdaptation`, ``"robust"`` the heavy-tail-robust median/MAD
    :class:`mimcs.adaptation.RobustCenteringAdaptation`. ``unit_vector_center`` fits the
    stereographic charts of ``adaptive=True`` unit vectors
    (:class:`mimcs.adaptation.UnitVectorCenteringAdaptation`)."""
    from ..samplers import make_sampler_class
    from ..hmc import HMC

    Cls = make_sampler_class(
        *_adaptation_mixins(mass_adapt, metric, center, unit_vector_center, terminate), HMC)

    def build(model, seed):
        init_position = init if init is not None else model.default_sample()
        return Cls(model, init_position=init_position, seed=seed,
                   n_leapfrog=n_leapfrog, step_size=step_size, metric=metric,
                   target_accept=target_accept, mass_min_samples=mass_min_samples,
                   **kwargs)

    return build


def nuts(*, init=None, max_tree_depth: int = 10, step_size: float = 0.5,
         metric: str = "diagonal", target_accept: float = 0.8,
         mass_min_samples: int = 50, mass_adapt: str = "covariance",
         center=False, unit_vector_center=False, terminate=False,
         extra_mixins: tuple = (), **kwargs) -> Builder:
    """Convenience builder for adaptive NUTS (Robbins--Monro step size + mass matrix).

    ``mass_adapt`` selects the mass adaptation: ``"covariance"`` (default) or ``"score"``
    (mass fit to the score covariance by KL-SGD, either ``metric``). ``center`` adds a centering
    reparametrization: ``True`` the mean/std :class:`mimcs.adaptation.CenteringAdaptation`,
    ``"robust"`` the median/MAD :class:`mimcs.adaptation.RobustCenteringAdaptation`.
    ``unit_vector_center`` fits the stereographic charts of ``adaptive=True`` unit vectors
    (:class:`mimcs.adaptation.UnitVectorCenteringAdaptation`). ``extra_mixins`` are placed
    **after** the adaptation mixins and before ``NUTS``, which is where a kernel-composing mixin
    such as :class:`~mimcs.samplers.DiscreteMetropolisWithinGibbs` belongs."""
    from ..samplers import make_sampler_class
    from ..hmc import NUTS

    Cls = make_sampler_class(
        *_adaptation_mixins(mass_adapt, metric, center, unit_vector_center, terminate),
        *extra_mixins, NUTS)

    def build(model, seed):
        init_position = init if init is not None else model.default_sample()
        return Cls(model, init_position=init_position, seed=seed,
                   max_tree_depth=max_tree_depth, step_size=step_size, metric=metric,
                   target_accept=target_accept, mass_min_samples=mass_min_samples,
                   **kwargs)

    return build


def nuts_gibbs(*, discrete_sweeps: int = 1, adapt_discrete: bool = False, **kwargs) -> Builder:
    """NUTS for the continuous block plus a Metropolis-within-Gibbs sweep for the discrete one.

    :func:`nuts` with :class:`~mimcs.samplers.DiscreteMetropolisWithinGibbs` mixed in ahead of the
    base algorithm; it takes every :func:`nuts` keyword. The mixin is inert on a model with no
    discrete parameters --- it adds no RNG draw components and does no work --- so this is a safe
    drop-in anywhere ``nuts`` is used, and an A/B against ``nuts`` on a continuous problem is
    bit-identical.

    ``adapt_discrete`` adds :class:`~mimcs.adaptation.DiscreteMarginalAdaptation`, which learns
    each coordinate's marginal pmf during warmup and proposes from it instead of uniformly. Off by
    default so ``nuts_gibbs`` remains the unadapted baseline an A/B is measured against.
    """
    from ..adaptation import DiscreteMarginalAdaptation
    from ..samplers import DiscreteMetropolisWithinGibbs
    extra = ((DiscreteMarginalAdaptation, DiscreteMetropolisWithinGibbs) if adapt_discrete
             else (DiscreteMetropolisWithinGibbs,))
    return nuts(extra_mixins=extra, discrete_sweeps=discrete_sweeps, **kwargs)


def rmhmc(*, metric, init=None, n_leapfrog: int = 20, n_fixed_point: int = 8,
          solver=None, step_size: float = 0.5, target_accept: float = 0.8,
          **kwargs) -> Builder:
    """Builder for implicit Riemannian HMC with a fixed (analytic) metric. **[experimental]**

    ``metric`` is a :class:`~mimcs.hmc.Metric` or a callable ``q -> SPD matrix``.
    ``solver`` selects the implicit-step fixed-point solver: ``None`` (Picard),
    ``"anderson"``, or a :class:`~mimcs.hmc.FixedPointSolver` object.
    """
    from ..samplers import make_sampler_class
    from ..adaptation import RobbinsMonroStepSize
    from ..hmc import RMHMC, Metric, AnalyticMetric
    from ..hmc.solvers import resolve_solver

    metric_obj = metric if isinstance(metric, Metric) else AnalyticMetric(metric)
    solver_obj = resolve_solver(solver, n_fixed_point)
    Cls = make_sampler_class(RobbinsMonroStepSize, RMHMC)

    def build(model, seed):
        init_position = init if init is not None else model.default_sample()
        return Cls(model, init_position=init_position, seed=seed, metric=metric_obj,
                   n_leapfrog=n_leapfrog, n_fixed_point=n_fixed_point, solver=solver_obj,
                   step_size=step_size, target_accept=target_accept, **kwargs)

    return build


def explicit_rmhmc(*, metrics=None, init=None, n_leapfrog: int = 20,
                   step_size: float = 0.5, target_accept: float = 0.8,
                   **kwargs) -> Builder:
    """Builder for explicit (block) Riemannian HMC with diagonal metrics.

    ``metrics`` is ``{block_name: BlockMetric(...) | MetricExpr}`` --- a given diagonal
    metric or a mass-matrix mini-language expression (``Exp("v") + Exp()``, ...). Learned
    blocks are adapted online by
    :class:`mimcs.adaptation.MetricAdaptation` (SGD on a KL objective); its knobs
    ``metric_adapt_kappa`` (0.75), ``metric_adapt_n0`` (5), ``metric_clip_frac`` (0.1) pass
    through ``kwargs``. With only given metrics the mixin is a no-op.
    """
    from ..samplers import make_sampler_class
    from ..adaptation import RobbinsMonroStepSize, MetricAdaptation
    from ..hmc import HMC, build_blocks, default_potentials, leapfrog

    Cls = make_sampler_class(RobbinsMonroStepSize, MetricAdaptation, HMC)

    def build(model, seed):
        init_position = init if init is not None else model.default_sample()
        blocks = build_blocks(model, metrics)
        potentials = default_potentials(model)
        integrator = leapfrog(potentials, blocks)
        return Cls(model, init_position=init_position, seed=seed, kinetics=blocks,
                   potentials=potentials, integrator=integrator, n_leapfrog=n_leapfrog,
                   step_size=step_size, target_accept=target_accept, **kwargs)

    return build


def explicit_rmnuts(*, metrics=None, init=None, max_tree_depth: int = 10,
                    step_size: float = 0.5, target_accept: float = 0.8,
                    **kwargs) -> Builder:
    """Builder for explicit (block) Riemannian NUTS: NUTS over a block-diagonal metric.

    Now just NUTS with the list of block kinetics (one per parameter) --- each block's
    ``velocity = M_i(q_{-i})^{-1} p_i`` drives the generalized U-turn and the unified
    ``leapfrog`` composes the closed-form block flows (no implicit solve). ``metrics`` is
    ``{block_name: BlockMetric(...) | MetricExpr}`` (mini-language, e.g. ``Exp("v") + Exp()``);
    learned blocks are adapted online by
    :class:`mimcs.adaptation.MetricAdaptation`.
    """
    from ..samplers import make_sampler_class
    from ..adaptation import RobbinsMonroStepSize, MetricAdaptation
    from ..hmc import NUTS, build_blocks, default_potentials, leapfrog

    Cls = make_sampler_class(RobbinsMonroStepSize, MetricAdaptation, NUTS)

    def build(model, seed):
        init_position = init if init is not None else model.default_sample()
        blocks = build_blocks(model, metrics)
        potentials = default_potentials(model)
        integrator = leapfrog(potentials, blocks)
        return Cls(model, init_position=init_position, seed=seed, kinetics=blocks,
                   potentials=potentials, integrator=integrator, max_tree_depth=max_tree_depth,
                   step_size=step_size, target_accept=target_accept, **kwargs)

    return build


def _relativistic_build(SamplerCls, model, seed, *, shape, inner_axes, mass, light_speed,
                        init, **kwargs):
    from ..hmc import RelativisticKinetic, leapfrog, default_potentials
    shape = tuple(shape) if shape is not None else (model.coord_dim,)
    kinetic = RelativisticKinetic(shape, inner_axes=inner_axes, mass=mass,
                                  light_speed=light_speed)
    if kinetic.dim != model.coord_dim:
        raise ValueError(
            f"relativistic shape {shape} (dim {kinetic.dim}) != model coord_dim "
            f"{model.coord_dim}")
    potentials = default_potentials(model)
    integrator = leapfrog(potentials, kinetic)
    init_position = init if init is not None else model.default_sample()
    return SamplerCls(model, init_position=init_position, seed=seed, kinetic=kinetic,
                      potentials=potentials, integrator=integrator, **kwargs)


def _relativistic_mixins(mass_adapt: bool, center: bool) -> tuple:
    """Adaptation mixins for the relativistic samplers. ``CenteringAdaptation`` sits just
    above the base so it recharts before the mass adaptation reads the score."""
    from ..adaptation import (
        RobbinsMonroStepSize, RelativisticMassAdaptation, CenteringAdaptation)
    mixins = [RobbinsMonroStepSize]
    if mass_adapt:
        mixins.append(RelativisticMassAdaptation)
    if center:
        mixins.append(CenteringAdaptation)
    return tuple(mixins)


def relativistic_hmc(*, shape=None, inner_axes=(), mass: float = 1.0,
                     light_speed: float = 1.0, mass_adapt: bool = False,
                     center: bool = False, init=None, n_leapfrog: int = 20,
                     step_size: float = 0.5, target_accept: float = 0.8, **kwargs) -> Builder:
    """Builder for relativistic HMC: HMC with the relativistic (bounded-velocity) kinetic.
    **[experimental]** --- see :mod:`mimcs.hmc.relativistic` on the fixed light speed.

    ``shape`` reshapes the flat coordinate momentum (default ``(coord_dim,)``) and
    ``inner_axes`` are summed inside each particle's square root (``()`` = every coordinate
    its own 1-D particle). ``mass_adapt=True`` adapts each particle's rest mass to the score
    covariance (:class:`mimcs.adaptation.RelativisticMassAdaptation`); ``center=True`` adds
    the centering reparametrization. ``light_speed`` is a fixed scalar --- which assumes centered coordinates, hence the
    experimental marker."""
    from ..samplers import make_sampler_class
    from ..hmc import HMC

    Cls = make_sampler_class(*_relativistic_mixins(mass_adapt, center), HMC)

    def build(model, seed):
        return _relativistic_build(
            Cls, model, seed, shape=shape, inner_axes=inner_axes, mass=mass,
            light_speed=light_speed, init=init, n_leapfrog=n_leapfrog, step_size=step_size,
            target_accept=target_accept, **kwargs)

    return build


def relativistic_nuts(*, shape=None, inner_axes=(), mass: float = 1.0,
                      light_speed: float = 1.0, mass_adapt: bool = False,
                      center: bool = False, init=None, max_tree_depth: int = 10,
                      step_size: float = 0.5, target_accept: float = 0.8,
                      **kwargs) -> Builder:
    """Builder for relativistic NUTS **[experimental]**: NUTS with the relativistic kinetic (its bounded
    ``velocity = c^2 p / T`` drives the U-turn). See :func:`relativistic_hmc` for args."""
    from ..samplers import make_sampler_class
    from ..hmc import NUTS

    Cls = make_sampler_class(*_relativistic_mixins(mass_adapt, center), NUTS)

    def build(model, seed):
        return _relativistic_build(
            Cls, model, seed, shape=shape, inner_axes=inner_axes, mass=mass,
            light_speed=light_speed, init=init, max_tree_depth=max_tree_depth,
            step_size=step_size, target_accept=target_accept, **kwargs)

    return build


def _multirate_integrator(model, potentials, kinetic, *, n, cheap=None):
    """The multi-rate (RESPA) integrator for ``model``, splitting on its declared component costs
    unless ``cheap`` names the cheap components explicitly (handy for the hand-written problems,
    which declare none)."""
    from ..hmc import multirate_leapfrog, split_potentials
    if cheap is None:
        cheap_pots, expensive = split_potentials(model, potentials)
    else:
        names = {f"V_{c}" for c in cheap} | {"V_jacobian"}
        cheap_pots = [p for p in potentials if p.id in names]
        expensive = [p for p in potentials if p.id not in names]
    return multirate_leapfrog(cheap_pots, expensive, kinetic, n=n)


def multirate_nuts(*, n: int = 4, cheap=None, init=None, max_tree_depth: int = 10,
                   step_size: float = 0.5, metric: str = "diagonal",
                   target_accept: float = 0.8, mass_min_samples: int = 50,
                   mass_adapt: str = "covariance", center=False, terminate=False,
                   **kwargs) -> Builder:
    """NUTS over the multi-rate (RESPA) integrator: the model's *expensive* components are kicked
    once per macro step, its cheap ones (plus the chart Jacobian) ``n`` times.

    The split comes from ``model.cheap_components``; pass ``cheap=("log_prior_v",)`` to split a
    model that declares nothing (the hand-written problems). A model with nothing cheap, or
    nothing expensive, raises --- see :func:`mimcs.hmc.multirate_leapfrog`."""
    from ..samplers import make_sampler_class
    from ..hmc import NUTS, default_potentials, make_kinetic

    Cls = make_sampler_class(
        *_adaptation_mixins(mass_adapt, metric, center, False, terminate), NUTS)

    def build(model, seed):
        kinetic = make_kinetic(metric)
        potentials = default_potentials(model)
        integrator = _multirate_integrator(model, potentials, kinetic, n=n, cheap=cheap)
        init_position = init if init is not None else model.default_sample()
        return Cls(model, init_position=init_position, seed=seed, kinetic=kinetic,
                   potentials=potentials, integrator=integrator,
                   max_tree_depth=max_tree_depth, step_size=step_size,
                   target_accept=target_accept, mass_min_samples=mass_min_samples, **kwargs)

    return build


def _walnuts_build(SamplerCls, model, seed, *, metric, schedule, error_thresholds, init,
                   p=None, multirate_n=None, cheap=None, **kwargs):
    from ..hmc import (
        LineSearchIntegrator, MarkovianLineSearchIntegrator, leapfrog,
        default_potentials, make_kinetic)
    kinetic = make_kinetic(metric)
    potentials = default_potentials(model)
    # ``multirate_n`` refines a whole multi-rate step instead of a plain leapfrog one: the line
    # search only ever calls ``base.step(istate, eps, ctx)``, so any integrator serves as a base.
    base = (leapfrog(potentials, kinetic) if multirate_n is None
            else _multirate_integrator(model, potentials, kinetic, n=multirate_n, cheap=cheap))
    if p is None:   # deterministic WALNUTS-D
        integrator = LineSearchIntegrator(base, potentials, kinetic, schedule=schedule,
                                          error_thresholds=error_thresholds)
    else:           # randomized Markovian variant
        integrator = MarkovianLineSearchIntegrator(
            base, potentials, kinetic, schedule=schedule,
            error_thresholds=error_thresholds, p=p)
    init_position = init if init is not None else model.default_sample()
    return SamplerCls(model, init_position=init_position, seed=seed, kinetic=kinetic,
                      potentials=potentials, integrator=integrator, **kwargs)


def _walnuts_sampler_class(base, adapt_step_size: bool):
    """Compose the WALNUTS sampler class. With ``adapt_step_size`` the proxy-driven
    :class:`~mimcs.adaptation.LineSearchStepSizeAdaptation` takes the top mixin slot (the position
    ``RobbinsMonroStepSize`` occupies for ordinary samplers); otherwise only the mass matrix is
    adapted (the historical WALNUTS default)."""
    from ..samplers import make_sampler_class
    from ..adaptation import MassMatrixAdaptation, LineSearchStepSizeAdaptation
    if adapt_step_size:
        return make_sampler_class(LineSearchStepSizeAdaptation, MassMatrixAdaptation, base)
    return make_sampler_class(MassMatrixAdaptation, base)


def wal_hmc(*, metric: str = "diagonal", schedule=None, error_thresholds: float = 1.0,
            n_macro: int = 10, step_size: float = 0.5, mass_min_samples: int = 50,
            adapt_step_size: bool = True, target_accept: float = 0.8,
            init=None, **kwargs) -> Builder:
    """Builder for WAL-HMC: fixed-length HMC over the within-orbit adaptive
    :class:`~mimcs.hmc.LineSearchIntegrator` (``n_macro`` adaptive macro steps).

    ``schedule`` is a list of ``(h_j, T_j)`` refinement levels (default doubling) and
    ``error_thresholds`` the per-level energy-error budget.

    The macro ``step_size`` is adapted by default, via the integrator's coarse-level *proxy*
    acceptance toward ``target_accept`` --- the same thing
    :func:`mimcs.make_sampler` builds for a line-search integrator, so the harness exercises
    what the factory ships. A line-search integrator refines until the energy error is within
    budget, so its real acceptance is ~1 whatever the macro step is and ordinary
    acceptance-driven adaptation would run the step away upward; the proxy is what makes the
    signal informative (``docs/design/06``). ``adapt_step_size=False`` pins the step instead."""
    from ..hmc import HMC

    Cls = _walnuts_sampler_class(HMC, adapt_step_size)

    def build(model, seed):
        return _walnuts_build(
            Cls, model, seed, metric=metric, schedule=schedule,
            error_thresholds=error_thresholds, init=init, n_leapfrog=n_macro,
            step_size=step_size, mass_min_samples=mass_min_samples,
            target_accept=target_accept, **kwargs)

    return build


def wal_nuts(*, metric: str = "diagonal", schedule=None, error_thresholds: float = 1.0,
             max_tree_depth: int = 10, step_size: float = 0.5, mass_min_samples: int = 50,
             adapt_step_size: bool = True, target_accept: float = 0.8,
             multirate_n=None, cheap=None, init=None, **kwargs) -> Builder:
    """Builder for WAL-NUTS: NUTS with the within-orbit adaptive
    :class:`~mimcs.hmc.LineSearchIntegrator` as the leaf step (each leaf is an adaptive macro
    step). See :func:`wal_hmc` for the schedule/threshold arguments and ``adapt_step_size``.

    ``multirate_n`` refines a *multi-rate* macro step instead of a plain leapfrog one (``cheap``
    names the cheap components for a model that declares none)."""
    from ..hmc import NUTS

    Cls = _walnuts_sampler_class(NUTS, adapt_step_size)

    def build(model, seed):
        return _walnuts_build(
            Cls, model, seed, metric=metric, schedule=schedule,
            error_thresholds=error_thresholds, init=init, max_tree_depth=max_tree_depth,
            step_size=step_size, mass_min_samples=mass_min_samples,
            target_accept=target_accept, multirate_n=multirate_n, cheap=cheap, **kwargs)

    return build


def mwal_nuts(*, metric: str = "diagonal", schedule=None, error_thresholds: float = 1.0,
              p: float = 0.5, max_tree_depth: int = 10, step_size: float = 0.5,
              mass_min_samples: int = 50, adapt_step_size: bool = True,
              target_accept: float = 0.8, init=None, **kwargs) -> Builder:
    """Builder for the *Markovian* WAL-NUTS: NUTS whose leaf is the randomized
    :class:`~mimcs.hmc.MarkovianLineSearchIntegrator`. The refinement level is chosen by a
    coarse-to-fine Markov chain with unforced-refinement probability ``p`` (see :func:`wal_nuts`
    for the schedule/threshold arguments and ``adapt_step_size``). Avoids invalidation entirely at
    the cost of a more spread-out micro-step-length distribution."""
    from ..hmc import NUTS

    Cls = _walnuts_sampler_class(NUTS, adapt_step_size)

    def build(model, seed):
        return _walnuts_build(
            Cls, model, seed, metric=metric, schedule=schedule,
            error_thresholds=error_thresholds, init=init, p=p, max_tree_depth=max_tree_depth,
            step_size=step_size, mass_min_samples=mass_min_samples,
            target_accept=target_accept, **kwargs)

    return build


def _block_kinetics(model, modes, dense_max_dim: int) -> list:
    """One constant-mass kinetic per model parameter (its own coordinate block): dense for a
    moderate multi-dimensional block (``1 < size <= dense_max_dim``), diagonal otherwise, or
    the explicit ``modes[name]`` override. This is the block-diagonal constant mass, now just a
    list of ordinary kinetics in ``BaseHMC`` (no composite)."""
    from ..hmc import DiagonalQuadraticKinetic, DenseQuadraticKinetic
    modes = modes or {}
    kinetics = []
    for p in model.parameters:
        s, e = model.coord_block(p.name)
        size = e - s
        mode = modes.get(p.name, "dense" if 1 < size <= dense_max_dim else "diagonal")
        cls = DenseQuadraticKinetic if mode == "dense" else DiagonalQuadraticKinetic
        kinetics.append(cls(id=p.name, slices=[(s, e)]))
    return kinetics


def _block_build(SamplerCls, model, seed, *, modes, dense_max_dim, init, **kwargs):
    from ..hmc import default_potentials, leapfrog
    kinetics = _block_kinetics(model, modes, dense_max_dim)
    potentials = default_potentials(model)
    integrator = leapfrog(potentials, kinetics)
    init_position = init if init is not None else model.default_sample()
    return SamplerCls(model, init_position=init_position, seed=seed, kinetics=kinetics,
                      potentials=potentials, integrator=integrator, **kwargs)


def block_hmc(*, modes=None, dense_max_dim: int = 50, init=None, n_leapfrog: int = 20,
              step_size: float = 0.5, target_accept: float = 0.8,
              mass_min_samples: int = 50, **kwargs) -> Builder:
    """Builder for HMC with a **block-diagonal constant mass** (one Diagonal/Dense kinetic per
    parameter, adapted by the block-aware :class:`mimcs.adaptation.MassMatrixAdaptation`). Mode
    per block: ``modes[name]`` override, else dense for a moderate block, diagonal otherwise."""
    from ..samplers import make_sampler_class
    from ..adaptation import RobbinsMonroStepSize, MassMatrixAdaptation
    from ..hmc import HMC

    Cls = make_sampler_class(RobbinsMonroStepSize, MassMatrixAdaptation, HMC)

    def build(model, seed):
        return _block_build(Cls, model, seed, modes=modes, dense_max_dim=dense_max_dim,
                            init=init, n_leapfrog=n_leapfrog, step_size=step_size,
                            target_accept=target_accept, mass_min_samples=mass_min_samples,
                            **kwargs)

    return build


def block_nuts(*, modes=None, dense_max_dim: int = 50, init=None, max_tree_depth: int = 10,
               step_size: float = 0.5, target_accept: float = 0.8,
               mass_min_samples: int = 50, **kwargs) -> Builder:
    """Builder for NUTS with a block-diagonal constant mass. See :func:`block_hmc`."""
    from ..samplers import make_sampler_class
    from ..adaptation import RobbinsMonroStepSize, MassMatrixAdaptation
    from ..hmc import NUTS

    Cls = make_sampler_class(RobbinsMonroStepSize, MassMatrixAdaptation, NUTS)

    def build(model, seed):
        return _block_build(Cls, model, seed, modes=modes, dense_max_dim=dense_max_dim,
                            init=init, max_tree_depth=max_tree_depth, step_size=step_size,
                            target_accept=target_accept, mass_min_samples=mass_min_samples,
                            **kwargs)

    return build


def rmnuts(*, metric, init=None, max_tree_depth: int = 10, n_fixed_point: int = 8,
           solver=None, step_size: float = 0.5, target_accept: float = 0.8,
           **kwargs) -> Builder:
    """Builder for Riemannian Manifold NUTS: NUTS over a position-dependent metric.
    **[experimental]** --- the implicit variant; see :mod:`mimcs.hmc.riemannian`.

    The modular design makes this a drop-in combination -- NUTS with the Riemannian
    kinetic (its ``velocity = G(q)^{-1} p`` drives the generalized U-turn) and the
    generalized (implicit) leapfrog as the leaf integrator. ``metric`` is a
    :class:`~mimcs.hmc.Metric` or a callable ``q -> SPD matrix``; ``solver`` selects the
    implicit-step fixed-point solver (``None`` Picard, ``"anderson"``, or an object).
    """
    from ..samplers import make_sampler_class
    from ..adaptation import RobbinsMonroStepSize
    from ..hmc import (
        NUTS, RiemannianKinetic, leapfrog, default_potentials, Metric, AnalyticMetric)
    from ..hmc.solvers import resolve_solver, PicardSolver

    metric_obj = metric if isinstance(metric, Metric) else AnalyticMetric(metric)
    solver_obj = resolve_solver(solver, n_fixed_point) or PicardSolver(n_fixed_point)
    Cls = make_sampler_class(RobbinsMonroStepSize, NUTS)

    def build(model, seed):
        init_position = init if init is not None else model.default_sample()
        kinetic = RiemannianKinetic(metric_obj, solver=solver_obj)
        potentials = default_potentials(model)
        integrator = leapfrog(potentials, kinetic)   # implicit work lives in kinetic.flow
        return Cls(model, init_position=init_position, seed=seed,
                   kinetic=kinetic, potentials=potentials, integrator=integrator,
                   max_tree_depth=max_tree_depth, step_size=step_size,
                   target_accept=target_accept, **kwargs)

    return build


def simple_nuts(*, init=None, max_tree_depth: int = 10, step_size: float = 0.5,
                metric: str = "diagonal", target_accept: float = 0.8,
                mass_min_samples: int = 50, **kwargs) -> Builder:
    """Builder for the reference (full-memory) NUTS, for testing against ``nuts``."""
    from ..samplers import make_sampler_class
    from ..adaptation import RobbinsMonroStepSize, MassMatrixAdaptation
    from ..hmc import SimpleNUTS

    Cls = make_sampler_class(RobbinsMonroStepSize, MassMatrixAdaptation, SimpleNUTS)

    def build(model, seed):
        init_position = init if init is not None else model.default_sample()
        return Cls(model, init_position=init_position, seed=seed,
                   max_tree_depth=max_tree_depth, step_size=step_size, metric=metric,
                   target_accept=target_accept, mass_min_samples=mass_min_samples,
                   **kwargs)

    return build


def randomized_hmc(*, init=None, n_leapfrog: int = 20, step_size: float = 0.5,
                   metric: str = "diagonal", target_accept: float = 0.8,
                   mass_min_samples: int = 50, **kwargs) -> Builder:
    """Convenience builder for HMC with randomized integration time (L ~ U{T..2T})."""
    from ..samplers import make_sampler_class
    from ..adaptation import RobbinsMonroStepSize, MassMatrixAdaptation
    from ..hmc import RandomizedHMC

    Cls = make_sampler_class(RobbinsMonroStepSize, MassMatrixAdaptation, RandomizedHMC)

    def build(model, seed):
        init_position = init if init is not None else model.default_sample()
        return Cls(model, init_position=init_position, seed=seed,
                   n_leapfrog=n_leapfrog, step_size=step_size, metric=metric,
                   target_accept=target_accept, mass_min_samples=mass_min_samples,
                   **kwargs)

    return build
