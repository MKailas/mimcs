"""Lower a :class:`~mimcs.factory.spec.SamplerSpec` onto the sampler-assembly machinery.

Each ``BlockSpec`` becomes one kinetic component (``DiagonalQuadraticKinetic`` /
``DenseQuadraticKinetic`` over the block's coordinate slice); ``BaseHMC`` holds the list of
them (doc 06), so a block-diagonal mass is just several kinetics with no composite. The
adaptation mixins are composed with the base algorithm via
:func:`~mimcs.samplers.base.make_sampler_class`; the (list-aware) ``ScoreMassAdaptation`` adapts
each block's mass to its score covariance over that block's own coordinate slice. Centering
(``RobustCenteringAdaptation``, standardizing centered parameters by median/MAD) is **opt-in**
via ``spec.centering`` and off by default (see :func:`~mimcs.factory.spec.default_spec`).
"""

from __future__ import annotations

from functools import partial

import numpy as np

from .._logging import get_logger
from ..samplers import make_sampler_class, DiscreteMetropolisWithinGibbs, StaticContinuous

log = get_logger(__name__)
from ..adaptation import (
    RobbinsMonroStepSize, LineSearchStepSizeAdaptation, ScoreMassAdaptation, MassMatrixAdaptation,
    RobustCenteringAdaptation, MetricAdaptation, ShapedMetricAdaptation, LowRankAdaptation,
    UnitVectorCenteringAdaptation,
    UniformInit, StepSizeLineSearch, ClassifierTermination, GelmanRubinTermination,
    DiscreteMarginalAdaptation)
from ..model.unit_vector import UnitVectorParameter

_TERMINATION = {"classifier": ClassifierTermination, "rhat": GelmanRubinTermination}
from ..hmc import (
    HMC, RandomizedHMC, NUTS, DiagonalQuadraticKinetic, DenseQuadraticKinetic,
    LowRankQuadraticKinetic, LineSearchIntegrator, MarkovianLineSearchIntegrator,
    build_block, default_potentials, split_potentials, leapfrog, multirate_leapfrog)
from ..pt.integrators import BUDGETED_INTEGRATORS, product_error_thresholds

#: ``"static"`` is the base for a model with **no continuous parameters**: it leaves the (empty)
#: continuous block alone so the Gibbs sweep has something to compose over. It has no kinetics, no
#: potentials, no integrator and no step size, so ``build_sampler`` takes a narrower path for it
#: --- selected by ``discrete_only_base_rule``, and refused for anything with a coordinate.
_BASE = {"nuts": NUTS, "hmc": HMC, "randomized_hmc": RandomizedHMC,
         "static": StaticContinuous}

#: Bases with no Hamiltonian machinery: no kinetics, potentials, integrator or step size.
_STATIC_BASES = frozenset({"static"})

#: The discrete proposal families selectable by ``SamplerSpec.discrete_proposal``, keyed by its
#: value. ``None`` (not in the table) leaves the sweep's own uniform-over-the-others proposal, the
#: placeholder an ordinal or count-valued proposal will replace (doc 14). The sweep itself is not
#: in here: it is not optional for a model that has labels to move.
_DISCRETE_PROPOSAL = {"marginal": DiscreteMarginalAdaptation}

#: Every base has a parallel-tempered counterpart (doc 13): the same algorithm run over the K-fold
#: product space, keeping the cold chain. Built by delegating to ``parallel_tempering`` rather than
#: by reassembling the mixin stack here --- the product model, tempered potentials, product
#: kinetics, ladder adaptation and swap move are all its job, and duplicating that split would let
#: the two paths drift.
_TEMPERED_PREFIX = "pt_"

#: ``tempering_params`` keys, validated like ``integrator_params`` so a typo cannot pass silently.
_TEMPERING_PARAMS = frozenset({
    "n_temperatures", "betas", "beta_min", "tempered", "adapt_ladder", "adapt_beta_min",
    "swap_target_accept", "ladder_adapt_n0", "ladder_adapt_kappa", "selection",
    "per_temperature_step_size"})

#: The mass adaptations selectable by ``SamplerSpec.mass_adapt``, keyed by its value. Both fit the
#: *quadratic* blocks (``mass_mode in ("diagonal", "dense")``) and so are mutually exclusive: one
#: is chosen, never both. The other adaptations below filter on ``mass_mode is None`` and are
#: orthogonal to this choice.
_MASS = {"score": ScoreMassAdaptation, "covariance": MassMatrixAdaptation}

#: Adaptations that must run **per temperature** rather than on the product chain: each fits one
#: rung's own width, and a hot rung is wider. Everything else (step size, termination, the
#: initialization mixins) is global. See :mod:`mimcs.pt.adaptation`.
#:
#: Derived from ``_MASS`` rather than listing the mass adaptations by name: omitting one here does
#: not fail loudly. A ``ProductKinetic`` copies ``mass_mode`` and ``slices`` from the block it
#: wraps, so a mass adaptation left out of this tuple still *passes* its own block filter on the
#: product chain and then gathers ``x[0:n]`` --- rung 0's coordinates. For a diagonal block that
#: broadcasts silently against the ``(K, n)`` parameters and every rung quietly gets the cold
#: rung's mass; for a dense block it raises somewhere inside ``chol_update``, naming nothing about
#: tempering.
_PER_TEMPERATURE_ADAPTATIONS = (
    *_MASS.values(), LowRankAdaptation, MetricAdaptation, ShapedMetricAdaptation)


def _base_name(spec) -> tuple[str, bool]:
    """``spec.base`` split into the underlying algorithm and whether it is tempered."""
    if spec.base.startswith(_TEMPERED_PREFIX):
        return spec.base[len(_TEMPERED_PREFIX):], True
    return spec.base, False

#: Default cheap sub-steps per expensive gradient when ``spec.integrator == "multirate"``.
MULTIRATE_DEFAULT_N = 4


def _build_leapfrog(model, potentials, kinetics, params):
    return leapfrog(potentials, kinetics)


def _build_multirate(model, potentials, kinetics, params):
    """The multi-rate (RESPA) splitting over the model's declared cheap/expensive components."""
    cheap, expensive = split_potentials(model, potentials)
    n = int(params.get("n", MULTIRATE_DEFAULT_N))
    log.debug("multi-rate integrator: n=%d, cheap %s, expensive %s", n,
              [p.id for p in cheap], [p.id for p in expensive])
    return multirate_leapfrog(cheap, expensive, kinetics, n=n)   # raises on a degenerate split


def _build_line_search(model, potentials, kinetics, params, *, markovian=False):
    """A WALNUTS line-search integrator over a *base* integrator, itself built from this table.

    ``params["base"]`` picks the base (``"leapfrog"`` or ``"multirate"``) and
    ``params["base_params"]`` configures it, so a line search can refine either a plain leapfrog
    step or a whole multi-rate step. The base's options are nested rather than flat so each
    level is validated on its own and neither can steal the other's keys.
    """
    base_name = params.get("base", "leapfrog")
    if base_name not in _LINE_SEARCH_BASES:
        raise ValueError(f"unknown line-search base integrator {base_name!r} "
                         f"(use one of {sorted(_LINE_SEARCH_BASES)})")
    base_params = dict(params.get("base_params", {}))
    _check_integrator_params(base_name, base_params)
    base = _INTEGRATOR[base_name](model, potentials, kinetics, base_params)
    kwargs = {k: params[k] for k in ("schedule", "error_thresholds") if k in params}
    if markovian:
        if "p" in params:
            kwargs["p"] = params["p"]
        return MarkovianLineSearchIntegrator(base, potentials, kinetics, **kwargs)
    return LineSearchIntegrator(base, potentials, kinetics, **kwargs)


#: Integrator builders keyed by ``SamplerSpec.integrator``; each takes
#: ``(model, potentials, kinetics, params)``.
_INTEGRATOR = {
    "leapfrog": _build_leapfrog,
    "multirate": _build_multirate,
    "line_search": _build_line_search,
    "markovian_line_search": partial(_build_line_search, markovian=True),
}

#: The ``integrator_params`` keys each builder accepts. A typo would otherwise be silently
#: ignored --- the same failure mode the duplicate-potential-id guard exists to prevent.
_INTEGRATOR_PARAMS = {
    "leapfrog": frozenset(),
    "multirate": frozenset({"n"}),
    "line_search": frozenset({"base", "base_params", "schedule", "error_thresholds"}),
    "markovian_line_search": frozenset({"base", "base_params", "schedule",
                                        "error_thresholds", "p"}),
}

#: Integrators usable as a line-search *base*: deterministic, and their ``step`` preserves the
#: ``integrator_data`` structure (the line search drives a base inside a ``fori_loop``).
_LINE_SEARCH_BASES = ("leapfrog", "multirate")


def _check_integrator_params(name: str, params: dict) -> None:
    unknown = sorted(set(params) - _INTEGRATOR_PARAMS[name])
    if unknown:
        raise ValueError(
            f"unknown integrator_params {unknown} for integrator {name!r} "
            f"(it takes {sorted(_INTEGRATOR_PARAMS[name]) or 'no options'})")


def _build_integrator(spec, model, potentials, kinetics):
    if spec.integrator not in _INTEGRATOR:
        raise ValueError(
            f"unknown integrator {spec.integrator!r} (use one of {sorted(_INTEGRATOR)})")
    params = dict(spec.integrator_params)
    _check_integrator_params(spec.integrator, params)
    return _INTEGRATOR[spec.integrator](model, potentials, kinetics, params)


def _has_adaptive_unit_vector(model) -> bool:
    """Does this model have a unit vector whose chart asks to be fitted?

    Unlike ``centered`` --- which defaults to ``False``, making ``spec.centering`` a free opt-in
    --- ``UnitVectorParameter.adaptive`` defaults to **True**: the parameter itself asks for the
    adaptation, and its docstring promises it. So the gate is a property of the model rather than
    a spec field, and a user who does not want it writes ``adaptive=False`` on the parameter.
    Gating this on ``spec.centering`` instead would silently override that opt-in and couple two
    unrelated knobs (see ``docs/design/09``).
    """
    return any(isinstance(p, UnitVectorParameter) and p.adaptive for p in model.parameters)


def _block_kinetic(block, model):
    """A kinetic for one block (its id joins the parameter names; the id becomes a
    draw-component field name, so it must be a valid identifier). The block's coordinates are a
    list of slices, possibly non-contiguous.

    ``"learned_metric"`` lowers to a position-dependent :class:`LearnedDiagonalBlock` built from
    ``block.params["metric"]`` (a :class:`~mimcs.hmc.metric_expr.MetricExpr`) and an optional
    fitted ``block.params["metric_init"]``; it must be a single (contiguous) parameter."""
    slices = list(block.coord_slices)
    kid = "__".join(block.names) if block.names else "blk_" + "_".join(
        f"{s}_{e}" for s, e in slices)
    if block.kind == "dense":
        return DenseQuadraticKinetic(id=kid, slices=slices)
    if block.kind == "diagonal":
        return DiagonalQuadraticKinetic(id=kid, slices=slices)
    if block.kind == "lowrank":
        return LowRankQuadraticKinetic(id=kid, slices=slices,
                                       rank=int(block.params.get("rank", 4)))
    if block.kind == "learned_metric":
        if len(block.names) != 1 or len(slices) != 1:
            raise NotImplementedError(
                "learned_metric requires a single (contiguous) parameter block")
        return build_block(model, block.names[0], block.params["metric"],
                           init=block.params.get("metric_init"),
                           shape=block.params.get("shape"))    # None: diagonal; "dense"/("lowrank",J)
    raise NotImplementedError(
        f"block kind {block.kind!r} is not implemented yet (relativistic is stage 2+)")


#: Whether each integrator emits a step-size proxy, read off the classes so the values cannot
#: drift. The untempered path gates on the *instance* attribute (see ``build_sampler``); the
#: tempered path cannot, because its integrator is built over product components that only exist
#: inside ``parallel_tempering``, so it consults this table by name instead.
#: Integrators that consume per-step randomness, keyed by ``SamplerSpec.integrator``. They are
#: only *randomized* under a base that declares the draws for them
#: (``BaseHMC.supplies_integrator_rng``); under any other base they degrade to their deterministic
#: variant without complaining, which is what ``build_sampler`` refuses rather than deliver
#: quietly. A name table for the same reason as ``_EMITS_PROXY``: the tempered path has no
#: integrator instance to ask, and ``n_rng_per_step`` is set per *instance* from the schedule
#: length rather than on the class, so it cannot be read off one.
_NEEDS_PER_STEP_RNG = frozenset({"markovian_line_search"})

_EMITS_PROXY = {
    "leapfrog": False,
    "multirate": False,
    "line_search": LineSearchIntegrator.emits_step_size_proxy,
    "markovian_line_search": MarkovianLineSearchIntegrator.emits_step_size_proxy,
}


def _tempered_integrator_builder(spec):
    """``(potentials, kinetics, K) -> integrator`` over the product components.

    A callable rather than an instance because the product potentials and kinetics are built
    inside ``parallel_tempering``. The *base* model is passed to the integrator builders, not the
    ``ProductModel``: the only builder that reads it is multi-rate, whose cheap/expensive split is
    a statement about the model's components (``split_potentials`` unwraps the tempered wrappers).
    """
    name, params = spec.integrator, dict(spec.integrator_params)
    _check_integrator_params(name, params)

    def build(potentials, kinetics, n_temperatures: int):
        p = dict(params)
        if name in BUDGETED_INTEGRATORS:
            # The budget is stated per temperature; ``mimcs.pt`` owns what that means over the
            # product space, so ask it rather than restate the rule (or the default) here.
            p["error_thresholds"] = product_error_thresholds(
                p.get("error_thresholds"), n_temperatures)
        return _INTEGRATOR[name](spec.model, potentials, kinetics, p)

    return build


def _build_tempered(spec, algo, kinetics, mixins, kwargs, *, seed, init):
    """Build a parallel-tempered sampler by delegating to :func:`~mimcs.pt.parallel_tempering`.

    The mixin list assembled for an ordinary sampler is *split* rather than reused wholesale: the
    mass and metric adaptations belong to each temperature (they fit that rung's own width, and a
    hot rung is wider), while the step size, warmup termination and the initialization mixins are
    global. That split is the whole reason PT keeps K adaptation hosts (doc 13).
    """
    from ..pt import parallel_tempering

    params = dict(spec.tempering_params)
    unknown = sorted(set(params) - _TEMPERING_PARAMS)
    if unknown:
        raise ValueError(f"unknown tempering_params {unknown} for base {spec.base!r} "
                         f"(it takes {sorted(_TEMPERING_PARAMS)})")
    if spec.centering:
        raise NotImplementedError(
            "centering is not supported with a tempered base: a chart's (mu, sigma) is shared by "
            "every temperature, so it is neither a per-rung quantity nor well defined from the "
            "product coordinate. Set spec.centering = False.")
    if _has_adaptive_unit_vector(spec.model):
        # The same argument as centering, and the failure would be quieter: a ProductModel's
        # `parameters` *is* the base model's list, so the mixin would find the adaptive unit
        # vectors and then rechart using base-model offsets against a (K * n) product
        # coordinate --- wrong, with nothing raising.
        raise NotImplementedError(
            "an adaptive unit vector is not supported with a tempered base: the chart's pole is "
            "shared by every temperature, so it is neither a per-rung quantity nor well defined "
            "from the product coordinate. Set adaptive=False on the unit-vector parameter(s).")
    per_temperature = [m for m in mixins if m in _PER_TEMPERATURE_ADAPTATIONS]
    global_mixins = [m for m in mixins if m not in _PER_TEMPERATURE_ADAPTATIONS]
    log.info("parallel tempering: global mixins %s, per-temperature %s",
             [m.__name__ for m in global_mixins], [m.__name__ for m in per_temperature])
    return parallel_tempering(
        spec.model, _init_position(spec, init, with_labels=False), base=algo, kinetics=kinetics,
        integrator=_tempered_integrator_builder(spec), step_size=spec.step_size, seed=seed,
        extra_mixins=tuple(global_mixins), adapt_mixins=tuple(per_temperature),
        **params, **kwargs)


def _check_static(spec, model, tempered: bool) -> None:
    """Guard the ``"static"`` base: it moves nothing continuous, so several spec fields are moot.

    Refused rather than silently ignored, the same policy the rest of ``build`` follows: a user who
    asked for an adapted step size and got a sampler with no step size at all has been handed a
    different algorithm from the one they asked for. ``discrete_only_base_rule`` sets all three
    consistently, so these fire only for a hand-edited spec.
    """
    if not model.discrete_dim:
        raise ValueError(
            "base 'static' moves nothing at all on a model with no discrete parameters: every "
            "coordinate would stay exactly where it started. It exists for a model that is *only* "
            "discrete, where the Gibbs sweep composed over it does all the moving.")
    if tempered:
        raise ValueError(
            "base 'pt_static' is not supported: parallel tempering builds product kinetics and "
            "tempered potentials over a continuous coordinate, and a static base has neither. "
            "Temper a discrete-only model by giving it a NUTS base over its (empty) continuous "
            "block, or sample it untempered.")
    if spec.blocks:
        raise ValueError(
            f"base 'static' has no kinetics, but the spec carries {len(spec.blocks)} block(s) "
            f"({', '.join('+'.join(b.names) for b in spec.blocks)}). A block *is* a kinetic over "
            f"a coordinate slice; set spec.blocks = [].")
    if spec.adapt_step_size:
        raise ValueError(
            "base 'static' takes no step: there is no step size to adapt. Set "
            "spec.adapt_step_size = False.")
    if spec.mass_adapt is not None:
        raise ValueError(
            f"base 'static' has no kinetics for mass_adapt={spec.mass_adapt!r} to fit. Set "
            f"spec.mass_adapt = None.")


def _init_position(spec, init, *, with_labels: bool = True):
    """The initial position, warm-started from the evidence when there is any.

    For a model with integer parameters this is a **dict** rather than a flat array, because that
    is the only channel that carries both halves of a state (``_as_sample_flat`` /
    ``_as_discrete_flat`` in ``mimcs/samplers/metropolis.py``). Warm-starting the continuous block
    to a fitted configuration while resetting the labels to their lower bound would pair a position
    with the wrong assignment --- for a mixture, every observation's coordinates fitted under one
    clustering and every label saying "cluster 0".

    ``with_labels=False`` forces the flat form for the **tempered** path: ``parallel_tempering``
    takes one rung's position and tiles it ``K``-fold (``np.asarray(init_position, float)``), which
    a dict cannot survive, and its labels come from ``ProductModel.default_discrete()`` either way.
    So a tempered run warm-starts its position but not its labels. That is a real limitation rather
    than an oversight --- giving each rung its own warm-started labels means tiling them through
    ``parallel_tempering``'s own init, which is its call to make, not ``build``'s.

    Note ``sampler.initialize()`` overwrites both halves regardless (``UniformInit`` the position,
    the Gibbs sweep's own hook the labels): "initialize" means start fresh, and that is unchanged.
    """
    if init is not None:
        log.debug("initial position: the one supplied to build()")
        return init
    model = spec.model
    ev = spec.evidence
    if ev is not None and ev.samples is not None and len(ev.samples):
        last = np.asarray(ev.samples[-1], dtype=float)
        z = getattr(ev, "discrete", None)
        if with_labels and model.discrete_dim and z is not None and len(z) == len(ev.samples):
            log.debug("initial position: warm start (position and labels) from the last of %d "
                      "evidence draw(s)", len(ev.samples))
            return {**model.unpack_sample(last),
                    **model.unpack_discrete(np.asarray(z[-1], dtype=np.int32))}
        if model.discrete_dim:
            log.debug("initial position: warm start from the last of %d evidence draw(s); the "
                      "labels start from the model's default (%s)", len(ev.samples),
                      "not carried by this evidence" if z is None else
                      "a tempered run tiles one rung's position and cannot take a dict")
        else:
            log.debug("initial position: warm start from the last of %d evidence draw(s)",
                      len(ev.samples))
        return last                                      # warm-start from the last draw
    log.debug("initial position: the model's default sample")
    return np.asarray(model.default_sample(), dtype=float)


def build_sampler(spec, *, seed: int = 0, init=None, buffer_size=None):
    """Construct the sampler described by ``spec``.

    ``buffer_size`` sizes the RNG buffer. ``None`` (the default) means "unspecified", leaving
    ``spec.algo_kwargs["buffer_size"]`` --- which has always reached the constructor --- or the
    1024 default in force; anything else overrides both. It is a **memory** knob, and the reason
    to reach for it is that the buffer is ``B * sum(prod(shape))`` floats per component: for NUTS
    that is dominated by ``leaf_select`` and so is ~21 MB at the defaults almost regardless of the
    model's dimension (doc 03). It is not stream-neutral --- two runs at the same seed and
    different ``buffer_size`` agree only up to the first refill.
    """
    model = spec.model
    algo_name, tempered = _base_name(spec)
    if algo_name not in _BASE:
        # ``pt_static`` is left out of the suggestions: a static base has no continuous
        # coordinate for tempered potentials or product kinetics to act on, so offering it here
        # would point at an option ``_check_static`` immediately refuses.
        raise ValueError(
            f"unknown base {spec.base!r} (use one of "
            f"{sorted(list(_BASE) + [_TEMPERED_PREFIX + b for b in _BASE if b not in _STATIC_BASES])})")
    if not tempered and spec.tempering_params:
        raise ValueError(
            f"tempering_params were given but base {spec.base!r} is not tempered; use "
            f"{_TEMPERED_PREFIX + algo_name!r}")
    # Validated whether or not the model has discrete parameters, so a typo raises on every model
    # rather than only on the ones where the field bites --- the same reason ``mass_adapt`` is
    # checked before the append that uses it.
    if spec.discrete_proposal is not None and spec.discrete_proposal not in _DISCRETE_PROPOSAL:
        raise ValueError(
            f"unknown discrete_proposal {spec.discrete_proposal!r} "
            f"(use {sorted(_DISCRETE_PROPOSAL)} or None for the uniform proposal)")
    static = algo_name in _STATIC_BASES
    if static:
        _check_static(spec, model, tempered)
    # Block kinetics are the model's *own* block structure either way; under tempering
    # ``parallel_tempering`` wraps each one to apply at every temperature (doc 13).
    kinetics = [_block_kinetic(b, model) for b in spec.blocks]
    if spec.integrator not in _INTEGRATOR:
        raise ValueError(
            f"unknown integrator {spec.integrator!r} (use one of {sorted(_INTEGRATOR)})")
    if (not static and spec.integrator in _NEEDS_PER_STEP_RNG
            and not _BASE[algo_name].supplies_integrator_rng):
        # Refused rather than delivered quietly: the integrator would still *build*, and still
        # run, but as its deterministic variant --- `integrate` has no per-step coins to give it,
        # so no unforced refinement ever fires. A user who asked for the randomized one would get
        # WALNUTS-D and no indication of it.
        raise ValueError(
            f"integrator {spec.integrator!r} needs per-step randomness, which base "
            f"{spec.base!r} does not supply: it integrates a whole trajectory in one call, so "
            f"the integrator would silently run as its deterministic variant. Use a NUTS base "
            f"(it draws per-leaf coins), or integrator='line_search' for the deterministic "
            f"variant by choice.")
    # The integrator is built before the mixins because the step-size mixin depends on it. Under
    # tempering it cannot be built yet (its components are the product ones), so the choice is
    # made from the integrator's name instead --- the same fact, read off the class.
    potentials = None
    if static or tempered:
        # A static base has no Hamiltonian at all; a tempered one cannot build its integrator yet
        # (its components are the product ones), so the proxy question is answered from the
        # integrator's *name* instead --- the same fact, read off the class.
        integrator, emits_proxy = None, False if static else _EMITS_PROXY[spec.integrator]
    else:
        potentials = default_potentials(model)
        integrator = _build_integrator(spec, model, potentials, kinetics)
        emits_proxy = integrator.emits_step_size_proxy

    mixins = []
    # Warmup termination goes first (outermost): it only observes the drawn sample --- after every
    # adaptation has had its say --- and decides via should_stop() whether to end warmup.
    term = getattr(spec, "terminate", None)
    if term:
        if term not in _TERMINATION:
            raise ValueError(
                f"unknown terminate {term!r} (use {sorted(_TERMINATION)} or None)")
        mixins.append(_TERMINATION[term])
    if spec.adapt_step_size and not static:
        # A line-search integrator refines until the energy error is within budget, so its *real*
        # acceptance is ~1 whatever the macro step size and acceptance-driven adaptation runs the
        # step away upward; it must be driven by the integrator's proxy instead. This is a
        # **swap**, not an addition: LineSearchStepSizeAdaptation subclasses RobbinsMonroStepSize,
        # so listing both would raise "Cannot create a consistent MRO". Gated on the integrator
        # attribute the samplers themselves read, so any future proxy-emitting integrator is
        # handled without touching this.
        mixins.append(LineSearchStepSizeAdaptation if emits_proxy else RobbinsMonroStepSize)
    # Validated on `is not None` rather than truthiness, and before the append, so that a typo
    # raises even for a model whose blocks are all lowrank/learned_metric (where the mixin would
    # be inert). The two are list-aware and share a block filter, so this is a swap, not a stack.
    if spec.mass_adapt is not None:
        if spec.mass_adapt not in _MASS:
            raise ValueError(f"unknown mass_adapt {spec.mass_adapt!r} "
                             f"(use {sorted(_MASS)} or None)")
        if not static:
            mixins.append(_MASS[spec.mass_adapt])  # fits each diagonal/dense block over its slice
    if any(b.kind == "lowrank" for b in spec.blocks):
        mixins.append(LowRankAdaptation)         # adapts the low-rank blocks (partitions with the
                                                 # mass adaptation, which skips them: mass_mode
                                                 # is None) --- so `mass_adapt` does not reach them
    learned = [b for b in spec.blocks if b.kind == "learned_metric"]
    if any(b.params.get("shape") is None for b in learned):
        mixins.append(MetricAdaptation)          # SGD-adapts the diagonal learned blocks (also
                                                 # skipped by the mass adaptation: mass_mode None)
    if any(b.params.get("shape") is not None for b in learned):
        mixins.append(ShapedMetricAdaptation)    # adapts the shaped (D(x)^1/2 A D(x)^1/2) blocks
    if spec.centering:
        mixins.append(RobustCenteringAdaptation)
    if _has_adaptive_unit_vector(model):
        # Lowest adaptation priority, so the chart is refitted *before* the mass adaptation reads
        # the score --- the same order the hand-built path uses (``mimcs.testing.runner``).
        mixins.append(UnitVectorCenteringAdaptation)
    # Initialization mixins (inert until sampler.initialize() is called). Ordered so the position
    # init runs before the step-size line search (deeper in the MRO = runs first). Both are
    # Hamiltonian-only: ``UniformInit`` redraws through ``state_at_coordinate``, which lives on
    # ``BaseHMC``, and there is no coordinate to draw for a static base anyway.
    if not static:
        mixins += [StepSizeLineSearch, UniformInit]
    # The discrete pair goes last, so the sweep sits immediately left of the base algorithm --- the
    # invariant every hand-built site holds (``mimcs/testing/runner.py``, ``examples/05_mixture.py``,
    # the tests), and the one the module docstring of ``samplers/gibbs.py`` states. Two consequences
    # worth naming, because both are load-bearing:
    #
    # * ``DiscreteMarginalAdaptation`` must be **left of** the sweep --- it writes the proposal
    #   tables the sweep reads.
    # * Being deeper than ``UniformInit`` means the sweep's ``_initialize_hooks`` randomizes the
    #   labels *before* the position is drawn, so ``UniformInit``'s finite-density retry tests the
    #   coordinate against the labels the chain will actually start from. ``state_at_coordinate``
    #   carries ``discrete`` through untouched, so nothing is lost the other way either.
    #
    # Under tempering only the adaptation is added: ``parallel_tempering`` injects the sweep itself
    # (between ``ReplicaExchangeMixin`` and the selection mixins, a position that cannot be
    # expressed from out here), so adding it as well would both misplace it and duplicate it into
    # an MRO error.
    if model.discrete_dim:
        if spec.discrete_proposal is not None:
            mixins.append(_DISCRETE_PROPOSAL[spec.discrete_proposal])
        if not tempered:
            mixins.append(DiscreteMetropolisWithinGibbs)

    kwargs = dict(spec.algo_kwargs)
    kwargs.setdefault("target_accept", 0.8)
    kwargs.setdefault("mass_min_samples", 50)
    if buffer_size is not None:
        kwargs["buffer_size"] = buffer_size      # an explicit argument beats the spec's own
    log.info("building %s over %d kinetic block(s) [%s], %s, adaptations %s, "
             "seed %d, rng buffer %s", spec.base, len(kinetics),
             ", ".join(f"{k.id}[{b.kind}]" for k, b in zip(kinetics, spec.blocks)),
             "no integrator (static base)" if static else
             (spec.integrator + (f" {spec.integrator_params}" if spec.integrator_params else "")
              + " integrator"),
             [m.__name__ for m in mixins], seed, kwargs.get("buffer_size", "default"))
    if tempered:
        return _build_tempered(spec, _BASE[algo_name], kinetics, mixins, kwargs,
                               seed=seed, init=init)

    Cls = make_sampler_class(*mixins, _BASE[algo_name])
    log.debug("composed class %s; integrator %s; potentials %s; algo kwargs %s",
              Cls.__name__, type(integrator).__name__,
              [type(p).__name__ for p in potentials or ()], kwargs)
    if static:
        # No kinetics, potentials, integrator or step size to hand it: ``StaticState`` has no
        # field for any of them, and passing them would only park them in ``self._kwargs``.
        return Cls(model, init_position=_init_position(spec, init), seed=seed, **kwargs)
    return Cls(model, init_position=_init_position(spec, init), seed=seed,
               kinetics=kinetics, potentials=potentials, integrator=integrator,
               step_size=spec.step_size, **kwargs)
