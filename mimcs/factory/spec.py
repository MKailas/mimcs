"""The sampler *prototype*: :class:`SamplerSpec`, :class:`BlockSpec` and :class:`DiscreteSpec`.

A ``SamplerSpec`` carries every decision needed to construct a sampler, in attributes a
user can inspect and mutate before building --- the configuration seam of the factory (no
``**kwargs`` soup). The coordinate space is always modelled as a list of ``BlockSpec``.
``default_spec`` starts with a single whole-space block, but the block-partition rule replaces it
immediately, and each block carries its own kinetic kind --- diagonal, dense, low-rank, or a
learned position-dependent metric.

The **discrete** space is modelled the same way: one :class:`DiscreteSpec` per integer parameter,
each carrying its own update method. That parallel is the point --- a discrete parameter is no more
obliged to share an update rule with its neighbours than a coordinate block is to share a kinetic.

``analyze`` (in :mod:`mimcs.factory`) produces a spec from a model and earlier results;
``spec.build()`` lowers it onto the existing sampler-assembly machinery (see
:mod:`mimcs.factory.build`). See ``docs/design/09_sampler_factory.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BlockSpec:
    """One coordinate block and the kinetic chosen for it.

    ``kind == "learned_metric"`` reads ``params["metric"]`` --- a
    :class:`~mimcs.hmc.metric_expr.MetricExpr` position-dependent diagonal metric over the other
    blocks (e.g. ``Exp("v") + Exp()``) --- and optional ``params["metric_init"]`` (pre-fitted
    parameters). Both are set by the metric-regression rule and can be overridden by hand::

        spec.blocks[i].kind = "learned_metric"
        spec.blocks[i].params = {"metric": Exp("v") + Exp()}

    ``kind == "lowrank"`` uses a :class:`~mimcs.hmc.LowRankQuadraticKinetic` (diagonal-whitened
    rank-J mass) with ``params["rank"]`` low-rank directions (default 4), adapted by
    :class:`~mimcs.adaptation.LowRankAdaptation`.
    """

    names: list[str]                       #: parameter names spanned by the block
    coord_slices: list                     #: list of (start, stop) coordinate slices (may be
                                           #: non-contiguous, so a block can fuse scattered params)
    kind: str = "diagonal"                #: "diagonal" | "dense" | "lowrank" | "learned_metric"
    params: dict = field(default_factory=dict)   #: kind-specific options (see above)

    def __str__(self) -> str:
        """``names[dim,kind]``, plus whatever the kind is configured with.

        Deliberately not aliased to ``__repr__``: a learned metric's ``params["metric_init"]``
        is an array pytree that prints over 70 lines, so ``repr`` stays the full dataclass for
        debugging while display paths use this.
        """
        dim = sum(e - s for s, e in self.coord_slices)
        out = f"{'+'.join(self.names)}[{dim}d,{self.kind}]"
        extra = []
        if "rank" in self.params:
            extra.append(f"rank={self.params['rank']}")
        if "metric" in self.params:
            extra.append(f"metric={self.params['metric']!r}")
        if self.params.get("shape") is not None:
            extra.append(f"shape={self.params['shape']!r}")
        if "metric_init" in self.params:
            extra.append("pre-fitted")
        return out + (" (" + ", ".join(extra) + ")" if extra else "")


@dataclass
class DiscreteSpec:
    """One discrete parameter and the update method chosen for it.

    The discrete peer of :class:`BlockSpec`, down to the ``kind`` + ``params`` shape, so a method
    that needs configuration has somewhere to put it without a second field appearing on
    ``SamplerSpec``. The deferred methods in ``docs/design/14_discrete_parameters.md`` --- an
    ordinal +-1 walk, a count-valued jump, a custom jump operator carrying a map ``T`` --- all slot
    in as values of ``kind`` with their options in ``params``.

    ``kind == "metropolis"`` is the Metropolis-within-Gibbs sweep, and reads
    ``params["proposal"]``: ``"marginal"`` (learn the coordinate's marginal pmf during warmup and
    propose proportional to it) or ``None`` (the sweep's own uniform-over-the-others proposal).

    ``kind == "exact"`` is exact conditional Gibbs, which has no proposal and nothing to adapt, so
    it ignores ``params``.

    ``kind == "random_walk"`` is Metropolis with a two-sided geometric step, clamped at a bound ---
    the method for an ordinal or unbounded integer, and the only one for an open side. It reads
    ``params["adapt"]`` (default ``True``): adapt the step scale toward 1/3 acceptance during warmup;
    and ``params["init_log_scale"]`` (optional): the starting ``rho`` of the mean step ``1 + e^rho``,
    a scalar or one entry per coordinate. With evidence the factory sets it from each coordinate's
    interquartile range (``2/p = IQR``); without, the walk starts at ``rho = 0``.

    Both live **per parameter**, which is the whole point: the proposal used to be one value for
    the whole model, so the *widest* parameter decided for every other one::

        spec.discrete[i].kind = "exact"
        spec.discrete[i].params = {}
    """

    name: str                              #: the discrete parameter this describes
    n_values: int | None                   #: its support width (``None``: an open side)
    kind: str = "metropolis"               #: "metropolis" | "exact" | "random_walk"
    params: dict = field(default_factory=dict)   #: kind-specific options (see above)
    ordinal: bool = False                  #: declared or implied ordinal, for display

    def __str__(self) -> str:
        width = "unbounded" if self.n_values is None else f"{self.n_values} values"
        out = f"{self.name}[{width}{',ordinal' if self.ordinal and self.n_values else ''},{self.kind}]"
        if self.kind == "metropolis":
            out += f" ({self.params.get('proposal') or 'uniform'})"
        elif self.kind == "random_walk":
            out += " (adapted" if self.params.get("adapt", True) else " (fixed scale"
            out += ", scale from evidence)" if "init_log_scale" in self.params else ")"
        return out


@dataclass
class SamplerSpec:
    """A mutable, inspectable prototype carrying everything needed to build a sampler."""

    #: the model this spec was analyzed from.
    model: object

    #: ``"nuts"`` | ``"hmc"`` | ``"randomized_hmc"``, each with a parallel-tempered counterpart
    #: ``"pt_nuts"`` | ``"pt_hmc"`` | ``"pt_randomized_hmc"`` (doc 13). A tempered base runs the
    #: same algorithm over the K-fold product space and keeps the cold chain, so everything
    #: downstream --- blocks, integrator, adaptations --- is unchanged; only ``tempering_params``
    #: is extra. No rule selects one yet: it is an explicit choice.
    base: str = "nuts"

    #: ladder options for a ``"pt_"`` base: ``{"n_temperatures", "betas", "beta_min", "tempered",
    #: "adapt_ladder", "adapt_beta_min", "swap_target_accept"}``. Ignored (and rejected) for an
    #: untempered base, so a typo cannot pass silently.
    tempering_params: dict = field(default_factory=dict)

    #: the coordinate space, one :class:`BlockSpec` per block.
    blocks: list[BlockSpec] = field(default_factory=list)

    #: ``"leapfrog"`` | ``"multirate"`` (RESPA over the model's cheap/expensive components) |
    #: ``"line_search"`` | ``"markovian_line_search"`` (WALNUTS).
    integrator: str = "leapfrog"

    #: the integrator's own options --- **not** ``algo_kwargs``, which is splatted into the
    #: sampler constructor. ``{"n": 4}`` for ``"multirate"``; ``{"base", "base_params",
    #: "schedule", "error_thresholds", "p"}`` for the line-search variants, whose ``"base"`` may
    #: itself be ``"multirate"``. Unknown keys raise.
    integrator_params: dict = field(default_factory=dict)

    #: initial leapfrog step size.
    step_size: float = 0.5

    #: adapt the step size (Robbins--Monro, or the line-search proxy variant under a line-search
    #: integrator --- a *swap*, since one subclasses the other).
    adapt_step_size: bool = True

    #: which mass adaptation fits the *quadratic* blocks: ``"score"`` (default,
    #: :class:`~mimcs.adaptation.ScoreMassAdaptation` --- SGD on a KL objective against the score
    #: covariance, writing from the first warmup step) | ``"covariance"``
    #: (:class:`~mimcs.adaptation.MassMatrixAdaptation` --- the empirical covariance of the
    #: positions, standard Stan-style) | ``None`` (identity mass, no adaptation).
    #:
    #: Three things to know about ``"covariance"``: it is a *partial* swap --- ``lowrank`` and
    #: ``learned_metric`` blocks are untouched and stay score-driven; it writes nothing for the
    #: first ``mass_min_samples`` (50) draws, so a shorter warmup silently leaves the mass at
    #: identity; and it reads ``mass_polyak``, which means the opposite thing to each mixin (an
    #: EMA of the SGD iterate for ``"score"``, a suffix average that *biases* the RM covariance
    #: here --- see :mod:`mimcs.adaptation.mass`).
    mass_adapt: str | None = "score"

    #: :class:`~mimcs.adaptation.RobustCenteringAdaptation` (opt-in, off by default: it only acts
    #: on ``centered=True`` params, and was measured to destabilize a fragile far-from-mode
    #: adaptation).
    centering: bool = False

    #: one :class:`DiscreteSpec` per integer parameter, in the model's declaration order ---
    #: how each one gets moved. Empty for a model with no integer parameters.
    #:
    #: This replaces a single model-wide ``discrete_proposal`` string. That field could only say
    #: one thing about every parameter at once, which is why ``discrete_update_rule``'s
    #: predecessor had to let the **widest** support decide for the whole model: the adaptation
    #: allocated every table together or none. Both halves are now per parameter.
    #:
    #: The **sweep itself is not a choice**: a model with integer parameters always gets it, since
    #: the alternative is a sampler that holds the labels frozen, which reports a perfect ESS and
    #: R-hat 1.000 while being arbitrarily wrong.
    discrete: list = field(default_factory=list)

    #: the order the discrete coordinates are visited in: ``"systematic"`` (default; every
    #: coordinate in declaration order) | ``"random"`` (each jump a uniformly chosen coordinate,
    #: ``algo_kwargs["discrete_jumps"]`` of them, default one per coordinate). No rule selects the
    #: random scan yet --- it waits for the blocked updates it is the base of.
    discrete_scan: str = "systematic"

    #: end warmup on a mixing criterion: ``"classifier"`` (the default) | ``"rhat"`` | ``None``
    #: (off). A criterion makes ``warmup(n)``'s ``n`` an upper bound and lets ``warmup()`` (no
    #: ``n``) run to the criterion or the mixin's ``max_warmup`` (set via ``algo_kwargs``).
    terminate: str | None = "classifier"

    #: an *input* to the block-partition rule rather than one of its decisions: a list of tuples
    #: of parameter names, each of which becomes one block, with any parameter left unnamed
    #: partitioned by the usual size heuristic. Set by ``analyze(model, blocks=...)``, which
    #: normalizes and validates it. Only the *grouping* is fixed; the refinement rules still pick
    #: each block's kind.
    block_override: list | None = None

    #: everything splatted into the sampler constructor --- ``target_accept``,
    #: ``max_tree_depth``, ``max_warmup``, and 80 more. See ``docs/reference/algo_kwargs.md``;
    #: unknown keys are silently ignored.
    algo_kwargs: dict = field(default_factory=dict)

    #: human-readable record of how the spec was decided, one line per arbitrated slot.
    rationale: list[str] = field(default_factory=list)

    #: the :class:`~mimcs.factory.Evidence` the spec was analyzed from.
    evidence: object = None

    def __str__(self) -> str:
        """A readable one-screen summary --- what was decided, not how it is stored.

        The default dataclass ``repr`` is unusable for the models this factory exists for: it
        inlines the evidence arrays and every fitted metric parameter, running to 100+ lines for a
        horseshoe. ``repr`` is left alone for debugging; this is what ``print(spec)`` gives.
        """
        blocks = ", ".join(str(b) for b in self.blocks) or "(none)"
        lines = [f"SamplerSpec: {self.base} over {len(self.blocks)} block(s) [{blocks}]"]
        # A static base has no Hamiltonian, so printing an integrator, a step size and a mass for
        # it would describe machinery that is not there --- and "step size 0.5 (fixed)" reads as a
        # value in use rather than one that is never consulted.
        if self.base != "static":
            integ = self.integrator + (f" {self.integrator_params}"
                                       if self.integrator_params else "")
            lines.append(f"  integrator     {integ}")
            lines.append(f"  step size      {self.step_size:g}"
                         f" ({'adapted' if self.adapt_step_size else 'fixed'})")
            lines.append(f"  mass           {self.mass_adapt or 'none (identity)'}")
        if getattr(self.model, "discrete_dim", 0):
            lines.append("  discrete       "
                         + (", ".join(str(d) for d in self.discrete) or "(none)")
                         + (f"; {self.discrete_scan} scan" if self.discrete_scan != "systematic"
                            else ""))
        lines.append(f"  terminate      {self.terminate or 'off'}")
        if self.centering:
            lines.append("  centering      on")
        if self.tempering_params:
            lines.append(f"  tempering      {self.tempering_params}")
        if self.block_override is not None:
            lines.append(f"  block override {self.block_override}")
        if self.algo_kwargs:
            lines.append(f"  algo_kwargs    {self.algo_kwargs}")
        ev = self.evidence
        present = {n: getattr(ev, n, None) for n in ("samples", "coordinates", "gradients")}
        have = {n: v for n, v in present.items() if v is not None and len(v)}
        if have:
            # Any of the three may be absent on its own -- a spec can carry gradients without
            # samples -- so take the row count from whichever is there.
            rows = len(next(iter(have.values())))
            lines.append(f"  evidence       {rows} row(s) of {'/'.join(have)}")
        if self.rationale:
            lines.append(f"  rationale      {len(self.rationale)} line(s)"
                         r" --- print(*spec.rationale, sep=chr(10))")
        return "\n".join(lines)

    def build(self, *, seed: int = 0, init=None, buffer_size=None):
        """Instantiate the sampler this spec describes.

        ``buffer_size`` sizes the RNG buffer (:class:`~mimcs.rng.RNGBuffer`); ``None`` leaves it to
        ``algo_kwargs`` or the default. Note it is **not** stream-neutral --- see ``build_sampler``.
        """
        from .build import build_sampler
        return build_sampler(self, seed=seed, init=init, buffer_size=buffer_size)


def default_spec(model, evidence=None) -> SamplerSpec:
    """The baseline spec: NUTS + one diagonal whole-space block + the default adaptations.

    A model with **no continuous parameters** gets no block at all rather than a degenerate
    ``(0, 0)`` one --- there is nothing for a kinetic to act on, and an empty block would lower to
    a zero-width mass matrix. ``discrete_only_base_rule`` then swaps the base for
    :class:`~mimcs.samplers.StaticContinuous`; this only makes ``default_spec(model).build()``
    work without going through ``analyze``.
    """
    blocks = [] if model.coord_dim == 0 else [
        BlockSpec(names=[p.name for p in model.parameters],
                  coord_slices=[(0, model.coord_dim)], kind="diagonal")]
    # Every discrete parameter on the Metropolis sweep with a learned marginal --- the library's
    # behaviour before per-parameter methods existed. `discrete_update_rule` revises it.
    # An open side, or an ordinal declaration over 3+ values, starts on the random walk instead:
    # nothing else can move the former, and `discrete_update_rule` would choose it for the latter.
    from ..samplers.discrete_updates import RW_MIN_VALUES
    discrete = []
    for p in getattr(model, "discrete_parameters", ()):
        n = getattr(p, "n_values", None)            # None: an open side
        walk = n is None or (getattr(p, "ordinal", False) and n >= RW_MIN_VALUES)
        discrete.append(DiscreteSpec(
            name=p.name, n_values=n, ordinal=bool(getattr(p, "ordinal", False)),
            kind="random_walk" if walk else "metropolis",
            params={"adapt": True} if walk else {"proposal": "marginal"}))
    return SamplerSpec(
        model=model, base="nuts", blocks=blocks, discrete=discrete, integrator="leapfrog",
        step_size=0.5, adapt_step_size=True, mass_adapt="score", centering=False,
        terminate="classifier", evidence=evidence,
        rationale=["default: NUTS + score-covariance mass + Robbins--Monro step size "
                   "+ classifier warmup termination"])
