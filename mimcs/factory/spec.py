"""The sampler *prototype*: :class:`SamplerSpec` and :class:`BlockSpec`.

A ``SamplerSpec`` carries every decision needed to construct a sampler, in attributes a
user can inspect and mutate before building --- the configuration seam of the factory (no
``**kwargs`` soup). The coordinate space is always modelled as a list of ``BlockSpec``.
``default_spec`` starts with a single whole-space block, but the block-partition rule replaces it
immediately, and each block carries its own kinetic kind --- diagonal, dense, low-rank, or a
learned position-dependent metric.

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
        integ = self.integrator + (f" {self.integrator_params}" if self.integrator_params else "")
        lines.append(f"  integrator     {integ}")
        lines.append(f"  step size      {self.step_size:g}"
                     f" ({'adapted' if self.adapt_step_size else 'fixed'})")
        lines.append(f"  mass           {self.mass_adapt or 'none (identity)'}")
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
    """The baseline spec: NUTS + one diagonal whole-space block + the default adaptations."""
    block = BlockSpec(names=[p.name for p in model.parameters],
                      coord_slices=[(0, model.coord_dim)], kind="diagonal")
    return SamplerSpec(
        model=model, base="nuts", blocks=[block], integrator="leapfrog",
        step_size=0.5, adapt_step_size=True, mass_adapt="score", centering=False,
        terminate="classifier", evidence=evidence,
        rationale=["default: NUTS + score-covariance mass + Robbins--Monro step size "
                   "+ classifier warmup termination"])
