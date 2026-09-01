# mimcs: Sampler factory

A user-facing front door for *constructing samplers*. Today a sampler is assembled by hand —
pick a base algorithm, a kinetic, an integrator, and a stack of adaptation mixins, then wire
them together (see the builders in `mimcs/testing/runner.py`). The sampler factory turns that
into one call:

```python
sampler = make_sampler(model)                 # a sensible default
sampler = make_sampler(model, earlier_output)  # tailored to that distribution
```

and — crucially — exposes everything in between so an advanced user can inspect and override
individual choices instead of accepting the automatic ones. This document specifies the
interface and the stage-1 implementation. It builds directly on the
[HMC](06_hamiltonian_monte_carlo.md) and [Riemannian HMC](07_riemannian_hmc.md) machinery and
targets the [`Model`](05_model_interface.md) backend.

## Goals and framing

The factory has two jobs, in priority order:

1. **Get the interface right.** It is meant to be *the* primary way users (and we ourselves,
   while building test problems and doing adaptation research) reach for a sampler. The shape
   of the call, and the object a user manipulates to customize a sampler, are the part that is
   expensive to change later — so they are the part to design carefully now.
2. **Choose well, heuristically.** Given earlier sampling results, pick a sampler likely to do
   well on that distribution. The heuristics are a deliberately open-ended *grab-bag* — they
   will grow over time and are individually replaceable. The interface is designed so they can
   be added, reordered, and overridden without disturbing anything else.

Because (2) is open-ended, the design is **staged**: a stage-1 skeleton (this document's main
subject) that nails the interface and ships one real heuristic, built so later stages slot in
without an interface change. Later-stage features are flagged **[stage 2+]** and gathered in
[Staging plan](#staging-plan).

## What the factory produces (the backend contract)

The factory's only job is to produce an instantiated sampler — the same kind of object the
`runner.py` builders produce today. Every existing builder follows one assembly pattern:

```python
Cls       = make_sampler_class(*adaptation_mixins, BaseAlgorithm)  # mimcs/samplers/base.py
kinetic   = ...                          # a KineticHamiltonian (mimcs/hmc/hamiltonians.py)
potentials = default_potentials(model)    # mimcs/hmc/samplers.py
integrator = leapfrog(potentials, kinetic)
sampler    = Cls(model, init_position, seed=seed,
                 kinetic=kinetic, potentials=potentials, integrator=integrator,
                 step_size=..., **algo_kwargs)
```

The factory is, at heart, a structured way to *decide the blanks in that template* and then
fill them in. Two facts about the backend shape the design:

- **The `Model` is required to build anything.** It supplies the coordinate dimension and
  per-parameter coordinate blocks (`model.parameters`, `model.coord_block(name)`,
  `model.coord_dim`, `model.ambient_dim`), the potentials (`default_potentials(model)`), and
  the natural default initial position (`np.zeros(model.ambient_dim)`). So the model is a
  **required first argument**, never inferred.
- **`BaseHMC` holds a single `kinetic`.** Block-wise kinetics (different kinetic per
  coordinate block) are therefore realized **[stage 2+]** by a *composite* kinetic that itself
  implements the `KineticHamiltonian` interface and dispatches to per-block sub-kinetics —
  *not* by rewriting `BaseHMC`. The interface models blocks from the start (below) so this is
  a drop-in, not a redesign.

## The call

```python
make_sampler(model, *results, seed=0, init=None, buffer_size=None, recompute_gradients=True) -> sampler
```

`*results` is *earlier sampling results*, and flexibility here is a first-class requirement.
It accepts, in any combination:

- nothing — return a reasonable default sampler;
- raw posterior **samples** (an `np.ndarray` of shape `(n, ambient_dim)` — the factory works
  in the flat layout, so it reads `get_samples_flat()`, not the by-name `get_samples()`);
- a **`SamplerOutput`** (`mimcs/testing/runner.py`) — samples plus diagnostics;
- a **live sampler** (a `BaseSampler` instance) — samples and diagnostics, plus the coordinates
  (recomputed) and scores (saved by the sampler, or recomputed) of its draws, so a prior run
  can drive the metric-regression rule;
- an explicit **bundle** — a `(samples, coordinates, gradients)` tuple or a dict of those —
  for callers who computed them outside a `mimcs` run.

`make_sampler` is a thin convenience wrapper over the two-stage pipeline that does the real
work, which is also exposed:

```python
spec    = analyze(model, *results)   # -> SamplerSpec  (a mutable prototype)
sampler = spec.build(seed=0, init=None, buffer_size=None)
```

`make_sampler(model, *results)` is exactly `analyze(model, *results).build(...)`. The point of
splitting them is the middle object.

## The pipeline

```
*results ──▶ Evidence ──▶ analyze ──▶ SamplerSpec ──▶ .build() ──▶ sampler
           (normalize)   (heuristics)  (prototype)    (lower)
```

Three layers, each independently usable and testable.

### 1. `Evidence` — normalize the heterogeneous input

The "flexibility is key" requirement is made concrete by funnelling every accepted input form
into one container by `isinstance`-dispatch:

| Input form | Populates |
|---|---|
| *(none)* | empty `Evidence` |
| `np.ndarray` | `samples` |
| `SamplerOutput` | `samples`; `diagnostics` (`accept_rate`, `ess`, `warmup_step_sizes`) |
| `BaseSampler` (live) | `samples` (`get_samples_flat()`); `coordinates` recomputed from the ambient samples (cheap); `gradients` from the sampler's **saved** scores (`get_gradients()`, on by default), else recomputed (a vmapped gradient pass, unless `recompute_gradients=False`) — so a prior run drives the metric-regression rule; `diagnostics` (`acceptance_rate()`, `warmup_step_sizes()`, and for NUTS `divergence_count()` / `divergence_rate()` / `mean_tree_depth()`); **validates `sampler.model` matches `model`** |
| `(samples, coordinates, gradients[, discrete])` tuple / dict | the named fields (a 3-tuple still means what it did) |

```python
@dataclass
class Evidence:
    samples: np.ndarray | None = None       # (n, ambient_dim)
    coordinates: np.ndarray | None = None    # (n, coord_dim)
    gradients: np.ndarray | None = None      # (n, coord_dim) — per-iteration score
    discrete: np.ndarray | None = None       # (n, discrete_dim) — integer labels (doc 14)
    diagnostics: Diagnostics | None = None

@dataclass
class Diagnostics:
    divergence_count: int | None = None
    divergence_rate: float | None = None
    mean_tree_depth: float | None = None
    accept_rate: float | None = None
    ess: np.ndarray | None = None
    warmup_step_sizes: np.ndarray | None = None
```

Any field may be `None`; heuristics check for what they need and otherwise fall through.
Multiple results merge into one `Evidence` (e.g. several chains' samples concatenated).

### 2. `SamplerSpec` — the inspectable, mutable prototype

This is the object the user described wanting: it *carries all the information needed to make a
sampler*, and its attributes can be examined and changed before the sampler is instantiated —
**not** a `make_sampler(model, *results, **kwargs)` where keyword soup carries the complexity.

```python
@dataclass
class SamplerSpec:
    base: str = "nuts"                  # "nuts" | "hmc" | "randomized_hmc" | "static" (the
                                        # discrete-only base; doc 14), the first three each with a
                                        # parallel-tempered counterpart "pt_nuts" | "pt_hmc" |
                                        # "pt_randomized_hmc" (doc 13). No rule selects one yet
                                        # -- tempering is an explicit choice.
    tempering_params: dict = ...        # ladder options for a "pt_" base: n_temperatures,
                                        # betas / beta_min, tempered, adapt_ladder, ...
    blocks: list[BlockSpec] = ...       # one or more coordinate blocks (see below)
    integrator: str = "leapfrog"        # "leapfrog" | "multirate" | "line_search"
                                        # | "markovian_line_search"
    integrator_params: dict = ...       # the integrator's own options -- NOT algo_kwargs,
                                        # which is splatted into the sampler constructor
    step_size: float = 0.5
    adapt_step_size: bool = True        # Robbins–Monro
    mass_adapt: str | None = "score"    # which mass adaptation fits the quadratic blocks:
                                        # "score" (default) | "covariance" | None (identity)
    centering: bool = False             # RobustCenteringAdaptation (opt-in, off by default)
    terminate: str | None = "classifier"  # end warmup on a mixing criterion (doc 10):
                                        # "classifier" (default) | "rhat" | None (off)
    discrete_proposal: str | None = "marginal"  # the discrete sweep's proposal, for a model with
                                        # integer parameters: "marginal" (default) | None (the
                                        # uniform placeholder). The *sweep* is not optional.
    algo_kwargs: dict = ...             # n_leapfrog, max_tree_depth, max_warmup, ...
    rationale: list[str] = ...          # human-readable: which heuristic set what, and why

@dataclass
class BlockSpec:
    names: list[str]                    # parameter names in this block
    coord_slices: list                  # list of (start, stop) slices, possibly non-contiguous
    kind: str = "diagonal"             # "diagonal" | "dense" | "lowrank" | "learned_metric"
                                       # | [stage 2+] "relativistic"
    params: dict = ...                  # kind-specific: init_mass; metric/metric_init; [stage 2+] light_speed
```

The single most important structural decision: **the coordinate space is always modelled as a
list of `BlockSpec`** — even the default is *one* block spanning the whole space. Block-wise
kinetics then become "more than one block" rather than a different code path, and nothing about
the interface changes when they arrive **[stage 2+]**.

A `BlockSpec.kind` maps to a concrete kinetic and its `mass_mode`:

| `kind` | kinetic | `mass_mode` | mass adaptation |
|---|---|---|---|
| `diagonal` | `DiagonalQuadraticKinetic` (over the block's slice) | `"diagonal"` | per `mass_adapt`† |
| `dense` | `DenseQuadraticKinetic` (over the block's slice) | `"dense"` | per `mass_adapt`† |
| `lowrank` | `LowRankQuadraticKinetic` (diagonal-whitened rank-J, `params["rank"]`) | `None` | `LowRankAdaptation` |
| `relativistic` **[stage 2+]** | `RelativisticKinetic` | `None` | `RelativisticMassAdaptation` |
| `learned_metric` | `LearnedDiagonalBlock` (mini-language `params["metric"]`) | `None` | `MetricAdaptation` |

† Only the two quadratic kinds. `mass_adapt` does **not** govern `lowrank`, `relativistic` or
`learned_metric`, whose adaptations filter on `mass_mode is None` and are orthogonal to it.

Each block is one slice-aware kinetic in `BaseHMC`'s kinetics list (doc 06); the chosen mass
adaptation is list-aware and fits every diagonal/dense block over that block's own coordinate
slice. **`spec.mass_adapt`** selects it: `"score"` (default — `ScoreMassAdaptation`, SGD on a KL
objective against the score covariance, writing from the first warmup step), `"covariance"`
(`MassMatrixAdaptation`, the standard empirical covariance of the positions), or `None` (identity
mass, no adaptation — the kinetic's own `initial_mass_params` still populate `ham_params`, so the
sampler is a fixed-metric one rather than broken). The two share the block filter
`mass_mode in ("diagonal", "dense")`, so the choice is a swap and never a stack.

Three things about `"covariance"` that the field's comment also records: it is a **partial** swap
(a mixed partition runs it beside the score-driven `LowRankAdaptation`/`MetricAdaptation`); it
writes **nothing** for the first `mass_min_samples` (50) draws, so a warmup shorter than that
silently leaves the mass at identity — safe at the default `min_warmup` of 500, a trap below it;
and it reads `mass_polyak`, which means the opposite thing to each mixin — an EMA of the SGD
iterate under `"score"`, and a suffix average that *biases* the Robbins–Monro covariance here.

Note also that the `mass_mode_rule` picks each block's *kind* from the score covariance's
spectrum. Under `"covariance"` the storage is therefore still chosen from scores while the mass
fitted into it comes from positions; for a Gaussian the two agree, but the rule is no longer a
statement about the estimator that will run.

**The default spec** (empty `Evidence`, before rules run) is `base="nuts"`, one whole-space
block, `adapt_step_size=True`, `terminate="classifier"`, and `centering=False`; the block-partition
rule then sets the actual blocks, and `build` composes **`ClassifierTermination` + NUTS + the
per-block kinetics + `ScoreMassAdaptation` + `RobbinsMonroStepSize`**.

**Centering is opt-in** (`centering=False` by default). `RobustCenteringAdaptation` acts only on
`centered=True` parameters (which the DSL emits only when a `ModelSpec` asks for it, doc 08, so
it is a no-op on a compiled model unless one did), and where it *does* act — a model built with
centered charts — it was measured to badly
destabilize a fragile far-from-mode adaptation (the stiff `poissonrandom` mu*=10: split-R̂ ~1.5→2.5,
median ESS/n ~1.0→0.15, Stein-flagged features ~3→62 of 102) while only matching the plain default
on an easy correlated Gaussian. Two sample-driven adaptations (centering's median/MAD standardization
and the score mass) co-adapting on a moving far-from-mode chain fight each other. It is therefore
kept off pending the mass-matrix work; enable per-spec with `centering=True`.

**Warmup termination** (doc 10) is on by default. `terminate` selects the criterion —
`"classifier"` (a logistic regression on the draws' features; the default, still under study),
`"rhat"` (`GelmanRubinTermination`, max split-R̂ < 1.01), or `None` to disable. `build` prepends the
chosen mixin *outermost*, since it only observes the drawn sample after every adaptation has run.
With a criterion, `warmup(n)`'s `n` is an upper bound and `warmup()` (no `n`) runs to the criterion
or the mixin's `max_warmup`; the criterion's knobs (`max_warmup`, `accuracy_threshold`,
`rhat_threshold`, `min_warmup`, `check_every`, ...) pass through `algo_kwargs`. The mixin draws no
RNG, so a `warmup(n)` that does *not* trigger an early stop is bit-identical to `terminate=None`.

### 3. `analyze` and `build`

**`analyze(model, *results) -> SamplerSpec`** starts from the default spec for `model` and lets
the **heuristic rules** revise it. Rules do *not* mutate the spec directly; each one reads what
it needs from `evidence` (falling through if absent) and emits zero or more **weighted
proposals**:

```python
def rule(spec: SamplerSpec, evidence: Evidence, model: Model) -> list[Proposal]: ...

@dataclass
class Proposal:
    slot: str       # the decision being made, e.g. "blocks[0].kind", "base", "step_size"
    value: object   # the proposed value for that slot
    weight: float   # the rule's confidence in this proposal (higher wins)
    reason: str     # human-readable justification
    rule: str = ""  # rule name (filled in by the registry)
```

An **arbiter** then resolves the proposals into the spec (next section). Rules are small,
side-effect-free, and individually unit-testable — this *is* the grab-bag, now expressed as a
set of independent voters rather than an order-sensitive mutation chain. The registry is the
only thing that grows as heuristics accumulate.

#### Conflicting heuristics: weighted arbitration

Different rules will disagree — the dimension rule may want a `diagonal` block while a
covariance or regression rule wants `dense`; a divergence rule may want `relativistic` where
another wants `dense`. Resolution is explicit and does not depend on rule order:

- Proposals are grouped by `slot`. For each slot, the **highest-weight proposal wins** and its
  value is written into the spec. The spec's default value is the baseline (weight 0), so any
  proposal with positive weight overrides it.
- **Weight is per-proposal, not per-rule** — a rule expresses graded confidence per situation
  (e.g. the dimension rule is near-certain about `diagonal` in 10 000-d but only mildly
  confident about `dense` near the threshold; a covariance check emits a high weight for
  `dense` only when off-diagonal mass is large). This is what lets weak, situational signals
  yield to strong ones automatically.
- Ties break **deterministically** by registration order, so `analyze` is reproducible.
- **Every proposal — winners and the losers it beat — is recorded in `spec.rationale`**, so the
  user can see not just what was chosen but what competed and why, and override accordingly.

For the categorical decisions that dominate (`kind`, `base`) arbitration is a choice, not a
blend. Slot-specific **combiners** — e.g. blending a covariance-derived and a regression-derived
`init_mass` rather than picking one — are a deliberate **[stage 2+]** extension point: the
arbiter dispatches per slot, defaulting to argmax-weight, and a numeric slot can later register
a combiner. User customization stays a separate, always-final step: after `analyze` returns, the
user mutates the resulting `SamplerSpec` directly, which by construction overrides any
heuristic outcome.

**`SamplerSpec.build(seed, init, buffer_size) -> sampler`** lowers the spec onto the backend template:

1. assemble the mixin tuple (MRO order: the `terminate` mixin outermost if any, then
   `RobbinsMonroStepSize`, `ScoreMassAdaptation`, `RobustCenteringAdaptation`, subject to
   `terminate`/`adapt_step_size`/`centering`), and `Cls = make_sampler_class(*mixins, BASE[spec.base])`;
2. build **one kinetic per block** — `DiagonalQuadraticKinetic` / `DenseQuadraticKinetic` over
   the block's coordinate slice, id-keyed by the block's parameter names — giving a
   `kinetics` list (one element for a single whole-space block);
3. `potentials = default_potentials(model)`; the integrator comes from the `_INTEGRATOR` dispatch
   on `spec.integrator` (`"multirate"` splits the potentials by `Model.cheap_components`; the two
   line-search variants wrap a *base* integrator named by `integrator_params["base"]`, which may
   itself be `"multirate"`). It is built **before** the mixin list, because the step-size mixin
   depends on it: a proxy-emitting integrator gets `LineSearchStepSizeAdaptation` *in place of*
   `RobbinsMonroStepSize` (it subclasses it, so listing both is an MRO error). The default
   `"leapfrog"` is the unified leapfrog over the kinetics list (doc 06);
4. instantiate `Cls(model, init_position, seed=seed, kinetics=, potentials=, integrator=,
   step_size=spec.step_size, **spec.algo_kwargs)`, with `init_position = init`, else the last
   row of `evidence.samples`, else `np.zeros(model.ambient_dim)`.

The override seam falls out for free:

```python
spec = analyze(model, output)
spec.blocks[0].kind = "dense"      # force a dense mass matrix
spec.base = "hmc"; spec.algo_kwargs["n_leapfrog"] = 32
sampler = spec.build()
```

## Heuristic rules

### Block partition from `model.parameters` (the structural rule)

The main structural rule proposes the whole `blocks` slot from the model's parameter list,
now that `BaseHMC` holds a **list of kinetics** (doc 06) — a block-diagonal mass is just
several `Diagonal`/`Dense` kinetics, adapted per-block by the list-aware `ScoreMassAdaptation`.
Each block's coordinates are a **list of slices** (`BlockSpec.coord_slices`), possibly
**non-contiguous**, so a block can fuse parameters that are scattered across the coordinate
vector; a kinetic gathers/scatters its block via those slices (doc 06). With `FUSE_MIN_DIM = 20`,
`DENSE_MAX_DIM = 50`:

- a parameter of dimension in `(50, LOWRANK_MAX_DIM]` (`LOWRANK_MAX_DIM = 1000`) → its own
  **low-rank** block (`lowrank_block_rule`, a refinement rule; a diagonal scale plus `J = 8`
  learned correlation directions — cheaper than dense, better than diagonal); above
  `LOWRANK_MAX_DIM` it stays **diagonal**. *(The threshold and rank are placeholders pending
  evidence.)*
- a parameter of dimension in `(20, 50]` → its own **dense** block (already dense-worthy, no
  fusion);
- the low-dimensional (`≤ 20`) parameters → **fused** — in declaration order but regardless of
  coordinate adjacency — into **dense** blocks, each **capped at `20`**: a parameter joins the
  current block only if it fits within the cap, else the block is flushed and a new one begun
  (adjacent slices coalesce; scattered ones stay a multi-slice block).

The rationale is hierarchical models: the correlations that matter most for a mass matrix are
among the naturally few high-level parameters, which are often declared *separately* (a mean
and a variance parameter, possibly with lower-level parameters between them in the coordinate
vector) and so are individually low-dimensional — fusing them into a dense block captures those
cross-parameter correlations even when the parameters are not adjacent, while high-dimensional
parameters stay diagonal. (Sample-based per-block refinement — e.g. downgrading a dense block to
diagonal when the draws show no correlation — is a natural future addition.)

The fusion is **capped**, not grown just past the threshold, because a fused *multi*-parameter
block is permanently ineligible for a **learned metric** (`learned_metric_rule` fits only
single-parameter blocks). Growing a block past the cap can swallow a whole vector parameter — a
20-dim `a` glued onto two scalars becomes a 22-dim block — which can then never receive the
position-dependent (funnel-whitening) metric it needs. Empirically that metric is decisive: on the
2PL IRT posterior (`tests/problems/irt_2pl`, three hierarchical funnels) giving each vector its own
`Exp("sigma") + Exp()` metric took split-R̂ from ~2.4 to ~1.005, cut Stein-flagged features from
~48/288 to ~17 (≈ the 5% null), and lifted median ESS/n ~40× over the constant-mass default. Capping
trades a little cross-parameter correlation for keeping such parameters eligible — a worthwhile
trade on the funnels where it matters.

#### The caller's override

Size is a proxy, and on some models it groups badly: on the `hmm_gaussian` HMM
(`tests/problems/hmm_gaussian`) all four parameters — `pi1`, `A`, `mu`, `sigma` — are individually
small and fuse into one 14-d dense block, which rules out a learned metric for any of them. So
`analyze(model, blocks=[...])` (forwarded by `make_sampler`) lets the caller state the grouping
directly: each entry is a group of parameter names becoming one block, with a bare string as
shorthand for a one-name group.

Three properties make it an override rather than a replacement:

- **It is partial by default.** Parameters the caller does not name are partitioned by the rule
  above, running on exactly those parameters — naming part of a model is the normal case.
- **It fixes the grouping only.** The refinement rules still run on the result, so each block's
  kind is still chosen from the evidence. That is the point: splitting a fused block is precisely
  what makes a single parameter *eligible* for `learned_metric_rule`, which skips anything that is
  not a single contiguous parameter.
- **It is validated up front**, in `analyze` before any rule runs (`normalize_block_override`),
  so an unknown name, a name in two groups, or an empty group is a `ValueError` pointing at the
  call rather than an obscure failure inside arbitration. Groups are sorted into model
  declaration order, because `block.names` drives the coordinate order and the kinetic id — a
  forced block should be indistinguishable from one the rule could have produced.

The override reaches the rule as `spec.block_override`, an *input* to the partition rather than
one of its decisions; with it unset the rule is exactly as described above.

### Metric regression → learned metric (a refinement rule)

`learned_metric_rule` (in `rules.py`; module `factory/regression.py`) uses `evidence.coordinates`
and `evidence.gradients` to decide, per block, whether a *position-dependent* diagonal metric is
warranted. For each single (contiguous) parameter block it enumerates simple mass-matrix
mini-language candidates over the **other** blocks (constant `Exp()` baseline; per-dependency
log-linear `Exp(d)` and bounded `Exp()*Sigmoid(d)`; pairwise additive and joint) — dimension-
aware: only candidates within `20 · block_dim` parameters, capped at 50 regressions, simplest
first. When a dependency **shares the block's dimension**, *sparse* elementwise candidates
(`SpExp(d)`, `Exp()*SpSigmoid(d)`) are added — the bijective row correspondence
common between equal-dimension arrays (a horseshoe's per-element scale), and often the *only*
viable position-dependent form for equal large dimensions, since the dense `Exp(d)` there
(`block_dim·dep_dim` params) blows the budget while the sparse form (`≈ 2·block_dim`) fits.
(Triggering on the total-dimension match alone needs no shape inference: a coincidental match
with incompatible shapes just fits poorly and is AIC-rejected.)

Every position-dependent form is enumerated **twice**: bare (`SpExp(d)`) and with an additive
constant floor (`SpExp(d) + Exp()`), bare first as the cheaper of the pair. Until 2026-08 the
floor was hard-coded onto every candidate, so the simpler form was not merely disfavoured but
*absent from the pool* — the direct reason `reg_horseshoe` selection always returned
`SpExp('lambda') + Exp()`. The floor is a real term on many targets (a hierarchical scale plus a
data-driven likelihood curvature), but where the truth has none it is a spare parameter with
nowhere to go: zeroing it needs its bias to run to `-∞`, which neither a capped L-BFGS nor a
Robbins–Monro warmup reliably reaches, so it inflates the fit exactly where the true metric is
smallest. The AIC arithmetic is a fair fight either way — the pair differs by one bias **per
coordinate**, and the loss is a sum over coordinates, so the floor pays for itself iff it improves
the *per-coordinate* KL loss by more than `1/N` nats. Each is fitted by minimising
the **batch KL loss** `mean_n ½ Σ_d (log M_d + g²_d/M_d)`
(the same objective `MetricAdaptation` descends online, minimiser `E[g²|q_{-i}]`) with the
L-BFGS in `mimcs.optim`, and the fits are ranked by **AIC** (`2k + 2N·mean_loss`). When the best
candidate is genuinely position-dependent and beats the constant baseline by a margin, the block
becomes `learned_metric`, its chosen expression and fitted parameters stashed in `block.params`
(`{"metric": …, "metric_init": …}`) — inspectable and hand-overridable on the spec. `build`
lowers it to a `LearnedDiagonalBlock` and adds `MetricAdaptation` (which partitions the kinetics
with `ScoreMassAdaptation` — learned blocks by `is_learned`, quadratic ones by `mass_mode`).

This is a **refinement** rule: it must read the *already-partitioned* `spec.blocks`, so `analyze`
runs two arbitration passes — the structural `RULES` (partition) first, then the
`REFINEMENT_RULES` against the final partition. It is inert without gradient evidence, so
evidence-free factory calls are unchanged. (AIC is a pragmatic first criterion — goodness-of-fit
vs. parameter count only. A known gap for later: a mass that departs *exponentially* from a good
value costs sampling efficiency exponentially, so a poorly-fit unbounded `Exp("x")` is far
costlier than a bounded gated form with a few more parameters; a cost-aware criterion should
eventually fold that in.)

#### Scale-aware initialisation (load-bearing, not a nicety)

Each fit starts from `expr.init_params(block_dim, dep_dims, target=…)`, and the regression passes
**`target` = the per-coordinate empirical second moment `mean_n g²_{n,d}`** — the exact minimiser
of the constant candidate — rather than the default `target = 1` (`M = I`). This is required for
correctness, not speed. The per-coordinate loss

    f(b) = ½ (b + ḡ² e^{−b}),   minimiser b* = log ḡ²

is **exponentially steep below `b*` and almost exactly linear with slope ½ above it**. Started at
`b = 0` on a target whose scores are ~1e5, the initial gradient is ≈ −½·1e5, so L-BFGS's first
line-search step is enormous — and is *correctly* Armijo-accepted, because the loss genuinely drops
when it lands in the flat region. From there the gradient is ½, and crawling back ~1e4 units is far
beyond `max_iter`.

Measured on `tests/problems/diamonds` (N=5000, `b` 24-d, true `log M ∈ [9.4, 12.7]`), with
`target = 1`:

| candidate | k | AIC (`target=1`) | AIC (scale-aware) | max\|param\| |
|---|---|---|---|---|
| `Exp()` (constant baseline) | 24 | 5.59e7 | **282,797** | 12.74 |
| `Exp(dep)+Exp()` | 96 | 425,933 | 282,917 | 10.25 |
| `Exp()*Sigmoid(dep)+Exp()` | 120 | 423,292 | 282,989 | 12.05 |

The dependency-*free* baseline fitted to `b ∈ [369.6, 10694]` instead of ≈ 11 (its final loss
27943.4 ≈ ½·Σb — the `ḡ²/M` term had wholly underflowed, i.e. the fit was sitting in the linear
regime). Two symptoms followed, and both are cured by the init:

1. **Unusable coefficients.** The selected metric put `log M` at 83.95 against a true 12.74 — `M =
   2.9e36`, with 4.77 log-units of headroom before float32 `exp` overflows at 88.72. Slightly
   different evidence tips it over: `M = inf` → `nan` in the first adaptation step.
2. **A spurious learned metric.** Because *the baseline every candidate is judged against* was the
   worst-fitted of all, any position-dependent form won by ~130×. With the scale-aware init all
   three reach the same loss (141.363–141.375) — the position-dependent terms buy nothing — so AIC
   picks the constant and diamonds correctly gets **no learned metric**.

Note this is a *scaling* failure, not a collinearity one: `Exp()` has no predictor at all. (Ridge
was tried and rejected — it leaves the overshoot untouched and shrinks `b` toward 0 when the truth
is ~11; at λ=1 the fitted `b` still reached 334 and the raw loss was `inf`.)

#### Non-finite guard

Two backstops, since a log-linear metric can carry *finite* parameters that still exponentiate to
`inf`:

* `regression.fit_is_usable` rejects a fitted candidate whose loss or parameters are non-finite, or
  whose **metric `M` is not finite and positive at the evidence**; `select_metric` scores it
  `AIC = inf`, ranking it last rather than dropping it (the constant baseline must stay available).
* `MetricAdaptation` skips any online step whose KL-loss gradient is non-finite, keeping the last
  finite parameters and counting the skip in `metric_nonfinite_count()`. A run therefore degrades
  instead of dying, and a persistently rising count says the *initial* metric was never usable.

**Iterating the loop sharpens the metric (bootstrapping).** The rule's selection is only as good as
the evidence, and the evidence is only as good as the sampler that produced it — so a single default
pilot on a hard target can select a *rough* metric, but feeding the improved draws back closes a
convergent loop. Measured on the 2PL IRT posterior (`tests/experiments/irt_metric.py --chain`, x64,
4 pilot seeds): a constant-mass default pilot mixes badly (split-R̂ ~1.9–2.7). Its draws are still
enough for `analyze` to propose a learned metric for every vector — but the *selection* is
seed-unstable where a dependency is weakly identified (the item difficulty `a`, whose dominant
curvature term `aᵢ²(θ−b)²` is not mini-language-representable — `aᵢ` is `a`'s own coordinate, `θ` is
over budget — so only its *subdominant* prior funnel `σ_a` is fittable, and under noisy evidence it
loses to spurious candidates). Sampling with that first metric already jumps R̂ to ~1.01 and cuts
Stein flags ~48→~15; re-`analyze`-ing on **those** cleaner draws then selects an identical, stable
metric across all seeds — `a` lands on its `σ_a` funnel every time — and the third-round sampler
matches or **beats** a hand-tuned metric (R̂ ~1.005, median ESS/n ~0.4–0.7 vs the hand-tuned ~0.34,
Stein flags at the ~5% null). It beats the hand-tuned one because the regression finds a real
dependency a modeller may miss: item `bᵢ`'s curvature scales with `aᵢ²` (the likelihood
`σ(aᵢ(θ−bᵢ))`), which it reliably picks as the *sparse* `SpExp('a')` — a stronger effect than the
prior funnel one would code by hand. The instability was thus **evidence-limited, not
representability-limited**: better draws, not a richer mini-language, are what stabilise it.
(Automating this iteration — pilot → refit → resample until the selection stops changing — is a
natural future rule; today it is driven by hand by passing a run back to `analyze`.)

#### Discrete dependencies

A metric may depend on the model's **integer** parameters as well as its coordinates, fitting
`E[g_i g_iᵀ | q_{-i}, z]` (doc 07). A discrete parameter is deliberately in no `BlockSpec` — it has
no coordinate and no kinetic — so it cannot arrive through the block loop the continuous
dependencies come from; `learned_metric_rule` enumerates it separately from
`model.discrete_parameters` and slices it from `Evidence.discrete`. Inert when the evidence carries
no labels, which is every continuous model.

Two codings, and the distinction only matters above two values. An **ordinal** label is
standardized by its *declared support* — not by the observed draws, because the transform must be
defined with no evidence at all (`MetricAdaptation` starts cold, and a hand-written metric never
sees a pilot); a metric that meant one thing fitted and another built cold would be a bad kind of
surprise. A **categorical** label is reference-coded into `k−1` indicators: full one-hot sums to
one and so duplicates the additive `+ Exp()`, leaving a direction the optimizer cannot resolve and
AIC still charges for. For `k = 2` the two coincide, so only one is offered.

Selection is **two-pass**. Pass 1 is the existing pool plus discrete-only forms. Pass 2 takes the
AIC-best *continuous* candidate and multiplies each discrete factor onto it —
`best * (Exp(ordinal=["z"]) + Exp())` — which is what "labels modulate the metric
multiplicatively" means operationally, and costs a handful of extra fits rather than a pool
multiplied by the number of labels. The risk in pass 2 is the obvious one, that a spurious factor
rides along on an already-good fit; AIC charges it for its parameters, and the test that decides
this is a target where the labels genuinely do not affect the block's scale, on which the constant
baseline must survive.

The sparse (elementwise) gate reads the **encoded** width, not the label count. A binary indicator
of the block's own length is the case that matters — spike-and-slab's `z_j` for `beta_j` — and
encodes to width `n`; a 3-level categorical over the same coordinates encodes to `2n` and is not an
elementwise correspondence.

### Evidence-informed mass mode (a refinement rule)

`mass_mode_rule` (in `rules.py`; module `factory/mode_select.py`) replaces the dimension-count
placeholder (`lowrank_block_rule`) *when there is gradient evidence*: it reads a block's mass mode —
**diagonal / low-rank(J) / dense** — off the data. The per-iteration score covariance `E[g g^T]` **is**
the target precision (the ideal mass); the decision is a property of its **whitened** spectrum
`R = D^{-1/2} S D^{-1/2}` (`D = diag(S)`), a sample correlation (unit diagonal, mean eigenvalue 1):
`R ~ I` → diagonal; a few eigenvalues isolated above the Marchenko–Pastur bulk edge
`(1+√(d/n))²` → low-rank, J = the spike count (Baik–Ben Arous–Péché: a true spike pops above the edge
only past the detectability threshold); a broadly-structured bulk → dense. It is `(d,n)`-aware through
`γ = d/n` and handles `d > n` (R singular → dense excluded; nonzero eigenvalues via the cheaper `n×n`
Gram). Weight 0.65: above the placeholder, below `learned_metric_rule`, so a learned-metric block
keeps its metric.

`select_mass_mode` provides three backends, **compared on synthetic spiked data with known ground
truth** (`tests/experiments/mode_select_study.py`): (**aic**, default) an AIC-penalised KL over the
low-rank order J — `AIC(J)-AIC(0) = penalty·(params_lr(J)-d) - 2n·Σ_{k≤J} ½(λ_k-1-log λ_k)`, whose
argmin is the number of directions worth a rank-one term (the MP bulk fails the marginal penalty, so
J is not inflated; dense is the fallback when that count exceeds `J_max`, *not* a raw
dense-vs-identity comparison, which would overfit the bulk); (**mp**) the analytic MP/BBP edge; and
(**parallel**) Horn's permutation-null parallel analysis. In the study **aic and mp agree almost
everywhere** — isotropic/wide-*diagonal* → diagonal (the diagonal `D` absorbs a wide scale, so it is
*not* mistaken for dense), k spikes → low-rank(k), a dense AR(1) → low-rank with J growing in n and
capped at `J_max = d/5` — differing by ±1–2 on J only when `d ≳ n`; **parallel over-detects near
`γ ≈ 1`** (false spikes on truly isotropic data, a multiple-testing miscalibration) and does not
scale, so it is dominated. Boundary gates: `d ≤ 10` → dense unless `~I`; `cond(R) < 10` (spectrum < 1
order of magnitude) → diagonal; dense only when `n > d` and `d ≤ 1000` (the `d(d+1)/2` penalty makes
it vanishingly rare above `d ~ few hundred` unless `n ≫ d` with a wide, structured spectrum).

The same whitened-spectrum machinery also chooses a **shaped learned metric**'s constant shape `A`
(`docs/design/07`): `learned_metric_rule`, having regressed `D(x)`, whitens the evidence scores by
the fitted `D(x)` (`regression.whitened_scores`) and runs `select_mass_mode` on the residual —
whose constant correlation *is* `A` — adding `shape` (`None` / `("lowrank", J)` / `"dense"`) to the
block's params. **Caveat (measured on IRT):** the MP null behind the AIC≡edge test assumes ~Gaussian
scores, so evidence with many **divergent transitions** (heavy-tailed scores) over-selects — a
7879-divergence IRT pilot spuriously picked `theta`→dense, corrected to `None` on a clean round-2
pilot (831 divergences).

**Divergence gate (the current safeguard).** Both `mass_mode_rule` and the shape branch of
`learned_metric_rule` are switched off when the pilot's `diagnostics.divergence_rate` (NUTS
`divergence_rate()`, carried through `Evidence`) exceeds `MODE_SELECT_MAX_DIVERGENCE_RATE` (0.10).
Above that, the evidence-based mode selection is skipped and the **defaults** stand: dimension-count
mass modes for constant blocks (the `lowrank_block_rule` placeholder is left in place), and a **plain
diagonal** metric for a learned-metric block (the `D(x)` regression is still adopted, only its
nondiagonal shape `A` is declined). NUTS counts warmup and sampling divergences separately
(`divergence_count` / `divergence_rate` take `include_warmup=False`, `include_sampling=True`), so the
rate the gate reads is over the **sampling phase** only — early-warmup divergences are common and
harmless, and it is the sampling draws that become the evidence. It is a coarse valve, not a fix — a
smoother scheme (a robust / divergence-trimmed score-covariance that also shrinks toward the
dimension-count prior) waits until we have more test cases; select shapes from a well-mixed pilot /
iterate meanwhile.

### Declared component costs → the multi-rate integrator (a structural rule)

`multirate_integrator_rule` (weight 0.6, evidence-free) proposes `integrator = "multirate"` with
`n = MULTIRATE_DEFAULT_N` (4, a placeholder with no evidence behind it) when the model declares
**both** a cheap and an expensive log-density component — that is, when
`Model.cheap_components` and `Model.expensive_components` are both non-empty. A DSL program gets
those labels from its `ModelSpec` (doc 08); a hand-written model declares them itself.

The gate is deliberately over *model components only*. The chart `JacobianPotential` is cheap by
construction and joins the cheap inner loop whenever a split is taken, but it can never *trigger*
one: it is not a model component, so it can never appear in `cheap_components`. Sub-stepping a
lone Jacobian would change the dynamics of **every constrained model** in exchange for a
stiffness gain nobody has measured, while costing `n` extra Jacobian gradients and drifts per
step. An undeclared model therefore keeps plain leapfrog, exactly as before this rule existed.

The two line-search integrators are **selectable but unruled**: `spec.integrator =
"line_search"` (or `"markovian_line_search"`) with `integrator_params` for the schedule,
thresholds, `p`, and the `base` — which may itself be `"multirate"`, so a line search can refine
a whole multi-rate step. A rule that reaches for them on a divergent pilot is still deferred
(see below).

### Deferred rules **[stage 2+]** — the rest of the grab-bag

Specified here so the interface and `Evidence` fields are shaped to receive them, but not yet
implemented:

- **Divergence count → relativistic / WALNUTS.** A high `diagnostics.divergence_count` suggests
  swapping quadratic kinetics for `relativistic` ones (bounded velocity), or — as an expensive
  last resort — WALNUTS. (Relativistic RMHMC, explicit variant, and a usable momentum
  partition-function approximation are prerequisites tracked elsewhere.)

## Discrete parameters

`analyze` refused a model with integer parameters until the release that wired them in. The
refusal was a guard against a *quiet* wrong answer rather than a repair of a broken partition:
discrete parameters are kept out of `model.parameters` entirely, so every rule above partitions
the continuous half perfectly well and would have handed back a sampler that simply never moves a
label — and a frozen coordinate has zero variance, so it reports a *perfect* ESS and R̂ 1.000.
Nothing the factory or the summary prints would have shown it.

Three things follow from that, and they shape the design.

**The sweep is not optional.** A model with integer parameters always gets
`DiscreteMetropolisWithinGibbs`, whatever else the spec says. There is no field to turn it off,
because the only alternative is the frozen-label sampler the refusal existed to prevent. `build`
appends it **last**, so it sits immediately left of the base algorithm — the invariant every
hand-composed site holds (`mimcs/testing/runner.py` and the tests; `examples/05_mixture.py` now
goes through the factory instead) and the
one `samplers/gibbs.py` states. `BaseSampler`'s `handles_discrete` check is the backstop: a stack
that failed to compose it raises rather than sampling with the labels held still.

**Only the *proposal* is a decision.** `spec.discrete_proposal` selects it — `"marginal"` (the
default) composes `DiscreteMarginalAdaptation`, `None` leaves the sweep's own uniform-over-the-
others proposal. A string rather than a bool because the next entries are already sketched (an
ordinal ±1 walk, a count-valued jump for an unbounded `int<lower=0>`; doc 14) and slot in as
values, not as a second flag.

`discrete_proposal_rule` (structural, evidence-free, weight 0.8) picks it from the **width of the
supports**. The learned marginal buys about a `k − 1` factor in label moves per iteration
(measured 1.93× at `k = 3`, 5.94× at `k = 8`, and *exactly* 1.00× at `k = 2` where the Hastings
term is identically zero — doc 14), at the cost of one `(size_i, n_i)` table per parameter. That
trade turns over as the support widens, because each value then collects only ~`1/n_i` of the
draws. The threshold is `WIDE_SUPPORT` (64), **imported from
`adaptation/discrete_marginal.py` rather than restated** — it is the same number the mixin already
warns at, and one definition is what keeps the two from drifting.

The **widest parameter decides for the whole model**. The mixin is all-or-nothing: one
`_postprocess_hooks` allocates and updates every parameter's table together, so there is no way to
adapt a narrow parameter and skip a wide one in the same model. Given the choice, the factory
declines rather than building a table it has just called too wide. Above the threshold the rule
**warns**, because the uniform proposal that stands instead is itself likely poor there — it
spends `(n_i − 2)/(n_i − 1)` of its attempts on values of essentially zero density. It is a
placeholder holding the seam for the proposals that are not built yet, not a considered choice,
and the log line says so.

**A discrete-only model needs a different base.** With `coord_dim == 0` there is no trajectory to
integrate, so `discrete_only_base_rule` (structural, weight 0.9) proposes
`base = "static"` (`StaticContinuous`) together with `adapt_step_size = False` and
`mass_adapt = None` — one rule setting the three slots that a zero-dimensional continuous space
makes moot, rather than three places each having to remember the case. `build` then takes a
narrower path: no potentials, no integrator, no kinetics, and neither initialization mixin
(`UniformInit` redraws through `state_at_coordinate`, which lives on `BaseHMC`). Warmup
termination is deliberately kept: a discrete parameter's features are real features, and a chain
still reassigning labels is not mixed. Each skipped field is **refused** rather than ignored if a
hand-edited spec sets it, on the same policy as the rest of `build` — a user who asked for an
adapted step size and got a sampler with no step size has been handed a different algorithm.

**Under tempering `build` adds only the adaptation.** `parallel_tempering` injects the sweep
itself, between `ReplicaExchangeMixin` and the selection mixins (doc 13) — a position that cannot
be expressed from the flat mixin list, since the sweep must be inside the replica exchange and the
per-lane NUTS selection mixin terminates the draw-component chain. Adding it here as well would
both misplace it and duplicate it into an MRO error. `DiscreteMarginalAdaptation` stays out of
`_PER_TEMPERATURE_ADAPTATIONS`: it reshapes `state.discrete` to `(K, n)` itself and learns every
rung's table in one jitted update, so it belongs on the product chain.

### Evidence carries the labels

`Evidence` gained a `discrete` field — `(n, discrete_dim)`, integer, row-aligned with
`coordinates`. This is not bookkeeping. A discrete model's coordinate-space density is
*conditional* on the labels, so a recomputed score has no meaning without the row's own `z`;
`_recomputed_scores` used to call `log_prob_at_coordinate` without them, which raised inside
`Model._require_discrete` and was swallowed by the caller's `try/except`, leaving `gradients=None`
and silently disabling `mass_mode_rule` and `learned_metric_rule`. Worth recording precisely,
because the prediction going in was that this bit on the *default* path and it does not: a live
sampler saves its gradients, so the failure only appeared with `save_gradients=False` or a bare
`(samples, coordinates)` bundle.

The labels also make the warm start a whole state. `_init_position` returns a **dict** for a
discrete model — the only channel that carries both halves — because warm-starting the continuous
block to a fitted configuration while resetting the labels to their lower bound would pair a
position with the wrong assignment: for a mixture, coordinates fitted under one clustering and
every label saying "cluster 0". `sampler.initialize()` still overwrites both, exactly as
`UniformInit` overwrites the continuous warm start; "initialize" means start fresh.

### What is still missing

Genuinely, not ceremonially:

- **No rule reaches for parallel tempering.** This is the one that matters, and it is not an
  oversight. A chain stuck in one mode reads as converged — that is the whole finding behind the
  spike-and-slab benchmark (doc 13): plain Gibbs was trapped on all 8 seeds while split-R̂ reported
  1.0000 on every one. So no *single* run's diagnostics can suggest tempering. The proposal on the
  table is to look for evidence **conflicted across rounds** — several chains started at different
  locations getting stuck in different places — which needs `Evidence` to record which round each
  row came from. Today every result is collapsed into one table, so the provenance column has to
  come first. Tracked in `TODO.md`.
- **No cost heuristic for a discrete block.** The sweep is one full log-density evaluation per
  discrete coordinate per sweep (measured 1.09× overhead at 10 labels, 2.47× at 300; doc 14), and
  nothing weighs that against a trajectory. Component-restricted recomputation is the fix and is
  the larger prize.
- **No ordinal or count-valued proposal**, which is what the wide-support warning is holding the
  seam for.

## Public API

Exported from `mimcs/__init__.py`:

- `make_sampler(model, *results, seed=0, init=None, buffer_size=None, recompute_gradients=True) -> sampler` — the one-liner.
- `analyze(model, *results, recompute_gradients=True) -> SamplerSpec` — produce the prototype for inspection/override.
- `SamplerSpec`, `BlockSpec`, `Evidence`, `Diagnostics` — the dataclasses, for advanced use.

## Forward-compatibility seams

| Seam | Why |
|---|---|
| Coordinate space is **always a list of `BlockSpec`** | block-wise kinetics become "n > 1 blocks", not a new path |
| Block kinetics are **one slice-aware `KineticHamiltonian` per block** in `BaseHMC`'s kinetics list (doc 06) | no composite object; a whole-space kinetic is a one-element list |
| Heuristics are **rule functions emitting weighted `Proposal`s**, resolved by an arbiter | conflicts resolve by weight, not list order; the grab-bag grows without rules having to know about each other |
| Arbiter dispatches **per slot**, defaulting to argmax-weight | numeric slots can later register **combiners** (blend rather than choose) without changing rules |
| `Evidence` already carries `coordinates`/`gradients`/`diagnostics` | the regression and divergence rules have their inputs the day they land |
| `spec.rationale` | explainability: the user can see *why* each choice was made, and which to override |

## Staging plan

- **Stage 1 (this round's implementation):** the full interface — `Evidence` normalization,
  `SamplerSpec`/`BlockSpec`, the `Proposal` + argmax-weight **arbiter**, `analyze`/`build`,
  `make_sampler` — the default sampler, and the dimension-count rule (single whole-space block;
  diagonal vs dense). The arbiter scaffold ships even though only one rule votes, so stage-2
  rules slot in without interface change.
- **Landed since:** the `model.parameters` block-partition rule (list-aware per-block mass); and
  the **metric-regression `learned_metric` rule** — a second (refinement) arbitration pass that
  fits mass-matrix mini-language candidates to each block's conditional score covariance (L-BFGS
  in `mimcs.optim`, AIC-ranked) and adopts the best position-dependent form.
- **Landed since (discrete):** `discrete_proposal_rule` and `discrete_only_base_rule`, the
  `"static"` base, and `Evidence.discrete` — see [Discrete parameters](#discrete-parameters).
- **[stage 2+]:** the divergence-count `relativistic` / WALNUTS rule; a cost-aware metric
  criterion (beyond AIC) — plus per-slot **combiners** for numeric slots; and a rule that reaches
  for parallel tempering, which needs per-round provenance in `Evidence` first.

  Two items previously listed here have shipped: **diagonal-plus-low-rank mass**
  (`LowRankQuadraticKinetic` + `LowRankAdaptation`, selected by both `lowrank_block_rule` and
  `mass_mode_rule`) and the **sample-based dense→diagonal downgrade**, which is what
  `mass_mode_rule` does — it picks diagonal / lowrank(J) / dense per block from the whitened
  score-covariance spectrum, overriding the dimension-count placeholder whenever evidence exists.

## Verification

- **Unit.** `Evidence` normalization for each input form (`ndarray` / `SamplerOutput` / live
  sampler / bundle), including the live-sampler model-mismatch guard; the default spec equals
  the stated default sampler; the dimension rule emits the expected `kind` proposal and weight
  across the threshold and under the covariance signal; the **arbiter** picks the highest-weight
  proposal per slot, breaks ties by registration order, records winners and losers in
  `rationale`, and lets a post-`analyze` `spec.blocks[0].kind` mutation override the outcome;
  `build` produces the expected mixin MRO and kinetic `mass_mode` for each spec.
- **End-to-end.** `make_sampler(model)` with no results returns a working NUTS + diagonal
  sampler that passes `mimcs.testing.evaluate(...)` / `report.assert_correct()` on
  `correlated_gaussian` (low-dim → dense path when fed prior samples) and a higher-dimensional
  Gaussian (diagonal path); the `analyze → mutate spec → build` flow yields the intended
  sampler.
