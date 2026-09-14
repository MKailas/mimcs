# Sampler factory — Reference Manual

A reference for the sampler factory as currently implemented (stage 1). The factory
constructs a sampler for a [`Model`](../design/05_model_interface.md), optionally tailored to
earlier sampling results. For the design and rationale, see
[`docs/design/09_sampler_factory.md`](../design/09_sampler_factory.md); features marked
**(not yet)** are planned but not implemented.

## Quick start

```python
from mimcs import make_sampler

sampler = make_sampler(model)        # a reasonable default sampler for `model`
sampler.initialize()                 # optional: U(-2,2) start + line-searched step size
sampler.warmup(2000)
samples = sampler.sample(8000)       # {name: (n, *ambient_shape)} of ndarrays
flat    = sampler.get_samples_flat() # (n, ambient_dim), if you want one block
```

`sampler.initialize()` (call it before `warmup()`, or omit it to start from the default
zeros/`init=` position) draws the initial coordinate from `U(-2, 2)` in the unconstrained space
(retrying to finite density) and sets the initial step size by a backtracking single-leapfrog
(MALA) line search from `0.5` to acceptance `0.9`. Factory samplers always include the
initialization mixins; nothing calls `initialize()` for you.

Pass earlier results to tailor the sampler to the target:

```python
sampler = make_sampler(model, previous_samples)   # e.g. a pilot run's draws
```

`make_sampler(model, *results, seed=0, init=None, buffer_size=None, blocks=None, recompute_gradients=True)` returns an
instantiated sampler (the same kind of object the `mimcs.testing` builders produce, with
`.warmup(n)`, `.sample(n)`, `.get_samples()`, `.get_samples_flat()`, `.get_gradients()`,
`.acceptance_rate()`). The
**`model` is always the first argument** — it defines the coordinate space, the potentials, and
the default starting position. A sampler saves the total score of each draw by default
(`save_gradients=True`), so passing a prior run back to the factory reuses those scores for the
metric-regression rule instead of recomputing them; `recompute_gradients=False` skips the
recompute for a prior run that did **not** save them (e.g. a very expensive model).

## What you can pass as results

`*results` is flexible: pass nothing, one result, or several (they merge). Each result may be:

| Form | Interpreted as |
|---|---|
| *(nothing)* | build the default sampler |
| an `np.ndarray` of shape `(n, ambient_dim)` | posterior **samples** |
| a `SamplerOutput` (from `mimcs.testing`) | samples + diagnostics |
| a live sampler (a `BaseSampler`) | samples + diagnostics (+ coordinates/gradients if recorded); its model must match `model` |
| a `(samples, coordinates, gradients)` tuple | the three arrays (any trailing ones omitted) |
| a dict with keys `samples` / `coordinates` / `gradients` | the named arrays |

Anything a particular result does not carry is simply absent; the heuristics use what is
available.

## The default sampler

With no results (or no information that warrants a change), `make_sampler(model)` builds:

**NUTS** + per-block kinetics + `ScoreMassAdaptation` + `RobbinsMonroStepSize` (Robbins–Monro
step-size adaptation) + `ClassifierTermination` (dynamic warmup termination).

The mass is fit to the **score covariance** per block (`ScoreMassAdaptation`, dense on fused
low-dimensional blocks, diagonal on large ones). Set `spec.mass_adapt = "covariance"` for the
standard empirical-covariance mass instead, or `None` for a fixed identity mass.

Four more things the factory composes, none of them a spec field:

* `UniformInit` and `StepSizeLineSearch`, which do nothing unless you call `sampler.initialize()`
  before warmup — a `U(−2, 2)` start on the unconstrained coordinates and a backtracking MALA
  search for the initial step size.
* `UnitVectorCenteringAdaptation`, **automatically**, whenever the model has a unit-vector
  parameter with `adaptive=True` (the default). It fits the stereographic chart's pole and scale,
  which matters because the unfitted pole sits at `e_d` — squarely in the target's bulk for a
  distribution concentrated there. Turn it off with `adaptive=False` on the parameter.
* `LowRankAdaptation` / `MetricAdaptation` / `ShapedMetricAdaptation` for blocks whose kind calls
  for them. These are **not** governed by `mass_adapt`, so `mass_adapt="covariance"` on a mixed
  partition runs a position-based adaptation beside score-based ones.
* Two build-time defaults not visible on the spec: `target_accept = 0.8` (the mixin's own default
  is 0.234) and `mass_min_samples = 50`.

If the spec carries evidence and you pass no `init`, the chain **warm-starts from the last
evidence draw** rather than the model's default point — for a model with `int` parameters, its
labels as well as its position, since a position fitted under one label configuration paired with
reset labels is not a state the chain was ever in. (`sampler.initialize()` still overwrites both:
"initialize" means start fresh.)

### Models with `int` parameters

The factory builds the Metropolis-within-Gibbs sweep automatically, and there is **no way to turn
it off** — the alternative is a sampler that holds the labels frozen, which no diagnostic would
show you (a frozen coordinate has zero variance, so it reports a perfect ESS and R-hat 1.000).

What you *can* choose is **how each parameter gets updated**, through `spec.discrete` — one entry
per `int` parameter, carrying a `kind` and its options, exactly as `spec.blocks` carries one entry
per coordinate block with its kinetic:

| `kind` | what it does |
|---|---|
| `"metropolis"` | the Metropolis-within-Gibbs sweep: propose among the `n_i − 1` values the coordinate is *not* at, accept on the ratio. Reads `params["proposal"]` — `"marginal"` (learn the coordinate's marginal pmf during warmup and propose proportional to it) or `None` (uniform over the other values). |
| `"exact"` | exact conditional Gibbs: evaluate the conditional at all `n_i` values and draw from it. No proposal, no acceptance test, nothing to adapt. |
| `"random_walk"` | Metropolis with a two-sided geometric step (a fair coin for the direction, `Geometric(p)` for the length, mean `1 + e^ρ`), clamped at a bound. Reads `params["adapt"]` (default `True`): adapt `ρ` per coordinate toward 1/3 acceptance during warmup; and `params["init_log_scale"]` (optional, scalar or per coordinate): the starting `ρ`, which with evidence the factory sets from each coordinate's interquartile range, `2/p = IQR`. The only method for a parameter with an open bound. |

**A parameter with an open bound, or one declared `ordinal` with at least 3 values, gets
`random_walk`** before any of the width rules below apply: the declaration says what the width cannot
(that neighbouring values are similar), and an open side leaves nothing else that can run. At 2
values an ordering carries no information and the walk would halve the move rate against the
Metropolis flip, so a binary `ordinal` parameter falls through to the rules below, and the rationale
says why.

Otherwise the factory chooses per parameter, from the support width and from whether the parameter is
*elementwise* — that is, whether every model component reading it is a `scan` component scanned
over it, which makes each candidate cost `O(1)` instead of a whole density:

| support | elementwise | chosen |
|---|---|---|
| 2–3 values | either | `metropolis` + `marginal` |
| 4–8 | no | `exact` |
| 4–64 | yes | `exact` |
| 9–64 | no | `metropolis` + `marginal` |
| > 64 | no | `metropolis` + uniform, **with a warning** |

**Narrow supports never get exact Gibbs**, and that is not a cost decision — it is the opposite of
one. The Metropolis sweep proposes from a learned table, which approximates each coordinate's
*marginal*; on a narrow support that is close enough to its *conditional* that proposing from it and
accepting beats drawing exactly. At 2 values it is provable (the proposal is forced, so Metropolis
moves with probability `min(1, π_b/π_a)` against Gibbs's `π_b` — asymptotic-variance ratios 5.0 at
`π_a = 0.6`, unbounded at 0.5, at half the evaluations); at 3 it is measured (0.91× label ESS over
8 paired seeds). From 4 values up the table degrades faster than the exact draw costs, and exact
wins by a margin that grows with the support — 1.3× at `k = 4`, 2.9× at `k = 16`. Spike-and-slab
indicators sit at 2 values and stay on the better kernel.

When the factory declines everything above 64 values, it **warns**. That is deliberate: the uniform
proposal left in place is itself poor on a wide support (it spends nearly every attempt on values
of essentially zero density), so the omission is a placeholder, not a recommendation. If the values
are ordered, declare the parameter `ordinal` and it gets the random walk; otherwise writing the
likelihood as a `scan` component over the labels is the way to get an exact draw there instead.
Override either way:

```python
spec = analyze(model)
spec.discrete[0].kind = "exact"                          # draw from the conditional
spec.discrete[0].kind = "metropolis"                     # ... or propose and accept
spec.discrete[0].params = {"proposal": "marginal"}       # adapt anyway, on a wide support
spec.discrete[0].params = {"proposal": None}             # or keep the uniform proposal
```

The entries must stay in the model's own parameter order — the rules address them by index, so a
permuted list is refused rather than silently giving a parameter another one's method.

A model with **only** discrete parameters gets `base="static"` (`StaticContinuous`), which leaves
the empty continuous block alone; the step size and mass adaptation are switched off with it,
since there is nothing for them to act on. Warmup termination stays on — a chain still reassigning
labels is not mixed.

The order the coordinates are visited in is `spec.discrete_scan`: `"systematic"` (the default, and
the only one any rule picks) or `"random"` (each jump a uniformly chosen coordinate; reversible, and
the base for future blocked updates). The sweep's own knobs go through `algo_kwargs` as usual:
`discrete_sweeps` (sweeps' worth per iteration), `discrete_jumps` (random scan: jumps per
iteration, default one per coordinate), and `discrete_lambda` / `discrete_min_samples` / `discrete_adapt_n0` /
`discrete_adapt_kappa` for the marginal adaptation, `discrete_rw_*` for the random walk's. See
`docs/reference/algo_kwargs.md`.

**Centering is opt-in, not a default** (`spec.centering`, default `False`).
`RobustCenteringAdaptation` standardizes each `centered=True` parameter by its **median and MAD**,
but it acts only on parameters declared `centered=True` (which the DSL never sets), and where it
*does* act it was measured to destabilize a fragile far-from-mode adaptation while merely matching
the default on an easy target — so it is left off pending the mass-matrix work. Enable it with
`spec.centering = True`.

Warmup **ends on a mixing criterion** rather than a fixed count: `warmup(n)`'s `n` becomes an upper
bound, and `warmup()` with no `n` runs until the chain looks to be mixing well (or `max_warmup`).
The default criterion is a logistic-regression classifier on the draws' features; set
`spec.terminate = "rhat"` for the Gelman–Rubin diagnostic, or `None` to disable. The criterion
draws no randomness, so a `warmup(n)` that never triggers an early stop is identical to leaving it
off.

## Configuring the sampler: the prototype

For anything beyond the one-liner, work with the **prototype** instead of keyword soup. `analyze`
returns a mutable `SamplerSpec` you can inspect and edit before building:

```python
from mimcs import analyze

spec = analyze(model, previous_samples)
print(spec.rationale)            # why each choice was made (incl. options it beat)
spec.blocks[0].kind = "dense"    # override a decision
spec.base = "hmc"
spec.algo_kwargs["n_leapfrog"] = 32
sampler = spec.build()           # or spec.build(seed=1, init=x0, buffer_size=256)
```

A `SamplerSpec` has these fields:

| Field | Meaning |
|---|---|
| `model` | the model the spec was analyzed from |
| `base` | large-scale algorithm: `"nuts"` (default), `"hmc"`, `"randomized_hmc"`, each with a parallel-tempered counterpart `"pt_nuts"`, `"pt_hmc"`, `"pt_randomized_hmc"` — the same algorithm over a K-fold product space, keeping the cold chain. No rule selects a tempered base; it is an explicit choice. Plus `"static"`, chosen automatically for a model with **no** continuous parameters (it moves nothing continuous, leaving the discrete sweep to do the work) |
| `tempering_params` | ladder options for a `pt_` base: `n_temperatures`, `betas`, `beta_min`, `tempered` (which components β scales — naming a subset gives a power posterior), `adapt_ladder`, `adapt_beta_min`, `swap_target_accept`. Rejected on an untempered base, so a typo cannot pass silently |
| `blocks` | list of `BlockSpec` (the coordinate space; see below) |
| `integrator` | `"leapfrog"` (default), `"multirate"` (RESPA over the model's declared cheap/expensive components), `"line_search"` or `"markovian_line_search"` (WALNUTS, within-orbit adaptive step size) |
| `integrator_params` | that integrator's own options — **not** `algo_kwargs`. `{"n": 4}` for multirate; `{"base", "base_params", "schedule", "error_thresholds", "p"}` for the line searches, whose `base` may itself be `"multirate"`. Unknown keys raise |
| `step_size` | initial leapfrog step size |
| `adapt_step_size` | whether to adapt it; default `True`. Robbins–Monro normally, but **swapped** for `LineSearchStepSizeAdaptation` under a line-search integrator, whose real acceptance is ≈1 and would drive the step size away |
| `mass_adapt` | which mass adaptation to fit the `diagonal`/`dense` blocks: `"score"` (default, the KL score covariance), `"covariance"` (the empirical covariance of the positions, written only after `mass_min_samples` draws), or `None` (identity mass, no adaptation). Does **not** affect `lowrank` / `learned_metric` blocks, which keep their own |
| `centering` | whether to include `RobustCenteringAdaptation` (acts only on `centered=True` params); **opt-in, default `False`** |
| `terminate` | warmup-termination criterion: `"classifier"` (default), `"rhat"`, or `None` (off) |
| `discrete` | one `DiscreteSpec` per `int` parameter — its update method (`"metropolis"` / `"exact"` / `"random_walk"`) and that method's options. Chosen per parameter from the support width and whether it is elementwise; the *sweep itself* is not optional — see [Models with `int` parameters](#models-with-int-parameters) |
| `block_override` | an *input* to the block-partition rule, not one of its outputs: a list of name tuples, each becoming one block. Set by `analyze(model, blocks=…)`, which validates it. Only the *grouping* is fixed — the refinement rules still pick each block's kind |
| `algo_kwargs` | everything splatted into the sampler constructor — 85 options across the composed mixins. See **[`algo_kwargs.md`](algo_kwargs.md)**; unknown keys are silently ignored |
| `rationale` | human-readable record of how the spec was decided, one line per arbitrated slot |
| `evidence` | the `Evidence` the spec was analyzed from (samples / coordinates / gradients) |

`print(spec)` gives a one-screen summary; `spec.rationale` says which rule set what and why. The
default `repr` is the full dataclass and runs to 70+ lines once a learned metric is fitted.

`buffer_size` is a **build-time** keyword rather than a spec field, like `seed` and `init`:
`make_sampler(model, buffer_size=256)` or `spec.build(buffer_size=256)`. It sizes the RNG buffer,
whose allocation is `buffer_size × Σ prod(shape)` floats — for NUTS ~21 MB at the defaults, almost
independently of the model's dimension, because it is dominated by the `leaf_select` component.
Note it is **not stream-neutral**: two runs at the same seed and different `buffer_size` agree only
up to the first refill, so a reproducible result must quote both.

Each `BlockSpec` describes one block of coordinates:

| Field | Meaning |
|---|---|
| `names` | the parameter names the block spans |
| `coord_slices` | list of `(start, stop)` slices in coordinate space (may be non-contiguous) |
| `kind` | `"diagonal"`, `"dense"`, `"lowrank"`, or `"learned_metric"` (`"relativistic"` is **(not yet)**) |
| `params` | kind-specific options: `{"rank": J}` for `"lowrank"` (default 4 at build, 8 when the rule sets it); for `"learned_metric"`, `{"metric": <MetricExpr>, "metric_init": <fitted params>, "shape": None \| "dense" \| ("lowrank", J)}` — `shape` upgrades the diagonal metric to `D(x)^½ A D(x)^½` |

Each block becomes one kinetic in the sampler's kinetics list; the mass is adapted per block over
that block's coordinate slice, by whichever adaptation `spec.mass_adapt` selects (`"score"` by
default). `lowrank` and `learned_metric` blocks are unaffected by that setting -- they carry their
own adaptations.

`make_sampler(model, *results)` is exactly `analyze(model, *results).build()`.

### Overriding the block partition

The partition is chosen by size alone (see the rule below), which is a heuristic and will
sometimes group your parameters wrongly — most consequentially by *fusing* them, since a fused
multi-parameter block can never carry a learned metric. `blocks` says what you want instead:

```python
spec    = analyze(model, blocks=[("mu", "sigma"), "theta"])   # two blocks, by name
sampler = make_sampler(model, pilot, blocks=["v", "x"])       # ... or in one line
```

Each entry is a group of parameter names that becomes one block; a bare string is shorthand for
a one-name group, and a group's parameters need not be adjacent in the coordinate vector.

**Naming only part of the model is the normal case** — any parameter you do not mention is
partitioned by the usual rule, running on exactly those parameters. So `blocks=["x"]` means
"give `x` its own block and decide the rest as usual".

Only the *grouping* is fixed. The refinement rules still run on the result, so each block's kind
(diagonal / dense / low-rank / learned metric) is still chosen from the evidence — which is the
point: splitting a fused block is what makes a single parameter eligible for a learned metric in
the first place.

Rejected with a `ValueError` before any rule runs: a name that is not a parameter of the model, a
name in two groups, or an empty group.

## How choices are made: heuristic rules and arbitration

`analyze` starts from the default spec and runs **heuristic rules** that each emit weighted
*proposals* for individual decisions. When rules disagree, the highest-weight proposal wins
(the current value is the baseline); ties break by rule order, and every proposal is recorded
in `spec.rationale`. Your own edits to the returned spec always take precedence (they happen
after analysis).

**Five rules run, in two passes.** `RULES` first (structural), then `REFINEMENT_RULES`, so a
later pass can override an earlier decision:

| Rule | Pass | Weight | Sets |
|---|---|---|---|
| `block_partition_rule` | structural | 0.80 | `blocks` — the partition and an initial kind from dimension alone |
| `multirate_integrator_rule` | structural | — | `integrator = "multirate"` when the model declares **both** cheap and expensive components (which a DSL model with large data usually does) |
| `lowrank_block_rule` | refinement | 0.60 | a diagonal block of dimension in (50, 1000] → `"lowrank"`, rank 8. A placeholder |
| `mass_mode_rule` | refinement | 0.65 | each block's kind from the **whitened score-covariance spectrum** of the evidence — diagonal / lowrank(J) / dense. Beats the placeholder whenever gradients exist. A block with fewer than `0.75·d` effective evidence rows stays **diagonal** whatever its spectrum shows: too few rows to aim a direction, however clearly one is detected (doc 09) |
| `learned_metric_rule` | refinement | 0.70 | single-parameter blocks → `"learned_metric"` when an AIC-ranked regression beats the constant baseline by 2.0 |

The weights are what decide a contested slot, so the ordering above *is* the precedence:
a learned metric beats an evidence-chosen mass mode, which beats the dimension-count placeholder.

**Both evidence-driven rules switch themselves off on a pathological pilot**: if more than 10% of
the pilot's transitions diverged (`MODE_SELECT_MAX_DIVERGENCE_RATE`), neither `mass_mode_rule` nor
`learned_metric_rule` proposes anything, on the grounds that a diverging chain's gradients describe
the sampler's failure rather than the target's geometry. If passing evidence seems to change
nothing, check the pilot's divergence rate first.

The main rule is the **block partition**, which sets `spec.blocks` from the model's
parameters:

- a parameter of dimension in **(50, 1000]** → its own **low-rank** block (a diagonal scale plus
  `J = 8` learned correlation directions); above **1000** it stays **diagonal**. *(Placeholder
  thresholds/rank, pending evidence.)*
- a parameter of dimension in **(20, 50]** → its own **dense** block;
- the low-dimensional (**≤ 20**) parameters → **fused** into **dense** blocks, each **capped at
  20** — a parameter joins the current block only if it fits within the cap, otherwise the block is
  flushed and a new one begun — regardless of whether they are adjacent in the coordinate vector
  (a block's coordinates are a list of slices, so it can span scattered parameters).

The idea (hierarchical models): the correlations that matter most for a mass matrix are among
the naturally few high-level parameters, which are often declared separately (a mean and a
variance parameter, possibly with lower-level parameters between them) and so are individually
low-dimensional — fusing them into a dense block captures their cross-parameter correlations
even when they are not adjacent, while high-dimensional parameters stay diagonal.

The fusion is **capped** rather than grown just past the threshold because a multi-parameter fused
block can never carry a **learned metric** (that rule works only on single-parameter blocks).
Without the cap, fusion can swallow a whole vector parameter (a 20-dim `a` glued onto two scalars →
a 22-dim block), which then can never be given the position-dependent metric a funnel needs. Capping
keeps such parameters as their own block and thus learned-metric-eligible — worth far more than the
small cross-parameter correlation it forgoes, on the hierarchical funnels where it matters.

## Public API

Exported from `mimcs`:

- `make_sampler(model, *results, seed=0, init=None, buffer_size=None, blocks=None, recompute_gradients=True) -> sampler`
- `analyze(model, *results, blocks=None, recompute_gradients=True) -> SamplerSpec`
- `SamplerSpec`, `BlockSpec` (the dataclasses, for advanced use)

`Evidence`, `default_spec` and `normalize` are in `mimcs.factory` rather than the top-level
namespace, which is deliberately small: the parameter types live in `mimcs.model`, the samplers in
`mimcs.hmc` and `mimcs.pt`, the adaptation mixins in `mimcs.adaptation`. Re-exporting all of those
would make `mimcs` both crowded and a thing to remember to update as the library grows.

## Not yet supported

For reference, the following are part of the design but not implemented yet: the
`"relativistic"` block kind; a cost-aware metric criterion beyond AIC; the
divergence-count-driven relativistic/WALNUTS selection; and **any rule that reaches for parallel
tempering** — a chain stuck in one mode reads as converged, so no single run's diagnostics can
suggest it, and the evidence table does not yet record which round a draw came from. Tempering
stays an explicit choice (`spec.base = "pt_nuts"`). Nor does any heuristic know what a discrete
block costs relative to a trajectory. (A label *can* now enter a learned metric: the regression
fits `E[g g' | q, z]`, offering each integer parameter as a dependency — standardized by its
declared support if ordinal, reference-coded into `k-1` indicators if categorical — and
multiplying the best discrete factor onto the best continuous one when AIC says it earns its
parameters.) (Sample-based per-block refinement was
listed here and **is** implemented — `mass_mode_rule` picks each block's kind from the evidence
spectrum, including downgrading a dense block to diagonal — as is the `"learned_metric"` kind and
its gradient-regression rule.) The rest is tracked as stage 2+ in the design document.

## Combinations the factory refuses

`analyze` / `make_sampler` raise on a model the factory cannot handle, and `build()` raises rather
than quietly hand back a different algorithm from the one asked for:

- **`integrator="markovian_line_search"` with a non-NUTS base** (`hmc`, `randomized_hmc`, and
  their `pt_` counterparts). The randomized line search consumes per-step coins, and only a NUTS
  base declares them — it calls the integrator once per leaf, while the others integrate a whole
  trajectory in one call and have nowhere to put them. The integrator would still build and run,
  as the *deterministic* WALNUTS-D, with nothing to indicate it. Note `randomized_hmc` is refused
  too: its randomization is of the trajectory *length*, not of the refinement. Use a NUTS base, or
  `integrator="line_search"` to choose the deterministic variant deliberately.
- **`centering=True` or an adaptive `unit_vector` parameter with a `pt_` base** — a chart's
  `(mu, sigma)`, or a sphere chart's pole, is shared by every temperature, so it is neither a
  per-rung quantity nor well defined from the product coordinate.
- **A `learned_metric` block spanning more than one parameter, or a non-contiguous one** —
  `NotImplementedError`; the metric regression works on single contiguous blocks.

And every enumerated field is validated, so a typo raises rather than silently reverting to a
default: unknown `base`, `integrator`, `terminate`, `mass_adapt`, discrete `kind` or discrete
`proposal`; a `discrete` list that does not match the model's parameters positionally; unknown keys in
`integrator_params` or `tempering_params`; `tempering_params` given at all on an untempered base;
an unknown line-search `base`. The one thing **not** validated is `algo_kwargs`, whose keys are
splatted into the constructor and silently ignored if unrecognised.

## Experimental

Two implemented features are deliberately **not** reachable from the factory, and are marked
`[experimental]` in their own docstrings:

- **Relativistic HMC/NUTS** (`mimcs.hmc.relativistic`, `mimcs.adaptation.RelativisticMassAdaptation`).
  The per-particle rest mass adapts, but the light speed `c` is a fixed scalar, and the argument
  that one value suffices assumes a centering reparametrization has standardized the coordinates —
  which is opt-in and off by default. A first-class option most likely needs an adaptation for `c`
  itself, which is open research.
- **Implicit RMHMC** (`mimcs.hmc.riemannian`, `mimcs.hmc.solvers`). Correct, and the reference the
  explicit path is tested against, but the interface for supplying `G(q)` by hand in a
  `SamplerSpec` needs design, and the fixed-point iterations run to a fixed count with no residual
  check. The **explicit** block-Riemannian path (the `"learned_metric"` block kind, backed by the
  mass-matrix mini-language) is the supported one and needs no implicit solve.

Both remain available by building a sampler directly, or through the `mimcs.testing` builders
(`relativistic_hmc`, `relativistic_nuts`, `rmhmc`, `rmnuts`).
