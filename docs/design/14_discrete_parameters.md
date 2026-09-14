# Discrete Parameters

## Motivation

Until this document, every parameter in mimcs was continuous. That is a real restriction rather
than a stylistic one: mixture and latent-class models, spike-and-slab variable selection, change
points, hidden discrete states and model-index parameters are all written with an integer
parameter, and none of them could be expressed. Stan's answer is to *marginalize* the discrete
parameter out by hand — `log_sum_exp` over its support inside the model block — which works, costs
`K` evaluations of the likelihood term, and is impossible when the discrete parameters are coupled
to each other.

There is also a second, longer-range reason. `04_manifold_parameters.md` describes an **atlas**: a
parameter covered by several charts, with an integer `chart_index` saying which is active. That
index *is* a discrete parameter, and the machinery here is what it would eventually be built on —
though nothing in stage 1 connects the two (see "What is deferred").

## Scope

Stage 1 ships:

- `IntegerParameter` — a bounded integer parameter, `int<lower=L, upper=U>` in the DSL, declarable
  as an array.
- A second flat array on the sampler state and through `Model`, of dtype `int`.
- `DiscreteMetropolisWithinGibbs` — a deterministic-scan Metropolis-within-Gibbs sweep, composed
  over any continuous base algorithm, plus `StaticContinuous` for a discrete-only model. (Since
  split into a family superclass and two scans, `SystematicScanMetropolisWithinGibbs` and
  `RandomScanMetropolisWithinGibbs`; see "Random scan".)
- `categorical` / `categorical_logit` in the DSL.
- Bare features and no Stein term for a discrete parameter.
- A **refusal** from the sampler factory and from parallel tempering (both lifted since).

Everything else is in "What is deferred", each with enough design to be built against.

## The state: two arrays, not one

A draw is now a pair — a flat `float` vector and a flat `int` vector — rather than one vector. The
alternative, widening the existing float array and rounding on read, was rejected outright: it
loses the property that makes a label a label, and it would put integers in front of every piece of
machinery that assumes a real coordinate (mass matrices, the score, the chart pullback).

Concretely, `MHState` and `HMCState` each gain one field:

```python
class HMCState(NamedTuple):
    coordinate: Array          # continuous position in coordinate space, flat
    sample: Array              # continuous position in ambient space, flat
    discrete: Array            # the discrete parameters, flat int32 -- shape (0,) if none
    ...
```

These are the *only two* state NamedTuples in the library — NUTS, RMHMC, WALNUTS and every
parallel-tempering sampler reuse `HMCState` — so "add a field to the state" is a two-line change.
Everything else rides through untouched, because every sampler mutates state through `_replace`.
That is the same property `hmc/state.py` relies on for integrator state, and it is why a
gradient-based sampler needs to know nothing at all about the discrete block: **HMC never moves
it**.

### The model keeps a parallel list, not a wider one

`Model` gains `discrete_parameters` beside `parameters`, with its own offsets and its own
`discrete_dim`. Discrete parameters contribute **nothing** to `coord_dim` or `ambient_dim`.

This is what makes the change small. The factory's block partitioner, every mass adaptation, the
chart machinery, the score pullback and the metric regression all iterate `model.parameters` and
slice by `_coord_offsets`; not one of them sees a discrete parameter, and not one of them needed
editing. A design that gave a discrete parameter `coord_dim == 0` and left it in the main list
would instead have produced a zero-width block in every one of those places, each needing its own
guard.

The density needs no special treatment either. `log_prob_fns` are pure functions of a
`{name: value}` dict, so `unpack_coordinate` merely seeds that dict with the discrete values and
every existing component sees them as ordinary entries.

### `None` is loud, not a default

`log_prob_at_coordinate`, `features` and `log_prob_flat` take the discrete block as a trailing
optional argument. Passing `None` is fine when the model has no discrete parameters, and **raises**
when it has. Every pre-existing call site is therefore unchanged, and a call site that forgets the
new argument on a discrete model fails immediately.

The alternative — defaulting to the lower bound, or to a stored value — would evaluate the density
at stale labels: right shapes, right dtypes, an entirely ordinary acceptance rate, and the wrong
answer. That failure mode is not hypothetical here. The parallel-tempering Woodbury work (v0.1.6)
found `ProductKinetic._lanes` silently dropping a `HamiltonianContext` field at the vmap boundary;
there the consequence was only lost performance. Here it would be a wrong posterior.

## No chart

A continuous parameter is *defined* by its charts. A discrete parameter has none, and the omission
is structural:

- There is nothing to reparameterize. The sampler proposes an integer and accepts or rejects it.
  There is no smooth coordinate in which the proposal is better conditioned, because there is no
  smoothness.
- There is nothing to differentiate. A chart exists so the coordinate-space density can be handed
  to `jax.grad`; a discrete coordinate never is.
- So sample space **is** coordinate space, and the change-of-variables log-Jacobian is identically
  zero.

`BaseDiscreteParameter` is therefore a separate, much smaller interface from `BaseParameter`,
rather than `BaseParameter` with three methods raising `NotImplementedError`. The two hierarchies
being distinct is what lets `Model` keep two lists without any risk of one leaking into the other.

### The one restriction that buys this

**A discrete parameter may not be the parent of a continuous parameter's chart.** `Model` raises if
one is.

With that restriction, the total log-Jacobian cannot depend on the discrete block, so a Gibbs sweep
changes `discrete` and `log_prob` and *nothing else* — `coordinate` and `sample` are untouched and
`JacobianPotential` needs no discrete argument at all. It is a genuine limitation (it is exactly
what a bound like `real<lower=0, upper=z> x;` would need) and it is cheap to lift later: recompute
`sample` inside the sweep from the new labels, and thread the discrete block into
`JacobianPotential.potential` the same way `ModelPotential` already has it.

## How the discrete block reaches the density inside HMC

Through `HamiltonianContext`, which already exists for exactly this purpose — per-trajectory
constants that a Hamiltonian component needs and must **not** close over:

```python
class HamiltonianContext(NamedTuple):
    chart_hyperparams: tuple
    chart_indices: tuple
    ham_params: dict
    discrete: Any = None       # the flat integer block
    betas: Any = None          # parallel tempering
    kinetic_cache: Any = None  # hoisted per-trajectory kinetic quantities
```

The discrete block is a trajectory constant in the strictest sense: HMC integrates
`pi(· | discrete)` at one fixed value of the labels for a whole trajectory. `ModelPotential` reads
`ctx.discrete` and forwards it; that is the entire HMC-side change.

It must be in the context and not closed over for the reason `_reseed_caches` documents at length:
a jitted reseed that closes over its context bakes in the *first* call's value as a compile-time
constant, and every later call then refreshes the cache against labels nobody is sampling — with
the right shapes, the right dtypes, and no error.

## The sampler

A scan over the discrete coordinates is a **kernel-composing mixin**, a new mixin category. Every
other mixin cooperates through the `_*_hooks` chain and never touches `kernel`; a scan overrides
`kernel` and calls `super().kernel`:

```python
cls = make_sampler_class(RobbinsMonroStepSize, SystematicScanMetropolisWithinGibbs, NUTS)
```

`SystematicScanMetropolisWithinGibbs` and `RandomScanMetropolisWithinGibbs` are siblings under
`DiscreteMetropolisWithinGibbs`, which holds everything but the visiting order and refuses to be
composed on its own.

That works with no change to any base algorithm because `BaseSampler.__init__` jits the
MRO-resolved bound method, so the composition compiles as a single function. The ordering rule is
the usual one: mixins before the base algorithm.

Composing two `pi`-invariant kernels leaves `pi` invariant. That is the whole correctness argument
for the composition, and it is why the sweep is allowed to be this simple.

### The proposal

Deterministic scan through the discrete coordinates in declaration order. At each, propose
uniformly among the `n_i - 1` values the coordinate is *not* currently at:

```
n_i    = upper_i - lower_i + 1
offset = 1 + floor(u * (n_i - 1))            # uniform on 1 .. n_i-1
prop   = lower_i + ((cur - lower_i) + offset) mod n_i
```

This is symmetric — `q(a -> b) = q(b -> a) = 1/(n_i - 1)` — so acceptance is the plain ratio
`min(1, pi(prop)/pi(cur))` with no Hastings term. A binary coordinate always proposes the flip,
which is what one wants and needs no special case; nor does `n_i = 1`, where the formula proposes
the current value, a no-op that is accepted and counted as no move.

The acceptance test is written `log(u) < delta`, not `u < exp(delta)`: `exp` overflows to `inf` for
a large improvement and underflows to `0` for a large worsening, while `log(0) = -inf` accepts
exactly when it should and a `NaN` delta compares false, i.e. rejects.

### What was checked before it was written

Two things, both in scratch scripts, both then promoted to tests:

1. **Detailed balance.** Building the single-coordinate transition matrix by enumeration and
   checking `pi_i K_ij = pi_j K_ji` gives a maximum error of ~1e-6, which is the resolution of the
   `u` grid used to build `q`. The controls fail it loudly, which is what makes the check mean
   something: a missing acceptance test gives 1.7e-1, an inverted ratio 1.7e-1, and an asymmetric
   proposal used without a Hastings correction 2.2e-2.

2. **The float32 rounding edge.** The worry was that `floor(u*(n-1))` could reach `n-1` for `u`
   just below 1, collapsing the proposal to the current value. It **cannot**: exhaustively, for
   every `n` in 2..200000 and the largest representable `u < 1` in both float32 and float64, the
   product rounds down. `jax.random.uniform` for float32 is generated on a `2^-24` grid and does
   not get that close to 1 in the first place. So there is no defensive clamp in the code, and this
   note is here so nobody adds one back.

The prediction going in was that the clamp *would* be needed. It was not — a case of a predicted
hazard failing to materialize, which is worth recording precisely because the pressure runs the
other way (a clamp is cheap, so it is easy to add one and never learn it was unnecessary).

### Refreshing the gradient cache

`BaseHMC` caches each potential's value **and gradient** at the current coordinate, and the leading
half-kick of the next trajectory reads that cache back verbatim. After a sweep moves the labels
those gradients are for `pi(· | old labels)`. So `DiscreteMetropolisWithinGibbs` calls a
cooperative `_after_discrete` hook, which `BaseHMC` overrides to recompute them via the existing
pure `_reseed_caches`.

**Without it the next trajectory integrates the previous labels' gradients.** Nothing raises,
nothing looks wrong, and the acceptance rate stays plausible. It is the single most dangerous thing
in this design, and the test for it carries a negative control (skip the refresh and the assertion
must fail).

The refresh is unconditional rather than guarded by a `lax.cond` on "did anything move". The branch
would save one gradient per iteration — against the tens a trajectory spends — only on a chain
whose labels are already stuck, which is a chain whose numbers should not be trusted anyway; and an
unconditional refresh leaves no state in which the cache and the labels can disagree.

For the same reason, `_after_discrete` discards the sweep's own `log_prob` and recomputes it as
`-sum(potential_values)` — that is how `BaseHMC.kernel` defines it, and the two differ in the last
bits by summation order. Keeping a single definition keeps the next sweep's first acceptance ratio
honest.

### Dtype, and float64

The discrete block is **`int32` always**, not the canonical integer dtype. That is deliberate: a
label never needs 64 bits, and pinning it makes the state's dtype independent of
`jax_enable_x64` — which matters because the kernel is jitted once and a field whose dtype changes
mid-run breaks the shape/dtype invariant of `01_state_and_kernel.md`. Every arithmetic step in the
sweep casts explicitly (`.astype(jnp.int32)` on the proposal offset) so nothing promotes.

Verified under x64: the continuous block comes back `float64`, `state.discrete` stays `int32`, and
the sweep behaves. There is deliberately no *test* for this — enabling x64 must happen before
`import mimcs` and leaks into the whole pytest process, shifting every float32-margin statistical
test in the suite (which is why `tests/conftest.py` refuses to collect the x64 scratch
directories).

### RNG

Two `(sweeps * discrete_dim,)` uniform draw components, added **only when the model has discrete
parameters**. This is not tidiness: `RNGBuffer` splits its key into one subkey *per draw component*,
so adding a component renumbers every other component's stream. Adding none on a continuous model
is what keeps every existing run bit-identical.

### Cost, measured

One full log-density evaluation per discrete coordinate per sweep, plus one to seed the sweep and
one gradient to refresh the caches. Each is gradient-free and so cheaper than a leapfrog step, but
the count is `discrete_dim`.

Measured on the mixture, against the *same* model with the labels baked into the closure as data —
so the continuous block, its dimension and its geometry are identical and the sweep is the only
difference (200 sampling iterations, ~64 gradient evaluations per iteration in both arms):

| labels | with sweep / without | compile |
|---|---|---|
| 10 | 1.09x | 1.06 s |
| 30 | 1.17x | 1.09 s |
| 100 | 1.20x | 1.11 s |
| 300 | 2.47x | 1.14 s |

**Compile time is flat** across a 30x range of `discrete_dim`, which is the check that the
`fori_loop` traces its body once rather than unrolling — if that column grew, the loop would be
unrolling and the design would be wrong.

The prediction going in was "well under 2x at 30 labels, and *sweep-dominated, several times
slower*, at a few hundred". The first half held (1.17x); the second did not — 300 labels cost
**2.47x**, not several. The reason is that the density is itself `O(n)` in the number of
observations, so growing `n` makes *both* arms more expensive: the sweep is `O(n)` evaluations of
`O(n)` work while the trajectory is ~64 gradients of `O(n)` work, and the crossover therefore
arrives around `n ~ 100` rather than immediately. The trend is real and superlinear
(1.09 -> 1.17 -> 1.20 -> 2.47) and extrapolates to ~8x at a thousand labels — but the measurement
is less alarming than the prediction, in the direction that flatters the change, which is the
direction to distrust. Restricted recomputation (below) is the fix, and it has since landed:
the same measurement now reads ~1x at every size, because the sweep no longer scales with `n`.

## Adapting the proposal: learned marginals

The uniform-over-others proposal wastes work in exactly the case the sweep exists for. In a
`k`-component mixture an ambiguous observation has posterior mass on perhaps two labels, so a
uniform proposal spends `(k-2)/(k-1)` of its attempts on labels of essentially zero density — each
costing a full log-density evaluation, each certain to be rejected.

`DiscreteMarginalAdaptation` learns each coordinate's **marginal pmf** during warmup and proposes
proportional to it, excluding the current value:

    q(a -> b) = p_b / (1 - p_a)

### The Hastings term

That proposal is **asymmetric**, so the plain Metropolis ratio is no longer valid:

    q(b -> a) / q(a -> b) = [p_a (1 - p_a)] / [p_b (1 - p_b)]

    log alpha = dlog pi  +  g(cur) - g(prop),    g(v) = log p_v + log1p(-p_v)

Two properties fall out of that algebra rather than being arranged, and both are load bearing:

* **It is identically zero for a binary coordinate.** With `p_b = 1 - p_a`, `g(a) = g(b)`. So a
  binary parameter has nothing to adapt — not as a heuristic, as an identity — and the adapted and
  unadapted samplers produce *the same draws*.
* **It is identically zero for a uniform table.** So an un-adapted run is unchanged.

Verified before implementation and kept as tests: on the enumerated single-coordinate kernel for an
*arbitrary* Dirichlet-drawn pmf, detailed balance holds to ~1e-18 with the term and breaks by
1e-2 to 1.3e-1 without it.

### The estimator

The library's shared Robbins–Monro gain (`_stochastic.rm_gain`) suits a pmf unusually well:

    p_hat <- p_hat + gain * (onehot(z) - p_hat)

is a convex combination of two points on the simplex, so it **stays on the simplex** with no
renormalization and no clipping, and it starts at uniform — the unadapted proposal exactly.

The estimate is then mixed with the uniform, `p = (1 - lambda) p_hat + lambda / n_i`
(`lambda = 0.05`). This is not cosmetic. A zero entry would make a value unproposable, which does
not break detailed balance but **does break irreducibility**: the chain would target the posterior
restricted to whatever it happened to visit during warmup, and no diagnostic here would flag it.
Mixing bounds `p` away from both 0 and 1 — the upper bound matters too, since `g` contains
`log1p(-p)`. `lambda = 1` recovers the uniform proposal exactly, so the knob spans the design;
`lambda = 0` is refused.

### One table per parameter

`state.discrete_proposal_params` is a **dict keyed by parameter name**, not one padded
`(discrete_dim, n_max)` array. A single array would bake in the assumption that every discrete
parameter has a finite enumerable support — exactly what a count-valued `int<lower=0>` breaks,
since its proposal is not a pmf over an enumeration but something like a random-walk scale. Keyed
per parameter, the entry's shape *and meaning* are the parameter type's business, precisely as
`ham_params` lets each kinetic decide what its entry means.

It pays off immediately: every coordinate of one `IntegerParameter` shares its support, so that
parameter's table is exactly `(size_i, n_i)` with `n_i` a **Python int** at trace time. The sweep
is therefore a static Python loop over parameters wrapping a `fori_loop` over each parameter's
coordinates, with a statically sized candidate axis — no padding, no masking. The RNG index stays
the *global* step so the draw order is unchanged from the unadapted sweep.

### Measured

Gaussian mixture, `n = 120`, `sep = 3`, 8 seeds, median over the label coordinates that vary,
against the same sampler without the mixin:

| `k` | moves/iteration | label ESS | cost/iteration |
|---|---|---|---|
| 2 | **1.00x** | **1.00x** | unchanged |
| 3 | 1.93x | >= 2.63x | unchanged |
| 8 | 5.94x | 7.14x | unchanged |

At `k = 2` the draws are **bit-identical across the two arms on all 8 seeds** — the analytic
identity above, confirmed at the measurement level. Divergences were 0 in every arm.

**The prediction going in was wrong, and the way it was wrong is the useful part.** I expected
~1.0x at `k = 3` and wrote that a large gain there would indicate a broken measurement. The
reasoning: a confidently-assigned point has a near-point-mass marginal, so excluding the current
(dominant) value leaves near-uniform weights over the rest and there is nothing to gain. That is
true — and irrelevant, because those coordinates barely move and so contribute almost nothing to
label ESS. Mixing is dominated by the **ambiguous** coordinates, and there the learned proposal
concentrates on the ~1 plausible alternative while uniform spreads over all `k - 1`. The gain is
therefore about `k - 1`, which is what all three measurements show.

Two cautions on reading that table. `ess_1d` returns `min(n/tau, n)`, so a well-mixed label sits at
the **ESS cap**: at `k = 3` the adapted arm is capped on 54% of coordinates even at 12000 draws, so
its ESS ratio is a censored *lower* bound. The **moves ratio is uncensored** and is the statistic
to trust — and it is what tracks `k - 1` most cleanly. And `k = 2` is the null control that the
mistaken `k = 3` prediction was meant to provide: a measurement that reported a gain where the
algebra forbids one would be broken.

## Diagnostics

A discrete parameter's features are **its bare value**, not the continuous default `[x, x^2]`. For
the common binary case `x^2 == x` exactly, so the second block would be a duplicate column that
doubles the multiplicity correction while carrying no information; for a categorical label the
square of a category index is not a quantity anyone reads.

There is **no Stein term**. The Langevin–Stein identity of `11_sample_evaluation.md` integrates by
parts against a density and its score, and a probability mass function has neither. `Model` reports
this per feature through `stein_defined`, and the summary prints a gap and the flag `discrete`
rather than a number; the "k of m flagged" line counts only testable features.

The implementation detail worth knowing: `stein_terms` pads the discrete block with **zeros, not
NaN**. `summarize` drops any draw whose Stein row is non-finite, so a NaN column would discard
*every* draw and silently empty the diagnostic for the continuous parameters too.

Discrete features **do** enter warmup termination — a chain still reassigning clusters is not mixed,
whatever the continuous block is doing. One caveat: a label that never moves has zero variance, so
`ess_1d` returns `n` and `split_rhat` returns 1.0 by their no-variance guards, and a *stuck*
coordinate therefore reads as perfectly converged. The `discrete_moves` diagnostic exists to catch
that, and it is the column to look at first when a discrete model's diagnostics look too good.

## The DSL

`int` was already a registered parameter kind — aliased to `real`, so
`parameters { int<lower=0,upper=1> z; }` compiled to a continuous `BoundedParameter` on a logit
link and was sampled by NUTS. It parsed, it ran, and it was not what anyone writing it meant. That
entry now has a real builder. `int` in a `data` block or a function signature is untouched: neither
reaches a builder, which is why the kind is deliberately *not* `parameter_only`.

Nothing in the lexer or parser changed: `takes_bounds=True` already parsed
`int<lower=..., upper=...>`, and the `array[N]` prefix already worked. Registering the kind is what
reserves the keyword — the registry is the seam, as `model/registry.py` describes.

`categorical(theta)` (1-based, matching Stan and the DSL's own 1-based indexing) and
`categorical_logit(alpha)` are added. Both index with `jnp.take` under an explicit range test, so
an out-of-range label is `-inf` rather than a silently clamped gather — JAX clamps out-of-bounds
indices instead of raising, which would otherwise turn a 1-based off-by-one into a plausible
density.

Using a discrete parameter as an index (`mu[z[n]]`, the whole point of a mixture model) needed no
work: the interpreter does not coerce scalar indices to static integers, and already converts a
numpy constant to a JAX array when the key is traced.

## Under tempering

Parallel tempering is the natural partner for discrete parameters, and for a sharper reason than
"both are about multimodality". A single-site Gibbs sweep moves one coordinate at a time, so two
configurations separated by a low-density intermediate are unreachable from each other however
long the chain runs — the barrier is *structural*, not a matter of step size. Tempering flattens
exactly that.

Each rung holds its **own copy of the labels**: `ProductModel.discrete_dim` is `K *
base.discrete_dim`, and everything discrete works in the `(K, base.discrete_dim)` view, exactly as
`PerTemperatureAdaptation` already reshapes the coordinate. `discrete_block(name)` therefore
returns the block within *one* rung.

The sweep gains a lane axis rather than a tempered variant: `z` is `(L, n)`, the density is `(L,)`,
a coordinate step updates the same column in every lane at once, and lanes **accept
independently** — each rung is its own chain against its own target, as
`IndependentAcceptanceMixin` already treats the continuous half. `L = 1` is an ordinary sampler and
the arithmetic is unchanged. The density comes from one overridable hook: untempered it is
`log_prob_at_coordinate`; under tempering it is `per_temperature_potential`, because a
`ProductModel` deliberately has no `log_prob_at_coordinate` (the ladder it would need is adapted,
so it travels in the Hamiltonian context).

The proposal tables are per rung too — a hot rung's marginal is flatter, and proposing from the
cold chain's concentrated marginal there would fight the exploration tempering exists to provide.
They are **not** exchanged by a swap: a table describes a temperature, not a state.

Placement in the mixin chain is forced twice over. The sweep must sit **inside** the replica
exchange, so each rung sweeps at its own temperature and the swap then moves whole replicas; and
**before** the selection mixins, because `PerTemperatureNUTSMixin.make_draw_components`
deliberately terminates the cooperative chain rather than calling `super()` — anything to its right
is never asked for its RNG draws.

### Measured

The benchmark is `spike_and_slab`: two near-collinear predictors, an inclusion indicator each, and
a sparsity prior. The two single-predictor models carry ~0.485 and ~0.514 of the posterior while
the state joining them carries ~2.9e-4, so a single-site sweep crosses about once in 2000 attempts.

Over 8 seeds, 8000 draws:

| | plain Gibbs | PT (K=6) |
|---|---|---|
| per-seed `p(1,0)` | **0,0,0,0,1,1,1,1** | 0.40–0.59 |
| mode crossings / run | median **0** | median **1286** |
| max \|freq − exact\| | 0.500 | (see below) |
| **split R-hat on labels** | **1.0000, 8/8 below 1.01** | 1.02–1.20 |

Every plain seed is *completely* trapped — four in each mode, the frequency 100% wrong — while
**R-hat reports a perfect 1.0000 on all eight**. A coordinate that never moves has no within-chain
variance to betray it, so the mixing diagnostic is blind to a maximally wrong answer. That is the
single most useful number here, and it is why `discrete_moves` exists.

PT is **unbiased**: over 8 seeds at 20000 draws, `p(1,0) = 0.5117 ± 0.0273` against an exact
0.5143 — a deviation of **0.09 standard errors**.

A caution earned the hard way. The per-seed standard deviation is 0.077, because the chain switches
modes in bursts and the effective number of independent mode observations is far below the raw
switch count. Reading *per-seed* errors of 0.08–0.20 as evidence of bias, and then treating a
four-seed mean as converged, cost a long detour before eight seeds showed the null. The oracle was
suspected too, and cleared independently: 2-d quadrature of the model's own density reproduces the
analytic marginalization to 4e-6. Exactness itself is asserted on the small enumerable targets,
where it holds robustly (1.4e-3 with a continuous parameter interacting with the labels), rather
than on this one.

## Restricted recomputation

The sweep used to evaluate the **whole** density once per discrete coordinate: `1 + discrete_dim`
evaluations per kernel call, which is what made a model with hundreds of labels sweep-dominated.

Two restrictions, and the second is the one that matters.

**Component-restricted.** A coordinate's Metropolis ratio needs only the components that read its
parameter; every other component contributes the same amount to both sides of the difference and
cancels exactly. `Model.component_reads` carries the per-component read sets --- the DSL's
`ComponentSpec.reads`, which the cost rule already computed. A component with **no recorded reads
counts as reading everything**, so a hand-written model is unaffected in either direction. The
chart Jacobian is dropped unconditionally and for free: a discrete parameter may not be a chart's
parent (`_validate_discrete`), so `total_log_jacobian` cannot depend on a label.

**Coordinate-restricted**, via scan components. On its own the component rule buys **nothing** on
the motivating model: the mixture has one component, and it reads `z`. A component is an opaque
scalar --- nothing in a JAX closure says it is a sum of per-observation terms. `model lik scan(z, y)`
declares exactly that (doc 08), and then moving `z[j]` needs only element `j`.

### The delta, not the total

The sweep no longer carries a running log density. It could not: a restricted `lp_prop` cannot be
compared against a full `lp`, and restricting both would still cost two evaluations per coordinate.
So it forms the **difference** directly, and the total it never needed is never built. One full
evaluation at the exit replaces the seed *and* the `n` in-loop ones --- and it also removes an
accumulation of `n` float32 increments that would have drifted.

Under tempering the difference is per rung and each component keeps **its own weight**:
`tempered=` may name a subset, so a restricted sum scaled by one global beta is right only when
every component happens to be tempered. Routing through each potential's `per_temperature_*` makes
that structural. The tempered path also hands each lane its own continuous values, stacked on a
leading `K` axis --- a Python list cannot be indexed by a traced lane, and reusing rung 0's values
would evaluate every rung at the cold chain's position. Both failure modes have a test with a
control, because neither produces anything but a plausible number.

### The fallback is verbatim

When no component can be skipped and none is elementwise in the swept parameter, the sweep runs the
original code unchanged. Every model that existed before this --- none has a scan component ---
keeps its draws **bit-for-bit**, verified across a hand-written Ising target, the mixture, a
factory-built sampler and a tempered spike-and-slab. The switch is per *model* rather than per
parameter, because the two paths carry different state and mixing them would mean carrying both.

### Measured

The mixture, `for`-loop model against scan-component model, same posterior:

| n | form | warmup + compile | sampling |
|---|---|---|---|
| 30 | loop | 5.8 s | 672 us/draw |
| 30 | scan | 2.1 s | 325 us/draw |
| 150 | loop | 28.6 s | 6822 us/draw |
| 150 | scan | **3.1 s** | **671 us/draw** |
| 600 | scan | 2.8 s | 708 us/draw |

**10.2x faster sampling at n=150, and the cost is now flat in `n`** --- 600 labels cost 5% more
than 150, not 4x. `examples/05_mixture.py` runs in 9.8 s where it took 74.5 s, printing identical
numbers. The `O(n^2)` hazard the per-coordinate array write would have become, once the density
stopped dominating, did **not** materialize: the single-column update lowers to an in-place dynamic
update inside the loop. Worth checking rather than assuming --- it is exactly the kind of cost that
only becomes visible after the original bottleneck is removed.

## From the factory

`make_sampler(model)` builds all of the above. The refusal that stood here is lifted; doc 09 holds
the rules and the reasoning, and three points belong on this side of the seam.

**The sweep is not a spec field.** A model with integer parameters always gets it. Every other
choice the factory makes is a trade-off between samplers that are all correct; holding the labels
frozen is not one of those, and it is invisible in every diagnostic the library prints. So there is
no knob for it, and `BaseSampler`'s `handles_discrete` check backstops the composition.

**What *is* a choice is the update method**, per parameter, and it is decided by support width
together with whether the parameter is *elementwise*. `spec.discrete` carries one `DiscreteSpec`
per integer parameter — `kind` plus `params`, the same shape `BlockSpec` uses for a kinetic — and
`discrete_update_rule` fills it in. The decision table is in doc 09.

Three things about it belong on this side of the seam.

*The thresholds are imported, not restated.* `WIDE_SUPPORT` lives in the mixin that warns at it and
`EXACT_MAX_VALUES` / `EXACT_MAX_VALUES_ELEMENTWISE` in the module implementing exact Gibbs, so the
number a rule gates on and the number the code is built around cannot drift apart. The last two
coincide with `WIDE_SUPPORT` in value and are deliberately not defined in terms of it: one prices
"a table this wide cannot be estimated from the draws", the other "this many restricted evaluations
are affordable".

*The widest parameter no longer decides for the whole model.* It used to, because
`_postprocess_hooks` allocated and updated every table in one pass and could not skip one. The
adaptation now owns only the parameters whose method reads a table, so a narrow parameter beside a
wide one keeps its learned marginal.

*The uniform placeholder is still where the deferred proposals attach.* An ordinal ±1 walk and a
count-valued jump are new values of `DiscreteSpec.kind`, and the state they would write is already
per parameter (`state.discrete_proposal_params`, above) with its shape and meaning the parameter
type's business. The warning exists so the gap stays visible: the uniform proposal on a 200-valued
coordinate spends 198/199 of its attempts on values of essentially zero density.

**A discrete-only model gets `StaticContinuous`.** That class was written here so the sweep could
be tested against an exactly enumerable target with the continuous block frozen; it turns out to be
exactly what `coord_dim == 0` needs in production too, and the factory selects it (with the step
size and mass switched off in the same rule).

One thing measured while wiring this, worth recording because it makes an obvious test vacuous:
the relative MRO order of `DiscreteMarginalAdaptation` and `SystematicScanMetropolisWithinGibbs`
**cannot change the draws**. They touch disjoint hooks --- the sweep composes on `kernel`, the
adaptation writes tables in `_postprocess_hooks` --- so swapping them is bit-identical at `k = 2`
*and* `k = 3`. "Compose it left of the sweep" is a readability convention. What actually constrains
the draws is the sweep sitting left of the *base algorithm*, and under tempering inside the replica
exchange.

## Per-parameter update methods

The sweep supplies the scan, the lane axis, the RNG indexing and the restricted density. *How* one
coordinate moves is a :class:`DiscreteUpdate` held per parameter in `sampler.discrete_updaters` ---
the discrete peer of `BaseHMC.kinetics`, where each block owns its own kinetic and each adaptation
filters the list down to what it owns. `MetropolisUpdate` is the method above; `ExactGibbsUpdate`
is the second; a custom jump operator will be the third.

Objects rather than a `{name: method}` dispatch, because a jump operator moves *continuous*
parameters alongside the label and carries `|det dT/dx|` in the ratio: that changes the carry, the
acceptance ratio and the per-parameter configuration. A string cannot hold `T`, and an `if` in the
sweep body cannot hold a different carry.

### Exact conditional Gibbs

Draw the coordinate from its exact conditional over all `n_i` values instead of proposing and
accepting. It is built entirely out of **differences** against the current value: a softmax is
shift-invariant, so

    p(v) = softmax_v [ log pi(v) - log pi(cur) ]

and the current value's own entry is identically zero. Three things fall out of that rather than
being arranged.

* It needs **no density hook of its own**, so the tempered override of `_discrete_delta` makes the
  per-rung path correct with nothing added, and the component/scan restriction applies for free.
* The `-inf` guard is forced. Under Metropolis a `NaN` delta compares false and rejects; inside a
  softmax it would propagate, make every cumulative comparison false and select index 0 --- a
  silent stay-put reporting acceptance 1.00. Non-finite entries are mapped to `-inf`, and the
  zero anchor guarantees at least one finite entry, so the draw is always well defined. The draw
  does therefore condition on `log pi(cur) > -inf`.
* `discrete_accept_prob` reads **1.00** for such a coordinate, because a Gibbs draw's acceptance
  probability *is* 1. That is the honest number, and it means `discrete_moves` is the only column
  left that can catch a frozen label there.

The candidate axis is `vmap`ped, not looped: `n_i` reaches 64 on the elementwise path and a Python
loop would put that many copies of the density into the `fori_loop` body. Measured flat --- 19
jaxpr equations at both `n_i = 3` and `n_i = 64`. It also does **not** cost a second evaluation of
the `cur` side of each difference, which was the worry: `cur` is not a batched operand, so `vmap`
leaves that half unbatched and it is computed once (checked in the jaxpr --- the primitive appears
exactly twice at every `n_i`, once batched and once scalar).

**Not for a narrow support**, which is the opposite of what the cost argument predicts and is the
main thing the measurement changed. Evaluating every candidate is *cheapest* when `n_i` is small,
so exact Gibbs was expected to pay off there; it loses there instead, and wins by a margin that
**grows** with the support.

The reason is that the Metropolis arm's learned table estimates a coordinate's **marginal** while
the draw needs its **conditional**. Those coincide on a narrow support and drift apart as it
widens, so the proposal degrades with `n_i` and an exact draw does not. At `n_i = 2` the proposal
is forced and *is* the restricted conditional, which makes the domination provable rather than
measured: Metropolis moves with probability `min(1, pi_b/pi_a)` against Gibbs's `pi_b` — Peskun,
at asymptotic-variance ratios of 5.0 at `pi_a = 0.6` and unbounded at 0.5, for *half* the density
evaluations. At `n_i = 3` the same shows end to end (0.91x label ESS, 8 paired seeds). From 4 up it
reverses: 1.30x / 1.28x / 1.98x / 2.91x at `k = 4 / 5 / 8 / 16`.

So the factory rule carries a **floor** (`EXACT_MIN_VALUES = 4`) as well as caps. Spike-and-slab
indicators sit at `n_i = 2` and stay on the better kernel. The runtime warns rather than refusing a
hand-built narrow exact updater --- it is worse, not wrong. See
`tests/experiments/writeups/discrete_exact.md`.

### The carry, and what stays bit-identical

Exact Gibbs never forms a running total, so it puts the **whole** sweep on the delta path
(`restriction_plans(force=True)`), where a parameter that gains nothing from component analysis
runs with every component slow and costs one extra evaluation per coordinate. That is the price
already paid for not carrying two kinds of state. A model whose every parameter is
Metropolis-updated *and* whose components offer no restriction still runs the original code, so its
draws are unchanged --- pinned bit-for-bit against draws captured before the old path was deleted
(`tests/data/golden_discrete.npz`).

The RNG layout does not depend on the mix of methods. An exact draw needs one uniform where
Metropolis needs two, and it reads `discrete_proposal[t]` and leaves `discrete_accept[t]`
**unread** rather than reusing it: both components stay allocated at unchanged shapes, and both
methods index by the same global step. Dropping the unused component would renumber every other
stream in the library.

### A latent bug this exposed

`_discrete_delta`'s `index` is the coordinate's position within *its own* parameter's block, but
every caller passed the model's flat index. The two coincide for the **first** discrete parameter,
which is every model that had a restriction plan before this, so it never showed. A second
parameter indexed past the end of its own array --- where `.at[i].set` **clamps** rather than
raising, so the wrong element moved and nothing reported it. It surfaced only once exact Gibbs
forced multi-parameter models onto the delta path, and then only through a mixed-model posterior
test whose all-Metropolis arm passed while every arm with an exact updater failed.

### Custom jump operators

The motivating case is a regression of spatial fields with spike-and-slab priors on the explanatory
fields and a Gaussian process for the error term. Switching an explanatory field on changes the fit
too much to be accepted --- but compensating with a matching, opposite change in the GP makes the
same jump routine. So a jump moves continuous parameters *alongside* the discrete coordinate, and
the acceptance ratio carries the Jacobian of that map:

    alpha = min(1, [pi(z', T(x)) |det dT/dx|] / pi(z, x))

A `JumpOperator` lives on the `Model` (`mimcs/model/jump.py`), so the sampler factory needs no
knowledge of it --- an operator is a property of the *model*, like a scan component, not a sampler
option. The DSL's `proposal` block is what produces one; see `docs/reference/model_dsl.md`.

**A jump is a modifier, not a `kind`.** It changes how a candidate is *evaluated*, not how one is
*chosen*, so it composes with both existing methods rather than becoming a third.
`build_discrete_updaters` substitutes the jump-aware variant of whichever method the parameter asked
for, which is what keeps `DiscreteSpec.kind` meaning the same thing with or without an operator.

#### Two balance conditions, not one

Writing `Phi_{a->v}` for "set the label to `v` and apply the map, from current label `a`":

* **the involution** `Phi_{b->a} . Phi_{a->b} = id`, which Metropolis needs, because the reverse
  move is this same operator run at the current value --- and it is why the ratio carries a single
  `|det|` rather than a forward and a reverse term;
* **the cocycle** `Phi_{b->v} . Phi_{a->b} = Phi_{a->v}`, which exact conditional Gibbs needs on
  top. It is the group-action condition of Liu and Sabatti's generalized Gibbs sampler, and it is
  what makes the orbit --- and so the weight vector up to a common factor --- the same seen from
  every member.

**The gap between them is real, and was measured rather than assumed.** A map negating a coordinate
whenever the label changes satisfies the involution and not the cocycle. It samples **correctly**
under Metropolis and **wrongly** under exact Gibbs: 0.003 against 0.076 on the label marginal, a 27x
gap. So the cocycle is demanded only of a parameter actually on an exact update; demanding it of
every operator would reject correct models.

Neither condition is statically checkable and neither failure is detectable downstream, so both are
checked **numerically at sampler construction and raise**. Not in `_initialize_hooks`:
`initialize()` is optional, so a check there would silently never run for a user who goes straight
to `warmup()` --- the same class of silence it exists to prevent. Probe points are the initial
coordinate plus random perturbations, because the charts' origin is often all-zeros and a
multiplicative map is accidentally involutive there.

*The identity at the current value holds only to rounding.* The idiom this library recommends,
`x + effect(g) - effect(z[j])`, evaluates as `(x + effect(a)) - effect(a)` when `g == a`, which
rounds twice and lands ~6e-8 away in float32. Requiring exactness would reject the canonical map
unless its author happened to parenthesise the difference first. So the check is by tolerance, and
a drawn value equal to the current one **skips the map entirely** --- otherwise a stay-put draw
would random-walk the continuous block by an ulp per sweep, with no acceptance test anywhere to
stop it, since nothing was proposed.

#### Volume, and what the declaration buys

An operator declares itself volume preserving by default, which a compensating shift is; the ratio
then carries no Jacobian term and the map costs only its own arithmetic. That default is what makes
the motivating problem affordable at all: the general path is `m` tangents plus an `O(m^3)`
determinant **per candidate**, which a GP field of a few thousand coordinates puts out of reach.

The claim is verified once at construction (affordable precisely because it is not per candidate),
and above 64 output coordinates it is warned about rather than checked. It matters: a scaling map
wrongly declared preserving biased the label marginal by ~5 standard errors over 6 seeds, with every
diagnostic looking ordinary --- and a **shift** map cannot detect the fault at all, because dropping
the Jacobian leaves a volume-preserving map correct. Every control on the Jacobian therefore runs on
a scaling map.

#### What moves, and what must not

The map is written in *sample* space and applied to the *coordinate*, with the Jacobian taken in
coordinate space --- which is what makes it correct with no separate chart-Jacobian term, since
`log_prob_at_coordinate` already carries that.

Only the output blocks are written back. The obvious spelling, unpack-substitute-repack, is wrong:
`to_coordinate(from_coordinate(x))` is not bitwise identity for a nonlinear chart in float32, so a
repack would perturb every *other* parameter in its last bits --- an unbiased-looking random walk on
everything, and balance checks failing for reasons unrelated to the map.

Three static refusals, each because the runtime failure is silent. An output may not be a
**projecting** chart (`unit_vector`, `simplex`, a doubly-bounded `ordered`, the matrix types): those
accept an off-manifold value and quietly project it back, violating the involution by exactly the
projection error while reporting a finite density. An output may not be a **chart parent**, or the
map would move a child's ambient value while its coordinate stands still. And an output may not be
discrete, which is deferred rather than wrong.

#### The carry, and what it costs

The sweep carry grows to hold the coordinate. Carrying an untouched array through a `fori_loop`
costs nothing, so every method takes the wider carry; what is *gated* on `moves_coordinate` is the
per-coordinate rebuild of the unpacked continuous values, which are otherwise computed once per
sweep and would be stale from the first accepted jump. A model with no operator therefore runs the
original path and is pinned bit-for-bit against `tests/data/golden_discrete.npz`.

A jump takes the **full-density** path, and the reason is worth recording rather than treating as
laziness: `_discrete_delta` deliberately omits the chart Jacobian because it cancels for a
label-only move. Under a jump it does not cancel, so the restricted path is not merely unhelpful but
*unsafe*. Restricted recomputation for jumps is deferred with that as its blocker --- and it is also
why the factory rule stops granting a scanned parameter the wide elementwise exact-Gibbs cap once it
carries an operator: each candidate is now a whole density, not `O(1)` element work.

**Three caches go stale**, all silently. `state.sample` is the worst: it is what `_retained_sample`
records, so leaving it means every stored continuous draw is the pre-jump value --- a wrong
posterior behind clean traces, reading as "my jump isn't helping". Two adaptations also read it and
write it straight back, welding an inconsistent pair into the state. The potential caches need only
the moved coordinate, which `_reseed_caches` already takes as an argument.

**Tempering needs nothing new.** Every density goes through `_discrete_log_prob`, so the per-rung
override resolves through the MRO --- the same reason `SweepEnv` carries the sampler rather than
bound copies of its hooks. The map runs per lane over the *base* model's layout, and the Jacobian
enters each rung **unscaled by beta**: it is a property of the state map, not of the density.

## Ordinal and unbounded integers: the random walk

An `int` needed both bounds because every method enumerated the support. A count (`int<lower=0>`)
has none to enumerate, and a wide *ordered* support — a change point over 200 positions — has one
that "propose among the others" wastes almost entirely. Both want a proposal that steps to nearby
values. `RandomWalkUpdate` is that method (`kind = "random_walk"`), and it is the only one that can
move a parameter with an open side.

### The type

`IntegerParameter` takes `lower=None` / `upper=None`. Python-side, `lower_value` / `upper_value` /
`n_values` become `None` — deliberately not a sentinel, so every consumer doing support arithmetic
(a table width, a candidate count) fails loudly rather than building something `2³¹` wide. The JAX
bound arrays *do* carry a sentinel, `±INT_BOUND = ±(2³⁰ − 1)`: that makes the clamp arithmetic
overflow-free in int32 by construction (`upper − cur ≤ 2³¹ − 2`) and lets the sweep clamp every
parameter the same way. `ordinal` is declared, or implied by an open side. A randomised start draws
from `init_range()` — the support when bounded, otherwise four values next to the finite bound or
`[−2, 2]` — mirroring `UniformInit`'s `U(−2, 2)` rather than drawing over `±2³⁰`.

### The proposal, and its Hastings term

A fair coin picks the direction; the step is `Geometric(p)` on `{1, 2, …}` with `1/p = 1 + exp(ρ)`,
per coordinate and per lane. A step past a bound **clamps to the endpoint**, which then collects the
whole geometric tail — `(1−p)^(d−1)` against an interior value's `p(1−p)^(d−1)` — so the proposal is
asymmetric exactly at the bounds, with a closed form:

    log q(b → a) − log q(a → b) = log p · (1[b at a bound] − 1[a at a bound])

Verified before implementation on an enumerated kernel (`tests/experiments/discrete_random_walk_kernel.py`):
the formula matches the proposal matrix to 1.2e-14, and detailed balance on Poisson(3) holds to
7e-18 with it and fails by 2.4e-2 without it. The tests keep both, with the kernel built from the
real `_propose` over a grid of uniforms. An open side's sentinel counts as a bound: it is a real clamp.

**No new RNG draw component.** The sweep already draws one proposal uniform per coordinate; the
direction is `u < ½` and the step comes from `u' = 1 − frac(2u) ∈ (0, 1]` by inversion. Adding a
component would renumber every seeded stream in the library. Halving the uniform's resolution leaves
the law symmetric in `|b − a|`, the same u-grid granularity the uniform proposal already documents.

Because it subclasses `MetropolisUpdate` and overrides only `prepare` / `_propose` /
`_log_hastings`, the Metropolis `step` — full path, restricted path, and tempering's per-rung
overrides — is reused unchanged, and a **jump operator composes by MRO alone**:
`JumpRandomWalkUpdate(JumpMetropolisUpdate, RandomWalkUpdate)` takes the jump step from the one and
the walk's proposal from the other. The balance checks enumerate `balance_values()`, eight values
next to the finite bound when a side is open.

### The scale adaptation, and the no-op it must ignore

`DiscreteRandomWalkAdaptation` moves `ρ ← clip(ρ + γₙ(ᾱ − 1/3), ρ_min, ρ_max)` during warmup, after
Vihola's robust adaptive Metropolis, with the step-size mixin's gain schedule. The per-coordinate
statistics ride in the sweep carry (a sixth slot, `{}` for every other method, so other models are
bit-identical — the golden file pins it) and land in the parameter's own proposal entry,
`{"log_scale", "accept_sum", "n_proposed"}`: doc 14's "the entry's shape and meaning are the
method's business", cashed in.

**1/3** is the IACT-optimal acceptance for a Laplace target (0.325 on the exact kernel). For
Gaussian-shaped targets the optimum is ≈0.44, but aiming at 1/3 there costs ~8% IACT, so one default
serves both.

**`ᾱ` averages genuine proposals only.** At a bound, the outward coin proposes the current value.
Its acceptance of 1 carries no information about the scale, and counting it is not a small bias —
on the exact kernel:

| target | best IACT | no-ops counted | no-ops excluded |
|---|---|---|---|
| Poisson(0.1) | 2.71 | signal ≥ 0.45 at every ρ → runs to the cap → IACT 3.2e5 | signal ≤ 0.17 → the ±1 floor → IACT 3.40 |
| Poisson(0.5) | 3.00 | root ρ≈3.3 → IACT ≈ 12 | root ρ≈1.05 → IACT ≈ 3.1 |
| Poisson(1.0) | 3.17 | root ρ≈2.6 → IACT ≈ 7 | root ρ≈1.05 → IACT ≈ 3.2 |

Poisson(0.1) is the case with no root at all: most of the mass sits on the bound, and the walk
degrades to ±1 steps, which the warmup-end report names as the intended fallback. A 4-seed smoke run
of the implementation reproduced the roots: Laplace b = 15 at ρ 3.88–4.20 (exact 4.0), Poisson(3) at
1.32–1.50 (exact 1.4), Poisson(0.1) at the floor.

### Measured

A change point over a 200-value support, `ordinal` against the uniform-over-the-others proposal the
factory would otherwise leave, 8 paired seeds:

| data | exact posterior sd of tau | walk / uniform ESS/s | wins | acceptance, uniform → walk |
|---|---|---|---|---|
| shift 1.5 | 2.8 | **6.54×** | 8/8 | 2.7% → 33.5% |
| shift 0.8 | 29.0 | 0.78× | 3/8 | 15% → 33% |

The support width is 200 in both rows; only the posterior's locality differs. Where it spans half the
support, far uniform jumps act like an independence sampler and beat a diffusive walk that moves
twice as often — which is why `ordinal` is a declaration the factory honours rather than something
it could infer from the width. On binomial-`N` estimation (an open-sided count) the adapted `ρ` lands
within 0.25 of the exact kernel's 1/3 root on 8/8 seeds, at a median 11% over the best achievable
IACT; see `tests/experiments/writeups/discrete_random_walk.md`.

### From the factory, and the DSL

`ordinal int<lower=1, upper=T> tau;` declares the ordering; an open side implies it.
`discrete_update_rule` gives every open-sided parameter, and every ordinal one with at least 3
values, `random_walk` ahead of the width rules. Two values is the exception: an ordering is vacuous
there and the walk would halve the move rate against the Peskun-optimal flip. An open-sided parameter
is excluded from learned-metric dependencies, since both encodings need a finite support.

### Starting scale from evidence

With labels in the evidence, the rule also sets each walk coordinate's starting `ρ` so that the
proposal's width `2/p` (mean jump left to mean jump right) equals the coordinate's interquartile
range: `ρ = log(IQR/2 − 1)`, floored at `RW_LOG_SCALE_MIN` (the ±1 walk) when the IQR is 2 or less.
Quantiles rather than a standard deviation, because the pilot need not have mixed or be light-tailed,
and the scale only has to be the right order of magnitude. On exact kernels it lands 1.3–1.8 below the
IACT-optimal `ρ` on Gaussian and Laplace targets (sd/b 3–60) — a mean jump 4–6× short — while the
no-evidence start of 0 is up to 5.5 below on the wide ones (fixed-scale IACT 10.8 against 2471 at
sd 60).

End to end it matters only when the warmup is short, because the adaptation's gain
`(n + 5)^−0.6` decays slowly enough to travel ~16 in `ρ` within 100 sweeps. Four unbounded Gaussian
coordinates (sd 3 / 30 / 300 / 3000), 8 paired seeds, evidence = exact draws: at warmup 30 the
sd-3000 coordinate's ESS is **2.44×** (8/8) and the others neutral; at warmup 100 and 1000 every
median ratio is 0.92–1.02. It is a cheap head start for a far-out scale, not a mixing improvement.

## Random scan

`RandomScanMetropolisWithinGibbs` runs `discrete_jumps` jumps per iteration (default
`discrete_sweeps * n`, one per coordinate). Each jump draws one coordinate uniformly from **all**
discrete coordinates, so a parameter is picked in proportion to its size, and moves it with that
parameter's own `DiscreteUpdate`. The updaters, proposals and adaptations are the systematic scan's,
unchanged. The choice is independent of the state, so every jump is a reversible `pi`-invariant
kernel and so is the product of an iteration's i.i.d. jumps; the fixed order of the systematic scan
is only `pi`-invariant. It is also the base for blocked updates: a block is another unit a jump
can pick.

Four points of design:

- **The scan owns the RNG indexing.** `DiscreteUpdate.step(env, prep, t, c, carry)` takes its draw
  row `t` from the scan instead of computing it. The systematic scan passes `s * n + start + c`,
  the formula the updaters used to compute, so its draws are bit-identical; the random scan passes
  the jump number.
- **The coordinate is shared across lanes.** Under tempering every rung updates the same column in
  a jump. Valid for the same reason as the scan itself --- the choice is independent of every rung's
  state --- and it keeps "update column `i` in every lane" the updaters' contract.
- **Dispatch is a `lax.switch`** over the parameters' updaters, one branch per parameter, each still
  statically sized by its own support. No switch at all for a single parameter.
- **Its RNG layout is its own:** `discrete_proposal` / `discrete_accept` at `(J, L)` plus
  `discrete_index` at `(J,)`. The extra component exists only in this class, so no systematic
  stream moved.

The adaptations need nothing new. A coordinate proposed twice in an iteration contributes two
acceptances to the random walk's statistics, and one not picked has `n_proposed = 0`, which the
adaptation already masks. The factory reaches it through `spec.discrete_scan = "random"` (and
`parallel_tempering(discrete_scan="random")`); no rule selects it.

**Measured**, at the same budget (`J = n`), 8 paired seeds: on 60 weakly coupled mixture labels the
random scan gets **0.50×** the systematic label ESS (0/8; the systematic arm is censored at the draw
count, so this is an upper bound) and 0.79× on the shared continuous shift. That is the refresh
arithmetic: a coordinate is revisited with probability `1 − e⁻¹` per iteration, an IACT of
`(1 + e⁻¹)/(1 − e⁻¹) ≈ 2.2` against i.i.d. On a single-coordinate change point the two scans are
bit-identical. Reversibility costs about half the ESS on independent coordinates, which is why the
systematic scan stays the default; see `tests/experiments/writeups/random_scan_gibbs.md`.

## What is deferred

Each of these has a place to attach, listed so it lands as a fill-in.

**Component-restricted recomputation** --- *now supported*; see "Restricted recomputation" above.

**Custom jump operators** --- *now supported*; see "Custom jump operators" above.

**Exact conditional Gibbs** --- *now supported*; see "Per-parameter update methods" below.

**Random-scan updates** --- *now supported*; see "Random scan" above.

**Blocked updates.** Several coordinates at once matter when labels are strongly coupled, as in a
hidden Markov model where a forward-backward sweep is the right move. They attach to the random scan
as another unit a jump can pick, and wait for problems to try them on; so does any factory rule
choosing between the two scans.

**An ordinal random walk** and **count-valued integers** --- *now supported*; see "Ordinal and
unbounded integers" above. Still open from that arc: an open-sided parameter as a learned-metric
dependency, and heavier-tailed step families as `params` of the same kind.

**Discrete-aware learned metrics** --- *now supported*; see doc 07 and doc 09.

**Parallel tempering** — *now supported*; see "Under tempering" above.

**Discrete Stein diagnostics.** There are Stein operators for discrete distributions, built from a
difference operator in place of the derivative: for a target on `{0..n-1}`, `A f(x) = f(x+1)
pi(x+1)/pi(x) - f(x)` is mean-zero under `pi`. It would slot straight into the existing per-feature
machinery via `stein_defined` and would give discrete parameters the same target-aware check the
continuous ones get.

**Chart indices.** The link back to `04_manifold_parameters.md`: `state.chart_indices` is a tuple of
scalar integer arrays that nothing writes. Making it a genuine discrete parameter would require the
no-discrete-parent restriction lifted first, since a chart index is *by definition* a chart's
parent. That ordering is the useful thing to record — the atlas work depends on the restriction
above, not on anything else here.

**Factory wiring** --- *now supported*; see "From the factory" above.
