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
  over any continuous base algorithm, plus `StaticContinuous` for a discrete-only model.
- `categorical` / `categorical_logit` in the DSL.
- Bare features and no Stein term for a discrete parameter.
- A **refusal** from the sampler factory and from parallel tempering.

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

`DiscreteMetropolisWithinGibbs` is a **kernel-composing mixin**, a new mixin category. Every other
mixin cooperates through the `_*_hooks` chain and never touches `kernel`; this one overrides
`kernel` and calls `super().kernel`:

```python
cls = make_sampler_class(RobbinsMonroStepSize, DiscreteMetropolisWithinGibbs, NUTS)
```

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
direction to distrust. Component-restricted recomputation (below) remains the fix; it is just less
urgent than predicted below ~100 labels.

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

## What is deferred

Each of these has a place to attach, listed so it lands as a fill-in.

**Component-restricted recomputation.** The sweep currently re-evaluates the *whole* density per
coordinate. A model knows its components (`log_prob_fns`) and the DSL knows which names each
component reads (`semantics.read_names`, already used by the cost rule). A coordinate's Metropolis
ratio only needs the components that read its parameter — for a mixture, the likelihood term for
observation `n` alone. This is the difference between `O(discrete_dim * N)` and `O(discrete_dim)`
per sweep and is the most valuable deferred item by a wide margin.

**Custom jump operators.** The motivating case: a regression of spatial fields with spike-and-slab
priors on the explanatory fields and a Gaussian process for the error term. Switching an
explanatory field on changes the fit too much to be accepted — but compensating with a matching,
opposite change in the GP makes the same jump routine. So a "jump" should be able to move
continuous parameters *alongside* the discrete coordinate. The design: a DSL block declaring, per
discrete coordinate, a map `T` on named continuous parameters; the acceptance ratio then carries
the Jacobian of `T`,

    alpha = min(1, [pi(z', T(x)) |det dT/dx|] / pi(z, x))

with `T` required to be invertible and its inverse used for the reverse move. `T` must be
differentiable so `|det dT/dx|` is available by autodiff, which the DSL's expression language
already supports. This is the largest deferred item and the one that most shapes what a `jump`
block should look like, which is why the acceptance ratio is written out here.

**Exact conditional Gibbs.** For small `n_i`, evaluating the density at *all* `n_i` values and
drawing exactly is better mixed than a Metropolis proposal, at `n_i` evaluations against 1. Worth
having as a per-parameter option once component-restricted recomputation makes those evaluations
cheap; the two compose naturally.

**Random-scan and blocked updates.** The scan is deterministic, which is `pi`-invariant but not
reversible. A random scan is reversible and is what a theory-facing user may expect; blocked updates
(several coordinates at once) matter when labels are strongly coupled, as in a hidden Markov model
where a forward-backward sweep is the right move.

**A ±1 ordinal random walk.** The learned-marginal proposal below is the first jump-probability
adaptation, and it treats a coordinate's values as unordered labels. For an *ordinal* integer — a
count, a change point, a discretized scale — a ±1 walk is the natural proposal and a
distance-weighted one the natural generalization. Both are symmetric or nearly so and would slot
into the same per-parameter proposal table (`state.discrete_proposal_params`), whose entry's shape
and meaning are already the parameter type's business.

**Count-valued integers.** `int<lower=0>` with no upper bound has no enumerable support, so it needs
a proposal that is not "uniform over the others" — a ±1 walk or a Poisson-tailed jump. The type
currently refuses it with that reason.

**Discrete-aware learned metrics.** `TODO.md` records the expected form,
`(Exp("discrete") + Exp()) * (Exp("continuous") + Exp())` — a discrete parameter modulating the
continuous metric multiplicatively. This is expected to matter: conditioning on different labels
gives genuinely different continuous geometries, and one mass matrix averaged over them is a
compromise. The score-covariance mass is more forgiving here than empirical covariance would be,
since it averages over modes rather than trying to span the distance between them, which is why
this is deferrable rather than blocking.

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

**Factory wiring.** No rule proposes the Gibbs mixin, and no heuristic knows what a discrete block
costs. `analyze` refuses rather than silently building a sampler that never moves a label.
