# Changelog

## Unreleased

- **Scan components, and a discrete sweep that no longer re-evaluates the whole density.** A
  `model <name> scan(a, b) { ... }` component is evaluated once per element of the named arrays,
  with each scanned name bound to *that element*; the component is their sum, so every continuous
  sampler sees exactly what it saw. What the declaration buys is that moving one label perturbs
  **one** term, which is the structure a single-site Gibbs sweep needs and which nothing in an
  opaque JAX closure can express --- the plain "recompute only the components that read this
  label" rule buys *nothing* on the motivating mixture, whose single component reads `z`. The
  sweep now forms the acceptance **difference** directly rather than carrying a running total
  (a restricted `lp_prop` cannot be compared against a full `lp`), skips every component that
  cannot depend on the label, drops the chart Jacobian unconditionally, and under tempering keeps
  each component's own beta weight so a power posterior stays correct. Measured on the mixture at
  n = 150: **sampling 10.2x faster** (6822 -> 671 us/draw) with the cost now **flat in n** (600
  labels cost 5% more than 150, not 4x), and warmup + compilation 28.6 s -> 3.1 s;
  `examples/05_mixture.py` runs in 9.8 s where it took 74.5 s, printing identical numbers. A model
  with no scan component takes the original code path **verbatim** and its draws are bit-identical,
  checked against `dev` across a hand-written Ising target, the mixture, a factory-built sampler
  and a tempered spike-and-slab. Two measurement notes worth keeping: the aggregate is a `vmap`
  and not a `lax.scan`, because a first benchmark with an almost-empty body put `scan` ahead while
  a realistic one put it **33x behind** at n=150; and the `O(n^2)` array write that would have
  become the next bottleneck did not materialize, which was checked rather than assumed.

- **`examples/05_mixture.py` goes through the factory.** It hand-composed
  `make_sampler_class(RobbinsMonroStepSize, MassMatrixAdaptation, DiscreteMetropolisWithinGibbs,
  NUTS)` because the factory refused a model with `int` parameters until v0.1.7; that is no longer
  true, and this example is the library's documentation for discrete parameters --- exactly where
  unnecessary assembly is most discouraging. It now calls `analyze(model)` and overrides
  `target_accept` on the spec, which demonstrates the prototype seam at the same time, and the
  docstring says plainly that `make_sampler(model)` alone would sample the model correctly.
  Everything the example claims is unchanged except that label moves per iteration go 4.9 -> 9.4,
  since the factory adds the learned-marginal proposal the hand-composed stack lacked.

## v0.1.7

- **The sampler factory builds discrete parameters.** `analyze` / `make_sampler` refused a model
  with `int` parameters; they no longer do. The refusal guarded against a *quiet* wrong answer:
  discrete parameters are kept out of `model.parameters`, so every rule would have partitioned the
  continuous half perfectly well and returned a sampler that never moves a label --- and a frozen
  coordinate has zero variance, so it reports a perfect ESS and split R-hat 1.000. The sweep is
  therefore not a spec field and has no off switch; it is always composed immediately left of the
  base algorithm. Only the *proposal* is a decision, through the new `spec.discrete_proposal`:
  `"marginal"` (the default) adds `DiscreteMarginalAdaptation`, `None` leaves the sweep's
  uniform-over-the-others placeholder, and `discrete_proposal_rule` chooses between them on the
  **widest** support against `WIDE_SUPPORT` (64, imported from the mixin rather than restated),
  warning when it declines because the uniform proposal left standing is itself poor there. A model
  with no continuous parameters gets the new `"static"` base (`StaticContinuous`) with the step size
  and mass switched off, and under tempering `build` adds only the adaptation, since
  `parallel_tempering` injects the sweep itself at a position the flat mixin list cannot express.

- **`Evidence` carries the labels** (`Evidence.discrete`, `(n, discrete_dim)` integer). A discrete
  model's coordinate-space density is *conditional* on the labels, so a recomputed score has no
  meaning without the row's own `z`: `_recomputed_scores` called `log_prob_at_coordinate` without
  them, which raised inside `Model._require_discrete` and was swallowed by the caller, leaving
  `gradients=None` and silently disabling the mass-mode and metric-regression rules --- though only
  off the default path, since a live sampler saves its gradients and the recompute never runs. The
  dimension guard now also compares `discrete_dim`, and a warm start carries the labels alongside
  the position, a fitted configuration paired with reset labels being a state the chain was never
  in.

- **Parallel tempering supports discrete parameters.** PT refused such a model; it no longer does,
  and this is the pairing that matters most for the feature, since a single-site Gibbs sweep moves
  one coordinate at a time and two configurations separated by a low-density intermediate are
  unreachable from each other however long the chain runs --- the barrier is *structural*, and
  tempering flattens exactly that. Four things had to be fixed, of which only the first two were
  known: `pt/kinetics.py` rebuilt each lane's `HamiltonianContext` positionally and dropped
  `discrete`; `ProductModel` tiled one flat *float* vector and hardcoded `discrete_dim = 0`;
  `_swap` permuted only `coordinate`, so replicas exchanged positions while leaving their labels
  behind, which is a state drawn from no rung's target at all; and `TemperedProductPotential`
  vmapped over lanes with the *shared* context, so every rung would have evaluated its density at
  the same labels. The sweep gained a **lane axis** rather than a tempered variant --- `z` is
  `(L, n)`, lanes accept independently, and each rung learns its own proposal table, which a swap
  deliberately does **not** exchange (a table describes a temperature, not a state). Measured on a
  new `spike_and_slab` benchmark, plain Gibbs is trapped on all 8 seeds while split R-hat on the
  labels reports 1.0000 on every one, and PT is unbiased to 0.09 SE.

- **Learned-marginal proposals for discrete parameters** (`DiscreteMarginalAdaptation`). The
  Metropolis-within-Gibbs sweep proposed uniformly over the values a coordinate is not currently
  at, which in a `k`-component mixture spends `(k-2)/(k-1)` of its attempts on labels of
  essentially zero density --- each a full log-density evaluation, each certain to be rejected ---
  so this learns each coordinate's marginal pmf during warmup and proposes from it instead. That
  proposal is **asymmetric**, so the sweep carries a Hastings term, `g(cur) - g(prop)` with
  `g(v) = log p_v + log1p(-p_v)`, which falls out identically **zero for a binary coordinate** and
  for a **uniform** table --- so binary parameters and unadapted runs are untouched exactly. The
  estimate is the library's shared Robbins--Monro gain, a convex combination that stays on the
  simplex with no renormalization, mixed with the uniform (`discrete_lambda`, default 0.05) because
  a zero entry would make a value unproposable and break **irreducibility** without breaking
  detailed balance, which no diagnostic here would flag. Measured over 8 seeds on a Gaussian
  mixture, moves per iteration go 1.00x at `k=2`, 1.93x at `k=3` and 5.94x at `k=8`, at unchanged
  cost per iteration.

- **The discrete proposal's parameters live in a dict keyed by parameter**
  (`state.discrete_proposal_params`), not one padded `(discrete_dim, n_max)` array. A single array
  would bake in the assumption that every discrete parameter has a finite enumerable support ---
  exactly what a count-valued `int<lower=0>` breaks, since its proposal is a random-walk scale
  rather than a pmf over an enumeration --- whereas keyed per parameter the entry's shape *and
  meaning* are the parameter type's business, as `ham_params` already lets each kinetic decide what
  its entry means. It pays off at once: every coordinate of one `IntegerParameter` shares its
  support, so that parameter's table is exactly `(size_i, n_i)` with `n_i` a Python int at trace
  time, and the sweep becomes a static loop over parameters with a statically sized candidate axis,
  no padding and no masking anywhere. The restructured `_discrete_sweep` is **bit-identical** to
  the previous flat one.

- **Documentation caught up with the discrete-parameter work.** Several places were not merely
  silent but *stale*: `docs/index.md` omitted design doc 14 from its table of contents, doc 01's
  `SamplerState` snippets were missing both new fields, doc 06's `HamiltonianContext` was missing
  `discrete`, and doc 04's atlas section still said Metropolis-within-Gibbs was machinery the
  library had none of --- the atlas blocker has since narrowed to the no-discrete-parent rule,
  since a chart index is by definition a chart's parent. Also added: `Model`'s second parameter
  list (doc 05), why a discrete feature gets no Stein z (doc 11), discrete features in warmup
  termination and the stuck-label caveat (doc 10), the kernel-composing mixin category (doc 02),
  and that `int` means something different in a `parameters` block than in `data` (doc 08).

- **Discrete (integer) parameters, and a Metropolis-within-Gibbs sampler for them.** A parameter
  may now be integer-valued --- `int<lower=L, upper=U>`, declarable as an array --- which puts
  mixture models, latent classes, spike-and-slab selection and change points in reach for the first
  time, with the labels **sampled** rather than marginalized out with `log_sum_exp` as a
  gradient-only library must. The state gains a second flat array, of dtype `int`, beside the
  existing float one, and `Model` keeps discrete parameters in a list of their own contributing to
  neither `coord_dim` nor `ambient_dim`; that separation is what kept the change small, since the
  block partitioner, every mass adaptation, the chart machinery and the score pullback all iterate
  `model.parameters` and never see a discrete one. A discrete parameter has **no chart** --- sample
  space is coordinate space, and there is nothing to reparameterize or differentiate.
  `DiscreteMetropolisWithinGibbs` is a new kind of mixin, composing with `kernel` rather than the
  `_*_hooks` chain, so `make_sampler_class(RobbinsMonroStepSize, DiscreteMetropolisWithinGibbs,
  NUTS)` is NUTS that also moves labels with no base algorithm edited; it scans the discrete
  coordinates in order and proposes uniformly among the `n-1` values each is not currently at,
  which is symmetric and so needs no Hastings term. The subtle part is the **gradient cache**:
  `BaseHMC` caches each potential's value and gradient at the current coordinate and the next
  trajectory's leading half-kick reads them back verbatim, so after a sweep those are the
  *previous* labels' gradients and `_after_discrete` must refresh them --- without it nothing
  raises, the acceptance rate stays plausible, and the chain targets the wrong density.

  New: `mimcs.model.IntegerParameter` / `BaseDiscreteParameter`,
  `mimcs.samplers.DiscreteMetropolisWithinGibbs` / `StaticContinuous`,
  `sampler.get_discrete_flat()`, `Model.discrete_parameters` / `discrete_dim`, `categorical` and
  `categorical_logit` in the DSL, the `gaussian_mixture` and `spike_and_slab` test problems and the
  `nuts_gibbs` builder, `examples/05_mixture.py`, and `docs/design/14_discrete_parameters.md`.

- **`int` in a `parameters` block changes meaning.** It was a registry alias for `real`, so
  `parameters { int<lower=0,upper=1> z; }` compiled to a continuous `BoundedParameter` sampled by
  NUTS on a logit link --- it parsed, it ran, and it was not what anyone writing it meant. It now
  builds an `IntegerParameter`, and both bounds are required and must be constant integers. No test
  or example declared one, so nothing in the repo changed behaviour. `int` in a `data` block or a
  function signature is **untouched**: neither reaches a parameter builder.

## v0.1.6

- **Parallel tempering hoists the low-rank Woodbury factors out of its trajectory loop too.**
  v0.1.3 gave `LowRankQuadraticKinetic` a `precompute` whose `O(J^2 d)` Sherman--Morrison recursion
  `BaseHMC.context` lifts to a per-kernel-call constant, but left tempering out: `ProductKinetic`
  defined no `precompute`, *and* rebuilt each lane's context from three positional fields, so the
  cache was neither filled nor forwarded across the `vmap`. It now delegates to the inner block and
  maps the cache along the same temperature axis as `ham_params`, so a lane reads the factors built
  from its own mass, while a block with nothing to precompute returns `None` and contributes no
  entry. The win is real but rank-dependent (rank 32, d=400, K=4: sampling 20.6 -> 15.8 s; rank 4,
  d=400: 6.29 -> 6.19 s) and smaller than the untempered 1.38x, since the removed term is
  `O(J^2 d)` against `O(J d)` for everything that survives. Diagonal and dense tempered runs are
  untouched and **bit-identical**; so, measured, are untempered and tempered low-rank runs.

- **`BaseHMC.context` gains `kinetic_cache=False`, and the reseeding callers use it.** Building the
  cache is free inside the jitted `kernel` but **eager** in `state_at_coordinate` and in the
  tempered ladder reseed, which dispatches the recursion primitive by primitive: 43.8 ms per call
  against 0.10 ms traced, at rank 8, d=200, K=4. Since the ladder reseeds once per warmup
  iteration, hoisting the factors without this opt-out made tempered warmup **3.6x slower**
  (5.4 -> 19.3 s) --- a regression three times larger than the speedup it was buying --- so those
  callers, which read only the potentials, now skip it. The switch is performance-only: every
  consumer keeps its inline fallback, so no number depends on it.

## v0.1.5

- **The NUTS leaf-selection draws are stored flat, saving 16 MiB per sampler.** `leaf_select` was a
  rectangular `(max_tree_depth, 2^(max_tree_depth-1))` table with a row per doubling, but only
  `2^j` entries of row `j` are ever read --- a factor `J*2^(J-1)/(2^J-1) ~ 5` of waste in the
  largest array a NUTS sampler holds. It is now flat, one entry per leaf, indexed at
  `2^depth - 1`: the layout `mimcs.pt` has used since it was written. **20.0 -> 4.0 MiB** in
  float32 and **40.0 -> 8.0 MiB** under x64, taking the whole RNG buffer from 20.9 to 4.9 MiB.
  The saving is permanent for the life of a sampler and independent of the model's dimension, so
  on small models it was most of the buffer. The `line_search` draws a randomized integrator asks
  for get the same treatment.

  > **This changes the random number stream.** Every seeded NUTS run now draws different numbers,
  > so results move --- not by rounding, but to a different chain. Anything pinned to a specific
  > seed will need re-baselining. The sampler itself is unchanged in distribution: verified at
  > 180 000 draws on a 57-dimensional model (mean relative variance error 1e-5) and at 600 000 on
  > a 2-d one, with the `NUTS`/`SimpleNUTS` bit-identity oracle passing throughout.

## v0.1.4

- **A tempered run stores only the cold chain.** It samples the `K`-fold product but reports one
  chain, so `(K-1)/K` of the draw store was discarded at read time — 87.5% at `K = 8`. The
  gradients were already narrowed when saved; the draws now are too (measured 1.83 -> 0.23 MiB at
  `K = 8`). `keep_all_temperatures=True` restores the old behaviour, which `get_samples_all()`
  needs.
- **`summary()` and the termination check allocate less.** `summary()`'s host peak falls 45 -> 35
  MiB at 4000 draws x 200 dimensions, and one termination check at 20 000 history rows falls
  262 -> 165 MiB: the saved gradients no longer take a float64 round trip, `ess` converts column by
  column instead of copying the whole feature matrix, `split_rhat` drops a stacked third copy of
  its segments, the train/validation split gathers into one preallocated array, standardization is
  done in place, and the logistic fit's row buffer is built at the dtype the device will hold.
  Every reported number is **bit-identical** — verified across eight models plus parallel
  tempering, on draws, gradients, all diagnostics and every `Summary` field.

## v0.1.3

- **Chart and ladder adaptation are several times faster.** Recharting a chart, or moving the
  temperature ladder, must refresh the cached potential values and gradients or the next step
  integrates a Hamiltonian nobody is simulating. That reseed ran eagerly, once per warmup
  iteration --- one full `value_and_grad` of every potential dispatched primitive by primitive.
  It is now one shared, compiled helper: an adaptive unit-vector chart takes warmup(500) from
  8.4s to 2.8s, and parallel tempering at K=6 from 4.2s to 2.3s. Compiling the unit-vector chart
  helpers takes vMF a further 2.9s -> 2.1s. Neutral on models that do not recharter. The test
  suite's centering-heavy chunk goes 768s -> 184s on the same idle machine, and the whole suite
  51 -> 31 minutes.
- **The low-rank mass hoists its Woodbury factors out of the trajectory.** `energy` and
  `velocity_into` each rebuilt the `O(q^2 d)` Sherman-Morrison recursion, several times per leaf,
  although it depends only on the mass and so is constant for a whole trajectory --- XLA
  eliminates the repeats *within* one loop iteration but will not hoist them *out* of the
  trajectory `while_loop`. They are now computed once per kernel call, in
  `HamiltonianContext.kinetic_cache`. On a blocked funnel with a rank-8 mass: warmup 1.66s ->
  1.02s at 200 dimensions, sampling 1.19s -> 0.86s at 400.
- **Fixed: `beta_min=0` drove the temperature ladder to NaN.** The adaptation is parametrized in
  temperatures, so a zero top rung is an infinite one and the first update produced NaN betas
  silently, on defaults. Such a ladder is now held fixed with a warning.
- Two further candidate optimizations were implemented, measured and rejected: compiling the
  Robbins-Monro step-size update was a 20% end-to-end regression on a 200-d model, and compiling
  the centering estimators made no measurable difference. Both are recorded in comments.

## v0.1.2

- **Sampling is ~2x faster and a run holds far less memory, with bit-identical output.** Four
  independent fixes, verified byte-for-byte against the previous release on six models plus
  parallel tempering (draws, gradients, every diagnostic and both step sizes unchanged):
  - The per-iteration stores kept `np.asarray` of each JAX array. On the CPU backend that is
    **zero-copy** --- a non-owning view over the live device buffer, which pins the whole
    `jax.Array` for the life of the store. Real copies instead: resident growth falls from
    29.6 KiB/iteration to 3.5 KiB/iteration, at no measurable time cost.
  - The per-step RNG draw sliced each component separately in Python, one eager dispatch each.
    One compiled gather instead: 0.44 ms -> 0.019 ms per step.
  - `initialize()`'s MALA line search probed an eagerly-bound `lax.while_loop` up to 60 times,
    recompiling on every probe. Now one cached `jax.jit` with the step size traced.
  - `PolyakLog.update` computed a tail average that all three of its callers discarded.

  End to end on a 6000-iteration warmup plus 2000 draws: Neal's funnel warmup 6.8s -> 4.8s,
  sampling 1.06s -> 0.45s, resident 309 -> 74 MiB; a 31-d blocked funnel sampling 2.07s -> 1.09s,
  resident 394 -> 159 MiB.

- **Warmup termination no longer spends most of its time compiling.** The classifier check refits
  on a history that grows every check, so each check handed XLA a new array shape. Rows are now
  buffered to a power of two with the padding at zero weight, and the fit runs under a cached
  `jax.jit` --- both are needed, since `minimize` is a bare `lax.while_loop` that recompiles on
  every call outside `jit` regardless of shape. On Neal's funnel through the factory default:
  compilation events 489 -> 98, XLA compilation 10.5s -> 2.5s, warmup wall 16.1s -> 5.5s, peak RSS
  981 -> 559 MiB, with warmup stopping at the same iteration.
- **`summary()` row labels now index from 1**, matching the DSL, so a model written in the DSL and
  the tables reporting on it no longer disagree: `x[1]` not `x[0]`, `S[2][1,1]` not `S[1][0,0]`, a
  scalar still bare. Labels only --- the arrays behind them are unchanged.
- **The warmup feature buffer is freed once warmup is over.** It is one `model.features` row per
  retained draw, grows for the whole of warmup, and is dead weight for the whole of sampling; the
  dynamic burn-in search's standardized copy of it goes too. Released when the criterion fires
  *or* `max_warmup` runs out — but not when a `warmup(n)` call merely returns, since the caller
  may continue with another one. `keep_features=True` holds on to it; the criterion's own
  reported history is unaffected either way.
- `fit_logistic` names the optimiser hyperparameters explicitly instead of forwarding `**opt`, so
  they can be static to the cached fit and a mistyped one raises instead of silently defaulting.

## v0.1.1

- **Parallel tempering: NUTS now selects independently at each temperature.** Each rung builds its
  own trajectory (its own direction, subtree and leaf draws) and picks its own next point; the
  rungs still stop together, on the *combined per-lane verdict* — any lane's U-turn or divergence
  ends the doubling for all — so the vectorized step is unchanged. Measured against the previous
  shared-trajectory scheme, cold-chain ESS per gradient evaluation improves 1.9-3.9x on a bimodal
  target, 4.6-7.4x on a funnel and 1.25x on a large-data HMM, with the margin growing in the number
  of temperatures. This is now what `parallel_tempering` does by default; it falls back to the old
  scheme when the integrator couples the temperatures (a WALNUTS line search), and
  `selection="joint"` restores it explicitly.
- Added `per_temperature_step_size` for a per-rung step size, which independent selection makes
  possible. Off by default: it is faster on uniform geometry but inflates the step until a funnel's
  neck cannot be integrated.
- **The factory's learned-metric regression now offers bare position-dependent forms.** Every
  candidate used to carry an additive constant floor (`SpExp(d) + Exp()`), so the simpler
  `SpExp(d)` could not be selected even when it was the right answer; both are now enumerated and
  AIC chooses between them. On a target whose metric genuinely has no floor (a Neal funnel) the
  bare form is now selected, as it should be. On `reg_horseshoe` and `irt_2pl` the selection is
  unchanged or quality-neutral, at the cost of ~1.6-1.9x longer `analyze` from the larger candidate
  pool — so this closes a structural blind spot rather than improving those models.

## v0.1.0

- Initial release.
