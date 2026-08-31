# Changelog

## Unreleased

- **Parallel tempering supports discrete parameters.** PT refused such a model; it no longer does.
  This is the pairing that matters most for the feature: a single-site Gibbs sweep moves one
  coordinate at a time, so two configurations separated by a low-density intermediate are
  unreachable from each other however long the chain runs --- the barrier is *structural*.
  Tempering flattens exactly that.

  The refusal's comment named two obstacles. Both were real, and there was a third it did not name
  which would have produced a **wrong answer** rather than a missing feature:

  * `pt/kinetics.py` rebuilt each lane's `HamiltonianContext` **positionally**, dropping `discrete`
    exactly as it already dropped `betas`. Both sites now forward every field by keyword, and
    `_lanes` **slices** `discrete` per lane rather than passing the whole product block.
  * `ProductModel` tiled one flat *float* vector and hardcoded `discrete_dim = 0`. It now carries
    `K` copies of the integer block, with `discrete_block` describing one rung.
  * **`_swap` permuted only `coordinate`.** A replica is its whole state: exchanging positions
    while leaving the labels behind hands each rung a configuration drawn from no target at all.
    `apply_swaps` was already generic over `(K, ...)`, so the fix is one call --- the danger was
    entirely in not noticing. The test for it was checked by reverting the fix (coordinates
    permuted to `[0,2,1,3]` while labels stayed `[0,1,2,3]`).
  * And, also unnamed: `TemperedProductPotential` vmapped over lanes with the **shared** context,
    so every rung would have evaluated its density at the same labels. All four entry points now
    route through one `_lane_potentials`; the continuous path is kept as literally the old
    expression, so a continuous tempered run emits the graph it always did.

  The sweep gained a **lane axis** rather than a tempered variant: `z` is `(L, n)`, the density is
  `(L,)`, and lanes accept independently --- each rung its own chain against its own target, as
  `IndependentAcceptanceMixin` already treats the continuous half. `L = 1` is an ordinary sampler.
  Proposal tables are per rung (a hot rung's marginal is flatter) and are deliberately **not**
  exchanged by a swap: a table describes a temperature, not a state.

  **Measured** on a new `spike_and_slab` benchmark --- two near-collinear predictors whose
  inclusion posterior has modes at 0.485 and 0.514 joined by a state at 2.9e-4, so a single-site
  sweep crosses about once in 2000 attempts. Over 8 seeds: plain Gibbs is trapped on **every**
  seed (four in each mode, 0 crossings, the frequency 100% wrong) while **split R-hat on the
  labels reports 1.0000 on all eight** --- a coordinate that never moves has no within-chain
  variance to betray it, so the mixing diagnostic is blind to a maximally wrong answer. PT crosses
  ~1300 times per run and is **unbiased**: `p(1,0) = 0.5117 +/- 0.0273` against an exact 0.5143, a
  deviation of **0.09 SE**.

  Two measurement mistakes on the way, both recorded in doc 14 because both were the kind that
  flatter or alarm without cause. The first benchmark put 0.23 of the mass on the joining state,
  making it a stepping stone rather than a barrier --- the comparison would have been vacuous.
  And per-seed errors of 0.08-0.20 were read as bias, and a four-seed mean treated as converged,
  before eight seeds showed the null; the per-seed sd is 0.077. The oracle was suspected too and
  cleared independently, by 2-d quadrature of the model's own density agreeing with the analytic
  marginalization to 4e-6.

  New: `mimcs.testing.spike_and_slab`, `ProductSpaceMixin.get_discrete_all`,
  `pt.lanes.per_temperature_potential` / `lane_discrete`, and `tests/test_pt_discrete.py`. The
  sampler **factory** still refuses discrete models --- that wiring is the remaining item.

- **Learned-marginal proposals for discrete parameters** (`DiscreteMarginalAdaptation`). The
  Metropolis-within-Gibbs sweep proposed uniformly over the values a coordinate is not currently
  at, which in a `k`-component mixture spends `(k-2)/(k-1)` of its attempts on labels of
  essentially zero density --- each a full log-density evaluation, each certain to be rejected.
  This learns each coordinate's marginal pmf during warmup and proposes from it instead.

  That proposal is **asymmetric**, so the sweep now carries a Hastings term,
  `g(cur) - g(prop)` with `g(v) = log p_v + log1p(-p_v)`. Two properties fall out of the algebra
  rather than being arranged: it is identically **zero for a binary coordinate** and identically
  zero for a **uniform** table, so binary parameters and unadapted runs are untouched *exactly*.
  Verified on the enumerated single-coordinate kernel for an arbitrary pmf: detailed balance holds
  to ~1e-18 with the term, and breaks by 1e-2 to 1.3e-1 without it.

  **Measured** on a Gaussian mixture (`n=120`, `sep=3`, 8 seeds, against the same sampler without
  the mixin): moves per iteration **1.00x at k=2, 1.93x at k=3, 5.94x at k=8**, label ESS
  1.00x / >=2.63x / 7.14x, at unchanged cost per iteration and 0 divergences throughout. At `k=2`
  the draws are **bit-identical across the two arms on all 8 seeds**.

  **The prediction going in was wrong, and instructively.** I expected ~1.0x at `k=3` and said a
  large gain there would indicate a broken measurement. Reasoning: a confidently-assigned point has
  a near-point-mass marginal, so excluding the current value leaves near-uniform weights over the
  rest. True, and irrelevant --- those coordinates barely move and contribute almost nothing to
  ESS. Mixing is dominated by the *ambiguous* coordinates, where the learned proposal concentrates
  on the ~1 plausible alternative while uniform spreads over all `k-1`. The gain is therefore about
  `k-1`, which all three measurements show. The `k=2` run is the null control the mistaken `k=3`
  prediction was meant to supply: a measurement reporting a gain where the algebra forbids one
  would be broken.

  Note when reading those numbers that `ess_1d` returns `min(n/tau, n)`, so a well-mixed label
  sits at the **ESS cap** --- at `k=3` the adapted arm is capped on 54% of coordinates even at
  12000 draws, making its ESS ratio a censored lower bound. The moves ratio is uncensored.

  The estimator is the library's shared Robbins--Monro gain, which suits a pmf unusually well:
  `p <- p + gain*(onehot - p)` is a convex combination of two points on the simplex, so it stays on
  the simplex with no renormalization and starts at uniform. It is then mixed with the uniform,
  `p = (1-lambda) p_hat + lambda/n` (`discrete_lambda`, default 0.05). That is not cosmetic: a zero
  entry makes a value unproposable, which does not break detailed balance but **does break
  irreducibility** --- the chain would target the posterior restricted to whatever warmup happened
  to visit, and no diagnostic here would flag it. `lambda = 0` is refused; `lambda = 1` is the
  uniform proposal exactly.

- **The discrete proposal's parameters live in a dict keyed by parameter**
  (`state.discrete_proposal_params`), not one padded `(discrete_dim, n_max)` array. A single array
  would bake in the assumption that every discrete parameter has a finite enumerable support ---
  exactly what a count-valued `int<lower=0>` breaks, since its proposal is a random-walk scale
  rather than a pmf over an enumeration. Keyed per parameter, the entry's shape *and meaning* are
  the parameter type's business, as `ham_params` already lets each kinetic decide what its entry
  means. It pays off at once: every coordinate of one `IntegerParameter` shares its support, so
  that parameter's table is exactly `(size_i, n_i)` with `n_i` a **Python int** at trace time, and
  the sweep becomes a static loop over parameters with a statically sized candidate axis --- no
  padding and no masking anywhere.

  `_discrete_sweep` was restructured for this and is **bit-identical** to the previous flat sweep,
  including a model with two discrete parameters of different widths and `discrete_sweeps=2`: with
  candidates ordered cyclically from `cur+1`, inverse-CDF selection at a uniform table reproduces
  the old `1 + floor(u*(n-1))` offset exactly (0 mismatches in 2.4e6 float32 cases, asserted in
  the suite so a later reworking cannot silently cost it).

- **Documentation caught up with the discrete-parameter work.** Several places were not merely
  silent but *stale*: `docs/index.md` omitted design doc 14 from its table of contents (and still
  said "four" examples); doc 01's `SamplerState` snippets were missing both new fields; doc 06's
  `HamiltonianContext` was missing `discrete`; and doc 04's atlas section still said
  Metropolis-within-Gibbs was machinery "the library has no ... for today", which it now has ---
  the atlas blocker has narrowed to the no-discrete-parent rule, since a chart index is by
  definition a chart's parent. Also added: `Model`'s second parameter list (doc 05), why a discrete
  feature gets no Stein z and why its padding is zeros rather than NaN (doc 11), discrete features
  in warmup termination and the stuck-label caveat (doc 10), the kernel-composing mixin category
  (doc 02), the factory's refusal (doc 09 and the factory reference), that `int` means something
  different in a `parameters` block than in `data` (doc 08), and the API module blurbs.

- **Discrete (integer) parameters, and a Metropolis-within-Gibbs sampler for them.** A parameter
  may now be integer-valued --- `int<lower=L, upper=U>`, declarable as an array --- which puts
  mixture models, latent classes, spike-and-slab selection and change points in reach for the
  first time. The labels are **sampled**, not marginalized out with `log_sum_exp` as a
  gradient-only library must.

  The state gains a second flat array, of dtype `int`, beside the existing float one; `Model`
  keeps discrete parameters in a list of their own, contributing to neither `coord_dim` nor
  `ambient_dim`. That separation is what kept the change small: the block partitioner, every mass
  adaptation, the chart machinery and the score pullback all iterate `model.parameters` and never
  see a discrete one. A discrete parameter has **no chart** --- sample space is coordinate space,
  and there is nothing to reparameterize or differentiate.

  `DiscreteMetropolisWithinGibbs` is a new kind of mixin: it composes with `kernel` rather than
  the `_*_hooks` chain, so `make_sampler_class(RobbinsMonroStepSize, DiscreteMetropolisWithinGibbs,
  NUTS)` is NUTS that also moves labels, with no base algorithm edited. It scans the discrete
  coordinates in order and proposes uniformly among the `n-1` values each is not currently at ---
  symmetric, so no Hastings term, and a binary coordinate always proposes the flip.
  `StaticContinuous` gives a discrete-only model a base to compose over.

  **Verified before it was written**, then kept as tests with controls: detailed balance of one
  coordinate update, built by enumeration, holds to ~1e-6 while a missing acceptance test (1.7e-1),
  an inverted ratio (1.7e-1) and an asymmetric proposal without a Hastings correction (2.2e-2) all
  fail it. End to end, the sweep recovers an exactly enumerable 8-state target to
  **max |empirical - exact| = 1.8e-3 over 40 000 draws** on a pmf spanning 0.005 to 0.546 --- a
  100x range, so a density-blind sweep could not pass --- and a perturbed density fails the same
  assertion. The DSL mixture recovers its generating means to within 0.02 and **97.3% of its
  generating labels**, at 0 divergences.

  The subtle part is the **gradient cache**. `BaseHMC` caches each potential's value and gradient
  at the current coordinate and the next trajectory's leading half-kick reads them back verbatim;
  after a sweep those are the *previous* labels' gradients. `_after_discrete` refreshes them.
  Without it nothing raises, the acceptance rate stays plausible, and the chain targets the wrong
  density --- so that test carries a negative control.

  **Cost, measured** against the same model with the labels baked in as data (so only the sweep
  differs): 1.09x at 10 labels, 1.17x at 30, 1.20x at 100, **2.47x at 300**, with compile time
  flat (1.06 -> 1.14 s) across that whole range — the check that the `fori_loop` traces its body
  once instead of unrolling. The prediction was "well under 2x at 30, sweep-*dominated* at a few
  hundred"; the first half held and the second did not, because the density is itself `O(n)` so
  growing the label count makes both arms costlier. Being wrong in the flattering direction is
  recorded rather than quietly dropped.

  A predicted hazard that did **not** materialize, recorded so nobody re-adds the guard: the
  concern that `floor(u*(n-1))` could reach `n-1` for `u` just below 1 and silently collapse the
  proposal to the current value. Exhaustively, for every `n` in 2..200000 and the largest
  representable `u < 1` in both float32 and float64, the product rounds down; JAX's float32
  uniform is generated on a `2^-24` grid and never gets that close. There is no clamp.

  Diagnostics: a discrete parameter's feature is its **bare value** (for a binary `z`, `z^2 == z`,
  so the usual second block would be a duplicate column), and it has **no Langevin--Stein term** ---
  the identity integrates by parts against a density and a score, and a pmf has neither. The
  summary prints a gap rather than a number. Note the padding is zeros and not NaN on purpose:
  `summarize` drops any draw with a non-finite Stein row, so a NaN column would have discarded
  *every* draw.

  New: `mimcs.model.IntegerParameter` / `BaseDiscreteParameter`,
  `mimcs.samplers.DiscreteMetropolisWithinGibbs` / `StaticContinuous`,
  `sampler.get_discrete_flat()`, `Model.discrete_parameters` / `discrete_dim`, `categorical` and
  `categorical_logit` in the DSL, the `gaussian_mixture` test problem and the `nuts_gibbs` builder,
  `examples/05_mixture.py`, and `docs/design/14_discrete_parameters.md`.

- **`int` in a `parameters` block changes meaning.** It was a registry alias for `real`, so
  `parameters { int<lower=0,upper=1> z; }` compiled to a continuous `BoundedParameter` sampled by
  NUTS on a logit link --- it parsed, it ran, and it was not what anyone writing it meant. It now
  builds an `IntegerParameter`, and both bounds are required and must be constant integers. No test
  or example declared one, so nothing in the repo changed behaviour. `int` in a `data` block or a
  function signature is **untouched**: neither reaches a parameter builder.

- **Deferred, with designs recorded** (`docs/design/14`): factory wiring (`analyze` and
  `parallel_tempering` *refuse* a discrete model rather than silently building one that never moves
  a label), custom jump operators that move continuous parameters alongside a label, exact
  conditional Gibbs, component-restricted recomputation (the fix for the `O(discrete_dim)` full
  density evaluations a sweep currently costs), random-scan and blocked updates, count-valued
  integers, discrete-aware learned metrics, PT x discrete, and discrete Stein diagnostics.

## v0.1.6

- **Parallel tempering hoists the low-rank Woodbury factors out of its trajectory loop too.**
  v0.1.3 gave `LowRankQuadraticKinetic` a `precompute` whose `O(J^2 d)` Sherman--Morrison recursion
  `BaseHMC.context` lifts to a per-kernel-call constant, but left tempering out: `ProductKinetic`
  defined no `precompute`, *and* rebuilt each lane's context from three positional fields, so the
  cache was neither filled nor forwarded across the `vmap`. It now delegates to the inner block and
  maps the cache along the same temperature axis as `ham_params`, so a lane reads the factors built
  from its own mass. A block with nothing to precompute returns `None` and contributes no entry.

  **Measured on a correlated Gaussian at K=4 with deep trees (63 leaves/iteration), sampling:
  rank 32, d=400, 20.6 -> 15.8 s (1.30x); rank 8, d=200, 2.14 -> 1.88 s (1.14x); rank 4, d=400,
  6.29 -> 6.19 s (1.02x).** The win is real but rank-dependent, and smaller than the untempered
  1.38x: the removed term is `O(J^2 d)` against `O(J d)` for everything that survives, so at the
  ranks the factory typically selects it is a small share of the tempered loop body. An initial
  reading that the gap did not widen with rank came from sweeping only 4 -> 16, too narrow to see
  it. Diagonal and dense tempered runs are untouched and **bit-identical**; so, measured, are
  untempered and tempered low-rank runs.

- **`BaseHMC.context` gains `kinetic_cache=False`, and the reseeding callers use it.** Building the
  cache is free inside the jitted `kernel` but **eager** in `state_at_coordinate` and in the
  tempered ladder reseed, which dispatches the recursion primitive by primitive: 43.8 ms per call
  against 0.10 ms traced, at rank 8, d=200, K=4. Since the ladder reseeds once per warmup
  iteration, hoisting the factors without this opt-out made tempered warmup **3.6x slower**
  (5.4 -> 19.3 s) — a regression three times larger than the speedup it was buying. Those callers
  read only the potentials, so they now skip it. The switch is performance-only: every consumer
  keeps its inline fallback, so no number depends on it.

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
