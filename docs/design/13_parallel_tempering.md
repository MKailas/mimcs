# Parallel Tempering

## Motivation

Everything the library samples with today is NUTS and refinements of it — learned metrics (doc 07), WALNUTS, multi-rate integration (doc 06). All of them improve how thoroughly a chain explores *one* mode. None of them helps it find another, because they all rely on following the gradient of a density whose modes are separated by regions the gradient points away from.

The standing example is `tests/problems/hmm_gaussian`, a 3-state Gaussian HMM. With weakly informative priors added, four factory-default seeds give:

| seed | divergences | result |
|---|---|---|
| 0, 1, 3 | 0 | μ ≈ [9.16, 18.65, 29.53], ESS 600–920, R̂ ≤ 1.003 |
| 2 | 0 | a different mode, μ ≈ [3.94, 12.9, 28.0], **ESS 2.8, R̂ 3.04**, max-depth trajectories |

Seed 2 is not badly conditioned — it is in the wrong place, and it stays there. No mass matrix, learned or otherwise, changes that. Mixture and hidden-state models are *generically* multimodal (label switching is only the most obvious source), so this is not an exotic failure.

**Parallel tempering** is the standard general answer. A ladder of inverse temperatures `1 = β_1 > β_2 > … > β_K ≥ 0` targets `π_k ∝ π^{β_k}`; the hot chains flatten the barriers between modes and traverse them freely, and periodic swaps between adjacent temperatures carry that mobility down to the cold chain, whose draws are the ones kept. It is imperfect — the ladder needs tuning, and the cost is K times the work per iteration — but it addresses multimodality without knowing in advance where the modes are, which nothing else here does.

Two requirements shape the design:

1. **Generic over base samplers.** PT should be a way of *running* RWMH, HMC or NUTS, not a fourth sampler. The same argument that produced the mixin architecture (doc 02) applies.
2. **Parallel across temperatures.** The K log-density and gradient evaluations per step are the whole cost, and they are independent — exactly what `vmap` is for, and what makes PT worth running on a GPU.

## Why not K separate samplers

The natural design is K sampler objects, one per temperature, stepped in turn with swaps between. It fails on requirement 2, and specifically on NUTS.

A NUTS transition takes a number of gradient evaluations that is **decided at run time** by its own U-turn criterion, and differs between temperatures — a hot chain's flatter target typically turns later than a cold one's. K independent trajectory builders therefore cannot share a single vectorised model evaluation: there is no common trip count to `vmap` over. One could `vmap` the whole kernel and let JAX's `while_loop` batching rule run every lane until the last one finishes, but that wastes exactly the work the vectorisation was meant to save, and it wastes more of it the more heterogeneous the ladder is — which is to say, always.

So the design relaxes the independence: **one sampler on the K-fold product space**, whose per-leaf work is vectorised over temperatures.

The product target is
```
Π(x_1, …, x_K) = Π_k π_k(x_k) = Π_k π(x_k)^{β_k}
```
a product measure, so a chain that leaves Π invariant leaves each π_k invariant marginally. Its gradient is the K independent gradients, which is precisely one `vmap` over a leading temperature axis.

## What "almost independent" means, per base sampler

The product-space chain should behave as K nearly-separate chains, coupled only where coupling is unavoidable. What is unavoidable differs by sampler.

### RWMH and HMC: exactly independent

Both propose and accept in one shot. Nothing forces the temperatures to share an accept/reject decision, so they do not: the acceptance test is a `(K,)` comparison and the state update a per-temperature `where`. Each temperature is then **exactly its own valid chain**, sharing only the vectorised gradient evaluation. This is emphatically *not* a Metropolis or HMC step in the product space, which would accept or reject all K together and mix far worse — one bad temperature would veto every proposal.

### NUTS: one trajectory, one selection

NUTS cannot be made independent *this* way, because a trajectory has no fixed length: the temperatures must agree on when to stop doubling. The default therefore builds the trajectory once in the product space, with the product-space U-turn criterion. (A per-lane construction that keeps the temperatures stopping together **is** possible and is now implemented — see "Resolved" below — but it is a different construction, not independent acceptance applied to a shared trajectory.)

This falls out of the existing code with no change. `BaseNUTS._is_turning` computes `dot(momentum_sum, velocity)` over the whole coordinate vector; with the coordinate being the stacked `(K·n,)` product vector, that dot product *is* `Σ_k dot(p_sum_k, v_k)` — the product-space criterion, by construction.

The remaining question is what to select from the trajectory, and it is a correctness question rather than a taste one.

> **Independent selection is not obviously valid.** Letting each temperature choose its own leaf index makes the new product point a *mix-and-match* — coordinate `k` at time `t_k` along its own orbit, with the `t_k` generally different. The selection probability does factorise in the right way (`Π_k w_k(t_k)` is proportional to the product target at that mixed point), so the *weights* are not the problem. The problem is reversibility: NUTS's detailed-balance argument requires that rebuilding the trajectory from the new state recovers the same orbit, and the stopping rule here depends on the whole product state. From a mixed point, each coordinate sits at a different phase of its own orbit, so the product-space U-turn evaluated there is not the criterion that produced the trajectory. The argument does not transfer, and the failure mode — a slightly biased β=1 marginal — is invisible to any test that does not compare against a known answer.

So v1 uses **joint selection**: one leaf index shared by all temperatures, weighted by the total energy `Σ_k H_k`. This is ordinary multinomial NUTS on the product target, and is correct for the same reason plain NUTS is. Selection, divergence and energy are all sums over temperatures — scalars — so `mimcs/hmc/nuts.py` needs **no changes at all**; PT-NUTS is the existing `NUTS` class pointed at product-space potentials and kinetics.

The price is real: acceptance is coupled across temperatures, so the cold chain's proposal is chosen partly by energy variation it does not care about.

#### Resolved: `selection="independent"` (`mimcs/pt/nuts.py`) — **now the default**

The objection above is about a **shared trajectory**. It dissolves if the tree construction is made per-lane as well, and the resolution needs two changes that are only valid together:

1. **Decoupled directions.** Each lane draws its own `tree_direction`, `tree_select` and `leaf_select`, so lane `k`'s trajectory is an ordinary NUTS orbit for `pi^beta_k`.
2. **Stopping combines per-lane *verdicts*, never per-lane *quantities*.** Stop when **any** lane's own U-turn fires (`min_k` of the test quantities `<= 0`) or **any** lane diverges (`max_k` of the per-lane energy ranges). Every lane then stops at the same doubling and builds the same number of leaves, so the vmapped product step stays efficient — no lane freezing, no masked accumulators, no wasted work.

*Why it is reversible.* Each lane's doubling checks **every canonical sub-block** of its own `T_k` (`read_level`'s `1..ntz(n+1)` bound). So a collection `(T_1..T_K)` is reachable iff no lane's criterion fires on any *proper* canonical sub-block and some lane fires on the full tree — both functions of the collection alone, with no offset in them. Any `z'` on `T` therefore meets only non-firing proper sub-blocks below level `J` and the identical check at `J`; both `z` and `z'` stop at `J` with direction probability `2^-JK`, and with per-lane weights `Π_k pi_k(z'_k)` the detailed-balance ratio is symmetric.

*The corner that does not work, and its measured cost.* Combining lanes **inside** a test quantity — the summed `Σ_k rho_k·v_k` — does not survive decoupled directions: that sum pairs the specific blocks the offsets select, and independent selection is what makes the offsets differ. K=2, depth 2, four points per lane: from offsets `(1,1)` the level-1 check is `rho_1{0,1}·v + rho_2{0,1}·v`; from `(1,2)` it is `rho_1{0,1}·v + rho_2{2,3}·v`. On a 1-d Gaussian at K=2 (200k draws × 8 seeds) that form biases the cold variance to **0.853** against 1.0 — a 90-sigma miss — while the min/max rule lands at **0.9985 ± 0.0023** and a per-lane-stopping reference at 1.0019 ± 0.0017. Two warnings from that study worth keeping: the cold **mean** is clean in all three arms (`z_mean` +0.14 for the *broken* one), and the broken arm has the **best** ESS-per-leaf — so neither means nor efficiency can be the screen. See `tests/experiments/writeups/pt_independent_bias.md`.

*The default.* `selection="auto"` picks independent for a NUTS base, **falling back to joint whenever the integrator couples the temperatures** — which a line search does, and which is why the fallback exists rather than an error. An *explicit* `selection="independent"` with such an integrator raises instead of silently downgrading.

*Consequences in the code.* Anything that lets lane `k`'s step depend on lane `j`'s state destroys the argument, so the line-search integrators — whose refinement level comes from the summed Hamiltonian — are **refused** on this path (doc's own open item, "a per-rung energy-error criterion", must be closed first). The divergence threshold **drops** its `× K` scaling, since `max_k (h_max_k − h_min_k)` is one Hamiltonian's range rather than a sum of K. And `accept_prob` stays a **scalar on the summed energy**: the step size is global by design, a `(K,)` signal would broadcast `RobbinsMonroStepSize` into per-rung steps, and the per-lane statistic sits on a different scale (0.914 vs 0.781 at the same step, K=4) which drove a step-size runaway to 1.28 where every trajectory U-turned at the first doubling.

*Measured against joint selection* (8 seeds/cell, factory-built so only the rule varies; writeup `tests/experiments/writeups/pt_selection_ab.md`). Cold-chain min-ESS per 1000 gradient evaluations:

| target | K=2 | K=4 | K=8 |
|---|---|---|---|
| bimodal | **1.91×** | **3.36×** | **3.85×** |
| neal_funnel_blocks | **4.60×** | **7.41×** | **7.40×** |
| hmm_gaussian | — | **1.25×** | — |
| correlated_gaussian | 1.12×\* | 1.69×\* | 1.29×\* |

\* that target cannot discriminate: joint's ESS is at the draw-count ceiling in every seed (2000/2000, and 20000/20000 when rerun with ten times the draws), so its ESS/gradient is a lower bound and the ratio an upper bound.

**The mechanism is one column: joint's tree depth grows with K, independent's does not.** 4.46 → 6.52 → **8.20** on the funnel against 2.38 → 2.40 → 2.62. The summed U-turn fires when the *aggregate* turns, so each extra rung is another chance to mask the lane that has already turned; where the hot rungs go flat and never turn, joint runs to 590 leaves per iteration while each lane would turn at ~2.6. That is the same fact the passive Step 0 study found as `j_sum > j_cold` — joint runs the cold chain past its own U-turn, and the over-run is close to pure waste.

End-to-end with the classifier termination restored and warmup gradients counted, the win survives at 1.45×–6.87×. Independent sometimes needs a longer warmup (funnel K=8: 1950 vs 700 iterations, giving back about half that cell's gain) and sometimes less (bimodal K=8: 950 vs 1900).

Bias guards: both arms match the analytic Gaussian moments to ≤0.01 over 8 × 20 000 draws; **8/8 seeds cross both modes in both arms** on the bimodal target at every K, so no arm is winning by sitting in one mode; nothing frozen. Divergence counts are **not** comparable across the arms — different thresholds and different statistics.

#### Per-temperature step size (`per_temperature_step_size=True`)

Independent selection is what makes this possible: each rung now has its own acceptance signal, which joint selection does not. It is **opt-in and should stay so** — see the failure below.

**Why it helps.** A single global step is tuned so the *summed* K-fold energy error meets a target calibrated for one chain, which implicitly demands each rung be `target^(1/K)` accurate — 0.95 at K=4, 0.97 at K=8. Per-rung tuning asks each rung for 0.8, the right question now that each rung *is* its own chain. Measured on `correlated_gaussian` at K=4: global 0.434 against per-rung 0.56–0.67. The gain therefore grows with K, and does: on the bimodal target **1.07× / 1.51× / 2.10×** at K = 2/4/8 on top of independent selection, and **1.89×** on `hmm_gaussian` at K=4 — all with zero divergences.

Note the adapted steps are barely spread (`[1.26, 1.25, 1.22, 1.27, 1.12, 1.03, 0.95, 0.98]` at K=8): **the gain is the overall level, not the per-rung differentiation**. The per-rung *mass* already absorbs each rung's width. One global step targeting `0.8^(1/K)` on the summed statistic ought to capture most of it, and is untested.

**Where it fails.** On `neal_funnel_blocks` the apparent 8.36× at K=8 is a failure to integrate the neck: **1082 of 2000 transitions diverge** (against 91), trees fall to 1.06 doublings, and the `v` marginal is under-dispersed at **std 2.73 against a true 3.0** (~3.5 s.e. low; joint gives 2.96, plain independent 2.89). Same pattern at K=4 (divergences 71 → 316).

The mechanism is the one that also nearly sank the *global* step: the acceptance statistic is a mean over the leaves actually taken, and the min-rule trees here are one or two leaves. Averaged over so few early leaves it cannot see the neck, so the step inflates until the neck is unintegrable — and the divergences that result are precisely what the statistic ignores. **ESS per gradient alone would have scored that an 85× win over joint**; only the divergence count and the dispersion of `v` catch it. Making the signal robust to a truncated tree (an energy-range or WALNUTS-style proxy rather than a mean over leaves taken) is what would let this be a default.

## Tempering: what β multiplies

The model exposes named log-density components (doc 05), so β can be applied per component:

```
log Π_k(x) = Σ_{c ∈ tempered} β_k · log π_c(x)  +  Σ_{c ∉ tempered} log π_c(x)
```

By default every component is tempered — textbook PT, where `β = 0` is a flat target. Naming a subset gives a **power posterior** (`tempered=["likelihood"]`: prior at full strength, likelihood flattened), which is usually better behaved: a flat target is improper whenever a parameter is unbounded, and `hmm_gaussian` has already shown what an improper direction does to a sampler. The choice is the caller's; the default matches the textbook.

**The chart Jacobian is never tempered.** `JacobianPotential` (doc 04) is not part of the target — it is the change of variables that makes the coordinate-space density the right thing to sample. Scaling it by β would sample a different distribution, not a flatter one.

## Architecture

```
ParallelTempering  (a BaseSampler)
├── product sampler         one ProductNUTS / ProductHMC / ProductRWMH over R^(K·n)
│   ├── potentials          one TemperedProductPotential per model component (vmapped over K)
│   └── kinetics            ProductKinetic: the model's own block structure (diagonal /
│                           dense / low-rank / learned metric), vmapped over K, each
│                           temperature carrying its own adapted parameters
├── swap step               adjacent-pair exchange, even/odd sweeps
└── K adaptation samplers   one per temperature, stepped for their mixins only
```

### The product-space sampler

`ProductSpaceMixin` teaches an existing sampler the `(K, n)` layout: `make_initial_state`, the coordinate↔sample maps, and `_current_score`. `state.sample` stays the **β=1 chain's** ambient vector, so `BaseSampler.postprocess`, `summary()` (doc 11) and the whole evaluation path work unchanged; the other temperatures ride in a separate field.

`TemperedProductPotential` wraps one model component: reshape `(K·n,)` to `(K, n)`, `vmap` the component's value-and-gradient over the temperature axis, scale by `β_k` if tempered, flatten back. One wrapper **per component**, so `split_potentials` and the multi-rate integrator (doc 06) keep working over a tempered product model.

### The kinetic energy: one structure, applied per temperature

Tempering must not cost the sampler its kinetic machinery. Every temperature therefore uses the **same block structure** — whatever partition the factory chose for the model (doc 09), with whatever per-block kinetics it chose: diagonal, dense, low-rank, or a learned position-dependent metric — and applies it **independently at each temperature**, with each temperature carrying its own adapted parameters.

Concretely, the kinetics are defined **once**, with slices relative to a *single* temperature's coordinate (`0 … n`), exactly as they are for a non-tempered sampler. A `ProductKinetic` wrapper reshapes the `(K·n,)` product vectors to `(K, n)` and `vmap`s the existing aggregation — `energy`, `velocity_into`, `sample_into` — over the temperature axis. It does not map a whole `HamiltonianContext`: it closes over the shared fields and rebuilds a per-lane context from the mapped ones, which today are `ham_params` and `kinetic_cache`, both carrying a leading `K` axis.

No kinetic class changes. The block classes are already written against `(istate, ctx)` with their own slices and their own `ham_params[id]`, which is precisely what `vmap` needs.

**The one thing that is not free: a context field the wrapper forgets to forward is silently lost.** `kinetic_cache` (doc 06) is filled per kernel call for any kinetic defining `precompute`, and a wrapper that neither defines one nor forwards the field costs nothing in correctness and everything in speed — which is exactly what happened between v0.1.3 and v0.1.6, where tempered low-rank runs rebuilt the Woodbury factors on every leaf while untempered ones did not. `ProductKinetic.precompute` now `vmap`s the inner block's over the same axis `_lanes` uses, so the cache row and the lane that reads it are sliced by one rule; a block with nothing to precompute returns `None` and contributes no entry, leaving every non-low-rank tempered graph unchanged. This is the general hazard the `mass_mode`/`slices` note below describes, in its performance form.

The same hazard has a *correctness* form waiting, which is why `parallel_tempering` refuses a model with discrete parameters (doc 14). `HamiltonianContext` now also carries `discrete`, the flat integer block a `ModelPotential` needs; `_lanes` and `ProductKinetic.precompute` build their per-lane contexts **positionally**, so a `discrete` field would be dropped there exactly as `betas` already is — and dropping *that* one does not make a tempered run slow, it makes it evaluate the density at the wrong labels. Wiring PT to discrete parameters means forwarding `discrete` at every one of those construction sites first.

The alternative reading — one `DiagonalQuadraticKinetic` per temperature, spanning that temperature's whole coordinate block — is *not* the design. It would restrict every temperature to a single diagonal mass, discard the block partition and the learned metrics with it, and eventually force a parallel reimplementation of the kinetic machinery. A second alternative, `K × B` ordinary kinetic objects with offset slices, is correct and needs no new code at all, but it unrolls the graph K times instead of vectorising it, which is the opposite of the point.

This was checked rather than assumed, on a deliberately heterogeneous structure — a 2-d **diagonal** block and a 3-d **dense** block, four temperatures, each with its own parameters:

```
vmapped total energy   12.352544
unrolled total energy  12.352543      (K×B separate kinetics, offset slices)
velocities agree       True
```

so the vmapped form and the explicit per-temperature form compute the same thing, with mixed block kinds.

Two consequences follow. `ham_params[block_id]` gains a leading `K` axis, which is what makes **per-temperature mass adaptation** natural: temperature `k`'s adaptation writes row `k`, and PT stacks the K adaptation samplers' parameters into that array. And the momentum draws declared by each block become `(K, block_size)` — the PT sampler asks the inner kinetics for their per-temperature shapes and prepends `K`.

### Per-temperature step sizes need no integrator change

A hot chain wants a larger step than a cold one, so ε must vary along the ladder. It already can: the kick is `p - eps * g` and the drift `q + eps * v`, both elementwise, and NUTS's direction flip `where(forward, ±step_size)` is elementwise too. A `step_size` of shape `(K·n,)`, each temperature's block filled with its own ε, therefore gives per-temperature steps through the existing integrator.

**Whether a per-temperature step size is *useful* turns out to depend on the sampler**, and this
was settled during implementation rather than in advance:

* **NUTS (joint selection): one global step size.** The product chain accepts as one, so there is
  a single acceptance signal and nothing to drive K different steps. That is not a loss, because
  the per-temperature *width* is absorbed by the per-temperature **mass**: a hot chain is wider,
  and width is exactly what a mass matrix models. Measured on a 2-d Gaussian with
  `beta = [1, 0.37, 0.14, 0.05]`, the learned inverse-masses came out in the ratio
  `1 : 2.7 : 9.7 : 20.7` — tracking `1/beta` as they should — and divergences went from
  **454/20000 to 0** once the mass was adapted per temperature rather than shared.
* **HMC and RWMH (independent acceptance): one step size per temperature.** Here each rung *does*
  have its own acceptance signal, so an ordinary `RobbinsMonroStepSize` adapts a `(K,)` vector,
  which the mixin expands to `(K·n,)` for the integrator.

So the vector step size the previous paragraph makes possible is used on the independent-acceptance
path and deliberately not on the NUTS path.

### The step-size and U-turn claims, checked

The kinetic claim above carries its own numbers; these are the other two — that the existing `NUTS` runs unmodified over a product coordinate with a *vector* step size, and that its U-turn is then the product-space one. Both were verified against the current code rather than argued.

A stand-in product target of two independent 2-d Gaussians with deliberately mismatched scales (sd 1 and 4, as a ladder would produce), sampled by the *unmodified* `NUTS` with two block-diagonal kinetics and `step_size = [0.4, 0.4, 1.2, 1.2]`:

```
accept 1.000   divergences 0   mean tree depth 3.24
target sd  [1.    1.    4.    4.  ]
sample sd  [1.025 0.993 4.063 4.022]
```

and, on random `(p_sum, v)`, `dot(p_sum, v)` over the stacked vector equals `dot(p_sum_1, v_1) + dot(p_sum_2, v_2)` exactly. So a per-temperature step size needs no integrator change, and the product-space U-turn needs no NUTS change.

### Swaps

Adjacent pairs only, in alternating even/odd sweeps, with all disjoint pairs of a sweep attempted at once (they touch disjoint states, so the sweep vectorises). For the pair `(k, k+1)`:

```
log α = (β_k − β_{k+1}) · (L(x_{k+1}) − L(x_k))
```

where `L` is the sum of the **untempered** tempered-components — the swap ratio involves the raw log-density, not the β-scaled one the potentials cache. v1 recomputes `L` per temperature at swap time (one vmapped, value-only evaluation), rather than plumbing untempered values through the gradient cache; against a NUTS trajectory of hundreds of gradients this is noise. Likewise, after a swap the cached potential values and gradients belong to the wrong β, so v1 re-seeds them with one vmapped `init_integrator_state`. Both are correctness-first choices with obvious optimisations available later.

Per-pair swap acceptance rate is a diagnostic, not a nicety: it is the quantity any ladder tuning is built on, and a ladder whose adjacent rates are near zero is a PT run that is doing nothing.

### The K adaptation-only samplers

Adaptation is where the "K separate samplers" idea earns its place. Rather than reimplement step-size and mass adaptation over a product state, `ParallelTempering` holds K ordinary sampler objects and uses them **only for their mixins**: after each product step it writes each temperature's slice into that sampler's state, runs its `_postprocess_hooks`, and reads the adapted `step_size` and `ham_params` back into the product state.

Every existing mixin — `RobbinsMonroStepSize`, `ScoreMassAdaptation`, `MassMatrixAdaptation`, the centering adaptations — then works per temperature, unmodified. This is the one place where the natural-but-unworkable design is exactly right, because adaptation is bookkeeping on a state, with none of the run-time-variable trip count that made K trajectory builders unworkable.

### Adapting the ladder

The measurements below (`hmm_gaussian`) turned "the ladder needs tuning" from a refinement into a
requirement: the usable temperature range narrows as the data grows, so a hand-set `beta_min`
cannot be right across models. `mimcs/pt/ladder.py` adapts it, following Miasojedow, Moulines &
Vihola (2013).

The parametrization is what makes the update safe. Adapt in *temperature*, and in the **log-gaps**
between neighbours:

```
T_1 = 1,   T_{k+1} = T_k + exp(ρ_k),   β_k = 1 / T_k
```

Every `ρ ∈ R^(K−1)` maps to a valid ladder — strictly increasing temperatures, `β_1 = 1` fixed by
construction — so the update needs no constraint and can never produce a crossed or duplicated
rung, which an unconstrained update of the βs themselves certainly could. The price is that `β = 0`
is not reachable (it needs an infinite temperature); a ladder wanting a genuinely flat top rung
should be fixed instead.

**The top rung is held fixed by default, and that correction is the main thing this section
records.** With the gaps left free — the literal MMV form — `ρ` runs away on any target whose hot
end goes *flat*. The mechanism is structural rather than a tuning accident: once two adjacent
tempered densities are both nearly flat their log-densities barely differ, so widening the gap
stops lowering that pair's swap acceptance. `α − α*` stays positive however far apart the rungs
are pushed, and nothing arrests the growth. Measured on the bimodal test target, `beta_min` ran
0.02 → 1.7e-4 in 200 warmup iterations and was still moving; at that point the hot chain's target
is flat enough that NUTS never U-turns, so every iteration builds a maximum-depth tree. The run
does not diverge or error — it simply stops making progress, which is how it was found (a test
that had been passing hung for over an hour).

So the gaps are instead **shares of a fixed total temperature range** (a softmax over `ρ`), which
pins both endpoints and adapts only the interior spacing. That also matches how the two quantities
differ in kind: `beta_min` states how deep a barrier the model has, a modelling judgement the user
can make, while the spacing is the tuning problem they cannot. `adapt_beta_min=True` restores the
unbounded form, with the above as its warning label.

#### When to free `beta_min`

The `hmm_gaussian` results below show the free form working well — the best arm in that study —
which sits oddly beside a runaway that motivated pinning the endpoint. Both are real, and the
condition that separates them is **whether anything holds the hot end down**.

Tempering scales the components it is given. Temper *everything* and `β → 0` is a genuinely flat
target: improper if the parameter space is unbounded, and flat enough that the swap acceptance
stops responding to the gap width and NUTS stops U-turning. That is the runaway, and it is
structural — the adaptation is not misbehaving, it is being asked to optimize a criterion that has
gone constant. Temper only the likelihood (`tempered=["likelihood"]`), and the prior stays at full
strength at every rung: the hot rung is the prior, which is proper, has finite width, and keeps
giving the acceptance signal something to respond to however far `β` falls. `hmm_gaussian` is that
second case, which is why its free ladder converged — consistently, to `beta_min ≈ 0.09` across
eight seeds — instead of running away.

So the guidance is:

* **Pinned is the right default**, because it is safe in both cases. It costs something only when
  the supplied range is wrong, and a wrong range is at least visible in the swap rates.
* **Free it when an untempered component holds the hot end down** — a power posterior with an
  untempered prior — *and* the achieved swap rates say the given range is wrong. At K=4 on
  `hmm_gaussian` the pinned ladder sat at ~0.02 per pair and, worse, could not repair itself: with
  both endpoints pinned the gaps are shares that must partition a fixed range, so a uniform
  `α − α*` cancels and the adaptation is a **no-op by construction**. Freeing the range is the only
  thing that can help there.
* **Do not free it when the range is already right.** At K=8 the geometric default already accepted
  0.21–0.36; freeing it let a slightly-above-target acceptance widen the range further, for 6× the
  divergences and no gain. Add rungs before reaching for the hot end.

One caveat on the evidence for the runaway itself: it was measured before the cache bug recorded
below was found. It is very unlikely to be an artefact of it — that bug's damage is `Δβ·V`, and the
bimodal test target has `V ~ O(1)`, so the corruption there is ~1e-3 nats against `hmm_gaussian`'s
~300 — and the mechanism above is structural rather than numerical. But it has not been re-measured
against the fixed code.

The update is Robbins–Monro on each gap, toward the optimal swap acceptance `α* = 0.234` from the
diffusion limit of Atchadé, Roberts & Rosenthal (2011) — the PT analogue of the familiar
random-walk Metropolis result:

```
ρ_k  ←  ρ_k + gain_n · (α_k − α*)
```

The sign is right: accepting too often pushes the rungs apart, too rarely pulls them together.

Two choices worth recording. The signal is the swap **acceptance probability**, not the binary
accept/reject — it comes free with the ratio and has far lower variance. And **every** gap is
updated every iteration via `all_pair_log_ratios`, using the ratio each pair *would* have accepted
with, rather than only the half this sweep's even/odd parity attempted; the unattempted ratios cost
nothing and using them keeps all gaps adapting at the same rate.

Adaptation is warmup-only and the gain decays, so diminishing adaptation holds. The ladder rides in
`state.ham_params` under a reserved key and reaches the potentials through a traced
`HamiltonianContext.betas` — a Python attribute would retrace the kernel on every update.

`adapt_ladder=True` is the default, which makes an explicit `betas=` a starting point for the
*interior* rungs; its endpoints are honoured either way, and `adapt_ladder=False` pins the lot.

### WALNUTS over the product space

Within-orbit adaptivity needs **no product variant**, for the same reason NUTS needed none.
`LineSearchIntegrator` refines against `total_energy(istate, potentials, kinetics, ctx)`, and over
the product space that is already the sum over temperatures — every tempered potential and every
`ProductKinetic` returns its own sum across rungs. The line search therefore sees one scalar
Hamiltonian, exactly as for an ordinary target, and refines the whole product step until that
scalar is within budget. `mimcs/pt/integrators.py` is a builder, not a subclass: the class it
constructs is the stock `LineSearchIntegrator`.

Two things do have to change, and both were measured on Neal's funnel (8 seeds, K=4,
`beta_min=0.05`, doubling schedule with 8 levels).

**The energy-error budget scales with K.** A threshold `δ` calibrated for one chain is a
statement about one Hamiltonian's discretization error; the product Hamiltonian is a sum of K of
them, and K independent errors of typical size `δ` sum to order `K·δ`. Holding the sum to `δ`
silently demands each rung be K times more accurate than intended and pays for it in gradients.
So thresholds are stated **per temperature** and scaled by K. That scaling — and the
untempered default it falls back to — is written down once, as `product_error_thresholds`
in `mimcs/pt/integrators.py`, and both users of it call that: `product_line_search` and the
factory's `pt_` bases (`_tempered_integrator_builder`, which asks the same module which
integrator names carry a budget at all, via `BUDGETED_INTEGRATORS`). The budget is parallel
tempering's own reading of the line search, so it belongs to `mimcs/pt/`, not to the factory.

**The divergence threshold scales with K too** — this doc previously listed that as an open
question, and it is now measured and fixed. `max(H) − min(H) > threshold` is over the summed
energy, so the K=1 default flags ordinary product-space energy ranges as divergences:

```
divergence_threshold   median div (of 2000)   frozen seeds   median v.min
1000  (K=1 default)            488                2/8            -9.19
4000  (= 1000·K)               304                2/8            -9.19
1e6   (effectively off)        140                1/8            -9.19
```

The neck depth is identical throughout, so the extra divergences at 1000 were **false
positives**, not real integration failures. `parallel_tempering` therefore defaults
`divergence_threshold` to `DEFAULT_DIVERGENCE_THRESHOLD * K` (an explicit value still wins).

**Tempered funnels need x64.** The seeds that stayed frozen above are a separate, numerical
failure: at `beta = 0.05` the funnel's v-marginal has sd `scale/√beta = 13.4`, so x's conditional
scale `e^(v/2)` reaches ~5e8 and float32 overflows. Under x64 with the K-scaled threshold, 0 of 8
seeds froze and the median divergence count fell to 152.

**What tempering buys on the funnel.** Both arms under x64, 8 seeds:

| arm | median v.min | median divergences | gradient evaluations |
|---|---|---|---|
| PT-WALNUTS (K=4) | **−8.82** | 152 | ~2.0e6 |
| WALNUTS (K=1) | −3.32 | 152 | ~1.3e6 |

Identical divergence counts — PT-WALNUTS is exactly as healthy — while reaching 2.7× deeper into
the neck for ~1.6× the gradients. This was *not* the prediction (the funnel is unimodal, so PT was
expected to cost something and gain nothing), and the mechanism is worth stating: a hot rung's
v-marginal is wider by `1/√beta`, so it visits neck depths the cold chain reaches only rarely, and
swaps carry them down. The funnel is not multimodal but it *is* a barrier in v, which is precisely
the coordinate tempering stretches.

Pair the line search with `LineSearchStepSizeAdaptation`, never plain `RobbinsMonroStepSize`: the
integrator refines until the energy error is in budget, so real acceptance is ~1 regardless of the
macro step and ordinary adaptation runs the step size away (measured here as ε = 12.2 with
acceptance 0.13, against 4.95 and correct moments once paired properly).

### The factory seam

Tempering is a property of the **base sampler**, not a separate machine, so `SamplerSpec.base`
gains a tempered counterpart for each algorithm — `pt_nuts`, `pt_hmc`, `pt_randomized_hmc` — and
`tempering_params` carries the ladder options (validated like `integrator_params`, so a typo
cannot pass silently). No rule selects one: it is an explicit choice.

`build_sampler` **delegates** to `parallel_tempering` rather than reassembling the mixin stack,
because the product model, tempered potentials, product kinetics, ladder adaptation and swap move
are all its job and two assembly paths would drift. What the factory contributes is the *split* of
the mixin list it has already computed:

* **per temperature** — `ScoreMassAdaptation`, `MassMatrixAdaptation`, `LowRankAdaptation`,
  `MetricAdaptation`, `ShapedMetricAdaptation`: each fits one rung's own width, and a hot rung is
  wider. The two mass adaptations are derived from `_MASS` rather than listed by name, because
  omitting one here does not fail loudly: a `ProductKinetic` copies `mass_mode` and `slices` from
  the block it wraps, so a mass adaptation left off this list still passes its own filter on the
  product chain and gathers rung 0's coordinates — which a diagonal block will broadcast across
  all `K` rungs without error.
* **global** — the step size, warmup termination, and the initialization mixins.

That split is the entire reason PT keeps K adaptation hosts. Two supporting changes were needed to
make the default spec work through it:

* `AdaptState` gained `potential_grads`, sliced per temperature. Every score-based adaptation reads
  it, including the factory default `ScoreMassAdaptation`; without it the hosts only supported
  `MassMatrixAdaptation`. Row `k` of a tempered potential's gradient is `beta_k · grad(V_k)` — the
  score of that rung's own target, which is what its mass should be fitted to.
* `split_potentials` now unwraps: cheapness is a property of the component under the
  `TemperedProductPotential`, not of the wrapper, so without unwrapping every tempered component
  read as expensive and multi-rate degenerated.

**Centering** is refused at build time rather than at the first transition: a chart's
`(mu, sigma)` is shared by every temperature, so it is neither a per-rung quantity nor well
defined from the product coordinate. It is opt-in and off by default, so this costs little.

### Learned metrics over the product space

A `learned_metric` block is **non-separable**: its mass depends on position, so there is no drift
`q += eps·M⁻¹p` for `ProductKinetic` to rebuild from a velocity. `_DiagBlock.flow` instead drifts
the block's own coordinates *and* kicks the dependency momenta by an autodiff force, so that flow
is vmapped per lane and both `q` and `p` come back changed.

This is exact, not an approximation. The product Hamiltonian is a sum of K per-temperature terms
with **no coupling between rungs** — rungs meet only in the swap move — so composing each lane's
own flow over disjoint coordinates *is* the flow of `Σ_k T_i^(k)`, and reversibility and volume
preservation hold lane-wise. The `depends` kick is a gradient with respect to that lane's `q_k`
alone, which is the full gradient because `T_i^(k)` depends on nothing else. A test asserts the
product flow equals applying the inner flow to each temperature separately *with distinct
parameters per rung*, which is what would catch cross-temperature leakage.

**Each rung learns its own metric.** `ProductKinetic.initial_mass_params` already broadcasts the
block's parameter pytree with a leading `K` axis, and `MetricAdaptation` runs in the K hosts,
each fitting rung `k`'s own `(q_k, score_k)`. A hot rung's funnel is shallower, so it genuinely
wants a different metric — the same argument that puts the mass per temperature.

The payoff is the reason this was worth building. A single shared mass cannot serve a funnel whose
scale varies by orders of magnitude, so PT with a constant diagonal mass stalls short of the neck.
On `neal_funnel_blocks` (4 seeds, K=3, `beta_min=0.5`, float32):

| block kind | median v.min | distinct draws | median divergences |
|---|---|---|---|
| learned metric | **−9.55** | 800 / 800 | **0** (all four seeds) |
| diagonal mass | −6.60 | 772 | 20 |

So the learned metric rescues PT on the funnel much as WALNUTS did, and for the same reason: it
absorbs scale variation a constant mass cannot. Deeper ladders on this problem still need x64.

Note the step size: `eps` is scalar under NUTS and `(K·n,)` under independent acceptance, where
it is built by `jnp.repeat` and so is uniform within a rung by construction. A genuinely
per-*coordinate* step size is not supported by non-separable blocks — nor is it untempered, since
the inner flow multiplies a block-sized velocity by `eps`.

### What the ladder adaptation does on a large-data model

With the cache bug above fixed, `hmm_gaussian` is measurable for the first time. Five arms, 8 seeds
each, 1000 warmup / 1000 draws, factory defaults otherwise (`tempered=["likelihood"]`):

| arm | good mode | min ESS median [range] | divergences median [max] | β_min reached | secs |
|---|---|---|---|---|---|
| `nuts` (no PT) | 7/8 | 368 [3, 570] | 0.000 | — | 22 |
| K=4, β_min held 0.01 | 8/8 | 427 [21, 662] | 0.007 [0.215] | 0.010 | 766 |
| **K=4, β_min adapted** | 8/8 | **586 [481, 720]** | **0.000 [0.000]** | 0.094 [0.040, 0.105] | 777 |
| K=8, β_min held 0.01 | 8/8 | 444 [110, 805] | 0.004 [0.042] | 0.010 | 818 |
| K=8, β_min adapted | 8/8 | 435 [103, 717] | 0.024 [0.044] | 0.004 [0.003, 0.005] | 841 |

"Good mode" is the cold chain's `mu` reaching [9.16, 18.65, 29.53]; plain NUTS's seed 2 instead
sits at [2.37, 3.75, 5.08] with R̂ 2.83 and ESS 3 — the stuck mode this doc has asked about since
the beginning, and every tempered arm escapes it.

**`adapt_beta_min` fixes a range that is too narrow and mildly spoils one that is already right.**
At K=4 the held ladder is disconnected (swap ~0.02 per pair) *and* cannot repair itself, because a
uniform `α − α*` is a no-op on softmax shares; freeing the range finds β_min ≈ 0.09, brings every
pair to 0.18–0.27 against the 0.234 target, and gives zero divergences on all 8 seeds. At K=8 the
geometric default already accepts 0.21–0.36 with nothing adapted, so freeing the range lets a
slightly-above-target acceptance widen it to β_min ≈ 0.004 — 6× the divergences for no gain. The
adaptation converged to a consistent ladder on every seed in both cases: it does not run away, in
either direction, once it is fed a chain that moves.

This is also the first evidence on **choosing K**. Held at β_min = 0.01 the achieved swap rate is
0.02 at K=4 and 0.25 at K=8, so the rule this doc proposed — grow the ladder while the rates sit
below target — picks K=8 and stops, which is the right answer here.

### Does PT actually escape the spurious modes? Yes — 48 paired seeds

At n=8 the mode-finding comparison was 8/8 against NUTS's 7/8, indistinguishable. Re-run at **48
paired seeds** it is not close:

| | NUTS | PT (K=4, free β_min) |
|---|---|---|
| stuck in a spurious mode | **17 / 48 (35.4%)** | **0 / 48** |
| min ESS, good seeds (min/med/max) | 285 / 437 / 638 | 342 / **599** / 833 |
| min ESS, stuck seeds | 2.8 / 21.7 / 62.6 | — |
| seconds/run (median) | 16.3 | 745.0 |

**McNemar exact, two-sided, 17 discordant pairs all favouring PT: p = 1.5 × 10⁻⁵**, and zero pairs
the other way — PT never broke a seed NUTS got right. PT also mixes *better* where NUTS succeeds:
median min ESS 599 against 437.

The comparison is **paired**, which an earlier version of this section denied. `UniformInit` draws
`U(-2,2)^coord_dim` and PT's `coord_dim` is `K·n`, but JAX's threefry is prefix-stable, so
`uniform(key, (K·n,))[:n] == uniform(key, (n,))` — the cold chain starts at the *identical* point
in both arms at a given seed (verified). So each discordant pair is one sampler escaping where the
other could not from the same initial state, and no separate "escape test" starting at the bad mode
is needed.

**Two things to carry away beyond the verdict.** First, the failures are a *family*: 14 of 17 are
the unoccupied-state mode (`mu ≈ [*, 9.26, 21.9]` with the third state unidentified and drifting
over −20.9 … +5.8), plus one all-below-the-data mode and one collapsed pair. Second, and worse for
anyone relying on standard practice: **14 of the 17 stuck seeds have max split-R̂ ≤ 1.09** with min
ESS 20–63. The spurious modes are well-mixed internally, so a within-chain R̂ has nothing to catch,
and restarting until R̂ looks acceptable *selects for* the failure rather than avoiding it.

The cost is 46×, and that is the honest frame: NUTS fails on 35% of starts and mostly fails
*quietly*, so the alternative is not "one cheap run" but "several cheap runs plus a way to tell
which is right".

Full writeup, with the caveats about generality (one dataset, one model, one PT configuration):
`tests/experiments/writeups/hmm_gaussian_pt_vs_nuts.md`.

### Moving a rung's beta invalidates that rung's cache

A swap moves a replica to another rung's β; the ladder update moves a rung's β under a replica.
**These are the same event seen from two sides, and both invalidate the same cache** — the
per-potential `potential_values` and `potential_grads`, which hold `w_k·V_k` and `w_k·∇V_k` at the
β that was current when they were computed. `_swap` has always re-seeded after an exchange. The
ladder update did not, and the omission is not a small inefficiency: the next trajectory's energy
baseline `H0` is seeded straight from those cached values, so every leaf of the tree appears to
carry an energy error of `Δβ·V`.

The damage is the **product** of two factors, which is why it went unseen for so long:

| | `V` | ladder motion | error/step | outcome |
|---|---|---|---|---|
| every existing ladder test (2-d Gaussian) | ~1 | ~1e-3 | ~1e-3 nats | invisible |
| `hmm_gaussian` (T = 500) | ~1e5 | ~3e-3 | **~300 nats** | chain destroyed |

On `hmm_gaussian` acceptance collapses to zero, Robbins–Monro drives the step size from 1.6e-2 to
**1e-13 within 25 iterations**, and NUTS then builds a maximum-depth tree every step without ever
U-turning — a ~400× cost for a K=4 product, and the per-temperature dense score-covariance mass
eventually goes singular on top. Measured at K=4, 300 warmup steps:

```
                                step size      tree depth
untempered                        0.739           3.02
K=4, adapt_ladder=False           0.367           4.69
K=4, ladder on, beta_min held     2.7e-13 / 1.6e-14 / 6.2e-12   10.00   (seeds 1,2,3)
K=4, ladder on, beta_min free     3.0e-19 / 1.1e-19 / 1.4e-19   10.00   (seeds 1,2,3)
K=4, ladder on, re-seeded (fix)   0.399           4.28
```

Both mass modes (dense and diagonal) are healthy with the ladder frozen and destroyed with it
adapting, so the mass — the obvious suspect, with its own record of collapse — is not implicated.
Neither is the choice to hold or free `beta_min`; freeing it only moves the ladder faster and so
digs the hole deeper. The **causal order** settles it: at iteration 25 the step size has already
fallen 100× while the ladder has barely moved and all three pairs are still swapping (0.111,
0.062, 0.222). The ladder deformation that follows is a consequence of the frozen chain, not its
cause — the reverse of the reading the end-state ladders invite.

Reproducing it in a test needs both factors, and the natural attempts fail: a d-dimensional
Gaussian only reaches `V ~ d/2β` (~240 at d = 200), and with `beta_min` held a ladder whose pairs
never swap does not move at all, because a uniform `α − α*` is a no-op on softmax shares. What
works is a **large additive constant** in the log-density (the stand-in for 500 observations)
*plus* a free `beta_min` to keep the ladder moving: 0.0024 broken vs 2.04 fixed, with the same
model at offset 1e3 unaffected as the control.

### The user-facing surface: what the product model is not

`ProductModel` is a *view*, not a model: the dimensions, the charts, and the coordinate↔sample
maps, `vmap`ped over the ladder. It deliberately does **not** answer
`log_prob_at_coordinate`. Over the product space the sampled target is the β-weighted sum the
tempered potentials evaluate, and the ladder they weight by is adapted, so it travels in the
Hamiltonian context rather than on the model. A method with `Model`'s signature could only bake
in some fixed ladder and would quietly return the wrong density the moment the ladder moved —
worse than not having it. `-sum(state.potential_values.values())` is that density, already
cached in the state.

Two user-facing entry points asked the model for things the view does not have, and both were
invisible until someone ran the whole lifecycle (the PT tests went straight to
`warmup`/`sample`):

* **`initialize()`.** `UniformInit` tested `model.log_prob_at_coordinate`. It now tests the
  candidate *state's* `log_prob` — the sampler's own target, correctly β-weighted at the current
  ladder. For an untempered sampler the two are the same number (the potentials are the model's
  components plus the chart Jacobian): measured identical to the last bit over 600 draws across
  a Gaussian, Neal's funnel, and `hmm_gaussian`'s simplex/ordered/positive charts. The candidate
  state is built either way, so the accepted draw costs nothing extra. Relatedly,
  `state_at_coordinate` now builds its context through `self.context(...)` instead of the
  constructor, so the ladder reaches the potentials there too.
* **`summary()`.** `ProductSpaceMixin` already narrows the retained draws and the saved scores to
  the cold chain; the *model* they are evaluated against has to be narrowed with them. Hence the
  `summary_model` seam on `BaseSampler` (its own model by default), which the mixin overrides to
  the base model. Handing `summarize` the product view asks for an `ambient_score` and
  `stein_terms` it has no business defining, over an `ambient_dim` that is K times too wide.

## What must be verified

The failure this design can produce quietly is a **biased β=1 marginal**. Diagnostics will not reveal it: a biased PT chain can have excellent R̂ and ESS. So the load-bearing tests compare against known answers.

1. **Exactness against an analytic reference** — `correlated_gaussian` and `positive_lognormal` through `evaluate`, whose energy-distance and mean/variance z-score checks are the suite's existing standard for "samples the right thing". With K > 1, the β=1 marginal must pass them.
2. **K = 1 reduces to the base sampler**, ideally bit-identically on the same seed stream.
3. **β = 0 gives the flat target / prior**, checkable in closed form on a Gaussian.
4. **Swaps in isolation**: equal β must accept with probability 1 and change nothing distributionally; a two-temperature Gaussian pair has an analytic swap rate to match.
5. **It finds modes** — the entire point. A well-separated Gaussian mixture where NUTS demonstrably sticks, over ≥8 seeds, comparing the fraction of seeds that recover both modes with and without PT. Per the working method (`CLAUDE.md`): never one seed, and state the expected number before measuring.
6. ~~**`hmm_gaussian`**: does seed 2 escape?~~ Answered: yes, and so do the other 16 seeds
   NUTS gets stuck on — see "Does PT actually escape the spurious modes?" above.

## Known costs and open questions

- ~~**Trajectories shorten as K grows.**~~ **Measured, and it is the other way round.** The claim
  here was that the summed U-turn is bounded by the earliest-turning temperature, giving shorter
  trees than any single chain would choose. The reasoning does not hold: a sum of K terms goes
  negative when the *aggregate* turns, which needs roughly the average lane to turn, not the
  earliest — the earliest lane's turn is masked by the K−1 that have not turned yet.

  Instrumenting each lane's own U-turn passively (500 warmup + 500 draws, 8 seeds per cell,
  step-size and per-rung mass adaptation on) puts the cold lane's own preferred depth `j_cold`
  *below* the summed rule's stopping depth in every cell measured:

  | target | K | j_cold | j_sum | j_sum − j_cold |
  |---|---|---|---|---|
  | correlated_gaussian | 2 / 4 / 8 | 2.08 / 2.19 / 2.36 | 2.53 / 2.92 / 3.00 | +0.44 / +0.70 / +0.64 |
  | neal_funnel_blocks | 2 / 4 / 8 | 1.59 / 1.92 / 2.81 | 1.72 / 2.12 / 2.68 | +0.18 / +0.52 / +1.03 |
  | hmm_gaussian | 4 | 4.79 | 5.03 | +0.41 (all 8 seeds positive, +0.20…+0.95) |

  So the cold end's inefficiency is **wasted gradients on an over-long trajectory**, not a truncated
  one. Writeup: `tests/experiments/writeups/pt_lane_turn_depths.md`.
- ~~**The divergence threshold scales with K.**~~ Measured and fixed — see "WALNUTS over the
  product space" above. It is now `DEFAULT_DIVERGENCE_THRESHOLD * K` by default.
- **Joint selection couples acceptance** (above). Measure PT against the plain base sampler on a *unimodal* target, where PT should cost something but not much — if it costs a lot, the coupling is the first suspect.
- ~~**The ladder must be scaled to the data, and the default is not**~~ — resolved: see "What the
  ladder adaptation does on a large-data model" above. `adapt_beta_min=True` finds the range at
  K=4, and at K=8 the geometric default is already right. The historical finding below was
  measured before the cache fix and its divergence/swap numbers describe a frozen chain.
  Original note: **the finding that keeps PT from helping `hmm_gaussian` today**. That model has T = 500 observations and a log-density of
  order 1e5, so adjacent rungs of a `beta_min = 0.02` ladder are thousands of nats apart: swap
  acceptance is **0.00 across every pair**, and the shared trajectory cannot integrate targets
  that differ so much, giving **~100% divergences**. Measured:

  ```
  K=6,  beta_min=0.02, threshold 1e3    div 499/500   swap [0, 0, 0, 0, 0]
  K=6,  beta_min=0.02, threshold 1e6    div 500/500   swap [0, 0, 0, 0, 0]
  K=12, beta_min=0.5,  threshold 1e6    div  15/500   swap [0.93, 0.92, 0.93, ..., 0.85]
  ```

  Note the middle row: the divergence threshold was the *obvious* suspect and is **not** the
  cause — raising it a thousandfold changed nothing. The usable ladder for this model spans
  `beta` 1 → 0.5, which barely flattens anything, so PT as configured here does not rescue
  `hmm_gaussian`'s stuck mode. The general lesson is that the usable temperature range narrows as
  the data grows, which is what makes **swap-rate-targeted ladder adaptation** the next piece of
  work rather than a refinement: a hand-set `beta_min` cannot be right across models.
- **The ladder adaptation was silently destroying large-data chains** until the cache re-seed
  above; every measurement of it on `hmm_gaussian` before that fix describes a frozen chain rather
  than the adaptation. A further structural point fell out: with `beta_min` **held**, a ladder
  none of whose pairs swap cannot move at all — a uniform `α − α*` is a no-op on softmax shares —
  so on such a model the held-endpoint adaptation is incapable of helping by construction, and
  `adapt_beta_min=True` is the only form that can.
- **The ladder is now adapted** (above), which resolves the `beta_min` guess but not the choice of
  **K**, still the caller's. Adaptation redistributes the rungs it is given; it cannot add one. On a
  model like `hmm_gaussian`, where every adjacent gap is already too wide at K = 8, the target rate
  is unreachable at any spacing and the ladder converges to the least-bad compromise rather than a
  working one. Choosing K from the achieved swap rates — grow the ladder while the rates sit below
  target — is the natural follow-on.
- **A per-rung energy-error criterion.** The line search compares the *summed* error against
  `K·δ`. Refining until the **worst** rung is within `δ` — `max_k |ΔH_k|` against an unscaled
  threshold — is the better criterion in principle, since a sum lets one badly-behaved hot rung
  hide behind K−1 well-behaved ones. It needs per-temperature energies threaded through the line
  search rather than the single scalar the integrator interface passes around, which is exactly
  what makes the sum form free; the sum form is what is implemented.
- **Out of scope**: a factory rule for deciding when to reach for PT. Wiring PT into
  `SamplerSpec.base` has since shipped (the `pt_` prefixes — see "The factory seam" above); what
  remains out of scope is a rule that *selects* it, which is still an explicit user choice.
