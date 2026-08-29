# `algo_kwargs` reference

Every key `SamplerSpec.algo_kwargs` accepts, and every keyword a hand-built sampler takes. There
are 85 of them across 18 modules, and they are invisible to introspection: adaptation mixins read
their options by name out of `**kwargs`, so no signature, type hint or autodoc page lists them.
Hence this table.

```python
spec = analyze(model)
spec.algo_kwargs = {"target_accept": 0.9, "max_tree_depth": 12}
sampler = spec.build()
```

Unknown keys are **silently ignored** — they flow through `**kwargs` to a terminal hook that
discards them. A typo costs you the setting with no error, which is the one thing to watch. (The
spec's own fields *are* validated: `base`, `integrator`, `mass_adapt`, `terminate`,
`integrator_params` and `tempering_params` all raise on an unknown value.)

**Only the mixins the factory actually composed read anything.** Which those are depends on
`spec.base`, `spec.mass_adapt`, `spec.terminate`, `spec.integrator` and each block's `kind`, so a
setting for a mixin that was not composed is inert. `spec.build()` logs the composed list at INFO.

---

## Three traps

These are the reason this page is hand-written rather than generated: no tool could infer any of
them from the source.

**1. `target_accept`'s effective default is 0.8, not the 0.234 in the source.** `RobbinsMonroStepSize`
defaults to 0.234 (the random-walk optimum), but `build_sampler` does
`kwargs.setdefault("target_accept", 0.8)`, so every factory-built sampler gets 0.8. A hand-built
one does not. `step_size_adapt_rate` is derived from whichever value applies.

**2. `mass_polyak` is read by five mixins, with two defaults and opposite meanings.**

| Mixin | Default | What it means there |
|---|---|---|
| `ScoreMassAdaptation` | `False` | freeze an EMA of the SGD iterate for sampling |
| `MassMatrixAdaptation` | `False` | a suffix average that **biases** this estimator — the RM covariance is a slow oscillating transient, so averaging it is wrong |
| `LowRankAdaptation` | `False` | Polyak-average the diagonal part |
| `MetricAdaptation` | **`True`** | freeze the averaged metric parameters |
| `RelativisticMassAdaptation` | **`True`** | as above |

So `algo_kwargs={"mass_polyak": True}` is good advice or bad advice depending on which mixins your
spec composed. There is no way to set it for one and not another.

**3. `step_size` has three defaults by algorithm**: `1.0` for `RandomWalkMH`, `0.5` for the HMC
family, and `0.5` again as `StepSizeLineSearch`'s starting point for its backtracking search.
`SamplerSpec.step_size` (a field, not an `algo_kwargs` key) is what the factory passes.

---

## Step size

`RobbinsMonroStepSize` — composed when `spec.adapt_step_size` is true.

| Key | Default | Meaning |
|---|---|---|
| `target_accept` | `0.234`, **0.8 via the factory** | acceptance the adaptation drives toward |
| `step_size_adapt_rate` | derived from `target_accept` | RM gain scale, `1/sqrt(a₀(1−a₀))` |
| `step_size_adapt_kappa` | `0.6` | RM decay exponent |
| `step_size_adapt_n0` | `5.0` | RM offset |
| `accelerated` | `False` | Kesten-style acceleration |

`LineSearchStepSizeAdaptation` **replaces** the above (it subclasses it) when the integrator emits
a step-size proxy — i.e. under `line_search` / `markovian_line_search`. Same keys: it differs only
in reading the integrator's proxy acceptance rather than the real one, which is ≈1 under WALNUTS
and would otherwise drive the step size away.

## Mass

`ScoreMassAdaptation` — the default (`spec.mass_adapt="score"`).

| Key | Default | Meaning |
|---|---|---|
| `score_mass_n0` | `5.0` | SGD offset |
| `score_mass_kappa` | `0.75` | SGD decay exponent |
| `score_mass_clip_frac` | `0.1` | target fraction of steps whose gradient is clipped, per coordinate |
| `score_mass_center_grad` | `True` | fit the score *covariance* rather than the second moment |
| `score_mass_lr_const` | `1.0` | learning-rate constant; also accepts the string `"1/d"` |
| `score_mass_polyak_warmup` | `False` | let the EMA drive warmup, not just sampling (implies `mass_polyak`) |
| `mass_polyak` | `False` | see trap 2 |

`MassMatrixAdaptation` — `spec.mass_adapt="covariance"`.

| Key | Default | Meaning |
|---|---|---|
| `mass_min_samples` | `50` (also `setdefault` by the factory) | **nothing is written before this many draws** — a shorter warmup silently leaves the mass at identity |
| `mass_adapt_n0` | `5.0` | RM offset |
| `mass_adapt_kappa` | `0.75` | RM decay exponent |
| `mass_polyak` | `False` | see trap 2 — off here *deliberately* |

`LowRankAdaptation` — composed when any block has `kind="lowrank"`. Unaffected by `mass_adapt`.

| Key | Default | | Key | Default |
|---|---|---|---|---|
| `lowrank_n0` | `5.0` | | `lowrank_mass_lr_const` | `1.0` |
| `lowrank_kappa` | `0.75` | | `lowrank_oja_const` | `1.0` (Sanger/Oja rate for the low-rank directions) |
| `lowrank_clip_frac` | `0.1` | | `lowrank_min_samples` | `50` |
| `lowrank_center_grad` | `True` | | `mass_polyak` | `False` |

## Learned metrics

`MetricAdaptation` — composed when a block has `kind="learned_metric"` with no `shape`.

| Key | Default | Meaning |
|---|---|---|
| `metric_adapt_kappa` | `0.75` | SGD decay exponent |
| `metric_adapt_n0` | `5.0` | SGD offset |
| `metric_clip_frac` | `0.1` | per-coordinate gradient clipping fraction |
| `metric_center_grad` | `False` | **off by default here**, unlike `ScoreMassAdaptation`: a single marginal mean distorts a *conditional* block's fit |
| `mass_polyak` | **`True`** | see trap 2 |

`ShapedMetricAdaptation` — when a learned-metric block sets `params["shape"]`.

| Key | Default | | Key | Default |
|---|---|---|---|---|
| `shaped_kappa` | `0.75` | | `shaped_oja_const` | `1.0` |
| `shaped_n0` | `5.0` | | `shaped_min_samples` | `50` |
| `shaped_clip_frac` | `0.1` | | | |

`RelativisticMassAdaptation` — **[experimental]**, not factory-reachable.
`rel_mass_n0` `5.0`, `rel_mass_kappa` `0.75`, `rel_mass_clip_frac` `0.1`, `mass_polyak` **`True`**.

## Chart adaptation

`RobustCenteringAdaptation` / `CenteringAdaptation` — composed when `spec.centering` is true. Acts
only on parameters declared `centered=True`.

| Key | Default | Meaning |
|---|---|---|
| `center_min_samples` | `50` | draws before the first recentering |
| `center_adapt_n0` | `5.0` | RM offset |
| `center_adapt_kappa` | `0.75` | RM decay exponent |
| `center_floor` | `1e-8` | lower bound on the fitted scale |

`UnitVectorCenteringAdaptation` — composed automatically when the model has a `UnitVectorParameter`
with `adaptive=True` (the default), so **no spec field turns this on**; `adaptive=False` on the
parameter turns it off.

| Key | Default | Meaning |
|---|---|---|
| `unit_vector_target_frac` | `0.5` | target fraction of draws inside the projection disc, which sets the plane offset |
| `unit_vector_min_samples` | `50` | draws before the first recharting |
| `unit_vector_adapt_n0` | `5.0` | RM offset |
| `unit_vector_adapt_kappa` | `0.75` | RM decay exponent |

## Warmup termination

`_WarmupTermination` — shared by both criteria.

| Key | Default | Meaning |
|---|---|---|
| `min_warmup` | `500` | no check before this. **Note the interaction with `mass_min_samples`** |
| `check_every` | `100` | iterations between checks |
| `max_warmup` | `50_000` | the budget `warmup()` with no `n` runs to |
| `burn_in_frac` | `0.1` | fraction of the buffer discarded before testing |
| `feature_thin` | `1` | thinning applied to the feature buffer |
| `patience` | `3` | consecutive passing checks required to stop |
| `keep_features` | `False` | hold on to the feature buffer after warmup instead of freeing it |

`keep_features` is about memory. The buffer is one `model.features` row per retained draw
(`rows x n_features x 4` bytes, and the dynamic burn-in search keeps a standardized `float64` copy
of the same size again), it grows for the whole of warmup, and it is dead weight for the whole of
sampling — usually the longer phase. It is therefore released once warmup is **over**: either the
criterion fired, or `max_warmup` ran out. Saving it is opt-in for *any* circumstance, so a warmup
that gave up without firing releases it too.

A `warmup(n)` call merely *returning* is not the end of warmup — `warmup(500)` twice is a
supported way to continue, and the second call's checks read what the first accumulated — so
nothing is freed there. What the criterion reported (`warmup_mixing_stats()`,
`warmup_burn_in_estimates()`, `warmup_terminated_early()`) survives either way; only the raw
observations go. Set `keep_features=True` to inspect them afterwards.

`GelmanRubinTermination` (`spec.terminate="rhat"`): `rhat_threshold` `1.01`.

`ClassifierTermination` (`spec.terminate="classifier"`, the default) — the largest surface here.

| Key | Default | Meaning |
|---|---|---|
| `accuracy_threshold` | `0.55` | held-out accuracy below which early and late draws are indistinguishable |
| `rule` | `"threshold"` | or `"permutation"` — **validated, raises** |
| `val_every` | `5` | validation split period |
| `l2` | `1e-2` | ridge penalty on the logistic fit |
| `n_perm` | `20` | permutations, under `rule="permutation"` |
| `perm_block_len` | `100` | block length for the permutation null |
| `perm_alpha` | `0.05` | permutation test level |
| `perm_seed` | `0` | permutation RNG seed |
| `burn_in` | `"fixed"` | burn-in mode — **validated**. The dynamic modes were measured and rejected (doc 10) |
| `burn_in_min` | `50` | |
| `burn_in_min_frac` | `0.0` | |
| `burn_in_max_frac` | `0.5` | |
| `burn_in_max_iter` | `4` | |
| `burn_in_tail_tol` | `0.1` | |
| `burn_in_null` | `"scaled"` | **validated** |
| `burn_in_null_slack` | `0.5` | |
| `burn_in_objective` | `_burnin.additive` | a **callable** |
| `burn_in_fit_opts` | `{}` | a nested dict; its sub-keys are not documented |

## Parallel tempering

`LadderAdaptation` — composed with any `pt_` base.

| Key | Default | Meaning |
|---|---|---|
| `adapt_ladder` | `True` | adapt the interior rungs toward the target swap rate |
| `swap_target_accept` | `0.234` | the Atchadé–Roberts–Rosenthal optimum |
| `ladder_adapt_n0` | `5.0` | RM offset |
| `ladder_adapt_kappa` | `0.75` | RM decay exponent |
| `adapt_beta_min` | `False` | free the hot end too. Read `docs/design/13`'s "When to free `beta_min`" first: safe when an untempered component holds the hot end down, unsafe otherwise |
| `keep_all_temperatures` | `False` | store every rung's draws, not just the cold chain |

`keep_all_temperatures` is about memory, and follows `keep_features` above: the extra data is kept
only if you ask. A tempered run samples the `K`-fold product but reports the cold chain, so storing
all of it means `(K-1)/K` of the draw store — 87.5% at `K = 8` — is thrown away when the draws are
read. The gradients were always narrowed at save time; the draws now are too. `get_samples_all()`
needs this flag and raises a message naming it otherwise; `get_samples()`, `get_samples_flat()`,
`summary()` and the evaluation harness are unaffected either way, since they only ever wanted the
cold chain.

The ladder itself (`n_temperatures`, `betas`, `beta_min`, `tempered`) goes in
`spec.tempering_params`, **not** here — and those keys *are* validated.

`PerTemperatureAdaptation` also reads `adapt_mixins` (a tuple of mixin classes), and
`ReplicaExchangeMixin` requires `betas`. Both are set by `parallel_tempering` and the factory;
they are the one pair on this page that is machinery rather than configuration.

## Base samplers

Named constructor parameters rather than `kwargs.get` lookups, so these are visible to
introspection, but they travel through `algo_kwargs` the same way.

| Key | Default | Where |
|---|---|---|
| `n_leapfrog` | `20` | `HMC`, `RandomizedHMC` |
| `max_tree_depth` | `10` | `BaseNUTS`. Costs buffer memory as `2^(J−1)` — see `docs/design/03` |
| `divergence_threshold` | `1000.0` | `BaseNUTS`; a `pt_` base scales it by K |
| `step_size` | `0.5` / `1.0` | see trap 3 |
| `buffer_size` | `1024` | RNG buffer; prefer the build keyword `build(buffer_size=…)` |
| `seed` | `0` | prefer the build keyword |
| `save_gradients` | `True` | `BaseSampler`; keeps coordinate gradients for `summary()`'s Stein diagnostic |

## Random-walk MH only

`DiagonalCovarianceAdaptation`: `cov_floor` `1e-8`, `cov_min_samples` `10`, `cov_adapt_n0` `5.0`,
`cov_adapt_kappa` `0.75`.

## Initialization

Inert unless `sampler.initialize()` is called. `UniformInit`: `init_radius` `2.0`,
`init_max_tries` `100`. `StepSizeLineSearch`: `step_size` `0.5`, `init_target_accept` `0.9`,
`init_step_size_min` `1e-6`, `init_step_size_max_halvings` `60`.

---

Shared constants: `DEFAULT_KAPPA = 0.75` and `DEFAULT_N0 = 5.0`
(`mimcs/adaptation/_stochastic.py`), `DEFAULT_L2 = 1e-2` (`_logistic.py`),
`DEFAULT_DIVERGENCE_THRESHOLD = 1000.0` (`hmc/nuts.py`), `OPTIMAL_SWAP_ACCEPT = 0.234`
(`pt/ladder.py`).
