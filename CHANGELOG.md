# Changelog

## Unreleased

- **Chart and ladder adaptation are several times faster.** Recharting a chart, or moving the
  temperature ladder, must refresh the cached potential values and gradients or the next step
  integrates a Hamiltonian nobody is simulating. That reseed ran eagerly, once per warmup
  iteration --- one full `value_and_grad` of every potential dispatched primitive by primitive.
  It is now one shared, compiled helper: an adaptive unit-vector chart takes warmup(500) from
  8.4s to 2.8s, and parallel tempering at K=6 from 4.2s to 2.3s. Compiling the unit-vector chart
  helpers takes vMF a further 2.9s -> 2.1s. Neutral on models that do not recharter. The test
  suite's centering-heavy chunk goes 768s -> 184s on the same idle machine, and the whole suite
  51 -> 31 minutes.
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
