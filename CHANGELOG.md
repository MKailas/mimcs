# Changelog

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
