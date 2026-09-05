# Dynamic Adaptation: Ending Warmup

## Motivation

Warmup length is otherwise a magic number. `warmup(4000)` runs exactly 4000 iterations whether the
chain settled at 300 — wasting 3700 — or is still nowhere near stationary at 4000, in which case
the draws that follow are wrong and nothing says so. Goal 1 of `00_overview.md` has always claimed
"adaptive MCMC determines its own adaptation schedule"; this is the component that makes it true.

The question "is the chain mixing well?" is easy to ask and hard to make precise. This document is
mostly about that difficulty.

## What we look at, and what we deliberately do not

### Not the adapted parameters

The tempting signal is the step size, the mass, the charts: watch them settle, then stop. It
answers the wrong question. Every adaptation in the library is a Robbins–Monro iterate whose
per-iteration change is `O(gain) → 0` **by construction** — so they settle whether or not the
chain has found the target. Their convergence is a property of the schedule, not evidence about
the draws.

That is also what makes stopping *safe*: diminishing adaptation is the condition adaptive MCMC
needs for validity (see `mimcs.adaptation.UnitVectorCenteringAdaptation`, where a chart that
provably never converges is nevertheless fine on exactly this ground). Since it holds throughout,
the criterion is free to be about the draws alone.

### Not the coordinates

Coordinates are a device for computation. They are relabelled whenever a chart adapts — the
centering and unit-vector mixins do this every iteration, holding the sample fixed and moving the
coordinate under it. A diagnostic computed in coordinate space would be measuring a moving ruler.
And for a manifold parameter the coordinate is not even the same dimension as the draw
(`coord_dim ≠ ambient_size` for `unit_vector`). What we care about is the samples.

### Features: the layer above samples

R̂, Geweke and a classifier are none of them computed from a draw. They are computed from
**observables** — fixed scalar functions of a draw. Usually the observable is left implicit
("R̂ of coordinate 3"), but it is a real layer, and naming it is what lets one criterion serve
Euclidean, bounded and manifold parameters without knowing which is which:

```
coordinate  ──charts──▶  sample  ──features──▶  observables  ──▶  diagnostic
(computation)            (what the             (what a           (mixing?)
                          model means)          diagnostic reads)
```

Each parameter declares its own (`BaseParameter.features`, `Model.features`). The default is
`[x_j, x_j²]` per ambient coordinate: enough to catch a chain whose location or whose spread is
still moving.

Two things fall out of taking features seriously:

- **`x²` gives plain R̂ the scale sensitivity it is famous for lacking.** R̂ compares means, so on a
  raw draw it is blind to two halves that differ only in scale — the defect *folded* R̂ (Vehtari et
  al. 2021) exists to repair. On `x²` a scale change *is* a mean change, so the plain statistic
  sees it. Measured: two halves differing 1× vs 2× in sd give R̂ = 1.00 on `x` and 1.12 on `x²`.
- **A discrete parameter contributes its bare value**, not `[x, x²]` — for a binary label
  `x² == x` exactly, so the second block would be a duplicate column. Its features *do* enter the
  criteria, and should: a chain still reassigning cluster labels is not mixed, whatever the
  continuous block is doing. One caveat worth knowing, because it points the wrong way: a label
  that never moves has zero variance, so `ess_1d` returns `n` and `split_rhat` returns 1.0 by their
  no-variance guards — a **stuck** coordinate reads as perfectly converged. The `discrete_moves`
  diagnostic is what catches that, and it is the column to look at first when a discrete model's
  numbers look too good (doc 14).
- **`unit_vector` must override the default.** A unit vector satisfies `Σ_j x_j² = 1` exactly, so
  its `d` squared features are perfectly collinear with the intercept of any model fitted on them:
  the design matrix is rank-deficient and coefficients are non-identified. Dropping `x_d²` costs
  nothing — it is `1 − Σ_{j<d} x_j²` — and restores full rank. This is precisely what a
  per-parameter feature declaration is for.

## The split

All warmup draws so far, minus a burn-in prefix (default 10%), cut into two equal halves: an
**early** period and a **late** one. If the chain has settled these are two samples from the same
distribution and nothing distinguishes them; if it is still travelling, they differ. Everything
below is a way of asking "do these two halves look alike?".

For R̂ the halves *are* the chains, which makes this Stan's split-R̂ for a single chain — and is why
no second chain is needed. Running several chains in parallel is the usual answer, and a direction
this library deliberately does not take.

## `GelmanRubinTermination`

`max R̂ < 1.01` over the features (`rhat_threshold`). The well-tested comparison point. Two
calibration facts, both measured:

- **R̂'s null is 1, and it needs no correction.** `R̂² = (n−1)/n + δ²/(2W)` for a half-to-half gap
  `δ`; under the null `E[δ²] = 2σ²/n_eff`, and the `(n−1)/n` term is *designed* to cancel it, so
  `E[R̂²] ≈ 1 + (τ−1)/n`. Autocorrelation inflates it only at `O(τ/n)`. This is why R̂ travels.
- **1.1 is loose, which is why the default is 1.01.** `R̂ = sqrt(1 + δ²/2)`, so 1.1 tolerates
  `δ = 0.65 sd` — and that tolerance is independent of `n`. Vehtari et al. recommend 1.01
  (`δ = 0.2 sd`) for this reason, and that is what `rhat_threshold` defaults to. The conventional
  1.1 fires almost immediately on easy targets: a 2-D Gaussian started at `[200, 200]` passes at
  the first check.

## `ClassifierTermination`

Label each draw by which half it came from; fit a logistic regression on the features; score it on
held-out draws. If the chain has settled, no rule can beat chance. While it is moving, a draw's
position gives its period away.

Linear-with-`[x, x²]` makes this close kin to Geweke: its power comes from mean differences in the
features, so it is a two-sample test on the observables with the contrast direction estimated
rather than fixed. The classifier framing is preferred because it leaves room for a nonlinear rule
later, where no analytic null exists.

### The ridge is load-bearing

Early in warmup the two halves are often linearly separable — exactly when the criterion must not
fire — and separable data has no maximum-likelihood optimum. What that looks like in practice is
worse than a failure: L-BFGS does not diverge or complain. The gradient underflows exponentially
as `|w|` grows, `gtol` is met, and `converged` comes back **true**, at whatever point the tolerance
happened to bite. Measured: tightening `gtol` from 1e-4 to 1e-10 walks `|w|` from 2.8 to 6.1 on the
same data, reporting success every time. The fit is a property of the stopping rule, not the data.
With a ridge, `|w|` is identical across those tolerances.

### "Better than chance" is not 1/2

The hard part, and the reason for the empirical study. Under the null each half still carries its
own sampling fluctuation — inflated by autocorrelation — and the fit finds the direction along
which the two halves happen to differ. Validation is **interleaved** (every 5th draw), so held-out
draws sit *inside* the same halves and share their low-frequency excursions: the held-out set does
not protect against a fluctuation it also experiences, and the null sits above 1/2.

The obvious alternative — hold out a **contiguous tail** of each half, so train and validation
excursions differ and the spurious direction does not transfer — was built, measured, and removed.
See "Why not a blocked validation split" below.

*Predictions, recorded before measuring:* interleaved ≈ `Φ(sqrt(p/(2·n_eff)))`, rising with the
feature count `p` (≈0.52 at `p=6, n_eff=1000`; ≈0.62 at `p=200`) — in which case no single
threshold travels across dimension. Blocked ≈ 0.5, flat in `p`.

Note the contrast with R̂, which needs no such calibration. The difference is that R̂ compares a
*fixed* statistic with a built-in finite-sample correction, while the classifier *searches* a
`p`-dimensional space for the most discriminating direction. Held-out scoring is supposed to
control that search; block-excursion sharing is what defeats it.

#### What the null actually looks like

Chains started from an exact draw; 8 seeds, 3000 warmup draws
(`tests/experiments/mixing_calibration.py`):

| problem | p | n_eff | interleaved (sd) | `Φ(sqrt(p/2n_eff))` | blocked (sd) | max R̂ |
|---|---|---|---|---|---|---|
| gaussian | 4 | 2233 | 0.514 (0.025) | 0.512 | 0.509 (0.027) | 1.0027 |
| gaussian | 20 | 2405 | 0.524 (0.026) | 0.526 | 0.500 (0.022) | 1.0045 |
| gaussian | 100 | 2503 | 0.522 (0.022) | **0.556** | 0.498 (0.031) | 1.0043 |
| von Mises–Fisher | 5 | 1150 | 0.513 (0.013) | 0.519 | 0.478 (0.031) | 1.0043 |
| lognormal | 2 | 2042 | 0.512 (0.029) | 0.509 | 0.516 (0.029) | 1.0005 |

**Blocked** came out as predicted: ≈0.5, flat in `p`.

**Interleaved** is inflated as predicted, and the formula tracks it at `p = 4` and `p = 20` — then
overshoots badly at `p = 100` (0.556 predicted against 0.522 measured, some 4 standard errors).
The inflation climbs and then *plateaus* near 0.52 rather than growing without bound.

The cause is the **ridge**. The prediction came from an LDA heuristic in which the classifier
exploits all `p` noise directions; L2 shrinkage caps how many it can, so the effective dimension
stops growing with `p`. The ridge was added to keep separable fits finite — it turns out to also
flatten the null's `p`-dependence, which is what makes a fixed threshold viable after all. The
concern that motivated building a second validation split at all ("no single constant travels
across dimension") was therefore wrong, and wrong because of a design decision taken for an
unrelated reason.

**R̂'s null**: 1.000–1.005 everywhere, nowhere near 1.1. Confirms both that R̂ needs no calibration
and that 1.1 is a very weak requirement. Note that the null's upper end sits only 0.005 below the
1.01 default, so the margin is much narrower than it was at 1.1 — narrow enough that the criterion
could in principle fail to fire on a well-mixed chain, in which case warmup runs to `max_warmup`.
Measured, it fires: 2-D Gaussian 700/900/1000 iterations over three seeds, funnel 1700/2000/2200,
against 700–900 for all six at 1.1.

### The decision rule

Pluggable, because the above is an empirical question:

- `rule="threshold"` — accuracy against a constant. Cheap, and only as good as the constant's
  calibration.
- `rule="permutation"` — build the null in situ: relabel *contiguous blocks* (longer than the
  autocorrelation time, so that under a stationary null they are exchangeable) at random, refit,
  and see how unusual the real split is. Absorbs whatever inflation the setup has, for any model,
  at `n_perm` refits per check. Shuffling individual draws would be wrong: it destroys the very
  correlation that causes the inflation and hands back a null that is too optimistic.

#### Power: what the statistic does when the chain has *not* settled

Same script, `--power`: chains start at the default point and have to travel. 8 seeds.

| problem | 300 draws | 1000 | 3000 | R̂ at 3000 |
|---|---|---|---|---|
| 10-D Gaussian, started 30 away | 0.516 (0.081) | 0.478 (0.043) | 0.492 (0.016) | 1.0008 |
| 10-D Gaussian, started 3 away | 0.551 (0.044) | 0.506 (0.029) | 0.518 (0.023) | 1.0013 |
| Neal's funnel | 0.644 (0.086) | 0.605 (0.056) | **0.573** (0.029) | **1.0281** |

**The Gaussian is not an alternative.** Even started 30 units from the mode, NUTS arrives so fast
that the transient is over inside the 10% burn-in discard, and both halves are converged by the
first check. The criterion reporting "mixed" there is correct, not blind.

**The funnel is**, and it is the case for the classifier over R̂: at 3000 draws the statistic still
sits at 0.573 against a 0.52 null, while R̂ has fallen to 1.028. Under the conventional `R̂ < 1.1`
that is victory declared, on a target famously not converged at 3000 — the 0.65-sd tolerance
showing up in practice rather than in algebra, and the measurement behind the 1.01 default. At
1.01 this particular case no longer slips through (1.028 > 1.01, so warmup continues), which
narrows the gap to the classifier without closing it: 1.01 is still a fixed threshold on a
statistic that only sees means and squares, while the classifier builds its null in situ.

### The fit is buffered, or warmup is mostly compilation

A check refits on the whole feature history, so the design matrix grows by
`check_every · (1 − burn_in_frac) · (1 − 1/val_every)` rows every time — 72 at the defaults. Every
check therefore handed XLA a shape it had never seen. Measured on Neal's funnel through the factory
default, a 2000-iteration warmup spent **10.5 of its 16.1 seconds compiling**, across 489
compilation events, and grew RSS from 381 to 981 MiB — a cost paid again by every sampler in every
test, and a real contributor to how little of this test suite fits in memory at once.

Two changes fix it, and neither works without the other:

1. **Buffered rows.** `_logistic.row_buffer` rounds the row count up to a power of two and the
   padding carries **zero weight**. `logistic_loss` normalizes as `sum(wt·l)/sum(wt)`, so padded
   rows leave numerator and denominator alike untouched — the same trick `class_weights` already
   used to hold the shape steady across the burn-in search's candidates, lifted one level up to
   hold it steady across checks. The padding must be *finite*, not junk: those rows still flow
   through `X @ w`, and `0 · nan` is `nan`.
2. **A cached `jit`.** `minimize` is a bare `lax.while_loop`; called outside `jit` it binds that
   loop eagerly, rebuilding the cond/body jaxprs and missing the dispatch cache on **every call, at
   any shape, even an identical one** (~0.2 s a call on a 300×6 fit). So buffering on its own saves
   nothing whatsoever. `_logistic._fit_jit` wraps the fit in a cached `jax.jit` with the data as
   *arguments* — a closed-over array would be a compile-time constant and key the cache on its
   contents.

`minimize` itself is deliberately left unjitted: it reports how it terminated on the host, which it
cannot do under trace, and `tests/test_logging.py` pins those messages.

| | before | after |
|---|---|---|
| compilation events | 489 | **98** |
| XLA compilation | 10.5 s | **2.5 s** |
| warmup wall | 16.1 s | **5.5 s** |
| peak RSS | 981 MiB | **559 MiB** |

Warmup stopped at the same iteration. `accuracy` moved onto the numpy `scores` path in the same
change, for the reason its siblings already were: the validation block is a new row count at every
check too.

#### The padding is wasted work, and finer buckets are still worse

Rounding to a power of two means a 4320-row history is fitted at 8192 — **47% of every GEMV is
padding**, on every L-BFGS iteration of every check. Powers of √2 cut that to ~16% at the cost of
roughly twice as many buckets, and the obvious prediction is that the flops win, since a
compilation is paid once per bucket and the flops on every iteration.

Measured, the prediction is **wrong**. Replaying the sequence of heights a real warmup produces
(`check_every=100`, `max_iter=1000`, p=6000), one process per arm:

| checks | scheme | buffers | mean padding | total |
|---|---|---|---|---|
| 20 | powers of 2 | 5 | 46.8% | 3.67 s |
| 20 | powers of √2 | 8 | 15.6% | 3.72 s |
| 60 | powers of 2 | 7 | 42.4% | **20.9 / 21.0 / 22.3 s** |
| 60 | powers of √2 | 12 | 18.5% | 25.1 / 24.9 / 25.4 s |

The padding really is wasted work — at the same history, 5793 rows fits in 421.8 ms against 489.7 ms
at 8192, roughly proportional to the row count and with no penalty for the odd shape — and a
compilation really does cost ~0.30 s, shape-independent. But those two account for only about a
third of the gap, so **the mechanism is not fully pinned down**; the decision rests on the
end-to-end number, which reproduces at both warmup lengths and both iteration caps. `row_buffer`
stays as it is. `tests/experiments/writeups/classifier_check_cost.md` has the controls.

#### The history is float32 the whole way through

The store is float32. It used to be promoted to float64 by `_term_split`, gathered in float64 by
`_train_val`, standardized in float64 — and then rounded straight back to float32 by `_buffered`,
because that is what JAX canonicalizes `float` to. Two full-width copies existed only to be
discarded: 288 MB and 259 MB of them on a 6000-draw, 6000-feature history.

The float64 *arithmetic* is still there; only the float64 *copies* are gone. `fit_logistic` takes a
`standardize=(mu, sd)` argument that folds the standardization into the buffering pass, a block of
rows at a time, writing each block into the device-dtype buffer with **one** rounding — the same
single rounding, in the same place. Measured at 2000 draws × 6000 features, one process per arm:
peak RSS above entry 344 → **266 MB**, one check 529 → **422 ms**, accuracy identical to the last
digit (0.9888888888888889, on a history with a planted drift so that the number could move).

Two things are load-bearing:

- `np.mean(x, axis=0, dtype=np.float64)` on the float32 store is bit-for-bit `np.mean` of a float64
  copy, because float32 → float64 is exact and the reduction order depends only on the shape.
  Reducing in float32 instead is *not*, and would have moved the dynamic burn-in search silently.
- Standardizing directly in float32 rounds **twice**, after the subtract and after the divide, and
  moves the matrix by ~1 ulp. `tests/test_termination.py` pins the one-rounding equality and carries
  the twice-rounded form as an explicit control.

The block size is `mimcs.config.chunk_bytes` (doc 15), the same budget the evidence, summary and
metric-fit row passes use.

#### Chunking the loss itself: measured worse, not done

By analogy with the metric fit, the loss looked like a candidate for `_chunked.sum_rows`. It is not.
`logistic_loss` is a single GEMV, so every AD intermediate is `(n,)` rather than `(n, p)` — XLA puts
the gradient's temporary working set at **0.03 MB against a 250 MB design matrix** — and the
`(n, p)` bytes are `X` itself, a primal input that chunking the loss does not eliminate. A
like-for-like `sum_rows` control is worse on both axes: +15% memory and +72% time, to save ~170 KB.
The structural difference from `fit_metric_expr` is that its per-row intermediate was
`(block_dim,)`; here it is one scalar.

### Why not a blocked validation split

The road not taken, since the reasoning for it was good and it still lost. Holding validation out
as a contiguous tail of each half does exactly what it promised — its null came out at 0.478–0.516,
centred on 0.5 and flat in `p`, against interleaved's 0.512–0.524. On calibration alone it wins.

It loses on everything else. On the funnel it reads 0.581 (sd **0.264**), 0.525 (0.117), 0.530
(0.098), against interleaved's 0.644 (0.086), 0.605 (0.056), 0.573 (0.029): **less signal and three
times the variance**. A contiguous tail is a single excursion, so it is a tiny effective sample
however many rows it contains — the same autocorrelation that inflates the interleaved null is what
makes a blocked tail uninformative. At 3000 draws blocked cannot separate the funnel (0.530) from
its own null (~0.50) at all, while interleaved reads 0.573.

And the calibration it buys turned out not to be needed, because the ridge flattens interleaved's
null anyway. Removed; `val_every` remains.

### Shipping defaults

`rule="threshold"`, `accuracy_threshold=0.55`, `patience=3`.

0.55 sits above every measured null (≤0.524) and below the funnel at every length (≥0.573). It is
about 1 sd from each, which is why `patience` exists: a single check is a noisy thing to bet on
and checks repeat every 100 draws, so betting on one would eventually stop a chain that merely got
lucky. Consecutive checks share almost all of their draws, so requiring 3 buys less than
independence would suggest — it costs a few hundred iterations and removes the single-unlucky-check
failure.

`rule="permutation"` remains for a model where the fixed threshold is suspect; the study says the
common cases do not need it.

### What the two criteria do in practice

With those defaults, `nuts(terminate=...)` and `warmup()`:

| problem | classifier | R̂ |
|---|---|---|
| `correlated_gaussian` (easy) | 700 | 700 |
| `neal_funnel` (hard) | **3300** | **1700** |

Easy targets stop at 700 either way — `min_warmup` (500) plus `patience` × `check_every`, i.e. the
floor, which is what `patience` costs. On the funnel the classifier warms up nearly twice as long,
which is the reason to prefer it: the extra 1600 iterations are the ones R̂'s 0.65-sd tolerance
gives away.

> **The funnel's 3300 is one seed, and unrepresentative.** Measured later across 8 seeds
> (`tests/experiments/burnin_selection.py`), the classifier's funnel stopping point is 3300, 2600,
> 1900, 1100, *never*, 1500, 2400, 3700 — median **2400**, and one seed that does not fire inside
> 4000 at all. The qualitative claim above survives (the classifier keeps warming up where R̂ stops)
> but the single number does not: seed-to-seed spread on this cell is over 3×, which is the usual
> lesson about single-seed MCMC numbers appearing in a table.

## Dynamic burn-in: measured, and rejected

The termination criteria above ask "have the draws stopped changing?". A natural extension asks the
sharper question "*which prefix*, once discarded, makes the rest look stationary?" — a burn-in
estimator, which would let warmup stop earlier and discard less. Three successive objectives were
built and measured over eight seeds on five targets. The full record, including the two synthetic
targets that could not decide it, is in `tests/experiments/writeups/dynamic_burnin.md`.

**It does not survive contact with the two real targets.** On `poissonrandom` at `mu* = 10` — the
motivating case, a cold start where warmup is supposed to be mostly burn-in — there was no benefit.
On `kilpisjarvi`, a regression on uncentered years whose intercept and slope correlate at −0.99999,
there was active harm: warmup fell 3×, and the draws paid for it with **558 divergences against 2**,
on six of eight seeds.

So `burn_in="fixed"` stays the default and the dynamic modes stay off. The machinery is kept
because the *diagnostic* — `warmup_burn_in_estimates()` and the tail-vs-null curve — is informative
even where the rule built on it is not.

The reason is worth stating plainly, because it is not a defect in any of the three searches. A
burn-in estimator asks which prefix to discard, but on these problems the thing that has to
converge is the **adaptation** (the mass), not the *location* of the draws. Discarding the prefix
removes the evidence that the adaptation is still moving while doing nothing to advance it, so the
criterion is satisfied earlier by a sampler that is no better. That is why kilpisjarvi gets worse
rather than merely cheaper, and it is a reason to doubt the framing rather than to try a fourth
objective.

## The feature buffer is released when warmup ends

The buffer is one `model.features` row per retained draw. It grows for the whole of warmup and is
then dead weight for the whole of sampling, which is usually the longer phase — `rows ×
n_features × 4` bytes, plus the dynamic burn-in search's standardized `float64` copy of the same
shape again. `_WarmupTermination._term_free_features` drops both; `ClassifierTermination` extends
it cooperatively for the burn-in copy. `keep_features=True` opts out.

The distinction that matters is **warmup ending** versus **a `warmup(n)` call returning**. The
first is terminal — the criterion fired, or `max_warmup` ran out — and both release the buffer,
because saving it is opt-in for any circumstance at all. The second is not: `warmup(500)` twice is
a supported way to continue, and the second call's checks read the history the first accumulated,
so freeing there would silently restart the criterion from an empty buffer and make the chain look
*less* converged than it is. Only the raw observations go; what the criterion reported
(`warmup_mixing_stats`, `warmup_burn_in_estimates`, `warmup_terminated_early`) is kept, as is the
burn-in search's small record of what it last chose.

## Interaction with the sampler

`BaseSampler.should_stop()` is the cooperative chain the mixins extend (terminal: `False`), and
`warmup(n=None)` consults it once per iteration. `n` becomes an upper bound; `n=None` means "run
to the criterion or the mixin's `max_warmup`". With no termination mixin `should_stop()` is always
false, so `warmup(n)` runs exactly `n` iterations and consumes the identical RNG stream — the
change is inert unless opted into.

The mixin buffers `state.sample` and converts to features in batches at each check (one vmapped
call per check, not a JAX dispatch per iteration). Features are what it retains: `n × p × 4` bytes,
with `feature_thin` for models where that matters. Warmup draws are not otherwise stored anywhere.

MRO: termination goes *first* (outermost). It only observes, and `state.sample` is invariant to
the recharting mixins, so the order is not load-bearing — first is simply where a diagnostic
belongs.

## Not built

- **Geweke's test** proper (spectral-density-corrected means). The classifier is its multivariate
  cousin; a Geweke mixin would slot into the same `should_stop()` chain.
- **Rank-normalized and folded R̂** (Vehtari et al. 2021). The `x²` feature covers the folded
  variant's motivating case; rank normalization would add heavy-tail robustness.
- **Multi-chain R̂.** Deliberate: it would require running chains in parallel.
- **Nonlinear classifiers.** The reason the classifier framing was chosen over Geweke's, but a
  linear rule is the right first step and the null is hard enough already.
- ~~A target that genuinely repays a long warmup.~~ **Found: `kilpisjarvi`** (above), and it
  settled the question against dynamic burn-in. Keep it in mind for any future change that
  shortens adaptation — it is the cheapest target in the suite that punishes one, and it punishes
  it in the divergence count rather than in ESS or R̂.
- **A per-candidate permutation null**, if `"scaled"`'s single reference fit ever proves too crude.
  Built (`null="permutation"`) but only spot-checked; it costs `n_perm` refits per rung.
- **R̂ opting into the burn-in estimate.** `_WarmupTermination._term_burn_count` is the seam;
  `GelmanRubinTermination` currently always takes the fixed fraction. Pointless until a dynamic
  mode is trustworthy.
- **Conformal burn-in.** Score each draw's nonconformity against the tail and cut at the last
  index whose p-value is below α. Calibration is free under exchangeability — which is exactly
  what autocorrelation breaks, so it would need block-conformal p-values.
