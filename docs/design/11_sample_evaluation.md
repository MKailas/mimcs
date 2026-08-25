# Sample Evaluation: `sampler.summary()`

## Motivation

Warmup termination (`10_warmup_termination.md`) decides when to stop adapting. This decides,
after sampling, whether to **accept the draws** — or hand them back to `analyze` for another
round. It is a sibling of the `Evidence`/`analyze` pipeline, which flows the *other* way (into
sampler construction); `summary()` only reports.

`sampler.summary()` returns a `Summary` (cached on the sampler, and it pretty-prints) with two
tables. A **posterior summary** per ambient coordinate — mean, its MCSE, sd, and a 5/50/95%
credible interval — the numbers a data analysis reports. And a **diagnostics** table per feature.

## Per feature, not per parameter

Stan and most packages report ESS and R̂ "per parameter". They are really **per feature** — per
*observable*, a fixed function of a draw. The distinction is the same coordinate → sample →
feature stack the warmup work introduced (`Model.features`, default `[x, x²]`): a diagnostic reads
observables, and the second moment `x²` mixes differently from the mean `x` (in a funnel the scale
lags), so reporting both is genuinely more informative than one row per parameter.

The descriptive stats stay per **coordinate** — quantiles of `x²` are the odd ones — so the two
tables have different row sets on purpose: `ambient_names` for the posterior table, `feature_names`
for the diagnostics.

## The blind spot ESS and R̂ share

Both ask only whether the draws look like a well-mixed sample from *some* distribution. Neither
looks at the **target**. A sample biased with respect to the actual posterior — wrong mean, wrong
tail — but internally consistent passes both. Demonstrated: 12000 i.i.d. draws from `N(0.3, 1)`
scored against an `N(0, 1)` target have ESS/n ≈ 0.96 and R̂ = 1.000, yet are badly biased.

## The Stein diagnostic

The fix is **target-aware**: a diagnostic that can only be satisfied by the actual target.

### The Langevin–Stein identity

For a target π and a test function `g` paired with direction `j`, integration by parts gives

    E_π[∂_j g + g ∂_j log π] = 0

when the boundary term vanishes. It uses only the **score** `∇ log π`, so it needs no normalizing
constant — a real advantage, and it means the diagnostic works even for a gradient-free sampler
(the score comes from the model). Each feature `φ = f(x_j)` yields one mean-zero function,
`A_j φ = f'(x_j) + f(x_j) s_j`:

    x_j  → 1 + x_j s_j        (the classic E[x_j ∂_j logπ] = −1)
    x_j² → 2 x_j + x_j² s_j

The sample average `ẑ_k` estimates zero; `z_k = ẑ_k / MCSE_k → N(0,1)` under the null (the MCSE
is autocorrelation-aware, so this is honest for a Markov chain). A large `|z_k|` is evidence of
bias that ESS and R̂ cannot see. On the biased sample above, `z = −6.5` (mean) and `−11` (second
moment) — flagged decisively, while the mixing diagnostics passed it.

### Why not a kernelized Stein discrepancy

A KSD would aggregate all this into one number via a kernel Gram matrix — and kernels are
unstable in high dimension. This is the deliberately basic version: per-feature scalar identities,
no kernel. Nor is there an aggregate Stein test, which would need to invert the `m × m` covariance
of the Stein terms — the same high-dimensional instability. Per-feature z's only.

### On the sphere

The flat formula is wrong on a manifold: the sphere is curved and *closed*, so the right operator
is the Langevin generator `L h = Δ_S h + ⟨∇_S h, ∇_S log p⟩`, whose integral over the (boundary-
free) sphere vanishes. In closed form, per unit vector with `P = I − xxᵀ`:

    L x_i  = −(d−1) x_i + (P g)_i
    L x_i² = 2 − 2d x_i² + 2 x_i (P g)_i

using `Δ_S(x_i^k|_S) = Δ_{R^d}(x_i^k) − k(k+d−2) x_i^k` and `∇_S log p = P ∇ log p`. The projection
`(P g)_i = g_i − x_i⟨x, g⟩` discards the normal component of the ambient score, so no extension
choice matters. Verified numerically: `E_vMF[·] = 0` for both terms. `UnitVectorParameter` drops
`x_d²` (as its features do — `Σ x_j² = 1` makes it collinear), so the Stein terms match the
feature list.

### The bounded boundary caveat

For a bounded parameter the flat formula applies to the constrained value, valid **iff**
`f(x)π(x) → 0` at the bounds. Often true, not always. Where it fails — e.g. Uniform(0,1), where
the score is 0 so `1 + x·0 = 1` is a constant — the Stein series has ~zero variance and a non-zero
mean. The summary detects that (tiny MCSE, non-zero offset) and prints `boundary?` rather than an
infinite z, and documents the limitation. It never auto-rejects.

## The score, reused not recomputed

The Stein terms need the **ambient** score `∇_sample log π`. The sampler saves the
**coordinate**-space gradient `s_coord = ∇_q[log π(x(q)) + log|J|]` (Jacobian included) — a
different quantity. But the two are related by the chain rule, and the expensive model gradient is
already inside `s_coord`:

    s_coord = Jᵀ s_ambient + ∇_q log|J|,  J = ∂x/∂q   ⟹   s_ambient = (Jᵀ)⁻¹(s_coord − ∇_q log|J|)

Both corrections are **model-free**: `∇_q log|J|` is the gradient of the parameter's own
`log_jacobian_det`, and `(Jᵀ)⁻¹(·)` is a vector–Jacobian product through `sample_to_coordinate`
(the chart maps only, differentiated by autodiff). So `Model.ambient_score` recovers the ambient
score from the *already computed* coordinate scores without re-differentiating the model — the
same reuse of saved scores that `analyze` relies on. As a VJP it never forms a dense Jacobian
(`O(chart)` per draw).

One mechanism covers everything:
- **Euclidean (uncentered): free.** `J = I`, `log|J| = 0` — the pullback is a passthrough, the
  saved gradient *is* the ambient score.
- **Parents:** the VJP differentiates the real `sample_to_coordinate`, so dynamic-bound cross
  terms are captured.
- **Sphere:** `J` is non-square, so `(Jᵀ)⁻¹` does not exist and the VJP returns a score with a
  spurious normal part — but `stein_terms` projects, and `JᵀP = Jᵀ` makes the projected result the
  correct tangential score. No sphere-specific pullback.

When no gradients were saved (gradient-free sampler, or `save_gradients=False`), the score is
recomputed directly, `jax.grad(model.log_prob)`. This is also the test oracle: pullback == recompute
across every chart type.

## Multiplicity

With `m` features, ~`0.05 m` of the z's exceed 1.96 by chance. The summary shows a per-feature 95%
flag and one line — "k of m features flagged (~0.05 m expected)". **No automated accept/reject**:
the summary informs the user's decision, it does not make it.

## Layout

`mimcs/diagnostics.py` is the runtime home for the numeric primitives (ESS, MCSE, autocorrelation,
split-R̂), moved out of `mimcs/testing/` (not a runtime dependency); everything imports it
directly.
`mimcs/summary.py` holds `summarize(model, draws, ...) → Summary` (a pure function of model +
draws) and the `Summary` dataclass. `BaseSampler.summary()` wires them together, passing the saved
gradients and frozen chart, and the model it evaluates against is `summary_model` rather than
`self.model` — the same by default, but a sampler that *narrows* what it retains overrides it so
the draws, the scores and the model stay the same width (parallel tempering keeps the cold chain
of a K-fold product; doc 13). The per-parameter `stein_terms`/`ambient_names` live beside `features`
in each parameter type's module under `mimcs/model/`; the score pullback and
`stein_terms`/`ambient_names` aggregation on `mimcs/model/model.py`.

## Not built

- Kernelized Stein / aggregate Stein test (high-dimensional instability, deliberately).
- Automated accept/reject (a user decision).
- Higher-order or non-`[x, x²]` Stein features (the feature layer already allows a parameter to
  declare more; the operator follows whatever `features` returns).
