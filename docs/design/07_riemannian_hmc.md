# Riemannian Manifold HMC

## Motivation

Basic HMC and NUTS use a **global** mass matrix `M` (constant over the state space).
Where the target's local geometry varies — the funnel's neck, a stiff banana's tail,
hierarchical scale parameters — no single `M` fits everywhere, and the sampler either
under-explores or diverges (we saw exactly this for NUTS on the deep funnel, doc 06).
Riemannian Manifold HMC (RMHMC) replaces the constant mass with a **position-dependent**
one, `G(q)`, so the kinetic geometry adapts locally.

This is orthogonal to the coordinate-chart / manifold-parameter machinery of doc 04:
the charts define the unconstrained sampling coordinates `q`; `G(q)` is the *kinetic
metric* on those coordinates. Both can be used together. RMHMC changes only the kinetic
energy and the integrator — the potentials (model log-densities + Jacobian, doc 06) are
untouched.

We will implement **two variants**, as separate kinetic energies sharing the machinery
that represents `G(q)`:

1. **Implicit RMHMC** (Girolami & Calderhead 2011): a fully general `G(q)`, requiring an implicit
   (generalized leapfrog) integrator — as a **block** kinetic over any (fused) set of coordinates,
   with a given metric or the clamped Hessian of the target, factory-reachable as
   `BlockSpec(kind="riemannian")`. Costly per step, so meant for low-dimensional blocks such as a
   model's hyperparameters.
2. **Explicit block RMHMC** (the explicit integrator noted by Kleppe; used adaptively by
   Kailas, Vihola & Wallin 2026): a *block-diagonal* `G(q)` where each block's mass
   matrix depends only on *other* blocks' positions. This admits a **closed-form explicit
   leapfrog**, so it is cheap and composes directly with NUTS.

### References

- M. Girolami and B. Calderhead, *Riemann manifold Langevin and Hamiltonian Monte Carlo
  methods*, JRSS B 73(2):123–214, 2011.
  <https://academic.oup.com/jrsssb/article-abstract/73/2/123/7034367>
- T. S. Kleppe, *Dynamically rescaled Hamiltonian Monte Carlo for Bayesian hierarchical
  models*, JCGS 28(3), 2019 (arXiv:1806.02068); and *Log-density gradient covariance and
  automatic metric tensors for Riemann manifold Monte Carlo methods*, 2024.
- M. Kailas, M. Vihola, J. Wallin, *Hierarchical Riemannian manifold Hamiltonian Monte
  Carlo algorithms*, arXiv:2604.09832, 2026.

## The RMHMC Hamiltonian

With a position-dependent metric `G(q)` (symmetric positive definite), the augmented
target is `π(q, p) ∝ π(q) · N(p; 0, G(q))`, giving the Hamiltonian

$$H(q, p) = V(q) \;+\; \tfrac12\, p^\top G(q)^{-1} p \;+\; \tfrac12 \log|G(q)|,$$

where `V(q) = -log π(q)` is the usual potential (our `ModelPotential`s + `JacobianPotential`).
The kinetic energy

$$T(q, p) = \tfrac12\, p^\top G(q)^{-1} p + \tfrac12 \log|G(q)|$$

depends on **both** `q` and `p`, so `H` is **non-separable**. Two consequences:

- **Momentum refresh is position-dependent**: `p ~ N(0, G(q))` at the current `q`. (For a
  constant metric this reduces to the usual `N(0, M)`.)
- **The U-turn velocity** is `∇_p T = G(q)^{-1} p` — already what `kinetic.velocity`
  returns, so NUTS's generalized U-turn works unchanged for any metric.

### Gradients via autodiff (a deliberate simplification)

The Girolami–Calderhead paper derives `∂H/∂q_i` analytically:

$$\frac{\partial H}{\partial q_i} = \frac{\partial V}{\partial q_i}
   + \tfrac12 \operatorname{tr}\!\big(G^{-1}\partial_i G\big)
   - \tfrac12\, p^\top G^{-1}(\partial_i G) G^{-1} p,$$

involving the metric derivatives `∂_i G = ∂G/∂q_i`. **We do not hand-derive these.** `T(q, p)`
is a scalar function of `q` (given `p`), so `jax.grad(T, argnums=q)` returns `∂_q T`
including all trace and quadratic terms automatically. This removes the single biggest
implementation burden of classical RMHMC and lets arbitrary metrics be plugged in.

## Variant 1: Implicit RMHMC (Girolami & Calderhead)

For a fully general `G(q)`, the flow of `T(q, p)` is not explicit (it moves both `q` and
`p`). The standard integrator is the **generalized (implicit) leapfrog** — a symmetric,
symplectic Störmer–Verlet whose first two half-steps are implicit. Note which parts: not only
the drift — the leading momentum half-kick is implicit in `p` too, because `∇_q T` depends on
`p`. What is explicit is the potential `V`:

1. **Implicit momentum half-kick** (solve for `p'`):
   $$p' = p - \tfrac{\epsilon}{2}\,\nabla_q H(q, p').$$
2. **Implicit position drift** (solve for `q'`):
   $$q' = q + \tfrac{\epsilon}{2}\big(\nabla_p H(q, p') + \nabla_p H(q', p')\big).$$
3. **Explicit momentum half-kick**:
   $$p'' = p' - \tfrac{\epsilon}{2}\,\nabla_q H(q', p').$$

Each implicit step is solved by **fixed-point iteration to a tolerance** (below).
`∇_q T` and `∇_p T = G(q)^{-1} p` are obtained by autodiff.

**Correctness.** The map is reversible and volume-preserving (symplectic) *when the
fixed points are solved exactly*. The solves therefore run to a residual tolerance, and a step
whose solve does not reach it is **not a valid proposal**: the kinetic returns a NaN phase point,
whose non-finite energy HMC rejects and NUTS flags as a divergence (so the step-size adaptation
shrinks the step). Verified in x64 at `tol = 1e-13`: `JᵀΩJ = Ω` to 6e-15 and step–flip–step returns
the start to 3e-15, for a given own-block metric and for the Hessian metric over a fused,
non-contiguous block (`tests/test_riemannian_block.py`).

**No dedicated integrator — the implicit flow lives in the kinetic.** A key
simplification: `∇_q V` is *constant within the implicit solve* (V depends only on q), so
V is already explicit. Splitting it out — an explicit V half-kick, the implicit flow of
`T` alone, then an explicit V half-kick — solves the *same* fixed-point equations and is
the *same integrator* as the classical monolithic generalized leapfrog. So the implicit
work lives in **`RiemannianKinetic.flow`** (a generalized leapfrog of `T` only, using a
swappable `FixedPointSolver`), and RMHMC just uses the ordinary `leapfrog(potentials,
kinetic)` splitting — no separate `GeneralizedLeapfrog` class, fully unified with the rest
of the HMC machinery (and it works under NUTS as `RiemannianKinetic.flow` is just the
leaf integrator's drift). Verified equivalent: bit-identical to the monolithic version for
a constant metric, at the float32 floor for mild metrics. A side benefit: the split form's
implicit momentum solve starts from the V-kicked `p − ½∇V` (closer to the fixed point), so
naive Picard converges *better* on stiff metrics — fewer divergences than the monolithic
form at the same iteration count.

**Component**: `RiemannianKinetic(metric, solver, id, slices)` with `separable = False` — a
slice-aware kinetic like every other, so it sits in `BaseHMC`'s kinetics list beside quadratic and
explicit block kinetics.

### The block form

Restricted to a block `i` (its `slices`, possibly fused and non-contiguous), the kinetic is
`T_i(q, p_i) = ½ p_iᵀ G_i(q)⁻¹ p_i + ½ log|G_i(q)|`, where `G_i` may depend on **all** of `q` —
the block's own coordinates included, which is exactly what the explicit blocks (variant 2) rule
out. Its flow is the generalized leapfrog of `T_i` over the whole `(q, p)`, with `h = ε/2`:

1. implicit block kick: `p_i' = p_i − h ∇_{q_i} T_i(q, p_i')`;
2. explicit dependency kick: `p_{−i} −= h ∇_{q_{−i}} T_i(q, p_i')` — the same gradient;
3. implicit drift: `q_i'' = q_i + h [G_i(q)⁻¹ + G_i(q'')⁻¹] p_i'`, `q''` being `q` with `q_i''`;
4. explicit kick of all momenta by `−h ∇_q T_i(q'', p_i')`.

`q_{−i}` never moves (`T_i` does not depend on `p_{−i}`). When `G_i` does not depend on `q_i` both
solves are exact after one evaluation and the map is **identical** to the explicit block flow of
variant 2 — bit for bit, which is the oracle test. `½ log|G_i|` stays inside `T_i` rather than
becoming a potential: a potential's value and gradient are cached in the sampler state, and the
cache would go stale whenever the warmup adaptation moves the metric's softness.

**One linearization per kick.** Step 1 holds `q` fixed, so the kinetic linearizes the metric's
*pre-image* `M(q)` once (`jax.vjp`) and each fixed-point iteration costs one pullback of `∂T/∂M` —
the forward work is never repeated inside the solve. The pullback is over the full `q`: its block
part drives the fixed point and the rest *is* the dependency kick.

### Metrics

- `AnalyticMetric(fn: q -> G)` — the low-level whole-coordinate form.
- `CallableMetric(fn(coords) -> G)` — `coords` keyed by parameter name (coordinate-space slices,
  plus discrete labels). `G` is `(k, k)` SPD (Cholesky) or `(k,)` positive (log space). What
  `BlockSpec(kind="riemannian", params={"metric": fn})` builds.
- `HessianMetric` — the default when no metric is given. `H = ∇²_{bb} V(q)` is the block Hessian of
  the coordinate-space target (Jacobian potential included), `k` Hessian-vector products; with
  `H = Q Λ Qᵀ` the metric is `G = Q f(Λ) Qᵀ`, `f(λ) = φ(bλ)/b`, for a clamp `φ`:
  - `softabs` (Betancourt 2013; the default): `φ(x) = x coth x`. A negative curvature keeps
    `≈ |λ|`.
  - `softplus`: `φ(x) = log(1 + eˣ)`. A negative curvature `−|λ|` keeps mass `≈ e^{−b|λ|}/b` —
    positive but small, so the direction moves fast (and the implicit solve is stiffest there).

  **On an indefinite target the clamp matters, and softplus is the worse one.** With `1/b`
  adapted to the *positive* curvatures (~0.14 on Rosenbrock, `b = 5`), softplus leaves a curvature
  of −0.8 a mass of ~0.006: a near-singular direction whose implicit step has no solution at almost
  any step size. Whole-space Hessian block, 4 seeds: softplus failed 47–69% of transitions with the
  step adapted down to 0.03–0.14; SoftAbs 30–40% at 0.23–0.36 (lower on every paired seed); both
  sampled the banana's moments correctly. Even SoftAbs loses a third of its transitions there — the
  metric turns fast along a banana — so a whole-space Hessian metric is not an efficient kinetic
  for one. Where the block curvature stays positive (a funnel's `v`), the clamp is inert and the
  choice does not matter. Centered eight schools (below) agreed: SoftAbs 1.4× the ESS/s of softplus
  with fewer failures. The default was softplus until that A/B and is now SoftAbs.

**What a Hessian hyperparameter block buys** (`tests/experiments/writeups/implicit_rmhmc_blocks.md`,
x64, 8 seeds per arm, medians). On *centered eight schools* the factory default (no evidence) is
biased on every seed — `log τ` at z = 6.45 [3.5, 11.9], sd 0.72 against the quadrature 1.17 — while
diverging on only 0.5% of transitions. A fused `(μ, τ)` Hessian block with `θ` on the learned
`Exp("tau")` metric is unbiased (z −0.1, sd 1.15) with `log τ` ESS 110 [66, 194]; the same `θ`
metric with a *constant* `(μ, τ)` mass is also unbiased at ~10× the Hessian arm's ESS per second
(52 vs 5.8), but erratic (`log τ` ESS 7–431, 3–32% divergences), and the Hessian arm fails 27–30% of
its implicit steps. On a *10-d funnel* (9 children of `v`) the Hessian `v` block buys nothing: the
conditional curvature `½Σx²e^{−v}` is nearly constant there, so a constant `v` mass is right, and the
Hessian arm runs at 0.14× its ESS per second. On a *2-d funnel* (one child, a wildly varying
curvature) it cut divergences ~8× (10.2% → 1.2%, paired ratio ≥3.3 on each of 8 seeds). So: the implicit block pays where a
hyperparameter's conditional curvature varies strongly, at 18–42× the gradients per iteration.

  Both are evaluated through `log φ`, which is finite across the whole float32 range (softplus's
  `φ` itself underflows to 0 below `x ≈ −88`). The derivative through the eigendecomposition is a
  `custom_jvp` (Daleckii–Krein divided differences, `mimcs/hmc/spectral.py`): naive autodiff
  through `eigh` is NaN at a repeated eigenvalue and — measured — *finite but wrong* on an
  indefinite matrix with one (−11.769 against the true −11.791; −2.85 against −1.47), while the
  custom rule matches central finite differences to ~1e-10 in every case.

**Softness adaptation** (`HessianSoftnessAdaptation`). `1/b` is a curvature scale: curvatures well
above it pass through, those around or below it are reshaped. It should sit below the real positive
curvatures without starving the negative ones, so warmup tracks — by a log-scale stochastic
approximation, the running quantile the learned-metric clip also uses — the 10% quantile `q` of the
positive block eigenvalues the chain visits, and sets `1/b = q / 3` (`φ(3)/3 = 1.016`: 90% of the
positive curvatures distorted by under 2%). On a 4-d funnel's `v` block the adapted `1/b` came
within 0.85–1.24 of the quantile of 4000 sampling draws' curvatures over 8 seeds.

### Fixed-point solvers

`PicardSolver` and `AndersonSolver` (default, depth 3) iterate **to a tolerance** in a
`lax.while_loop`: stop at `norm(g(x) − x) ≤ tol` or `max_iter` (30) evaluations, and report
convergence. The kinetic measures residuals in the metric's own units — a momentum residual
`G^{−1/2}`-whitened, a position residual `G^{1/2}`-whitened — so one absolute `tol` means the same
on any scale; the default is `√eps` of the working float type (3.5e-4 float32, 1.5e-8 x64).
Anderson's ridge is relative to `tr(RRᵀ)` (an absolute one swamps the normal equations near
convergence), and its history resets when a residual grows 10× — resetting on *any* growth was
measured to cost convergence (one failure in 40 random stiff 6-d contractions; none at 10×). On
those contractions Anderson's median evaluation count is 13 against Picard's 43.5.

**Cost accounting.** The kinetic adds its model work to `grad_evals`, in gradient equivalents: an
HVP inside the block Hessian counts `1.0` and a pullback through one `1.3` — the top of the ranges
measured on hierarchical logistic regressions (0.58–1.00 and 0.63–1.32, N = 200..20000;
`tests/experiments/rmhmc_cost_ratios.py`), so the count is never flattering. The sampler's own
energy and U-turn evaluations of the kinetic are not in it. `fp_iters` / `fp_failures` ride in
`integrator_data` into `state.diagnostics`; `sampler.fixed_point_failure_rate()` separates solver
failure from ordinary energy-error divergence, and sampling ends with a WARNING above 1%.

## Variant 2: Explicit Block RMHMC (Kleppe; Kailas–Vihola–Wallin)

### Block structure

Partition the coordinates into blocks `q = (q_1, …, q_k)` (naturally, the blocks are the
`Model`'s parameters — see below). Use a **block-diagonal** metric

$$G(q) = \operatorname{blockdiag}\big(M_1(q_{-1}), …, M_k(q_{-k})\big),$$

with the **key constraint**: block `i`'s mass `M_i` depends only on *other* blocks'
positions `q_{-i}`, **never on `q_i` itself**. The canonical (two-block, "hierarchical")
case of Kailas–Vihola–Wallin is `q = (q_A, q_B)` with `M_A` constant and `M_B(q_A)` — e.g.
the funnel's `q_A = v` (scale) and `q_B = x`, where `M_B(v)` learns the `exp(v)` scaling.

The kinetic energy is the sum of per-block terms

$$T(q, p) = \sum_i T_i,\qquad
  T_i = \tfrac12\, p_i^\top M_i(q_{-i})^{-1} p_i + \tfrac12 \log|M_i(q_{-i})|.$$

### Why it is explicit

The flow of one block term `T_i` holds `q_{-i}` and `p_i` fixed (because
`∂T_i/∂p_j = 0` for `j ≠ i` and `∂T_i/∂q_i = 0`). With those fixed, `M_i(q_{-i})` is
constant, so the remaining updates are linear and **closed-form**:

- **drift** `q_i ← q_i + ε · M_i(q_{-i})^{-1} p_i` (constant velocity), and
- **kick** the *other* momenta `p_j ← p_j + ε · (-∂_{q_j} T_i)` for the blocks `j` that
  `M_i` depends on (constant force, including the `log|M_i|` term via autodiff).

This combined drift+kick is a volume-preserving shear and is exactly reversible. It is a
single **elementary flow** in our framework. The full integrator is then a *palindromic
splitting* of `V` and the `T_i`, built from our existing `SplittingIntegrator` — **no new
implicit machinery, no fixed-point solves**. For example, two blocks with `M_A` constant:

```
SplittingIntegrator([
    Op(V,    0.5),     # kick all momenta by the potential
    Op(T_A,  0.5),     # drift q_A   (M_A constant: pure drift)
    Op(T_B,  1.0),     # drift q_B + kick p_A   (M_B(q_A) constant during this flow)
    Op(T_A,  0.5),
    Op(V,    0.5),
])
```

(The Kailas–Vihola–Wallin Algorithm 4 uses a closely related symmetric composition that
averages `M_B` at the old and new `q_A`; the Strang splitting above evaluates `M_B` at the
single current `q_A`. Both are second-order, reversible, volume-preserving explicit
integrators; we adopt the splitting form because it reuses `SplittingIntegrator` directly.
Exact composition ordering for ≥3 blocks / cyclic dependencies is an implementation
detail to pin down — see Open Questions.)

**Component**: one `BlockRiemannianKinetic(block_index, metric, depends_on)` per block,
each implementing the combined explicit `flow`. Because the integrator is explicit, this
variant plugs **directly into NUTS** (the headline advantage in Kailas–Vihola–Wallin).

## Shared Machinery: Representing `G(q)`

Both variants need a position-dependent SPD matrix. A `Metric` abstraction serves both
(one full `G(q)` for the implicit variant; one `M_i(q_{-i})` per block for the explicit
one):

```python
class Metric:
    def init_params(self): ...                  # learnable parameters φ (None if fixed)
    def matrix(self, q_inputs, params) -> Array: ...   # SPD matrix
    # the kinetic energy and ALL its q-derivatives are obtained by autodiff of
    #   T(q_block, p_block) = 0.5 p^T matrix(q_inputs)^{-1} p + 0.5 logdet(matrix(q_inputs))
```

SPD-ness is guaranteed by construction (diagonal `exp(...)`, or a Cholesky / log-Cholesky
parametrization for dense metrics). Two implementations:

### Analytic metric (for testing)

`AnalyticMetric(fn)` wraps a user-supplied `q_inputs -> SPD matrix` (e.g. the Fisher
information, derived by hand as in Girolami–Calderhead). Derivatives come from autodiff of
`fn`. This is how we validate the samplers against a known metric — and, with the implicit
variant, against the reference RMHMC algorithm.

### Learned metric (the practical default)

Following Kailas–Vihola–Wallin, fit a parametric model for the **conditional covariance of
the score**. For block `i`, let `g_i = ∇_{q_i} log π(q)` (the block's gradient — already
computed for the leapfrog and cached in `potential_grads`). Minimize, over the metric
parameters `φ`, the expected loss

$$\ell(\varphi;\, q_{-i}, g_i) = \log\big|M_i(q_{-i};\varphi)\big| + g_i^\top M_i(q_{-i};\varphi)^{-1} g_i,$$

whose minimizer sets `M_i(q_{-i}) = E[g_i g_i^\top \mid q_{-i}]`, the conditional second
moment of the score (a Fisher-information-like quantity). This is exactly the KL-divergence
objective between the joint of `(q_{-i}, g_i)` and the parametric model.

**Diagonal model** (used in the experiments): per coordinate, `M_{i}(q_{-i}; φ) =
exp(φ^\top x(q_{-i}))` (or a sum of exponentials), with feature vector `x(q_{-i})`. The
loss gradient has the clean form

$$\nabla_{\varphi}\,\ell = \big(1 - g_{i}^2 / M_{i}\big)\,\nabla_{\varphi}\log M_{i}.$$

**Adaptation** is Markovian stochastic approximation during warmup (an adaptation mixin,
doc 02): each step does an SGD update `φ ← φ − η_n ∇ℓ(φ; q_{-i}^{(n)}, g_i^{(n)})` using
the score from the current draw, stabilized by a **variation of** the paper's Algorithm 5 — a
running mean of the gradient, centering, and clipping at an adaptively learned quantile — to
prevent early large gradients from corrupting the metric. It is a variation rather than the
algorithm as published: the clip here is **per target coordinate** rather than per block, and
gradient-mean centering is **off by default** for a learned (conditional) block, for the reasons
given under "Learned metric" below.
The learned `φ` lives in `state.ham_params` (the reparameterization principle), updated in
`postprocess`, read by the kinetic in the kernel.

**Conditioning on discrete parameters.** A model with integer parameters conditions on those too:
the fitted quantity is `M_i(q_{-i}, z) = E[g_i g_iᵀ | q_{-i}, z]`, not its average over `z`. That
matters because labels change a mode's *scale* — under a spike-and-slab indicator `beta_j` has
scale `tau` or near zero, and one mass is wrong under both. A label enters the metric exactly as
it enters the density: as a **trajectory constant**, carried in `ctx.discrete`. Two consequences
fall out rather than being arranged. The metric-derivative kick is unaffected, because
`M = f(z)·g(q)` has `dM/dq = f(z)·dg/dq` — the discrete factor *scales* the kick rather than
contributing one; and a metric depending only on labels is constant along a trajectory, so it
needs no kick at all (`depends` keys off the *continuous* dependencies alone).

A discrete dependency is declared, not inferred: `Exp("sigma", categorical=["z"], ordinal=["k"])`.
The distinction only bites above two values — for a binary label the reference indicator is an
affine function of the standardized value, so the two codings span the same models. Encoding
happens *outside* the expression language (`mimcs/hmc/metric_encode.py`), which is why none of the
atoms' arithmetic changed: an atom sees a real feature vector either way.

## Integration With the Existing Architecture

- **Potentials / charts unchanged.** RMHMC only swaps the kinetic + integrator; the
  potentials (model + Jacobian) and the chart framework are untouched. RMHMC and
  constrained parameters compose.
- **`ham_params`** holds the metric parameters `φ` (learned) exactly as it holds mass
  matrices today; mass adaptation becomes metric adaptation.
- **NUTS** works for both variants via `velocity = G(q)^{-1}p`; the explicit variant is
  the one to pair with NUTS in practice.
- **Momentum refresh needs `q`.** `sample_momentum` must see the current position to draw
  `p ~ N(0, G(q))` (block-wise `p_i ~ N(0, M_i(q_{-i}))`). This requires a small interface
  change — pass the position (or the seed `IntegratorState`) to `sample_momentum`; existing
  diagonal/dense kinetics ignore it.
- **Testing strategy** mirrors NUTS/SimpleNUTS: the implicit variant with an analytic
  metric is the trustworthy reference; the explicit variant and the learned metric are
  validated against it (and against analytic-reference targets like the funnel via the
  existing `evaluate` harness). The funnel becomes the showcase where RMHMC should *fix*
  the neck that global-metric NUTS cannot.

## Implementation Status

**Both variants are complete and factory-reachable.** Both were merged to `main` on 2026-06-25 with
given and learned diagonal metrics, in fixed-length and NUTS forms. Variant 2 (explicit) is the
cheap, scalable one and what the factory's rules select. Variant 1 (implicit) was a whole-space
reference until 2026-09, when it became a block kinetic with a clamped-Hessian default metric,
tolerance-checked solves and a `BlockSpec(kind="riemannian")` (see "The block form" above); no rule
selects it yet. Variant 1 handles a general `G_i(q)` — dependent on the block's own coordinates —
via the generalized leapfrog; variant 2 handles block-diagonal `M_i(q_{-i})` in closed form.

A word on what "diagonal" does and does not mean here, since the two senses are easy to conflate.
`G(q)` is block-diagonal *across* blocks throughout — that is the constraint that buys the explicit
integrator. *Within* a block the metric need not be diagonal: `ShapedLearnedBlock` (below) is
`M(x) = D(x)^{1/2} A D(x)^{1/2}` with a dense or low-rank constant shape `A`, and it is shipped and
factory-selectable. What stays diagonal is the **position-dependent part** `D(x)`. A block whose
dependence on `q_{-i}` is itself dense — a fully position-varying `M_i(q_{-i})` — is the piece still
deferred (Open Question 5).

**Variant 1 (implicit RMHMC) is implemented** (`mimcs/hmc/riemannian.py`): `Metric` /
`AnalyticMetric` / `CallableMetric` / `HessianMetric`, the slice-aware `RiemannianKinetic`
(non-separable; its `flow` is the block generalized leapfrog of `T_i` with a swappable
`FixedPointSolver`; `∂_q T` via autodiff, no hand-derived metric terms), and the whole-space `RMHMC`
convenience sampler, which uses the ordinary `leapfrog(potentials, kinetic)` splitting — no
dedicated integrator. Validated: with a
constant metric it matches the analytic standard-leapfrog map to the float32 floor; RMHMC
samples a Gaussian and a `(1+x0²)I` banana correctly, and samples Neal's funnel correctly
with a good metric (`diag(1, e^{-v})`) — the geometry global-metric NUTS could not handle.

Empirical note worth keeping: the conformal metric `exp(-v) I` (suggested as a test case)
under-explores the funnel's x-tail. Its `e^{-v}` *v-block* makes the large-`v` dynamics
stiff/unstable, so the chain mixes too slowly into the large-`v`/large-`x` corner (x std
plateaus well below the truth even at 50k draws) — while still sampling `v` correctly and
reaching deep into the neck. This is a vivid illustration of the implicit integrator's
sensitivity to the metric and motivates the stability work below. The natural metric
(constant `v`-block, `e^{-v}` `x`-block) does not have this problem.

**Variant 2 (explicit block RMHMC) is implemented with given input metrics**
(`mimcs/hmc/block_riemannian.py`). The metric is block-diagonal,
`G(q) = blockdiag(M_1(q_{-1}), ..., M_k(q_{-k}))`, where block `i`'s mass depends only on
*other* blocks' positions (never `q_i` itself) — the constraint that makes integration
explicit. The simplest block is **diagonal** (`DiagonalBlock`): `M_i` is a vector and its
kinetic term is `T_i = ½ Σ p_i²/M_i + ½ Σ log M_i`. (`ShapedLearnedBlock`, below, is the
nondiagonal one.) The block dependence is declared via
the `depends_on=` interface on `BlockMetric`; an `fn` maps the named dependency slices to
the diagonal vector (e.g. `BlockMetric(depends_on=("v",), fn=lambda d: jnp.exp(-d["v"]))`),
and a block with no `fn` is a constant identity metric. Blocks are composed in the
topological/declaration order of the model parameters (`q_{-i}`-dependence only, so no
cycle); order-tuning is deferred.

Each term `T_i`'s flow holds `q_{-i}` and `p_i` fixed (so `M_i` is constant along it),
drifting `q_i` linearly and kicking `p_{-i}` by a constant force — fully closed-form, **no
implicit solve**. The metric-derivative kick is taken by `jax.grad` (no hand-derived terms,
as in the implicit variant). The full step is the palindromic Strang splitting
`V_1/2…V_p/2 T_1/2…T_{k-1}/2 T_k T_{k-1}/2…T_1/2 V_p/2…V_1/2` — but this is exactly what the
**unified `leapfrog(potentials, kinetics)`** produces from the list of block kinetics (doc 06):
each block is an ordinary slice-aware `KineticHamiltonian` in `BaseHMC`'s kinetics list, and
explicit block RMHMC is just `BaseHMC` with that list (no `BlockDiagonalKinetic` composite, no
`block_leapfrog`, no `ExplicitRMHMC` sampler class). `build_block` resolves each block's
coordinate slice via `Model.coord_block` and raises if a block depends on itself. Validated (`tests/test_explicit_rmhmc.py`): constant metric matches the
analytic standard-leapfrog map to the float32 floor; all-constant explicit RMHMC = standard
HMC on a Gaussian; and the natural funnel metric (constant `v`-block, `e^{-v}` `x`-block)
samples Neal's funnel correctly — neck and mouth — much faster than the implicit variant
(no fixed-point iterations).

**The learned/adaptive diagonal metric is also implemented** (`LearnedDiagonalBlock` in
`mimcs/hmc/block_riemannian.py`; `MetricAdaptation` mixin in `mimcs/adaptation/metric.py`).
A learned metric is given by a **mass-matrix mini-language** expression
(`mimcs/hmc/metric_expr.py`, `MetricExpr`), a small algebra over the *other* blocks'
coordinates built from two atoms and two combinators:

- **Dense atoms** `Exp(*deps, features=…)` and `Sigmoid(*deps, features=…)`, each
  `link(Σ_d W_d · feat(coord_d) + b)` with `link ∈ {exp, σ}` — coordinate `j` of the block
  couples to *every* dependency coordinate through the matrix `W_d` (shape `block_dim × dep_dim`).
  Every atom carries a per-coordinate bias `b`, so a dep-less atom `Exp()` / `Sigmoid()` is the
  pure-bias case `link(b)` — a learnable per-coordinate constant (an `Exp()` term is a baseline,
  an `Exp()` factor a positive scale). `feat` is identity by default or `"quadratic"`
  (`[x, x²]`, no interactions).
- **Sparse atoms** `SpExp(*deps, features=…)` and `SpSigmoid(*deps, features=…)`, each
  `M_j = link(Σ_d W_{d,j} · feat(coord_{d,j}) + b_j)` — *elementwise*: coordinate `j` of the
  block depends only on coordinate `j` of each dependency (a bijective row correspondence, e.g. a
  horseshoe's per-element scale `λ_j` for `x_j`), with **no sum over the other dependency
  coordinates**. Each dependency must have `dep_dim == block_dim`. This is both the natural
  geometry for equal-dimension arrays and far cheaper — `2·block_dim` parameters vs the dense
  `block_dim·dep_dim + block_dim`. Implemented as `_SparseAtom` (overriding only the numeric
  methods) combined with a dense atom's link via a diamond (`SpExp(_SparseAtom, Exp)`), so no
  link code is duplicated.
- **Combinators** `+` (`Sum`) and `*` (`Product`), elementwise over the block's coordinates.

Examples: `{"x": Exp("v") + Exp()}` gives `M_x = exp(W v + b) + exp(b0)` (the previous
sum-of-exponentials); `{"y": Exp()*Sigmoid("v","x") + Exp()}` gives a gated form
`exp(b1)·σ(W v + U x + c) + exp(b2)`; `{"x": SpExp("lambda")}` gives the elementwise
`M_j = exp(W_j λ_j + b_j)`.

**Weights shared across the block's coordinates.** Every parameter above is per-coordinate: `W` has
`block_dim` rows, `b` has `block_dim` entries. A weight of shape `(block_dim, feat)` has a
broadcastable sibling of shape `(1, feat)` — one value serving every coordinate — declared on the
atom as `Exp(d, shared_weights=(0,))` / `shared_bias=(0,)`, where the tuple indexes the *block*
axes (only axis 0 exists today; the feature axis is deliberately not shareable, since a dense
atom's `W @ feat` cannot broadcast its contracted axis). This is the right model whenever the
geometry has a single cause — a funnel's `e^{-v}`, a horseshoe's `log lambda` — and it is what lets
the regression fit a 30-coordinate funnel with 2 parameters instead of 60.

`evaluate` needs no change: `(1, feat)` broadcasts in both atom families. Four things around it did,
and none of them raised — a length-1 axis broadcasts rather than failing. The block must
`broadcast_to((size,))` its mass, or a fully-shared expression's `(1,)` makes `_energy`'s
`jnp.sum(jnp.log(M))` sum one element instead of `size`: a wrong log-determinant, and since `flow`
differentiates it, wrong *dynamics* (`metric_loss` is accidentally immune — it broadcasts inside its
sum). `MetricAdaptation` must key its update scale off each *leaf's* rows, or a shared leaf is
promoted back to `(block_dim, feat)` on the first warmup step. The scale-aware init must reduce a
per-coordinate `target` in link space for a shared bias (the geometric mean of the scales), or
`zeros((1,)) + log(target)` broadcasts it back up. And `metric_expr.check_params` validates a
supplied `metric_init` against the expression, structure and shape, since that is the one place a
mismatched tree can enter.

A shared unit's gradient is **divided by the coordinates it serves**. `L(w) = sum_d l_d(w)` scales
gradient and curvature together, so a first-order step needs `eta < 2/(n h_1)` where a
per-coordinate one needs `eta < 2/h_1`, and the adaptive clip cannot absorb it (its threshold
tracks the observed norm, so both sides scale). It is a per-unit learning rate; the objective is
unchanged, which matters because the online and offline losses are deliberately the same one.

**Positivity is structural** — `exp>0`, `σ∈(0,1)`, and
sums/products of positives are positive — so every expression is a valid diagonal mass.
Dependency blocks may be fused/non-contiguous (referred to by the `x__y` name). Each node exposes `deps()`, `init_params()` (weights zero, biases set so
`M_i ≈ I` at init), `evaluate()`, and `n_params()` (for the factory's dimension-aware
candidate budget). The parameters live in `state.ham_params[kinetic.id]` as a pytree
mirroring the expression, read back through `HamiltonianContext` during integration — so the
same block machinery serves given and learned metrics.

`MetricAdaptation` adapts them online during warmup by SGD on the per-block KL objective
`L_i(φ) = ½ Σ_d (log M_i[d] + g_i[d]²/M_i[d])`, where `g_i` is the current potential
gradient (the score) restricted to block `i`. The per-sample minimiser is `M_i[d]=g_i[d]²`
and the expected-loss minimiser is the *conditional gradient second moment*
`E[g_i[d]² | q_{-i}]` — the metric that whitens the local geometry; the gradient `∂L_i/∂φ`
is taken by `jax.grad` (no hand-derived terms). Both this loss and the block's kinetic energy
are evaluated in **log space** — `log M_i` from the expression itself (`MetricExpr.log_evaluate`),
the energy as `½|p·exp(−log M/2)|² + ½Σ log M`, the loss with `g·exp(−log M/2)` — because
`g²/M` and `p²/M` differentiated by autodiff form `M⁻²`, which overflows float32 once
`M < ~5e-20`. That made the metric-derivative kick infinite at any step size on a hot PT rung of
Neal's funnel (`D` = 2.2e-20 at `v` ≈ 25) and froze every lane. The shaped block is whitened the same
way: with `u = p·exp(−log D/2)` its energy is `½uᵀA⁻¹u + ½(Σ log D + log|A|)`, so the dense/low-rank
algebra acts on `A` alone and `D` never appears in a denominator. SGD defaults: step size `(n+n₀)^{-κ}` with
`κ=0.75`, `n₀=5`; per-block gradient clipping at an adaptive threshold tracked (online
log-scale quantile) so a target fraction (default 10%) of steps are clipped. A third
Kailas–Vihola–Wallin regularizer, **gradient mean estimation** (centre the score by a running
mean `ḡ`, fitting the covariance rather than the second moment), is implemented but **off by
default** here (`metric_center_grad`): `E[g]→0` at stationarity so it is inert there, but a
single *marginal* `ḡ` distorts a *conditional* block's fit (it adds a constant floor `ḡ²` that
flattens e.g. the funnel's `e^{-v}`), so it suits only effectively marginal / constant blocks.
It is on by default in the marginal counterpart `ScoreMassAdaptation` (constant diagonal
mass), where it also helps the chain move downhill early. Adaptation runs in warmup only and
is frozen for sampling.

Centring has a known transient side effect from a **far-from-mode start**: the early scores are
enormous, `ḡ` chases them up, and because it is an RM running mean it stays inflated for ~10⁴ steps
after the chain reaches the mode, driving the fitted mass *up* precisely once the chain has arrived.
It is **benign** — a frozen mass taken from anywhere on the hump recovers the posterior at the same
ESS — because it only appears where the basin is already easy. A two-copy AIR fix was tried and
rejected (2026-07-18): every variant lost to the plain adaptation, monotonically in refreshment
frequency, because a periodic mass refresh makes the mass bumpy and destabilizes the coupled
step-size adaptation. The cure is worse than the disease; see
`tests/experiments/writeups/air_two_copy_metric.md`.

Validated (`tests/test_learned_metric.py`): on the funnel `x|v ~ N(0,e^v)` the ideal metric
is `E[g_x²|v] = e^{-v}`, and the learned `M_x = exp(W v + b)` recovers it — `W → -1`,
`b → 0` (the quadratic feature variant keeps the `v²` weight `≈ 0`); a constant `Exp()` block
recovers `diag(precision)` on a Gaussian; and the learned metric (given *no* metric, only the
expression `Exp("v")`) samples the funnel correctly, matching the given-ideal-metric case.
The mini-language itself is unit-tested in `tests/test_metric_expr.py` (init ≈ I, evaluation,
positivity, param counts, differentiability).

**Explicit RM-NUTS** is a drop-in composition (`explicit_rmnuts` builder; no new sampler
code): `NUTS` with the list of block kinetics (each block's `velocity = M_i(q_{-i})^{-1} p_i`
drives the generalized U-turn) and the unified `leapfrog` as the leaf integrator. It pairs NUTS's
automatic trajectory length with the explicit block metric — given *or* learned, with
`MetricAdaptation` composing unchanged. Validated (`tests/test_explicit_rmnuts.py`): a
constant metric reduces to ordinary NUTS on a Gaussian; the given natural metric, and a
fully learned metric (no metric supplied, no trajectory tuning), both sample the funnel
correctly. Because there is no implicit solve, explicit RM-NUTS is far cheaper than the
implicit RM-NUTS — and the learned variant is the most automatic of the family.

**Shaped (nondiagonal) learned metric.** A diagonal `D(x)` cannot capture *within-block*
correlation. `ShapedLearnedBlock` (`block_riemannian.py`) extends it to
`M(x) = D(x)^{1/2} A D(x)^{1/2}`, keeping the position-dependent part **diagonal** (`D(x)`, the
mini-language) and adding a **constant** shape `A` — either **dense** `A = K K^T` or **low-rank**
`A = I + Σ_j γ_j v_j v_jᵀ` (`γ_j ≥ 0`). Only the cheap whitening varies with position; the
parameter-heavy shape is constant, so few new parameters are added (a shaped metric already carries
many, and adaptation must stay stable). The kinetic reuses the existing algebra: dense via
triangular solves with `L(x) = diag(√D(x)) K` (as `DenseQuadraticKinetic`); low-rank via the
`mimcs.hmc.lowrank` Woodbury/`apply_chol` primitives with `Dd = D(x)`, `V[j] = √γ_j √D(x) v_j`. The
explicit block flow (drift + autodiff kick) is inherited unchanged — only `D(x)` depends on `q`, so
the metric-derivative kick generalizes for free once `_energy` is the shaped energy.

Adaptation (`ShapedMetricAdaptation`) is **decoupled and reuses both existing score adapters**:
`D(x)` by *the same code* as `MetricAdaptation` (`metric._make_kl_step` + `_PerUnitClip`: a clip
threshold per coordinate and per shared leaf, a shared leaf's gradient divided by the coordinates it
serves, non-finite units skipped, the same optional EMA), and `A` by feeding the
`D(x)^{-1/2}`-whitened block score to a dense `ScoreMassAdaptation` block (`K`) or the low-rank
low-rank tracker shared with `LowRankAdaptation`: by default the held-basis subspace iteration of
`mimcs.adaptation._subspace`, or the original Sanger/Oja tracker under `shaped_tracker="sanger"`
(its module docstring and `docs/reference/algo_kwargs.md`, *Low-rank trackers*: Sanger reads
autocorrelated warmup scores as anisotropy, which is what closed `irt_2pl`'s collapse loop).
Because `D`
whitens the diagonal, `A` is fit as a **correlation** matrix (unit diagonal) — well conditioned,
which is what keeps the shape estimate stable; a short burn-in lets `D(x)` settle first (mirroring
`LowRankAdaptation`). Gradient mean estimation is the same key here as for the diagonal metric
(`metric_center_grad`, off by default for the same reason), and one centred score feeds both `D(x)`
and `A`'s whitening — as `_LowRankBlock` centres once for its diagonal and its Sanger step. It was
read only by `MetricAdaptation` until 2026-09, so it was silently inert on shaped blocks. By default
the *raw* `D(x)` iterate drives warmup and whitens `A`, and an EMA of it is frozen for sampling
(`mass_ema` defaults on for the learned metrics only). The mass-averaging
keys every mass adaptation shares (`mass_ema`, `mass_ema_warmup`; `mimcs/adaptation/_ema.py`) act on
`D(x)` exactly as in `MetricAdaptation`, so plain and shaped blocks run the same `D(x)`: an
exponential moving average of the `D(x)` parameters with the SGD's Robbins–Monro gain, frozen for
sampling, and under `mass_ema_warmup` also driving the simulation and whitening `A`, so the shape is
fitted under the metric the chain actually runs with (and adapts more slowly). On `irt_2pl` neither
prevents the shape runaway (`tests/experiments/writeups/irt_shaped_dx_ema.md`). A position-*varying* shape `A(x)` is a deliberate non-goal for now. Settable by
hand via a `learned_metric` block with `params={"metric": …, "shape": "dense" | ("lowrank", J)}`, and
**auto-selected by the factory** (`learned_metric_rule`, `docs/design/09`): once `D(x)` is regressed,
the shape is chosen by running the *same* mass-mode selector (`mode_select.select_mass_mode`) on the
`D(x)^{-1/2}`-whitened *conditional* scores — whose constant correlation is exactly `A` — giving
`None` / `("lowrank", J)` / `"dense"`. That selector applies its **rank guard** here too, so a
block whose pilot has fewer than `0.75·d` effective rows gets no shape at all — the case that drove
a 2000-coordinate spike-and-slab block to a 1e-24 step size (doc 09). Its three gates apply too
(effective sample size of the second moments, a stiffness floor of 2, split replication; doc 09):
on `irt_2pl` they remove the `theta` shape that the MP test alone chose on 9 of 15 pilots, all of it
noise (`tests/experiments/writeups/mode_select_gates.md`). Validated
(`tests/test_shaped_metric.py`): the kinetic algebra
matches a dense reference for both shapes; on a funnel-with-correlation target (ideal metric
`e^{-v} A`) the fit recovers `D(v)` and `corr(A)` and both shapes sample it correctly; `shape=None` is
exactly the diagonal metric; and a pilot fed back to `analyze` selects a shape when the conditional
scores are correlated and declines one (`None`) when they are not.

## Design Decisions To Pin Down (Open Questions)

1. **Fixed-point solver** for the implicit integrator. ✅ *Resolved (2026-09).* A swappable
   `FixedPointSolver` (`mimcs/hmc/solvers.py`): `PicardSolver` and `AndersonSolver` (the default)
   iterate to a tolerance in metric-whitened units and report convergence; an unconverged step is
   rejected as a divergence. On the misspecified conformal funnel metric Anderson fails fewer steps
   than Picard at the same cap (means .53 vs .63 over 4 seeds), and raising the cap from 8 to 30
   lowers neither: those failures, like the funnel `v`-block's, are steps at which the implicit
   equations **have no solution** (a 1-d kick is a quadratic with no real root past a step size), not
   slow convergence — so a Newton solver would not cure them either. It remains a possible
   follow-up for convergence speed on larger blocks.
2. **Block composition for ≥3 blocks / cyclic dependence.** ✅ *Resolved (for the DAG
   case).* Blocks are composed in the topological/declaration order of the `Model`
   parameters and the palindrome mirrors it, so the integrator stays reversible; `depends_on`
   referencing only *other* blocks rules out cycles by construction. Whether a *different*
   order improves integrator efficiency is left open (no clear picture yet) — revisit later.
3. **Block partition API.** ✅ *Resolved.* One block per `Model` parameter, with an explicit
   `depends_on` per block and the hard constraint `i ∉ depends_on(i)` (raised at build time).
   The funnel: `v` block (constant mass), `x` block (`depends_on = ("v",)`).
4. **Feature map `x(q_{-i})`** for the learned diagonal metric. ✅ *Resolved.* `feat` is the
   identity by default, optionally `"quadratic"` (`[x, x²]`, no interactions); configurable
   **per atom** via `Exp("v", features="quadratic")`, so the terms of a sum may differ.
5. **Metric parametrization beyond diagonal**: dense/low-rank for correlated blocks.
   ✅ *Partly resolved.* `ShapedLearnedBlock` + `ShapedMetricAdaptation` ship a nondiagonal
   within-block metric `M(x) = D(x)^{1/2} A D(x)^{1/2}`, with `A` dense (`K K^T`) or low-rank,
   fitted from the `D`-whitened conditional scores and auto-selected by the factory. **What
   remains open** is a *position-dependent* dense metric — a shape that varies with `q_{-i}`
   rather than a constant `A`. That is what has no clear reference: a stable KL objective and
   parametrization for a full `M_i(q_{-i})`. The constant-shape decomposition was chosen partly
   to sidestep it (few new parameters, and the shape is fitted as a well-conditioned correlation
   matrix).

## Planned Classes

| Layer | Class | Role | Status |
|---|---|---|---|
| Metric | `Metric` (base) | position-dependent SPD matrix + learnable params | ✅ |
| | `AnalyticMetric` / `CallableMetric` | a given `q -> G` (flat coordinate) / `coords -> G` (by parameter name; the factory form) | ✅ |
| | `HessianMetric` | the block Hessian of the target, eigenvalue-clamped (`softabs` default / `softplus`, `mimcs/hmc/spectral.py`) | ✅ |
| | `MetricExpr` (mini-language) / `LearnedDiagonalBlock` | `M_i` = expression (`Exp`/`Sigmoid`, `+`/`*`) over `q_{-i}`, KL-fit to conditional score cov | ✅ |
| | `ShapedLearnedBlock` | nondiagonal `M = D(x)^{1/2} A D(x)^{1/2}`; `D` from the mini-language, constant `A` dense or low-rank | ✅ |
| | position-*varying* dense metric | `M_i(q_{-i})` via (log-)Cholesky, KL-fit | deferred (no reference) |
| Kinetics | `RiemannianKinetic` | general `G_i(q)` over a (fused) block, non-separable; `flow` = the block generalized leapfrog of `T_i`; factory `kind="riemannian"` | ✅ |
| | `DiagonalBlock` / `LearnedDiagonalBlock` / `ShapedLearnedBlock` | per-block `M_i(q_{-i})` slice-aware kinetic (explicit flow); listed in `BaseHMC` | ✅ |
| Integrators | (`SplittingIntegrator` / unified `leapfrog`) | reused as-is — composes each block's flow; RMHMC needs no dedicated integrator | ✅ |
| Solvers | `PicardSolver` / `AndersonSolver` | swappable fixed-point solver for the implicit flow, to a tolerance, reporting convergence | ✅ |
| Samplers | `RMHMC` (implicit) / explicit block RMHMC | fixed-length; explicit = `HMC` with the block-kinetics list (`explicit_rmhmc` builder, no sampler subclass) | ✅ |
| | RM-NUTS / explicit RM-NUTS | NUTS composed with the Riemannian kinetic / block-kinetics list (no new code) | ✅ |
| Adaptation | `MetricAdaptation` | SGD on the KL loss from cached scores, with adaptive grad clipping | ✅ |
| | `ShapedMetricAdaptation` | `D(x)` by the same KL-SGD, constant shape `A` from the `D`-whitened score (dense or Sanger low-rank) | ✅ |
| | `HessianSoftnessAdaptation` | a Hessian metric's softness `1/b` = the 10% quantile of the positive block curvatures / 3 | ✅ |
