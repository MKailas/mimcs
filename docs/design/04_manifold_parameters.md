# Parameters, Charts, and Reparameterization

## Motivation

The chart framework serves two distinct but mathematically unified purposes.

### 1. Manifold-typed parameters

Many statistical models have parameters that live on manifolds rather than unconstrained Euclidean space:

| Manifold | Example parameter | Ambient embedding |
|---|---|---|
| S^(d-1) (hypersphere) | Directional parameter, unit quaternion | ℝ^d |
| SO(3) | 3D rotation | ℝ^(3×3) or ℝ^4 (quaternion) |
| Gr(k, n) (Grassmannian) | Subspace of ℝ^n of dimension k | ℝ^(n×k) |
| SPD(d) (symmetric positive-definite matrices) | Covariance matrix | ℝ^(d×d) |
| Δ^(d-1) (probability simplex) | Categorical probabilities | ℝ^d |
| ℝ^+ | Scale parameter | ℝ (via log transform) |

A chart gives the sampler a local Euclidean coordinate system in which standard proposals (Gaussian perturbations, leapfrog steps) can operate. The chart framework is the correct abstraction for this even when the library provides only a fixed set of charts for each manifold type.

### 2. Automated reparameterization for Euclidean parameters

Even for parameters on unconstrained Euclidean space, reparameterization can dramatically improve sampling efficiency. Well-known examples include:

- **Centering / non-centering**: In a hierarchical model `θ_i ~ N(μ, σ²)`, sampling the centered parameterization `θ_i` is inefficient when data is sparse; the non-centered parameterization `ε_i = (θ_i - μ)/σ` removes the funnel geometry.
- **Affine standardization**: Sampling `u = (x - μ̂) / σ̂` where `μ̂` and `σ̂` are estimates of the marginal location and scale reduces correlations between the MCMC step size and the natural scale of the parameter. The estimates can be the running **mean and standard deviation** (`CenteringAdaptation`) or, for heavy-tailed marginals where the empirical variance chases tail excursions, the robust **median and MAD** (`RobustCenteringAdaptation`; the median-absolute-deviation is adapted on the log scale so the scale stays positive).
- **Whitening**: Sampling `u = L⁻¹(x - μ̂)` where `L` is the Cholesky factor of the estimated covariance removes posterior correlations that would otherwise force the sampler to take small steps.

Manual reparameterization requires model-specific expertise and is error-prone. The chart framework makes automated reparameterization a first-class feature: a reparameterization is simply a chart on Euclidean space, and the chart's hyperparameters (mean, scale, Cholesky factor) can be adapted online from the running sample statistics, just as step size and mass matrix are adapted.

## Core Concepts

### Ambient representation vs. coordinate representation

The **ambient representation** (`sample`) is the canonical position: what gets stored in the sample store and what the model's log-prob function expects. For manifold parameters it is the embedding in Euclidean space; for Euclidean parameters it is the parameter value itself.

The **coordinate representation** (`coordinate`) is the image of the ambient point under the active chart map. It lives in ℝⁿ where n is the intrinsic dimension of the space (equal to the ambient dimension for Euclidean parameters; less than the ambient dimension for manifold parameters). This is the space where the sampler operates.

### Chart

A **chart** (U, φ, h) consists of:
- An open subset U ⊂ M (domain)
- A homeomorphism φ_h: U → ℝⁿ (the coordinate map, parameterized by **hyperparameters** h)
- An inverse φ_h⁻¹: ℝⁿ → U (the parameterization or retraction)

For fixed charts (all manifold cases, and Euclidean identity), h is empty and φ does not depend on it. For adaptive reparameterizations, h contains learnable quantities (mean, log-scale, log-Cholesky entries) that are updated during adaptation.

### Chart hyperparameters

Chart hyperparameters `h` are continuous JAX arrays that parameterize the chart map. They live in `state.chart_hyperparams` as a tuple, one pytree per parameter. They are:

- **In the JAX state**, so the kernel can use them to convert proposed coordinates back to ambient and to evaluate the log-Jacobian.
- **Updated by `postprocess`** via `state._replace(chart_hyperparams=...)`, exactly as step size and mass matrix are updated.
- **Fixed structure throughout the run**: only values change, not shapes. This is required by JAX's JIT constraint that pytree structure must be static.

For parameters with no learnable hyperparameters (fixed charts), `chart_hyperparams[i] = None`. JAX treats `None` as an empty pytree leaf, so it flows through JIT without overhead.

### Chart index and atlas dispatch

An **atlas** is a collection of charts that together cover M. For most parameters a single chart suffices. For manifold parameters with genuine topological obstructions (e.g., no single smooth chart can cover S¹), an atlas of two or more charts is needed.

> **Status: not implemented.** Nothing in the library dispatches on a chart index today. `n_charts()` / `chart_contains()` are never called, `state.chart_indices` is initialized to zeros and never written, and every implemented parameter (including `UnitVectorParameter`, whose adaptive pole removes the need — see below) has a single chart. This section and "Chart Transitions in `preprocess`" describe a *design*, not code. Read them as a specification to build against, not a description of what runs.

The active chart for each parameter is indicated by an integer index stored in `state.chart_indices` as a tuple of scalar integer arrays. When a chart index is needed inside the JIT kernel (for example, to call the correct `from_coordinate` for a proposed position), dispatch uses `jax.lax.switch`:

```python
# Inside kernel: convert proposed coordinate back to ambient
sample_proposed = jax.lax.switch(
    state.chart_indices[i],
    [lambda u: param.from_coordinate(u, hp, 0),
     lambda u: param.from_coordinate(u, hp, 1)],
    coordinate_proposed_i,
)
```

`BaseParameter` subclasses with multiple charts implement each chart as a separate code path. The Python-facing chart methods also accept a Python integer `chart_index` (for use in `preprocess` / `postprocess`), which can use plain `if` / `elif` branching.

Chart transitions (switching chart index) happen exclusively in `preprocess` on the Python side, where `chart_contains` is evaluated and the new chart index, new coordinate, and new log_prob are computed before the kernel is called.

## The `BaseParameter` Class

```python
from typing import Any
import jax
import jax.numpy as jnp
from jax import Array

class BaseParameter:
    """
    Abstract base for a typed model parameter.

    A parameter may be scalar, vector, matrix, or tensor-valued, and may live
    on a manifold or in Euclidean space.  Charts may be fixed (manifold cases,
    identity) or adaptive (reparameterizations with learnable hyperparameters).
    """

    name: str                       # parameter name in model
    ambient_shape: tuple[int, ...]  # shape of the ambient-space representation
    coord_dim: int                  # intrinsic dimension (= dim of coordinate chart ℝⁿ)
    parents: tuple[str, ...] = ()   # names of parameters this chart depends on (see below)

    # --- Hyperparameter interface ---

    def init_hyperparams(self) -> Any:
        """
        Return the initial chart hyperparameters as a JAX pytree.
        Called once at sampler construction to initialize state.chart_hyperparams[i].
        Return None for parameters with fixed charts (no learnable hyperparameters).
        """
        return None

    # --- Chart interface ---
    # The chart methods accept `hyperparams`, `chart_index`, and `parents`.
    # Implementations for fixed charts may ignore `hyperparams` (it will be None).
    # Implementations for single-chart parameters may ignore `chart_index`.
    # `parents` is a dict {parent_name: ambient_value} of this parameter's parents'
    # values (empty for parameters with no parents); see "Parameters with parents".

    def to_coordinate(
        self,
        sample: Array,
        hyperparams: Any,
        chart_index: int = 0,
        parents: dict | None = None,
    ) -> Array:
        """Map from ambient to coordinate chart. Returns shape (coord_dim,)."""
        raise NotImplementedError

    def from_coordinate(
        self,
        coordinate: Array,
        hyperparams: Any,
        chart_index: int = 0,
        parents: dict | None = None,
    ) -> Array:
        """Map from coordinate chart to ambient. Returns shape ambient_shape."""
        raise NotImplementedError

    def log_jacobian_det(
        self,
        coordinate: Array,
        hyperparams: Any,
        chart_index: int = 0,
        parents: dict | None = None,
    ) -> Array:
        """
        Log |det J_{φ⁻¹}(u; h)|: log absolute Jacobian of the parameterization
        φ_h⁻¹: ℝⁿ → M w.r.t. the coordinate u.

        Added to the model log-prob to correct the density so the stationary
        distribution is the target distribution under the Riemannian volume measure.

        Returns a scalar.
        """
        raise NotImplementedError

    def n_charts(self) -> int:
        """Number of charts in the atlas. Default: 1."""
        return 1

    def chart_contains(self, sample: Array, chart_index: int) -> bool:
        """
        Python-side check: is `sample` in the domain of `chart_index`?
        Used by preprocess to decide when a chart transition is needed.
        May use Python control flow; not called inside JIT.
        """
        return True
```

## Fixed-Chart Parameter Examples

### `EuclideanParameter` (identity chart)

```python
class EuclideanParameter(BaseParameter):
    """Unconstrained Euclidean parameter with no reparameterization."""

    def init_hyperparams(self):
        return None

    def to_coordinate(self, sample, hyperparams, chart_index=0):
        return sample.ravel()

    def from_coordinate(self, coordinate, hyperparams, chart_index=0):
        return coordinate.reshape(self.ambient_shape)

    def log_jacobian_det(self, coordinate, hyperparams, chart_index=0):
        return jnp.zeros(())
```

### `PositiveParameter` (log transform, fixed chart)

```python
class PositiveParameter(BaseParameter):
    """Scalar parameter constrained to ℝ⁺. Chart: log transform."""

    ambient_shape = ()
    coord_dim = 1

    def init_hyperparams(self):
        return None

    def to_coordinate(self, sample, hyperparams, chart_index=0):
        return jnp.log(sample).ravel()

    def from_coordinate(self, coordinate, hyperparams, chart_index=0):
        return jnp.exp(coordinate[0])

    def log_jacobian_det(self, coordinate, hyperparams, chart_index=0):
        # φ⁻¹(u) = eᵘ; |dφ⁻¹/du| = eᵘ
        return coordinate[0]
```

## `UnitVectorParameter`: S^(d-1) in an adaptive stereographic chart

`UnitVectorParameter` (DSL: `unit_vector[d]`, or `array[n] unit_vector[d]` for an array of them,
ambient shape `(n, d)`) is the first parameter for which **coord_dim ≠ ambient size**: `d`
constrained ambient components, `d - 1` free coordinates. It is also the case that makes the
sample/coordinate distinction load-bearing — the ambient value is what the model's density and
any sample-space diagnostic see; the coordinate is only the sampler's working space.

Unlike the other manifold entries below, this one is a **single adaptive chart** rather than a
fixed atlas. See "Why one chart" below.

### The chart

With `H_v y = y − 2v⟨v,y⟩` the reflection about a unit `v` (an isometry of the sphere, an
involution, `|det| = 1`), and `z = H_v x`:

```
u = s · z[:-1] / (1 − z[-1])          # to_coordinate
w = u / s ;  z = [2w, |w|²−1] / (|w|²+1) ;  x = H_v z     # from_coordinate
log|J| = (d−1) · (log 2 − log1p(|w|²) − log s)
```

The Jacobian is the standard stereographic conformal factor `(2/(1+|w|²))^(d-1)` times
`s^-(d-1)` from `w = u/s`; the reflection contributes nothing. Note it is a **Gram** determinant
— `from_coordinate: ℝ^(d-1) → ℝ^d` is not square, so the volume element is
`0.5 log det(JᵀJ)`, which is what the test checks against autodiff.

Two hyperparameters place the chart (`SphereChart`), carrying `d` degrees of freedom in total —
the same count as the von Mises–Fisher family's `(μ, κ)`:

- **`householder`** `v` fixes the **pole** `p = H_v e_d`, the one point the chart cannot
  represent. Because `H_v` is symmetric, `z[-1] = ⟨x, p⟩`: the last rotated component *is* the
  pole alignment.
- **`log_scale`** `log s` places the projection plane.

### Why the plane's normal is tied to the pole

Project from pole `e_d` onto the plane `{y_d = c}`: the line through `e_d` and `x` meets it at
`u = (1−c)·x_{1:d-1}/(1−x_d)`. So with the normal parallel to the pole, **the plane's offset `c`
is purely a global scale**. Normalizing so the sphere∩plane circle is the *unit* circle of
coordinate space divides by `sqrt(1−c²)`, giving a clean bijection:

```
s = sqrt((1−c)/(1+c))          c = (1−s²)/(1+s²)          c ∈ (−1,1) ↔ s ∈ (0,∞)
```

A **tilted** plane (normal free of the pole) adds nothing worth having: projecting from `p` onto
a tilted plane equals projecting onto the perpendicular plane and then applying a *projective*
(non-conformal) map of coordinate space. It costs `d−2` extra hyperparameters and does not buy
elliptical/anisotropic control — that belongs to the mass matrix or a whitening chart, per "The
reparameterization principle and mass matrix adaptation" above. `SphereChart` is a NamedTuple so
that a tilt could be added as extra fields later without restructuring the pytree.

### The identity behind the adaptation

Since `|u|² = s²(1+z_d)/(1−z_d)`:

```
|u| < 1   ⟺   z_d < (1−s²)/(1+s²) = c   ⟺   ⟨x, p⟩ < c
```

"Inside the unit circle" means "on the far side of the cutting plane from the pole", so **`c` is
exactly the quantile of `⟨x, p⟩` that the chart targets**. Aiming for half the draws inside makes
`c` the median of `⟨x, p⟩`, and needs nothing more than a scalar Robbins–Monro step. See
`mimcs.adaptation.UnitVectorCenteringAdaptation` for the estimators: the scale is a stochastic
quantile, and the pole is stepped along the **geodesic** toward each draw's antipode, which
estimates the **Fréchet (Karcher) mean of the antipodal draws**.

### Estimate the pole intrinsically

This generalizes, so it is worth stating here rather than only in the mixin: for a chart
hyperparameter that lives *on* a manifold, prefer an intrinsic estimator to an extrinsic one.
The obvious extrinsic choice for the pole is the anti-mean-direction `−normalize(E[x])`, and it
is **undefined exactly where it is needed**. A running mean tends to `0` on any diffuse target,
so normalizing it is `0/0`: the magnitude vanishes while the direction does not converge, and the
estimate wanders the sphere. No gate repairs this — a floor test on the mean resultant only turns
the wander into a flicker, because a hard threshold on a fluctuating quantity leaks on every
crossing. The defect is in the estimand, not the filtering.

Riemannian SGD on `F(p) = ½·E[d(p, −x)²]` — step along the geodesic toward `−x`, gain from the
usual Robbins–Monro schedule — never forms that ratio. Every step is a bounded move (`gain·θ`, at
most `gain·π`) toward a well-defined point. It also strictly dominates: it finds the axis of a
great-circle (axial) target, which the mean cannot see at all since `E[x] = 0` there, while still
recovering `−μ` on a concentrated one. (The second moment `E[x xᵀ]` also finds the axis, but
cannot distinguish `+μ` from `−μ`, so it fails the concentrated case.)

`SO3Parameter` and `GrassmannParameter` will face the same choice; the same reasoning applies.

**Convergence is not the guarantee.** A uniform target has no Fréchet mean — every point
minimizes `F` — so the pole keeps drifting, and that is fine. What adaptive MCMC requires is
*diminishing adaptation*: the chart's change per iteration is `O(gain) → 0`, which holds whether
or not the estimator has a limit.

### Why the pole is stored indirectly

Solving `H_v p = e_d` for `v` gives `v ∝ p − e_d`, which degenerates as `p → e_d`. Storing `p`
would put a `0/0` guard in `from_coordinate`'s gradient path. Storing `v` and *defining*
`p := H_v e_d` makes the chart unconditionally well-defined; the degenerate inverse then only
arises in the Python-side adaptation, which can branch freely. (At `p = e_d` every `v ⊥ e_d`
works, since such a reflection fixes `e_d`.)

### Why one chart, not the two-chart atlas

An earlier sketch of this class used a fixed two-chart atlas (north/south stereographic
projections) with `chart_contains`-driven transitions. The adaptive chart makes that
unnecessary: coordinate space `ℝ^(d-1)` is complete, the pole is never reachable at finite
coordinates, and the adaptation actively parks the singularity where the density is lowest. So
`n_charts()` stays 1 and no transition ever fires.

The **limitation** this accepts: on a near-uniform target there is no pole to find — every
placement is as good as any other — so the pole drifts and the coordinates keep heavy tails. NUTS
may struggle near the pole of a target with real mass there. No chart placement fixes that; only
an atlas would. (Axial targets are *not* in this bucket: the Fréchet-mean estimator handles them,
which is why it is the one the mixin uses.)

Should a second chart be wanted later, note that **the atlas machinery in this document is not
implemented**: `n_charts()` and `chart_contains()` are defined on `BaseParameter` but never
called; `state.chart_indices` is initialized to zeros and never written; `preprocess`
(`samplers/base.py`) only injects the RNG draw; `RandomWalkMH` has no `state_at_coordinate`; and
the `_maybe_switch_chart` sketch below references four `Model` methods that do not exist
(`unpack_coordinate_per_param`, `unpack_sample_per_param`, `pack_coordinate_from_parts`,
`log_prob_flat`). Also note that `chart_indices[i]` is a **traced** `int32` array, so the Python branching used in
the sketches (`if chart_index == 0`) will not JIT — this document once said dispatch with
`jax.lax.switch` while `05_model_interface.md` read the index Python-side with
`int(chart_indices[i])`. That disagreement is now moot: the current design sketch (below) makes
the chart index a *sampled* quantity, which settles where it lives.

## `SimplexParameter`: Δ^(d-1) by stick breaking

A probability vector: `d` positive components summing to one. Like `unit_vector`, its coordinate space is smaller than its ambient space — `d - 1` free coordinates for `d` components — because the sum-to-one constraint removes exactly one degree of freedom.

The chart is the standard **stick-breaking** transform. Break a unit stick in `K - 1` steps, taking a fraction `z_k = sigmoid(y_k + offset_k)` of whatever is left:

```
p_k = z_k · (1 − Σ_{i<k} p_i),   k = 1 … K−1;    p_K = 1 − Σ_{i<K} p_i
```

Two implementation notes that matter:

* **It vectorizes.** The stick remaining after `k` steps is `Π_{i≤k} (1 − z_i)` — a *cumulative product* — so no sequential scan is needed. Working in logs (`log_sigmoid` plus a `cumsum`) keeps it stable when a part is tiny, and the inverse takes its ratios from *suffix* sums rather than `1 − prefix`, which cancels catastrophically once the prefix approaches one.
* **The offsets place `y = 0` at the uniform point** `p_k = 1/K`, via `offset_k = −log(K − k)`. This is not cosmetic: `Model.default_sample` evaluates every chart at the coordinate origin, so the offsets are what make the origin a sensible default rather than a corner of the simplex.

### Features and the Stein operator on the simplex

The generic `[x, x²]` features are rank-deficient here, but for the **opposite** reason to the unit vector. There the constraint is `Σ x_k² = 1`, so the *squares* are collinear with an intercept; here it is `Σ x_k = 1`, so the *linear* terms are. Dropping just `x_d` is not enough: `x_d² = (1 − Σ_{k<d} x_k)²` expands into squares *and cross terms*, and while the cross terms put it outside the feature span for `d > 2`, at `d = 2` there are none and `x_2²` is again an exact combination of `1, x_1, x_1²`. Dropping the last component entirely is rank-safe for every `d` and leaves exactly the default features on the `d − 1` free coordinates — two per degree of freedom.

For the Stein terms the simplex is **flat** (it lies in the affine hull `Σ x_k = 1`), so unlike the sphere there is no curvature term. With `P = I − (1/d)11ᵀ` — subtracting the mean — the Langevin generator gives

```
L x_i   = (P g)_i
L x_i²  = 2(1 − 1/d) + 2 x_i (P g)_i
```

using `Δ_V(x_i²) = 2 P_ii = 2(1 − 1/d)`. Only `P g` appears, which is what makes this independent of how the score was obtained: the ambient score is defined only up to the constraint normal `1`, and the projection discards exactly that.

The simplex, unlike the sphere, has a **boundary**, so `E_π[L h] = 0` needs the boundary flux to vanish — true when the density goes to zero on the faces, the same assumption `BoundedParameter` makes at its bounds. A Dirichlet with any concentration below one puts mass on a face and breaks it, so a Stein flag there is reporting the assumption, not the sampler. Verified against 200k exact Dirichlet draws with every `α_k > 1`: all `|z| < 4`, while a deliberately mis-scaled score is caught at `|z| > 10`.

## `OrderedParameter`: an increasing vector, bounded or not

The type for mixture-component locations and ordinal cutpoints, where without ordering the posterior is invariant under relabelling and multimodal by construction. Ordering costs no dimension — an increasing vector is an *open subset* of `R^d` — so there are `d` coordinates for `d` components, and the default features and flat Stein terms apply unchanged.

Which link is used depends on which bounds are present, mirroring `BoundedParameter`:

| Bounds | Construction | `log|J|` |
|---|---|---|
| neither | `x_1 = y_1`, then gaps `exp(y_k)` | `Σ_{k>1} y_k` |
| lower `L` | `x_1 = L + exp(y_1)`, then gaps `exp(y_k)` | `Σ_k y_k` |
| upper `U` | the reflection `x ↦ U − x` of the lower case | `Σ_k y_k` |
| both `(L, U)` | stick breaking on the gaps (below) | simplex term `+ d log(U − L)` |

The upper-only case being the *reflection* of the lower-only one is the same relationship the reflected-log link has to the log link on a scalar — worth pinning in a test, since it is the reason the two share an implementation up to a flip.

The doubly-bounded case is the interesting one. Its `d + 1` **gaps** `(x_1 − L, x_2 − x_1, …, U − x_d)` are positive and sum to `U − L`, so **a doubly-bounded ordered vector *is* a scaled `(d+1)`-part simplex**. That identity is why `_stick_breaking.py` exists as a shared module rather than living inside `simplex.py`: two apparently unrelated constrained objects turn out to be the same object, and both have exactly `K − 1` free coordinates. It also hands over the coordinate origin for free — the uniform simplex point becomes the *evenly spaced* vector across `(L, U)`.

`ordered_uniform` is the sharp test of that Jacobian: the order statistics of `d` i.i.d. uniforms have a *constant* density, so the chart's Jacobian carries the entire target, exactly as `uniform_sphere` does for the stereographic chart.

### Two caveats worth writing down

**Precision.** A gap is `exp(y_k)`, so a coordinate deep in the negatives asks for two entries closer together than the floats at that magnitude can express; below about `1.2e-7` relative in float32 the gap is absorbed and the entries come out *equal*, so the ordering stops being strict. No chart can avoid this — they are the same float — and the log-Jacobian is unaffected because it is computed from the coordinate, never from the differences. But a model taking `log(x_{k+1} − x_k)` sees an infinity long before the sampler is in real trouble. That is a reason to reach for x64, not to guard the chart.

**The Stein boundary.** The ordered region's boundary includes the *internal* faces `x_k = x_{k+1}`, and a density need not vanish there — the order statistics of an i.i.d. sample, the canonical ordered target, are perfectly happy with two coordinates coinciding. Where that happens the Stein z-scores carry a bias that belongs to the target, not the sampler. This is the same caveat `BoundedParameter` records at its bounds, but much easier to trip over, since a repulsive prior between neighbours is the exception rather than the rule. R-hat and ESS on the same draws are unaffected.


## `CovMatrixParameter` / `CholeskyFactorCovParameter`: covariance matrices

Stan's `cov_matrix[K]` and `cholesky_factor_cov[K]`, and the first pair of types that differ only
in their **ambient value**: `Σ` for one, its Cholesky factor `L` with `Σ = L Lᵀ` for the other. They
share a chart, a feature set and a Stein operator, and so share a module (`cholesky_cov.py`); the
factor exists as its own type because a multivariate normal density *uses* the factor, so a model
written over `L` never forms `Σ` at all.

### The chart is log-Cholesky

The `m = K(K+1)/2` coordinates are the strict lower triangle of `L` together with `log L_ii`:

```
L_ii = exp(z_ii),   L_ij = z_ij  (i > j),   Σ = L Lᵀ
```

Positive definiteness is structural rather than checked, the coordinates are unconstrained, and
`z = 0` is the identity — so the charts' origin, which `Model.default_sample()` returns, is a
sensible start. The two Jacobians compose, and both match Stan:

| Type | `log|J|` |
|---|---|
| `cholesky_factor_cov` | `Σ_i z_ii = Σ_i log L_ii` |
| `cov_matrix` | `K log 2 + Σ_i (K − i + 2) log L_ii` |

This is the same parametrization `ScoreMassAdaptation` fits a dense mass in and `PolyakLog`
averages in (doc 06) — the third place the library reaches for log-Cholesky, and for the same
reason each time: it is the parametrization in which a positive definite matrix has no constraint
left to enforce.

Only the **square** case ships. Stan's rectangular `cholesky_factor_cov[M, N]` has a singular `Σ`,
which is not a point of the space the Stein operator below lives on, so it would be a different
type with a weaker diagnostic rather than an extra size argument on this one.

### Features are taken from `Σ`, never from `L`

The lower triangle of `Σ` and its squares — `2m` per matrix — whichever of the two is the ambient
value. The covariance is what a model means and what a reader reports, and it is the invariant
one: two chains can agree on `Σ` while their factors differ. Nothing is dropped here, unlike the
simplex and the unit vector, because `vech(Σ)` ranges over an open set with no constraint tying
the entries together, so `[x, x²]` over them is already full rank.

For `cholesky_factor_cov` the ambient array is the full `(K, K)` matrix with a structurally zero
strict upper triangle, which shows up in the posterior summary as `K(K−1)/2` rows of exact zeros.
That is deliberate: the DSL needs the matrix shape so a model can write `L * L'`, and hiding
zero-variance rows is precisely what a frozen coordinate must *not* do. No diagnostic is computed
from them, since the features come from `Σ`.

### The Stein operator: Brownian motion on the SPD matrices

The flat operator is available and wrong-shaped here — the natural geometry of covariance matrices
is not Euclidean. Take `P(K)`, the positive definite matrices under the **affine-invariant** metric

```
⟨A, B⟩_Σ = tr(Σ⁻¹ A Σ⁻¹ B),
```

the Riemannian symmetric space `GL(K)/O(K)`, on which every `Σ ↦ A Σ Aᵀ` is an isometry. Its
volume element is `det(Σ)^{−(K+1)/2} dvech(Σ)`, so a target with Lebesgue density `p` has
Riemannian density `π = p · det(Σ)^{(K+1)/2}`, and the generator of the Langevin diffusion
`L h = Δh + ⟨∇h, ∇log π⟩` satisfies `E_π[L h] = 0`.

Two facts give the terms. **Linear coordinates are eigenfunctions of the Laplacian**,
`Δ tr(CΣ) = ((K+1)/2) tr(CΣ)`: at `Σ = I` summing `E_k²` over an orthonormal basis of the symmetric
matrices gives `((K+1)/2) I`, and the isometries carry that to every `Σ`. The product rule with
`‖∇ tr(CΣ)‖² = tr((CΣ)²)` gives the squares. Writing `S` for the score in the `D log p[V] = tr(SV)`
convention, so that `∇ log π = Σ S Σ + ((K+1)/2) Σ`:

```
L Σ_ab   = (K+1) Σ_ab + (Σ S Σ)_ab
L Σ_ab²  = (2K+3) Σ_ab² + Σ_aa Σ_bb + 2 Σ_ab (Σ S Σ)_ab
```

Both were checked by hand at `K = 1`, where `P(1)` is the log-scale line and they collapse to
`2v + v²s` and `6v² + 2v³s`.

**This is the first Stein operator in the library that needs no boundary assumption.** `P(K)` is
complete and boundaryless — the singular matrices are at *infinite* distance in this metric — so
unlike `SimplexParameter` (whose faces need a vanishing density) and `BoundedParameter` (likewise
at its bounds), nothing has to be assumed about the target beyond integrability. A flag means the
draws, not the fine print. It is a good argument for choosing the geometry to match the parameter
rather than the storage.

### Getting `S` from the factor's score

`Σ S Σ` is the only form of the score the operator needs, and for `cov_matrix` it is immediate:
symmetrize the ambient cotangent (only the symmetric part is a direction of the manifold) and
multiply. For `cholesky_factor_cov` the score arrives as `∂ log p_L/∂L` — the density of the
*factor*, which differs from `p_Σ` both by the chain rule through `Σ = L Lᵀ` and by that map's
Jacobian. Undoing both is short but not obvious. With the Jacobian term removed,
`B = tril(S_L) − diag((K − i + 1)/L_ii)` satisfies `tril(2 S L) = B`, and the symmetry of `S` pins
the strict upper triangle that `tril` dropped: writing `Z = B + U` for `U` strictly upper and
requiring `Lᵀ Z` symmetric forces `Lᵀ U = −triu(Lᵀ B − Bᵀ L, 1)`, one triangular solve. Then
`Σ S Σ = L Y Lᵀ` with `Y = Lᵀ Z / 2`.

That derivation is exactly the kind that fails silently, so it is checked against a deliberately
dumb reference — autodiff of the same density written in `Σ` — in the same way `SimpleNUTS` checks
`NUTS`. The operator as a whole is checked against exactly sampled Wishart draws (Bartlett's
decomposition produces the Cholesky factor directly, so no MCMC is in the loop), with a control
that fires when the score is deliberately mis-scaled.


## `CorrMatrixParameter` / `CholeskyFactorCorrParameter`: correlation matrices

Stan's `corr_matrix[K]` and `cholesky_factor_corr[K]`, and the same two-views-one-chart arrangement
as the covariance pair: `Omega` with a unit diagonal, or the lower-triangular `L` whose *rows* have
unit norm. They matter because a covariance is usually better modelled as
`diag(sigma) Omega diag(sigma)`, with scales and dependence carrying separate priors.

### The chart

Stan's: canonical partial correlations `z = tanh(y)`, then stick breaking on **sums of squares**,
each row spending what is left of a unit budget.

```
L_11 = 1,  L_i1 = z_i1,  L_ij = z_ij sqrt(1 − Σ_{k<j} L_ik²),  L_ii = sqrt(1 − Σ_{k<i} L_ik²)
```

The diagonal is *determined*, not free, so the `m = K(K−1)/2` coordinates match the `m` free ambient
entries — the strict lower triangle — and `y = 0` is the identity. Both log-Jacobians were checked
against autodiff at `K = 2..5`:

| Type | `log|J|` |
|---|---|
| `cholesky_factor_corr` | `Σ log(1 − z_ij²) + ½ Σ log(1 − Σ_{k<j} L_ik²)` |
| `corr_matrix` | the same, plus `Σ_i (K − 1 − i) log L_ii` |

The stick breaking is triangular in row-major order — `L_ij` depends only on `z_ij` and on earlier
entries of *its own row* — which is why the second factor is just a product of `dL_ij/dz_ij`.

One identity earns its keep twice: **`det Omega = Π_i L_ii² = Π_{i>j} (1 − z_ij²)`**. It gives the
LKJ density in the coordinates for free, and it is what makes exact LKJ sampling available for the
tests — the canonical partial correlations are independent, `z_ij ~ 2 Beta(b, b) − 1` with
`b = eta + (K − 2 − j)/2` on 0-based column `j`. That law was *derived* from the identity rather
than looked up, and verified to 1e-15.

### Features

The **strict** lower triangle of `Omega` and its squares. The diagonal is constantly 1 and is
excluded — a constant feature is a frozen coordinate, whose ESS is perfect and whose R̂ is
undefined, which is the failure mode `CLAUDE.md` warns about. Nothing else is dropped; the
elliptope's interior is open in `R^m`.

### The Stein operator, and why it is not written as a Laplacian

Take the correlation matrices as an **embedded submanifold** of the SPD symmetric space (doc 04's
covariance section): tangent space the symmetric matrices with zero diagonal, `Sym₀`, carrying the
*induced* affine-invariant metric `⟨A, B⟩_Ω = tr(Ω⁻¹AΩ⁻¹B)` unchanged.

The projection that goes with it is clean. The metric-orthogonal complement of `Sym₀` in `Sym` is
`{Ω D Ω : D diagonal}` — the residual is orthogonal to every zero-diagonal `B` exactly when
`Ω⁻¹(V − P(V))Ω⁻¹` is diagonal — and requiring `diag(P(V)) = 0` pins `D`:

```
P(V) = V − Ω diag(d) Ω,     (Ω ∘ Ω) d = diag(V)
```

with `∘` the Hadamard product. `Ω ∘ Ω` is positive definite by the Schur product theorem, so the
`K × K` solve is well posed. So `grad_E φ = P(Ω C Ω)`.

**The Laplacian is where the covariance recipe stops working.** There the terms come from
`Δφ + ⟨∇φ, ∇log π⟩` with a closed-form `Δ`, because linear coordinates are its eigenfunctions. On a
submanifold `Δ_E` differs from the ambient Laplacian by the mean curvature, and the second
fundamental form of the elliptope in this metric is not something that could be derived here with
any confidence. Rather than guess, the operator is assembled in its **divergence form**. In any
coordinates `x` whose reference measure is Lebesgue with density `p`,

```
A φ = div_x(v) + v · s,     v = grad_E φ in those coordinates,   s = ∇_x log p
```

and this is *identically* `Δ_E φ + ⟨∇φ, ∇log π_riem⟩`: the volume element relating `p` to `π_riem`
cancels exactly the difference between the Euclidean and Riemannian divergences. It is therefore
the operator induced by Brownian motion on the submanifold — what was wanted — only computed
without ever forming `Δ_E`. The coordinates are the strict lower triangle of `Omega`, and `div_x`
comes from `jax.jacfwd`: **exact, not an approximation**, at `m` extra tangent evaluations per draw.

That cost is a diagnostic's, paid once per draw in `summary()` and never in a kernel; it grows about
as `m² K³`, which is comfortable at the sizes correlation matrices are used at. A closed form would
be an optimization, not a correction.

Two things justify trusting it without one, and both are tests:

* applied to the SPD geometry the same construction **reproduces the covariance closed forms**
  exactly — the oracle, in the `NUTS`/`SimpleNUTS` mould;
* against LKJ the terms integrate to zero by quadrature (`K = 2` to 7e-9, `K = 3` to 1e-8) and
  average to zero over exact draws, with a mis-scaled-score control that fires.

**No boundary assumption**, as for the covariance types: distances within a submanifold are at least
the ambient ones, and `P(K)`'s boundary is already infinitely far.

### The score for the factor type

`corr_matrix` receives a `(K, K)` cotangent; only its symmetric part is a direction of the manifold,
and a coordinate moves *two* entries, so the Lebesgue score in coordinate `(a, b)` is `2 sym(G)_ab`.
`cholesky_factor_corr` needs more care, because `L`'s diagonal is **determined by its row** rather
than free: what arrives is `∂ log p/∂L` over the strict lower triangle with the diagonal's
contribution already folded in by the chain rule, and converting it to `Omega`'s coordinates is the
generic change of variables `s_Ω = J⁻ᵀ(s_L − ∇_L log|det J|)`. `J` comes from autodiff — the map is
a plain matrix product, so there is nothing to gain from deriving it, and an `m × m` solve per draw
is affordable for a diagnostic. (Differentiating with the diagonal held *independent* gives a
different and wrong score; that mistake cost an afternoon, and is now a test.)


## Adaptive Reparameterizations

> **Status: what shipped has a different shape from the sketch this section used to carry.** The
> affine standardization is **not** a parameter class (`AffineReparameterizedParameter`) but a
> `centered=True` *option* composed onto any chart: `mimcs/model/_centering.py` provides
> `init_centering` / `standardize` / `unstandardize`, and `EuclideanParameter(centered=True)` and
> `BoundedParameter(centered=True)` use it. The hyperparameters are a plain `(mu, sigma)` pair,
> fitted online by `CenteringAdaptation` (mean/sd) or `RobustCenteringAdaptation` (median/MAD).
> The **whitening** (full-Cholesky) chart is *not built* in any form.

For Euclidean parameters the chart framework enables reparameterizations whose hyperparameters are
adapted online: the parameter's chart map takes the hyperparameters as an argument, and an
adaptation mixin updates them in `postprocess` from running sample statistics. Because the chart
is a bijection with a tractable Jacobian, the target is unchanged — only its coordinates are.

**Affine (diagonal), shipped.** `x_coord = (x - mu) / sigma`, with `log|J| = -sum(log sigma)`. It
is opt-in per parameter (`centered=True`) *and* opt-in per sampler (`spec.centering`), and off by
default in both: it only acts on `centered=True` parameters, and it was measured to destabilize a
fragile far-from-mode adaptation (doc 09). Note the interaction with the mass matrix — both are
scale absorbers fitted to the same quantity, and under a position-based mass they can fight.

**Whitening (full Cholesky), not built.** `x_coord = L^{-1}(x - mu)` for a running covariance
`Sigma = LL^T` would do for correlation what centering does for scale. It is not implemented, and
the argument for building it is weaker than it looks: a dense adapted mass already addresses
correlation from inside the kernel, and the two overlap almost entirely. The one thing a whitening
*chart* would add over a dense *mass* is that it also reshapes what the chart-dependent parts of
the model see — which matters only when other charts depend on this parameter.

## Density Correction

When the target π is a density over M (with respect to the Riemannian volume or Lebesgue measure), sampling in coordinate space requires the Jacobian correction:

```
log π_coord(u; h) = log π(φ_h⁻¹(u)) + log |det J_{φ_h⁻¹}(u; h)|
```

The `log_jacobian_det` method provides the per-parameter correction. For a model with parameters p₁, …, pₖ, the total correction is:

```
log |J_total| = Σᵢ pᵢ.log_jacobian_det(uᵢ, hᵢ, chart_idx_i)
```

This sum is computed inside `model.log_prob_flat` and included in `state.log_prob` (see `05_model_interface.md`). When `jax.grad` is taken through `log_prob_flat` with respect to `coordinate` (needed by HMC leapfrog), the gradient of `log_jacobian_det` w.r.t. `coordinate` is included automatically via JAX's autodiff. The chart hyperparameters `h` are treated as constants in this differentiation (they are not the variable being differentiated).

## Parameters with Parents (Dynamic Bounds)

A parameter's chart may depend on the values of **other parameters**, not only on its own coordinate and hyperparameters. The motivating case is a constraint whose bound is itself a parameter: e.g. `b ~ Uniform(0, a)`, where `b`'s upper bound is the current value of parameter `a`. More generally, charts for ordered parameters, hierarchical scales, and triangular structures all need parent values.

### Interface

A parameter declares the names of the parameters it depends on in its `parents` attribute, and the three chart methods accept a `parents` argument — a dict `{parent_name: ambient_value}` supplying the *ambient* (sample-space) values of those parents:

```python
class BoundedParameter(BaseParameter):
    def __init__(self, name, shape=(), *, lower=None, upper=None, parents=()):
        ...
        # bounds may be constants, a parent name (str), or a callable of `parents`;
        # parent names are collected from string bounds plus the explicit `parents` arg.
        self.parents = (...)

    def from_coordinate(self, coordinate, hyperparams, chart_index=0, parents=None):
        L, U = self._bounds(parents)        # L or U may be a parent's value
        return L + (U - L) * jax.nn.sigmoid(coordinate)   # logit chart
```

### Why this stays simple: the Jacobian is still a sum

The full map from coordinates `(u₁, …, uₖ)` to samples `(x₁, …, xₖ)` is now coupled: `xᵢ = φᵢ⁻¹(uᵢ; x_parents(i))`. But because dependencies form a **DAG**, in topological order the Jacobian `∂x/∂u` is **block lower-triangular** — `xᵢ` depends on `uᵢ` and (through its parents' samples) only on the coordinates of its ancestors. The determinant of a block-triangular matrix is the product of its diagonal blocks, and the diagonal block `∂xᵢ/∂uᵢ` is exactly `pᵢ`'s own chart Jacobian *with the parents held fixed*. Therefore:

```
log |J_total| = Σᵢ pᵢ.log_jacobian_det(uᵢ, hᵢ, chart_idx_i, parents_i)
```

The total log-Jacobian remains the same simple sum as in the no-parent case; each term just receives the parent values and differentiates only with respect to its own coordinate. **No off-diagonal Jacobian terms are needed.** This is the key reason parent support is a small, local addition rather than a structural change.

### Evaluation order and gradients

`Model` computes a topological order of the parameters once at construction (raising on cycles or unknown parent names). `unpack_coordinate` then walks parameters in that order, building the sample dict incrementally so each child's `from_coordinate` receives its parents' already-computed ambient values (see `05_model_interface.md`).

For HMC, the coordinate-space gradient `∂/∂u [log π(x(u)) + log|J_total|]` is obtained by `jax.grad` through this construction. Autodiff automatically captures the cross-dependencies — the gradient w.r.t. a parent's coordinate `u_parent` flows through both the parent's sample `x_parent` *and* its effect on every child's bound and Jacobian — without any of those off-diagonal terms appearing explicitly. The triangular structure makes the determinant a sum; autodiff handles the chain rule for the gradient.

### Worked example (validates the design)

For `a ~ U(0,1)`, `b ~ U(0,a)` with conditional density `1/a` (so `log π = -log a`), the `-log a` from the model density exactly cancels the `+log a` in `b`'s chart Jacobian (`log(U−L) = log a`), leaving the coordinates `u_a, u_b` as **independent standard logistics**. A correct implementation reproduces this cancellation; a wrong parent-dependent Jacobian would not. (This is exactly the `nested_uniform` test problem.)

## Chart Transitions

> **Status: not implemented.** Two designs are recorded here. The **deterministic** one below
> (switch in `preprocess` when the state nears a singularity) is the older sketch; it calls four
> `Model` methods that do not exist — `unpack_coordinate_per_param`, `unpack_sample_per_param`,
> `pack_coordinate_from_parts`, `log_prob_flat` — and `BaseSampler.preprocess` has no transition
> step. The **current thinking**, immediately below, is different in kind and supersedes it.

### Current design sketch: the chart index as a sampled parameter

A deterministic switch has an awkward property: it is a state-dependent change of measure applied
outside the kernel, so its validity has to be argued separately from the sampler's. The alternative
is to stop treating the chart index as bookkeeping and **treat it as a parameter, inferred like any
other**. The target becomes a joint distribution over `(chart index, coordinate)`, the chart index
is part of the state the sampler updates, and correctness follows from the usual invariance
argument rather than from a hand-proof about when a switch is safe.

The transition kernel for the index is where the manifold's geometry enters. Proposals need not be
uniform over charts: on the sphere the natural choice makes a transition *likely* as the state
approaches the pole of the current stereographic projection and negligible elsewhere, which is
exactly where a chart change is worth making. That is a modelling statement about the manifold,
written once per parameter type, not a heuristic tuned per problem.

**Two open problems**, both real:

1. **The index is discrete**, so it cannot be moved by HMC. The natural fit is
   Metropolis-within-Gibbs — alternate an HMC update of the continuous coordinates at a fixed
   chart with a Metropolis update of the index at fixed position — which the mixin architecture can
   express, but which the library has no machinery for today.
2. **Adaptation across a moving chart.** Every adapted quantity the sampler holds — the mass, a
   learned metric, the step size — is fitted to the geometry *of the current chart*. If the chart
   keeps changing, either those quantities must be per-chart (multiplying the adaptation state by
   the number of charts, and slowing each one's convergence), or they must be transported across
   a transition, which needs the chart-change Jacobian to act on the metric. Neither is worked out.
   This is the harder of the two and the reason the layer is not built.

Note that the sphere does **not** motivate any of this: its adaptive chart already parks the
singularity where the density is lowest, so no transition ever fires (see "Why one chart, not the
two-chart atlas"). The motivating cases are manifolds with genuine topological obstructions —
Grassmannians, positive-definite matrices — where no single chart covers the space.

### Older sketch: a deterministic switch in `preprocess`

Chart transitions happen exclusively on the Python side in `preprocess`, before the kernel is called. For a manifold parameter whose active chart is approaching a singularity:

```python
def _maybe_switch_chart(self, state):
    hyperparams = list(state.chart_hyperparams)
    chart_indices = list(state.chart_indices)
    coordinate_parts = self.model.unpack_coordinate_per_param(state.coordinate)
    sample_parts = self.model.unpack_sample_per_param(state.sample)
    changed = False

    for i, param in enumerate(self.model.parameters):
        if param.n_charts() <= 1:
            continue
        ci = int(chart_indices[i])
        if param.chart_contains(sample_parts[i], ci):
            continue
        # transition to the first chart whose domain contains the current sample
        new_ci = next(
            j for j in range(param.n_charts())
            if param.chart_contains(sample_parts[i], j)
        )
        coordinate_parts[i] = param.to_coordinate(sample_parts[i], hyperparams[i], new_ci)
        chart_indices[i] = jnp.array(new_ci)
        changed = True

    if not changed:
        return state

    new_coordinate = self.model.pack_coordinate_from_parts(coordinate_parts)
    # Recompute log_prob to pick up the new Jacobian determinant
    new_log_prob = self.model.log_prob_flat(
        state.sample, tuple(hyperparams), tuple(chart_indices)
    )
    return state._replace(
        coordinate=new_coordinate,
        chart_indices=tuple(chart_indices),
        log_prob=new_log_prob,
    )
```

## State Layout for Multiple Parameters

`state.coordinate` and `state.sample` are flat concatenations of the per-parameter values:

```
state.coordinate    = concat([pᵢ.to_coordinate(xᵢ, hᵢ, cᵢ) for i in ...])
state.sample        = concat([xᵢ.ravel() for i in ...])
state.chart_hyperparams = (h₁, h₂, ..., hₖ)   # tuple of pytrees, one per parameter
state.chart_indices     = (c₁, c₂, ..., cₖ)   # tuple of scalar int arrays
```

The `Model` object holds the layout information (per-parameter offsets and shapes) needed to split and pack these flat arrays (see `05_model_interface.md`).

## Module Layout

Parameter types and `Model` live in one package, `mimcs/model/` — a parameter is defined by its charts, and `Model` is what composes a list of them, so they are one logical whole. The rule is **one parameter type per module**, so the table below maps one-to-one onto files:

| Module | Holds |
|---|---|
| `model.py` | `Model` (doc 05) |
| `parameter.py` | `BaseParameter`, the naming helpers, `flat_size` |
| `_centering.py` | the `(mu, sigma)` standardization shared by every `centered=True` type |
| `euclidean.py` | `EuclideanParameter` |
| `bounded.py` | `BoundedParameter`, `PositiveParameter`, `IntervalParameter` (one link family) |
| `unit_vector.py` | `UnitVectorParameter`, `SphereChart` |
| `simplex.py` | `SimplexParameter` |
| `ordered.py` | `OrderedParameter` |
| `_stick_breaking.py` | the simplex transform, shared by `simplex` and doubly-bounded `ordered` |
| `_bounds.py` | bound specs (constant / parent name / callable), shared by `bounded` and `ordered` |
| `registry.py` | `ParameterKind`, `PARAMETER_KINDS` |

Closely related types stay together — `Positive` and `Interval` are the same link machinery under different bounds — but genuinely different geometry gets its own file.

### The registry: how a front end learns about a type

A front end (today only the DSL, `mimcs/dsl/`) needs more from a parameter type than a constructor: whether its keyword takes a size argument (`unit_vector[d]`), whether a declaration may carry `<lower=…, upper=…>`, which chart options apply (`centered` vs `adaptive`), and whether it may be declared outside the `parameters` block. Each of those used to be a hard-coded `unit_vector` branch in the lexer, parser, semantic pass and factory — six of them — so adding a type meant finding all six.

`PARAMETER_KINDS` inverts that: the parameter types own the knowledge and the DSL reads it. Registering a kind *is* what reserves its keyword in the grammar (`parser._TYPE_KEYWORDS` is derived from the table).

The builders take **plain Python values** — resolved shapes as ints, bounds as `float | parent-name str | None` — and never DSL objects. This is load-bearing: it keeps `mimcs.model` from importing `mimcs.dsl`, which would be an import cycle. The DSL evaluates its own constants and resolves its own bounds, then calls `kind.build(...)` last; a builder rejects bad arguments with `ValueError`, which the DSL re-raises as a span-carrying `DslError` against the offending declaration.

Adding a parameter type is therefore: write its module, add a `ParameterKind` entry, export it from the package `__init__`, and add a test file under `tests/model/`.

## Planned Parameter Classes

Status: ✅ = implemented, ⏳ = planned.

| Class | Space | Chart hyperparameters | Status |
|---|---|---|---|
| `EuclideanParameter` | ℝ^d | None (identity chart) | ✅ |
| `BoundedParameter` | interval / half-line | None (log / reflected-log / logit); bounds may depend on parents | ✅ |
| `PositiveParameter` | ℝ⁺ | None (log transform; convenience for `BoundedParameter`) | ✅ |
| `IntervalParameter` | (a, b) | None (logit; convenience for `BoundedParameter`) | ✅ |
| `UnitVectorParameter` | S^(d-1) | `SphereChart(householder, log_scale)` (adaptive stereographic) | ✅ |
| `SimplexParameter` | Δ^(d-1) | None (stick-breaking) | ✅ |
| `OrderedParameter` | `x_1 < … < x_d`, optionally bounded at either end | None (cumulative exp; stick-breaking when doubly bounded) | ✅ |
| `SO3Parameter` | SO(3) | None (exponential map) | ⏳ |
| `SPDParameter` | SPD(d) | None (Cholesky / matrix log) | ⏳ |
| `GrassmannParameter` | Gr(k, n) | None (local QR coordinates) | ⏳ |
| `centered=True` option (not a class) | ℝ^d, bounded | `(mu, sigma)` via `_centering.py` | ✅ |
| whitening chart (full Cholesky) | ℝ^d | — | ⏳ not built; a dense adapted mass covers most of it |
