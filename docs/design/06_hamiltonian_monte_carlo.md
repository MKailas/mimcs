# Hamiltonian Monte Carlo: Modular Design

## Three Axes of Variation

An HMC sampler is a complex object, but its complexity factors cleanly into three independent axes. The design keeps them independent so that any combination is expressible without writing new code for each combination.

| Axis | What varies | Mechanism | Examples |
|---|---|---|---|
| **1. Hamiltonians** | The energy function $H(q,p)$ | Composition: a list of pluggable component objects | quadratic / relativistic / position-dependent kinetic; model + Jacobian potentials |
| **2. Integrator** | How the flow of $H$ is approximated | Composition: a (recursively) composable splitting scheme | leapfrog, higher-order, nested multi-rate (RESPA) |
| **3. Large-scale structure** | How a trajectory is built and a point selected | Inheritance from `BaseHMC`; a per-sampler `IntegratorState` | fixed-length HMC, randomized HMC, NUTS (MALT *planned*) |

Axes 1 and 2 are realized by **composition** (components held by the sampler); axis 3 is realized by **inheritance** (`HMC`, `RandomizedHMC`, `NUTS` subclass `BaseHMC`) and by each sampler defining its own **`IntegratorState`**. This mirrors the broader pattern in the library: base sampler classes via inheritance, behavior via composition and mixins (see `02_sampler_classes.md`).

## Mathematical Background

HMC targets $\pi(q) \propto e^{-V(q)}$ on the coordinate space $q \in \mathbb{R}^n$ (the chart coordinate; see `04_manifold_parameters.md`). It augments the state with a momentum $p$ and a kinetic energy $T(q,p)$, giving the **Hamiltonian**

$$H(q, p) = V(q) + T(q, p),$$

whose joint Boltzmann distribution $e^{-H(q,p)} = e^{-V(q)}\, e^{-T(q,p)}$ has $\pi$ as its $q$-marginal. Hamilton's equations

$$\dot q = \frac{\partial H}{\partial p}, \qquad \dot p = -\frac{\partial H}{\partial q}$$

define a flow that preserves $H$ and phase-space volume. HMC simulates this flow approximately with a reversible, volume-preserving integrator and corrects the discretization error with a Metropolis accept/reject on $\Delta H$.

**The decomposition that matters.** Both $V$ and $T$ are *sums of components*:

$$H(q,p) = \underbrace{\sum_i V_i(q)}_{\text{potentials}} + \underbrace{\sum_j T_j(q,p)}_{\text{kinetics}}.$$

- The $V_i$ come from the model's decomposed log-densities (prior, likelihood, …) and from the chart Jacobian corrections. This is precisely the decomposition exposed by `Model.log_prob_fns` (doc 05) and `log_jacobian_det` (doc 04).
- The $T_j$ come from the parameters: by default **one kinetic component per parameter** (a block-diagonal mass matrix), with cross-parameter coupling expressed as additional kinetic components.

Each component generates its own elementary flow. The integrator (axis 2) is a recipe for composing those flows. This is the single idea that makes all three axes independent.

## The Phase-Space State: `IntegratorState`

The object that flows through an integrator is **not** just a position–momentum pair. It is an `IntegratorState`, which additionally carries a **cache of per-potential gradients (and values)** and is **extended per sampler** to hold trajectory-level bookkeeping. The design follows Blackjax in making the integrator's working state a first-class, extensible structure.

### Base contract

```python
class IntegratorState(NamedTuple):
    q: Array                 # coordinate (position), shape (n,)
    p: Array                 # momentum, shape (n,)
    potential_values: dict   # {potential_id: scalar}  cached V_i(q)
    potential_grads: dict    # {potential_id: Array}   cached ∇_q V_i(q)
    log_weight: Array        # scalar: accumulated log-weight for this phase point
    integrator_data: dict = {}   # integrator-specific outputs (WALNUTS proxy energy, …)
```

The six fields above are the **integrator contract**: the integrator (axis 2) reads and writes only these, always via `NamedTuple._replace`. Any extra fields a sampler adds (axis 3) are therefore preserved untouched as they ride through `integrator.step`.

`integrator_data` is a dict for integrator-specific outputs a sampler may consume beyond the scalar `log_weight` — the line-search integrators write a coarse-level **proxy energy** (for step-size adaptation) and a refinement-count diagnostic here. Each integrator declares its schema via `init_integrator_data()` (empty for deterministic ones); the samplers seed it once per trajectory and it rides through every `_replace` untouched, exactly like `log_weight`. Fixed keys per run keep it a clean pytree.

`potential_values`/`potential_grads` are dicts keyed by the static component ids (e.g. `"V_log_likelihood"`, `"V_jacobian"`). As JAX pytrees with fixed keys, they trace cleanly under JIT. Caching the *value* alongside the *gradient* is free — both come from one `jax.value_and_grad` call — and the value is needed for trajectory energies (NUTS divergence checks, multinomial weights) without recomputation.

`log_weight` is a scalar log-probability weight that the integrator may **accumulate** for the phase point it produces. For a standard symplectic integrator (leapfrog, Yoshida, RESPA) the map is deterministic and exactly volume-preserving, so `log_weight` stays at its initial `0` and is ignored. It exists in the base contract for integrators that are themselves *randomized or adaptive* and must report a correction to keep the overall sampler exact. The motivating case is **WALNUTS** (Bou-Rabee & Carpenter), whose within-orbit adaptive step-size selection (a local "line search" that subdivides a leapfrog step until an energy-error criterion is met) makes the integrator a randomized map; the log of the forward/reverse transition ratio of that adaptive choice accumulates into `log_weight` so that detailed balance still holds. Because the weight is produced *inside* the integrator's step, it must live in the contract the integrator is allowed to write — hence a base field rather than a sampler-level extension.

Axis-3 samplers consume `log_weight` in selection and acceptance. The general Metropolis acceptance over a trajectory becomes

$$\alpha = \min\!\big(1,\; \exp(H_0 - H_1 + \Delta\,\texttt{log\_weight})\big),$$

which reduces to the usual $\min(1, e^{-\Delta H})$ whenever `log_weight` is zero (every integrator in this build so far). NUTS-style multinomial selection likewise adds `log_weight` to each leaf's $-H$ before normalizing.

```python
class HamiltonianContext(NamedTuple):
    """Everything a component needs besides the IntegratorState, held constant per trajectory."""
    chart_hyperparams: tuple   # from state.chart_hyperparams
    chart_indices: tuple       # from state.chart_indices
    ham_params: dict           # adapted component parameters keyed by component id
    betas: Any = None          # parallel tempering only (doc 13): the traced ladder
    kinetic_cache: Any = None  # {kinetic id: whatever its precompute() returned}
```

**`kinetic_cache` is where a loop constant goes.** `BaseHMC.context` builds the context once per
kernel call, *before* the trajectory, so anything placed there is a constant of the whole
`while_loop`. A kinetic may opt in by defining `precompute(ctx)`; returning `None` means *nothing
to cache* and contributes no entry. The motivating case is
`LowRankQuadraticKinetic`, whose $O(J^2 d)$ Sherman--Morrison recursion depends only on the mass
and is otherwise rebuilt on every leaf --- XLA common-subexpression-eliminates the repeats *within*
one loop-body iteration but will not hoist them *out* of the trajectory loop. Every consumer keeps
a fallback that computes its own factors when the cache is absent, so the cache is an
**optimization only**: it can never change a number, which is what lets `context(state,
kinetic_cache=False)` exist for callers that read only the potentials.

That opt-out is not a micro-optimization. `kernel` is jitted, so `precompute` there is traced once
and free; the reseeding callers (chart adaptation's `state_at_coordinate`, and the tempered ladder
at doc 13) run **eagerly**, dispatching the recursion primitive by primitive at ~440x the traced
cost. Building a cache they never read made the tempered ladder --- which reseeds once per warmup
iteration --- 3.6x *slower* in warmup than not hoisting at all.

### Per-sampler extension (axis 3)

Each large-scale structure defines its own `IntegratorState` by extending the base in one of two ways:

- **Flat extension (ride-along).** Add sibling fields that the integrator preserves through `_replace`. Used for any *additional* sampler-specific accumulator beyond the base `log_weight` — the integrator carries it obliviously. (The common log-weight accumulator is already a base field, so a partial-refresh sampler like the planned MALT, which only needs that, would use the base contract directly.)
- **Compositional extension.** Embed one or more base `IntegratorState`s plus summary fields. Used when the structure tracks several phase-space points at once — e.g. NUTS's left/right endpoints and selected proposal.

Both are detailed under Axis 3 below. The crucial invariant: **the integrator only ever sees the base contract.** In the flat case it carries extra fields obliviously; in the compositional case the sampler hands the integrator a single base leaf and folds the result back in itself.

## Axis 1: Hamiltonian Components

### The `Hamiltonian` protocol

Every component exposes its energy and the elementary flow it generates. A flow maps an `IntegratorState` to an `IntegratorState`:

```python
class Hamiltonian:
    id: str                       # unique key into ctx.ham_params and the gradient cache

    def energy(self, istate: IntegratorState, ctx: HamiltonianContext) -> Array:
        """Scalar energy contribution of this component at (istate.q, istate.p)."""
        raise NotImplementedError

    def flow(self, istate: IntegratorState, eps: Array, ctx: HamiltonianContext,
             use_cache: bool = False) -> IntegratorState:
        """Exact flow of this component alone for time `eps`."""
        raise NotImplementedError

    # Optional adaptation hooks, routed by BaseHMC into the sampler hook chain (doc 02).
    def init_params(self) -> Any: return None
    def preprocess_hook(self, sampler, state): return state
    def postprocess_hook(self, sampler, state): return state
```

### Potential Hamiltonians

A potential depends on $q$ only. Its flow is a **kick** — it changes momentum, never position:

$$\Phi^{V}(\epsilon): \quad q \mapsto q, \qquad p \mapsto p - \epsilon\, \nabla_q V(q).$$

The kick consults the gradient cache when `use_cache=True`, and otherwise computes a fresh value-and-gradient and writes it back into the cache:

```python
class PotentialHamiltonian(Hamiltonian):
    def potential(self, q: Array, ctx: HamiltonianContext) -> Array:
        raise NotImplementedError

    def value_and_grad(self, q, ctx):
        return jax.value_and_grad(self.potential)(q, ctx)   # (V(q), ∇_q V(q))

    def energy(self, istate, ctx):
        # use the cached value if present (valid at leaf boundaries); else compute
        return istate.potential_values.get(self.id) or self.potential(istate.q, ctx)

    def flow(self, istate, eps, ctx, use_cache=False):   # kick
        if use_cache:
            g = istate.potential_grads[self.id]          # reuse — no recomputation
            return istate._replace(p=istate.p - eps * g)
        v, g = self.value_and_grad(istate.q, ctx)
        return istate._replace(
            p=istate.p - eps * g,
            potential_values={**istate.potential_values, self.id: v},
            potential_grads={**istate.potential_grads, self.id: g},
        )
```

Two concrete kinds map directly onto the existing model/chart interfaces:

```python
class ModelPotential(PotentialHamiltonian):
    """One model log-density component, e.g. 'log_prior' or 'log_likelihood'."""
    def __init__(self, model, component_name):
        self.model, self._name = model, component_name
        self.id = f"V_{component_name}"
    def potential(self, q, ctx):
        sample = self.model.unpack_coordinate(q, ctx.chart_hyperparams, ctx.chart_indices)
        return -self.model.log_prob_fns[self._name](sample)

class JacobianPotential(PotentialHamiltonian):
    """The chart change-of-variables correction, summed over parameters."""
    def __init__(self, model):
        self.model, self.id = model, "V_jacobian"
    def potential(self, q, ctx):
        # log_jacobian_det takes the coordinate directly (doc 04): cheap, closed-form in q.
        return -self.model.total_log_jacobian_from_coordinate(
            q, ctx.chart_hyperparams, ctx.chart_indices)
```

> **Why separate potentials matter.** `ModelPotential` gradients are expensive (full model autodiff); `JacobianPotential` gradients are cheap and often closed-form. Distinct components let the integrator evaluate them at *different frequencies* (multi-rate splitting) and cache them *independently* (per-id cache slots).

### Kinetic Hamiltonians

A kinetic component supplies the kinetic energy, the **velocity** $\nabla_p T$ (which drives the position drift and the NUTS U-turn criterion), the momentum-refresh distribution, and the random draws it needs. Its flow is a **drift** (for separable kinetics) and ignores `use_cache`:

```python
class KineticHamiltonian(Hamiltonian):
    separable: bool   # True if T depends on p only (∂T/∂q = 0)

    def kinetic(self, istate, ctx) -> Array: ...
    def velocity(self, istate, ctx) -> Array:          # ∇_p T  (physical velocity q̇)
        ...
    def energy(self, istate, ctx):
        return self.kinetic(istate, ctx)

    def flow(self, istate, eps, ctx, use_cache=False):  # drift (separable only)
        assert self.separable
        return istate._replace(q=istate.q + eps * self.velocity(istate, ctx))

    # Momentum refresh: connects to the RNG buffer design (doc 03).
    def make_draw_components(self, dim) -> list["DrawComponent"]: ...
    def sample_momentum(self, draw, ctx) -> Array: ...
```

A drift changes `q`, which **staleness-invalidates** every potential's cache entry. The cache is *not* cleared here (that would cost work and complicate the pytree); instead, validity is tracked statically by the integrator builder via the `cached_gradient` annotation (see below). Stale entries are simply never read, and are overwritten by the next non-cached kick.

The library ships several kinetic components. **The diagonal vs. dense mass-matrix distinction is just a choice of kinetic component:**

```python
class DiagonalQuadraticKinetic(KineticHamiltonian):
    """T = ½ pᵀ diag(m)⁻¹ p.  Mass `m` adapted; lives in ctx.ham_params[self.id]."""
    separable = True
    def kinetic(self, istate, ctx):
        inv_m = ctx.ham_params[self.id].inv_mass
        return 0.5 * jnp.sum(inv_m * istate.p**2)
    def velocity(self, istate, ctx):
        return ctx.ham_params[self.id].inv_mass * istate.p
    def make_draw_components(self, dim):
        return [DrawComponent("momentum", (dim,), jax.random.normal)]
    def sample_momentum(self, draw, ctx):                    # reparameterization (doc 03)
        sqrt_m = jnp.sqrt(1.0 / ctx.ham_params[self.id].inv_mass)
        return sqrt_m * draw.momentum

class DenseQuadraticKinetic(KineticHamiltonian):   # T = ½ pᵀ M⁻¹ p; M via Cholesky; p = L z
    separable = True
class LowRankQuadraticKinetic(KineticHamiltonian): # M = D^½(I + Σⱼ γⱼ vⱼvⱼᵀ)D^½, rank-J (mimcs.hmc.lowrank)
    separable = True                               # mass_mode=None; adapted by LowRankAdaptation (Oja)
class RelativisticKinetic(KineticHamiltonian):     # bounded q̇; non-Gaussian custom draw
    """T = Σᵢ √(mᵢ²cᵢ⁴ + cᵢ²|pᵢ|²).  Velocity c²p/T is capped by the light speed c, so the
    integrator can't shoot off in light tails / funnels (Lu et al. 2017). The "particle"
    structure is set by reshaping the flat momentum to `shape` and choosing `inner_axes`
    (summed inside the √; the rest index particles): inner_axes=() → every coordinate a
    1-D particle; all axes → one particle over the block. Separable, so it uses the
    ordinary leapfrog and HMC/NUTS unchanged. Momentum ~ exp(-T) is refreshed exactly:
    per particle, |p| from a precomputed (u, m) inverse-CDF (tracks the adapting mass),
    direction uniform. Per-particle mass mᵢ lives in ham_params and is score-adapted
    (RelativisticMassAdaptation: SGD on ½(d·log m + |g|²/m) → m = E[|g|²]/d, the per-particle
    ScoreMassAdaptation; |g|²/m → χ²_d, mean d); c is a fixed scalar, which suffices only
    if a centering reparametrization has standardized the coordinates -- and centering is
    opt-in and off by default, so this is [experimental]: a first-class relativistic option
    most likely needs an adaptation for c itself."""
    separable = True
class RiemannianKinetic(KineticHamiltonian):        # [experimental]; doc 07
    """RMHMC: T = ½ pᵀ G(q)⁻¹ p + ½ log det G(q). NON-separable: ∂T/∂q ≠ 0,
    so flow() is not an explicit drift — see implicit integrators below."""
    separable = False
```

> **Naming.** We call $G(q)$ a *position-dependent mass matrix* to keep it distinct from the manifold-typed parameters of doc 04. RMHMC's geometry lives in the **kinetic energy**; the parameter manifolds live in the **coordinate charts**. They are orthogonal and combinable.

### One kinetic per parameter (default); adapted params and hooks

By default `BaseHMC` builds one kinetic component per parameter (block-diagonal mass matrix); cross-parameter coupling is an extra component spanning both blocks. Adapted kinetic parameters (mass matrices, relativistic masses) live in `state.ham_params` and are updated in `postprocess` by adaptation mixins, following the reparameterization principle shared with `chart_hyperparams` (doc 04). `BaseHMC` routes each component's `preprocess_hook`/`postprocess_hook` into the cooperative hook chain (doc 02).

### Separability taxonomy

| Component | Depends on | Flow | Treatment |
|---|---|---|---|
| `PotentialHamiltonian` | $q$ | kick: $p \mathrel{-}= \epsilon \nabla_q V$ | explicit, exact, **cacheable** |
| separable `KineticHamiltonian` | $p$ | drift: $q \mathrel{+}= \epsilon \nabla_p T$ | explicit, exact |
| non-separable `KineticHamiltonian` | $q, p$ | coupled | **implicit** (generalized leapfrog) |

## Axis 2: Integrators

An integrator approximates the flow of the full Hamiltonian over one step of size $\epsilon$ by composing the elementary flows of the components. It operates on `IntegratorState` (reading/writing only the base contract) and must satisfy two invariants so the simple $\Delta H$ acceptance rule is valid:

- **Reversibility** (time-symmetry): `step` followed by momentum negation is an involution. Guaranteed by **palindromic** flow sequences.
- **Volume preservation**: each elementary flow is a shear; compositions are too. Symplecticity holds for the explicit schemes and for the implicit generalized leapfrog.

### Elementary composition and the `cached_gradient` annotation

```python
@dataclass(frozen=True)
class Op:
    target: "Hamiltonian | Integrator"   # a component flow, or a nested integrator
    coeff: float                          # fraction of eps applied to this target
    cached_gradient: bool = False         # reuse the cached ∇V instead of recomputing

class SplittingIntegrator:
    """One step = a palindromic sequence of component / sub-integrator flows."""
    def __init__(self, ops: list[Op]):
        self.ops = ops                    # must read the same forwards and backwards
    def step(self, istate, eps, ctx) -> IntegratorState:
        for op in self.ops:
            istate = op.target.flow(istate, op.coeff * eps, ctx, use_cache=op.cached_gradient)
        return istate
    def flow(self, istate, eps, ctx, use_cache=False):   # so integrators nest as Op targets
        return self.step(istate, eps, ctx)
```

`cached_gradient=True` is meaningful **only on an `Op` whose target is a single `PotentialHamiltonian`** (a kick). On kinetic flows and nested integrators it is ignored (those manage their own caching internally).

**`leapfrog(potentials, kinetics)` takes the whole kinetics list** and builds the palindromic `V…/2 · [T₁/2 … Tₖ₋₁/2 · Tₖ · Tₖ₋₁/2 … T₁/2] · V…/2`. With one kinetic this is the ordinary kick–drift–kick; with several it composes their flows in a palindrome. Crucially, the integrator **never branches on `separable`** — it just calls each component's `flow`, whether that is a separable position drift, an explicit cross-block kick (block RMHMC), or an implicit generalized-leapfrog solve (dense RMHMC). So one integrator subsumes what used to be a separate block-leapfrog; `separable` now only decides whether a component inherits the base drift-`flow` or overrides it.

**Standard leapfrog** with gradient reuse — one expensive gradient per step instead of two:

```python
# Leading kick reuses the cache; trailing kick recomputes and refreshes it.
SplittingIntegrator([
    Op(V, 0.5, cached_gradient=True),   # uses ∇V from init or previous step's trailing kick
    Op(T, 1.0),                          # drift: changes q, invalidating the V cache
    Op(V, 0.5),                          # recomputes ∇V at the new q, stores it
])
```

**Split-potential leapfrog** ($V = V_\text{model} + V_\text{jac}$), each kick cached independently:

```python
SplittingIntegrator([
    Op(V_model, 0.5, cached_gradient=True), Op(V_jac, 0.5, cached_gradient=True),
    Op(T, 1.0),
    Op(V_jac, 0.5), Op(V_model, 0.5),
])
```

### Gradient caching: the validity contract

A cached entry for potential component $C$ is **valid at the current $q$** as long as **no drift (kinetic flow) has executed since the entry was written**. Kicks change only $p$, so they never invalidate any potential cache; drifts change $q$, invalidating every potential cache.

Because the op sequence is a static Python structure, validity is verifiable by inspection at construction time. The builder sets `cached_gradient=True` exactly where the following holds: tracing backward from this kick to the most recent write of $C$'s cache — which is either (a) a prior non-cached kick of $C$ in the same step, (b) the trajectory's initial state (`init_integrator_state` pre-populates all caches), or (c) the trailing kick of the previous step in a repeated loop — **no drift intervenes**.

For standard leapfrog this is immediate: the leading kick reads the cache, the drift invalidates it, the trailing kick rewrites it; across steps the loop preserves the invariant "*$V$ valid at entry*," because the trailing kick of step $n$ and the leading kick of step $n{+}1$ are separated by no drift and act at the same $q$. The reuse is therefore **mathematically exact**, not an approximation.

> **Safety.** A debug mode (`integrator(check_cache=True)`) recomputes each cached gradient and asserts equality with the cached value, catching mis-annotated `Op` lists during development. In production the check is compiled out.

`init_integrator_state` (provided by `BaseHMC`) computes the value-and-gradient of every potential at $q_0$, so all `cached_gradient=True` kicks at the first step are valid.

### Nested / multi-rate (RESPA-style) integrators

**Which component is cheap is now told to us.** `Model.cheap_components` (doc 05) carries the
split: a DSL program's `ModelSpec` labels each `model` component by whether it touches large data
(doc 08), and a hand-written model can declare it. `JacobianPotential` is cheap by construction —
it never touches data — and is not a model component, so it never appears in that set; the
builder should treat it as cheap regardless. An unlabelled model has nothing cheap, so the
builder falls back to plain `leapfrog`. What remains is `RepeatedIntegrator` below and the
selection rule in the sampler factory (`mimcs/factory/build.py`, where `leapfrog` is hardcoded).

Evaluate the expensive model gradient **once** per $N$ evaluations of the cheap Jacobian gradient and the drift — a *nested* splitting,

$$\Phi^{V_\text{model}}\!\Big(\tfrac{\epsilon}{2}\Big)\;\Big[\,\Phi^{V_\text{jac}}\!\Big(\tfrac{\delta}{2}\Big)\,\Phi^{T}(\delta)\,\Phi^{V_\text{jac}}\!\Big(\tfrac{\delta}{2}\Big)\Big]^{N}\;\Phi^{V_\text{model}}\!\Big(\tfrac{\epsilon}{2}\Big), \quad \delta=\tfrac{\epsilon}{N},$$

expressed by composing integrators recursively. **Implemented** as `RepeatedIntegrator` (the inner
loop) and the builder `multirate_leapfrog`:

```python
cheap, expensive = split_potentials(model, potentials)   # the Jacobian is always cheap
integrator = multirate_leapfrog(cheap, expensive, kinetics, n=4)

# which is
inner = leapfrog(cheap, kinetics)                        # the ordinary palindrome, at eps/n
SplittingIntegrator([Op(V, 0.5, cached_gradient=True) for V in expensive]
                    + [Op(RepeatedIntegrator(inner, n), 1.0)]
                    + [Op(V, 0.5) for V in reversed(expensive)])
```

`RepeatedIntegrator` and `SplittingIntegrator` both expose `flow`, so they compose to arbitrary
depth; $N$ is a static int, so the inner loop is a fixed-trip `fori_loop` (invariant 5).
`RepeatedIntegrator` also implements `step` / `integrate` / `init_integrator_data`, so it is a
complete integrator and can stand alone or serve as a `LineSearchIntegrator` base; it delegates
`emits_step_size_proxy` to its inner and rejects a *randomized* inner (there is no per-sub-step
randomness to thread). It deliberately defines **no** `value_and_grad`, which is what makes an
enclosing `SplittingIntegrator` count it as zero while the inner steps accumulate their own cost.

`split_potentials(model, potentials)` does the grouping: the components named in
`Model.cheap_components`, plus `JacobianPotential` (cheap by construction — it never touches
data, and being no model component it can never be *named* cheap). A degenerate split — nothing
cheap, or nothing expensive — is a `ValueError` rather than a silent fallback, because the layer
that has the model and must record a reason is the factory (doc 09), not the integrator library.
`n = 1` is legal and reproduces `leapfrog(expensive + cheap, kinetics)` op for op.

#### Cache validity of the multi-rate op list

The only ops that move $q$ are the inner drifts, which gives the contract above four cases:

| cached kick | last write of that cache | in between | verdict |
|---|---|---|---|
| leading expensive, first step | `init_integrator_state` at $q_0$ | the other leading kicks | valid |
| leading expensive, later steps | the previous outer step's trailing expensive kick | kicks only | valid |
| leading cheap, inner iteration 1 | the previous outer step's last trailing cheap kick | the two expensive kick groups | valid |
| leading cheap, iterations 2..n | the previous inner iteration's trailing kick | nothing | valid |

Every trailing kick recomputes, so at each outer-step **endpoint** every potential's value *and*
gradient is fresh — which is what NUTS needs to resume from either end, and what `total_energy`
needs, since `PotentialHamiltonian.energy` trusts a cached value unconditionally. (Mid-inner-loop
the expensive cache is stale; nothing reads it there.) **All of this holds only because no
kinetic appears in the outer list**: putting one there would separate the trailing and leading
expensive kicks by a drift and silently invalidate the annotation.

#### What it costs, and what we can and cannot measure

Per outer step: `#expensive + n * #cheap` gradients, against `#expensive + #cheap` for plain
leapfrog at the same $\epsilon$. The saving is *wall-clock*, and only when the expensive gradient
dominates it while the cheap part limits the stable step size.

**The `grad_evals` diagnostic cannot show that saving**, because it counts every gradient as one,
whatever it cost. Measured on a 2-D Gaussian split into two precision components (8000 draws,
`n=4`, one seed): ESS per 1000 gradients was 191 for leapfrog against 24 for multi-rate with
similar-scale components, and 192 against 43 when the *cheap* component was 400× stiffer — i.e.
multi-rate loses by that counter even in the case built to favour it. Against a genuinely
expensive likelihood (a 20000×8 design matrix) with a stiff cheap prior, wall-clock per draw was
**identical** (1.45 s for 1000 draws either way) while the counter still called multi-rate 2.3×
worse. One seed, so the ~7% ESS/s edge measured there is not a result — but the *direction* of
the discrepancy is structural, not noise.

**Open question:** count gradients *per potential* (a dict keyed by potential id) instead of
summing heterogeneous ones into a single scalar. That would also fix
`LineSearchIntegrator._grad_evals_by_level`, whose `2 + 2 Σ T_i + T_j` counts *base steps* and so
already undercounts for any base costing more than one gradient per step — a multi-rate base most
of all.

#### Why multi-rate is off by default: the `diamonds` study

`diamonds_split` is the natural candidate — a 5000×24 design matrix, so the likelihood gradient is
the only cost that matters. Eight seeds, 1000 warmup + 1000 draws, x64, comparing min-ESS per
*expensive* gradient (`tests/experiments/diamonds_multirate.py`): **no arm differs from the
baseline**, paired ratios 0.91–1.18 with confidence intervals spanning 1, and the non-monotonicity
in `n` is the signature of noise — seed-to-seed spread within one arm is 2×.

The mechanism is the flat step size: 0.526–0.538 across every arm, identical tree depth and leaves
per draw. Multi-rate changed the dynamics not at all, because the step size is not prior-limited.
Measuring the two components' curvature at a warmed-up point settles it — the largest Hessian
eigenvalue of $-\log p$ is **1.00** for the prior (a `normal(0, 1)`) and **1.06 × 10⁶** for the
likelihood. Sub-stepping the prior four times refines a term contributing one part in a million of
the stiffness.

The lesson generalizes past this problem. A tight prior is tight in *location* — here the posterior
mean sits ~6 prior sds out, which is real prior–likelihood tension — but multi-rate splits on
**curvature**, not on tension. And in a Bayesian model with plenty of data the curvature is
dominated by the likelihood *by construction*: the prior contributes O(1) precision while the
likelihood contributes O(N). This is the reverse of the molecular-dynamics setting RESPA was
designed for, where the cheap forces are the fast (stiff) ones.

**So: multi-rate helps only when the cheap component is also the stiff one.** Candidates worth
testing are a hierarchical prior whose funnel neck is the stiff geometry with an expensive
likelihood over it, or a strongly informative prior against weak data — not a regression with
5000 rows. Until such a case is measured, the integrator is correct, cheap to keep, and off by
default for every model that does not declare a split.

#### What the `reg_horseshoe` learned-metric studies settled

Thirteen studies over `reg_horseshoe` and the funnel — most of them negative — are recorded in
full, with their scripts and data, in `tests/experiments/writeups/reg_horseshoe_learned_metric.md`.
Three of their conclusions bear on the design and are kept here:

* **The RESPA mechanism works; the economics usually do not.** On `reg_horseshoe` the split does
  what it claims — the expensive likelihood gradient is evaluated once per `n` cheap sub-steps —
  but the saving never repays the extra cheap steps, for the same curvature reason the `diamonds`
  study gives above.
* **A frozen chain reports a perfect ESS.** A coordinate that never moves has zero variance and so
  an ESS equal to the draw count; several arms looked excellent until the *distinct-draw count*
  was checked. Any comparison table over these problems needs a frozen-coordinate column, and this
  is why `evaluate` grew one.
* **Coordinate-wise gradient clipping is load-bearing**, not a refinement: clipping each coordinate
  against its own running threshold rather than the block against a shared one took the freeze rate
  from 50% to 12.5% over 8 seeds. The `(d,)` `log_clip` vector in `_ScoreBlock` and
  `MetricAdaptation` is that change; see `docs/design/09` for the adaptation side.

### Higher-order integrators

The 4th-order Yoshida integrator composes three leapfrog steps with $w_1 = 1/(2 - 2^{1/3})$, $w_0 = 1 - 2w_1$ — no new machinery:

```python
leap = SplittingIntegrator([Op(V, 0.5, cached_gradient=True), Op(T, 1.0), Op(V, 0.5)])
yoshida4 = SplittingIntegrator([Op(leap, w1), Op(leap, w0), Op(leap, w1)])
```

### Within-orbit adaptivity: `LineSearchIntegrator` (WALNUTS)

WALNUTS's within-orbit step-size adaptivity lives entirely in axis 2, so it is an
integrator that **wraps a base integrator** (leapfrog) as a component and uses it for the
forward and the reversibility-preserving backward sub-steps. Each macro `step` refines the
discretization until the energy error over the step is within a budget. Composing it with
fixed-length `HMC` gives WAL-HMC; with `NUTS` (each leaf an adaptive macro step) gives
WAL-NUTS — no sampler changes (NUTS just folds `leaf.log_weight` into the leaf weight).

```python
class LineSearchIntegrator:                    # base integrator + schedule + thresholds
    def step(self, istate, eps, ctx):
        L_f, cand, div_f = self._line_search(istate,  eps, ctx)   # coarsest level err<=delta_j
        L_b, _,    div_b = self._line_search(cand,    -eps, ctx)   # required level backward
        L = max(L_f, L_b)                                          # symmetric -> reversible
        z = self._integrate_level(istate, L, eps, ctx)            # re-integrate at level L
        return z._replace(log_weight=z.log_weight + where(div_f|div_b, -inf, 0.0))
```

The **schedule** is an arbitrary list of `(h_j, T_j)` levels (`T_j` base steps of relative
size `h_j`, level 0 coarsest) with a per-level energy-error budget `delta_j`. Classical
WALNUTS halves the step and doubles the count (`h_j T_j` constant); but `h_j T_j` need not
be fixed, so a refinement may also *shorten* the integration time — useful when stiff
(funnel) geometry is localized. The reversibility weight (`-inf` on a true divergence)
accumulates into `log_weight`; the macro `step_size` sets the coarsest time granularity and
cannot be adapted to *acceptance* (refinement keeps the error in budget regardless, so
acceptance is ~1 whatever the step). It **is** adapted, from the integrator's coarse-level
**proxy** acceptance, by `LineSearchStepSizeAdaptation` — which *replaces* `RobbinsMonroStepSize`
rather than joining it (it subclasses it, so listing both is an MRO error); the sampler factory
performs that swap automatically on `integrator.emits_step_size_proxy`. On Neal's funnel one orbit traverses the neck (tiny steps) and
mouth (large steps), cutting divergences by ~40x versus fixed-step NUTS. This
`LineSearchIntegrator` is the **deterministic (WALNUTS-D)** variant: the micro-step length is
concentrated on the single coarsest valid level, and a forward/backward level disagreement is
*invalidated* (`-inf`).

**What it buys, and what it costs.** On `reg_horseshoe` — the stiffest problem in the repertoire,
where plain leapfrog with the best available learned metric still gave a mean 89% divergence rate
and froze on 1 seed in 8 — WALNUTS removed divergence *entirely*: 0.000 on all 8 seeds, accept
0.80–0.91, 8/8 healthy against leapfrog's 2/8, median ESS 3–20× higher. The cost was two orders of
magnitude worse than predicted: ~2.1–3.3M sampling gradient evaluations against leapfrog's ~4k
median, i.e. **400–800× more expensive per unit of ESS** on the seeds where leapfrog worked at all.

So the value here is reliability, not efficiency: WALNUTS turns an unreliable coin flip into a sure
thing, and leapfrog wins on ESS-per-gradient whenever it happens to work. That is the honest way to
choose between them, and it is why the factory does not select `line_search` on its own. Full study
in `tests/experiments/writeups/reg_horseshoe_learned_metric.md`.

#### `MarkovianLineSearchIntegrator` — a randomized alternative (no invalidation)

**It needs a host that gives it per-step randomness, and only NUTS does.** NUTS calls `step` once
per leaf and declares a per-leaf coin array keyed on the integrator's `n_rng_per_step`; a
fixed-length base integrates a whole trajectory in one `integrate` call and has nowhere to put
coins, so the chain runs at the coarsest valid level — exactly WALNUTS-D, with no indication.
`RandomizedHMC` is no different: it randomizes the trajectory *length*, not the refinement. That
capability is declared as `BaseHMC.supplies_integrator_rng` (False; `BaseNUTS` overrides to True),
and the factory refuses the combination rather than deliver a different algorithm silently. A
randomized fixed-length integrator (WAL-HMC) would flip that flag.

An alternative chooses the refinement level by a **coarse-to-fine Markov chain** instead of the
coarsest valid level, trading a *more spread-out* micro-step-length distribution for the ability
to **never invalidate** a step (except on a genuine non-finite energy). At level `j < n`: if
`err_j > delta_j` take the finer level (**forced**); else take it with probability `p`
(**unforced**) or **stop** at `j` with probability `1 - p`. The finest level `n` always stops
(infinite budget). Because stopping at the chosen level `J` always has positive reverse
probability — the backward error at `J` equals the forward one (leapfrog is reversible, energy
error is symmetric), hence within budget, or `J = n` — the move is reversible **in general**, so
no step needs invalidating and steps that refine all the way to the finest scale are *accepted*.

The reversibility weight follows from detailed balance under the codebase's convention
(`log_weight` a reward added to `-H`; samplers fold it as `exp(H0 - H1 + Δlog_weight)`):

```
C(z -> z') = log P_rev(J | z') - log P_fwd(J | z) = (u_rev - u_fwd) · log p
```

where `u_fwd` is the number of *unforced forward* moves and `u_rev` the number of coarser levels
`j' < J` whose *backward* error is within budget. So **each unforced forward move contributes
`-log p`** (a boost — the state was proposed with probability `~ p`) and **each backward
within-budget coarser level contributes `+log p`** (a penalty); the `(1 - p)` stop factors cancel
between the two directions. `-inf` is reserved for non-finite energy alone. A clean consequence,
unit-tested, is the antisymmetry `C(z -> z') = -C(z' -> z)`.

Implementation: the **forward** pass consumes `n_levels` uniform coins (one per level), so the
integrator is the first *randomized* one — NUTS declares a per-leaf `line_search` draw
(`(J, max_subtree, n_levels)`) **only when the integrator requests it** (via `n_rng_per_step`),
keeping every existing seed stream byte-identical; the **backward** pass is deterministic
accounting of `log P_rev`. `NUTS` and `SimpleNUTS` thread the per-leaf coins identically (the
bit-identical oracle still holds). Builder: `mwal_nuts(..., p=0.5)`.

Illustrative single-seed run on Neal's funnel (`scale=3`, doubling schedule, `delta=0.8`,
`p=0.5`; ~40 min multi-seed study deferred with the step-size/schedule arc): fixed-step NUTS
diverged 519/4000 and reached only `v_min ≈ -4`; both WALNUTS variants reached `v_min < -9` with
0–2 divergences (WALNUTS-D 0, Markovian 2 — within seed noise). The ~250x divergence reduction is
the robust effect; the 0-vs-2 gap between the variants is not meaningful at one seed.

#### Step-size adaptation via a coarse-level *proxy energy*

Ordinary acceptance-driven step-size adaptation (`RobbinsMonroStepSize`, targeting the real
acceptance) **fails by construction** for the line-search integrators: they refine until the energy
error is within budget, so the real acceptance is ~1 regardless of the macro step size `h`, and `h`
runs away upward until the integrator is pinned to its finest scales. Nor can we adapt on the raw
coarsest-level energy error — in exactly the stiff geometry where WALNUTS earns its keep, that error
is huge, and *forced* refinement (the correct response) would wrongly drag `h` down.

Instead the integrator emits a **proxy energy** whose per-macro-step increment transports the actual
energy change to a coarsest-level-equivalent scale, then re-expresses it over the macro step's *actual*
integration time. The mean energy error at level `j` is `~ h² h_j²`, so transporting the mean error
`ΔH/(T_j h h_j)` (energy change ÷ integration time) to the coarsest level divides by `h_j²`; multiplying
back by the actual integration time `T_j h h_j` gives

```
proxy_ΔH_macro = (H_end − H_start)_macro / h_j²   ≈  h²,
```

the coarsest-level-equivalent energy change — **growing with `h`** (the restoring signal step-size
adaptation needs) and, crucially, **accounting for the shortened integration time** on finer levels.
It coincides with the earlier `ΔH/(T_j h_j³)` exactly when the integration time is constant
(`T_j h_j = 1`, the classical doubling schedule) and differs only for schedules that shorten the
integration time on finer levels — the earlier form implicitly used the *coarsest* integration time
`h` rather than the actual `T_j h h_j`, over-counting by `1/(T_j h_j)`. This is the more principled
form (a normalization refinement), but on its own it is empirically neutral. The `(2⁻ʲ,1)` step-size
**blowup** on stiff cold starts (`h` running to hundreds/thousands with near-total divergence) is cured
instead by a **finest-level divergence signal**: when forced refinement drives the coarsest valid level
all the way to the finest — no coarser level was within budget, i.e. the coarsest step is simply too
large — the proxy emits `+∞` energy, so the proxy acceptance collapses to 0 and `h` is pulled down. It
is self-scaling (it stops firing once `h` shrinks enough that a coarser level is valid again, so it
needs no tuned cap) and inert on easy targets (the finest is rarely reached). On the Poisson this
tames `h` from ~210/4848 to ~5/18 with near-zero divergences and lifts classical doubling to 4/4
reached. It also **clarified an artifact**: the earlier apparent `(2⁻ʲ,1)`·Markovian "win" (reaching
the mode ~4× cheaper) came *through* the blown-up `h` (chaotic large jumps that happened to land near
the mode); with a healthy step size `(2⁻ʲ,1)` merely creeps (deep refinement → tiny time per leaf), so
doubling·deterministic is the clear Poisson winner too. The proxy is evaluated at the **coarsest valid
(forward) level `j*`** — the refinement that was actually *needed* (the first within budget, or the
finest) — and
its endpoint's energy change, *not* the realized level. This matters for the randomized (Markovian)
variant: normalizing by the realized level would credit its *unforced* extra refinement as if it
were necessary, making the proxy optimistic and inflating `h`; using `j*` makes both variants agree.
`LineSearchStepSizeAdaptation` drives `h` toward `target_accept` using the proxy acceptance
`min(1, exp(−proxy_energy))`; the *real* energy still governs the actual NUTS multinomial / HMC
Metropolis accept. The proxy energy accumulates in `integrator_data["proxy_energy"]` (cumulative from
the trajectory start); NUTS averages the per-leaf proxy acceptance (a `sum_proxy_accept` accumulator
mirroring `sum_accept`), HMC reads the endpoint's cumulative value. A `mean_refinements()` diagnostic
(mean *realized* level) rides in `integrator_data` too.

This is **on by default** in both places a WALNUTS sampler gets built: `make_sampler` swaps it in
whenever the integrator emits a step-size proxy, and the `wal_hmc` / `wal_nuts` / `mwal_nuts` test
builders default `adapt_step_size=True` to match — so the suite exercises the configuration the
factory actually ships. `adapt_step_size=False` pins the macro step instead. (The builders
defaulted to *off* until the adaptation was validated, which for a while left every sampling test
in `test_walnuts.py` / `test_mwalnuts.py` running a configuration nobody uses.) Turning it on
changed no pinned margin in those files, and adaptation demonstrably engages: from a 0.5 start the
step settles at 0.70–0.82 on a benign Gaussian and 0.17–0.44 on the funnel's stiff neck, over four
seeds, never overlapping.

Validation on a benign correlated Gaussian: the deterministic WAL-NUTS proxy holds `h` at a finite
plateau with shallow refinement, whereas naive real-acceptance adaptation runs `h` several-fold
higher into deep refinement. With the *realized*-level normalization the Markovian variant inflated
`h` ~1.9× the deterministic one (single seed); the coarsest-valid-level normalization removes that —
over 8 seeds `h` = 0.64 ± 0.04 (Markovian) vs 0.85 ± 0.23 (deterministic), ratio 0.76, with sampling
correctness intact.

On **Neal's funnel** (dim 2, scale 3, x64, 8 seeds) the picture is consistent and encouraging: no
inflation (ratio 0.93) and no collapse to the finest scale (mean refinement stays shallow, 0.7–1.6).
The proxy is *more conservative* than a hand-set step — it settles `h ≈ 0.22` (vs a fixed 0.5) to
hit `target_accept = 0.8` across the mouth's heterogeneous `e^v` scales — and that conservatism cuts
divergences ~20× (≈0.1 vs 3.5 per run) and reaches ~1.5 deeper into the neck, at a ~15% raw-ESS cost
vs the fixed step. So the proxy auto-tunes to a *safer* operating point; a lower `target_accept` is a
natural knob to trade some of that safety back for ESS. The energy-error-threshold `δ_j` / schedule
study (and tuning `target_accept`) is the follow-on empirical work this machinery enables.

#### Schedule study: the classical doubling schedule wins on the funnel

Using the gradient-evaluation diagnostic (`docs/design/02`), a first schedule comparison on Neal's
funnel (dim 2, scale 3, x64, 4 seeds, factory-default sampler with a line-search integrator + proxy
step-size adaptation, efficiency = min-ESS per 1000 gradient evals). Three `(h_j, T_j)` schedules ×
three integrator configs; the threshold is scheduled `δ_j = ε₀·h_j·T_j` (reduce the raw-error budget
by the integration-time decrease). Result: the **classical doubling `(2⁻ʲ, 2ʲ)` (constant
`h_j T_j = 1`) with the deterministic integrator wins decisively** — efficiency 2.5, vs 1.3 for
`(3⁻ʲ, 2ʲ)` and 0.56 for `(2⁻ʲ, 1)` (≈4.5×), all deterministic; 0 divergences everywhere. The
shorter-integration-time schedules are *worse*: because they cover tiny time per leaf in the neck,
the NUTS trajectory stalls there — ≈2× the leaves (15 vs 7) and lower ESS — so *progress-per-leaf*,
not deep-refinement grad-cost, dominates on this (mild) funnel. The Markovian variant is pure
overhead here (deterministic ≫ p=⅕ ≫ p=½, up to 13× worse): unforced refinement burns gradients and
the funnel needs no invalidation-avoidance.

A second problem reverses two of these on a *stiff cold start* — the Poisson random-effects model
(`mu*=10`, 51-d; cumulative grad-evals to reach the mode from `U(-2,2)`, diagonal score-mass, 4
seeds). Here the cheaper-deep `(2⁻ʲ, 1)` (`2j+1` grad-evals) with the **Markovian** integrator reaches
the mode ~**4× cheaper** than robust doubling (54k vs 208k grad-evals, both 3/4 reliable) — cheap deep
refinement creeps through the stiff region cheaply. And the Markovian **no-invalidation property is
essential** here (opposite of the funnel): the *deterministic* shorter schedules fail catastrophically
(0/4, ~1000 divergences, frozen) because when cold-start stiffness exceeds the finest level the
deterministic line search invalidates (`−∞`), destabilizing the step-size proxy so `h` blows up; the
Markovian never invalidates and creeps through. Two open issues this exposed: the step-size proxy
adaptation breaks on stiff cold starts (`h` blows up; the line-search refinement compensates), and the
factory-default *dense* eta score-mass collapses to non-PD there (`docs/design/09`; diagonal is stable).
A third problem, the **2PL IRT** posterior (144-d, three funnels; ESS per grad-eval, factory default
= dense hyper blocks + low-rank theta + dense a/b), shows the same lean in the mixing regime: a
seed-dependent bistability (good regime ESS 700–1500 vs a stuck regime where every trajectory
diverges) that the schedule controls — `(2⁻ʲ, 1)` reaches the good regime most reliably (2–3/4 seeds,
best with Markovian), classical doubling least (0–1/4). So across the three: **classical
constant-time doubling·deterministic wins raw efficiency on the mild funnel, but on the geometrically
hard problems (stiff cold start, funnel mixing) the shorter-integration-time `(2⁻ʲ, 1)`·Markovian is
more robust/efficient** — and the Markovian no-invalidation property, pure overhead on the easy
funnel, earns its keep exactly where the geometry is hard. The clear lever to make `(2⁻ʲ, 1)`·Markovian
broadly competitive is fixing the step-size proxy's cold-start `h`-blowup (its instability is what
tips seeds into the bad regime on the hard problems).

### Implicit flow for non-separable kinetics (RMHMC) **[experimental]**

When the kinetic is non-separable (`separable == False`), `T`'s flow moves both `q` and `p`, so it is not an explicit drift. Instead of a *separate* implicit integrator, the implicit work lives in the **kinetic's `flow`**: a generalized (implicit) leapfrog of `T` alone, with fixed-point iterations (static count → shape-stable). The surrounding `leapfrog` splitting handles the potentials as ordinary explicit kicks — valid because `∇_q V` is constant within the implicit solve. So RMHMC reuses the standard `SplittingIntegrator`; no dedicated integrator class is needed (see `docs/design/07_riemannian_hmc.md`).

```python
class RiemannianKinetic(KineticHamiltonian):
    separable = False
    def flow(self, istate, eps, ctx, use_cache=False):
        # implicit generalized leapfrog of T alone (solver = Picard / Anderson):
        # 1. implicit momentum half-kick:  p ← p − (ε/2) ∇_q T(q, p)        [fixed-point in p]
        # 2. implicit position drift:      q ← q + (ε/2)(∇_p T(q,p)+∇_p T(q',p)) [fixed-point in q']
        # 3. explicit momentum half-kick at the new q
        ...
```

Only the kinetic's flow differs; the `PotentialHamiltonian`/`KineticHamiltonian` interfaces and the `SplittingIntegrator` are unchanged. This is why axes 1 and 2 stay independent.

## Axis 3: Large-Scale Structure (`BaseHMC`, `HMC`, `RandomizedHMC`, `NUTS`)

### `BaseHMC`

`BaseHMC` is a base sampler class (doc 02). It **holds the components** and provides shared services; it does **not** decide trajectory length or selection. It holds a **list of kinetics** — one per coordinate block — exactly as it holds a list of potentials, and aggregates them (`Σ Tⱼ` energy, the assembled velocity/momentum). A single whole-space kinetic is a one-element list; a block-diagonal mass is several blocks; a relativistic-on-one-block sampler is just a relativistic kinetic in the list beside a diagonal one. (A single `kinetic=`/`metric=` is accepted and wrapped into a one-element list.) A kinetic's coordinates are a **list of slices** (`kinetic.slices`, `None` = whole space) which may be **non-contiguous**, so one block can fuse scattered parameters; the base provides `_gather(x)` (concatenate the slices) and `_scatter(base, vals)` (successive `.at[s:e].set`), and `velocity_into`/`sample_into`/`energy` are written in terms of them. It also owns `init_integrator_state`, which seeds the gradient caches.

```python
class BaseHMC(BaseSampler):
    integrator_state_class = IntegratorState     # overridden by subclasses that extend it

    def __init__(self, model, *, potentials, kinetics, integrator, **kw):
        self.potentials = potentials             # list[PotentialHamiltonian]
        self.kinetics = kinetics                 # list[KineticHamiltonian], one per block
        self.integrator = integrator
        super().__init__(model, **kw)

    # --- shared services: aggregate the potentials and kinetic components ---
    def total_energy(self, istate, ctx):
        """H = Σ Vᵢ(q) + Σ Tⱼ(q,p), using cached potential values (valid at leaf boundaries)."""
        v = sum(istate.potential_values.values())
        return v + sum(k.energy(istate, ctx) for k in self.kinetics)

    def sample_momentum(self, draw, q, ctx):     # each block fills its slice
        p = jnp.zeros(self.model.coord_dim)
        for k in self.kinetics: p = k.sample_into(p, draw, q, ctx)
        return p

    def kinetic_velocity(self, istate, ctx):     # ∇_p Σ Tⱼ; drives the U-turn test
        v = jnp.zeros_like(istate.q)
        for k in self.kinetics: v = k.velocity_into(v, istate, ctx)
        return v

    def make_draw_components(self, model, **kw):  # id-namespaced draws, no collisions
        comps = [c for k in self.kinetics for c in k.make_draw_components(model.coord_dim)]
        comps.append(DrawComponent("accept_threshold", (), jax.random.uniform))
        return comps

    def build_trajectory_and_select(self, state, ctx) -> IntegratorState:
        """Subclass responsibility. Returns the chosen base IntegratorState."""
        raise NotImplementedError
```

The kernel assembles the context and delegates:

```python
def kernel(self, state):
    ctx = HamiltonianContext(state.chart_hyperparams, state.chart_indices, self._ham_params(state))
    chosen = self.build_trajectory_and_select(state, ctx)          # base IntegratorState
    new_sample = self.model.coordinate_to_sample(chosen.q, ctx.chart_hyperparams, ctx.chart_indices)
    new_log_prob = self.model.log_prob_flat(new_sample, ctx.chart_hyperparams, ctx.chart_indices)
    return state._replace(coordinate=chosen.q, sample=new_sample,
                          log_prob=new_log_prob, momentum=chosen.p)
```

### `HMC` — fixed-length trajectory (base `IntegratorState`)

```python
class HMC(BaseHMC):
    integrator_state_class = IntegratorState     # no extension needed
    def __init__(self, *a, n_leapfrog=20, **kw):
        self.n_leapfrog = n_leapfrog
        super().__init__(*a, **kw)

    def build_trajectory_and_select(self, state, ctx):
        p0 = self.refresh_momentum(state, ctx)
        istate0 = self.init_integrator_state(state.coordinate, p0, ctx)
        H0 = self.total_energy(istate0, ctx)
        proposed = jax.lax.fori_loop(
            0, self.n_leapfrog, lambda _, s: self.integrator.step(s, state.step_size, ctx), istate0)
        H1 = self.total_energy(proposed, ctx)
        # general acceptance; for leapfrog log_weight stays 0 and this is min(1, e^{-ΔH})
        log_alpha = H0 - H1 + (proposed.log_weight - istate0.log_weight)
        accept = state.rng_draw.accept_threshold < jnp.exp(log_alpha)
        return jax.tree.map(lambda a, b: jnp.where(accept, a, b), proposed, istate0)
```

### `MALT` — accumulating into the base `log_weight` *(planned, not implemented)*

**Not implemented.** It is kept here because it is what the base `log_weight` field was designed
for, and so explains a field that does ship. MALT (Metropolis-Adjusted Langevin Trajectories) runs
a leapfrog trajectory with **partial momentum refreshment between steps**, accumulating a log-weight used in the end-of-trajectory correction. This accumulator is exactly the base `log_weight` field, so MALT uses the base `IntegratorState` directly — no extension needed. The difference from WALNUTS is only *where* the weight is written: MALT accumulates it in the sampler loop *between* integrator steps, WALNUTS *inside* the integrator step; both target the same base field.

```python
class MALT(BaseHMC):
    integrator_state_class = IntegratorState     # base contract suffices
    def __init__(self, *a, n_steps=20, friction=0.5, **kw):
        self.n_steps, self.friction = n_steps, friction
        super().__init__(*a, **kw)

    @classmethod
    def _extra_draw_components(cls, model, n_steps=20, **kw):
        # one fresh Gaussian noise vector per step for the partial refreshment
        return [DrawComponent("ou_noise", (n_steps, model.coord_dim), jax.random.normal)]

    def build_trajectory_and_select(self, state, ctx):
        p0 = self.refresh_momentum(state, ctx)
        istate = self.init_integrator_state(state.coordinate, p0, ctx)   # log_weight = 0

        def body(i, s):
            s = self.integrator.step(s, state.step_size, ctx)        # advances q, p, caches
            p_new, dlw = self._partial_refresh(s.p, ctx.ham_params, state.rng_draw.ou_noise[i])
            return s._replace(p=p_new, log_weight=s.log_weight + dlw)  # accumulate weight

        traj = jax.lax.fori_loop(0, self.n_steps, body, istate)
        accept = state.rng_draw.accept_threshold < jnp.exp(traj.log_weight)
        return jax.tree.map(lambda a, b: jnp.where(accept, a, b), traj, istate)
```

### `NUTS` — compositional extension (tree of endpoints)

NUTS builds an adaptive trajectory by recursive tree doubling, using the **same** `integrator.step` as the leaf operation and the **same** `total_energy`/`velocity`. Its trajectory state is a *compositional* extension of `IntegratorState`: it embeds base `IntegratorState`s for the two endpoints (the resume points for further integration) and the currently selected proposal, plus summary fields.

> **Two implementations (`BaseNUTS` + subclass).** The shared machinery — outer doubling loop, U-turn test, multinomial selection, reversible divergence, kernel/diagnostics — lives in `BaseNUTS`; the variants differ only in how one subtree is built (`_build_subtree`):
> - **`NUTS`** (canonical, `mimcs/hmc/nuts.py`): memory-efficient. For the subtree U-turn checks it keeps only a per-level **checkpoint** of the *velocity* and cumulative momentum at each open subtree's left endpoint — O(`max_tree_depth`) state, not O(2^depth). Storing the velocity (all the U-turn needs) also reduces per-leaf velocity computations from O(depth) to one. The read/write checkpoint levels are the contiguous ranges `1..ntz(n+1)` / `1..ntz(n)`, looped over directly.
> - **`SimpleNUTS`** (`mimcs/hmc/simple_nuts.py`): stores every leaf of the subtree and indexes them directly. Simpler and obviously correct; kept as a **reference oracle**.
>
> Both yield the same transition kernel — in fact, given the same RNG they trace the *bit-for-bit identical* trajectory (an exact-match test guards this). The efficient version is ~1.1× faster at low dimension and ~1.8× at moderate dimension, with much smaller working memory (enabling deeper trees / higher dimension without blowing up).

```python
class NUTSIntegratorState(NamedTuple):
    left: IntegratorState      # leftmost endpoint  (q, p, caches) — resume point for backward growth
    right: IntegratorState     # rightmost endpoint (q, p, caches) — resume point for forward growth
    proposal: IntegratorState  # currently selected sample
    momentum_sum: Array        # Σ p over the trajectory, for the generalized U-turn test
    log_weight: Array          # log Σ exp(−H) over the trajectory (multinomial weights)
    depth: Array               # current tree depth (diagnostic)
    turning: Array             # bool: a sub-trajectory made a U-turn
    diverging: Array           # bool: |ΔH| exceeded the divergence threshold
```

Tree doubling:

- **Direction** for each doubling is drawn from `rng_draw.tree_uniforms`. The relevant endpoint (`left` or `right`) is handed to `integrator.step` as a plain base `IntegratorState`; the new leaf is folded back into the tree state. Because the endpoints **store their gradient caches**, integration resumes from either end with no gradient recomputation — the primary motivation for caching gradients in `IntegratorState`.
- **No-U-turn termination** uses the velocity from the kinetic component (correct under any metric — the *generalized* / Betancourt criterion). For a sub-trajectory with extreme endpoints carrying momenta $p^-, p^+$ and **summed momentum** $\rho = \sum p$ over that sub-trajectory:
  $$\langle\, \rho,\; \nabla_p T(p^-) \,\rangle \le 0 \quad\text{or}\quad \langle\, \rho,\; \nabla_p T(p^+) \,\rangle \le 0,$$
  i.e. `self.velocity(...)` dotted with the momentum sum. The summed momentum $\rho$ is essential: it is what composes correctly across merged subtrees. The Euclidean form $\langle M^{-1}p,\,\Delta q\rangle$ with the position difference $\Delta q = q^+ - q^-$ is only equivalent for a *single* leapfrog leg (where $M^{-1}\rho \approx \Delta q$) and is **incorrect for merged subtrees** — do not use it. Implemented in `mimcs/hmc/nuts.py::NUTS._is_turning`.
- **Divergence** uses the **reversible** test $\max H - \min H > \tau$ over all states in the trajectory (default $\tau = 1000$), tracked as running min/max — not $H$ relative to the start (the start-relative test, as in Stan, is not exactly reversible, though the difference is negligible in practice). Energies come from cached `potential_values` + kinetic, no recomputation.
- **Selection** is multinomial across the trajectory weighted by $e^{-H}$ via `logaddexp` (unbiased progressive within a subtree, biased toward the newer subtree at each merge). The per-leaf weighting is isolated in `_leaf_log_weight` so the original slice-sampler variant can be added later by overriding it.

The iterative tree-doubling, generalized U-turn, and biased progressive multinomial follow the established implementations in **NumPyro** and **Blackjax**.

```python
class NUTS(BaseHMC):
    integrator_state_class = IntegratorState   # the *leaf* type the integrator advances
    def __init__(self, *a, max_tree_depth=10, **kw):
        self.max_tree_depth = max_tree_depth
        super().__init__(*a, **kw)

    @classmethod
    def _extra_draw_components(cls, model, max_tree_depth=10, **kw):
        return [DrawComponent("log_slice", (), jax.random.uniform),
                DrawComponent("tree_uniforms", (max_tree_depth,), jax.random.uniform)]

    def build_trajectory_and_select(self, state, ctx):
        p0 = self.refresh_momentum(state, ctx)
        leaf0 = self.init_integrator_state(state.coordinate, p0, ctx)
        tree = NUTSIntegratorState(left=leaf0, right=leaf0, proposal=leaf0,
                                   momentum_sum=p0, log_weight=-self.total_energy(leaf0, ctx),
                                   depth=jnp.array(0), turning=jnp.array(False),
                                   diverging=jnp.array(False))
        # ... recursive doubling: extend `left`/`right` via integrator.step, test U-turn with
        #     self.velocity, update proposal by multinomial weight, until turning/diverging/max depth.
        return tree.proposal                      # a base IntegratorState
```

Note that `NUTS.integrator_state_class` is the **base** `IntegratorState`: that is the *leaf* type handed to the integrator. The `NUTSIntegratorState` is the *trajectory* type that NUTS itself manages, composing leaves. This makes explicit that the integrator never sees NUTS-specific fields.

### The central guarantee

`HMC`, `RandomizedHMC`, and `NUTS` consume only `integrator.step`, `total_energy`, `velocity`, `refresh_momentum`, and `init_integrator_state`. Therefore **any** combination of components (axis 1) and integrator (axis 2) works with **any** large-scale structure (axis 3).

## State Additions for HMC

Building on `BaseSamplerState` (doc 01), the persistent sampler state stores the post-kernel momentum and the adapted Hamiltonian parameters. The `IntegratorState` is **transient** (internal to a single kernel call / trajectory) and is *not* persisted.

```python
class HMCSamplerState(NamedTuple):
    # base fields (doc 01): coordinate, sample, log_prob, rng_draw,
    #                       chart_hyperparams, chart_indices
    ...
    momentum: Array            # post-kernel momentum (n,)
    step_size: Array           # scalar, adapted by dual averaging
    ham_params: dict           # adapted per-component params keyed by component id
                               # e.g. {"T_mu": DiagMass(inv_mass=...), "T_sigma": ...}
```

`ham_params` has fixed pytree structure for the run; only values change — no JIT recompilation, the same constraint as `chart_hyperparams` (doc 04). NUTS adds diagnostic fields (`log_weight` summary, `tree_depth`) as needed.

## Construction Example

```python
from mimcs import Model
from mimcs.hmc import (NUTS, ModelPotential, JacobianPotential, DiagonalQuadraticKinetic,
                      SplittingIntegrator, RepeatedIntegrator, Op)

model = Model(parameters=[...], log_prob_fns={"log_prior": ..., "log_likelihood": ...})

# Axis 1: components
V_prior = ModelPotential(model, "log_prior")
V_lik   = ModelPotential(model, "log_likelihood")
V_jac   = JacobianPotential(model)
kinetic = DiagonalQuadraticKinetic.per_parameter(model)

# Axis 2: multi-rate integrator with gradient caching (model grad once per outer step,
#          Jacobian + drift sub-stepped N=4 times)
inner = SplittingIntegrator([Op(V_jac, 0.5, cached_gradient=True), Op(kinetic, 1.0), Op(V_jac, 0.5)])
integ = SplittingIntegrator([
    Op(V_prior, 0.5, cached_gradient=True), Op(V_lik, 0.5, cached_gradient=True),
    Op(RepeatedIntegrator(inner, n=4), 1.0),
    Op(V_lik, 0.5), Op(V_prior, 0.5),
])

# Axis 3: NUTS + adaptation mixins (doc 02)
NUTSAdaptive = make_sampler_class(DualAveragingStepSize, WindowedCovariance, NUTS)
sampler = NUTSAdaptive(model, potentials=[V_prior, V_lik, V_jac], kinetic=kinetic,
                       integrator=integ, seed=0, max_tree_depth=10)
```

## Design Invariants

1. **Additivity.** The total Hamiltonian is the sum of component energies; the total gradient is the sum of component gradients. No component may depend on the existence of another.
2. **Reversibility + volume preservation.** Every integrator must be reversible and volume-preserving so acceptance is exactly $\min(1, e^{-\Delta H})$. Enforced by palindromic flow sequences (explicit) or symmetric fixed-point schemes (implicit).
3. **Integrator contract.** The integrator reads/writes only `q, p, potential_values, potential_grads`, always via `_replace`. Sampler-specific `IntegratorState` fields ride through untouched (flat extension) or are managed outside the integrator (compositional extension).
4. **Cache validity.** A `cached_gradient=True` kick is permitted only where the potential's cache is provably valid — i.e. no drift has changed `q` since it was written. This makes the reuse exact, not approximate. `init_integrator_state` seeds all caches at `q0`.
5. **Static structure.** Trajectory lengths, repetition counts $N$, fixed-point iteration counts, and `max_tree_depth` are static so the kernel is shape-stable under JIT.
6. **Adapted parameters in state.** All adapted component parameters live in `ham_params` (a JAX-state field), updated in `postprocess`, following the reparameterization principle shared with `chart_hyperparams` (doc 04).

## Implementation Status

Everything below ships except the two rows marked ⏳.

| Layer | Class | Role |
|---|---|---|
| Phase state | `IntegratorState` ✅ | base contract: `q, p, potential_values, potential_grads`, `log_weight`, `integrator_data` |
| Potentials | `ModelPotential` ✅ | one model log-density component |
| | `JacobianPotential` ✅ | chart change-of-variables correction |
| Kinetics | `DiagonalQuadraticKinetic` / `DenseQuadraticKinetic` / `LowRankQuadraticKinetic` ✅ | diagonal / dense / diagonal-whitened rank-J mass matrix |
| | `RelativisticKinetic` ✅ | bounded-velocity kinetic — **[experimental]** |
| | `RiemannianKinetic` ✅ | implicit RMHMC position-dependent metric — **[experimental]** (doc 07) |
| | `DiagonalBlock` / `LearnedDiagonalBlock` / `ShapedLearnedBlock` ✅ | explicit block-Riemannian metrics, the supported position-dependent path (doc 07) |
| Integrators | `SplittingIntegrator` ✅ | palindromic composition of flows (with caching) |
| | `RepeatedIntegrator` ✅ | nested multi-rate (RESPA) sub-stepping; built by `multirate_leapfrog` |
| | `LineSearchIntegrator` ✅ / `MarkovianLineSearchIntegrator` ✅ | within-orbit adaptive (WALNUTS) over any base integrator |
| | (non-separable kinetics carry their own implicit `flow` — no separate integrator; see doc 07) | |
| Structure | `BaseHMC` ✅ | holds components; shared services; `init_integrator_state` |
| | `HMC` ✅ | fixed-length trajectory (base `IntegratorState`) |
| | `RandomizedHMC` ✅ | trajectory length drawn per step from `{T, …, 2T}` |
| | `NUTS` ✅ / `BaseNUTS` ✅ / `SimpleNUTS` ✅ | tree-doubling trajectory (compositional extension: endpoints); `SimpleNUTS` is the O(2^depth) reference oracle |
| | ⏳ `MALT` | partial-refresh trajectory (flat extension: `log_weight`) — **planned** |
| | ⏳ `MALA` | single-step Langevin — **planned** (doc 02) |
