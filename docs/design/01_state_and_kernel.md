# State, Kernel, and the Pre/Postprocess Protocol

## The `SamplerState` NamedTuple

Every sampler operates on a state that is a `NamedTuple`. This makes state a valid JAX pytree—fields are leaves, and JAX can trace through the structure under `jit`, `grad`, and `vmap` without special registration.

### Base fields

All sampler states share a minimal set of fields:

```python
from typing import NamedTuple
import jax.numpy as jnp
from jax import Array

class BaseSamplerState(NamedTuple):
    coordinate: Array          # position in the active coordinate chart (ℝⁿ, flat)
    sample: Array              # position in ambient space (manifold embedding, flat)
    log_prob: Array            # scalar: log π(sample) + Σ log |J_i| corrections
    rng_draw: NamedTuple       # typed draw struct, injected by preprocess (sampler-specific)
    chart_hyperparams: tuple   # per-parameter chart hyperparameters (tuple of pytrees)
    chart_indices: tuple       # per-parameter active chart index (tuple of scalar int arrays;
                               # always zeros today -- the atlas layer is not built, see doc 04)
    diagnostics: dict          # per-transition diagnostics, schema from init_diagnostics()
```

**`coordinate`** is what the sampler's proposal mechanism acts on. For a Euclidean parameter it is identical to `sample`; for a manifold-typed parameter it is the image of `sample` under the current chart map.

**`sample`** is always the position in the ambient (embedded) representation. It is what gets stored when a sample is accepted. For Euclidean parameters this is the same array as `coordinate`; for manifold parameters it is the ambient embedding (e.g., a unit vector in ℝ³ for a parameter on S²).

**`log_prob`** includes the Jacobian correction from the chart change-of-variables so that the stationary distribution is correct regardless of which chart is active.

**`chart_hyperparams`** is a tuple of per-parameter pytrees containing the learnable hyperparameters of each parameter's active chart (e.g., `AffineHyperparams(mean, log_scale)` for an affine reparameterization, `None` for fixed charts). It is part of the JAX state so the kernel can call `from_coordinate` and `log_jacobian_det` with current hyperparameter values for proposed positions. Updated by `postprocess` via adaptation mixins (see `04_manifold_parameters.md`).

**`chart_indices`** is a tuple of scalar integer arrays, one per parameter, identifying the active chart in that parameter's atlas. **Every shipped parameter type is single-chart, so the value is always 0 and nothing ever writes it** — the field is threaded through the state and passed to every chart call so the atlas layer can be added without an interface change. How transitions would work, and the open problems in making them work, are in `04_manifold_parameters.md`.

**`rng_draw`** carries the random numbers needed for one step as a typed NamedTuple (e.g., `NUTSRngDraw(momentum=..., log_slice=..., tree_uniforms=...)`). It is deposited by `preprocess` before each kernel call and consumed deterministically inside the kernel. The NamedTuple class is generated from the sampler's `DrawComponent` list and is specific to each base sampler class (see `03_rng_management.md`). Because NamedTuples are valid JAX pytrees, the nested structure in `state.rng_draw` is handled transparently by `jax.jit`.

### Sampler-specific extensions

Sampler subclasses extend the base state with algorithm-specific fields:

```python
class HMCSamplerState(NamedTuple):
    coordinate: Array
    sample: Array
    log_prob: Array
    rng_draw: NamedTuple
    chart_hyperparams: tuple
    chart_indices: tuple
    diagnostics: dict
    momentum: Array          # current momentum vector
    step_size: Array         # scalar, may be adapted
    ham_params: dict         # adapted per-component Hamiltonian params (e.g. mass matrices)
    potential_values: dict   # cached V_i at `coordinate`, keyed by potential id
    potential_grads: dict    # cached grad V_i at `coordinate` -- the leading half-kick reads
                             # these back rather than recomputing (doc 06, gradient caching)

```

`RandomWalkMH` adds `proposal_scale`. `HMCState` is shared by **every** HMC-family sampler,
including NUTS: there is no separate `NUTSSamplerState`, and NUTS's `tree_depth` is one entry in
the uniform `diagnostics` dict rather than a field of its own (doc 02, "Diagnostics").

The mass matrix is not a bare field but lives inside `ham_params`, because it is the adapted parameter of a *kinetic Hamiltonian component*. Diagonal vs. dense vs. low-rank vs. relativistic vs. position-dependent mass matrices are different components, not different state layouts (see `06_hamiltonian_monte_carlo.md`).

Adaptation-related quantities (running mean/variance estimates, Robbins--Monro accumulators) are **not** stored in the `NamedTuple` state because they are consumed and updated on the Python side in `postprocess`. Keeping them in Python avoids exposing adaptation bookkeeping to JIT and keeps the kernel pure.

> **Decision rationale.** Putting adaptation state inside the JAX state would require either (a) carrying it through JIT as large arrays or (b) using `jax.experimental.io_callback`. Both complicate the kernel. Since adaptation decisions are inherently Python-level (they may consult sample history, change hyperparameters, trigger chart switches), keeping them in Python is the cleaner split.

## The JIT Kernel Contract

The kernel is a **pure function** with signature:

```python
def kernel(state: SamplerState) -> SamplerState:
    ...
```

### Invariants the kernel must satisfy

1. **Purity.** No side effects. No Python-level I/O, no mutation of external buffers.
2. **Stationarity.** The Markov kernel must leave the target measure π invariant. For Metropolis-based kernels this means the acceptance probability is correctly computed, including the Jacobian correction in `log_prob`.
3. **Determinism given `rng_draw`.** The kernel produces a deterministic output given identical `state`. All randomness enters through `state.rng_draw`.
4. **Shape stability.** All output arrays have the same shapes and dtypes as corresponding input arrays. This is required for `jax.lax.scan` and repeated JIT calls.
5. **No chart switching.** Chart selection and the chart transition logic live in `preprocess`/`postprocess`. The kernel operates in the chart that is already active.

### JIT compilation

```python
import jax
kernel_jit = jax.jit(kernel)
```

The kernel is compiled once and reused across all iterations. Because state is a NamedTuple with fixed structure, re-compilation only occurs if field shapes or dtypes change—which they must not.

## The `preprocess` Method

`preprocess` is a Python method on the `Sampler` object. It runs **before** each kernel call.

```python
class Sampler:
    def preprocess(self, state: SamplerState) -> SamplerState:
        ...
```

### Responsibilities

1. **RNG injection.** Call `self._rng_buffer.next()` to get a `dict[str, Array]` of pre-generated draws, wrap it in the sampler's typed `RngDraw` NamedTuple (constructed once at sampler init from the sampler's `DrawComponent` list), and write it into `state.rng_draw`. If the buffer is exhausted, it replenishes automatically (see `03_rng_management.md`).

2. **Chart management.** Two distinct updates may be needed:
   - *Atlas transition* (**not implemented**, doc 04): switch chart when a parameter approaches its chart's singularity, recomputing `state.coordinate` and `state.log_prob` and updating `state.chart_indices[i]`. What ships instead is chart *hyperparameter* adaptation — the sphere's pole is moved to keep the singularity away from the mass, so no transition is needed.
   - *Hyperparameter injection*: if an adaptation mixin has updated chart hyperparameters during `postprocess`, the new values are already in `state.chart_hyperparams` (written there by `postprocess`). No separate injection step is needed in `preprocess`; the new values will be used by the kernel automatically.

3. **Pre-step adaptation hooks.** Some adaptation strategies need to act before the proposal (e.g., refreshing the momentum distribution using an updated mass matrix). These hooks are defined in mixin classes and called here.

`preprocess` returns a new state (preserving immutability); it does not mutate `state` in place.

## The `postprocess` Method

`postprocess` is a Python method on the `Sampler` object. It runs **after** each kernel call.

```python
class Sampler:
    def postprocess(self, state: SamplerState) -> SamplerState:
        ...
```

### Responsibilities

1. **Sample storage.** If the sampler decides the current state is a sample to be retained (not discarded as part of adaptation), it appends `state.sample` to the sample store. The decision of whether to save is made here, not inside the kernel.

2. **Adaptation updates.** Compute running statistics from the accepted sample (mean, covariance, acceptance rate) and update the Python-side adaptation state. The adaptation state is held by the `Sampler` object itself, not in the JAX state.

3. **Hyperparameter updates.** Apply adaptation schedules to kernel hyperparameters embedded in `state` (step size, mass matrix, chart hyperparameters). If updated, return a new state with the modified fields via `_replace`. Chart hyperparameter updates follow the same pattern as mass matrix updates: compute new values from Python-side accumulators, write into `state.chart_hyperparams` as a new tuple. Because the structure of `chart_hyperparams` is fixed (same pytree shape throughout the run), this does not trigger JIT recompilation.

4. **Stopping criterion.** Update any convergence diagnostics. The `sampler.should_stop()` method reads from state updated by `postprocess`.

5. **Post-step adaptation hooks.** Mixins may register hooks here (e.g., Robbins--Monro step-size adaptation after each transition).

### Why saving samples is in `postprocess`

With adaptive MCMC, there is no predetermined boundary between warmup and sampling. Instead, the `Sampler` maintains an internal phase variable (e.g., `ADAPTING`, `SAMPLING`, `THINNING`) and `postprocess` decides whether the current iterate is a keepable sample based on that phase. This logic cannot live inside the kernel because phase transitions are Python-level decisions that may depend on sample history.

## Iteration Loop Summary

```
preprocess:
  ├── pop RNG draw from buffer (replenish if needed)
  ├── [optional] switch coordinate chart
  └── [optional] pre-step adaptation hooks
          │
          ▼
kernel(state):         ← jax.jit, pure
  ├── construct proposal in coordinate space
  ├── evaluate log_prob at proposal (via model + Jacobian)
  ├── Metropolis / HMC / NUTS accept/reject
  └── return new state
          │
          ▼
postprocess:
  ├── save sample (if in sampling phase)
  ├── update Python-side adaptation accumulators
  ├── update hyperparameters in state (step size, metric, …)
  └── update convergence diagnostics
```
