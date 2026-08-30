# mimcs: Architecture Overview

## Project Goals

**mimcs** is a general-purpose Bayesian inference library built on JAX, designed around three first-class requirements that shape every architectural decision:

1. **Adaptive MCMC from the ground up.** Adaptation is not an afterthought. The sampling loop is designed to support continuous, dynamic adaptation throughout a run rather than a fixed warmup/sampling split.

2. **Manifold-typed parameters.** Parameters may live on Riemannian manifolds embedded in Euclidean space. A parameter is defined by its *charts* — maps from the ambient sample to the unconstrained coordinate the sampler works in — and the library applies the density corrections (Jacobian determinants) natively. Shipped today: the unit sphere `S^(d-1)` with an adaptive stereographic chart, the simplex, ordered vectors, covariance and correlation matrices (and their Cholesky factors), and bounded/positive scalars. **Not shipped:** multi-chart *atlases* with transitions between charts, which none of these need — the sphere's adaptive chart parks its singularity where the density is lowest, and the positive definite matrices are covered globally by one chart — but Grassmannians would. See `04_manifold_parameters.md`, which sketches the design and states the open problems.
   A parameter may also be **discrete** — integer-valued, `int<lower=L, upper=U>` — which is a
   different thing rather than another manifold: it has no chart and no gradient, so it lives in a
   flat `int` array of its own and is moved by a Metropolis-within-Gibbs sweep composed over any
   continuous sampler (`14_discrete_parameters.md`). Mixtures, latent classes and spike-and-slab
   selection need one.

3. **Flexible, composable MCMC methods.** The design supports a family of advanced samplers — Metropolis–Hastings, HMC, randomized HMC, NUTS, WALNUTS (within-orbit adaptive step size), explicit block-Riemannian HMC with learned metrics, and parallel tempering over any of them — and allows mixing base algorithms with interchangeable adaptation strategies via Python's class system. Relativistic HMC and implicit Riemannian HMC are implemented but marked **[experimental]** (see `06`, `07`).

## High-Level Component Map

```
┌───────────────────────────────────────────────────────────────────┐
│                          User / Script                            │
└───────────────────────────┬───────────────────────────────────────┘
                            │  constructs
                            ▼
┌───────────────────────────────────────────────────────────────────┐
│  Sampler  (dynamically constructed Python class)                  │
│                                                                   │
│  ┌──────────────┐   ┌────────────────────┐   ┌────────────────┐  │
│  │  preprocess  │   │   JIT kernel       │   │  postprocess   │  │
│  │  (Python)    │──▶│   next_state(s)    │──▶│  (Python)      │  │
│  │              │   │   (JAX / pure)     │   │                │  │
│  │ • replenish  │   │ • proposal         │   │ • save samples │  │
│  │   RNG buffer │   │ • acceptance       │   │ • adapt        │  │
│  │ • inject RNG │   │ • manifold maps    │   │ • decide stop  │  │
│  └──────────────┘   └────────────────────┘   └────────────────┘  │
│                                                                   │
│  Base class (MH / HMC / NUTS / …)                                 │
│  + Mixin(s) (RobbinsMonroStepSize / ScoreMassAdaptation / …)     │
└─────────────────────────────┬─────────────────────────────────────┘
                              │  wraps
                              ▼
┌───────────────────────────────────────────────────────────────────┐
│  Model                                                            │
│  • list[Parameter] + list[DiscreteParameter]                      │
│  • log_prob_components() → dict[str, Array]                       │
└─────────────────────────────┬─────────────────────────────────────┘
                              │  contains
                              ▼
┌───────────────────────────────────────────────────────────────────┐
│  Parameter (BaseParameter subclasses)                             │
│  • chart: to_coordinate / from_coordinate / log_jacobian_det      │
│  • single chart today; atlas/transitions sketched, not built      │
│  BaseDiscreteParameter subclasses: no chart, own flat int block   │
└───────────────────────────────────────────────────────────────────┘
```

## Iteration Pattern

The outer sampling loop looks like:

```python
for _ in range(max_iterations):
    state = sampler.preprocess(state)   # Python: inject RNG draw from buffer
    state = sampler.kernel(state)       # JAX JIT: pure proposal + acceptance
    state = sampler.postprocess(state)  # Python: save, adapt, check stop
    if sampler.should_stop():
        break
```

The `kernel` function is the only thing that runs under `jax.jit`. `preprocess` and `postprocess` are deliberately left in Python so that they can make dynamic decisions (buffer replenishment, adaptive scheduling, streaming sample storage) that are either impractical or unnecessary to JIT-compile.

## Design Constraints and Rationale

| Constraint | Rationale |
|---|---|
| State is a `NamedTuple` | Compatible with JAX pytree tracing; zero-copy between Python and JIT |
| RNG is injected by `preprocess`, not inside `kernel` | Allows large batched generation; keeps kernel pure and reproducible |
| No fixed warmup length | Adaptive MCMC determines its own adaptation schedule (`10_warmup_termination.md`: `warmup()` ends on a mixing criterion, not a count) |
| Samples saved in `postprocess`, not inside kernel | Decouples storage from sampling; enables streaming and dynamic stopping |
| Manifold embedding always available in `state.sample` | Samplers that exploit geometry can access it; generic samplers ignore it |
| Log-prob is decomposable (prior / likelihood / …) | Some advanced samplers (e.g., tempered, SMC-within-MCMC hybrids) exploit the decomposition |

## Document Index

| Document | Topic |
|---|---|
| `01_state_and_kernel.md` | `SamplerState` design, JIT kernel contract, pre/postprocess protocol |
| `02_sampler_classes.md` | Base sampler classes, mixin system, dynamic class construction |
| `03_rng_management.md` | JAX PRNG buffering strategy, replenishment, injection into state |
| `04_manifold_parameters.md` | `BaseParameter`, charts, atlases, Jacobian corrections, adaptive reparameterizations |
| `05_model_interface.md` | `Model` class, log-prob decomposition, parameter registry |
| `06_hamiltonian_monte_carlo.md` | Modular HMC: Hamiltonian components, composable integrators, `BaseHMC`/`HMC`/`NUTS` |
| `07_riemannian_hmc.md` | Riemannian Manifold HMC: position-dependent metrics (implicit dense, explicit block), learned metrics |
| `08_model_dsl.md` | Stan-like model DSL: imperative log-density, blocks, the compiler (lexer/parser/interpreter), staging |
| `09_sampler_factory.md` | Sampler factory: `make_sampler(model, *results)`, the `Evidence`→`SamplerSpec`→`build` pipeline, heuristic rules, staging |
| `10_warmup_termination.md` | Dynamic adaptation: features/observables, `should_stop()`, split-R̂ and classifier criteria for ending warmup |
| `11_sample_evaluation.md` | `sampler.summary()`: per-feature ESS/R̂, the target-aware Langevin–Stein diagnostic, and the ambient-score pullback |
| `12_logging.md` | Library-wide `logging`: per-module loggers under `mimcs`, the INFO default, what belongs at each level, and the end-of-phase reporting hooks |
| `13_parallel_tempering.md` | Parallel tempering for multimodality: the K-fold product space, why not K separate samplers, per-component tempering, swaps, and the joint-vs-independent selection question |
| `14_discrete_parameters.md` | Discrete (integer) parameters: the second flat array, why they have no chart, the Metropolis-within-Gibbs sweep and the gradient cache it must refresh, and the deferred jump operators |
