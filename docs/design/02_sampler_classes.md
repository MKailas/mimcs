# Sampler Classes and the Mixin System

## Design Motivation

The space of MCMC algorithms can be factored into two mostly orthogonal axes:

- **Proposal mechanism** (Metropolis–Hastings random walk, HMC leapfrog, NUTS tree building, Relativistic HMC, Riemannian HMC, …)
- **Adaptation strategy** (fixed hyperparameters, dual averaging, windowed covariance estimation, preconditioning, …)

A class hierarchy that combines both axes into a single inheritance tree would require an exponential number of concrete classes. Instead, we use Python's multiple-inheritance mixin system: **base classes** implement proposal mechanisms, **mixin classes** implement adaptation strategies, and the concrete sampler class is assembled at runtime by the user (or eventually by automated heuristics).

## Base Sampler Classes

Base classes implement the JAX kernel (`kernel`) and define the state type for their algorithm. They inherit from `BaseSampler` and must not implement adaptation logic.

```python
class BaseSampler:
    """Abstract base. Defines the pre/postprocess protocol and iteration loop."""

    state_class: type  # the NamedTuple subclass this sampler uses

    def preprocess(self, state) -> "SamplerState":
        raise NotImplementedError

    def postprocess(self, state) -> "SamplerState":
        raise NotImplementedError

    def kernel(self, state) -> "SamplerState":
        raise NotImplementedError

    def should_stop(self) -> bool:
        return False
```

### Concrete base classes (planned)

| Class | Algorithm | Key state fields |
|---|---|---|
| `RandomWalkMH` | Metropolis–Hastings with isotropic Gaussian proposal | `coordinate`, `sample`, `log_prob`, `rng_draw`, `step_size` |
| `BaseHMC` | Abstract HMC base: holds Hamiltonian components + integrator | + `momentum`, `step_size`, `ham_params` |
| `HMC` | Fixed-length HMC (subclass of `BaseHMC`) | inherits + `n_leapfrog` |
| `NUTS` | No-U-Turn Sampler (subclass of `BaseHMC`) | inherits + `tree_depth` |
| `MALA` | Metropolis-Adjusted Langevin Algorithm | + `grad_log_prob` |

The HMC family is itself modular: relativistic and Riemannian-manifold variants are **not** separate base classes but combinations of Hamiltonian components and integrators plugged into `HMC` or `NUTS`. See `06_hamiltonian_monte_carlo.md` for the three-axis decomposition (Hamiltonians × integrators × large-scale structure).

Each base class implements:

- `state_class` — the `NamedTuple` type for that algorithm's state
- `make_initial_state(init_position) -> SamplerState` — construct a valid initial state
- `kernel(state) -> SamplerState` — the JIT-compilable pure function
- `_preprocess_base(state) -> SamplerState` — base implementation (RNG injection; see §Composition below)
- `_postprocess_base(state) -> SamplerState` — base implementation (phase check, sample save)

## Mixin Classes

Mixins implement adaptation strategies. They **do not** define a `kernel`; they augment `preprocess` and `postprocess` to update hyperparameters and manage adaptation state.

### Mixin protocol

Each mixin must implement a subset of:

```python
class AdaptationMixin:
    def _init_hooks(self, **kwargs) -> None:
        """Read options and initialize Python-side state. Must call super()."""
        super()._init_hooks(**kwargs)

    def _preprocess_hooks(self, state):
        """End of preprocess. May modify state fields."""
        return super()._preprocess_hooks(state)

    def _postprocess_hooks(self, state):
        """Start of postprocess -- where adaptation runs (warmup only)."""
        return super()._postprocess_hooks(state)
```

Every hook is a cooperative `super()` chain, so a mixin that forgets the `super()` call silently
disables everything below it. Six chains exist: `_init_hooks`, `_init_state_hooks`,
`_preprocess_hooks`, `_postprocess_hooks`, `_finalize_hooks` (once, at the warmup→sampling
transition — where an averaged mass is frozen), and `_initialize_hooks` (once, on
`sampler.initialize()`, before warmup). Termination mixins additionally vote through
`should_stop()`.

Mixins store their adaptation state as ordinary Python instance attributes on the `Sampler` object (e.g., `self._step_size_accum`, `self._cov_estimate`). This state is not part of the JAX state NamedTuple and is never JIT-compiled.

### The mixins that exist

Grouped by what they write. Within a group the entries are usually **alternatives**, not additions
— `ScoreMassAdaptation` and `MassMatrixAdaptation` share a block filter and must not be stacked,
and `LineSearchStepSizeAdaptation` *subclasses* `RobbinsMonroStepSize`, so listing both is an MRO
error rather than a configuration.

| Mixin | Writes | What it does |
|---|---|---|
| `RobbinsMonroStepSize` | `state.step_size` | Robbins–Monro on the log step size toward a target acceptance |
| `LineSearchStepSizeAdaptation` | `state.step_size` | the same, driven by the line-search integrator's *proxy* acceptance — the real one is ≈1 under WALNUTS, so ordinary adaptation runs the step size away |
| `ScoreMassAdaptation` | `ham_params[k.id]` | SGD on a KL objective against each diagonal/dense block's **score** covariance (the default) |
| `MassMatrixAdaptation` | `ham_params[k.id]` | the classical empirical covariance of the **positions**, Stan-style |
| `LowRankAdaptation` | `ham_params[k.id]` | diagonal-plus-rank-J mass for a `LowRankQuadraticKinetic` block (diagonal by log-mass SGD, low-rank by Sanger/Oja) |
| `MetricAdaptation` | `ham_params` | SGD on a learned position-dependent diagonal metric written in the `MetricExpr` mini-language |
| `ShapedMetricAdaptation` | `ham_params` | the same, plus a constant shape `A` for a shaped metric `D(x)^½ A D(x)^½` |
| `RelativisticMassAdaptation` | `ham_params[k.id]` | per-particle rest mass for the relativistic kinetic — **[experimental]** |
| `CenteringAdaptation` | `state.chart_hyperparams` | standardize `centered=True` charts by running **mean / sd** |
| `RobustCenteringAdaptation` | `state.chart_hyperparams` | the same by **median / MAD** (heavy-tail robust); the one the factory composes |
| `UnitVectorCenteringAdaptation` | `state.chart_hyperparams` | fit an `adaptive=True` unit vector's stereographic chart (pole + scale) |
| `DiagonalCovarianceAdaptation` | `state.proposal_scale` | random-walk MH only: proposal scale from the running diagonal covariance |
| `UniformInit` | (initial state) | draw the starting coordinate from `U(−r, r)`, retrying until the target is finite |
| `StepSizeLineSearch` | (initial state) | backtracking MALA line search for the initial step size |
| `GelmanRubinTermination` | — | end warmup when max split-R̂ over the features drops below a threshold |
| `ClassifierTermination` | — | end warmup when a logistic regression cannot tell early draws from late ones (the default) |

The last four hang off different chains from the adaptations: `UniformInit` and
`StepSizeLineSearch` run once in `_initialize_hooks` and are inert unless `initialize()` is called;
the two termination mixins observe the drawn sample and vote through `should_stop()`
(`10_warmup_termination.md`).

> **Convention.** A mixin that adapts the step size must write the new step size back into the state before returning from `_postprocess_hook`, so the kernel picks it up on the next call without the kernel being aware of adaptation.

## Dynamic Class Construction

The concrete sampler class is assembled using Python's `type()` built-in:

```python
def make_sampler_class(*bases) -> type:
    """
    Assemble a concrete Sampler class from a base class and zero or more mixins.
    Resolution order: mixins are listed left-to-right in increasing MRO priority,
    with the base algorithm class last.
    """
    name = "_".join(cls.__name__ for cls in bases)
    return type(name, bases, {})
```

Usage:

```python
from mimcs.hmc import NUTS
from mimcs.adaptation import RobbinsMonroStepSize, ScoreMassAdaptation

NUTSAdaptive = make_sampler_class(RobbinsMonroStepSize, ScoreMassAdaptation, NUTS)
sampler = NUTSAdaptive(model, init_position=..., seed=42)
```

The MRO ensures that mixin hooks are called in the right order relative to the base class methods through standard cooperative multiple inheritance (`super()` chains).

## Method Resolution and `super()` Chains

The pre/postprocess methods follow the cooperative inheritance pattern. Each class in the MRO calls `super()`, so hooks compose automatically:

```python
class BaseSampler:
    def preprocess(self, state):
        state = self._inject_rng(state)      # always runs
        state = self._preprocess_hooks(state) # calls mixin chain
        return state

    def _preprocess_hooks(self, state):
        return state  # terminal; mixins call super()

class RobbinsMonroStepSize:
    def _preprocess_hooks(self, state):
        state = super()._preprocess_hooks(state)
        # (no pre-step action needed)
        return state

    def _postprocess_hooks(self, state):
        state = super()._postprocess_hooks(state)
        new_step_size = self._rm_update(state)
        return state._replace(step_size=new_step_size)
```

This pattern means the order of mixins in the class definition controls the order of hook execution, and adding a new mixin never requires modifying existing code.

A third cooperative hook, `_finalize_hooks(state)`, runs **once at the warmup→sampling transition** (the sampler calls it at the start of `sample()` when leaving the warmup phase). Adaptation mixins use it to freeze into the state the value they want held fixed for sampling when that differs from the last warmup iterate — the motivating case is **Polyak–Ruppert averaging** of an adapted mass: the raw stochastic-approximation iterate keeps driving the warmup dynamics (so step-size adaptation and the sampling path are unperturbed), while the suffix average of that iterate is frozen in only here, for the sampling run to use.

A fourth cooperative hook, `_initialize_hooks(state)`, runs **once, before warmup**, when the user calls the `sampler.initialize()` method (a no-op if no initialization mixin is present; nothing auto-calls it). Initialization mixins use it to set a reasonable starting point — currently `UniformInit` (draw the initial coordinate from `U(-2, 2)` in the unconstrained space, retrying to finite density) and `StepSizeLineSearch` (backtrack the step size from `0.5` to a single-leapfrog/MALA acceptance of `0.9`). It runs no adaptation and does not step the chain. A future joint initializer (Pathfinder: position + metric + step size) slots in as another such mixin. Ordering follows the same MRO convention — the position mixin is placed last (closest to the base) so its coordinate is set before the step-size search reads it.

## `Sampler.__init__` and Initialization

The `__init__` method is defined on `BaseSampler` and calls `_init_adaptation` on each mixin via the MRO:

```python
class BaseSampler:
    def __init__(self, model, init_position, *, seed=0, **kwargs):
        self.model = model
        self._rng_buffer = RNGBuffer(seed=seed, ...)  # see 03_rng_management.md
        self._phase = Phase.WARMUP
        self._samples = []
        self._init_hooks(**kwargs)  # calls the mixin chain in MRO order
        self.state = self.make_initial_state(init_position)
        self._kernel_jit = jax.jit(self.kernel)

    def _init_hooks(self, **kwargs):
        pass  # terminal

class RobbinsMonroStepSize:
    def _init_hooks(self, **kwargs):
        # options are read from **kwargs, never popped, so every mixin sees them all
        self._target_accept = float(kwargs.get("target_accept", 0.234))
        super()._init_hooks(**kwargs)
```

## Phase Management

`BaseSampler` maintains the phase itself, in `self._phase` — there is no scheduler mixin:

```python
class Phase(enum.Enum):
    WARMUP   = "warmup"      # adaptation runs; draws are discarded
    SAMPLING = "sampling"    # adaptation frozen; draws are retained
```

Two phases, not three: thinning is the caller's business, and every adaptation gates on
`self._phase is Phase.WARMUP`.

The warmup→sampling transition is driven by the cooperative `should_stop()` chain, which `warmup()` consults once per iteration (terminal answer: no, so `warmup(n)` without a termination mixin runs exactly `n` iterations). A termination mixin overrides it with a convergence criterion computed from the draws' *features* — see `10_warmup_termination.md`, which also explains why the adapted parameters are deliberately not consulted.

## Diagnostics

Per-transition diagnostics are uniform across samplers. Every `State` carries a single
`diagnostics: dict` that the kernel fills each step; the sampler declares its schema via
`init_diagnostics()` (a cooperative `super()` chain, seeded in `make_initial_state` so the initial
pytree matches the kernel's output — the same discipline as `IntegratorState.integrator_data`).
`RandomWalkMH` reports `{accept_prob, accepted}`; `BaseHMC` adds `{grad_evals, mean_refine,
proxy_accept_prob}`; `BaseNUTS` adds `{diverging, tree_depth, n_leaves}`.

`BaseSampler.postprocess` then generically appends every `state.diagnostics` entry — plus the
post-adaptation `step_size` — into one phase-tagged store (`self._diag`, a dict of per-name lists,
with a parallel SAMPLING/WARMUP flag). The accessors are thin views over it: `acceptance_rate`,
`warmup_step_sizes`, `divergence_count`/`divergence_rate` (default sampling-phase), `mean_tree_depth`,
`mean_refinements`, `mean_n_leaves`, `total_grad_evals`, and the general
`diagnostics(phase="sampling"|"warmup"|"all")` returning a dict of arrays. Adaptation mixins read
their signal from the same dict (`state.diagnostics["accept_prob"]` /
`["proxy_accept_prob"]`), so there is one source of truth.

**Gradient-evaluation cost** (the standard HMC efficiency denominator) is accumulated by the
integrator into `integrator_data["grad_evals"]` (cumulative): a leapfrog step costs one gradient per
non-cached potential kick (P; 1 for a single-potential target), and a line-search macro step costs
`2 + 2·Σ_{i=1}^{j-1} T_i + T_j` for finest level `j` (forward search + reversibility backward search
+ re-integration). NUTS sums the per-leaf increment into `NUTSTree.sum_grad_evals`; HMC reads the
trajectory endpoint. `total_grad_evals()` and `mean_n_leaves()` give ESS-per-gradient efficiency for
comparing samplers and integration schedules.

## Example: Full Sampler Construction

```python
from mimcs import Model, make_sampler_class
from mimcs.hmc import NUTS
from mimcs.adaptation import (
    RobbinsMonroStepSize, ScoreMassAdaptation, ClassifierTermination)

# Model defined elsewhere (see 05_model_interface.md)
model = Model(...)

# Assemble sampler class
NUTSFull = make_sampler_class(
    ClassifierTermination, RobbinsMonroStepSize, ScoreMassAdaptation, NUTS)

sampler = NUTSFull(
    model,
    init_position={"mu": 0.0, "sigma": 1.0},
    seed=0,
    step_size=0.1,
    target_accept=0.8,
)

# Run
for _ in range(10_000):
    sampler.state = sampler.preprocess(sampler.state)
    sampler.state = sampler._kernel_jit(sampler.state)
    sampler.state = sampler.postprocess(sampler.state)
    if sampler.should_stop():
        break

samples = sampler.get_samples()      # {name: (n, *ambient_shape)}
flat    = sampler.get_samples_flat() # (n, ambient_dim), the layout the factory uses
```
