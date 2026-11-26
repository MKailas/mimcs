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
| `DiscreteMarginalAdaptation` | `state.discrete_proposal_params` | each integer coordinate's marginal pmf, so the Gibbs sweep proposes from it rather than uniformly (doc 14) |
| `UniformInit` | (initial state) | draw the starting coordinate from `U(−r, r)`, retrying until the target is finite |
| `StepSizeLineSearch` | (initial state) | backtracking MALA line search for the initial step size |
| `GelmanRubinTermination` | — | end warmup when max split-R̂ over the features drops below a threshold |
| `ClassifierTermination` | — | end warmup when a logistic regression cannot tell early draws from late ones (the default) |

`SystematicScanMetropolisWithinGibbs` and its sibling `RandomScanMetropolisWithinGibbs`
(`mimcs/samplers/gibbs.py`, doc 14; their shared superclass `DiscreteMetropolisWithinGibbs` is not
composable on its own) are **kernel-composing** mixins — a category of their own. Every mixin above cooperates through the `_*_hooks` chain and never
touches `kernel`; that one overrides `kernel` and calls `super().kernel`, appending a
Metropolis-within-Gibbs sweep over the model's integer parameters to whatever continuous algorithm
it is composed over. It works with no change to any base algorithm because `BaseSampler.__init__`
jits the MRO-resolved bound method, so the composition compiles as one function; and the ordering
rule is the usual one, mixins before the base. `DiscreteMarginalAdaptation` is an ordinary
adaptation mixin that feeds it, and sits to its left.

The last four hang off different chains from the adaptations: `UniformInit` and
`StepSizeLineSearch` (the position and step-size initializers) run once in `_initialize_hooks` and
are inert unless `initialize()` is called;
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

A third cooperative hook, `_finalize_hooks(state)`, runs **once at the warmup→sampling transition** (the sampler calls it at the start of `sample()` when leaving the warmup phase). Adaptation mixins use it to freeze into the state the value they want held fixed for sampling when that differs from the last warmup iterate — the motivating case is the **exponential moving average** of an adapted mass (`mass_ema`, shared by every mass adaptation, `mimcs/adaptation/_ema.py`): the raw stochastic-approximation iterate keeps driving the warmup dynamics (so step-size adaptation and the sampling path are unperturbed), while the EMA of that iterate is frozen in only here, for the sampling run to use. (`mass_ema_warmup` lets the EMA drive warmup too. Both are off by default, except `mass_ema` for the learned metrics, whose raw last iterate is not safe to sample with.)

A fourth cooperative hook, `_initialize_hooks(state)`, runs **once, before warmup**, when the user calls the `sampler.initialize()` method (a no-op if no initialization mixin is present; nothing auto-calls it). It runs no adaptation and does not step the chain. Initialization mixins come in two independent kinds, and a sampler has at most one of each (the factory's `position_init` / `step_size_init` slots, `09_sampler_factory.md`, make that structural):

* a **position** initializer sets the starting point --- `UniformInit` (draw the coordinate from `U(-2, 2)` in the unconstrained space, retrying until both the density and its gradient are finite) `PathfinderInit` (start at a draw from Pathfinder's normal approximation, below), or `ADVIInit` (a draw from a mean-field ADVI fit, below);
* a **step-size** initializer sets the starting step at whatever position it is handed --- `StepSizeLineSearch` (backtrack from `step_size` to a single-leapfrog/MALA acceptance of `0.9`).

Ordering follows the MRO convention: the position mixin sits right of the step-size one, so its coordinate is set before the search reads it. The discrete labels are drawn by the Gibbs sweep's own `_initialize_hooks`, deeper still, so the position is tested against the labels the chain will start from; the sweep's `init_labels` kwarg (default `True`; the factory sets it from `position_init`) switches that draw off.

**The fixed set — the contract every initializer honours.** `sampler.init_fixed` names the parameters (continuous or discrete) whose start the caller *supplied*, and an initializer must leave them where they are. It comes from the `init_fixed` constructor kwarg, which defaults to the keys of a dict `init_position` (an array fixes nothing: the constructor requires one, so it cannot express an opinion); the factory passes it explicitly from `build(init=...)`. The *values* need no channel of their own: `make_initial_state` builds the state from them, so they are in `state.sample` / `state.discrete` when `initialize()` runs. A partial dict is completed there by `Model.complete_sample` / `complete_discrete` (chart origin, re-charted under any named parent; values checked against support by a chart round trip).

Fixed values are kept in **sample** space, through one primitive, `Model.pin_fixed(coordinate, sample, fixed, ...)`: in topological order, a fixed parameter takes its sample and is re-charted under its parents' values in the result, and a free one keeps its coordinate. A coordinate-space mask would be wrong whenever a fixed child has a free parent (its coordinate moves with the parent's bound or centring). `UniformInit` draws the full box and overlays the fixed set on each candidate, so the free coordinates are exactly the unfixed run's and the default path is bit-identical to the old one (`tests/data/golden_init.npz`); a fully fixed position is kept and refused if not finite. An optimizing initializer (Pathfinder) maps its free coordinates to a full one through the same primitive --- it is pure and differentiable --- with `Model.coordinate_mask(names)` giving which coordinates are free to move.

**Pathfinder** (`PathfinderInit`, `mimcs/vi/`; Zhang, Carpenter, Gelman & Vehtari 2022, with the paper's defaults) runs L-BFGS up the log density from a `U(-radius, radius)` start, builds a normal approximation at every iterate from the path's own curvature (diagonal-plus-rank-`2J` inverse Hessian, sampled and scored through a thin QR and a `2J x 2J` Cholesky, or densely when `2J >= n`), estimates each one's ELBO with `K = 5` draws, and starts the chain at a draw from the best. Multi-path runs `I = 20` paths vmapped in one compilation and Pareto-smoothed-importance-resamples the pooled draws (`mimcs/vi/psis.py`, held to `arviz_stats` in the tests). Its target is the **free coordinates**: each free vector is scattered into the state's coordinate and `pin_fixed` re-charts the fixed parameters, and the fixed parameters' own chart Jacobians are subtracted (`Model.log_jacobian`), because the density of the free parameters given ones fixed in *sample* space has none (for a fixed child of a free parent that term moves with the parent; otherwise it is a constant). The labels are held at the sweep's draw, which runs first. Under parallel tempering it runs once on the base (`beta = 1`) posterior and gives each rung its own draw. Two misprints in the paper are not copied (Algorithm 3's pair filter printed reversed, Algorithm 7's initial scale printed as the Hessian's); the convergence test is L-BFGS-B's `|Δf| / max(|f_l|, |f_{l+1}|, 1)`, what the paper's own implementation (R's `optim`) computes, because the printed `|Δlog p| / |log p|` never converges when the log density at the mode is near 0 (every path on a Gaussian with `max log p = 0` ran all 1000 iterations); the line search is capped at `max_halvings` trials (first Armijo step if no trial meets both Wolfe conditions); a step with no curvature pair held is scaled to unit length, L-BFGS-B's first step (from a far start the raw gradient at `lambda = 1` reaches no finite trial in 30 halvings: every cscm path failed at iteration 0, start gradient 2.9e20); and the ELBO is evaluated as the path is made, keeping only the best factors (O(Jn) memory). An optional `patience` (default `None`, the paper's run) ends a path once that many iterations pass without an ELBO improvement. Funnel paths otherwise run all `L = 1000` iterations heading for the pole, long after their best iterate (about 10–30). ELBO traces (`tests/experiments/writeups/pathfinder_patience.md`) show no single value fits: funnels keep their pick from `P = 8`, while slow climbs (hmm, diamonds, bci) stall for 130–240 iterations between improvements and need `P` of 100–200. So the default is unchanged. The covariance factors were checked against the dense BFGS recursion and the sampler's `log q` against a dense normal density before the module was written; both checks are tests. A path with no finite ELBO "fails" in the paper's sense --- e.g. any target with a hard boundary, such as the conditional of a free scale given a fixed child it bounds, where every normal approximation has infinite KL --- and the chain then starts at the last L-BFGS iterate. `sampler.init_report` records iterations, the selected iteration, ELBO, Pareto k, distinct draws and evaluations in the sampler's gradient units.

**Mean-field ADVI** (`ADVIInit`, `mimcs/vi/meanfield.py`; Kucukelbir et al. 2017) is plain variational inference, kept as an alternative and a control for Pathfinder. It fits a diagonal normal `N(mu, diag(exp(2 omega)))` over the same free-coordinate target, from the same `U(-radius, radius)` start: it shares Pathfinder's init substream, so a seed starts both from one point. The fit begins at `omega = 0` and runs Adam with its standard constants (learning rate 1e-3) on the reparameterised `K = 5`-draw gradient, with the entropy's gradient exact. It stops once `patience` iterations pass without the ELBO estimate improving, or at `max_iters`, and returns the **last** iterate. The estimate it stops on is `mean(log p - log q)` over the same draws, free, with zero variance at an exact fit; the exact-entropy form keeps a `sqrt(n / 2K)` floor. A draw with a non-finite density or gradient is dropped from the gradient, which averages over the finite draws; dropping the whole update instead strands a fit whose unit-scale start straddles a support boundary. The *estimate* of such an iteration is `-inf`, since averaging the rest is biased high while q leaks, and patience would stop on that stale record. The ELBO traces (`tests/experiments/writeups/advi_init.md`) set the provisional defaults `patience` 2000 and `max_iters` 10000. At learning rate 1e-3 the problem is slowness rather than noise. `omega` moves at most about `learning_rate` per step, and heavy-tailed gradients make Adam's normalised step far smaller than that: on bci's log-Gaussian Cox process it barely moves in 20000 iterations. Patience below 2000 stops even a 10-d Gaussian more than 1 nat short, and kilpisjarvi, diamonds, hmm and bci are still climbing at 10000. Two options exist for the cluster study's sweep, both off by default: a learning-rate schedule (`lr_schedule`: `"power"`, `lr * (1 + t / lr_tau)^-lr_power`, horizon-free so it suits the patience stop; `"cosine"`, from `lr` to `lr * lr_final` over `max_iters`), and a start scale `init_log_sd` below 0, which keeps the first draws near the start where unit-scale draws exponentiate a latent field (bci's ELBO at `omega = 0` is about -1e14).

**Fixed-draw mean-field VI** (`DADVIInit`, `mimcs/vi/deterministic.py`; DADVI, Giordano, Ingram & Broderick 2024) is the compromise between the two. Its optimisation objective is the ELBO, like ADVI's, but it is deterministic, so a second-order optimiser applies. It draws `K = 5` standard-normal vectors `g_k` once and maximises `F(mu, omega) = mean_k [log p(x_k) - log q(x_k)]`, with `x_k = mu + exp(omega) * g_k`, from Pathfinder's start (the same substream), by the paper's trust-region Newton-CG (`optimizer="trust_ncg"`, default) or Pathfinder's L-BFGS (`"lbfgs"`). The maths, each point checked numerically (`tests/experiments/dadvi_init/derivation_checks.py`, the identities also tests):
* **A point cloud.** Since `omega_i = log sd_k(x_ki) - log sd_k(g_ki)`, `F` is the mean log density of `K` points, constrained to be a coordinatewise-affine image of the fixed pattern, plus the log of each coordinate's spread. `K = 1` is unbounded: one point can sit at the mode while `sigma -> inf`. Hence `elbo_draws >= 2`.
* **Funnels.** The spread term cancels a hierarchical scale's normaliser: down centred eight schools' neck, `log p` rises like `-7 log tau`, but the profile of `F` *falls* like `+2 log tau` (the Jacobian, plus the hyper-mean's spread, which must shrink with `tau`). The neck is not a maximum, unlike for Pathfinder's L-BFGS on `log p`.
* **Gaussian targets.** The realised centroid `xbar = mu + sigma * gbar` is the exact mean, and the cloud's spread the exact mean-field sd. The fitted `sigma` is that spread divided by the pattern's sample sd: noisy and inflated (`E[sigma^2]/sd^2 = K/(K-3)`, 2.5 at `K = 5`).
* **The `"lr"` read-out (default)** is the linear-response covariance of the fit's own mean estimate, `J^T (-hess F)^-1 J` with `J = d xbar / d(mu, omega) = [I; diag(sigma * gbar)]`, around `xbar`. It is exact on a Gaussian target for any draws. The plain mean block of `(-hess F)^-1` is not: it gives `sd^2 (1 + gbar^2 / 2 shat^2)`. The Hessian is formed exactly, from `2n` Hessian-vector products of `F` (batched `lax.map`, counted as `4nK` gradients), and the `n x n` sandwich is Cholesky-factored.
* **`"lr_lbfgs"`** uses a secant inverse-Hessian estimate instead, from the last `history` steps' pairs (Pathfinder's `alpha`-recovered diagonal plus rank `2J` over the `2n` coordinates; under `"trust_ncg"` the pairs are the trust-region steps'). The sandwich keeps that form, `diag(alpha_mu + v^2 alpha_omega) + beta' gamma beta'^T` with `beta' = beta_mu + v * beta_omega` and `v = sigma * gbar`, so it is drawn with Pathfinder's Algorithm 4 at no extra cost. But it underestimates the variances 3–7x (median, up to e^7), so its starts sit at the mode in high dimension (lp percentile 0.97–1.00 on irt, bci, diamonds), and chains started from it start alike. The exact sandwich's are posterior-like (0.13–0.71), which is why `"lr"` is the default. A non-positive-definite exact sandwich (5/8 diamonds fits, all stopped at the iteration cap) falls back to `"lr_lbfgs"`, and that to `"diag"`. The exact Hessian is a dense `2n x 2n` matrix with an `O(n^3)` factorisation. In the cluster study (2026-11-24) memory bound it, not time: the read-out's extra peak is about 6 of those matrices (the Hessian, its symmetrised and negated copies, the Cholesky factor and the sandwich's solves), 4 GB at n = 5019 in float64, while the time was about a minute on 32 CPUs. So `lr_max_gb` (default 1 GiB, n = 2364 in float64 and 3344 in float32) caps it by `lr_memory_gb`: above the budget `"lr"` draws from `"lr_lbfgs"` without forming it (`lr_capped`, not a failure). The Hessian is also formed only for a converged fit (a `rel_tol` stop). In the study 125 of the 132 unconverged large-target fits (line-search stops on bci, the 1000-iteration cap on cscm and reg_horseshoe) fell back after forming it, while 4 of the 8 converged ones used it, so an unconverged fit now skips it (`lr_skipped`); `max_iters` is 5000 for the same reason (cscm_observed then converges 4/4, with Pareto k 0.63 against 3.2).
* **Other read-outs.** `"diag"` draws from the fitted `q` itself. `standardize=True` centres and scales the pattern, which removes the `sigma` noise.

The run:
* The start is `(x0, omega = init_log_sd)`, 0 by default. A non-finite start lowers `omega` by `log 4` up to 10 times.

`optimizer="trust_ncg"` (default) is SciPy's `trust-ncg` (`scipy.optimize._trustregion`, `_trustregion_ncg`), which the paper's code (martiningram/dadvi) runs with SciPy's defaults, ported to JAX and checked iterate for iterate against SciPy (`tests/test_trust_ncg.py`):
* Each step minimises the exact local quadratic model of `-F` over the ball `|p| <= radius` in the `2n` coordinates, by Steihaug's CG (`mimcs/optim/trust_ncg.py`: forcing tolerance `min(1/2, sqrt|g|) |g|`, to the boundary on leaving the ball or on non-positive curvature). Hessian-vector products come from one `jax.linearize` of `grad F` per step, each counted as `2K` gradients (`n_hvp`).
* `rho` = actual / predicted gain. The step is accepted if `rho > trust_eta` (0.15); the radius is quartered if `rho < 1/4` and doubled (to `max_trust_radius`, 1000) for a boundary step with `rho > 3/4`. It starts at `initial_trust_radius` (1), which also does the job of L-BFGS's unit-length first step.
* It stops on `"tolerance"` when `|grad F| < gtol` (SciPy's 1e-4), on `"trust region"` when a boundary step predicts no gain (SciPy's failure exit) or the radius collapses, and at `max_iters` iterations, rejected ones included (SciPy's count; `n_iter` counts accepted steps).
* Departures from SciPy: (1) a proposal with a non-finite `F` or gradient is rejected with `rho = -inf` (SciPy's NaN `rho` leaves the radius and so the step unchanged until `maxiter`); (2) a second `"tolerance"` test, the Newton decrement: an interior step (CG converged inside the ball) predicting a gain below `max(rel_tol, eps) max(|F|, 1)`, or none at all, which for CG only rounding produces. It is scale-free and reachable in float32, where `|grad F| < 1e-4` often is not: on eight schools in float32 the gains fall below F's resolution first, and without the floor the rejections from rounding collapse the radius (`rel_tol = 0` turns it off); (3) `H p` is carried through the CG recurrences, so the model's value costs no product (SciPy spends one per step); (4) an optional CG cap, `cg_max_iters` (default `2n`, which exact-arithmetic CG never exceeds; SciPy has none). A solve cut short below `2n` is counted (`n_cg_capped`) and never triggers the decrement test.
* Why it replaced L-BFGS as the default: the bci failure below. Each trust-region model is the exact Hessian at the current point, so there are no stale secant pairs to couple the field to the length scale, and every step is checked against the gain its model predicted, not only against the total `F`. On bci 10 m from the uniform start (acaldi, x64, seeds 0-3, 2026-11-26) all 4 fits converge, where L-BFGS failed 4/4. The final F is -1816 to -1868, and max log ls is 4.2-4.5 against L-BFGS's 8.7-13.4. Each fit takes 109-277 iterations and 1261-1588 products, n_grad 13.7k-17.4k, in 53-61 s on 8 CPUs. On cscm_observed both seeds converge in 140-178 iterations (L-BFGS: 1529-1654) and use the exact read-out, Pareto k 0.48 / 0.39 (L-BFGS: 4.06 / 1.00). `jax.linearize` is 1.29x faster per product than a jvp of the gradient, at the same peak memory.
* The secant read-out under `"trust_ncg"` uses CG pairs. Above `lr_max_gb`, or for a fit that has not converged, the read-out is `"lr_lbfgs"`. Pairs made from the trust region's few long steps gave a poor one: the draws' ELBO was -40k on cscm_imputed (L-BFGS's own pairs: -32.6k) and -1e6 to -1.6e10 on bci. So `secant="cg"`, the default under `"trust_ncg"`, makes the pairs instead from `cg_pairs` (200) iterations of plain CG on `-hess F` at the fit, from a random right-hand side (`_cg_secant`). Each direction and its product is an exact curvature pair. All of them are kept, so the low-rank part is CG's exact inverse on its Krylov space, which is exact on a Gaussian at `cg_pairs >= 2n` (a test). Measured on 2 seeds each (2026-11-26):

  | target | step pairs | 50 CG pairs | 200 CG pairs | 1000 CG pairs | exact read-out |
  |---|---|---|---|---|---|
  | bci 10 m | -1e6 to -1.6e10 | -9.5k to -13.8k | -6.8k to -9.7k | -7.2k to -7.5k | -3.2k |
  | cscm_imputed | -40k to -42k | -40k to -41k | -31k to -33k | -29k | |

  The exact read-out was run on 1 seed; the cells are the draws' ELBO, with in-sample F ≈ -1840 on bci. 200 pairs cost 1-3 s and 2 x 200 x 2n of memory; 1000 pairs cost 30 s and +0.6 GB. Under `"lbfgs"`, the steps' own pairs are as good (cscm_imputed: -32.6k against -33.3k to -34.5k from 200 CG pairs), so `secant=None` keeps them there. Every one of these large-n read-outs has Pareto k 10-18, the exact one included (bci: 7.4, against my predicted 0.5-1). With K = 5 draws at n = 5019, the linear-response normal is itself far from this posterior.

`optimizer="lbfgs"`:
* A non-finite point makes `F = -inf`, which the line search backtracks from.
* A step taken with no curvature pair in the history (the first, or after every pair so far was rejected) is scaled to unit length (L-BFGS-B's rule). From a unit-scale cloud on bci the gradient is ~1e14, and after a rejected first pair ~1e11; 30 halvings of `lambda = 1` never reach a finite F.
* It stops on L-BFGS-B's relative test, a failed line search, or `max_iters` (5000).
* The failure that motivated `"trust_ncg"` (cluster study, 2026-11-25): on bci's 10 m grid (n = 5019) from the uniform
  start, every fit (20/20) first shrinks the field scale `sigma` to about `e^-10`. The length
  scale then runs past its prior's mode to where the Matérn Cholesky is NaN (log ls ≈ 13.7), and
  the line search stops. ls's own gradient pulls it back (the prior's −5 per unit; the
  likelihood's part is about 0), but the L-BFGS direction moves it up regardless: the secant
  pairs, all taken while `sigma` fell and ls rose together, map the field's gradient onto an
  "ls up" step. Clearing the history at the overshoot recovers the fit. From the prior-centred
  start it does not happen.

Departures from DADVI: `K = 5` (theirs 30), the trust-region details above, and the read-out's fallbacks (`"lr_lbfgs"`, `"diag"`). The study is `tests/experiments/writeups/dadvi_init.md`.

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
