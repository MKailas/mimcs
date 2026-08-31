# Model Interface

## Purpose

The `Model` class is the bridge between the user's probabilistic model and the MCMC sampler. It provides:

1. A list of typed parameters (with their manifold structure).
2. A decomposed log-probability function, returning named components rather than a single scalar.
3. The machinery to convert between the sampler's flat `coordinate` / `sample` vectors and the structured parameter space the user works in.

## Why Decomposed Log-Prob

Some advanced samplers exploit the structure of the log-prob:

- **Tempered transitions / parallel tempering**: operate on `β * log_likelihood + log_prior`.
- **SMC-within-MCMC hybrids**: require the likelihood separately.
- **Bouncy Particle Sampler**: may exploit conditional independence structure.
- **Riemannian geometry**: the Fisher information metric is the Hessian of the log-likelihood, not the full log-prob.

A single fused `log_prob` function is always available for samplers that don't need decomposition, but the decomposed form is the primary interface so that no information is thrown away.

## The `Model` Class

```python
class Model:
    """
    A probabilistic model: a collection of parameters and a log-prob decomposition.
    """

    def __init__(
        self,
        parameters: list[BaseParameter],
        log_prob_fns: dict[str, Callable[[dict[str, Array]], Array]],
        *,
        cheap_components: Iterable[str] = (),
        discrete_parameters: Iterable[BaseDiscreteParameter] = (),
    ):
        """
        Args:
            parameters: ordered list of model parameters (BaseParameter subclasses)
            log_prob_fns: named log-prob components, e.g.
                {
                    "log_prior": fn,
                    "log_likelihood": fn,
                }
                Each fn takes a dict {param_name: ambient_array} and returns a scalar.
            cheap_components: the components whose *gradient* is cheap -- what a multi-rate
                (RESPA) integrator sub-steps on (doc 06). Metadata about evaluation cost only:
                it never changes the density. Everything unnamed is expensive, and the empty
                default means "nothing is known cheap", under which a multi-rate builder finds
                nothing to nest. A DSL-compiled model gets this from its `ModelSpec` (doc 08);
                a hand-written one passes it here.
            discrete_parameters: the integer-valued parameters (doc 14). A *separate* list with a
                separate flat `int` layout, contributing to neither `coord_dim` nor `ambient_dim`
                -- a discrete parameter has no chart and no gradient, so nothing that partitions
                coordinates, adapts a mass or differentiates a density should ever see one. The
                log-density components do: they are functions of a `{name: value}` dict, and the
                discrete values are simply further entries in it.
        """
        self.parameters = parameters
        self._log_prob_fns = log_prob_fns
        self._layout = _build_layout(parameters)   # offsets and shapes for flat ↔ structured

    # --- Primary interface ---

    def log_prob_components(
        self, sample_dict: dict[str, Array]
    ) -> dict[str, Array]:
        """
        Evaluate each named log-prob component at the given parameter values.
        Returns a dict of scalars.
        """
        return {name: fn(sample_dict) for name, fn in self._log_prob_fns.items()}

    def log_prob(self, sample_dict: dict[str, Array]) -> Array:
        """Sum of all log-prob components. Convenience method."""
        return sum(self.log_prob_components(sample_dict).values())

    # --- Flat-vector interface (used by kernel) ---

    def log_prob_flat(self, sample_flat: Array) -> Array:
        """
        Evaluate the model's own total log-prob at a flat ambient vector.

        NOTE this is the *ambient* density: it takes no chart arguments and applies
        NO Jacobian correction. The coordinate-space target the kernel actually
        differentiates is log_prob_at_coordinate (below).
        """
        return self.log_prob(self.unpack_sample(sample_flat))

    def log_prob_at_coordinate(
        self,
        coord_flat: Array,
        chart_hyperparams: tuple,
        chart_indices: tuple,
    ) -> Array:
        """
        The coordinate-space target: log pi(from_coordinate(q)) + log|J|.

        This is the density the sampler's Markov kernel is reversible with respect
        to, and what state.log_prob stores. Differentiable in coord_flat
        (chart_hyperparams and chart_indices are non-differentiated inputs).
        """
        sample_dict = self.unpack_coordinate(coord_flat, chart_hyperparams, chart_indices)
        return (self.log_prob(sample_dict)
                + self.total_log_jacobian(coord_flat, sample_dict,
                                          chart_hyperparams, chart_indices))

    def log_prob_components(self, sample_dict: dict) -> dict[str, Array]:
        """
        Per-component log-densities, keyed by component name. The Jacobian is a
        separate potential (JacobianPotential, doc 06), not an entry here.
        """
        return {name: fn(sample_dict) for name, fn in self.log_prob_fns.items()}

    # --- Pack / unpack helpers ---

    def pack_sample(self, sample_dict: dict[str, Array]) -> Array:
        """Flatten structured parameter dict to a single 1D ambient vector."""
        return jnp.concatenate([sample_dict[p.name].ravel() for p in self.parameters])

    def unpack_sample(self, sample_flat: Array) -> dict[str, Array]:
        """Inverse of pack_sample."""
        out = {}
        offset = 0
        for p in self.parameters:
            size = math.prod(p.ambient_shape)
            out[p.name] = sample_flat[offset:offset + size].reshape(p.ambient_shape)
            offset += size
        return out

    def pack_coordinate(
        self,
        sample_dict: dict[str, Array],
        chart_hyperparams: tuple,
        chart_indices: tuple,
    ) -> Array:
        """
        Convert structured ambient dict to flat coordinate vector,
        applying each parameter's chart map with the given hyperparams and indices.
        """
        parts = []
        for i, p in enumerate(self.parameters):
            parts.append(
                p.to_coordinate(sample_dict[p.name], chart_hyperparams[i], int(chart_indices[i]))
            )
        return jnp.concatenate(parts)

    def unpack_coordinate(
        self,
        coord_flat: Array,
        chart_hyperparams: tuple,
        chart_indices: tuple,
    ) -> dict[str, Array]:
        """
        Convert flat coordinate vector to structured ambient dict,
        applying each parameter's inverse chart map.
        """
        out = {}
        offset = 0
        for i, p in enumerate(self.parameters):
            coords = coord_flat[offset:offset + p.coord_dim]
            out[p.name] = p.from_coordinate(coords, chart_hyperparams[i], int(chart_indices[i]))
            offset += p.coord_dim
        return out

    @property
    def coord_dim(self) -> int:
        """Total coordinate-space dimension (intrinsic dimensionality)."""
        return sum(p.coord_dim for p in self.parameters)

    @property
    def ambient_dim(self) -> int:
        """Total ambient dimension."""
        return sum(math.prod(p.ambient_shape) for p in self.parameters)

    # --- Jacobian ---

    def _total_log_jacobian(
        self,
        sample_flat: Array,
        chart_hyperparams: tuple,
        chart_indices: tuple,
    ) -> Array:
        """Sum of log_jacobian_det over all parameters."""
        total = jnp.zeros(())
        offset = 0
        for i, p in enumerate(self.parameters):
            # Compute coordinate for this parameter (needed when log_jacobian_det
            # depends on the coordinate, e.g. stereographic projection)
            x = sample_flat[self._sample_offsets[i]:self._sample_offsets[i+1]]
            x = x.reshape(p.ambient_shape)
            u = p.to_coordinate(x, chart_hyperparams[i], int(chart_indices[i]))
            total = total + p.log_jacobian_det(u, chart_hyperparams[i], int(chart_indices[i]))
            offset += p.coord_dim
        return total
```

## Constructing a `Model`

The user constructs a model by defining parameters and log-prob components as pure JAX functions:

```python
import jax.numpy as jnp
from mimcs.model import Model, EuclideanParameter, PositiveParameter

mu = EuclideanParameter("mu", shape=(3,))
sigma = PositiveParameter(name="sigma")

def log_prior(params):
    return (
        -0.5 * jnp.dot(params["mu"], params["mu"])          # N(0, I) prior on mu
        - params["sigma"]                                     # Exp(1) prior on sigma
    )

def log_likelihood(params):
    # data is captured from outer scope (pure function over params)
    residuals = (data - params["mu"]) / params["sigma"]
    return -0.5 * jnp.sum(residuals ** 2) - data.shape[0] * jnp.log(params["sigma"])

model = Model(
    parameters=[mu, sigma],
    log_prob_fns={
        "log_prior": log_prior,
        "log_likelihood": log_likelihood,
    },
)
```

## Design Constraints

### Log-prob functions must be JAX-traceable

All functions in `log_prob_fns` are called inside `log_prob_at_coordinate`, which is in turn called from the kernel (under `jax.jit` and potentially `jax.grad`). They must be pure JAX functions: no Python control flow that depends on array values, no side effects.

Data (observations) should be captured as constants from the enclosing scope, as in the example above. Do not pass data as a runtime argument to the model.

### Grad of the coordinate-space target

HMC-family samplers require the gradient of the *coordinate-space* target. `log_prob_flat` is the ambient density and is not what the kernel differentiates:

```python
grad_log_prob = jax.grad(model.log_prob_at_coordinate)  # w.r.t. the coordinate
```

Written out, that is the gradient of `log_prob ∘ from_coordinate + log|J|`:

```python
def log_prob_in_coordinate(coordinate, chart_hyperparams, chart_indices):
    return model.log_prob_at_coordinate(coordinate, chart_hyperparams, chart_indices)

grad_in_coord = jax.grad(log_prob_in_coordinate)
```

The Jacobian correction from `log_jacobian_det` is included in `log_prob_at_coordinate` and therefore appears in the gradient automatically. Chart hyperparameters are not differentiated (they are non-differentiated inputs to `jax.grad`), which is correct: they are constants during a single kernel call, updated only between steps by `postprocess`.

### Immutability of `Model`

A `Model` instance is effectively immutable after construction: parameters and log-prob functions do not change during a run. If a hierarchical model requires conditioning on hyperparameters, construct a new `Model` with the conditioned log-prob.

## Discrete parameters

`Model` carries **two** parameter lists. `parameters` are continuous and define the flat
`sample` / `coordinate` pair; `discrete_parameters` are integer-valued and define a flat `int`
block of their own, with `discrete_dim` and `discrete_block(name)` mirroring `coord_dim` and
`coord_block`.

Keeping them apart is what made discrete parameters a small change rather than a pervasive one:
the factory's block partitioner, every mass adaptation, the chart machinery and the score pullback
all iterate `model.parameters` and never encounter a discrete one, so none of them needed a guard.
The density needs no special treatment either, since `log_prob_fns` are pure functions of a
`{name: value}` dict and `unpack_coordinate` simply seeds that dict with the discrete values.

The methods that take the discrete block --- `log_prob_at_coordinate`, `features`,
`log_prob_flat` --- take it as a trailing optional argument. Passing `None` is fine when the model
has no discrete parameters and **raises** when it has: a defaulted-away block would evaluate the
density at stale labels, with the right shapes, the right dtypes and nothing raising. See doc 14.

## Relationship to `SamplerState`

The sampler kernel calls `model.log_prob_at_coordinate` to evaluate the target density and `model.sample_to_coordinate` / `model.coordinate_to_sample` to move between the two spaces. Both require `state.chart_hyperparams` and `state.chart_indices`. The model does not hold any reference to the sampler or the state; the flow of information is one-way:

```
state.sample, state.chart_hyperparams, state.chart_indices
    ──▶ model.log_prob_at_coordinate(...) ──▶ scalar (for kernel)

state.sample, state.chart_hyperparams, state.chart_indices
    ──▶ model.pack_coordinate(...) ──▶ state.coordinate
```
