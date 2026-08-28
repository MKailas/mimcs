# Random Number Management

## The JAX PRNG Model

JAX uses an explicit, functional PRNG: every call to a random function requires a `key`, and keys must be split before reuse. When a sampler runs for tens of thousands of iterations, dispatching individual draws one at a time accumulates significant overhead (one Python→XLA round-trip per draw).

**The solution:** generate random numbers in large batches and feed them into the kernel one step at a time through `state.rng_draw`. This amortizes dispatch overhead and keeps the kernel a pure, deterministic function.

## Design Overview

Two concerns are separated cleanly:

- **What** to generate per step — defined by each base sampler class via a list of `DrawComponent` objects.
- **How** to buffer and replenish — handled generically by `RNGBuffer`, which is agnostic to what it is generating.

The flow from construction to use:

```
BaseSampler.make_draw_components(model, **kwargs)
    │  returns list[DrawComponent]
    ▼
RNGBuffer(seed, draw_components, buffer_size)
    │  at replenishment: calls each component's generator with a fresh subkey
    │  next() returns dict[str, Array]
    ▼
preprocess:
    raw = buffer.next()
    state = state._replace(rng_draw=RngDrawClass(**raw))
    │
    ▼
kernel(state):
    state.rng_draw.momentum       # typed, named access
    state.rng_draw.slice_threshold
    ...
```

## `DrawComponent`

A `DrawComponent` specifies one named random variable that a sampler needs per step.

```python
from dataclasses import dataclass
from typing import Callable
from jax import Array

@dataclass(frozen=True)
class DrawComponent:
    name: str
    shape: tuple[int, ...]
    generator: Callable[[Array, tuple[int, ...]], Array]
    dtype: type = jnp.float32
```

- `name` identifies the variable in the typed draw struct and in the buffer dict.
- `shape` is the shape of *one step's* draw (not the buffered batch shape).
- `generator` is a pure JAX function with the signature `(key, shape) -> Array`. It must be compatible with `jax.jit`. Practically any `jax.random.*` function qualifies directly, and arbitrary distributions can be expressed as callables following this signature.

### Examples

```python
import jax
from functools import partial

# Standard normal draw, shape (d,)
DrawComponent("momentum", shape=(d,), generator=jax.random.normal)

# Uniform draw, scalar
DrawComponent("accept_threshold", shape=(), generator=jax.random.uniform)

# Uniform draws for tree sampling, shape (max_depth,)
DrawComponent("tree_select", shape=(max_depth,), generator=jax.random.uniform)

# Student-t proposal (df=3) — arbitrary callable with (key, shape) signature
def sample_student_t(key, shape, df=3.0):
    # reparameterize: t = N / sqrt(Chi²/df)
    key_n, key_g = jax.random.split(key)
    normal = jax.random.normal(key_n, shape)
    gamma = jax.random.gamma(key_g, df / 2.0, shape)
    return normal / jnp.sqrt(2.0 * gamma / df)

DrawComponent("proposal", shape=(d,), generator=partial(sample_student_t, df=3.0))
```

## `RNGBuffer`

`RNGBuffer` maintains a batch-generated buffer for each component and hands out one step's worth at a time.

```python
import collections
import jax
import jax.numpy as jnp

class RNGBuffer:
    """
    Generic random number buffer. Generates draw_components in batches;
    returns one step's worth per call to next().

    Args:
        seed: integer seed for the initial JAX PRNG key
        draw_components: list of DrawComponent defining what to generate
        buffer_size: number of steps to generate per batch (default 1024)
    """

    def __init__(
        self,
        seed: int,
        draw_components: list[DrawComponent],
        buffer_size: int = 1024,
    ):
        self._key = jax.random.PRNGKey(seed)
        self._components = draw_components
        self._buffer_size = buffer_size
        self._buffers: dict[str, Array] | None = None
        self._cursor: int = 0
        self._replenish_jit = jax.jit(self._make_replenish_fn())

    def next(self) -> dict[str, Array]:
        """Return the next draw: {component_name: array_of_component_shape}."""
        if self._buffers is None or self._cursor >= self._buffer_size:
            self._replenish()
        draws = {name: buf[self._cursor] for name, buf in self._buffers.items()}
        self._cursor += 1
        return draws

    def _replenish(self) -> None:
        self._key, subkey = jax.random.split(self._key)
        self._buffers = self._replenish_jit(subkey)
        self._cursor = 0

    def _make_replenish_fn(self):
        components = self._components
        buffer_size = self._buffer_size

        def replenish(key: Array) -> dict[str, Array]:
            subkeys = jax.random.split(key, len(components))
            return {
                comp.name: comp.generator(
                    subkeys[i], (buffer_size, *comp.shape)
                ).astype(comp.dtype)
                for i, comp in enumerate(components)
            }

        return replenish
```

The replenishment function is JIT-compiled once at construction time. This means each call to `_replenish` pays only the XLA kernel dispatch cost, not the tracing cost.

## How Base Sampler Classes Define Their Draws

Each base sampler class implements `make_draw_components` as an **instance** method — it must be, since it reads `self.kinetics` and `self.integrator`, both of which are set before `BaseSampler.__init__` runs. It also has the model (for dimensions) and the constructor kwargs:

```python
class BaseSampler:
    def make_draw_components(self, model: "Model", **kwargs) -> list[DrawComponent]:
        raise NotImplementedError
```

### Example: `RandomWalkMH`

```python
class RandomWalkMH(BaseSampler):
    def make_draw_components(self, model, **kwargs):
        d = model.coord_dim
        return [
            DrawComponent("proposal_noise", shape=(d,), generator=jax.random.normal),
            DrawComponent("accept_threshold", shape=(), generator=jax.random.uniform),
        ]
```

### Example: `HMC`

```python
class HMC(BaseSampler):
    def make_draw_components(self, model, **kwargs):
        d = model.coord_dim
        return [
            # N(0, I); mass matrix M applied as p = L @ z (Cholesky) inside kernel
            DrawComponent("momentum", shape=(d,), generator=jax.random.normal),
            DrawComponent("accept_threshold", shape=(), generator=jax.random.uniform),
        ]
```

### Example: `NUTS`

```python
class BaseNUTS(BaseHMC):
    def make_draw_components(self, model, **kwargs):
        comps = super().make_draw_components(model, **kwargs)   # per-kinetic momentum + accept
        J = self.max_tree_depth
        comps.append(DrawComponent("tree_direction", (J,), jax.random.uniform))
        comps.append(DrawComponent("tree_select", (J,), jax.random.uniform))
        comps.append(DrawComponent("leaf_select", (J, self._max_subtree), jax.random.uniform))
        # Only when the integrator is randomized (MarkovianLineSearchIntegrator): its per-leaf
        # coins. Declared *only* if asked for, so plain NUTS keeps the identical seed stream.
        n_rng = getattr(self.integrator, "n_rng_per_step", 0)
        if n_rng > 0:
            comps.append(DrawComponent(
                "line_search", (J, self._max_subtree, n_rng), jax.random.uniform))
        return comps
```

There is no `log_slice`: selection is multinomial over the tree rather than slice-based, so the
draws are one direction coin and one selection uniform per doubling, plus one per *leaf* of the
deepest possible subtree. `max_tree_depth` is fixed at construction, so all the shapes are static
and the kernel indexes into them; unused entries (for shallow trees) are simply not consumed.

Note the size: `leaf_select` alone is `J · 2^(J-1)` floats per step — 5120 at the default `J = 10`
— which dominates the buffer for any model of moderate dimension. See *Buffer Sizing* below.

## The Typed `RngDraw` Struct

The `RNGBuffer.next()` dict is converted to a typed NamedTuple before being written into the state. This gives the kernel named, type-checked access to its random inputs.

The NamedTuple class is generated automatically from the component list, so that the component specification is the single source of truth:

```python
import collections

def make_rng_draw_class(sampler_name: str, draw_components: list[DrawComponent]) -> type:
    """
    Build a NamedTuple class whose fields correspond to draw_components.
    The result is a valid JAX pytree.
    """
    field_names = [comp.name for comp in draw_components]
    return collections.namedtuple(f"{sampler_name}RngDraw", field_names)
```

`BaseSampler.__init__` calls this once and stores the result:

```python
class BaseSampler:
    def __init__(self, model, seed, buffer_size=1024, **kwargs):
        draw_components = self.make_draw_components(model, **kwargs)
        self._rng_draw_class = make_rng_draw_class(type(self).__name__, draw_components)
        self._rng_buffer = RNGBuffer(seed, draw_components, buffer_size)
        ...
```

The `preprocess` method then assembles the typed draw:

```python
class BaseSampler:
    def preprocess(self, state):
        raw = self._rng_buffer.next()                     # dict[str, Array]
        rng_draw = self._rng_draw_class(**raw)            # typed NamedTuple
        state = state._replace(rng_draw=rng_draw)
        state = self._preprocess_hooks(state)
        return state
```

## Adaptation and the Reparameterization Principle

Some adaptation strategies update distribution parameters that the kernel uses to interpret the draw. The clearest example is the mass matrix in HMC: the momentum distribution is N(0, M⁻¹), but M changes during adaptation.

The buffer always generates from a fixed distribution (N(0, I) for momentum), and the kernel applies the current transformation at runtime:

```python
# Inside HMC kernel: transform standard normal draw to correlated momentum
z = state.rng_draw.momentum            # N(0, I) draw from buffer
p = state.metric_chol @ z             # N(0, M⁻¹) if metric_chol = chol(M)^{-T}
```

This reparameterization principle keeps the buffer static (no need to invalidate or regenerate on adaptation updates) while allowing the effective distribution to change continuously with the mass matrix.

The same principle applies to proposal distributions in MH: if the proposal covariance is adapted, the buffer still generates N(0, I) and `preprocess` or the kernel scales by the current proposal covariance Cholesky factor.

## Distribution Summary by Sampler

| Sampler | Component | Distribution | Why |
|---|---|---|---|
| `RandomWalkMH` | `proposal_noise` | N(0, I) | Proposal covariance applied in kernel |
| `RandomWalkMH` | `accept_threshold` | U(0, 1) | MH acceptance comparison |
| `HMC` | `momentum` | N(0, I) | Mass matrix Cholesky applied in kernel |
| `HMC` | `accept_threshold` | U(0, 1) | MH acceptance comparison |
| `NUTS` | `{id}_momentum` | N(0, I) | As above |
| `NUTS` | `tree_direction` | U(0, 1) | Which way to extend at each doubling |
| `NUTS` | `tree_select` | U(0, 1) | Progressive selection between subtrees |
| `NUTS` | `leaf_select` | U(0, 1) | Multinomial selection within a subtree, `(J, 2^(J-1))` |
| `NUTS` | `line_search` | U(0, 1) | *Only* under a randomized integrator (`MarkovianLineSearchIntegrator`) |
| `RelativisticKinetic` | `{id}_mom_direction`, `{id}_mom_radius` | see note | Non-Gaussian; custom generator callable |
| `RiemannianKinetic` | `{id}_momentum` | N(0, I) | Position-dependent metric applied in kernel |

**Relativistic HMC note.** The relativistic kinetic energy requires a non-Gaussian momentum distribution (a multivariate Cauchy or related). This is implemented as a custom `generator` callable in the `DrawComponent`, with the same `(key, shape) -> Array` signature, so no changes to `RNGBuffer` are needed.

## Reproducibility

The draw sequence is fully determined by `seed`, the number of calls to `buffer.next()`, **and
`buffer_size`**. The last of those is easy to overlook and is the reason this section was wrong
until it was measured: draw `i` is served from refill `i // B` slot `i % B`, and refill `r` uses
the key after `r + 1` sequential splits, so two runs at the same seed and different `B` agree for
exactly `min(B₁, B₂)` draws and then diverge for good. (Measured through a NUTS sampler at seed 0:
`B` = 8 / 32 / 64 each match `B = 1024` for precisely the first `B` draws.) **So `buffer_size`
belongs beside `seed` whenever a seed-pinned result is quoted**, and 1024 is the pinned default the
existing suite was calibrated at.

The agreement up to the boundary — rather than divergence from draw 0 — is JAX's doing: threefry
addresses the batch axis in counter mode, so `normal(k, (16, 3))[:8] == normal(k, (8, 3))`. That is
an implementation detail rather than a documented guarantee, and `tests/test_rng.py` pins it
separately from the mimcs-level behaviour so that a JAX change is diagnosable rather than merely
confusing.

An earlier version of this section proposed `jax.random.fold_in(base_key, replenishment_count)` as
a way to decouple the sequence from the batch size. **It would not work**: keying by refill counter
still leaves draw `i` in refill `i // B`, so a different `B` still lands it on a different key.
Genuine independence needs per-*draw* key derivation (`fold_in(base_key, i)` for the global draw
index), which gives up part of the batching win. That is deliberately not done — the batching is
the point of this design — so the knob is documented as stream-affecting instead.

## Both ends of the buffer are compiled

Batching the *generation* is only half of it. Handing a step its draw means taking row `cursor`
out of every component, and doing that in Python dispatches an eager `slice`+`squeeze` per
component per step — four or five of them for a default NUTS draw, at ~0.44 ms all told against
**0.019 ms** for a single compiled gather. On a cheap target that was a larger share of the
iteration than the jitted kernel itself.

So `RNGBuffer` holds two compiled functions: `_replenish_jit` (once per `buffer_size` steps) and
`_gather_jit` (once per step). The cursor is passed to the gather as a **traced** argument. That
detail is load-bearing: as a static argument it would key the cache on its value and compile
`buffer_size` separate executables, which is worse than the eager indexing it replaced.
`tests/test_rng.py` asserts the cache stays at one entry.

The gather returns the same slices of the same buffers, so the stream is untouched — the draws are
bit-identical, which is what makes this safe to change under seed-pinned tests.

## Buffer Sizing

The allocation is `buffer_size × Σ_components prod(shape)` floats. Sizing guidance stated in terms
of the model's dimension `d` is misleading, because for NUTS `d` is usually the *smallest* term:

| sampler | floats per step | at `B = 1024`, float32 |
|---|---|---|
| RWMH / HMC | `d + 1` | 414 KB at `d = 100` |
| NUTS, `max_tree_depth = J` | `d + 2J + J·2^(J-1)` | **~21 MB** at `J = 10`, near-independent of `d` |
| NUTS + `markovian_line_search` | `d + 2J + (1+r)·J·2^(J-1)` | ~84 MB at `J = 10, r = 3` |

The NUTS row is dominated by the `leaf_select (J, 2^(J-1))` component — ~5240 floats per step at
the default `J = 10` — so raising `max_tree_depth` costs far more buffer than raising `d` does:
`J = 12` alone takes it to ~86 MB.

- **Default: 1024**, which amortizes dispatch cost to well under 1% of runtime.
- Below ~32 steps, dispatch overhead from the JIT call dominates the benefit.
- Above ~65536 steps, memory pressure grows without proportional gain.

The buffer size affects only performance and memory, **never the validity of a run** — every size
gives a correct i.i.d. stream. It does, however, change *which* stream (see Reproducibility above),
so it is not "safe to change" in the sense of leaving results comparable.

**Setting it.** `buffer_size` is a `BaseSampler` constructor argument, and on the factory route a
build-time keyword alongside `seed`: `make_sampler(model, seed=0, buffer_size=256)` or
`spec.build(seed=0, buffer_size=256)`. `None` means unspecified, leaving
`spec.algo_kwargs["buffer_size"]` (which has always reached the constructor) or the default in
force. `RNGBuffer` rejects sizes below 1 at construction, and exposes a read-only `buffer_size`.

## Thread Safety

`RNGBuffer` is not thread-safe. Each sampler instance (and each parallel chain) owns a separate `RNGBuffer` initialized with a distinct seed.
