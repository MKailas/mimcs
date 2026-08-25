"""Random number management.

Implements the batched-generation strategy from ``docs/design/03_rng_management.md``:
a sampler declares the random variables it needs per step as a list of
:class:`DrawComponent` objects; :class:`RNGBuffer` generates those in large
batches and hands out one step's worth at a time.

The kernel never draws random numbers itself --- all randomness enters through a
typed ``rng_draw`` NamedTuple that ``preprocess`` writes into the sampler state.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array

from ._logging import get_logger

log = get_logger(__name__)

Generator = Callable[[Array, tuple], Array]


@dataclass(frozen=True)
class DrawComponent:
    """Specification of one named random variable drawn each step.

    Attributes:
        name: identifier; becomes a field of the typed ``rng_draw`` NamedTuple.
        shape: shape of *one step's* draw (the batch dimension is added by the buffer).
        generator: pure JAX function ``(key, shape) -> Array``. Any ``jax.random.*``
            function works directly; custom distributions follow the same signature.
        dtype: float dtype for the draw.
    """

    name: str
    shape: tuple
    generator: Generator = field(default=jax.random.normal)
    dtype: jnp.dtype = float          # canonical float: float32, or float64 under jax_enable_x64


def make_rng_draw_class(name: str, draw_components: list[DrawComponent]) -> type:
    """Build a NamedTuple class whose fields are the draw component names.

    The result is a valid JAX pytree (NamedTuples are registered automatically).
    """
    field_names = [c.name for c in draw_components]
    return collections.namedtuple(name, field_names)


def zero_draw(rng_draw_class: type, draw_components: list[DrawComponent]):
    """A draw of the right structure filled with zeros (placeholder before step 1)."""
    return rng_draw_class(
        **{c.name: jnp.zeros(c.shape, c.dtype) for c in draw_components}
    )


class RNGBuffer:
    """Pre-generated buffer of random numbers, replenished in batches.

    Args:
        seed: integer seed for the initial PRNG key.
        draw_components: what to generate each step. Names must be unique.
        buffer_size: number of steps to generate per batch. Memory is
            ``buffer_size * sum(prod(c.shape) for c in draw_components)`` floats --- which for
            NUTS is dominated by the ``leaf_select`` component and is therefore near-independent
            of the model's dimension (see ``docs/design/03``).

    **``buffer_size`` is not stream-neutral.** Draw ``i`` is served from refill ``i // B`` slot
    ``i % B``, and refill ``r`` uses the key after ``r + 1`` sequential splits, so two buffers with
    the same seed and different ``B`` agree for ``min(B1, B2)`` draws and then diverge for good. A
    seed-pinned result reproduces only at the same ``buffer_size``.
    """

    def __init__(
        self,
        seed: int,
        draw_components: list[DrawComponent],
        buffer_size: int = 1024,
    ):
        self._key = jax.random.PRNGKey(seed)
        self._components = list(draw_components)
        names = [c.name for c in self._components]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            # Unreachable through a sampler --- ``make_rng_draw_class`` builds a namedtuple first
            # and raises there --- but ``RNGBuffer`` is public, and used directly the dict
            # comprehension in ``_make_replenish_fn`` would just keep the last of each name.
            raise ValueError(f"duplicate draw component name(s): {duplicates}")
        self._buffer_size = int(buffer_size)
        if self._buffer_size < 1:
            # Checked here, at construction, because the natural failure is late and points
            # somewhere else: buffer_size=0 refills on every call and then indexes a zero-length
            # array, surfacing as an IndexError from ``next()`` mid-warmup, and a negative one
            # dies inside XLA lowering at the first draw.
            raise ValueError(f"buffer_size must be >= 1, got {buffer_size!r}")
        self._buffers: dict | None = None
        self._cursor: int = 0
        self._refills: int = 0
        self._replenish_jit = jax.jit(self._make_replenish_fn())

    @property
    def buffer_size(self) -> int:
        """Steps generated per batch. Read-only: it is baked into the JIT-compiled replenish
        function at construction, so changing it afterwards would desync the cursor bound."""
        return self._buffer_size

    def next(self) -> dict:
        """Return the next draw as ``{component_name: array_of_component_shape}``."""
        if self._buffers is None or self._cursor >= self._buffer_size:
            self._replenish()
        draw = {name: buf[self._cursor] for name, buf in self._buffers.items()}
        self._cursor += 1
        return draw

    def _replenish(self) -> None:
        self._refills += 1
        log.debug("RNG buffer refill %d: %d step(s) of %s", self._refills, self._buffer_size,
                  [c.name for c in self._components])
        self._key, subkey = jax.random.split(self._key)
        self._buffers = self._replenish_jit(subkey)
        self._cursor = 0

    def _make_replenish_fn(self) -> Callable[[Array], dict]:
        components = self._components
        n = len(components)
        buffer_size = self._buffer_size

        def replenish(key: Array) -> dict:
            subkeys = jax.random.split(key, n)
            return {
                c.name: c.generator(subkeys[i], (buffer_size, *c.shape)).astype(c.dtype)
                for i, c in enumerate(components)
            }

        return replenish
