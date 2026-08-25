"""Tests for ``mimcs/model/model.py``: the layout side of ``Model``.

``Model`` owns two layouts (ambient and coordinate) and the maps between the flat vectors the
sampler works in and the named, shaped quantities the model is written in. These check the
stacked unpacking that :meth:`mimcs.samplers.BaseSampler.get_samples` is built on.
"""

import numpy as np
import jax.numpy as jnp

from mimcs.model import (Model, EuclideanParameter, PositiveParameter, UnitVectorParameter,
                        SimplexParameter)


def _mixed_model():
    """Every shape that matters: a scalar, a vector, and a batched manifold parameter."""
    return Model(
        [EuclideanParameter("a"), EuclideanParameter("b", (3,)), PositiveParameter("c"),
         UnitVectorParameter("u", 3, (2,)), SimplexParameter("w", 4)],
        {"m": lambda v: jnp.zeros(())})


def test_unpack_draws_shapes_and_names():
    """A scalar comes out ``(n,)``, a vector ``(n, d)``, a batch of unit vectors ``(n, b, d)``."""
    model = _mixed_model()
    n = 7
    draws = np.arange(n * model.ambient_dim, dtype=float).reshape(n, model.ambient_dim)
    out = model.unpack_draws(draws)

    assert list(out) == ["a", "b", "c", "u", "w"]        # declaration order
    assert out["a"].shape == (n,)
    assert out["b"].shape == (n, 3)
    assert out["c"].shape == (n,)
    assert out["u"].shape == (n, 2, 3)
    assert out["w"].shape == (n, 4)


def test_unpack_draws_agrees_with_the_single_draw_unpacking():
    """The plural of ``unpack_sample``: row ``i`` of the stack must be that row unpacked."""
    model = _mixed_model()
    rng = np.random.default_rng(0)
    draws = rng.normal(size=(5, model.ambient_dim))
    stacked = model.unpack_draws(draws)
    for i in (0, 3):
        one = model.unpack_sample(jnp.asarray(draws[i], float))
        for name, value in one.items():
            assert np.allclose(stacked[name][i], np.asarray(value), atol=1e-5), name


def test_unpack_draws_preserves_the_array_type():
    """Numpy in, numpy out --- the sampler stores numpy, and plotting or ArviZ will want it.

    Written with plain slicing rather than ``jnp`` for exactly this reason; going through
    ``jnp.asarray`` would silently demote a float64 stack to float32 under the default config.
    """
    model = _mixed_model()
    draws = np.zeros((4, model.ambient_dim), dtype=np.float64)
    out = model.unpack_draws(draws)
    assert all(isinstance(v, np.ndarray) for v in out.values())
    assert all(v.dtype == np.float64 for v in out.values())

    jout = model.unpack_draws(jnp.zeros((4, model.ambient_dim), float))
    assert all(isinstance(v, jnp.ndarray) for v in jout.values())


def test_unpack_draws_handles_an_empty_stack():
    """No draws yet is a shape, not a special case --- what ``get_samples()`` returns before use."""
    model = _mixed_model()
    out = model.unpack_draws(np.empty((0, model.ambient_dim)))
    assert out["a"].shape == (0,) and out["b"].shape == (0, 3) and out["u"].shape == (0, 2, 3)
