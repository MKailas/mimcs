"""Tests for the RNG machinery (``mimcs/rng.py``, ``docs/design/03_rng_management.md``).

Everything random in the library comes through here: the kernel never draws, so a sampler declares
its per-step random variables as :class:`DrawComponent` objects and :class:`RNGBuffer` generates
them in batches and hands out one step's worth per transition. Roughly thirty test files depend on
that being deterministic, and until this file none of them asserted it.

Two properties are worth separating. **Determinism given a seed and a buffer size** is the contract
the seed-pinned suite rests on. **Independence from the buffer size is not a property at all** ---
draw ``i`` is served from refill ``i // B`` slot ``i % B``, and refill ``r`` uses the key after
``r + 1`` sequential splits, so changing ``B`` realigns the stream from the first refill onward.
That makes ``buffer_size`` part of the reproducibility contract, alongside ``seed``.

Fast by construction: tiny shapes, no sampling. ``mimcs.testing`` is deliberately not imported (it
pulls in matplotlib).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import mimcs
from mimcs.rng import DrawComponent, RNGBuffer, make_rng_draw_class, zero_draw


#: One of each interesting kind: a scalar, a vector, a 2-D component with a non-default generator
#: (shaped like NUTS's ``leaf_select``), and a non-default dtype. ``float16`` rather than
#: ``float64`` --- with x64 off the latter silently truncates to float32 and warns, so asserting on
#: it would be vacuous.
COMPONENTS = [
    DrawComponent("scalar", ()),
    DrawComponent("vector", (3,)),
    DrawComponent("matrix", (2, 4), jax.random.uniform),
    DrawComponent("half", (2,), jax.random.normal, jnp.float16),
]


def _model():
    """A one-parameter model, for the few tests that need a real sampler."""
    return mimcs.compile_model("parameters { real mu; }\nmodel { mu ~ normal(0, 1); }", {})


# --- the shape of a draw ------------------------------------------------------ #

def test_a_draw_has_exactly_the_declared_components():
    """No more, no fewer. (Key *order* is deliberately not asserted: jit canonicalises dict
    pytrees to sorted keys, so an order assertion would be testing jit, not mimcs.)"""
    draw = RNGBuffer(0, COMPONENTS, 8).next()
    assert set(draw) == {"scalar", "vector", "matrix", "half"}


def test_each_component_has_its_declared_shape_and_dtype():
    """The batch axis is stripped, exactly: a scalar component must come back ``()``, not ``(1,)``.
    A stray axis here would propagate into every kernel that reads the draw."""
    draw = RNGBuffer(0, COMPONENTS, 8).next()
    for c in COMPONENTS:
        assert draw[c.name].shape == c.shape, c.name
        assert draw[c.name].dtype == jnp.zeros(c.shape, c.dtype).dtype, c.name
    assert draw["scalar"].shape == ()


def test_a_custom_generator_is_called_with_the_key_and_the_batched_shape():
    """Pins the ``(key, shape) -> Array`` convention doc 03 advertises, and that exactly one
    leading axis is added."""
    seen = []

    def recording(key, shape):
        seen.append(shape)
        return jnp.full(shape, 2.5)

    draw = RNGBuffer(0, [DrawComponent("c", (2, 3), recording)], 5).next()
    assert seen == [(5, 2, 3)]                     # one batch axis, prepended
    assert draw["c"].shape == (2, 3)
    assert np.all(np.asarray(draw["c"]) == 2.5)


def test_two_components_of_the_same_shape_are_independent():
    """Each component gets its own subkey from ``split(key, n)``. The correlation check is the
    load-bearing half: plain inequality would pass if one were a shifted copy of the other."""
    comps = [DrawComponent("x", (4,)), DrawComponent("y", (4,))]
    buf = RNGBuffer(3, comps, 32)
    xs, ys = zip(*((buf.next()["x"], buf.next()["y"]) for _ in range(16)))
    X, Y = np.asarray(xs).ravel(), np.asarray(ys).ravel()
    assert not np.array_equal(X, Y)
    assert abs(np.corrcoef(X, Y)[0, 1]) < 0.5


# --- determinism -------------------------------------------------------------- #

def test_the_same_seed_and_buffer_size_give_the_same_stream():
    """The property the whole seed-pinned suite rests on. Twenty draws at ``B = 8`` crosses two
    refill boundaries, so this covers the key advance as well as the batch contents."""
    a, b = RNGBuffer(0, COMPONENTS, 8), RNGBuffer(0, COMPONENTS, 8)
    for _ in range(20):
        da, db = a.next(), b.next()
        for c in COMPONENTS:
            assert np.array_equal(np.asarray(da[c.name]), np.asarray(db[c.name])), c.name


def test_different_seeds_give_different_streams():
    """Checked on every component: one whose generator ignored its key would otherwise hide."""
    da, db = RNGBuffer(0, COMPONENTS, 8).next(), RNGBuffer(1, COMPONENTS, 8).next()
    for c in COMPONENTS:
        assert not np.array_equal(np.asarray(da[c.name]), np.asarray(db[c.name])), c.name


def test_the_cursor_advances_within_a_buffer():
    """A cursor stuck at slot 0 would re-serve one draw forever, and every other test here would
    still pass."""
    buf = RNGBuffer(0, [DrawComponent("vector", (3,))], 8)
    draws = np.stack([np.asarray(buf.next()["vector"]) for _ in range(8)])
    assert np.unique(draws, axis=0).shape[0] == 8


# --- the refill boundary ------------------------------------------------------ #

def test_the_first_buffer_is_filled_lazily():
    """Constructing a sampler must not allocate a batch it may never draw from."""
    buf = RNGBuffer(0, COMPONENTS, 8)
    assert buf._buffers is None and buf._refills == 0 and buf._cursor == 0
    buf.next()
    assert buf._refills == 1 and buf._buffers is not None


def test_the_buffer_refills_exactly_at_the_boundary():
    """Draw ``i`` comes from refill ``i // B`` slot ``i % B``. The guard is ``>=``, so a fencepost
    error here would silently re-serve a draw rather than fail --- hence the per-draw bookkeeping
    assertions, and the check that draw B differs from draw 0 (the key really advanced)."""
    B = 8
    buf = RNGBuffer(0, [DrawComponent("vector", (3,))], B)
    first = np.asarray(buf.next()["vector"])
    assert buf._refills == 1 and buf._cursor == 1
    for k in range(1, B):
        buf.next()
        assert buf._refills == 1, "refilled early"
        assert buf._cursor == k + 1
    assert buf._cursor == B                             # exactly full, still one refill

    at_boundary = np.asarray(buf.next()["vector"])      # the (B+1)-th draw triggers the refill
    assert buf._refills == 2 and buf._cursor == 1
    assert not np.array_equal(first, at_boundary), "the batch was re-served, not regenerated"


def test_a_buffer_of_one_refills_on_every_draw():
    """The degenerate lower bound must be correct, not merely non-crashing."""
    buf = RNGBuffer(0, [DrawComponent("vector", (3,))], 1)
    draws = np.stack([np.asarray(buf.next()["vector"]) for _ in range(5)])
    assert buf._refills == 5 and buf._cursor == 1
    assert np.unique(draws, axis=0).shape[0] == 5


# --- buffer_size and the draw stream ------------------------------------------ #

def test_jax_random_is_prefix_stable_in_the_batch_dimension():
    """Pinned in isolation, with no mimcs involved, because the next test leans on it.

    If this fails, JAX changed how it addresses the batch axis (``threefry_partitionable`` is the
    precedent) and the whole seed-pinned suite has moved."""
    key = jax.random.PRNGKey(0)
    assert np.array_equal(np.asarray(jax.random.normal(key, (16, 3))[:8]),
                          np.asarray(jax.random.normal(key, (8, 3))))
    assert np.array_equal(np.asarray(jax.random.uniform(key, (16, 2))[:4]),
                          np.asarray(jax.random.uniform(key, (4, 2))))


def test_buffer_size_shifts_the_stream_only_at_the_refill_boundary():
    """``buffer_size`` is a memory knob, but it is not stream-neutral: two buffers with the same
    seed agree for exactly ``min(B1, B2)`` draws and then diverge for good.

    CHARACTERIZATION, not a contract. Only the *divergence* is ours: refill ``r`` uses the key
    after ``r + 1`` sequential splits, so a different ``B`` realigns the keys from the second
    refill on, and nothing could make the streams agree past the first boundary. The *agreement*
    up to it is JAX's --- threefry generates the batch axis in counter mode, pinned separately in
    ``test_jax_random_is_prefix_stable_in_the_batch_dimension``. A JAX upgrade may drop that; if
    the first half of this test ever fails, check that one first, because the cause is upstream and
    the right fix is to relax this test rather than to change ``RNGBuffer``.

    Why it is worth pinning: ``buffer_size`` is part of the reproducibility contract. A seed-pinned
    result reproduces only at the same buffer size.
    """
    comps = [DrawComponent("vector", (3,))]
    a, b = RNGBuffer(0, comps, 4), RNGBuffer(0, comps, 8)
    agree = [np.array_equal(np.asarray(a.next()["vector"]), np.asarray(b.next()["vector"]))
             for _ in range(12)]
    assert all(agree[:4]), "the streams should agree until the smaller buffer refills"
    assert not agree[4], "the streams should diverge at the first refill boundary"
    assert not agree[11], "and stay diverged"


# --- zero_draw and the typed draw class --------------------------------------- #

def test_zero_draw_matches_the_structure_of_a_real_draw():
    """``zero_draw`` is what ``make_initial_state`` puts in ``state.rng_draw`` before step 1; if it
    disagrees with what ``preprocess`` injects afterwards, the jitted kernel retraces or fails at
    the first step. Compared leaf by leaf: ``tree_structure`` alone checks neither shape nor
    dtype, so asserting only that would be the vacuous version."""
    cls = make_rng_draw_class("T", COMPONENTS)
    real = cls(**RNGBuffer(0, COMPONENTS, 8).next())
    zeros = zero_draw(cls, COMPONENTS)
    assert jax.tree_util.tree_structure(zeros) == jax.tree_util.tree_structure(real)
    for z, r, c in zip(jax.tree_util.tree_leaves(zeros),
                       jax.tree_util.tree_leaves(real), COMPONENTS):
        assert z.shape == r.shape == c.shape, c.name
        assert z.dtype == r.dtype, c.name
        assert np.all(np.asarray(z) == 0), c.name


def test_the_draw_class_fields_are_in_declaration_order():
    """A real contract, not a formality: ``_init_momentum_draw`` zips ``split(key, n)`` against
    ``self._draw_components`` *positionally*, and pytree leaf order follows ``_fields`` rather than
    the draw dict's sorted keys."""
    cls = make_rng_draw_class("T", COMPONENTS)
    assert cls._fields == tuple(c.name for c in COMPONENTS)
    real = cls(**RNGBuffer(0, COMPONENTS, 8).next())
    assert len(jax.tree_util.tree_leaves(real)) == len(COMPONENTS)
    assert [leaf.shape for leaf in jax.tree_util.tree_leaves(real)] == [c.shape for c in COMPONENTS]


def test_a_draw_dict_populates_the_typed_class_by_name():
    """``preprocess`` does ``self._rng_draw_class(**raw)``, so the dict's sorted key order must not
    matter. Asserted by identity, so it fails if anyone turns the dict into a positional tuple."""
    draw = RNGBuffer(0, COMPONENTS, 8).next()
    typed = make_rng_draw_class("T", COMPONENTS)(**draw)
    assert typed.vector is draw["vector"] and typed.scalar is draw["scalar"]


# --- duplicate component names ------------------------------------------------ #

def test_duplicate_component_names_are_rejected():
    """Two kinetics sharing an ``id`` is the realistic way to reach this, since
    ``KineticHamiltonian.make_draw_components`` namespaces its draw as ``f"{id}_momentum"``. Both
    guards are checked: the namedtuple (which a sampler hits first) and the buffer's own."""
    dupes = [DrawComponent("a", ()), DrawComponent("a", (3,))]
    with pytest.raises(ValueError, match="duplicate field name"):
        make_rng_draw_class("T", dupes)
    with pytest.raises(ValueError, match="duplicate draw component name"):
        RNGBuffer(0, dupes, 4)


def test_two_kinetics_with_the_same_id_are_rejected_at_construction():
    """The user-facing symptom, named after the draw so the message points at the cause."""
    from mimcs.hmc import HMC, make_kinetic

    with pytest.raises(ValueError, match="T_momentum"):
        HMC(_model(), jnp.zeros(1),
            kinetics=[make_kinetic("diagonal"), make_kinetic("diagonal")])


# --- validation --------------------------------------------------------------- #

@pytest.mark.parametrize("bad", [0, -1])
def test_a_non_positive_buffer_size_is_rejected_at_construction(bad):
    """At construction is the point. Unchecked, ``0`` refills on every call and then indexes a
    zero-length array --- an ``IndexError`` raised from ``next()`` mid-warmup, pointing nowhere
    near the argument that caused it --- and a negative one dies inside XLA lowering."""
    with pytest.raises(ValueError, match="buffer_size"):
        RNGBuffer(0, COMPONENTS, bad)


def test_buffer_size_is_readable_and_read_only():
    """Read-only because it is baked into the JIT-compiled replenish function at construction, so
    changing it afterwards would desync the cursor bound from the batch length."""
    buf = RNGBuffer(0, COMPONENTS, 16)
    assert buf.buffer_size == 16
    with pytest.raises(AttributeError):
        buf.buffer_size = 32


def test_a_sampler_rejects_a_bad_buffer_size_at_construction():
    """Now that the factory exposes the knob, a bad value is reachable from user config."""
    from mimcs.hmc import HMC

    with pytest.raises(ValueError, match="buffer_size"):
        HMC(_model(), jnp.zeros(1), buffer_size=0)
