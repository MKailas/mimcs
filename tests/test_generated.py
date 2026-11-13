"""Derived quantities through `Model` and the sampler loop: `GeneratedQuantities`.

What matters, hardest first:

* **Adding them changes nothing about the chain.** Their key is appended to the sampler's draw
  components, which is stream-neutral only because `jax.random.split(key, n)` is prefix-stable ---
  so the parameter draws and every per-transition diagnostic must be *bit-identical* with and
  without them, including across a buffer refill and under parallel tempering.
* **They are the draw's.** Each saved value is `fn` of the saved draw (the cold chain's under PT),
  with a fresh key per draw.
* **Layouts survive.** Integer outputs stay integer, shapes come out `(n, *shape)`, and the
  keywords that drop them restore exactly the old dict.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

import mimcs
from mimcs.model import Model, EuclideanParameter, PositiveParameter, GeneratedQuantities
from mimcs.testing import nuts


def _model(generated=None):
    return Model([EuclideanParameter("x", (2,)), PositiveParameter("s")],
                 {"lp": lambda d: -0.5 * jnp.sum(d["x"] ** 2) - d["s"]}, generated=generated)


def _gq(rng=True):
    def fn(v, key):
        out = {"x_sum": jnp.sum(v["x"]), "x_sq": v["x"] ** 2,
               "big": (v["s"] > 1.0).astype(jnp.int32)}
        if rng:
            out["noise"] = jax.random.normal(key, (3,))
            out["count"] = jax.random.poisson(key, 4.0)
        return out
    kinds = {"x_sum": "transformed", "x_sq": "transformed", "big": "generated"}
    if rng:
        kinds.update(noise="generated", count="generated")
    return GeneratedQuantities(fn, kinds, rng=rng)


def _run(model, n_warm=150, n=300, seed=2, **kw):
    s = nuts(**kw)(model, seed)
    s.warmup(n_warm)
    s.sample(n)
    return s


# --- the Model contract ---------------------------------------------------- #

def test_layout_is_traced_and_integers_get_their_own_block():
    m = _model(_gq())
    assert m.generated_shapes == {"x_sum": (), "x_sq": (2,), "big": (), "noise": (3,),
                                  "count": ()}
    assert m.generated_float_dim == 1 + 2 + 3 and m.generated_int_dim == 2
    fl, it = m.generated_flat(m.default_sample(), None, jnp.zeros(2, jnp.uint32))
    assert fl.shape == (6,) and it.dtype == jnp.int32 and it.shape == (2,)
    assert m.generated_ambient_names("x_sq") == ["x_sq[1]", "x_sq[2]"]
    assert [c.name for c in m.generated_draw_components] == ["generated_key"]
    assert _model(_gq(rng=False)).generated_draw_components == []


def test_a_derived_name_may_not_reuse_a_parameter_name():
    gq = GeneratedQuantities(lambda v, k: {"s": v["s"]}, {"s": "transformed"})
    with pytest.raises(ValueError, match="also parameter names"):
        _model(gq)


def test_an_unknown_kind_is_refused():
    with pytest.raises(ValueError, match="unknown generated-quantity kind"):
        GeneratedQuantities(lambda v, k: {}, {"a": "posterior"})


def test_a_trace_error_surfaces_at_sampler_construction_not_after_warmup():
    gq = GeneratedQuantities(lambda v, k: {"bad": v["x"] @ jnp.ones(5)}, {"bad": "generated"})
    with pytest.raises(TypeError):
        nuts()(_model(gq), 0)


def test_ambient_and_feature_blocks_cover_their_tables():
    m = _model()
    assert m.ambient_blocks == [("x", 0, 2), ("s", 2, 3)]
    assert m.feature_blocks == [("x", 0, 4), ("s", 4, 6)]
    assert m.ambient_blocks[-1][2] == len(m.ambient_names)
    assert m.feature_blocks[-1][2] == len(m.feature_names)


# --- the sampler loop ------------------------------------------------------- #

def test_the_chain_is_bit_identical_with_and_without_derived_quantities():
    """The column that would catch a stream shift: buffer 64 so the 450 steps cross six refills."""
    a = _run(_model(), buffer_size=64)
    b = _run(_model(_gq()), buffer_size=64)
    assert np.array_equal(a.get_samples_flat(), b.get_samples_flat())
    da, db = a.diagnostics("all"), b.diagnostics("all")
    assert da.keys() == db.keys()
    assert all(np.array_equal(da[k], db[k]) for k in da)


def test_saved_values_are_the_functions_of_the_saved_draws():
    s = _run(_model(_gq()))
    d = s.get_samples()
    assert d["x_sq"].shape == (300, 2) and d["noise"].shape == (300, 3)
    np.testing.assert_array_equal(d["x_sq"], (d["x"] ** 2).astype(np.float32))
    np.testing.assert_allclose(d["x_sum"], d["x"].sum(1), rtol=1e-6)
    np.testing.assert_array_equal(d["big"], (d["s"] > 1.0).astype(np.int32))
    assert d["big"].dtype == np.int32 and d["count"].dtype == np.int32
    # a fresh key per draw: no two draws share their noise (control: one key repeats exactly)
    assert len({tuple(r) for r in d["noise"]}) == 300
    key = jnp.zeros(2, jnp.uint32)
    assert np.array_equal(jax.random.normal(key, (3,)), jax.random.normal(key, (3,)))


def test_the_variates_have_the_distribution_they_claim():
    """Poisson(4) over 600 draws: mean within 4 MCSE. Control: the same draws reject a rate of 5."""
    d = _run(_model(_gq()), n=600).get_samples()
    c = d["count"].astype(float)
    se = c.std() / np.sqrt(len(c))
    assert abs(c.mean() - 4.0) < 4 * se
    assert abs(c.mean() - 5.0) > 4 * se
    z = d["noise"].ravel()
    assert abs(z.mean()) < 4 / np.sqrt(z.size) and abs(z.var() - 1) < 0.15


def test_keywords_drop_them_and_save_generated_false_skips_them():
    s = _run(_model(_gq()))
    assert set(s.get_samples(transformed=False)) == {"x", "s", "big", "noise", "count"}
    assert set(s.get_samples(generated=False)) == {"x", "s", "x_sum", "x_sq"}
    plain = s.get_samples(transformed=False, generated=False)
    assert set(plain) == {"x", "s"}
    off = _run(_model(_gq()), save_generated=False)
    assert set(off.get_samples()) == {"x", "s"} and not off._generated_float
    assert np.array_equal(off.get_samples_flat(), s.get_samples_flat())
    assert off.get_generated_flat()[0].shape == (0, 6)


def test_the_stores_hold_copies_not_views():
    s = _run(_model(_gq()), n=20)
    assert all(a.flags["OWNDATA"] for a in s._generated_float + s._generated_int)


def test_parallel_tempering_evaluates_the_cold_chain_even_when_keeping_every_rung():
    m = _model(GeneratedQuantities(lambda v, k: {"x_copy": v["x"]}, {"x_copy": "transformed"}))
    for keep in (False, True):
        s = mimcs.pt.parallel_tempering(m, n_temperatures=3, seed=1, keep_all_temperatures=keep)
        s.warmup(60)
        s.sample(80)
        d = s.get_samples()
        np.testing.assert_array_equal(d["x_copy"], d["x"])
