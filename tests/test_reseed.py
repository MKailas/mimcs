"""Tests for the cache reseed --- the one place potential values and gradients are refreshed
outside the kernel.

Chart adaptation and ladder adaptation both move the *target as a function of the coordinate*, so
the cached ``V_i`` and ``grad V_i`` they inherit describe a Hamiltonian nobody is integrating: the
next leading half-kick reads that gradient back out (``cached_gradient=True``) and ``log_prob`` is
the acceptance baseline. :meth:`BaseHMC._reseed_caches` is what discharges that, and it runs under
a cached ``jax.jit``.

**A wrong reseed does not raise.** It returns arrays of the right shape and dtype holding numbers
that look plausible, and the chain quietly integrates the wrong thing --- on ``hmm_gaussian`` a
stale ladder cache collapsed acceptance to zero and drove the step size to 1e-13 in 25 iterations
(``mimcs.pt.LadderAdaptation._reseed_at_new_betas``). These tests are the guard, and they check the
properties that a plausible-looking wrong answer would violate.

Seeds are fixed, so pass/fail is deterministic.
"""

import numpy as np
import jax
import jax.numpy as jnp

from mimcs.hmc.integrators import init_integrator_state
from mimcs.model import Model, EuclideanParameter, UnitVectorParameter
from mimcs.pt import parallel_tempering
from mimcs.testing import nuts
from mimcs.testing.problems import correlated_gaussian, von_mises_fisher


def _centered_model(d=3):
    def log_post(params):
        return -0.5 * jnp.sum(params["x"] ** 2)
    return Model([EuclideanParameter("x", (d,), centered=True)], {"lp": log_post})


def _sphere_model(d=3):
    def log_post(params):
        return 5.0 * params["u"][0]
    return Model([UnitVectorParameter("u", d)], {"lp": log_post})


def _eager(sampler, coordinate, ctx):
    """The reseed as it was written before it was compiled --- the reference."""
    seed = init_integrator_state(sampler.potentials, coordinate,
                                 jnp.zeros_like(coordinate), ctx)
    return seed.potential_values, seed.potential_grads


def _assert_same(a, b, *, rtol=1e-5, atol=1e-6):
    """Per potential id, not on the sum: a sign error in one component can cancel in the total."""
    va, ga = a
    vb, gb = b
    assert set(va) == set(vb) and set(ga) == set(gb)
    for pid in va:
        assert np.allclose(np.asarray(va[pid]), np.asarray(vb[pid]), rtol=rtol, atol=atol), pid
        assert np.allclose(np.asarray(ga[pid]), np.asarray(gb[pid]), rtol=rtol, atol=atol), pid


# --- the compiled reseed agrees with the eager one it replaced --------------------- #

def test_compiled_reseed_matches_the_eager_one_on_every_chart_type():
    for name, model in [("euclidean", correlated_gaussian().model),
                        ("centered", _centered_model()),
                        ("unit_vector", _sphere_model())]:
        s = nuts(terminate=None)(model, 0)
        s.initialize()
        ctx = s.context(s.state)
        _assert_same(s._reseed(s.state.coordinate, ctx),
                     _eager(s, s.state.coordinate, ctx))


def test_compiled_reseed_matches_the_eager_one_for_a_moved_ladder():
    s = parallel_tempering(correlated_gaussian().model, n_temperatures=4, beta_min=0.05, seed=0)
    s.warmup(20)                                  # let the ladder move off its initial geometry
    ctx = s.context(s.state)
    assert ctx.betas is not None, "a tempered context must carry the ladder"
    _assert_same(s._reseed(s.state.coordinate, ctx), _eager(s, s.state.coordinate, ctx))


# --- the arguments are traced, not baked in ---------------------------------------- #

def test_the_chart_is_not_baked_into_the_compiled_reseed():
    """The failure this whole design is arranged against.

    Closing over the hyperparameters instead of passing them compiles the *first* chart in as a
    constant, so every later reseed refreshes the cache at a chart nobody is integrating --- right
    shapes, right dtypes, no error. Reseeding under two different charts must give two different
    answers, each matching a fresh eager computation at its own chart.
    """
    s = nuts(terminate=None)(_centered_model(), 0)
    s.initialize()
    q = s.state.coordinate
    chart_a = ((jnp.zeros(3), jnp.ones(3)),)                      # mu = 0, sigma = 1
    chart_b = ((jnp.full(3, 2.0), jnp.full(3, 3.0)),)             # a genuinely different chart
    got_a = s._reseed(q, s.context(s.state._replace(chart_hyperparams=chart_a)))
    got_b = s._reseed(q, s.context(s.state._replace(chart_hyperparams=chart_b)))
    _assert_same(got_a, _eager(s, q, s.context(s.state._replace(chart_hyperparams=chart_a))))
    _assert_same(got_b, _eager(s, q, s.context(s.state._replace(chart_hyperparams=chart_b))))
    # and the two must actually differ -- otherwise the test above passes vacuously
    same = all(np.allclose(np.asarray(got_a[0][k]), np.asarray(got_b[0][k])) for k in got_a[0])
    assert not same, "the chart made no difference: it has been compiled in as a constant"


def test_the_ladder_is_not_baked_into_the_compiled_reseed():
    """The same failure for parallel tempering, where it is worse: the tempered potential falls
    back to its *construction-time* ``self.betas`` whenever ``ctx.betas`` is ``None``, so a baked-in
    context would freeze the initial geometric ladder while the adaptation goes on reporting that
    it is moving them."""
    s = parallel_tempering(correlated_gaussian().model, n_temperatures=3, beta_min=0.1, seed=0)
    q = s.state.coordinate
    ctx = s.context(s.state)
    hot = ctx._replace(betas=jnp.asarray([1.0, 0.5, 0.1]))
    cold = ctx._replace(betas=jnp.asarray([1.0, 0.9, 0.8]))
    got_hot, got_cold = s._reseed(q, hot), s._reseed(q, cold)
    _assert_same(got_hot, _eager(s, q, hot))
    _assert_same(got_cold, _eager(s, q, cold))
    same = all(np.allclose(np.asarray(got_hot[0][k]), np.asarray(got_cold[0][k]))
               for k in got_hot[0])
    assert not same, "the ladder made no difference: it has been compiled in as a constant"


def test_the_reseed_is_compiled_once_not_once_per_call():
    """Building the ``jax.jit`` inside the call would retrace every iteration --- all of the risk
    of compiling and none of the benefit, and nothing would fail."""
    s = nuts(unit_vector_center=True, terminate=None)(von_mises_fisher(kappa=5.0).model, 0)
    s.initialize().warmup(120)                    # past ``unit_vector_min_samples``, so it reseeds
    assert s._reseed_jit._cache_size() == 1
