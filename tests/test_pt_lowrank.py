"""Parallel tempering with a low-rank mass, and the Woodbury factors hoisted out of its loop.

Two things are new here, and only the first is an optimization.

**The combination itself had never been tested.** `docs/design/13` names low-rank as one of the
block kinds every temperature may carry, and the factory reaches it (a `lowrank` block puts
`LowRankAdaptation` in the per-temperature mixins), but no test ran the two together --- doc 13's
own mixed-block check used a diagonal and a dense block. So the end-to-end test below is a gap
being closed, independent of the caching.

**The cache.** `LowRankQuadraticKinetic.precompute` puts the `O(J^2 d)` Sherman--Morrison
recursion into `HamiltonianContext.kinetic_cache` once per kernel call, because XLA will not hoist
it out of the trajectory `while_loop` (v0.1.3). `ProductKinetic` defined no `precompute` and
rebuilt the lane context with three positional arguments, so a tempered run got neither half: the
factors were never computed and would have been dropped at the vmap boundary if they had been.

The vacuity traps are the reason for the shape of these tests, and one of them is specific to
tempering. At initialization `gamma = 0` makes `V` all zeros, the rank term inert, and *any* cache
whatsoever gives the same answer. Under tempering that is not a property of the run but of the
**rung**: a hot chain is wider, its whitened score covariance can have no eigenvalue above 1, and
`LowRankAdaptation` then leaves that rung at `V[k] = 0` while the cold one carries a live rank
term (observed on a 4-dimensional Gaussian at K=3 --- whether it happens depends on the target and
the ladder). A control that fires on only some rungs is therefore the correct outcome, and pinning
*which* is a sharper test than pinning that it fires somewhere at all.
"""

import numpy as np
import jax
import jax.numpy as jnp

from mimcs.hmc import LowRankQuadraticKinetic, DiagonalQuadraticKinetic, lowrank
from mimcs.hmc.state import IntegratorState, HamiltonianContext
from mimcs.adaptation import LowRankAdaptation, RobbinsMonroStepSize
from mimcs.pt import parallel_tempering, ProductKinetic
from mimcs.testing import correlated_gaussian, evaluate


def _random_masses(K, d, J, seed):
    """A genuine low-rank mass per temperature: `(D, V)` stacked on a leading K axis, `V != 0`."""
    rng = np.random.default_rng(seed)
    D = jnp.asarray(rng.uniform(0.5, 3.0, (K, d)))
    V = jnp.asarray(rng.standard_normal((K, J, d)) * 0.7)
    return D, V


def _dense(D_k, V_k):
    """The mass this row stands for, formed explicitly: `M = diag(D) + V^T V`."""
    D_k, V_k = np.asarray(D_k), np.asarray(V_k)
    return np.diag(D_k) + V_k.T @ V_k


def _hand_built(K, d, J, seed):
    """A `ProductKinetic` over a low-rank block, with a context carrying its own cache."""
    pk = ProductKinetic(LowRankQuadraticKinetic(id="T", rank=J), K, d)
    D, V = _random_masses(K, d, J, seed)
    bare = HamiltonianContext(chart_hyperparams={}, chart_indices={}, ham_params={"T": (D, V)})
    return pk, bare, bare._replace(kinetic_cache={"T": pk.precompute(bare)}), D, V


def _istate(p):
    return IntegratorState(q=jnp.zeros_like(p), p=p, potential_values={}, potential_grads={},
                           log_weight=jnp.zeros(()), integrator_data={})


def _pt_lowrank(model, K, J, seed=0, **kw):
    return parallel_tempering(
        model, n_temperatures=K, seed=seed,
        kinetics=[LowRankQuadraticKinetic(id="T", rank=J)],
        adapt_mixins=(LowRankAdaptation,), extra_mixins=(RobbinsMonroStepSize,), **kw)


def _stiff_gaussian(d, seed):
    """A correlated Gaussian whose leading directions are stiff enough to earn a rank term."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    spectrum = np.geomspace(8.0, 0.3, d)
    cov = Q @ np.diag(spectrum) @ Q.T
    return correlated_gaussian(mean=tuple(np.zeros(d)), cov=tuple(map(tuple, cov)))


# --- the factors are stacked per temperature --------------------------------- #

def test_product_precompute_stacks_the_inner_factors_per_temperature():
    """Row `k` of the cache must be the inner block's factors for row `k` of the mass.

    That correspondence is the whole safety argument, and it holds by construction only because
    `precompute` and `_lanes` vmap the *same* dict along the *same* axis.
    """
    K, d, J = 4, 6, 2
    pk, bare, ctx, D, V = _hand_built(K, d, J, seed=3)

    beta, t = ctx.kinetic_cache["T"]
    assert beta.shape == (K, J)
    assert t.shape == (K, J, d)
    for k in range(K):
        ref_beta, ref_t = lowrank.inv_factors(D[k], V[k])
        assert np.allclose(np.asarray(beta[k]), np.asarray(ref_beta), rtol=1e-6)
        assert np.allclose(np.asarray(t[k]), np.asarray(ref_t), rtol=1e-6)


def test_a_block_with_nothing_to_precompute_leaves_the_cache_empty():
    """A diagonal inner block yields `None`, and `context` must then store no entry at all.

    Storing `{"T": None}` would be worse than storing nothing: `LowRankQuadraticKinetic`'s cache
    lookup tests `self.id in cache`, so a `None` under a live id would be unpacked as a pair. This
    is also what keeps every non-low-rank tempered run's emitted graph unchanged.
    """
    pk = ProductKinetic(DiagonalQuadraticKinetic(id="T"), 3, 5)
    ctx = HamiltonianContext(chart_hyperparams={}, chart_indices={},
                             ham_params={"T": jnp.ones((3, 5))})
    assert pk.precompute(ctx) is None

    problem = _stiff_gaussian(d=4, seed=0)
    sampler = parallel_tempering(problem.model, n_temperatures=3, metric="diagonal", seed=0)
    assert sampler.context(sampler.state).kinetic_cache is None


# --- the cache reaches the lane, and is read there ---------------------------- #

def test_the_cached_and_inline_paths_agree_with_the_dense_mass():
    """Three ways to the same energy: through the cache, through the fallback, and through an
    explicitly formed `M`. The dense reference is what stops the first two agreeing on a shared
    mistake."""
    K, d, J = 4, 6, 2
    pk, bare, ctx, D, V = _hand_built(K, d, J, seed=4)
    p = jnp.asarray(np.random.default_rng(11).standard_normal(K * d))
    istate = _istate(p)

    cached = np.asarray(pk.per_temperature_energy(istate, ctx))
    inline = np.asarray(pk.per_temperature_energy(istate, bare))
    assert np.allclose(cached, inline, rtol=1e-6)

    pm = np.asarray(p).reshape(K, d)
    dense = np.array([0.5 * pm[k] @ np.linalg.solve(_dense(D[k], V[k]), pm[k]) for k in range(K)])
    assert np.allclose(cached, dense, rtol=1e-4, atol=1e-5)

    velocity = np.asarray(pk.velocity_into(jnp.zeros(K * d), istate, ctx)).reshape(K, d)
    for k in range(K):
        assert np.allclose(velocity[k], np.linalg.solve(_dense(D[k], V[k]), pm[k]),
                           rtol=1e-4, atol=1e-5)


def test_a_wrong_cache_changes_the_answer_in_every_lane():
    """The control. Without it every assertion above would pass just as well against a
    `precompute` whose result the lane silently ignores --- which is precisely the state of the
    world this change is fixing, so it is the one failure mode that must be excluded directly.

    Every rung is perturbed here because `_random_masses` gives every rung a live `V`; the
    adapted case, where some rungs have `V = 0`, is
    `test_a_wrong_cache_is_inert_exactly_where_the_rank_term_is`.
    """
    K, d, J = 4, 6, 2
    pk, bare, ctx, D, V = _hand_built(K, d, J, seed=5)
    istate = _istate(jnp.asarray(np.random.default_rng(12).standard_normal(K * d)))

    wrong = bare._replace(kinetic_cache={
        "T": jax.vmap(lambda D_k, V_k: lowrank.inv_factors(D_k * 7.0 + 1.0, V_k))(D, V)})
    cached = np.asarray(pk.per_temperature_energy(istate, ctx))
    spoiled = np.asarray(pk.per_temperature_energy(istate, wrong))
    assert not np.any(np.isclose(cached, spoiled)), \
        "the lane is not reading the cache: a wrong one made no difference"


# --- against a real, adapted, tempered mass ----------------------------------- #

def test_the_warmed_up_cache_is_consistent_with_its_own_mass():
    """`BaseHMC.context` fills the cache and `ham_params` from one state and nothing replaces
    either afterwards, so they cannot disagree; a stale cache would be a wrong `M^-1` of the right
    shape, which no shape check would catch.

    Warmed up first, and the cold rung's `V` asserted non-zero: at initialization `gamma = 0`
    makes the rank term inert and the check would hold whatever the cache contained.
    """
    problem = _stiff_gaussian(d=6, seed=0)
    sampler = _pt_lowrank(problem.model, K=4, J=2, seed=0)
    sampler.warmup(400)

    ctx = sampler.context(sampler.state)
    D, V = ctx.ham_params["T"]
    assert float(jnp.max(jnp.abs(V[0]))) > 0.0, "the cold rung's V is zero: the test is vacuous"

    beta, t = ctx.kinetic_cache["T"]
    for k in range(4):
        ref_beta, ref_t = lowrank.inv_factors(D[k], V[k])
        assert np.allclose(np.asarray(beta[k]), np.asarray(ref_beta), rtol=1e-6)
        assert np.allclose(np.asarray(t[k]), np.asarray(ref_t), rtol=1e-6)


def test_a_wrong_cache_is_inert_exactly_where_the_rank_term_is():
    """The per-rung refinement of the control: the cache must be sliced *per lane*, not shared.

    With `V = 0` the factors reduce to `t = 0`, `beta = 1` and `apply_inv_factored` returns
    `u / D` --- and `D` is read from `ham_params`, not from the cache. So a rung with no rank term
    is provably immune to a corrupted cache, while a rung with one is not. Asserting both
    directions in one context is sharper than "the control fired somewhere": a cache that leaked
    one lane's factors into another would move the inert rungs too.

    The mix is built by hand rather than adapted into existence. `LowRankAdaptation` does leave
    `gamma = 0` on hot rungs for some targets --- it is exactly the tempered form of the
    initialization trap this module's docstring describes --- but which rungs go inert depends on
    the target and the ladder, so relying on it would make the test's sharpness accidental.
    """
    K, d, J = 4, 6, 2
    pk = ProductKinetic(LowRankQuadraticKinetic(id="T", rank=J), K, d)
    D, V = _random_masses(K, d, J, seed=6)
    live = np.array([True, False, True, False])
    V = V * jnp.asarray(live, float)[:, None, None]        # rungs 1 and 3 get no rank term

    bare = HamiltonianContext(chart_hyperparams={}, chart_indices={}, ham_params={"T": (D, V)})
    ctx = bare._replace(kinetic_cache={"T": pk.precompute(bare)})
    wrong = bare._replace(kinetic_cache={
        "T": jax.vmap(lambda D_k, V_k: lowrank.inv_factors(D_k * 7.0 + 1.0, V_k))(D, V)})

    istate = _istate(jnp.asarray(np.random.default_rng(13).standard_normal(K * d)))
    cached = np.asarray(pk.per_temperature_energy(istate, ctx))
    spoiled = np.asarray(pk.per_temperature_energy(istate, wrong))

    # Still the dense truth, so a mass that silently lost its rank term cannot pass.
    dense = np.array([0.5 * np.asarray(istate.p).reshape(K, d)[k]
                      @ np.linalg.solve(_dense(D[k], V[k]),
                                        np.asarray(istate.p).reshape(K, d)[k]) for k in range(K)])
    assert np.allclose(cached, dense, rtol=1e-4, atol=1e-5)

    for k in range(K):
        if live[k]:
            assert not np.isclose(cached[k], spoiled[k]), \
                f"rung {k} has a live rank term but ignored a corrupted cache"
        else:
            assert np.isclose(cached[k], spoiled[k]), \
                f"rung {k} has V = 0, so the cache cannot matter, yet the answer moved"


# --- the opt-out that keeps the hoist from costing more than it saves --------- #

def test_the_cache_is_not_rebuilt_once_per_warmup_iteration():
    """The regression guard, and the reason `context` has a `kinetic_cache` switch at all.

    `kernel` is jitted, so `precompute` runs there once at trace time and is free. The reseeding
    callers --- chart adaptation, the swap, and above all the **tempered ladder**, which reseeds
    once per warmup iteration --- run *eagerly*, where the `O(J^2 d)` recursion dispatches
    primitive by primitive: 43.8 ms per call against 0.10 ms traced, at rank 8, d = 200, K = 4.
    Building a cache those callers never read made tempered warmup 3.6x *slower* (5.4 -> 19.3 s),
    turning the optimization into a regression three times the size of its own speedup.

    So the property to pin is not a wall-clock number, which would be flaky, but the shape of the
    cost: **eager `precompute` calls must not scale with the number of warmup iterations.** Two
    warmups differing 4x in length must make the same number of them.
    """
    counts = []
    for n_warmup in (5, 20):
        sampler = _pt_lowrank(_stiff_gaussian(d=4, seed=0).model, K=3, J=2, seed=0)
        block = sampler.kinetics[0]
        calls = []
        inner = block.precompute
        # An instance attribute shadows the class method, and `context` reads it with `getattr`.
        block.precompute = lambda ctx, _f=inner, _c=calls: (_c.append(1), _f(ctx))[1]
        sampler.warmup(n_warmup)
        counts.append(len(calls))

    assert counts[0] == counts[1], (
        f"eager precompute calls scale with warmup length ({counts[0]} at 5 iterations, "
        f"{counts[1]} at 20): a reseeding caller is building a kinetic cache it never reads")


def test_the_opt_out_changes_no_number():
    """`kinetic_cache=False` is a performance switch, so it must be invisible in the answers ---
    every consumer keeps an inline fallback. Pinning this is what makes it safe to add the flag at
    a new caller without re-deriving whether that caller touches the kinetics."""
    sampler = _pt_lowrank(_stiff_gaussian(d=5, seed=2).model, K=3, J=2, seed=0)
    sampler.warmup(200)

    with_cache = sampler.context(sampler.state)
    without = sampler.context(sampler.state, kinetic_cache=False)
    assert without.kinetic_cache is None
    assert with_cache.kinetic_cache is not None
    assert np.array_equal(np.asarray(without.betas), np.asarray(with_cache.betas))

    istate = _istate(jnp.asarray(np.random.default_rng(21).standard_normal(3 * 5)))
    block = sampler.kinetics[0]
    assert np.allclose(np.asarray(block.per_temperature_energy(istate, with_cache)),
                       np.asarray(block.per_temperature_energy(istate, without)), rtol=1e-6)


# --- and the combination samples correctly ------------------------------------ #

def test_parallel_tempering_with_a_lowrank_mass_is_correct(artifacts_dir):
    """The end-to-end check, through the same `evaluate` harness as the rest of the suite.

    PT's characteristic failure is a biased cold marginal that R-hat and ESS both pass, so the
    cold chain is compared against a distribution whose answer is known.
    """
    problem = _stiff_gaussian(d=5, seed=1)

    def build(model, seed):
        return _pt_lowrank(model, K=4, J=2, seed=seed, beta_min=0.05)

    report = evaluate(problem, {"pt_lowrank": build}, n_warmup=2000, n_samples=20000, seed=0,
                      out_dir=str(artifacts_dir / "pt_lowrank"))
    print("\n" + report.summary())
    report.assert_correct()
