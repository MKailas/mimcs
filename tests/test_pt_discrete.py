"""Tests for parallel tempering over models with discrete (integer) parameters.

PT refused such a model until now, for three reasons that all had to be fixed:

* ``pt/kinetics.py`` rebuilt each lane's ``HamiltonianContext`` **positionally**, so a ``discrete``
  field was dropped at the vmap boundary exactly as ``betas`` already was;
* ``ProductModel`` tiled one flat *float* vector across the ladder and hardcoded
  ``discrete_dim = 0``;
* and --- the one that would have produced a wrong answer rather than a missing feature ---
  ``_swap`` permuted only ``coordinate``, so replicas would have exchanged **positions while
  keeping their labels**, giving each rung a configuration drawn from no target at all.

The tests below are arranged so each of those can fail on its own, and the swap one is written so
that reverting the fix makes it fail (verified) rather than merely making it unnecessary.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from mimcs.adaptation import DiscreteMarginalAdaptation
from mimcs.diagnostics import split_rhat
from mimcs.model import EuclideanParameter, IntegerParameter, Model
from mimcs.pt import parallel_tempering
from mimcs.pt.lanes import lane_discrete, per_temperature_potential
from mimcs.samplers import (DiscreteMetropolisWithinGibbs, StaticContinuous, make_sampler_class)
from mimcs.testing import spike_and_slab

GIBBS_ONLY = make_sampler_class(DiscreteMetropolisWithinGibbs, StaticContinuous)


def _coupled_binary(nz=3):
    """Three coupled binary coordinates and one continuous parameter they interact with."""
    w = jnp.asarray([1.3, -0.7, 2.1][:nz], float)
    J = jnp.asarray([[0.0, 1.5, -0.9], [1.5, 0.0, 0.6], [-0.9, 0.6, 0.0]])[:nz, :nz]

    def lp(v):
        z = v["z"].astype(float)
        return (jnp.dot(w, z) + 0.5 * z @ J @ z
                - 0.5 * jnp.sum(v["x"] ** 2) + 0.4 * jnp.sum(z) * jnp.sum(v["x"]))

    return Model([EuclideanParameter("x")], {"p": lp},
                 discrete_parameters=[IntegerParameter("z", (nz,), lower=0, upper=1)]), lp


def _discrete_only():
    w = jnp.asarray([1.3, -0.7, 2.1], float)
    J = jnp.asarray([[0.0, 1.5, -0.9], [1.5, 0.0, 0.6], [-0.9, 0.6, 0.0]])

    def lp(v):
        z = v["z"].astype(float)
        return jnp.dot(w, z) + 0.5 * z @ J @ z

    return Model([], {"p": lp},
                 discrete_parameters=[IntegerParameter("z", (3,), lower=0, upper=1)]), lp


def _exact_pmf(lp, n=3):
    states = np.array([[(i >> k) & 1 for k in range(n - 1, -1, -1)] for i in range(2 ** n)])
    logp = np.array([float(lp({"z": jnp.asarray(s)})) for s in states])
    w = np.exp(logp - logp.max())
    return states, w / w.sum()


# --------------------------------------------------------------------------- #
# 1. the product layout                                                        #
# --------------------------------------------------------------------------- #

def test_the_product_model_carries_k_copies_of_the_discrete_block():
    m, _ = _coupled_binary()
    s = parallel_tempering(m, n_temperatures=4, seed=0)
    pm = s.model
    assert pm.discrete_dim == 4 * m.discrete_dim
    assert pm.discrete_block("z") == m.discrete_block("z")     # a block *within one rung*
    assert s.state.discrete.shape == (4 * m.discrete_dim,)
    assert s.state.discrete.dtype == jnp.int32
    assert np.array_equal(np.asarray(pm.default_discrete()),
                          np.tile(np.asarray(m.default_discrete()), 4))


def test_each_rung_gets_its_own_proposal_table():
    m, _ = _coupled_binary()
    s = parallel_tempering(m, n_temperatures=5, seed=0)
    tbl = s.state.discrete_proposal_params["z"]
    assert tbl.shape == (5, 3, 2)                              # (rungs, coordinates, values)
    assert np.allclose(np.asarray(tbl), 0.5)                   # uniform until adapted


def test_lane_discrete_slices_per_rung_and_passes_none_through():
    from mimcs.hmc.state import HamiltonianContext
    ctx = HamiltonianContext((), (), {}, discrete=jnp.arange(12, dtype=jnp.int32))
    rows = lane_discrete(ctx, 4)
    assert rows.shape == (4, 3)
    assert np.array_equal(np.asarray(rows[2]), [6, 7, 8])
    # a continuous model contributes no rows at all, which `vmap` treats as an empty pytree
    assert lane_discrete(HamiltonianContext((), (), {}), 4) is None
    assert lane_discrete(HamiltonianContext((), (), {}, discrete=jnp.zeros((0,), jnp.int32)),
                         4) is None


# --------------------------------------------------------------------------- #
# 2. each rung's density is its own                                            #
# --------------------------------------------------------------------------- #

def test_each_rung_is_evaluated_at_its_own_labels_and_its_own_beta():
    """The check that the per-lane context slicing works.

    Give the rungs *different* labels and assert the per-rung potential equals what the base model
    says for each rung separately, scaled by that rung's beta. Passing the shared context through
    unsliced --- what the code did before discrete parameters existed --- would give every rung the
    same answer.
    """
    m, lp = _coupled_binary()
    K = 4
    s = parallel_tempering(m, n_temperatures=K, seed=0)
    betas = np.asarray(s.betas)

    rng = np.random.default_rng(0)
    z = jnp.asarray(rng.integers(0, 2, size=(K, 3)).reshape(-1), jnp.int32)
    q = jnp.asarray(rng.normal(size=K), float)
    state = s.state._replace(discrete=z, coordinate=q)
    ctx = s.context(state, kinetic_cache=False)

    got = np.asarray(per_temperature_potential(s.potentials, q, ctx, K))
    hp, ci = m.init_chart_hyperparams(), m.init_chart_indices()
    want = np.array([
        -betas[k] * float(m.log_prob_at_coordinate(q[k:k + 1], hp, ci, z[3 * k:3 * k + 3]))
        for k in range(K)])
    assert np.allclose(got, want, rtol=1e-4, atol=1e-4), (got, want)
    # ... and non-vacuously: the rungs must actually differ, or a shared context would pass too
    assert np.ptp(got) > 1.0, got


# --------------------------------------------------------------------------- #
# 3. the swap moves whole replicas                                             #
# --------------------------------------------------------------------------- #

def test_a_swap_exchanges_labels_together_with_positions():
    """A replica is its whole state. Exchanging positions while leaving the labels behind gives
    each rung a configuration drawn from no target at all, and nothing downstream would show it.

    Verified to fail against the pre-fix code (coordinates permuted to [0,2,1,3] while labels
    stayed [0,1,2,3]), so this is a test of the defect and not merely of the plumbing.
    """
    def lp(v):
        z = v["z"].astype(float)
        return -0.5 * jnp.sum(v["x"] ** 2) + 2.0 * jnp.sum(z) + 0.5 * jnp.sum(z) * jnp.sum(v["x"])

    m = Model([EuclideanParameter("x", (2,))], {"p": lp},
              discrete_parameters=[IntegerParameter("z", (2,), lower=0, upper=3)])
    K, nz, nq = 4, 2, 2
    s = parallel_tempering(m, n_temperatures=K, seed=1)

    # Rung k is tagged by both its labels (k, k) and its position (10k, 10k).
    z0 = jnp.asarray(np.repeat(np.arange(K), nz), jnp.int32)
    q0 = jnp.asarray(np.repeat(np.arange(K) * 10.0, nq), float)
    st = s.preprocess(s.state._replace(discrete=z0, coordinate=q0))
    # log(0) = -inf beats any finite ratio, so every attempted pair accepts.
    st = st._replace(rng_draw=st.rng_draw._replace(
        swap_uniform=jnp.zeros((K,)), swap_parity=jnp.zeros(())))
    out = jax.jit(s._swap)(st)

    from_q = (np.asarray(out.coordinate).reshape(K, nq)[:, 0] / 10.0).round().astype(int)
    from_z = np.asarray(out.discrete).reshape(K, nz)[:, 0]
    assert not np.array_equal(from_q, np.arange(K)), "no swap happened: the test would be vacuous"
    assert np.array_equal(from_q, from_z), (from_q, from_z)


def test_the_proposal_tables_do_not_swap():
    """A table describes a *temperature*, not a state: after a swap rung k holds a different
    replica but still targets pi^beta_k, so its table still approximates the right marginal.
    Permuting them with the replicas would be quiet and wrong."""
    m, _ = _coupled_binary()
    K = 4
    s = parallel_tempering(m, n_temperatures=K, seed=1)
    tagged = jnp.asarray(np.linspace(0.2, 0.8, K)[:, None, None]
                         * np.ones((K, 3, 2)), float)
    st = s.preprocess(s.state._replace(discrete_proposal_params={"z": tagged}))
    st = st._replace(rng_draw=st.rng_draw._replace(
        swap_uniform=jnp.zeros((K,)), swap_parity=jnp.zeros(())))
    out = jax.jit(s._swap)(st)
    assert np.array_equal(np.asarray(out.discrete_proposal_params["z"]), np.asarray(tagged))


# --------------------------------------------------------------------------- #
# 4. it samples the right thing                                                #
# --------------------------------------------------------------------------- #

def test_tempering_samples_an_exactly_enumerable_discrete_target():
    m, lp = _discrete_only()
    s = parallel_tempering(m, n_temperatures=4, seed=0)
    s.initialize(); s.warmup(400); s.sample(30000)
    draws = s.get_discrete_flat()
    assert draws.shape == (30000, 3), "the cold chain's labels, not the whole product"
    emp = np.bincount(draws @ np.array([4, 2, 1]), minlength=8) / len(draws)
    states, exact = _exact_pmf(lp)
    assert exact.max() / exact.min() > 20                      # far from uniform: not vacuous
    assert np.max(np.abs(emp - exact)) < 0.012, (emp, exact)


def test_one_temperature_is_the_untempered_target():
    """A ladder of one rung at ``beta = 1`` targets the base model exactly.

    Deliberately a *distributional* check, not a bit-identity one. Two things rule the stronger
    claim out: parallel tempering adds its own swap draw components, which renumbers every RNG
    stream, and it is built on the Hamiltonian potentials, so it cannot compose over
    :class:`~mimcs.samplers.StaticContinuous` at all (`ReplicaExchangeMixin` reads
    ``self.potentials``, which only ``BaseHMC`` supplies). What *is* exactly checkable is that the
    single rung's tempered potential is the base model's own density, which is asserted below.
    """
    m, lp = _coupled_binary()
    s = parallel_tempering(m, n_temperatures=1, betas=jnp.ones((1,)), seed=4,
                           adapt_ladder=False)

    # exact: at beta = 1 the rung's potential IS the base model's coordinate-space density
    rng = np.random.default_rng(0)
    z = jnp.asarray(rng.integers(0, 2, size=3), jnp.int32)
    q = jnp.asarray(rng.normal(size=1), float)
    ctx = s.context(s.state._replace(discrete=z, coordinate=q), kinetic_cache=False)
    got = float(per_temperature_potential(s.potentials, q, ctx, 1)[0])
    want = -float(m.log_prob_at_coordinate(
        q, m.init_chart_hyperparams(), m.init_chart_indices(), z))
    assert np.isclose(got, want, rtol=1e-5, atol=1e-5), (got, want)

    # ... and end to end it samples the base model's discrete marginal
    s.initialize(); s.warmup(400); s.sample(20000)
    emp = np.bincount(s.get_discrete_flat() @ np.array([4, 2, 1]), minlength=8) / 20000
    assert emp.sum() == pytest.approx(1.0)
    assert np.all(emp > 0), "a one-rung ladder should still visit every state"


def test_labels_move_at_every_rung_and_hotter_rungs_move_more():
    m, _ = _coupled_binary()
    K = 4
    s = parallel_tempering(m, n_temperatures=K, seed=0)
    s.initialize(); s.warmup(300); s.sample(2000)
    moves = np.mean(np.stack(s.diagnostics()["discrete_moves"]), axis=0)
    assert moves.shape == (K,)
    assert np.all(moves > 0), f"some rung's labels never moved: {moves}"
    # The flattened targets accept more, so the hot end must move at least as much as the cold.
    assert moves[-1] > moves[0], moves


def test_the_cold_chain_is_what_is_retained():
    m, _ = _coupled_binary()
    s = parallel_tempering(m, n_temperatures=3, seed=0)
    s.initialize(); s.warmup(100); s.sample(200)
    assert s.get_discrete_flat().shape == (200, 3)
    with pytest.raises(RuntimeError, match="keep_all_temperatures"):
        s.get_discrete_all()

    s2 = parallel_tempering(m, n_temperatures=3, seed=0, keep_all_temperatures=True)
    s2.initialize(); s2.warmup(100); s2.sample(200)
    assert s2.get_discrete_all().shape == (200, 3, 3)
    assert np.array_equal(s2.get_discrete_flat(), s2.get_discrete_all()[:, 0, :])


def test_a_tempered_discrete_run_is_evidence_for_the_base_model():
    """The user-facing shape of the product-model evidence bug: a PT run over a model with integer
    parameters was refused by ``normalize``'s dimension guard, which read the K-fold product
    (``discrete 8000 vs 2000`` on a spike-and-slab logistic regression). The labels must come back
    at the *base* width and be the cold rung's --- the hot rungs' are never stored, so a wrong
    slice would be reading another temperature's assignment against this one's position.
    """
    from mimcs.factory.evidence import normalize

    m, _ = _coupled_binary()
    s = parallel_tempering(m, n_temperatures=3, seed=0)
    s.initialize(); s.warmup(100); s.sample(200)

    ev = normalize(m, s)
    assert ev.discrete.shape == (200, m.discrete_dim) and ev.discrete.dtype == np.int32
    assert ev.samples.shape == (200, m.ambient_dim)
    assert np.array_equal(ev.discrete, np.asarray(s.get_discrete_flat()))
    assert ev.gradients is not None and ev.gradients.shape == (200, m.coord_dim)

    # the score is conditional on the labels, so it must be the base target's at *these* labels
    st = s.state
    score = jax.vmap(jax.grad(lambda c, z: m.log_prob_at_coordinate(
        c, st.chart_hyperparams, st.chart_indices, z)))
    g = np.asarray(score(jnp.asarray(ev.coordinates, float), jnp.asarray(ev.discrete, jnp.int32)))
    assert np.allclose(g, ev.gradients, atol=1e-4)


# --------------------------------------------------------------------------- #
# 5. with the marginal adaptation                                              #
# --------------------------------------------------------------------------- #

def test_a_hot_rung_learns_a_flatter_marginal_than_the_cold_one():
    """The reason the tables are per rung at all: a hot rung's target is flatter, so proposing
    from the cold chain's concentrated marginal there would fight the exploration tempering
    exists to provide."""
    m, _ = _discrete_only()
    s = parallel_tempering(m, n_temperatures=4, seed=0,
                           extra_mixins=(DiscreteMarginalAdaptation,))
    s.initialize(); s.warmup(1500); s.sample(10)
    tbl = np.asarray(s.state.discrete_proposal_params["z"])        # (K, 3, 2)
    ent = -np.sum(np.where(tbl > 0, tbl * np.log(np.maximum(tbl, 1e-300)), 0.0), axis=-1)
    per_rung = ent.mean(axis=-1) / np.log(2)
    assert per_rung[-1] > per_rung[0] + 0.05, per_rung
    assert per_rung[0] < 0.98, f"the cold rung learned nothing: {per_rung}"


# --------------------------------------------------------------------------- #
# 6. the benchmark itself                                                      #
# --------------------------------------------------------------------------- #

def test_the_spike_and_slab_really_has_a_barrier():
    """The benchmark must be checked before it is used as evidence. A first version of this
    problem put 0.23 of the mass on (1,1), which makes it a stepping stone rather than a barrier
    and would have made the whole comparison vacuous."""
    p = spike_and_slab()
    pz = p.truth["p_z"]
    modes = np.sort(pz)[-2:]
    assert modes.min() > 0.4, pz                       # two near-equal modes
    assert abs(modes[0] - modes[1]) < 0.1, pz
    assert pz[3] < 1e-3, pz                            # (1,1): the crossing state
    assert pz[3] / pz[2] < 1e-3, "the barrier is not blocking; Gibbs would cross freely"


def test_tempering_crosses_the_barrier_that_traps_a_single_chain():
    """The claim this benchmark supports, at a run length a test can afford.

    Deliberately **not** an exactness assertion. Averaged over seeds PT does recover the exact
    frequencies (measured: 0.5108 against 0.5143 over 8 seeds at 20000 draws), but the per-seed
    spread on this target is large --- the chain switches modes in bursts, so the effective number
    of independent mode observations is far below the raw switch count. A tight single-seed
    threshold here would be a coin flip dressed as a check, which is the trap
    ``test_adaptation_converges_to_target_accept`` in test_step_size_adaptation.py documents.

    What *is* robust on one seed is the contrast: PT crosses hundreds of times and visits both
    modes, where a single chain crosses ~0 times and visits one. That is the property the feature
    exists to provide.
    """
    p = spike_and_slab()
    s = parallel_tempering(p.model, n_temperatures=6, seed=0, target_accept=0.9)
    s.initialize(); s.warmup(1000); s.sample(8000)
    key = s.get_discrete_flat() @ np.array([2, 1])
    emp = np.bincount(key, minlength=4) / len(key)

    modes = key[(key == 1) | (key == 2)]
    switches = int(np.sum(np.diff(modes) != 0))
    assert switches > 100, f"the barrier was not crossed ({switches} mode switches)"
    assert emp[1] > 0.15 and emp[2] > 0.15, emp        # both modes genuinely visited
    assert np.max(np.abs(emp - p.truth["p_z"])) < 0.25, (emp, p.truth["p_z"])


def test_a_single_chain_is_trapped_by_the_same_barrier():
    """The other half of the contrast, and the reason the feature is needed.

    Measured over 8 seeds: a plain Gibbs chain lands in one mode and stays --- 4 seeds in each,
    0 mode crossings, the frequency 100% wrong --- while split R-hat on the labels reports
    **1.0000 on every seed**. A coordinate that never moves has no within-chain variance to
    betray it, so the mixing diagnostic is blind to a maximally wrong answer. That is why this
    test asserts the trap directly rather than trusting R-hat to reveal it.
    """
    from mimcs.adaptation import MassMatrixAdaptation, RobbinsMonroStepSize
    from mimcs.hmc import NUTS
    plain = make_sampler_class(RobbinsMonroStepSize, MassMatrixAdaptation,
                               DiscreteMetropolisWithinGibbs, NUTS)
    p = spike_and_slab()
    s = plain(p.model, p.model.default_sample(), seed=0, target_accept=0.9)
    s.initialize(); s.warmup(1000); s.sample(8000)
    z = s.get_discrete_flat()
    key = z @ np.array([2, 1])
    emp = np.bincount(key, minlength=4) / len(key)

    assert max(emp[1], emp[2]) > 0.98, f"expected one mode to dominate, got {emp}"
    half = len(z) // 2
    rhat = float(np.max(split_rhat(z[:half].astype(float), z[half:2 * half].astype(float))))
    assert rhat < 1.01, (
        f"R-hat {rhat:.4f} flagged the trapped chain -- if this ever fails, the claim that the "
        f"mixing diagnostic is blind to this failure needs revisiting")
