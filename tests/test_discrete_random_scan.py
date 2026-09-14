"""Tests for random-scan Metropolis-within-Gibbs (``RandomScanMetropolisWithinGibbs``).

Each jump picks one coordinate uniformly from **all** discrete coordinates and moves it with that
parameter's own update method --- the same updaters, proposals and adaptations as the systematic
scan. What is new, and silent when wrong, gets a check with a control that must fail:

* **dispatch** across parameters of unequal size and different methods: an exact joint, against a
  perturbed density; and uniform per-coordinate visit counts, which a parameter-uniform choice
  would fail;
* **reversibility**, the reason a random scan exists: the empirical transition flux is symmetric,
  where the systematic scan's is measurably not;
* the adaptations still work when a coordinate can go unvisited in an iteration.
"""

import itertools

import numpy as np
import jax
import jax.numpy as jnp
import pytest
from scipy.special import gammaln

from mimcs.adaptation import (DiscreteMarginalAdaptation, DiscreteRandomWalkAdaptation,
                              RobbinsMonroStepSize)
from mimcs.hmc import NUTS
from mimcs.model import EuclideanParameter, IntegerParameter, Model
from mimcs.model.jump import JumpOperator
from mimcs.samplers import (DiscreteMetropolisWithinGibbs, RandomScanMetropolisWithinGibbs,
                            StaticContinuous, SystematicScanMetropolisWithinGibbs,
                            make_sampler_class)

RS_ONLY = make_sampler_class(RandomScanMetropolisWithinGibbs, StaticContinuous)
SYS_ONLY = make_sampler_class(SystematicScanMetropolisWithinGibbs, StaticContinuous)
RS_ADAPT = make_sampler_class(DiscreteMarginalAdaptation, DiscreteRandomWalkAdaptation,
                              RandomScanMetropolisWithinGibbs, StaticContinuous)
RS_RW = make_sampler_class(DiscreteRandomWalkAdaptation, RandomScanMetropolisWithinGibbs,
                           StaticContinuous)
NUTS_RS = make_sampler_class(RobbinsMonroStepSize, RandomScanMetropolisWithinGibbs, NUTS)


# --------------------------------------------------------------------------- #
# the family and the budget                                                    #
# --------------------------------------------------------------------------- #

def _binary(n=3):
    w = jnp.asarray([1.3, -0.7, 2.1][:n], float)
    J = jnp.asarray([[0.0, 1.5, -0.9], [1.5, 0.0, 0.6], [-0.9, 0.6, 0.0]])[:n, :n]

    def lp(v):
        z = v["z"].astype(float)
        return jnp.dot(w, z) + 0.5 * z @ J @ z

    return Model([], {"p": lp},
                 discrete_parameters=[IntegerParameter("z", (n,), lower=0, upper=1)]), lp


def test_the_bare_family_cannot_be_composed():
    m, _ = _binary()
    with pytest.raises(TypeError, match="RandomScanMetropolisWithinGibbs"):
        make_sampler_class(DiscreteMetropolisWithinGibbs, StaticContinuous)(
            m, m.default_sample(), seed=0)


def test_the_default_budget_is_one_jump_per_coordinate_and_is_overridable():
    m, _ = _binary()
    s = RS_ONLY(m, m.default_sample(), seed=0)
    assert s._n_discrete_jumps == m.discrete_dim == 3
    assert s.state.rng_draw.discrete_index.shape == (3,)
    assert RS_ONLY(m, m.default_sample(), seed=0, discrete_sweeps=2)._n_discrete_jumps == 6
    s = RS_ONLY(m, m.default_sample(), seed=0, discrete_jumps=7)
    assert s._n_discrete_jumps == 7
    s.sample(200)
    d = s.diagnostics()
    assert np.all(np.asarray(d["discrete_moves"]) <= 7)
    acc = np.asarray(d["discrete_accept_prob"])
    assert np.all((acc >= 0) & (acc <= 1)) and acc.mean() > 0.05
    for bad in (0, -1, 2.5):
        with pytest.raises(ValueError, match="discrete_jumps"):
            RS_ONLY(m, m.default_sample(), seed=0, discrete_jumps=bad)


# --------------------------------------------------------------------------- #
# dispatch across parameters                                                   #
# --------------------------------------------------------------------------- #

_A = jnp.asarray([0.0, 1.2, -0.8])
_B = jnp.asarray([1.0, -0.6, 1.4])
_E = jnp.asarray([0.0, 0.9, -0.4, 1.3])


def _mixed_lp(v):
    """Four parameters of sizes 1 / 3 / 2 / 2, coupled across parameters (a-c, b-e)."""
    b, c, e = v["b"].astype(float), v["c"].astype(float), v["e"].astype(float)
    a = v["a"]
    return (_A[a] + jnp.dot(_B, b) + 0.9 * b[0] * b[1]
            - 0.5 * jnp.sum(((c - 1.5 - 1.2 * a.astype(float)) / 1.1) ** 2)
            + jnp.sum(_E[v["e"]]) + 0.8 * e[0] * b[2])


def _mixed():
    return Model([], {"p": _mixed_lp}, discrete_parameters=[
        IntegerParameter("a", (), lower=0, upper=2),
        IntegerParameter("b", (3,), lower=0, upper=1),
        IntegerParameter("c", (2,), lower=0, upper=5, ordinal=True),
        IntegerParameter("e", (2,), lower=0, upper=3)])


MIXED_METHODS = {"a": "metropolis", "b": "metropolis", "c": "random_walk", "e": "exact"}


def _mixed_exact(scale=1.0):
    """Every coordinate's marginal pmf and two cross-parameter moments, by full enumeration."""
    rows = list(itertools.product(range(3), *[range(2)] * 3, *[range(6)] * 2, *[range(4)] * 2))
    S = np.asarray(rows, np.int32)
    vals = {"a": S[:, 0], "b": S[:, 1:4], "c": S[:, 4:6], "e": S[:, 6:8]}
    lp = np.asarray(jax.vmap(lambda a, b, c, e: _mixed_lp({"a": a, "b": b, "c": c, "e": e}))(
        *(jnp.asarray(vals[k]) for k in "abce")), float) * scale
    w = np.exp(lp - lp.max())
    w /= w.sum()
    return _summaries({k: np.asarray(v) for k, v in vals.items()}, w)


def _summaries(vals, w=None):
    n = len(vals["a"])
    w = np.full(n, 1.0 / n) if w is None else w
    out = {}
    for k, ni in (("a", 3), ("b", 2), ("c", 6), ("e", 4)):
        col = vals[k].reshape(n, -1)
        for j in range(col.shape[1]):
            out[f"{k}[{j}]"] = np.bincount(col[:, j], weights=w, minlength=ni)
    out["E[a c0]"] = np.sum(w * vals["a"] * vals["c"].reshape(n, -1)[:, 0])
    out["E[b2 e0]"] = np.sum(w * vals["b"].reshape(n, -1)[:, 2] * vals["e"].reshape(n, -1)[:, 0])
    return out


@pytest.fixture(scope="module")
def mixed_run():
    m = _mixed()
    s = RS_ADAPT(m, m.default_sample(), seed=0, discrete_update=MIXED_METHODS)
    s.initialize(); s.warmup(600); s.sample(40000)
    return s


def _max_errors(emp, exact):
    pmf = max(float(np.abs(emp[k] - exact[k]).max()) for k in emp if k.endswith("]"))
    mom = max(abs(float(emp[k] - exact[k])) for k in emp if k.startswith("E["))
    return pmf, mom


def test_a_mixed_model_recovers_its_exact_joint_under_every_method(mixed_run):
    s = mixed_run
    d = s.get_samples()
    emp = _summaries({k: np.asarray(d[k]) for k in "abce"})
    pmf, mom = _max_errors(emp, _mixed_exact())
    assert pmf < 0.02, f"a coordinate marginal is off by {pmf}"
    assert mom < 0.1, f"a cross-parameter moment is off by {mom}"
    # non-vacuity: labels move, and every parameter's every coordinate took several values
    assert float(np.mean(s.diagnostics()["discrete_moves"])) > 0.5
    assert all(len(np.unique(np.asarray(d[k]).reshape(len(d[k]), -1)[:, j])) > 1
               for k in "abce" for j in range(np.asarray(d[k]).reshape(len(d[k]), -1).shape[1]))


def test_a_perturbed_density_fails_the_same_comparison(mixed_run):
    """CONTROL: the run tracks *this* joint, not a plausible-looking one on the same support."""
    d = mixed_run.get_samples()
    emp = _summaries({k: np.asarray(d[k]) for k in "abce"})
    pmf, _ = _max_errors(emp, _mixed_exact(scale=0.5))
    assert pmf > 0.05


def test_a_coordinate_is_chosen_uniformly_not_a_parameter():
    """Visit counts over one long iteration are uniform per *coordinate*. An unbounded walk never
    proposes its current value, so ``n_proposed`` counts every visit."""
    m = Model([], {"p": lambda v: -0.5 * (v["p"] / 5.0) ** 2 - 0.5 * jnp.sum((v["q"] / 5.0) ** 2)},
              discrete_parameters=[IntegerParameter("p"), IntegerParameter("q", (5,))])
    J = 6000
    s = RS_ONLY(m, m.default_sample(), seed=3, discrete_jumps=J,
                discrete_update={"p": "random_walk", "q": "random_walk"})
    s.sample(1)
    params = s.state.discrete_proposal_params
    counts = np.concatenate([np.asarray(params["p"]["n_proposed"]).ravel(),
                             np.asarray(params["q"]["n_proposed"]).ravel()])
    assert counts.sum() == J

    def chi2(expected):
        return float(np.sum((counts - expected) ** 2 / expected))

    uniform = np.full(6, J / 6)
    assert chi2(uniform) < 20.5                         # df 5, p = 0.001
    # CONTROL: picking a *parameter* uniformly would give p half the jumps
    assert chi2(np.asarray([J / 2] + [J / 10] * 5)) > 1000


# --------------------------------------------------------------------------- #
# reversibility                                                                #
# --------------------------------------------------------------------------- #

def _pair_asymmetry(s, n):
    d = s.get_samples()["z"].reshape(n, 2)
    key = 3 * d[:, 0] + d[:, 1]
    F = np.zeros((9, 9))
    np.add.at(F, (key[:-1], key[1:]), 1.0)
    F /= F.sum()
    return float(np.abs(F - F.T).max())


def _coupled_pair():
    w = jnp.asarray([[0.0, 0.8, -0.5], [0.3, -0.9, 1.1]])

    def lp(v):
        z = v["z"]
        return w[0, z[0]] + w[1, z[1]] + 1.6 * (z[0] == z[1]).astype(float) \
            + 0.7 * (z[0] - z[1]).astype(float)

    return Model([], {"p": lp},
                 discrete_parameters=[IntegerParameter("z", (2,), lower=0, upper=2)])


def test_one_jump_per_iteration_is_a_reversible_kernel():
    """Detailed balance shows as a symmetric empirical flux ``F[a, b] ~ F[b, a]``."""
    m, n = _coupled_pair(), 200000
    s = RS_ONLY(m, m.default_sample(), seed=0, discrete_jumps=1)
    s.initialize(); s.sample(n)
    assert _pair_asymmetry(s, n) < 3e-3


def test_the_systematic_scan_is_not_reversible_on_the_same_target():
    """CONTROL: a fixed order of reversible updates is only ``pi``-invariant, and the same
    statistic sees it --- so the random scan's symmetry above is a property, not the test's
    blindness."""
    m, n = _coupled_pair(), 200000
    s = SYS_ONLY(m, m.default_sample(), seed=0)
    s.initialize(); s.sample(n)
    assert _pair_asymmetry(s, n) > 1e-2


# --------------------------------------------------------------------------- #
# adaptations, jumps, open sides                                               #
# --------------------------------------------------------------------------- #

def _laplace(size=4, b=15.0):
    return Model([], {"p": lambda v: -jnp.sum(jnp.abs(v["g"].astype(float))) / b},
                 discrete_parameters=[IntegerParameter("g", (size,))])


def test_only_the_visited_coordinate_moves_its_scale():
    m = _laplace()
    s = RS_RW(m, m.default_sample(), seed=0, discrete_jumps=1,
              discrete_update={"g": "random_walk"})
    rho0 = np.asarray(s.state.discrete_proposal_params["g"]["log_scale"]).copy()
    s.warmup(1)
    changed = np.asarray(s.state.discrete_proposal_params["g"]["log_scale"]) != rho0
    assert int(changed.sum()) == 1


def test_the_random_walk_scale_adapts_to_the_same_root_under_a_random_scan():
    """Exact-kernel root for Laplace b = 15 at 1/3 acceptance: rho ~ 4.0 (systematic smoke:
    3.88-4.20)."""
    m = _laplace()
    s = RS_RW(m, m.default_sample(), seed=1, discrete_update={"g": "random_walk"})
    s.initialize(); s.warmup(2000)
    rho = np.asarray(s.state.discrete_proposal_params["g"]["log_scale"])
    assert 3.4 < float(np.median(rho)) < 4.6, rho


def test_an_open_sided_count_is_sampled_exactly():
    lam = 3.0
    m = Model([], {"p": lambda v: jnp.sum(v["n"].astype(float) * jnp.log(lam)
                                          - jax.scipy.special.gammaln(v["n"] + 1.0))},
              discrete_parameters=[IntegerParameter("n", (3,), lower=0)])
    s = RS_RW(m, m.default_sample(), seed=0, discrete_update={"n": "random_walk"})
    s.initialize(); s.warmup(500); s.sample(20000)
    draws = np.asarray(s.get_samples()["n"]).ravel()
    k = np.arange(9)
    exact = np.exp(k * np.log(lam) - lam - gammaln(k + 1))
    emp = np.bincount(draws, minlength=40)[:9] / len(draws)
    assert np.abs(emp - exact).max() < 0.015


W = jnp.asarray([0.2, 0.5, 0.3])
MU = jnp.asarray([-1.0, 0.4, 2.0])
SIG = jnp.asarray([0.7, 1.3, 0.5])


def _toy_logp(v):
    z, x = v["z"], jnp.reshape(v["x"], ())
    return (jnp.log(W[z]) - 0.5 * jnp.log(2 * jnp.pi * SIG[z] ** 2)
            - 0.5 * ((x - MU[z]) / SIG[z]) ** 2)


def _shift(values, c, v):
    return (values["x"] + (MU[v] - MU[values["z"]]),)


def test_a_jump_operator_samples_its_closed_form_joint_under_a_random_scan():
    m = Model([EuclideanParameter("x", ())], {"p": _toy_logp},
              discrete_parameters=[IntegerParameter("z", (), lower=0, upper=2)],
              jump_operators={"z": JumpOperator("z", ("x",), _shift, volume_preserving=True)})
    s = NUTS_RS(m, m.default_sample(), seed=0, discrete_update={"z": "metropolis"})
    s.initialize(); s.warmup(600); s.sample(30000)
    d = s.get_samples()
    z, x = np.asarray(d["z"]).ravel(), np.asarray(d["x"]).ravel()
    emp = np.bincount(z, minlength=3) / len(z)
    assert float(np.abs(emp - np.asarray(W)).max()) < 0.02
    assert max(abs(x[z == k].mean() - float(MU[k])) for k in range(3)) < 0.05


# --------------------------------------------------------------------------- #
# tempering and the factory                                                    #
# --------------------------------------------------------------------------- #

def _exact_pmf(lp, n=3):
    states = np.array([[(i >> k) & 1 for k in range(n - 1, -1, -1)] for i in range(2 ** n)])
    logp = np.array([float(lp({"z": jnp.asarray(s)})) for s in states])
    w = np.exp(logp - logp.max())
    return w / w.sum()


def test_tempering_with_a_random_scan_samples_the_exact_target():
    from mimcs.pt import parallel_tempering
    m, lp = _binary()
    s = parallel_tempering(m, n_temperatures=4, seed=0, discrete_scan="random")
    assert isinstance(s, RandomScanMetropolisWithinGibbs)
    assert s.state.rng_draw.discrete_index.shape == (3,)      # shared across the 4 lanes
    s.initialize(); s.warmup(400); s.sample(30000)
    draws = s.get_discrete_flat()
    emp = np.bincount(draws @ np.array([4, 2, 1]), minlength=8) / len(draws)
    assert np.abs(emp - _exact_pmf(lp)).max() < 0.015
    with pytest.raises(ValueError, match="discrete_scan"):
        parallel_tempering(m, n_temperatures=2, seed=0, discrete_scan="zigzag")


def _continuous_and_discrete():
    w = jnp.asarray([1.3, -0.7, 2.1], float)

    def lp(v):
        z = v["z"].astype(float)
        return jnp.dot(w, z) - 0.5 * jnp.sum(v["x"] ** 2) + 0.4 * jnp.sum(z) * jnp.sum(v["x"])

    return Model([EuclideanParameter("x")], {"p": lp},
                 discrete_parameters=[IntegerParameter("z", (3,), lower=0, upper=1)])


def test_the_factory_reaches_the_random_scan_but_never_chooses_it():
    from mimcs import analyze
    m = _continuous_and_discrete()
    spec = analyze(m)
    assert spec.discrete_scan == "systematic"
    assert isinstance(spec.build(seed=0), SystematicScanMetropolisWithinGibbs)
    spec.discrete_scan = "random"
    assert "random scan" in str(spec)
    s = spec.build(seed=0)
    assert isinstance(s, RandomScanMetropolisWithinGibbs)
    assert not isinstance(s, SystematicScanMetropolisWithinGibbs)
    spec.base, spec.tempering_params = "pt_nuts", {"n_temperatures": 2}
    assert isinstance(spec.build(seed=0), RandomScanMetropolisWithinGibbs)
    spec.discrete_scan = "zigzag"
    with pytest.raises(ValueError, match="discrete_scan"):
        spec.build(seed=0)
