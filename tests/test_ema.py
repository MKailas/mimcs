"""The shared EMA of every mass adaptation (``mimcs.adaptation._ema``).

Six mixins adapt a mass --- ScoreMassAdaptation, MassMatrixAdaptation (each diagonal and dense),
LowRankAdaptation, MetricAdaptation, ShapedMetricAdaptation and RelativisticMassAdaptation --- and
all read the same two keys: ``mass_ema`` (keep an EMA of the estimate and freeze it for sampling;
the raw iterate still drives warmup; off by default except for the two learned metrics) and
``mass_ema_warmup`` (the EMA drives warmup too; off by default everywhere). The unit tests pin the
helper. The integration tests pin the behaviours for every mixin against the one recursion ``e_1 = x_1, e_n = e_{n-1} + eta_n (x_n - e_{n-1})``, taken in log
(diagonal), log-Cholesky (dense) or parameter (learned metric) space, with the mixin's RM gain.
"""

from typing import Callable, NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from mimcs import Model
from mimcs.model import EuclideanParameter
from mimcs.adaptation import RobbinsMonroStepSize, LowRankAdaptation
from mimcs.adaptation._ema import LogEMA, ema_options, tree_ema, _to_log
from mimcs.adaptation._stochastic import rm_gain
from mimcs.samplers import make_sampler_class
from mimcs.hmc import NUTS, LowRankQuadraticKinetic
from mimcs.hmc.metric_expr import Exp
from mimcs.factory import analyze
from mimcs.factory.spec import BlockSpec
from mimcs.testing import correlated_gaussian, nuts, relativistic_hmc


# --- the helper (unit) --------------------------------------------------------- #

def test_log_ema_diagonal_is_the_recursion_in_log_space():
    e = LogEMA("diagonal")
    assert e.value() is None
    e.update(np.array([2.0, 8.0]), 0.9)                 # the first update just sets it
    assert np.allclose(e.value(), [2.0, 8.0])
    e.update(np.array([8.0, 2.0]), 0.5)                 # gain 1/2 in log space: geometric mean
    assert np.allclose(e.value(), [4.0, 4.0])
    e.update(np.array([1.0, 16.0]), 0.25)
    want = np.exp(np.log([4.0, 4.0]) + 0.25 * (np.log([1.0, 16.0]) - np.log([4.0, 4.0])))
    assert np.allclose(e.value(), want)


def test_log_ema_dense_averages_log_cholesky_and_stays_a_valid_factor():
    e = LogEMA("dense")
    e.update(np.array([[2.0, 0.0], [1.0, 3.0]]), 0.3)
    e.update(np.array([[8.0, 0.0], [3.0, 12.0]]), 0.5)
    out = e.value()
    assert np.allclose(np.triu(out, 1), 0.0) and np.all(np.diag(out) > 0)
    assert np.allclose(np.diag(out), [4.0, 6.0])        # geometric mean on the diagonal
    assert np.isclose(out[1, 0], 2.0)                   # arithmetic mean below it


def test_tree_ema_is_the_recursion_leafwise():
    a = {"W": [jnp.array([[1.0]])], "b": jnp.array([0.0, 2.0])}
    x = {"W": [jnp.array([[3.0]])], "b": jnp.array([4.0, 2.0])}
    out = tree_ema(a, x, 0.25)
    assert np.allclose(out["W"][0], 1.5) and np.allclose(out["b"], [1.0, 2.0])


def test_ema_options_defaults_implication_and_removed_keys():
    assert ema_options({}) == (False, False)
    assert ema_options({}, sample_default=True) == (True, False)     # the learned metrics' default
    assert ema_options({"mass_ema": False}, sample_default=True) == (False, False)
    assert ema_options({"mass_ema": True}) == (True, False)
    assert ema_options({"mass_ema_warmup": True}) == (True, True)    # warmup implies sampling
    for old, new in (("mass_polyak", "mass_ema"), ("score_mass_polyak_warmup", "mass_ema_warmup"),
                     ("metric_ema_warmup", "mass_ema_warmup")):
        with pytest.raises(ValueError, match=f"use '{new}'"):
            ema_options({old: False})               # even False: a stale key must not pass silently


def test_a_removed_key_raises_when_the_sampler_is_built():
    prob = correlated_gaussian(mean=[1.0, -2.0], cov=[[2.0, 1.4], [1.4, 1.5]])
    with pytest.raises(ValueError, match="mass_polyak"):
        nuts(mass_adapt="score", mass_polyak=True)(prob.model, 0)


# --- every mass adaptation (integration) --------------------------------------- #

def _gauss2():
    return correlated_gaussian(mean=[1.0, -2.0], cov=[[2.0, 1.4], [1.4, 1.5]]).model


def _gauss4():
    rng = np.random.default_rng(1)
    B = rng.standard_normal((4, 4))
    return correlated_gaussian(mean=np.zeros(4), cov=(B @ B.T + np.eye(4)).tolist()).model


def _funnel(n=3, scale=1.5):
    def log_post(p):
        v = jnp.squeeze(p["v"])
        return -0.5 * v ** 2 / scale ** 2 - 0.5 * jnp.exp(-v) * jnp.sum(p["x"] ** 2) - 0.5 * n * v
    return Model([EuclideanParameter("v", ()), EuclideanParameter("x", (n,))], {"lp": log_post})


def _lowrank(model, **algo):
    Cls = make_sampler_class(RobbinsMonroStepSize, LowRankAdaptation, NUTS)
    return Cls(model, init_position=np.zeros(model.ambient_dim), seed=0,
               kinetics=[LowRankQuadraticKinetic(id="T", rank=1)],
               max_tree_depth=10, step_size=0.5, target_accept=0.8, **algo)


def _learned(shape):
    def build(model, **algo):
        spec = analyze(model)
        vs, ve = model.coord_block("v")
        xs, xe = model.coord_block("x")
        params = {"metric": Exp("v") + Exp()}
        if shape is not None:
            params["shape"] = shape
        spec.blocks = [BlockSpec(["v"], [(vs, ve)], "diagonal"),
                       BlockSpec(["x"], [(xs, xe)], "learned_metric", params=params)]
        spec.terminate = None
        spec.algo_kwargs = {**spec.algo_kwargs, **algo}
        return spec.build(seed=0)
    return build


def _leaves(tree):
    return [np.asarray(l, dtype=float) for l in jax.tree_util.tree_leaves(tree)]


def _K_from_L(L):
    """The dense score mass averages K = chol(M); the kinetic is written L = chol(M^{-1})."""
    L = np.asarray(L, float)
    return np.linalg.cholesky(np.linalg.inv(L @ L.T))


def _block_emas(blocks, attr):
    emas = [getattr(b, attr) for b in (blocks or {}).values()]
    return None if not emas or any(e is None for e in emas) else emas


class Case(NamedTuple):
    model: Callable
    build: Callable                     # build(model, **algo) -> sampler
    view: Callable                      # ham_params -> the averaged quantity (list of arrays)
    ema: Callable                       # sampler -> the mixin's own EMA in the same space, or None
    start: int = 0                      # warmup steps before the first mass write
    default_ema: bool = False           # the mixin's documented `mass_ema` default


CASES = {
    "score_diag": Case(
        _gauss2, lambda m, **a: nuts(mass_adapt="score", **a)(m, 0),
        lambda hp: [-np.log(np.asarray(hp["T"], float))],                  # written M^-1; avg log M
        lambda s: None if (e := _block_emas(s._sm_blocks, "ema")) is None
        else [np.log(e[0].value())]),
    "score_dense": Case(
        _gauss2, lambda m, **a: nuts(mass_adapt="score", metric="dense", **a)(m, 0),
        lambda hp: [_to_log(_K_from_L(hp["T"]), "dense")],
        lambda s: None if (e := _block_emas(s._sm_blocks, "ema")) is None
        else [_to_log(e[0].value(), "dense")]),
    "cov_diag": Case(
        _gauss2, lambda m, **a: nuts(mass_adapt="covariance", **a)(m, 0),
        lambda hp: [np.log(np.asarray(hp["T"], float))],
        lambda s: [np.log(s._mm_ema_avg["T"].value())] if s._mm_ema_avg else None, start=50),
    "cov_dense": Case(
        _gauss2, lambda m, **a: nuts(mass_adapt="covariance", metric="dense", **a)(m, 0),
        lambda hp: [_to_log(hp["T"], "dense")],
        lambda s: [_to_log(s._mm_ema_avg["T"].value(), "dense")] if s._mm_ema_avg else None,
        start=50),
    "lowrank": Case(
        _gauss4, _lowrank,
        lambda hp: [np.log(np.asarray(hp["T"][0], float))],                 # (D, V): average D
        lambda s: None if (e := _block_emas(s._lr_blocks, "_ema")) is None
        else [np.log(e[0].value())]),
    "learned": Case(
        _funnel, _learned(None), lambda hp: _leaves(hp["x"]),
        lambda s: _leaves(s._metric_ema["x"]) if s._metric_ema else None, default_ema=True),
    "shaped": Case(
        _funnel, _learned(("lowrank", 1)), lambda hp: _leaves(hp["x"]["diag"]),
        lambda s: _leaves(s._shp_ema["x"]) if s._shp_ema else None, default_ema=True),
    "relativistic": Case(
        _gauss2, lambda m, **a: relativistic_hmc(shape=(2,), mass_adapt=True, n_leapfrog=20,
                                                 step_size=0.4, **a)(m, 0),
        lambda hp: [np.log(np.asarray(hp["T"], float))],
        lambda s: [np.log(s._rm_ema.value())] if s._rm_ema is not None else None),
}
N_WARMUP = 150


def _run(case, **algo):
    """Warm up while recording, each step, the written mass and the mixin's own EMA (both in the
    averaged space); then enter sampling. Returns (writes, emas, frozen, sampler)."""
    c = CASES[case]
    s = c.build(c.model(), **algo)
    hook, writes, emas = s._postprocess_hooks, [], []

    def recording(state):
        state = hook(state)
        writes.append(c.view(state.ham_params))
        emas.append(c.ema(s))
        return state

    s._postprocess_hooks = recording
    s.warmup(N_WARMUP)
    del s._postprocess_hooks                        # back to the class's hook for sampling
    s.sample(1)                                     # `_finalize_hooks` runs on the first `sample`
    return writes, emas, c.view(s.state.ham_params), s


def _same(a, b):
    return all(np.array_equal(x, y) for x, y in zip(a, b))


@pytest.mark.parametrize("case", CASES)
def test_off_keeps_no_average_and_samples_with_the_last_raw_iterate(case):
    writes, emas, frozen, _ = _run(case, mass_ema=False)
    assert all(e is None for e in emas)             # the average is not even computed
    assert _same(frozen, writes[-1])


@pytest.mark.parametrize("case", CASES)
def test_the_default_is_each_mixins_documented_default(case):
    """Off for every mass but the learned metrics, whose raw last iterate is unsafe to sample with."""
    writes, _, frozen, _ = _run(case)
    ref_writes, _, ref_frozen, _ = _run(case, mass_ema=CASES[case].default_ema)
    assert all(_same(a, b) for a, b in zip(writes, ref_writes)) and _same(frozen, ref_frozen)


@pytest.mark.parametrize("case", CASES)
def test_mass_ema_freezes_the_ema_and_leaves_warmup_untouched(case):
    start = CASES[case].start
    base, _, _, _ = _run(case, mass_ema=False)
    writes, _, frozen, _ = _run(case, mass_ema=True)
    assert all(_same(a, b) for a, b in zip(base, writes))     # the raw iterate still drives warmup
    ema = None                                                # the recursion over those raw writes
    for n, w in enumerate(writes[start:], start=start + 1):
        ema = list(w) if ema is None else [e + rm_gain(n) * (x - e) for e, x in zip(ema, w)]
    for got, want in zip(frozen, ema):
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)
    assert not all(np.allclose(a, b, rtol=1e-3) for a, b in zip(frozen, writes[-1]))  # control


@pytest.mark.parametrize("case", CASES)
def test_mass_ema_warmup_drives_every_warmup_step_with_the_ema(case):
    start = CASES[case].start
    base, _, _, _ = _run(case, mass_ema=False)
    writes, emas, frozen, _ = _run(case, mass_ema_warmup=True)
    for w, e in zip(writes[start:], emas[start:]):            # each write IS the mixin's EMA
        for a, b in zip(w, e):
            np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)
    assert _same(frozen, writes[-1])                          # and it is what sampling keeps
    # Control: the EMA fed back, so the warmup trajectory departs from the raw-driven one.
    assert not all(_same(a, b) for a, b in zip(base[start:], writes[start:]))
