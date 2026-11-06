"""Tests for the metric regression's kick guard (``regression.KICK_GUARD``).

A learned block's metric depends on other coordinates, and its kinetic energy pushes them: the
force on dependency coordinate ``h`` is ``-1/2 sum_lanes d_h log M (1 - p^2/M)``. Pinned here: the
closed form :func:`~mimcs.factory.regression.kick_ratios` uses for that force's sd (against autodiff
of the kinetic energy at sampled momenta, with a control that must miss), and that the guard ranks
an AIC winner last for pushing too hard and keeps it under a looser limit; and that the default
limit of 2 demotes a winner kicking ~4 but keeps one kicking ~1, as a matched funnel metric does.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from mimcs.hmc.metric_expr import Exp
from mimcs.factory import regression
from mimcs.factory.regression import kick_ratios, select_metric


def _force_sd(expr, params, rows, scale, rng, draws=2000):
    """Monte Carlo sd of the kinetic energy's force on the dependency, momenta ``scale * N(0, M)``,
    averaged over ``rows`` in quadrature --- the slow, obviously right reference."""
    def T(p, v):
        lm = expr.log_evaluate(params, {"v": v})
        return 0.5 * jnp.sum(p ** 2 * jnp.exp(-lm)) + 0.5 * jnp.sum(lm)

    var = []
    for v in rows:
        M = jnp.exp(expr.log_evaluate(params, {"v": v}))
        p = scale * jnp.sqrt(M) * jnp.asarray(rng.normal(size=(draws, M.shape[0])), float)
        f = jax.vmap(lambda pp: -jax.grad(T, argnums=1)(pp, v))(p)
        var.append(np.var(np.asarray(f), axis=0))
    return np.sqrt(np.mean(var, axis=0))


def test_the_kick_ratio_is_the_kinetic_energys_force_on_the_dependency():
    """With unit score sd the ratio is the force's sd itself; momenta at twice the scale (a
    force four times as large) is the control that must miss."""
    rng = np.random.default_rng(0)
    L, m, R = 20, 3, 8
    expr = Exp("v") + Exp()
    params = jax.tree_util.tree_map(lambda a: a + 0.4 * rng.normal(size=np.shape(a)),
                                    expr.init_params(L, {"v": m}))
    v = rng.normal(size=(R, m))
    coords = np.c_[np.zeros((R, L)), v]
    grads = np.c_[np.zeros((R, L)), rng.normal(size=(R, m))]
    grads[:, L:] /= grads[:, L:].std(axis=0)                     # unit score sd: ratio == force sd
    got = kick_ratios(expr, params, L, {"v": list(range(L, L + m))}, coords, grads)["v"]
    rows = [jnp.asarray(r, float) for r in v]
    np.testing.assert_allclose(got, _force_sd(expr, params, rows, 1.0, rng), rtol=0.05)
    assert np.all(np.abs(_force_sd(expr, params, rows, 2.0, rng) / got - 1) > 2.0)


def _coupled_evidence(lanes=32, n=600, seed=1):
    """``lanes`` block coordinates whose score variance is ``exp(v)``, with ``v ~ N(0, 1)``: the
    winning metric ``Exp('v') + ...`` has ``d log M / dv ~ 1`` on every lane, so it kicks ``v`` at
    ``sqrt(lanes / 2) = 4`` times ``v``'s own (unit) force."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=n)
    g_block = np.exp(0.5 * v)[:, None] * rng.normal(size=(n, lanes))
    coords = np.c_[rng.normal(size=(n, lanes)), v]
    return list(range(lanes)), {"v": [lanes]}, coords, np.c_[g_block, -v]


def test_the_guard_ranks_a_hard_pushing_winner_last():
    block, deps, coords, grads = _coupled_evidence()
    free = select_metric(block, deps, coords, grads, kick_guard=None)
    assert free[0].expr.deps() == {"v"} and np.isnan(free[0].kick)     # unguarded: not scored

    loose = select_metric(block, deps, coords, grads, kick_guard=100.0)
    assert repr(loose[0].expr) == repr(free[0].expr)                    # passes, and is scored
    assert 3.0 < loose[0].kick < 5.5, loose[0].kick

    tight = select_metric(block, deps, coords, grads, kick_guard=1.0)
    assert tight[0].kick <= 1.0 and repr(tight[0].expr) != repr(free[0].expr)
    demoted = next(r for r in tight if repr(r.expr) == repr(free[0].expr))
    assert demoted.aic == float("inf") and demoted.kick > 3.0


def test_the_default_guard_is_two_and_demotes_a_hard_pusher():
    assert regression.KICK_GUARD == 2.0
    block, deps, coords, grads = _coupled_evidence()                    # winner kicks ~4
    a = select_metric(block, deps, coords, grads)
    b = select_metric(block, deps, coords, grads, kick_guard=2.0)
    assert [(repr(r.expr), r.aic) for r in a] == [(repr(r.expr), r.aic) for r in b]
    assert a[0].kick <= 2.0 and a[0].expr.deps() != {"v"}


def test_the_default_guard_keeps_a_gently_pushing_winner():
    block, deps, coords, grads = _coupled_evidence(lanes=2)             # winner kicks ~1
    a = select_metric(block, deps, coords, grads)
    off = select_metric(block, deps, coords, grads, kick_guard=None)
    assert repr(a[0].expr) == repr(off[0].expr) and a[0].expr.deps() == {"v"}
    assert 0.5 < a[0].kick < 2.0, a[0].kick
    assert [r.aic for r in a] == [r.aic for r in off]


def test_the_guard_off_scores_nothing():
    block, deps, coords, grads = _coupled_evidence(lanes=8, n=300, seed=2)
    off = select_metric(block, deps, coords, grads, kick_guard=None)
    assert all(np.isnan(r.kick) for r in off)
