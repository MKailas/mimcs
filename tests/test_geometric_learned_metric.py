"""The geometric-mean target on learned metrics (``MetricAdaptation`` / ``ShapedMetricAdaptation``
under ``mass_target="geometric"``).

The per-sample loss (``block_riemannian._geometric_log_loss``) is

    1/2 sum_d ( g_d^2 / M_d(q) + (u_d^2 / c) * 1/2 (M_d(q_new) + M_d(q_old)) ):

the score term where the score was computed, the spread term split evenly between the spread's two
endpoints. Its fixed point for a conditional metric ``M(v)`` is the conditional geometric mean
``sqrt(F(v) / S(v))`` when the dependency ``v`` moves little within the spread's window.

The fixed-point tests drive the mixins through the parallel-tempering adaptation host (the same
``AdaptState`` a rung receives, with explicit ``start``/``end``) on synthetic streams whose ``F(v)``
and ``S(v)`` are known; each has the score target as its control. Seeds are fixed.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import sqrtm

from mimcs.adaptation import MetricAdaptation, ShapedMetricAdaptation
from mimcs.hmc import Exp
from mimcs.hmc.block_riemannian import build_block
from mimcs.model import EuclideanParameter, Model
from mimcs.pt.adaptation import AdaptState, _AdaptationHost
from mimcs.samplers.base import Phase

DX = 3


class _Owner:
    _phase = Phase.WARMUP


def _model():
    return Model([EuclideanParameter("v", ()), EuclideanParameter("x", (DX,))],
                 {"lp": lambda p: -0.5 * (p["v"] ** 2 + jnp.sum(p["x"] ** 2))})


def _host(mixin, shape=None, **kwargs):
    m = _model()
    blk = build_block(m, "x", Exp("v"), shape=shape)
    host = type("Host", (mixin, _AdaptationHost), {})(m, [blk], _Owner(), **kwargs)
    return host, blk, m


def _drive(host, blk, stream, n):
    """Feed ``n`` steps of ``stream() -> (start, end, score)``; returns the final ham params."""
    params = {blk.id: blk.initial_mass_params(1 + DX)}
    for _ in range(n):
        start, end, g = stream()
        st = AdaptState(coordinate=jnp.asarray(end), ham_params=params, step_size=jnp.asarray(0.1),
                        potential_grads={"lp": jnp.asarray(g)}, start=jnp.asarray(start),
                        end=jnp.asarray(end))
        params = host._postprocess_hooks(st).ham_params
    return params[blk.id]


def _conditional_stream(a, f, rho_v=0.999, seed=0, corr=None, gcov=None):
    """``v`` a slow AR(1); ``x | v ~ N(0, e^{a v} C)`` drawn afresh each step (so the spread's
    endpoints share ``v`` to within ``sqrt(1 - rho_v)``); the score ``g ~ N(0, f e^{-a v} G)``
    independent of ``x``. ``C``, ``G`` default to the identity."""
    rng = np.random.default_rng(seed)
    LC = np.linalg.cholesky(np.eye(DX) if corr is None else corr)
    LG = np.linalg.cholesky(np.eye(DX) if gcov is None else gcov)
    state = {"v": 0.0, "q": np.zeros(1 + DX)}

    def step():
        v = rho_v * state["v"] + np.sqrt(1 - rho_v ** 2) * rng.standard_normal()
        x = np.exp(0.5 * a * v) * (LC @ rng.standard_normal(DX))
        q = np.r_[v, x]
        g = np.r_[0.0, np.sqrt(f) * np.exp(-0.5 * a * v) * (LG @ rng.standard_normal(DX))]
        start, state["v"], state["q"] = state["q"], v, q
        return start, q, g
    return step


def _wb(params):
    """The Exp(v) atom's per-coordinate slope and bias: log M = W v + b."""
    leaves = jax.tree_util.tree_leaves(params)
    W = np.asarray(leaves[0], float).reshape(DX, -1)[:, 0]
    b = np.asarray(leaves[1], float).reshape(DX)
    return W, b


def test_geometric_learned_metric_is_the_conditional_geometric_mean():
    """``F(v) = 4 e^{-v}``, ``S(v) = e^{v}``: the geometric target's metric is
    ``sqrt(F/S) = 2 e^{-v}`` --- slope -1, bias log 2. Control: the score target fits ``F`` itself,
    bias log 4 (less its ~0.2 clip bias, as for the constant score mass)."""
    host, blk, _ = _host(MetricAdaptation, mass_target="geometric", mass_ema=False)
    W, b = _wb(_drive(host, blk, _conditional_stream(a=1.0, f=4.0), 15_000))
    np.testing.assert_allclose(W, -1.0, atol=0.12)
    np.testing.assert_allclose(b, np.log(2.0), atol=0.12)
    host, blk, _ = _host(MetricAdaptation, mass_ema=False)
    W_s, b_s = _wb(_drive(host, blk, _conditional_stream(a=1.0, f=4.0), 15_000))
    np.testing.assert_allclose(W_s, -1.0, atol=0.12)
    assert np.all(b_s > np.log(2.0) + 0.4), b_s          # score: ~log 4 - clip bias


def test_the_spread_term_is_split_between_the_window_endpoints():
    """The loss's spread term is ``u^2 / c * 1/2 (M(q_new) + M(q_old))``: checked against a direct
    evaluation, and its gradient differs from evaluating it at ``q_new`` alone (control)."""
    from mimcs.hmc.block_riemannian import _geometric_log_loss
    _, blk, _ = _host(MetricAdaptation)
    params = jax.tree_util.tree_map(lambda x: x + 0.3, blk.init_params())
    q, qn, qo = (jnp.asarray(np.r_[v, np.ones(DX)]) for v in (0.2, 0.5, -1.0))
    g = jnp.asarray(np.r_[0.0, [1.0, 2.0, 3.0]])
    u = jnp.asarray(np.r_[0.0, [0.5, -0.5, 1.5]])
    got = blk.geometric_loss(params, q, qn, qo, None, g, u, 2.0)
    lm = lambda x: blk._log_mass(x, None, params)
    want = 0.5 * jnp.sum(g[1:] ** 2 * jnp.exp(-lm(q))
                         + u[1:] ** 2 / 2.0 * 0.5 * (jnp.exp(lm(qn)) + jnp.exp(lm(qo))))
    np.testing.assert_allclose(got, want, rtol=1e-6)
    one = _geometric_log_loss(lm(q), lm(qn), lm(qn), g[1:], u[1:], 2.0)
    assert abs(float(one) - float(got)) > 1e-3


def test_the_learned_metric_trust_region_binds_only_in_the_transient():
    """300 steps of scores ~1e6 drive the clip threshold up; the trust region holds every step's
    change of log M to 1 nat (it binds), and on the stationary stream that follows it stops binding
    while the fit still reaches the conditional geometric mean."""
    host, blk, _ = _host(MetricAdaptation, mass_target="geometric", mass_ema=False)
    rng = np.random.default_rng(3)
    calm = _conditional_stream(a=1.0, f=4.0, seed=4)

    def wild():
        s, e, g = calm()
        return s, e, g * 1e6 * (1 + rng.uniform())
    _drive(host, blk, wild, 300)
    n_transient = host._metric_capped.get("x", 0)
    assert n_transient > 50
    W, b = _wb(_drive(host, blk, calm, 20_000))
    np.testing.assert_allclose(W, -1.0, atol=0.15)
    np.testing.assert_allclose(b, np.log(2.0), atol=0.15)


def test_dense_shaped_geometric_metric_is_the_whitened_geometric_mean():
    """``x | v ~ N(0, e^{v} C)`` and an independent score ``~ N(0, e^{-v} G)`` with ``G != C^{-1}``.
    The geometric shaped metric at ``v = 0`` is ``C^{-1} # G``; the score target gives ``G``
    (control: the two differ by 59%)."""
    C = np.array([[1.0, 0.6, 0.2], [0.6, 1.0, 0.3], [0.2, 0.3, 1.0]])
    G = np.array([[6.0, 1.5, 0.0], [1.5, 1.0, 0.2], [0.0, 0.2, 0.4]])

    def gm(A, B):
        Ah = np.real(sqrtm(A)); Ai = np.linalg.inv(Ah)
        return Ah @ np.real(sqrtm(Ai @ B @ Ai)) @ Ah
    target = gm(np.linalg.inv(C), G)

    def fitted(**kw):
        host, blk, m = _host(ShapedMetricAdaptation, shape="dense", mass_ema=False, **kw)
        params = _drive(host, blk, _conditional_stream(1.0, 1.0, corr=C, gcov=G, seed=5), 20_000)
        q0 = jnp.zeros(1 + DX)
        Dh = np.exp(0.5 * np.asarray(blk._log_D(q0, None, params["diag"])))
        K = np.asarray(params["shape"])
        return Dh[:, None] * (K @ K.T) * Dh[None, :]
    M = fitted(mass_target="geometric")
    assert np.abs(M - target).max() / np.abs(target).max() < 0.12, (M, target)
    M_s = fitted()
    assert np.abs(M_s - G).max() / np.abs(G).max() < 0.2
    assert np.abs(target - G).max() / np.abs(G).max() > 0.5


def test_low_rank_shaped_geometric_metric_runs_and_uses_the_aux_stream():
    host, blk, _ = _host(ShapedMetricAdaptation, shape=("lowrank", 1), mass_target="geometric",
                         geometric_lag=4)
    params = _drive(host, blk, _conditional_stream(1.0, 1.0, seed=6), 400)
    assert host._shp_shape["x"].lam_aux is not None
    W, gamma = params["shape"]
    assert np.all(np.isfinite(np.asarray(gamma)))


def test_factory_geometric_reaches_learned_metric_blocks_and_runs_under_pt():
    """``spec.mass_adapt="geometric"`` sets ``mass_target``, which the learned metrics read too. Under
    PT the hosts get each rung's start and pre-swap end."""
    from mimcs.factory import BlockSpec, default_spec
    from mimcs.testing import neal_funnel_blocks
    m = neal_funnel_blocks(dim=3, scale=2.0).model
    cb = m.coord_block
    for base in ("nuts", "pt_nuts"):
        spec = default_spec(m)
        spec.mass_adapt = "geometric"
        spec.base = base
        if base == "pt_nuts":
            spec.tempering_params = {"n_temperatures": 3}
        spec.blocks = [BlockSpec(["v"], [cb("v")], "diagonal", {}),
                       BlockSpec(["x"], [cb("x")], "learned_metric", {"metric": Exp("v")})]
        spec.algo_kwargs = {**spec.algo_kwargs, "geometric_lag": 4}
        s = spec.build(seed=0)
        s.warmup(60)
        s.sample(20)
        hosts = s._adapt_hosts if base == "pt_nuts" else [s]
        for h in hosts:
            assert h._metric_track.active and h._metric_track.k == 4
            assert h._metric_count == 60


def test_new_point_puts_both_spread_halves_at_the_newest_coordinate():
    from types import SimpleNamespace as NS
    from mimcs.adaptation._geometric import SpreadTracker
    t = SpreadTracker({"mass_target": "geometric", "geometric_lag": 3,
                       "geometric_metric_point": "new"})
    xs = [np.full(2, float(i)) for i in range(5)]
    for i in range(4):
        sp = t.observe(NS(coordinate=xs[i + 1], start=xs[i], end=xs[i + 1]))
    np.testing.assert_array_equal(sp.u, xs[4] - xs[1])        # the window is unchanged
    np.testing.assert_array_equal(sp.q_old, sp.q_new)         # only the evaluation point moves
    with pytest.raises(ValueError, match="geometric_metric_point"):
        SpreadTracker({"geometric_metric_point": "old"})


def test_split_and_new_points_share_a_fixed_point_on_a_reversible_chain():
    """For a stationary reversible chain ``(q_t, q_{t-k})`` and ``(q_{t-k}, q_t)`` are equal in law and
    ``u^2`` is symmetric, so ``E[u^2 M(q_old)] = E[u^2 M(q_new)]``: splitting the spread term between
    the endpoints or putting it at the newest one has the SAME expected loss. With a lag over which
    ``v`` moves (``rho_v = 0.6``, k = 4) both fits flatten alike --- the flattening is the window's,
    not the evaluation point's. Control: k = 1 on the slow stream keeps the slope near -1.

    Checked on the unclipped SGD (``metric_clip_frac=0``), which is what the argument covers: the
    10% adaptive clip acts on the per-step gradients, whose distribution differs between the two
    (the split spreads the spread term over two points), so the clipped fits separate slightly
    (measured: slopes -0.74 split vs -0.80 new with the clip; -0.77 vs -0.79 without)."""
    fits = {}
    for point in ("split", "new"):
        host, blk, _ = _host(MetricAdaptation, mass_target="geometric", geometric_lag=4,
                             geometric_metric_point=point, mass_ema=False, metric_clip_frac=0.0)
        fits[point] = _wb(_drive(host, blk, _conditional_stream(1.0, 4.0, rho_v=0.6, seed=8),
                                 20_000))
    (Ws, bs), (Wn, bn) = fits["split"], fits["new"]
    np.testing.assert_allclose(Ws.mean(), Wn.mean(), atol=0.05)
    np.testing.assert_allclose(bs.mean(), bn.mean(), atol=0.05)
    assert Ws.mean() > -0.8 and Wn.mean() > -0.8               # both flattened by the window
    host, blk, _ = _host(MetricAdaptation, mass_target="geometric", mass_ema=False)
    W1, _ = _wb(_drive(host, blk, _conditional_stream(1.0, 4.0, seed=8), 15_000))
    np.testing.assert_allclose(W1, -1.0, atol=0.12)
