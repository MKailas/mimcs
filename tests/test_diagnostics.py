"""Tests for the uniform sampler-diagnostics structure and gradient-evaluation counting.

Every sampler State carries a single ``diagnostics`` dict the kernel fills; the generic
``postprocess`` accumulates it into one phase-tagged store, over which the (unchanged) accessors
are re-expressed. Gradient evaluations and tree size are counted for HMC-cost comparisons.
Seeds fixed.
"""

import numpy as np
import jax.numpy as jnp

from mimcs.hmc import (
    LineSearchIntegrator, leapfrog, default_potentials, make_kinetic, doubling_schedule)
from mimcs.testing import (
    correlated_gaussian, nuts, simple_nuts, hmc, adaptive_mh, wal_nuts, draw_samples)

PROB = correlated_gaussian(mean=[1.0, -2.0], cov=[[2.0, 1.4], [1.4, 1.5]])


def test_diagnostics_dict_shape_and_phases():
    s = nuts(max_tree_depth=8, step_size=0.3)(PROB.model, seed=0)
    s.warmup(120); s.sample(200)
    d = s.diagnostics("sampling")
    assert {"accept_prob", "accepted", "diverging", "tree_depth", "n_leaves", "grad_evals",
            "mean_refine", "proxy_accept_prob", "step_size"} <= set(d)
    assert all(len(v) == 200 for v in d.values())              # one entry per sampling transition
    assert len(s.diagnostics("warmup")["accepted"]) == 120
    assert len(s.diagnostics("all")["accepted"]) == 320


def test_accessors_match_the_store():
    s = nuts(max_tree_depth=8, step_size=0.3)(PROB.model, seed=1)
    s.warmup(150); s.sample(300)
    d = s.diagnostics("sampling")
    assert s.divergence_count() == int(d["diverging"].sum())
    assert np.isclose(s.divergence_rate(), d["diverging"].mean())
    assert np.isclose(s.acceptance_rate(), d["accepted"].mean())
    assert np.isclose(s.mean_n_leaves(), d["n_leaves"].mean())
    assert np.isclose(s.total_grad_evals(), d["grad_evals"].sum())
    # mean_tree_depth is over all phases (preserved semantics)
    assert np.isclose(s.mean_tree_depth(), s.diagnostics("all")["tree_depth"].mean())
    assert s.warmup_step_sizes().shape == (150,)


def test_oracle_still_bit_identical():
    """The new sum_grad_evals accumulator must not perturb the NUTS trajectory."""
    cfg = dict(max_tree_depth=10, step_size=0.3, metric="dense")
    Xn = draw_samples(nuts(**cfg)(PROB.model, seed=0), 100, 300)
    Xs = draw_samples(simple_nuts(**cfg)(PROB.model, seed=0), 100, 300)
    assert np.array_equal(Xn, Xs)


def test_grad_evals_leapfrog_is_one_per_leaf():
    """Plain NUTS (single potential leapfrog) costs one gradient per leaf, so total grad evals
    equals the summed leaf count."""
    s = nuts(max_tree_depth=8, step_size=0.3)(PROB.model, seed=2)
    s.warmup(100); s.sample(250)
    d = s.diagnostics("sampling")
    assert s.total_grad_evals() == d["n_leaves"].sum()
    hp = hmc(n_leapfrog=13, step_size=0.3)(PROB.model, seed=2)
    hp.warmup(100); hp.sample(120)
    assert np.allclose(hp.diagnostics("sampling")["grad_evals"], 13.0)   # n_leapfrog, single potential


def test_line_search_grad_eval_formula():
    """The per-level gradient-eval count is 2 + 2·Σ_{i=1}^{j-1} T_i + T_j (forward + backward search
    + re-integration): [3,4,10,22] for the doubling schedule."""
    model = PROB.model
    pot, kin = default_potentials(model), make_kinetic("diagonal")
    lsi = LineSearchIntegrator(leapfrog(pot, kin), pot, kin, schedule=doubling_schedule(4))
    assert np.array_equal(np.asarray(lsi._grad_evals_by_level), [3, 4, 10, 22])
    # for the extreme (2^-j, 1) schedule it reduces to 2j+1 for j>=1 (and 2+T_0=3 at j=0)
    flat = LineSearchIntegrator(leapfrog(pot, kin), pot, kin,
                                schedule=[(2.0 ** -j, 1) for j in range(4)])
    assert np.array_equal(np.asarray(flat._grad_evals_by_level), [3, 3, 5, 7])


def test_wal_nuts_costs_more_gradients_than_leaves():
    w = wal_nuts(step_size=0.5, schedule=doubling_schedule(6), error_thresholds=0.8,
                 max_tree_depth=8)(PROB.model, seed=0)
    w.warmup(120); w.sample(250)
    d = w.diagnostics("sampling")
    # every line-search leaf costs at least 2 + T_0 = 3 gradient evals (fwd+bwd probe + reintegrate)
    assert w.total_grad_evals() >= 3.0 * d["n_leaves"].sum() - 1e-6


def test_mh_has_minimal_diagnostics():
    m = adaptive_mh(step_size=0.5)(PROB.model, seed=0)
    m.warmup(100); m.sample(200)
    assert set(m.diagnostics("sampling")) == {"accept_prob", "accepted", "step_size"}
    assert 0.0 <= m.acceptance_rate() <= 1.0
    assert np.isnan(m.mean_n_leaves()) and m.total_grad_evals() == 0.0   # untracked for MH
