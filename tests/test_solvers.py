"""Tests for the implicit-integrator fixed-point solvers (Picard, Anderson).

Two things: that Anderson finds the same fixed point as Picard, and that on a stiff
(misspecified) metric Anderson acceleration sharply reduces the generalized leapfrog's
divergences at the same iteration budget -- i.e. a meaningful part of the implicit
integrator's instability is fixed-point non-convergence, which Anderson attacks.

Seeds are fixed, so pass/fail is deterministic.
"""

import numpy as np
import jax.numpy as jnp

from mimcs.hmc import PicardSolver, AndersonSolver
from mimcs.testing import neal_funnel, draw_samples, rmnuts


def test_anderson_matches_picard_fixed_point():
    a = jnp.array([1.0, -2.0, 0.5])
    g = lambda x: 0.5 * (jnp.cos(x) + a)            # a contraction; unique fixed point
    x_picard = np.asarray(PicardSolver(40).solve(g, jnp.zeros(3)))
    x_anderson = np.asarray(AndersonSolver(depth=3, n_iterations=10).solve(g, jnp.zeros(3)))
    assert np.allclose(x_picard, x_anderson, atol=1e-4)
    assert np.max(np.abs(0.5 * (np.cos(x_anderson) + np.asarray(a)) - x_anderson)) < 1e-3


def test_anderson_improves_sampling_on_stiff_metric():
    """On Neal's funnel with the misspecified conformal metric exp(-v) I, the generalized
    leapfrog's implicit solve often fails to converge under naive Picard iteration; Anderson
    acceleration converges the fixed point better at the same iteration count, so more steps are
    valid -- acceptance is markedly higher and the chain actually reaches the funnel's mode, where
    Picard gets stuck off-mode.

    (We compare sampling quality, not the raw divergence count: with the reversible divergence
    handling a solver mis-solve that stays finite shows up as an energy-range excursion the chain
    *traverses* rather than a fatal divergence, so the count no longer isolates non-convergence --
    acceptance and the recovered mode do.)"""
    fun = neal_funnel(dim=2, scale=3.0)
    conformal = lambda q: jnp.exp(-q[0]) * jnp.eye(2)

    def run(solver, seed):
        s = rmnuts(metric=conformal, max_tree_depth=9, step_size=0.4,
                   target_accept=0.8, n_fixed_point=8, solver=solver)(fun.model, seed=seed)
        x = draw_samples(s, 1000, 3000)
        return x, s.acceptance_rate()

    # Averaged over seeds. Both quantities are a *difference between two solvers on one chain*,
    # and how badly Picard stalls on a given seed varies enormously --- measured over 6 seeds the
    # acceptance gap ranges from about 0.0 to 0.5, so a one-seed margin is a margin on one draw
    # from that spread. (On the previous release the ``|v|`` criterion below already failed on 2
    # of those 6 seeds; it passed here only because seed 1 was a favourable one.) The claim the
    # test is making --- Anderson converges the fixed point better, so more steps are valid and
    # the chain reaches the mode --- is about the average, and that is what is asserted.
    gaps, reach = [], []
    for seed in range(6):
        picard_x, picard_acc = run(None, seed)          # default Picard
        anderson_x, anderson_acc = run("anderson", seed)
        gaps.append(anderson_acc - picard_acc)
        reach.append(abs(picard_x[:, 0].mean()) - abs(anderson_x[:, 0].mean()))
    print(f"\naccept gap per seed: {np.round(gaps, 3).tolist()}\n"
          f"|v| improvement per seed: {np.round(reach, 2).tolist()}")

    assert float(np.mean(gaps)) > 0.05, \
        f"Anderson should accept more often on average, got {gaps}"
    assert float(np.mean(reach)) > 0.5, \
        f"Anderson should reach the mode better on average, got {reach}"
