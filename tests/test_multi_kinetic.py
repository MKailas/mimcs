"""Tests for the multi-kinetic ``BaseHMC`` refactor.

``BaseHMC`` holds a *list* of kinetic components, each acting on a coordinate block --- so a
block-diagonal constant mass is just a list of ``Diagonal`` / ``Dense`` kinetics (no bespoke
composite), and a per-block *kind* (e.g. relativistic on one block, diagonal on another)
composes for free. These tests cover the block-diagonal mass end to end and the headline
composition the refactor unlocks.
"""

import numpy as np

from mimcs.testing import block_gaussian, block_hmc, block_nuts, evaluate
from mimcs.hmc import (
    RelativisticKinetic, DiagonalQuadraticKinetic, HMC, default_potentials, leapfrog)
from mimcs.samplers import make_sampler_class
from mimcs.adaptation import RobbinsMonroStepSize, MassMatrixAdaptation


def test_block_kinetics_modes_and_flat_ham_params():
    """One kinetic per parameter, keyed by its own id in a flat ham_params dict."""
    prob = block_gaussian()
    s = block_hmc(modes={"a": "dense", "b": "diagonal"})(prob.model, 0)
    kinds = {k.id: type(k).__name__ for k in s.kinetics}
    assert kinds == {"a": "DenseQuadraticKinetic", "b": "DiagonalQuadraticKinetic"}
    assert s.kinetics[0].slices == [(0, 2)]     # block 'a' slice
    assert s.kinetics[1].slices == [(2, 5)]     # block 'b' slice
    assert set(s.state.ham_params) == {"a", "b"}


def test_block_nuts_samples_block_gaussian_end_to_end():
    """A dense-'a' / diagonal-'b' block-diagonal mass samples the block Gaussian correctly."""
    prob = block_gaussian()
    report = evaluate(prob, {"block": block_nuts(modes={"a": "dense", "b": "diagonal"})},
                      n_warmup=2000, n_samples=8000, seed=0, make_plots=False)
    print("\n" + report.summary())
    report.assert_correct()


def test_block_dense_beats_single_diagonal_on_correlated_block(artifacts_dir):
    """The dense-'a' block whitens 'a''s correlation that a single diagonal mass misses, so it
    mixes better on 'a' (higher ESS) while both remain correct."""
    prob = block_gaussian(cov_a=((3.0, 2.7), (2.7, 3.0)))
    report = evaluate(
        prob,
        {"block": block_nuts(modes={"a": "dense", "b": "diagonal"}),
         "diag": block_nuts(modes={"a": "diagonal", "b": "diagonal"})},
        n_warmup=2500, n_samples=10000, seed=0, out_dir=str(artifacts_dir / "block_gaussian"))
    print("\n" + report.summary())
    report.assert_correct()
    assert report.outputs["block"].ess[:2].min() > report.outputs["diag"].ess[:2].min()


def test_relativistic_block_composed_with_diagonal_block():
    """The headline of the refactor: a *relativistic* kinetic on block 'a' composed with a
    *diagonal* kinetic on block 'b' --- just two components in the list, no new class. The
    diagonal block is mass-adapted (the relativistic one keeps a fixed mass; mass_mode=None)."""
    prob = block_gaussian()
    model = prob.model
    sa, ea = model.coord_block("a")
    sb, eb = model.coord_block("b")
    kinetics = [RelativisticKinetic((ea - sa,), id="a", slices=[(sa, ea)]),
                DiagonalQuadraticKinetic(id="b", slices=[(sb, eb)])]
    Cls = make_sampler_class(RobbinsMonroStepSize, MassMatrixAdaptation, HMC)

    def builder(m, seed):
        pots = default_potentials(m)
        return Cls(m, init_position=np.zeros(m.ambient_dim), seed=seed, kinetics=kinetics,
                   potentials=pots, integrator=leapfrog(pots, kinetics),
                   n_leapfrog=20, step_size=0.4, target_accept=0.8)

    report = evaluate(prob, {"rel_a+diag_b": builder},
                      n_warmup=2000, n_samples=8000, seed=0, make_plots=False)
    print("\n" + report.summary())
    report.assert_correct()
    # only the diagonal 'b' block is covariance-adapted; the relativistic 'a' keeps fixed mass
    assert set(report.outputs) == {"rel_a+diag_b"}
