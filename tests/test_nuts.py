"""Correctness tests for the No-U-Turn Sampler.

NUTS is validated through the same framework as HMC: against analytic references
(Gaussian diagonal+dense, mild banana, constrained positive) and against fixed-length
HMC as an oracle. Two NUTS-specific checks confirm the tree machinery: zero divergences
and naturally-terminating (sub-max) tree depth on an easy target.

Neal's funnel is handled honestly. Basic NUTS with a *global* metric does not sample the
deep funnel cleanly -- it misses the neck and diverges there (the classic funnel
pathology, the motivation for Riemannian / reparameterized methods). So we test a mild
funnel (scale=1) strictly, and use the deep funnel (scale=3) to confirm the divergence
diagnostic correctly flags the pathological geometry rather than asserting correctness.

Seeds are fixed, so pass/fail is deterministic.
"""

import numpy as np

from mimcs.testing import (
    correlated_gaussian, rosenbrock, positive_lognormal, neal_funnel,
    evaluate, nuts, simple_nuts, randomized_hmc, draw_samples)


def test_nuts_matches_simple_nuts():
    """The memory-efficient NUTS must trace the IDENTICAL chain as the reference
    full-memory SimpleNUTS given the same RNG -- they make the same U-turn and
    multinomial decisions, so the checkpoint bookkeeping is verified bit-for-bit."""
    problem = correlated_gaussian(mean=[1.0, -2.0], cov=[[2.0, 1.4], [1.4, 1.5]])
    cfg = dict(max_tree_depth=10, step_size=0.3, metric="dense")
    Xn = draw_samples(nuts(**cfg)(problem.model, seed=0), 500, 2000)
    Xs = draw_samples(simple_nuts(**cfg)(problem.model, seed=0), 500, 2000)
    assert np.array_equal(Xn, Xs)


def _stan_build_tree(P, V, depth, start=0):
    """Stan's recursive ``build_tree`` (since 2019) on precomputed leaves: ``(valid, n_consumed)``.

    Returns early on an invalid half; at every merge of ``[a..m]`` with ``[m+1..b]`` demands no
    U-turn over ``[a..b]``, ``[a..m+1]`` and ``[m..b]``. No ntz/checkpoint bookkeeping in it."""
    if depth == 0:
        return True, 1
    half = 1 << (depth - 1)
    ok, c = _stan_build_tree(P, V, depth - 1, start)
    if not ok:
        return False, c
    ok, c2 = _stan_build_tree(P, V, depth - 1, start + half)
    if not ok:
        return False, c + c2
    a, m, b = start, start + half - 1, start + 2 * half - 1

    def turns(lo, hi):
        rho = P[lo:hi + 1].sum(0)
        return rho @ V[lo] <= 0.0 or rho @ V[hi] <= 0.0

    return not (turns(a, b) or turns(a, m + 1) or turns(m, b)), c + c2


def test_subtree_builders_match_stans_recursive_rule():
    """Both builders' subtree verdicts against a recursive reference of Stan's 2019 rule, on exact
    leapfrog orbits of a near-isotropic Gaussian -- the regime where the extra checks matter.

    Control: with ``extra_uturn_checks=False`` the builders must disagree with the reference on
    some cases, so the match is not vacuous (were the extra checks never to fire, it would be).
    """
    import jax
    import jax.numpy as jnp
    from mimcs.hmc import NUTS, SimpleNUTS
    from mimcs.hmc.state import IntegratorState

    d, J = 6, 8
    model = correlated_gaussian(mean=np.zeros(d), cov=np.eye(d)).model   # U = |q|^2/2, unit mass
    q0 = np.linspace(-1.0, 1.0, d)
    rng = np.random.default_rng(0)
    cases = [(rng.standard_normal(d), eps, depth)
             for eps in (0.3, 0.55, 0.8, 1.1, 1.4) for depth in (2, 4, 6, 7) for _ in range(3)]

    def reference(p0, eps, depth):
        q, p = q0.copy(), p0.copy()
        P = np.empty((1 << depth, d))
        for k in range(1 << depth):
            p = p - 0.5 * eps * q
            q = q + eps * p
            p = p - 0.5 * eps * q
            P[k] = p
        valid, n = _stan_build_tree(P, P, depth)
        return (not valid), n

    def outcomes(Cls, extra):
        s = Cls(model, q0, seed=0, step_size=0.5, max_tree_depth=J, extra_uturn_checks=extra)
        st = s.state
        ctx = s.context(st)

        @jax.jit
        def subtree(p0, eps, depth):
            z0 = IntegratorState(
                q=st.coordinate, p=p0, potential_values=st.potential_values,
                potential_grads=st.potential_grads, log_weight=jnp.zeros(()),
                integrator_data=s.integrator.init_integrator_data())
            sub = s._build_subtree(z0, eps, depth, s.total_energy(z0, ctx),
                                   jnp.full(((1 << J) - 1,), 0.5), None, ctx)
            return sub.terminated, sub.n_leaves

        return [tuple(np.asarray(x).item() for x in subtree(
                    jnp.asarray(p0, float), jnp.asarray(eps, float), jnp.int32(depth)))
                for p0, eps, depth in cases]

    ref = [reference(*c) for c in cases]
    for Cls in (NUTS, SimpleNUTS):
        assert outcomes(Cls, True) == ref, Cls.__name__
    n_differ = sum(o != r for o, r in zip(outcomes(NUTS, False), ref))
    assert n_differ > 0, "control: the original rule never differed, so nothing was tested"


def test_extra_uturn_checks_stop_the_isotropic_runaway():
    """The defect the extra checks fix, with the chain oracle riding on it.

    On an isotropic Gaussian with an exactly right (unit) mass, the original rule lets a U-turn
    straddling two subtrees go unseen and trajectories loop around the mode: mean tree depth
    6.3-6.9 at d = 20, step 0.78 (measured on these seeds), against ~2.8 with the checks. This is
    also where the NUTS/SimpleNUTS bit-identity is non-vacuous for the new checkpoint arrays,
    since the checks demonstrably change the chain here.
    """
    from mimcs.hmc import NUTS, SimpleNUTS

    model = correlated_gaussian(mean=np.zeros(20), cov=np.eye(20)).model
    init = np.asarray(model.default_sample(), float)

    def run(Cls, seed, extra):
        s = Cls(model, init, seed=seed, step_size=0.78, extra_uturn_checks=extra)
        s.sample(150)
        return s.get_samples_flat(), s.diagnostics("sampling")["tree_depth"]

    for seed in range(3):
        x_on, depth_on = run(NUTS, seed, True)
        x_off, depth_off = run(NUTS, seed, False)
        assert np.array_equal(x_on, run(SimpleNUTS, seed, True)[0]), seed
        assert not np.array_equal(x_on, x_off), seed
        assert depth_on.mean() < 3.5 and depth_on.max() < 10, (seed, depth_on.mean())
        assert depth_off.mean() > 5.5, (seed, depth_off.mean())   # control: the runaway is real
        assert abs((x_on ** 2).mean() - 1.0) < 0.15, (seed, (x_on ** 2).mean())


def test_nuts_correlated_gaussian_diagonal(artifacts_dir):
    problem = correlated_gaussian(mean=[1.0, -2.0], cov=[[2.0, 1.4], [1.4, 1.5]])
    report = evaluate(
        problem, {"nuts": nuts(max_tree_depth=10, step_size=0.3, metric="diagonal")},
        n_warmup=2000, n_samples=8000, seed=0,
        out_dir=str(artifacts_dir / "nuts_gaussian_diagonal"))
    print("\n" + report.summary())
    report.assert_correct()


def test_nuts_correlated_gaussian_dense(artifacts_dir):
    problem = correlated_gaussian(mean=[1.0, -2.0], cov=[[2.0, 1.4], [1.4, 1.5]])
    report = evaluate(
        problem, {"nuts": nuts(max_tree_depth=10, step_size=0.3, metric="dense")},
        n_warmup=2000, n_samples=8000, seed=1,
        out_dir=str(artifacts_dir / "nuts_gaussian_dense"))
    print("\n" + report.summary())
    report.assert_correct()


def test_nuts_rosenbrock_banana(artifacts_dir):
    # Mild banana (b=1); a stiff banana is a known global-metric limitation (see HMC tests).
    problem = rosenbrock(a=1.0, b=1.0)
    report = evaluate(
        problem, {"nuts": nuts(max_tree_depth=10, step_size=0.2, target_accept=0.9)},
        n_warmup=2000, n_samples=8000, seed=2,
        out_dir=str(artifacts_dir / "nuts_rosenbrock"))
    print("\n" + report.summary())
    report.assert_correct()


def test_nuts_positive_parameter(artifacts_dir):
    problem = positive_lognormal(sigma=1.0)
    report = evaluate(
        problem, {"nuts": nuts(init=np.array([1.0]), max_tree_depth=10, step_size=0.3)},
        n_warmup=2000, n_samples=8000, seed=3,
        out_dir=str(artifacts_dir / "nuts_positive"))
    print("\n" + report.summary())
    report.assert_correct()


def test_nuts_agrees_with_randomized_hmc(artifacts_dir):
    """Oracle: NUTS (no trajectory-length tuning) and randomized HMC must agree, and both
    match analytic. (Randomized HMC is used rather than fixed-length HMC because the
    latter is resonance-prone with a dense metric on a well-conditioned Gaussian.)"""
    problem = correlated_gaussian(
        mean=[0.0, 0.0, 0.0],
        cov=[[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]])
    report = evaluate(
        problem,
        {
            "nuts": nuts(max_tree_depth=10, step_size=0.3, metric="dense"),
            "rhmc": randomized_hmc(n_leapfrog=20, step_size=0.3, metric="dense"),
        },
        n_warmup=2000, n_samples=8000, seed=4,
        out_dir=str(artifacts_dir / "nuts_vs_randomized_hmc"))
    print("\n" + report.summary())
    report.assert_correct()


def test_nuts_no_divergences_and_terminates_on_gaussian():
    """Tree machinery sanity on an easy target: no divergences, and the U-turn fires
    (mean depth modest, max depth below the cap -- not just hitting max_tree_depth)."""
    problem = correlated_gaussian(mean=[1.0, -2.0], cov=[[2.0, 1.4], [1.4, 1.5]])
    sampler = nuts(max_tree_depth=10, step_size=0.3, metric="dense")(problem.model, seed=0)
    draw_samples(sampler, 1500, 5000)
    assert sampler.divergence_count() == 0
    assert 1.0 <= sampler.mean_tree_depth() <= 7.0
    assert sampler.diagnostics("all")["tree_depth"].max() < 10   # natural termination, not the cap


def test_nuts_counts_warmup_and_sampling_divergences_separately():
    """Warmup and sampling divergences are bucketed apart: the default counts sampling-phase only,
    ``include_warmup=True`` adds the warmup bucket, and the phases partition the total. The deep
    funnel diverges in both phases, so it exercises both buckets."""
    problem = neal_funnel(dim=2, scale=3.0)
    sampler = nuts(max_tree_depth=10, step_size=0.3, target_accept=0.9)(problem.model, seed=5)
    draw_samples(sampler, 2000, 4000)
    sampling = sampler.divergence_count()                              # default: sampling only
    warmup = sampler.divergence_count(include_warmup=True, include_sampling=False)
    both = sampler.divergence_count(include_warmup=True, include_sampling=True)
    # cross-check against the uniform diagnostics store
    assert sampling == int(sampler.diagnostics("sampling")["diverging"].sum())
    assert warmup == int(sampler.diagnostics("warmup")["diverging"].sum())
    assert both == warmup + sampling
    assert warmup > 0 and sampling > 0                                # the funnel diverges in both
    # the rate is over the selected phase(s), matching the count
    assert np.isclose(sampler.divergence_rate(), sampling / 4000)


def test_nuts_mild_funnel(artifacts_dir):
    """A mild funnel (scale=1) is within global-metric NUTS's reach: strict check."""
    problem = neal_funnel(dim=2, scale=1.0)
    problem.hard = False   # mild enough for a real correctness assertion
    report = evaluate(
        problem, {"nuts": nuts(max_tree_depth=10, step_size=0.3, target_accept=0.9)},
        n_warmup=2000, n_samples=8000, seed=3,
        out_dir=str(artifacts_dir / "nuts_mild_funnel"))
    print("\n" + report.summary())
    report.assert_correct()


def test_nuts_divergence_diagnostic_on_deep_funnel(artifacts_dir):
    """Deep funnel (scale=3): basic NUTS with a global metric cannot sample the neck and
    diverges there. We do not assert distributional correctness (known limitation); we
    confirm the divergence diagnostic fires, alerting the user to the pathology. Plots
    are written for inspection."""
    problem = neal_funnel(dim=2, scale=3.0)
    # plots for inspection (the deep funnel is hard=True, so no distributional assertion)
    report = evaluate(
        problem, {"nuts": nuts(max_tree_depth=10, step_size=0.3, target_accept=0.9)},
        n_warmup=2000, n_samples=8000, seed=5,
        out_dir=str(artifacts_dir / "nuts_deep_funnel"))
    report.assert_correct()            # skipped (problem.hard is True)

    # read the divergence diagnostic directly from a sampler instance
    sampler = nuts(max_tree_depth=10, step_size=0.3, target_accept=0.9)(problem.model, seed=5)
    draw_samples(sampler, 2000, 8000)
    print(f"deep funnel divergences: {sampler.divergence_count()} / 8000")
    assert sampler.divergence_count() > 10   # the diagnostic detects the pathological neck
