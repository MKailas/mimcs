"""Tests for the multi-rate (RESPA) integrator (``docs/design/06_hamiltonian_monte_carlo.md``).

One multi-rate step over a Gaussian split into two precision components is a *linear map*,

    K_exp(eps/2) [ K_cheap(d/2) D(d) K_cheap(d/2) ]^n K_exp(eps/2),   d = eps/n,

so the same checks `test_integrators.py` applies to leapfrog apply here: exactness against the
analytically composed matrix, reversibility, and second-order error in the step size. Two more
are specific to this integrator: the cost accounting (`#expensive + n * #cheap` gradients per
outer step --- the whole point of the thing) and the freshness of every potential's cache at the
outer-step endpoint, which is what `total_energy` and NUTS's resumption rely on.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest
import scipy.linalg

from mimcs.model import Model, EuclideanParameter, PositiveParameter
from mimcs.hmc import (HamiltonianContext, LineSearchIntegrator, RepeatedIntegrator,
                      default_potentials, doubling_schedule, init_integrator_state, leapfrog,
                      make_kinetic, multirate_leapfrog, split_potentials)
from mimcs.testing import correlated_gaussian, evaluate, nuts, multirate_nuts, TargetProblem


P_CHEAP = np.array([[1.0, 0.2], [0.2, 1.5]])
P_EXPENSIVE = np.array([[3.0, -0.4], [-0.4, 2.0]])
INV_MASS = np.array([1.3, 0.7])


def _split_model(cheap=("cheap",)):
    """A 2-D Gaussian whose precision is split into a declared-cheap and an expensive part."""
    return Model(
        [EuclideanParameter("x", (2,))],
        {"cheap": lambda p: -0.5 * p["x"] @ jnp.asarray(P_CHEAP) @ p["x"],
         "expensive": lambda p: -0.5 * p["x"] @ jnp.asarray(P_EXPENSIVE) @ p["x"]},
        cheap_components=cheap)


def _setup(model=None):
    model = model if model is not None else _split_model()
    potentials = default_potentials(model)
    kinetic = make_kinetic("diagonal")
    ctx = HamiltonianContext(model.init_chart_hyperparams(), model.init_chart_indices(),
                             {"T": jnp.asarray(INV_MASS, float)})
    cheap, expensive = split_potentials(model, potentials)
    return model, potentials, kinetic, ctx, cheap, expensive


def _state(potentials, q, p, ctx, integrator=None):
    st = init_integrator_state(potentials, jnp.asarray(q, float), jnp.asarray(p, float), ctx)
    if integrator is not None:
        st = st._replace(integrator_data=integrator.init_integrator_data())
    return st


# --- the analytic reference -------------------------------------------------- #

def _kick(precision, eps):
    m = np.eye(4)
    m[2:, :2] = -eps * precision           # p <- p - eps * grad V,  grad V = precision @ q
    return m


def _drift(inv_mass, eps):
    m = np.eye(4)
    m[:2, 2:] = eps * np.diag(inv_mass)    # q <- q + eps * M^-1 p
    return m


def _multirate_step_matrix(p_cheap, p_expensive, inv_mass, eps, n):
    inner = _kick(p_cheap, 0.5 * eps / n) @ _drift(inv_mass, eps / n) @ _kick(p_cheap, 0.5 * eps / n)
    # matrices compose right-to-left, so the leading kick is the rightmost factor
    return (_kick(p_expensive, 0.5 * eps)
            @ np.linalg.matrix_power(inner, n)
            @ _kick(p_expensive, 0.5 * eps))


# --- the integrator ---------------------------------------------------------- #

@pytest.mark.parametrize("n", [1, 4])
def test_multirate_matches_the_analytic_linear_map(n):
    _, potentials, kinetic, ctx, cheap, expensive = _setup()
    integ = multirate_leapfrog(cheap, expensive, kinetic, n=n)
    q0, p0 = np.array([0.5, 0.3]), np.array([-0.4, 0.9])
    eps = 0.1
    step = _multirate_step_matrix(P_CHEAP, P_EXPENSIVE, INV_MASS, eps, n)

    for n_steps in (1, 5, 20):
        out = integ.integrate(_state(potentials, q0, p0, ctx), eps, n_steps, ctx)
        ref = np.linalg.matrix_power(step, n_steps) @ np.concatenate([q0, p0])
        assert np.allclose(np.asarray(out.q), ref[:2], atol=1e-4, rtol=1e-4)
        assert np.allclose(np.asarray(out.p), ref[2:], atol=1e-4, rtol=1e-4)


def test_multirate_with_one_substep_is_leapfrog():
    """``n = 1`` is the same op sequence as leapfrog over the union (expensive first)."""
    _, potentials, kinetic, ctx, cheap, expensive = _setup()
    q0, p0 = [0.3, -0.5], [1.0, 0.7]
    out = multirate_leapfrog(cheap, expensive, kinetic, n=1).step(
        _state(potentials, q0, p0, ctx), 0.3, ctx)
    ref = leapfrog(expensive + cheap, kinetic).step(_state(potentials, q0, p0, ctx), 0.3, ctx)
    assert np.allclose(np.asarray(out.q), np.asarray(ref.q), atol=1e-6)
    assert np.allclose(np.asarray(out.p), np.asarray(ref.p), atol=1e-6)


def test_multirate_is_reversible():
    _, potentials, kinetic, ctx, cheap, expensive = _setup()
    integ = multirate_leapfrog(cheap, expensive, kinetic, n=3)
    q0, p0 = jnp.array([0.7, -0.2]), jnp.array([0.1, 0.5])
    eps, n_steps = 0.15, 20
    fwd = integ.integrate(_state(potentials, q0, p0, ctx), eps, n_steps, ctx)
    back = integ.integrate(_state(potentials, fwd.q, -fwd.p, ctx), eps, n_steps, ctx)
    assert np.allclose(np.asarray(back.q), np.asarray(q0), atol=1e-4)
    assert np.allclose(np.asarray(back.p), -np.asarray(p0), atol=1e-4)


def test_multirate_is_second_order_accurate():
    _, potentials, kinetic, ctx, cheap, expensive = _setup()
    integ = multirate_leapfrog(cheap, expensive, kinetic, n=2)
    q0, p0 = np.array([0.6, -0.4]), np.array([0.2, 0.8])
    total_time = 1.0
    generator = np.zeros((4, 4))
    generator[:2, 2:] = np.diag(INV_MASS)
    generator[2:, :2] = -(P_CHEAP + P_EXPENSIVE)
    exact = scipy.linalg.expm(total_time * generator) @ np.concatenate([q0, p0])

    def error_at(eps):
        out = integ.integrate(_state(potentials, q0, p0, ctx), eps,
                              int(round(total_time / eps)), ctx)
        got = np.concatenate([np.asarray(out.q), np.asarray(out.p)])
        return float(np.linalg.norm(got - exact))

    ratio = error_at(0.05) / error_at(0.025)
    assert 3.0 < ratio < 5.0, f"order ratio {ratio:.2f} is not ~4 (second order)"


def test_multirate_leaves_log_weight_zero():
    _, potentials, kinetic, ctx, cheap, expensive = _setup()
    integ = multirate_leapfrog(cheap, expensive, kinetic, n=4)
    out = integ.integrate(_state(potentials, [0.4, 0.1], [0.2, -0.6], ctx), 0.2, 10, ctx)
    assert float(out.log_weight) == 0.0


def test_multirate_costs_one_expensive_gradient_per_n_cheap_ones():
    """The point of the integrator, asserted on the cost counter the samplers report."""
    _, potentials, kinetic, ctx, cheap, expensive = _setup()
    n = 4
    integ = multirate_leapfrog(cheap, expensive, kinetic, n=n)
    assert integ._grad_evals == len(expensive)      # the outer's own (non-cached) kicks

    st = _state(potentials, [0.4, 0.1], [0.2, -0.6], ctx, integ)
    per_step = len(expensive) + n * len(cheap)
    for k in (1, 2, 7):
        out = integ.integrate(st, 0.1, k, ctx)
        assert float(out.integrator_data["grad_evals"]) == pytest.approx(k * per_step)
    # ... against plain leapfrog's one gradient per potential per step
    plain = leapfrog(potentials, kinetic)
    out = plain.integrate(_state(potentials, [0.4, 0.1], [0.2, -0.6], ctx, plain), 0.1, 7, ctx)
    assert float(out.integrator_data["grad_evals"]) == pytest.approx(7 * len(potentials))


def test_every_cache_is_fresh_at_the_outer_step_endpoint():
    """The executable form of the cache-validity argument: `total_energy` and NUTS's resumption
    read cached values, so no potential may be left stale at a step boundary."""
    _, potentials, kinetic, ctx, cheap, expensive = _setup()
    integ = multirate_leapfrog(cheap, expensive, kinetic, n=3)
    out = integ.integrate(_state(potentials, [0.4, 0.1], [0.2, -0.6], ctx), 0.2, 4, ctx)
    for potential in potentials:
        value, grad = potential.value_and_grad(out.q, ctx)
        assert np.allclose(np.asarray(out.potential_values[potential.id]), np.asarray(value),
                           atol=1e-6), f"{potential.id} value is stale"
        assert np.allclose(np.asarray(out.potential_grads[potential.id]), np.asarray(grad),
                           atol=1e-6), f"{potential.id} gradient is stale"


@pytest.mark.parametrize("kwargs, match", [
    (dict(n=0), "n >= 1"),
    (dict(cheap=[]), "no cheap potential"),
    (dict(expensive=[]), "no expensive potential"),
])
def test_multirate_rejects_a_degenerate_split(kwargs, match):
    _, potentials, kinetic, ctx, cheap, expensive = _setup()
    groups = {"cheap": cheap, "expensive": expensive, "n": 4}
    groups.update(kwargs)
    with pytest.raises(ValueError, match=match):
        multirate_leapfrog(groups["cheap"], groups["expensive"], kinetic, n=groups["n"])


def test_repeated_integrator_rejects_a_randomized_inner():
    _, potentials, kinetic, ctx, cheap, expensive = _setup()
    from mimcs.hmc import MarkovianLineSearchIntegrator
    randomized = MarkovianLineSearchIntegrator(leapfrog(potentials, kinetic), potentials, kinetic,
                                               schedule=doubling_schedule(3))
    with pytest.raises(ValueError, match="deterministic inner integrator"):
        RepeatedIntegrator(randomized, 4)


# --- the split ---------------------------------------------------------------- #

def test_split_potentials_puts_the_jacobian_with_the_cheap_group():
    model = Model([EuclideanParameter("x", (2,)), PositiveParameter("s")],
                  {"cheap": lambda p: -0.5 * jnp.sum(p["x"] ** 2),
                   "expensive": lambda p: -jnp.log(p["s"])},
                  cheap_components={"cheap"})
    cheap, expensive = split_potentials(model, default_potentials(model))
    assert [p.id for p in cheap] == ["V_cheap", "V_jacobian"]
    assert [p.id for p in expensive] == ["V_expensive"]


def test_split_potentials_calls_an_undeclared_model_all_expensive():
    """A model that declares nothing cheap leaves only the Jacobian in the cheap group --- which
    is why the factory rule needs a *declared* component before it goes multi-rate."""
    model = Model([EuclideanParameter("x", (2,)), PositiveParameter("s")],
                  {"lp": lambda p: -0.5 * jnp.sum(p["x"] ** 2) - jnp.log(p["s"])})
    cheap, expensive = split_potentials(model, default_potentials(model))
    assert [p.id for p in cheap] == ["V_jacobian"]
    assert [p.id for p in expensive] == ["V_lp"]


# --- as a line-search base ---------------------------------------------------- #

def test_line_search_runs_over_a_multirate_base():
    """`LineSearchIntegrator` drives any base with a structure-preserving `step`, so a whole
    multi-rate step can be the thing it refines."""
    _, potentials, kinetic, ctx, cheap, expensive = _setup()
    base = multirate_leapfrog(cheap, expensive, kinetic, n=2)
    lsi = LineSearchIntegrator(base, potentials, kinetic, schedule=doubling_schedule(3))
    out = lsi.integrate(_state(potentials, [0.4, 0.1], [0.2, -0.6], ctx), 0.3, 5, ctx)
    assert bool(jnp.isfinite(out.q).all()) and float(out.integrator_data["grad_evals"]) > 0
    # the per-level table still counts *base steps* (a known undercount for a base that costs
    # more than one gradient per step --- doc 06's open question)
    assert np.array_equal(np.asarray(lsi._grad_evals_by_level), [3, 4, 10])


# --- end to end --------------------------------------------------------------- #

def test_multirate_nuts_samples_the_right_distribution(artifacts_dir):
    """Correctness against the analytic Gaussian, and agreement with plain leapfrog NUTS ---
    `evaluate` compares every named sampler against the reference and against each other."""
    cov = np.linalg.inv(P_CHEAP + P_EXPENSIVE)
    chol = np.linalg.cholesky(cov)
    problem = TargetProblem(
        name="split_gaussian", model=_split_model(), dim=2, labels=["x0", "x1"],
        exact_sampler=lambda n, rng: rng.standard_normal((n, 2)) @ chol.T,
        mean=np.zeros(2), cov=cov)
    report = evaluate(problem,
                      {"leapfrog": nuts(step_size=0.5), "multirate": multirate_nuts(n=4,
                                                                                    step_size=0.5)},
                      n_warmup=2000, n_samples=8000, seed=0, make_plots=False)
    print("\n" + report.summary())
    report.assert_correct()


def test_nuts_and_simple_nuts_agree_with_the_multirate_integrator():
    """The standing oracle invariant, with the multi-rate integrator as the leaf stepper."""
    from mimcs.hmc import NUTS, SimpleNUTS
    from mimcs.samplers import make_sampler_class
    from mimcs.testing import draw_samples

    def build(Cls):
        model, potentials, kinetic, _, cheap, expensive = _setup()
        integ = multirate_leapfrog(cheap, expensive, kinetic, n=3)
        return make_sampler_class(Cls)(
            model, init_position=model.default_sample(), seed=0, kinetic=kinetic,
            potentials=potentials, integrator=integ, step_size=0.4, max_tree_depth=6)

    assert np.array_equal(draw_samples(build(NUTS), 100, 300),
                          draw_samples(build(SimpleNUTS), 100, 300))
