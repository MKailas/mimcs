"""Jump operators with **discrete outputs**, and **restricted recomputation** for jumps.

Discrete outputs: a jump on ``g[c]`` may also rewrite *other* integer parameters. A label moves under
counting measure, so it adds no Jacobian; what it adds is validation --- a returned label that is
non-integral or out of support is a proposal of target density 0 and must be rejected, never
written. JAX clamps an out-of-range gather silently, so without the mask an invalid move would be
evaluated at a *different*, plausible state.

Restricted recomputation: a jump's density difference needs only the components reading a moved
name, plus the moved continuous outputs' own chart-Jacobian difference, which --- unlike for a
label-only move --- does not cancel. The algebra was checked before any library code
(``tests/experiments/jump_restricted_algebra.py``); the checks below keep it, each with a control.
"""

import itertools

import numpy as np
import jax
import jax.numpy as jnp
import pytest

import mimcs
from mimcs.adaptation import RobbinsMonroStepSize
from mimcs.hmc import NUTS
from mimcs.model import EuclideanParameter, IntegerParameter, Model
from mimcs.model.bounded import BoundedParameter, PositiveParameter
from mimcs.model.jump import JumpOperator
from mimcs.samplers import (StaticContinuous, SystematicScanMetropolisWithinGibbs,
                            make_sampler_class)
from mimcs.samplers.discrete_updates import JumpMap
from mimcs.samplers.gibbs import jump_restriction_plan

NUTS_GIBBS = make_sampler_class(RobbinsMonroStepSize, SystematicScanMetropolisWithinGibbs, NUTS)
GIBBS_ONLY = make_sampler_class(SystematicScanMetropolisWithinGibbs, StaticContinuous)


# ================================================================ A. discrete outputs

W = jnp.asarray([0.2, 0.5, 0.3])
MU = jnp.asarray([-1.0, 0.4, 2.0])
SIG = jnp.asarray([0.7, 1.3, 0.5])
PS = jnp.asarray([[0.8, 0.2], [0.3, 0.7], [0.5, 0.5]])    # p(s | z)


def _toy_logp(v):
    """z ~ Cat(W); s | z ~ Cat(PS[z]); x | z ~ N(MU_z, SIG_z). Closed-form joint."""
    z, s, x = v["z"], v["s"], jnp.reshape(v["x"], ())
    return (jnp.log(W[z]) + jnp.log(PS[z, s]) - 0.5 * jnp.log(2 * jnp.pi * SIG[z] ** 2)
            - 0.5 * ((x - MU[z]) / SIG[z]) ** 2)


def _shift_and_flip(values, c, v):
    """A continuous shift plus an indicator flip on a parity change: both telescope, so the joint
    map is an involution and a cocycle."""
    g = values["z"]
    return values["x"] + (MU[v] - MU[g]), jnp.mod(values["s"] + (v - g), 2)


def _toy(fn=_shift_and_flip, outputs=("x", "s")):
    return Model([EuclideanParameter("x", ())], {"p": _toy_logp},
                 discrete_parameters=[IntegerParameter("z", (), lower=0, upper=2),
                                      IntegerParameter("s", (), lower=0, upper=1)],
                 jump_operators={"z": JumpOperator("z", outputs, fn)})


def _joint_errors(z, s, x):
    joint = np.asarray(W)[:, None] * np.asarray(PS)
    emp = np.zeros((3, 2))
    np.add.at(emp, (z, s), 1.0)
    emp /= emp.sum()
    dm = max(abs(x[z == k].mean() - float(MU[k])) for k in range(3))
    return float(np.abs(emp - joint).max()), dm


def test_a_discrete_output_builds_and_its_own_parameter_does_not():
    _toy()
    with pytest.raises(ValueError, match="itself among its outputs"):
        _toy(outputs=("x", "z"))


def test_a_labels_only_operator_may_not_claim_to_scale():
    with pytest.raises(ValueError, match="rewrites only discrete"):
        Model([], {"p": lambda v: jnp.zeros(())},
              discrete_parameters=[IntegerParameter("z", (), lower=0, upper=2),
                                   IntegerParameter("s", (), lower=0, upper=1)],
              jump_operators={"z": JumpOperator("z", ("s",), lambda v, c, x: (v["s"],),
                                                volume_preserving=False)})


def test_the_map_writes_the_discrete_output_and_flags_invalid_ones():
    m = _toy()
    jm = JumpMap(m.jump_operators["z"], m)
    h, idx = m.init_chart_hyperparams(), m.init_chart_indices()
    x = jnp.asarray([0.3])
    z = jnp.asarray([0, 1], jnp.int32)                       # z = 0, s = 1
    x2, z2, ok = jm.move(x, z, 0, jnp.int32(1), h, idx)
    assert bool(ok) and np.asarray(z2).tolist() == [0, 0]      # s flipped; z itself NOT set here
    assert abs(float(x2[0]) - (0.3 + float(MU[1] - MU[0]))) < 1e-6

    bad = _toy(lambda values, c, v: (values["x"], values["s"] + 2 * v))   # out of support
    _, z3, ok3 = JumpMap(bad.jump_operators["z"], bad).move(x, z, 0, jnp.int32(1), h, idx)
    assert not bool(ok3) and int(z3[1]) in (0, 1)              # rejected, and clipped in range
    frac = _toy(lambda values, c, v: (values["x"], values["s"] * 0.5))    # non-integral
    _, _, ok4 = JumpMap(frac.jump_operators["z"], frac).move(x, z, 0, jnp.int32(1), h, idx)
    assert not bool(ok4)


@pytest.mark.parametrize("kind", ["metropolis", "exact"])
def test_a_label_moving_jump_samples_the_closed_form_joint(kind):
    m = _toy()
    s = NUTS_GIBBS(m, m.default_sample(), seed=0, discrete_update={"z": kind, "s": "metropolis"})
    s.initialize(); s.warmup(600); s.sample(30000)
    d = s.get_samples()
    dz, dm = _joint_errors(np.asarray(d["z"]).ravel(), np.asarray(d["s"]).ravel(),
                           np.asarray(d["x"]).ravel())
    assert dz < 0.02, f"joint pmf of (z, s) off by {dz}"
    assert dm < 0.05, f"conditional mean off by {dm}"
    assert float(np.mean(s.diagnostics()["discrete_moves"])) > 0.05


def test_a_label_moving_jump_samples_right_under_tempering():
    from mimcs.pt import parallel_tempering
    m = _toy()
    s = parallel_tempering(m, n_temperatures=3, seed=0,
                           discrete_update={"z": "metropolis", "s": "metropolis"})
    s.initialize(); s.warmup(600); s.sample(20000)
    d = s.get_samples()
    dz, dm = _joint_errors(np.asarray(d["z"]).ravel(), np.asarray(d["s"]).ravel(),
                           np.asarray(d["x"]).ravel())
    assert dz < 0.03 and dm < 0.06, (dz, dm)


def test_a_non_involutive_discrete_output_is_refused():
    """CONTROL for the positive tests: moving `s` without telescoping (set it to 1 whenever the
    label changes) is not reversible, and the balance check must say so."""
    m = _toy(lambda values, c, v: (values["x"] + (MU[v] - MU[values["z"]]),
                                   jnp.where(v == values["z"], values["s"], 1)))
    with pytest.raises(ValueError, match="involution|identity"):
        NUTS_GIBBS(m, m.default_sample(), seed=0)


def _labels_only_model(mask_invalid=True):
    """Discrete-only: z in {0,1,2}, s in {0..3}; the jump moves s by the change in z. Exactly
    enumerable, so it runs under StaticContinuous. Out of support when s + (v - z) leaves 0..3."""
    lw = jnp.asarray([[0.1, 0.9, -0.4, 0.3], [0.7, -0.2, 0.5, 0.0], [-0.6, 0.4, 0.8, 1.1]])

    def lp(v):
        return lw[v["z"], v["s"]]

    return Model([], {"p": lp},
                 discrete_parameters=[IntegerParameter("z", (), lower=0, upper=2),
                                      IntegerParameter("s", (), lower=0, upper=3)],
                 jump_operators={"z": JumpOperator(
                     "z", ("s",), lambda values, c, v: (values["s"] + (v - values["z"]),))}), lw


def test_a_labels_only_jump_runs_on_a_static_base_and_is_exact():
    """Invalid moves (s pushed out of 0..3) are rejected, never written: the exact joint still
    comes out. A continuous-output jump under StaticContinuous still raises."""
    m, lw = _labels_only_model()
    s = GIBBS_ONLY(m, m.default_sample(), seed=0,
                   discrete_update={"z": "metropolis", "s": "metropolis"})
    s.initialize(); s.warmup(200); s.sample(60000)
    d = s.get_samples()
    emp = np.zeros((3, 4))
    np.add.at(emp, (np.asarray(d["z"]).ravel(), np.asarray(d["s"]).ravel()), 1.0)
    emp /= emp.sum()
    exact = np.exp(np.asarray(lw)); exact /= exact.sum()
    assert np.abs(emp - exact).max() < 0.012
    assert np.asarray(d["s"]).min() >= 0 and np.asarray(d["s"]).max() <= 3
    with pytest.raises(TypeError, match="freezes every continuous"):
        GIBBS_ONLY(_toy(), _toy().default_sample(), seed=0)


# ============================================================ B. restricted recomputation

E = jnp.asarray([-1.0, 0.5, 1.7])


def _restricted_model(reads=True, extra_component=True):
    """Several components, a logit-chart and a log-chart continuous output, and a discrete output.
    With `reads`, most components are skippable for a jump on `g`."""
    params = [EuclideanParameter("x", (3,)), PositiveParameter("tau", ()),
              EuclideanParameter("eta", (4,)), BoundedParameter("sb", (), lower=0.0, upper=5.0),
              PositiveParameter("lam", ())]
    disc = [IntegerParameter("g", (3,), lower=0, upper=2),
            IntegerParameter("d", (2,), lower=0, upper=1)]
    fns = {
        "lik": lambda v: (-0.5 * jnp.sum((v["eta"] - jnp.sum(E[v["g"]]) * v["sb"]) ** 2)
                          + 0.3 * jnp.sum(v["d"]) * jnp.log(jnp.reshape(v["lam"], ()))),
        "prior_x": lambda v: -0.5 * jnp.sum(v["x"] ** 2),
        "prior_tau": lambda v: -jnp.reshape(v["tau"], ()),
        "prior_eta": lambda v: -0.5 * jnp.sum(v["eta"] ** 2) / jnp.reshape(v["tau"], ()),
        "prior_d": lambda v: jnp.sum(v["d"] * v["x"][:2]),
        "prior_g": lambda v: jnp.sum(jnp.log(jnp.asarray([0.2, 0.5, 0.3])[v["g"]])),
        "prior_rest": lambda v: -0.5 * (jnp.reshape(v["sb"], ()) - 2.0) ** 2
                                - jnp.reshape(v["lam"], ()),
    }
    rd = {"lik": {"eta", "g", "sb", "d", "lam"}, "prior_x": {"x"}, "prior_tau": {"tau"},
          "prior_eta": {"eta", "tau"}, "prior_d": {"d", "x"}, "prior_g": {"g"},
          "prior_rest": {"sb", "lam"}}

    def jump(values, c, v):
        g = values["g"][c]
        return (values["eta"] + E[v] - E[g],
                values["sb"] * (1.0 + 0.1 * v) / (1.0 + 0.1 * g),
                values["lam"] * jnp.exp(0.4 * (v - g)),
                jnp.mod(values["d"] + (v - g), 2))

    op = JumpOperator("g", ("eta", "sb", "lam", "d"), jump, volume_preserving=False)
    return Model(params, fns, discrete_parameters=disc,
                 component_reads=rd if reads else {}, jump_operators={"g": op})


def test_the_plan_skips_what_no_moved_name_reaches():
    m = _restricted_model()
    fast, slow = jump_restriction_plan(m, "g")
    assert fast == [] and set(slow) == {"lik", "prior_eta", "prior_d", "prior_g", "prior_rest"}
    # CONTROL: without recorded reads everything reads everything, and nothing is gained
    assert jump_restriction_plan(_restricted_model(reads=False), "g") is None


def _full_vs_restricted(s, m, n=24, seed=0, jac_outputs=None):
    """Worst |restricted - full| over probe states, plus the same with the Jacobian dropped."""
    u = next(u for u in s.discrete_updaters if u.name == "g")
    h, idx = m.init_chart_hyperparams(), m.init_chart_indices()
    rng = np.random.default_rng(seed)
    worst = worst_nojac = 0.0
    for _ in range(n):
        x = jnp.asarray(rng.normal(size=(1, m.coord_dim)) * 0.5)
        z = jnp.asarray([np.concatenate([rng.integers(0, 3, 3), rng.integers(0, 2, 2)])],
                        jnp.int32)
        c, v = int(rng.integers(0, 3)), int(rng.integers(0, 3))
        cur = z[:, c]
        prop = jnp.full((1,), v, jnp.int32)
        x2, z2, ok = jax.vmap(u.map.move, in_axes=(0, 0, None, 0, None, None))(x, z, c, prop, h, idx)
        z2 = z2.at[:, c].set(v)
        full = float(m.log_prob_at_coordinate(x2[0], h, idx, z2[0])
                     - m.log_prob_at_coordinate(x[0], h, idx, z[0]))
        st = s.state
        got = float(s._jump_delta(st, x, x2, z, z2, "g", c, cur, prop, u.plan, u.map.outputs)[0])
        nojac = float(s._jump_delta(st, x, x2, z, z2, "g", c, cur, prop, u.plan, [])[0])
        worst = max(worst, abs(got - full))
        worst_nojac = max(worst_nojac, abs(nojac - full))
    return worst, worst_nojac


def test_the_restricted_jump_delta_equals_the_full_density_difference():
    m = _restricted_model()
    s = NUTS_GIBBS(m, m.default_sample(), seed=0, discrete_update={"g": "metropolis"})
    assert s.discrete_updaters[0].plan is not None                 # the new path is really taken
    worst, worst_nojac = _full_vs_restricted(s, m)
    tol = 1e-10 if jnp.zeros(()).dtype == jnp.float64 else 2e-4
    assert worst < tol, worst
    # CONTROL: the chart Jacobian does not cancel for a jump (logit and log charts here)
    assert worst_nojac > 1e-2, worst_nojac


def test_skipping_a_component_that_reads_an_output_would_be_wrong():
    """CONTROL on the plan: if `prior_eta` (which reads the continuous output) or `prior_d` (which
    reads the discrete output) were skipped, the difference would be off --- so the plan must keep
    them, and does."""
    m = _restricted_model()
    s = NUTS_GIBBS(m, m.default_sample(), seed=0, discrete_update={"g": "metropolis"})
    u = s.discrete_updaters[0]
    fast, slow = u.plan
    for comp in ("prior_eta", "prior_d"):
        assert comp in slow
        broken = (fast, [c for c in slow if c != comp])
        u.plan = broken
        worst, _ = _full_vs_restricted(s, m, n=8)
        assert worst > 1e-2, (comp, worst)
    u.plan = (fast, slow)


def test_a_restricted_jump_samples_the_same_posterior_as_the_full_path():
    """End to end. The full path is forced by clearing the plan on the same model and seed."""
    m = _restricted_model()

    def run(restricted, seed):
        s = NUTS_GIBBS(m, m.default_sample(), seed=seed,
                       discrete_update={"g": "metropolis", "d": "metropolis"})
        if not restricted:
            s.discrete_updaters[0].plan = None
        s.initialize(); s.warmup(400); s.sample(6000)
        d = s.get_samples()
        return np.concatenate([np.asarray(d["g"], float).mean(0),
                               np.asarray(d["d"], float).mean(0),
                               np.asarray(d["eta"]).mean(0)])

    a, b = run(True, 0), run(False, 0)
    # Different arithmetic, so different draws after the first rounding difference; the same
    # posterior within a generous Monte Carlo band.
    assert np.abs(a - b).max() < 0.15, (a, b)


def test_the_tempered_restricted_delta_keeps_each_component_s_own_beta():
    """With `tempered=` naming only `lik`, per-rung restricted deltas must equal per-rung full
    differences of the power posterior (tempered components scaled, the rest and the Jacobian not)."""
    from mimcs.pt import parallel_tempering
    from mimcs.pt.lanes import per_temperature_potential
    m = _restricted_model()
    K = 3
    s = parallel_tempering(m, n_temperatures=K, seed=0, tempered=["lik"],
                           discrete_update={"g": "metropolis"})
    u = next(u for u in s.discrete_updaters if u.name == "g")
    assert u.plan is not None
    st = s.state
    base = m
    h, idx = base.init_chart_hyperparams(), base.init_chart_indices()
    rng = np.random.default_rng(1)
    x = jnp.asarray(rng.normal(size=(K, base.coord_dim)) * 0.5)
    z = jnp.asarray(np.tile(np.concatenate([rng.integers(0, 3, 3), rng.integers(0, 2, 2)]),
                            (K, 1)), jnp.int32)
    c, v = 1, 2
    cur, prop = z[:, c], jnp.full((K,), v, jnp.int32)
    x2, z2, _ = jax.vmap(u.map.move, in_axes=(0, 0, None, 0, None, None))(x, z, c, prop, h, idx)
    z2 = z2.at[:, c].set(v)
    got = np.asarray(s._jump_delta(st, x, x2, z, z2, "g", c, cur, prop, u.plan, u.map.outputs))

    def full(xx, zz):
        return np.asarray(s._discrete_log_prob(st._replace(coordinate=xx.reshape(-1)),
                                               zz.reshape(-1)))

    want = full(x2, z2) - full(x, z)
    tol = 1e-8 if jnp.zeros(()).dtype == jnp.float64 else 5e-4
    assert np.abs(got - want).max() < tol, (got, want)
    # CONTROL: the ladder is non-trivial, so a single global beta could not have matched
    betas = np.asarray(s.context(st, kinetic_cache=False).betas)
    assert betas.min() < 0.9 * betas.max()


def test_another_label_s_update_reads_the_coordinate_a_jump_just_moved():
    """A regression test for a latent bug this work exposed. In a jump model every *other*
    parameter's label update runs on the delta path with no pre-sweep context, and it must take the
    continuous values from the **carried** coordinate --- an earlier accepted jump may have moved
    it. It used to crash; fixing only the crash would have read the stale pre-sweep coordinate.

    CONTROL: the same difference taken at the stale state, which must be wrong here."""
    from mimcs.samplers.discrete_updates import SweepEnv

    def lp(v):
        x = jnp.reshape(v["x"], ())
        return (jnp.log(W[v["z"]]) - 0.5 * ((x - MU[v["z"]]) / SIG[v["z"]]) ** 2
                + 1.3 * v["s"] * x)                                     # s reads the moved x

    m = Model([EuclideanParameter("x", ())], {"p": lp},
              discrete_parameters=[IntegerParameter("z", (), lower=0, upper=2),
                                   IntegerParameter("s", (), lower=0, upper=1)],
              jump_operators={"z": JumpOperator(
                  "z", ("x",), lambda values, c, v: (values["x"] + (MU[v] - MU[values["z"]]),))})
    s = NUTS_GIBBS(m, m.default_sample(), seed=0)
    s.initialize()
    st = s.state
    us = next(u for u in s.discrete_updaters if u.name == "s")
    env = SweepEnv(sampler=s, state=st, sweep_ctx=None, plans=s._restricted(force=True),
                   tables=st.discrete_proposal_params, u_prop=None, u_acc=None, n_lanes=1,
                   lane_dim=m.discrete_dim)
    x_moved = (st.coordinate + 2.5).reshape(1, -1)                 # as if a jump had moved it
    z = st.discrete.reshape(1, -1).at[:, 1].set(0)
    cur, prop = z[:, 1], jnp.asarray([1], jnp.int32)
    plan = env.plans["s"]
    h, idx = m.init_chart_hyperparams(), m.init_chart_indices()
    want = float(m.log_prob_at_coordinate(x_moved[0], h, idx, z[0].at[1].set(1))
                 - m.log_prob_at_coordinate(x_moved[0], h, idx, z[0]))
    st2, ctx = us._delta_env(env, x_moved)
    got = float(s._discrete_delta(st2, ctx, z, "s", 0, cur, prop, plan)[0])
    assert abs(got - want) < 1e-4, (got, want)
    stale = float(s._discrete_delta(st, s._sweep_context(st), z, "s", 0, cur, prop, plan)[0])
    assert abs(stale - want) > 1.0, (stale, want)                  # CONTROL


def test_an_existing_jump_model_keeps_the_full_path():
    """Gating: a model where every component reads the moved names has no plan, so its draws are
    unchanged from before this work (the jump tests pin them)."""
    m = _toy()
    s = NUTS_GIBBS(m, m.default_sample(), seed=0)
    assert next(u for u in s.discrete_updaters if u.name == "z").plan is None
