"""Tests for the library's logging.

Two things are checked: the plumbing (a configured ``mimcs`` logger, level control, one handler)
and the messages the rest of the library promises to emit --- a DSL compile error at ERROR
*before* the exception escapes, the L-BFGS iteration cap and a warmup that runs out of budget at
WARNING, post-warmup divergences at WARNING, the termination checks and the factory's choices at
INFO, and the factory's rejected alternatives at DEBUG.

Records are captured off the ``mimcs`` logger with a handler of our own rather than pytest's
``caplog``: the library turns ``propagate`` off (so it never double-prints into an application's
root logger), which is exactly what ``caplog`` relies on.
"""

import logging
import os

import jax.numpy as jnp
import numpy as np
import pytest

import mimcs
from mimcs._logging import (FMT_MAX_ELEMENTS, LEVEL_ENV_VAR, ROOT_LOGGER, STRICT_ENV_VAR,
                            _SAFE_FORMATTING,
                            SafeFormatting, _rescue, _resolve_level, configure_logging, fmt,
                            get_logger, log_level, set_log_level)
from mimcs.optim import minimize
from mimcs.testing import correlated_gaussian, nuts


class _Capture(logging.Handler):
    """Collects records emitted anywhere under ``mimcs``."""

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)

    def messages(self, level=None, name=None):
        return [r.getMessage() for r in self.records
                if (level is None or r.levelno == level)
                and (name is None or r.name.startswith(name))]

    def text(self, level=None, name=None):
        return "\n".join(self.messages(level, name))


@pytest.fixture
def captured():
    """Capture every ``mimcs`` record at DEBUG for the duration of a test."""
    logger = logging.getLogger(ROOT_LOGGER)
    handler = _Capture()
    previous = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)


# --- the plumbing ------------------------------------------------------------- #

def test_importing_mimcs_configures_the_library_logger():
    logger = logging.getLogger(ROOT_LOGGER)
    # ``MIMCS_LOG_LEVEL`` is *meant* to win here, and running the whole suite under
    # ``MIMCS_LOG_LEVEL=DEBUG`` is how the formatting hazards in `mimcs._logging` get found --- so
    # the assertion is on the resolved default, not on INFO unconditionally.
    expected = _resolve_level(os.environ.get(LEVEL_ENV_VAR) or None)
    assert logger.level == expected, "the library prints at INFO unless MIMCS_LOG_LEVEL says else"
    assert logger.propagate is False, "the library must not double-print into the root logger"
    assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)


def test_configure_logging_is_idempotent():
    # (a test runner may have handlers of its own on the logger; the count must not *grow*)
    logger = logging.getLogger(ROOT_LOGGER)
    before = len(logger.handlers)
    assert configure_logging() is logger
    assert configure_logging() is logger
    assert len(logger.handlers) == before


def test_set_log_level_accepts_names_and_numbers():
    logger = logging.getLogger(ROOT_LOGGER)
    original = logger.level
    try:
        set_log_level("WARNING")
        assert logger.level == logging.WARNING
        set_log_level(logging.DEBUG)
        assert logger.level == logging.DEBUG
        with pytest.raises(ValueError, match="unknown log level"):
            set_log_level("LOUD")
    finally:
        logger.setLevel(original)


def test_log_level_context_manager_restores_the_level():
    logger = logging.getLogger(ROOT_LOGGER)
    original = logger.level
    with log_level("DEBUG"):
        assert logger.level == logging.DEBUG
    assert logger.level == original


def test_module_loggers_are_named_after_their_module(captured):
    mimcs.compile_model("parameters { real mu; } model { mu ~ normal(0, 1); }")
    assert any(r.name.startswith("mimcs.dsl") for r in captured.records)


# --- ERROR: DSL compile errors ------------------------------------------------ #

def test_a_compile_error_is_logged_before_it_is_raised(captured):
    with pytest.raises(mimcs.DslError):
        mimcs.compile_model("parameters { real mu } model { }")
    errors = captured.messages(logging.ERROR)
    assert len(errors) == 1, "the error must be reported exactly once, not per compiler stage"
    assert "1:22" in errors[0] and "expected ';'" in errors[0]


def test_a_missing_data_error_is_logged_with_its_source_line(captured):
    source = "data { int n; }\nparameters { real mu; }\nmodel { mu ~ normal(0, n); }"
    with pytest.raises(mimcs.DslError):
        mimcs.compile_model(source, data={})
    errors = captured.messages(logging.ERROR)
    assert len(errors) == 1
    assert "missing data 'n'" in errors[0]
    assert "data { int n; }" in errors[0], "the caret line should show the offending source"


def test_a_successful_compile_logs_no_error(captured):
    mimcs.compile_model("parameters { real mu; } model { mu ~ normal(0, 1); }")
    assert captured.messages(logging.ERROR) == []


# --- WARNING: a silently ignored DSL block ------------------------------------ #

def test_generated_quantities_warns_that_it_is_ignored(captured):
    """`generated quantities` parses, so the program compiles and samples perfectly well while
    producing none of the quantities it asks for. That was a DEBUG note --- invisible by default,
    so someone porting a Stan program would just find their quantities missing."""
    src = """
    data { int<lower=1> N; }
    parameters { real mu; }
    model { mu ~ normal(0, 1); }
    generated quantities { real mu2 = mu * mu; }
    """
    model = mimcs.compile_model(src, {"N": 3})
    assert list(model.log_prob_fns) == ["target"]        # it still compiles ...
    warnings = captured.text(logging.WARNING, "mimcs.dsl")
    assert "generated quantities" in warnings and "NOT implemented" in warnings
    assert captured.messages(logging.WARNING, "mimcs.dsl") != []


def test_no_generated_quantities_block_warns_about_nothing(captured):
    """The control: the warning must not fire for a program without the block."""
    mimcs.compile_model("""
    data { int<lower=1> N; }
    parameters { real mu; }
    model { mu ~ normal(0, 1); }
    """, {"N": 3})
    assert captured.messages(logging.WARNING, "mimcs.dsl") == []


# --- WARNING / DEBUG: the L-BFGS optimizer ------------------------------------ #

def test_lbfgs_warns_when_it_runs_out_of_iterations(captured):
    import jax.numpy as jnp
    # Rosenbrock from a cold start: nowhere near converged in two iterations.
    fun = lambda x: (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
    res = minimize(fun, jnp.array([-3.0, 5.0]), max_iter=2)
    assert not bool(res.converged)
    warnings = captured.text(logging.WARNING, "mimcs.optim")
    assert "max_iter=2" in warnings and "without converging" in warnings


def test_lbfgs_logs_its_termination_at_debug(captured):
    import jax.numpy as jnp
    res = minimize(lambda x: jnp.sum((x - 2.0) ** 2), jnp.zeros(3))
    assert bool(res.converged)
    debug = captured.text(logging.DEBUG, "mimcs.optim")
    assert "terminated after" in debug and "converged=True" in debug
    assert captured.messages(logging.WARNING, "mimcs.optim") == []


# --- INFO / WARNING: warmup termination --------------------------------------- #

def test_every_termination_check_is_logged_at_info(captured):
    sampler = nuts(terminate="rhat", min_warmup=100, check_every=50,
                   max_warmup=300)(correlated_gaussian().model, 0)
    sampler.warmup()
    checks = [m for m in captured.messages(logging.INFO) if "check at warmup iteration" in m]
    assert len(checks) == len(sampler.warmup_mixing_stats()) >= 2
    assert "GelmanRubinTermination" in checks[0], "the mixin, not the composed sampler class"
    assert "max split-R-hat" in checks[0]


def test_running_out_of_warmup_budget_warns(captured):
    """``max_warmup`` reached without the criterion firing: the run continues, so it must warn."""
    sampler = nuts(terminate="classifier", min_warmup=20, check_every=10,
                   max_warmup=40, accuracy_threshold=0.0)(correlated_gaussian().model, 0)
    sampler.warmup()                                  # threshold 0 => can never look "mixed"
    assert not sampler.warmup_terminated_early()
    warnings = captured.text(logging.WARNING, "mimcs.adaptation.termination")
    assert "maximum of 40 iteration(s)" in warnings


def test_stopping_on_the_criterion_does_not_warn(captured):
    sampler = nuts(terminate="rhat", max_warmup=8000)(correlated_gaussian().model, 0)
    sampler.warmup()
    assert sampler.warmup_terminated_early()
    assert captured.messages(logging.WARNING, "mimcs.adaptation.termination") == []
    assert "criterion met" in captured.text(logging.INFO, "mimcs.adaptation.termination")


# --- WARNING: NUTS divergences ------------------------------------------------ #

def _fake_sampling_diagnostics(sampler, diverging):
    """Pretend the sampler just drew these transitions in the SAMPLING phase."""
    sampler._diag["diverging"] = [np.asarray(bool(d)) for d in diverging]
    sampler._diag_phase = [True] * len(diverging)


def test_nuts_warns_about_post_warmup_divergences(captured):
    sampler = nuts()(correlated_gaussian().model, 0)
    _fake_sampling_diagnostics(sampler, [False] * 97 + [True] * 3)
    sampler._sample_end_hooks(sampler.state)
    warnings = captured.text(logging.WARNING, "mimcs.hmc.nuts")
    assert "3 of 100 post-warmup transition(s) diverged" in warnings


def test_nuts_says_nothing_when_no_transition_diverged(captured):
    sampler = nuts()(correlated_gaussian().model, 0)
    _fake_sampling_diagnostics(sampler, [False] * 50)
    sampler._sample_end_hooks(sampler.state)
    assert captured.messages(logging.WARNING, "mimcs.hmc.nuts") == []
    assert "no post-warmup divergences" in captured.text(logging.DEBUG, "mimcs.hmc.nuts")


def test_a_clean_short_run_warns_about_nothing(captured):
    """The common path must stay quiet: an easy target, no warnings anywhere."""
    sampler = nuts(terminate="rhat", max_warmup=4000)(correlated_gaussian().model, 0)
    sampler.warmup()
    sampler.sample(200)
    assert captured.messages(logging.WARNING) == []


# --- INFO / DEBUG: the sampler factory ---------------------------------------- #

def test_the_factory_logs_its_choices_at_info(captured):
    model = correlated_gaussian().model
    spec = mimcs.analyze(model)
    chosen = [m for m in captured.messages(logging.INFO, "mimcs.factory") if m.startswith("chose")]
    assert chosen, "every adopted proposal is logged"
    assert any("blocks =" in m for m in chosen)
    # the same content the spec carries, without having to keep the spec
    assert any("block_partition" in m for m in chosen) and spec.rationale


def test_the_factory_logs_rejected_choices_at_debug(captured):
    """Two rules want the same slot: the winner at INFO, the one it beat at DEBUG."""
    from mimcs.factory.rules import Proposal, arbitrate
    from mimcs.factory.spec import default_spec

    spec = default_spec(correlated_gaussian().model)
    arbitrate(spec, [Proposal("base", "hmc", 0.3, "cheaper per step", "cheap_rule"),
                     Proposal("base", "nuts", 0.9, "no trajectory length to tune", "nuts_rule")])
    assert spec.base == "nuts"
    debug = captured.text(logging.DEBUG, "mimcs.factory.rules")
    assert "rejected base = 'hmc'" in debug and "beaten by nuts_rule" in debug
    assert "chose base = 'nuts'" in captured.text(logging.INFO, "mimcs.factory.rules")


def test_a_rule_that_cannot_fire_says_so_at_debug(captured):
    """Evidence-free analysis: the evidence-based rules are inert, and that is on the record."""
    mimcs.analyze(correlated_gaussian().model)
    debug = captured.text(logging.DEBUG, "mimcs.factory.rules")
    assert "learned_metric rule inert" in debug
    assert "mass_mode rule inert" in debug
    assert "rule block_partition_rule emitted 1 proposal(s)" in debug


def test_building_a_sampler_reports_the_composition(captured):
    mimcs.make_sampler(correlated_gaussian().model)
    info = captured.text(logging.INFO, "mimcs.factory.build")
    assert "building nuts" in info and "adaptations" in info


# --- formatting: a log call must never raise ---------------------------------- #
#
# The hazard is real, not defensive padding. A DEBUG line in `mimcs/hmc/samplers.py` formatted the
# initial step size with `%.4g`; parallel tempering with a per-rung acceptance signal makes that a
# length-K array, `%` calls `float()` on it, and `TypeError` came out of the *handler*. The
# standard library swallows that, pytest's capture handler re-raises it, and so four builds died
# at DEBUG while passing at the default level. Both halves of the fix are pinned here.

def test_fmt_formats_a_scalar_exactly_as_percent_would():
    assert fmt(0.5) == "0.5" == "%.4g" % 0.5
    assert fmt(1 / 3, ".6g") == "%.6g" % (1 / 3)
    assert fmt(jnp.float32(2.5)) == "2.5"
    assert fmt(jnp.asarray(2.5)) == "2.5", "a 0-d array is a scalar"
    assert fmt(None) == "n/a"


def test_fmt_formats_a_vector_elementwise():
    assert fmt(jnp.asarray([0.5, 0.25, 0.125])) == "[0.5, 0.25, 0.125]"
    assert fmt(np.asarray([1.5, 2.5]), ".1f") == "[1.5, 2.5]"
    assert fmt(np.arange(6.0).reshape(2, 3), ".0f").startswith("(2, 3) ["), "shape when ndim > 1"


def test_fmt_elides_a_long_vector_but_says_how_long():
    out = fmt(np.arange(100.0))
    assert out.count(",") == FMT_MAX_ELEMENTS, "one separator per shown element, then the elision"
    assert "(100 total)" in out


def test_fmt_never_raises_on_anything_it_is_handed():
    for x in (object(), "a string", ["a", "b"], np.array(["a", "b"]), {"k": 1}, [1, 2, 3]):
        assert isinstance(fmt(x), str)


def test_the_step_size_line_survives_a_per_temperature_vector(captured):
    """The exact regression: build a PT sampler whose step size is one value per rung."""
    from mimcs.hmc import HMC
    from mimcs.pt import parallel_tempering

    model = correlated_gaussian(mean=[1.0, -2.0], cov=[[2.0, 1.4], [1.4, 1.5]]).model
    with log_level("DEBUG"):
        s = parallel_tempering(model, n_temperatures=3, base=HMC, step_size=0.5,
                               per_temperature_step_size=True, seed=0)
    assert np.ndim(s.state.step_size) == 1, "the premise: the step size really is a vector here"
    line = [m for m in captured.messages(logging.DEBUG) if m.startswith("HMC components:")]
    assert len(line) == 1
    assert "initial step size [0.5, 0.5, 0.5]" in line[0], line[0]


def test_a_bad_format_is_rescued_rather_than_raised(captured, monkeypatch):
    """The net, exercised on a line no site actually writes: a numeric spec on a vector."""
    monkeypatch.delenv(STRICT_ENV_VAR, raising=False)
    log = get_logger("mimcs.testing._fmt_probe")
    with log_level("DEBUG"):
        log.debug("step %.4g over %d rung(s)", jnp.asarray([0.5, 0.25]), 2)
    assert captured.messages(logging.DEBUG) == ["step [0.5, 0.25] over 2 rung(s)"]


def test_the_net_is_load_bearing(monkeypatch):
    """Control for the test above: without the rescue, that same call raises.

    Strict mode is how the suite stays an audit --- a net that silently absorbs the defect would
    make `MIMCS_LOG_LEVEL=DEBUG` pass on the very sites it is run to find."""
    monkeypatch.setenv(STRICT_ENV_VAR, "1")
    log = get_logger("mimcs.testing._fmt_probe")
    with log_level("DEBUG"):
        with pytest.raises(TypeError):
            log.debug("step %.4g", jnp.asarray([0.5, 0.25]))
        log.debug("step %s", jnp.asarray([0.5, 0.25]))     # a good call is untouched by strictness


def test_strict_mode_can_be_set_on_the_filter_itself(monkeypatch):
    monkeypatch.delenv(STRICT_ENV_VAR, raising=False)
    assert SafeFormatting()._strict() is False
    assert SafeFormatting(strict=True)._strict() is True
    monkeypatch.setenv(STRICT_ENV_VAR, "yes")
    assert SafeFormatting()._strict() is True
    assert SafeFormatting(strict=False)._strict() is False, "an explicit setting beats the env"


def test_the_net_is_not_reached_for_a_message_that_formats(captured):
    """Control: the rescue must not touch a well-formed record, or the test above proves nothing.

    `%.4g` of 0.5 is "0.5"; `fmt`'s vector rendering would have bracketed it."""
    log = get_logger("mimcs.testing._fmt_probe")
    with log_level("DEBUG"):
        log.debug("step %.4g over %d rung(s)", 0.5, 2)
    assert captured.messages(logging.DEBUG) == ["step 0.5 over 2 rung(s)"]


@pytest.mark.parametrize("msg, args, expected", [
    ("a %s b %d c %%", ("x", 2), "a x b 2 c %"),                      # literal %% is not an arg
    ("%d only", (np.arange(3),), "[0, 1, 2] only"),                   # integer spec on a vector
    ("%s", (np.arange(2), "spare"), "[0 1] | unused args ('spare',)"),  # too many arguments
    ("%s and %s", ("one",), "one and %s"),                           # too few arguments
])
def test_the_rescue_keeps_the_record_readable(msg, args, expected):
    assert _rescue(msg, args) == expected


def test_a_mapping_message_is_kept_whole():
    """`%(name)s` args are a dict, which the by-position walk cannot handle; it must not mangle."""
    out = _rescue("%(x)s and %(y).4g", {"x": 1, "y": np.arange(2)})
    assert out.startswith("%(x)s and %(y).4g | args ")


def test_every_library_logger_carries_the_net():
    for name in ("mimcs", "mimcs.hmc.samplers", "mimcs.factory.rules"):
        assert any(isinstance(f, SafeFormatting) for f in get_logger(name).filters), name
    assert get_logger("mimcs").filters.count(_SAFE_FORMATTING) == 1, "added once, not per call"
