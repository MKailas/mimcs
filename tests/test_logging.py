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

import numpy as np
import pytest

import mimcs
from mimcs._logging import ROOT_LOGGER, configure_logging, log_level, set_log_level
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
    assert logger.level == logging.INFO, "INFO is the level the library prints at by default"
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
