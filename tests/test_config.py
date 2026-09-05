"""Tests for :mod:`mimcs.config` --- the settings that depend on the machine, not the model.

Two things are checked: the resolver (explicit setting > environment variable > default, and the
byte-count grammar), and that the settings actually *reach* the code they claim to govern. The
second half is the one that matters. A configuration module is easy to test vacuously --- set a
value, read it back, watch it agree with itself --- so every override test here also asserts a
control: that the library's own behaviour changed, or that it had not already changed before the
override was applied.

That habit is not general caution. Two tests in this suite were silently vacuous for exactly this
reason: they lowered a chunk budget that was bound as a default argument, so both arms ran the
unchunked path and the comparison proved nothing.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import mimcs
from mimcs import config
from mimcs.testing import correlated_gaussian
from mimcs._chunked import map_rows, rows_per_chunk, sum_rows


@pytest.fixture(autouse=True)
def _clean_settings():
    """No test may leak a setting into the next one; the state is process-wide by design."""
    config.reset()
    yield
    config.reset()


# --- the byte-count grammar --------------------------------------------------- #

@pytest.mark.parametrize("text, expected", [
    ("8388608", 8 * 1024 ** 2),
    ("8M", 8 * 1024 ** 2),
    ("8MiB", 8 * 1024 ** 2),
    ("64 mib", 64 * 1024 ** 2),
    ("1K", 1024),
    ("2G", 2 * 1024 ** 3),
    ("none", None),
    ("OFF", None),
])
def test_parse_bytes_reads_the_shapes_a_shell_would_write(text, expected):
    assert config.parse_bytes(text) == expected


def test_suffixes_are_base_two_throughout():
    """`M` and `MiB` mean the same thing --- 1048576, not 1000000.

    Every use of this number is a working set against a memory budget, where the hardware deals in
    powers of two. A decimal `MB` would be a second meaning for the same letter, so it is rejected
    rather than quietly rounded."""
    assert config.parse_bytes("1M") == config.parse_bytes("1MiB") == 1048576
    with pytest.raises(ValueError, match="cannot read"):
        config.parse_bytes("1MB")


@pytest.mark.parametrize("bad", ["", "lots", "8 gallons", "-1", "1.5M"])
def test_an_unreadable_byte_count_raises_rather_than_defaulting(bad):
    with pytest.raises(ValueError):
        config.parse_bytes(bad)


def test_a_zero_or_negative_budget_is_refused():
    """It would ask for chunks of no rows. `None` is how you turn chunking off."""
    for bad in (0, -1):
        with pytest.raises(ValueError, match="must be positive"):
            config.parse_bytes(bad)


# --- resolution: explicit > environment > default ----------------------------- #

def test_the_default_is_used_when_nothing_says_otherwise(monkeypatch):
    monkeypatch.delenv(config.CHUNK_BYTES_ENV_VAR, raising=False)
    assert config.chunk_bytes() == config.DEFAULT_CHUNK_BYTES


def test_the_environment_beats_the_default(monkeypatch):
    monkeypatch.setenv(config.CHUNK_BYTES_ENV_VAR, "64MiB")
    assert config.chunk_bytes() == 64 * 1024 ** 2
    assert config.chunk_bytes() != config.DEFAULT_CHUNK_BYTES, "control: it really moved"


def test_an_explicit_setting_beats_the_environment(monkeypatch):
    monkeypatch.setenv(config.CHUNK_BYTES_ENV_VAR, "64MiB")
    config.set_chunk_bytes("1M")
    assert config.chunk_bytes() == 1024 ** 2


def test_the_environment_is_read_at_call_time_not_at_import(monkeypatch):
    """The property the whole module rests on. A value read once at import could not be overridden
    afterwards, and a test that tried would compare a code path against itself."""
    monkeypatch.delenv(config.CHUNK_BYTES_ENV_VAR, raising=False)
    assert config.chunk_bytes() == config.DEFAULT_CHUNK_BYTES
    monkeypatch.setenv(config.CHUNK_BYTES_ENV_VAR, "1M")
    assert config.chunk_bytes() == 1024 ** 2, "the same call must see the new environment"


def test_none_is_a_setting_and_not_an_absence(monkeypatch):
    """`None` means *never chunk*. Storing it must not read as "nothing has been set", or the
    default would come back and quietly re-enable chunking."""
    monkeypatch.setenv(config.CHUNK_BYTES_ENV_VAR, "64MiB")
    config.set_chunk_bytes(None)
    assert config.chunk_bytes() is None
    config.reset()
    assert config.chunk_bytes() == 64 * 1024 ** 2, "reset really does forget it"


# --- the scoped override ------------------------------------------------------ #

def test_options_restores_the_previous_setting():
    config.set_chunk_bytes("4M")
    with config.options(chunk_bytes=64):
        assert config.chunk_bytes() == 64
    assert config.chunk_bytes() == 4 * 1024 ** 2


def test_options_restores_on_the_way_out_of_an_exception():
    """What makes it safe in a test: a failing assertion inside the block must not leak."""
    with pytest.raises(RuntimeError):
        with config.options(chunk_bytes=64):
            raise RuntimeError("boom")
    assert config.chunk_bytes() == config.DEFAULT_CHUNK_BYTES


def test_options_restores_unset_as_unset(monkeypatch):
    """Leaving the block must restore "nothing set", not the resolved value it happened to have ---
    otherwise the first `options()` call would freeze the environment in place."""
    with config.options(chunk_bytes=64):
        pass
    monkeypatch.setenv(config.CHUNK_BYTES_ENV_VAR, "1M")
    assert config.chunk_bytes() == 1024 ** 2


def test_an_unknown_option_is_an_error_rather_than_a_silent_typo():
    with pytest.raises(TypeError, match="unknown option"):
        with config.options(chnuk_bytes=64):
            pass


# --- the settings reach the code they govern ---------------------------------- #

def test_the_budget_reaches_rows_per_chunk():
    before = rows_per_chunk(64)
    with config.options(chunk_bytes=64):
        assert rows_per_chunk(64) == 1
    assert before > 10_000, "control: nothing this small chunks at the default budget"


def test_the_budget_reaches_map_rows():
    """`map_rows` is bit-identical at every budget, so the control cannot be the *values* --- it is
    that the chunked path was really taken."""
    x = np.arange(600.0).reshape(300, 2)
    whole = map_rows(lambda r: r * 2.0, x)
    with config.options(chunk_bytes=32):
        assert rows_per_chunk(2 * x.dtype.itemsize) < 300, "control: it really chunks"
        chunked = map_rows(lambda r: r * 2.0, x)
    assert np.array_equal(whole, chunked)
    assert chunked.flags.owndata


def test_never_chunk_takes_the_whole_array_path():
    """`chunk_bytes=None` is the accelerator escape hatch: `map_rows` copies each chunk back with
    `np.asarray`, which is a device-to-host transfer on a GPU, so being able to turn chunking off
    outright matters more there than the budget's exact value does."""
    x = np.arange(600.0).reshape(300, 2)
    with config.options(chunk_bytes=None):
        assert rows_per_chunk(1_000_000_000) > 300, "one pass, however wide the row"
        out = map_rows(lambda r: r * 2.0, x)
    assert np.array_equal(out, x * 2.0)
    assert out.flags.owndata, "the whole-array path must still own its data"


def test_never_chunk_reaches_sum_rows():
    x = jnp.arange(300.0).reshape(150, 2)
    whole = float(sum_rows(lambda r: jnp.sum(r ** 2), x, budget=None))
    with config.options(chunk_bytes=None):
        assert float(sum_rows(lambda r: jnp.sum(r ** 2), x)) == whole
    with config.options(chunk_bytes=32):
        assert rows_per_chunk(2 * 8) < 150, "control: this budget really chunks"
        chunked = float(sum_rows(lambda r: jnp.sum(r ** 2), x))
    assert chunked == pytest.approx(whole, rel=1e-6)


def test_the_budget_reaches_the_summary_quantile_blocks():
    from mimcs.summary import QUANTILES, _quantiles_by_column_block
    a = np.arange(400.0).reshape(20, 20)
    whole = _quantiles_by_column_block(a, QUANTILES, budget=2 ** 30)
    with config.options(chunk_bytes=64):
        blocked = _quantiles_by_column_block(a, QUANTILES)
    with config.options(chunk_bytes=None):
        unblocked = _quantiles_by_column_block(a, QUANTILES)
    assert np.array_equal(whole, blocked), "column blocks are order statistics: bit-identical"
    assert np.array_equal(whole, unblocked)


# --- x64 ---------------------------------------------------------------------- #

def test_x64_enabled_reports_jax_and_not_a_setting_of_our_own():
    """It is process-global JAX state, so the honest answer is whatever JAX says right now."""
    assert config.x64_enabled() is bool(jax.config.jax_enable_x64)


def test_enable_x64_switches_what_float_canonicalizes_to():
    """Turned back off in the same test: leaving it on would shift every float32-margin
    statistical test in the rest of the suite --- the hazard `tests/conftest.py` documents."""
    was = config.x64_enabled()
    try:
        config.enable_x64(True)
        assert config.x64_enabled()
        assert jnp.zeros(3, float).dtype == jnp.float64
        config.enable_x64(False)
        assert jnp.zeros(3, float).dtype == jnp.float32, "control: the switch goes both ways"
    finally:
        config.enable_x64(was)
    assert config.x64_enabled() is was


def test_switching_x64_after_import_works_at_all():
    """The reason `enable_x64` can exist as a function and not only as an environment variable.

    Nothing in this library captures a dtype at import: the precision reads are all
    `jnp.result_type(float)` *inside* functions, and the module-level `jax.jit`s key their caches
    on avals. So a caller who flips this immediately after `import mimcs` gets a consistent
    library --- which is what the assertion below is really pinning."""
    was = config.x64_enabled()
    try:
        config.enable_x64(True)
        model = correlated_gaussian().model
        assert jax.tree.leaves(model.default_sample())[0].dtype == jnp.float64
    finally:
        config.enable_x64(was)


def test_the_x64_env_var_is_named_and_read_at_import(monkeypatch):
    """`_configure_from_env` is what `mimcs/__init__` calls before importing any subpackage --- the
    only ordering-proof route, since an environment variable cannot be set too late."""
    assert config.X64_ENV_VAR == "MIMCS_ENABLE_X64"
    was = config.x64_enabled()
    try:
        monkeypatch.setenv(config.X64_ENV_VAR, "1")
        config.enable_x64(False)                       # control: it is off before we start
        config._configure_from_env()
        assert config.x64_enabled()
    finally:
        config.enable_x64(was)


@pytest.mark.parametrize("text, expected", [("1", True), ("true", True), ("YES", True),
                                            ("on", True), ("0", False), ("", False),
                                            ("maybe", False)])
def test_the_env_var_truthiness_matches_the_logging_module(text, expected):
    """One spelling of "on" across the library; `_logging.SafeFormatting` uses the same set."""
    assert config._truthy(text) is expected
    assert config.TRUE_STRINGS == ("1", "true", "yes", "on")


# --- reachability from the top level ------------------------------------------ #

def test_config_is_reachable_from_the_package_namespace():
    assert mimcs.config is config
    assert "config" in mimcs.__all__


def test_the_loss_gate_is_deliberately_not_configurable():
    """`CHUNK_LOSS_BYTES` decides *whether* a path that reorders accumulation is taken, and sits
    above the largest fit the suite performs so no seed-pinned expectation can move underneath it.
    Making it settable would be a correctness change dressed as a tuning change."""
    assert not hasattr(config, "chunk_loss_bytes")
    assert "chunk_loss_bytes" not in config._settings
    assert os.environ.get("MIMCS_CHUNK_LOSS_BYTES") is None
