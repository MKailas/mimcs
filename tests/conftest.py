"""Shared pytest fixtures: a per-test artifacts directory, and the chunk-budget setter."""

import contextlib
import pathlib

import pytest

# `tests/experiments/` and `tests/problems/` are untracked research scratch (see CLAUDE.md), not
# part of the suite. Keep pytest from collecting them: a scratch file named `*_test.py` would
# otherwise be imported, and some enable x64 at import (jax.config.update) -- which leaks into the
# whole run and shifts float32-margin statistical tests.
collect_ignore_glob = ["experiments/*", "problems/*"]

ARTIFACTS = pathlib.Path(__file__).parent / "artifacts"


@pytest.fixture
def artifacts_dir():
    ARTIFACTS.mkdir(exist_ok=True)
    return ARTIFACTS


@pytest.fixture
def chunk_budget():
    """Set the row-chunking budget inside one test, restoring the previous setting afterwards.

    Call it as ``chunk_budget(64)`` (or ``chunk_budget(None)`` for never-chunk). It replaces the
    older ``monkeypatch.setattr("mimcs._chunked.CHUNK_BYTES", ...)``: the budget now lives in
    :mod:`mimcs.config`, which is the seam a *user* has, so a test that goes through it is
    exercising the same path they would.

    The property that makes any of this work is unchanged --- ``mimcs._chunked`` resolves the
    budget at call time, so a test can still lower it and force the chunked path on a small array.
    A value bound at definition time would leave every "chunked" arm silently unchunked, comparing
    the whole-array path against itself.
    """
    from mimcs import config
    with contextlib.ExitStack() as stack:
        yield lambda value: stack.enter_context(config.options(chunk_bytes=value))
