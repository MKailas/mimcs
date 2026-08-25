"""Shared pytest fixtures: a per-test artifacts directory for plot output."""

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
