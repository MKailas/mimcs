"""Library-wide logging: one standard-library logger per module, under the ``mimcs`` root.

Every module gets its own logger with ``log = get_logger(__name__)``, so the name of a record
(``mimcs.factory.rules``, ``mimcs.adaptation.termination``, ...) says where it came from and any
subtree can be silenced or turned up on its own::

    import logging
    logging.getLogger("mimcs.adaptation").setLevel(logging.DEBUG)

Importing :mod:`mimcs` calls :func:`configure_logging`, which attaches one stream handler to the
``mimcs`` logger at **INFO** --- so the library prints its INFO-level messages out of the box,
without the application having to configure logging. ``MIMCS_LOG_LEVEL`` overrides the default
level (``MIMCS_LOG_LEVEL=DEBUG python run.py``), and :func:`set_log_level` / :func:`log_level`
change it at runtime. The handler is attached to ``mimcs`` rather than the root logger, and
``propagate`` is turned off, so the library never touches an application's own logging setup and
never double-prints into it.

**What goes at which level** (the convention this codebase follows):

* ``ERROR`` --- an operation is failing and an exception is about to be raised (DSL compile errors).
* ``WARNING`` --- the run continues, but a result is suspect: an optimiser hit its iteration cap,
  warmup ran out of budget without its criterion firing, sampling produced divergences.
* ``INFO`` --- the few decisions and outcomes a user wants without asking: what the factory chose,
  each warmup-termination check, how a run ended.
* ``DEBUG`` --- the detail for debugging: rejected alternatives, per-candidate fits, shapes and
  dimensions, frozen adaptation state. Assume nobody reads it unless something is wrong.

Log calls use ``%``-style lazy formatting (``log.debug("n=%d", n)``), so a DEBUG message costs
almost nothing while DEBUG is off --- which matters for the calls that sit near the sampling loop.
"""

from __future__ import annotations

import contextlib
import logging
import os

#: The root logger name of the library; every module logger is a child of it.
ROOT_LOGGER = "mimcs"
#: Level used when neither an argument nor ``MIMCS_LOG_LEVEL`` says otherwise.
DEFAULT_LEVEL = logging.INFO
#: Environment variable read by :func:`configure_logging` (a level name or number).
LEVEL_ENV_VAR = "MIMCS_LOG_LEVEL"
DEFAULT_FORMAT = "%(levelname)s %(name)s: %(message)s"


def get_logger(name: str | None = None) -> logging.Logger:
    """The logger for a module: ``log = get_logger(__name__)``.

    ``name`` is a dotted module name (already under ``mimcs``); omitting it gives the library
    root logger.
    """
    return logging.getLogger(name if name else ROOT_LOGGER)


def _resolve_level(level) -> int:
    """A level from an int, a level name (``"DEBUG"``), ``MIMCS_LOG_LEVEL``, or the default."""
    if level is None:
        level = os.environ.get(LEVEL_ENV_VAR) or DEFAULT_LEVEL
    if isinstance(level, str):
        named = logging.getLevelName(level.strip().upper())
        if not isinstance(named, int):
            raise ValueError(f"unknown log level {level!r}")
        return named
    return int(level)


def configure_logging(level=None, *, stream=None, fmt: str = DEFAULT_FORMAT,
                      force: bool = False) -> logging.Logger:
    """Attach a stream handler to the ``mimcs`` logger and set its level (INFO by default).

    Called once on ``import mimcs``. Idempotent: a second call re-uses the existing handler and
    only updates the level, unless ``force`` replaces the handlers outright.

    Args:
        level: a level number, a level name, or ``None`` for ``MIMCS_LOG_LEVEL`` / INFO.
        stream: handler stream (default ``sys.stderr``).
        fmt: format string for the handler's formatter.
        force: drop any handlers already on the ``mimcs`` logger and install a fresh one.

    Returns:
        The configured ``mimcs`` logger.
    """
    logger = logging.getLogger(ROOT_LOGGER)
    if force:
        for h in list(logger.handlers):
            logger.removeHandler(h)
    if not logger.handlers:
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    logger.setLevel(_resolve_level(level))
    logger.propagate = False        # ours to print; never duplicate into an application's root
    return logger


def set_log_level(level) -> None:
    """Set the level of the ``mimcs`` logger (``set_log_level("DEBUG")``)."""
    logging.getLogger(ROOT_LOGGER).setLevel(_resolve_level(level))


@contextlib.contextmanager
def log_level(level):
    """Context manager: run a block with the library at ``level``, then restore.

        with log_level("DEBUG"):
            sampler.warmup(1000)
    """
    logger = logging.getLogger(ROOT_LOGGER)
    previous = logger.level
    logger.setLevel(_resolve_level(level))
    try:
        yield logger
    finally:
        logger.setLevel(previous)


__all__ = ["ROOT_LOGGER", "DEFAULT_LEVEL", "LEVEL_ENV_VAR", "get_logger",
           "configure_logging", "set_log_level", "log_level"]
