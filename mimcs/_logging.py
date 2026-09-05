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

**A log call must never be able to fail a run.** Lazy formatting defers ``msg % args`` to whichever
handler emits the record, so a mistyped conversion raises *there*, far from the call site --- and
in a numerical library the usual mistake is a quantity that is a scalar in one configuration and a
vector in another. The real case: ``"initial step size %.4g"`` on a step size that parallel
tempering makes **one per rung**, where ``%.4g`` calls ``float()`` on a length-K array and raises
``TypeError``. The standard library swallows that (``Handler.handleError`` prints
``--- Logging error ---`` to stderr and moves on), but pytest's capture handler re-raises it, so it
crashed a build --- and the four tests it crashed pass at the default level, which is how it went
unseen. Two things guard against it now:

* :func:`fmt` formats a scalar **or a vector** with one spec, and never raises. Use it (with
  ``%s``) wherever a logged quantity might not be a scalar.
* :class:`SafeFormatting`, attached to every logger :func:`get_logger` hands out, rescues a record
  whose arguments do not fit its format string: the message is re-rendered through :func:`fmt`,
  one conversion at a time, so the record still says what it was going to say. It is a net, not a
  licence --- a rescued record is a defect at its call site. Because the net would otherwise hide
  those, ``MIMCS_STRICT_LOGGING=1`` makes it re-raise instead, and

      MIMCS_LOG_LEVEL=DEBUG MIMCS_STRICT_LOGGING=1 python -m pytest tests/ -q

  is the audit that walks every DEBUG line the library has.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re

import numpy as np

#: The root logger name of the library; every module logger is a child of it.
ROOT_LOGGER = "mimcs"
#: Level used when neither an argument nor ``MIMCS_LOG_LEVEL`` says otherwise.
DEFAULT_LEVEL = logging.INFO
#: Environment variable read by :func:`configure_logging` (a level name or number).
LEVEL_ENV_VAR = "MIMCS_LOG_LEVEL"
#: Environment variable that makes :class:`SafeFormatting` re-raise instead of rescuing. Set it
#: (with ``MIMCS_LOG_LEVEL=DEBUG``) to audit the library's own log calls; see that class.
STRICT_ENV_VAR = "MIMCS_STRICT_LOGGING"
DEFAULT_FORMAT = "%(levelname)s %(name)s: %(message)s"


#: How many elements of a vector :func:`fmt` prints before eliding the rest.
FMT_MAX_ELEMENTS = 8

#: One ``%``-conversion of a printf-style format string, e.g. ``%s``, ``%.4g``, ``%(name)-5d``.
#: Used only by :class:`SafeFormatting` to walk a message it could not format in one go.
_CONVERSION = re.compile(
    r"%(?:\((?P<key>[^)]*)\))?(?P<flags>[-+ #0]*)(?P<width>\*|\d+)?"
    r"(?P<precision>\.(?:\*|\d+))?[hlL]?(?P<type>[diouxXeEfFgGcrsa%])")


def _format_one(v, spec: str) -> str:
    """One element under one format spec, falling back rather than raising.

    Two attempts, because the element's own type may not accept the spec even when its value does:
    a ``numpy.float64`` refuses ``"d"``, and a 0-d array refuses every numeric spec.
    """
    try:
        return format(v, spec)
    except (TypeError, ValueError):
        pass
    try:
        return format(float(v), spec)
    except (TypeError, ValueError):
        return str(v)


def fmt(x, spec: str = ".4g") -> str:
    """Format a logged quantity --- a scalar **or a vector** --- under one spec. Never raises.

    ``None`` is ``"n/a"``; a scalar (Python number, numpy scalar, 0-d array) formats as
    ``format(float(x), spec)`` does; an array of one or more dimensions formats element by element
    inside brackets, elided after :data:`FMT_MAX_ELEMENTS` with the total count; anything else
    falls back to ``str``.

    The vector case is not hypothetical padding: parallel tempering turns several per-chain
    quantities into one value per rung (the step size when acceptance is per-temperature, the log
    density on the product state), so the same log line is scalar on an ordinary chain and a
    length-K vector under PT. Pair it with ``%s``, never with a numeric conversion --- ``%.4g`` on
    the *result* of this function would raise for the same reason the vector did.
    """
    if x is None:
        return "n/a"
    try:
        return format(float(x), spec)            # Python numbers, numpy scalars, 0-d arrays
    except (TypeError, ValueError):
        pass
    try:
        a = np.asarray(x)
    except Exception:                            # not array-like at all (a list of names, say)
        return str(x)
    if a.ndim == 0 or a.dtype.kind not in "fiub":
        return str(x)
    flat = a.reshape(-1)
    shown = ", ".join(_format_one(v, spec) for v in flat[:FMT_MAX_ELEMENTS])
    elided = "" if flat.size <= FMT_MAX_ELEMENTS else f", ... ({flat.size} total)"
    shape = "" if a.ndim == 1 else f"{a.shape} "
    return f"{shape}[{shown}{elided}]"


def _rescue(msg, args) -> str:
    """Render ``msg % args`` conversion by conversion, using :func:`fmt` where ``%`` fails.

    Only reached once the ordinary formatting has already raised, so this trades the record's
    exact layout for the guarantee that it still carries its information. A ``%(name)s`` mapping
    message is kept whole with its arguments appended --- the by-position walk below cannot read
    it, and mangling is worse than not trying.
    """
    text = str(msg)
    if isinstance(args, dict) or "%(" in text:
        return f"{text} | args {args!r}"
    values = list(args if isinstance(args, tuple) else (args,))
    out, last, i = [], 0, 0
    for m in _CONVERSION.finditer(text):
        out.append(text[last:m.start()])
        last = m.end()
        if m.group("type") == "%":
            out.append("%")
            continue
        if i >= len(values):                     # too few arguments; keep the conversion verbatim
            out.append(m.group(0))
            continue
        value, i = values[i], i + 1
        try:
            out.append(m.group(0) % value)
        except (TypeError, ValueError):
            # Precision and type carry over; a field *width* does not, since padding a
            # bracketed list to a column width says nothing.
            kind = m.group("type")
            out.append(fmt(value, (m.group("precision") or "") + kind)
                       if kind in "diouxXeEfFgG" else str(value))
    out.append(text[last:])
    rendered = "".join(out)
    extra = values[i:]
    return rendered if not extra else f"{rendered} | unused args {tuple(extra)!r}"


class SafeFormatting(logging.Filter):
    """Keeps a badly-formatted log record from raising out of the log call.

    ``%``-style formatting happens inside the *handler*, so a bad argument raises in
    ``Handler.emit``. The standard library catches it there and prints ``--- Logging error ---``,
    but pytest's capture handler re-raises, which is how a DEBUG line that formatted a
    per-temperature step-size vector with ``%.4g`` came to fail four builds. A *logger* filter runs
    in ``Logger.handle``, before any handler sees the record, so fixing the record here protects
    every handler --- ours, an application's, and pytest's alike.

    **Strict mode.** The net would otherwise hide exactly the defect it exists for: with it in
    place a bad call site quietly produces a rescued message, and running the suite at DEBUG stops
    being an audit. ``MIMCS_STRICT_LOGGING=1`` re-raises instead, from inside ``Logger.handle`` ---
    so the traceback points at the *call site* rather than at whichever handler happened to format
    it. The audit is::

        MIMCS_LOG_LEVEL=DEBUG MIMCS_STRICT_LOGGING=1 python -m pytest tests/ -q

    which is the only run that exercises every DEBUG line the library has.

    The cost is one extra ``record.getMessage()`` per record that passes the level check, which is
    why it is worth keeping to the convention that nothing logs per transition.
    """

    def __init__(self, strict: bool | None = None):
        """``strict=None`` reads :data:`STRICT_ENV_VAR` at every record, so a test (or a shell)
        can turn the audit on without rebuilding the loggers that already hold this filter."""
        super().__init__()
        self.strict = strict

    def _strict(self) -> bool:
        if self.strict is not None:
            return self.strict
        return os.environ.get(STRICT_ENV_VAR, "").strip().lower() in ("1", "true", "yes", "on")

    def filter(self, record: logging.LogRecord) -> bool:
        if record.args:
            try:
                record.getMessage()
            except Exception:
                if self._strict():
                    raise
                record.msg, record.args = _rescue(record.msg, record.args), ()
        return True


#: One shared instance; filters hold no per-logger state, so every logger can use the same one.
_SAFE_FORMATTING = SafeFormatting()


def get_logger(name: str | None = None) -> logging.Logger:
    """The logger for a module: ``log = get_logger(__name__)``.

    ``name`` is a dotted module name (already under ``mimcs``); omitting it gives the library
    root logger.

    Every logger handed out here carries :class:`SafeFormatting`, so no ``log.debug(...)`` in the
    library can raise out of its call site. It has to be attached per module logger rather than
    once to ``mimcs``: a *logger* filter runs only for records logged on that logger itself, not
    for records propagating up from its children.
    """
    logger = logging.getLogger(name if name else ROOT_LOGGER)
    if _SAFE_FORMATTING not in logger.filters:
        logger.addFilter(_SAFE_FORMATTING)
    return logger


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


__all__ = ["ROOT_LOGGER", "DEFAULT_LEVEL", "LEVEL_ENV_VAR", "STRICT_ENV_VAR",
           "FMT_MAX_ELEMENTS", "fmt",
           "SafeFormatting", "get_logger", "configure_logging", "set_log_level", "log_level"]
