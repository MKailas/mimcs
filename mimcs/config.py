"""Global configuration: the few knobs that depend on the **machine**, not on the model.

Almost every constant in this library is a statement about the mathematics --- a step-size target,
an AIC margin, the Marchenko--Pastur edge --- and belongs beside the code that reasons about it. A
handful are not: they are statements about the box the code is running on, and the right value for
one machine is the wrong value for another. Those live here, where a user can change them without
editing the library::

    import mimcs
    mimcs.config.set_chunk_bytes("64MiB")     # or MIMCS_CHUNK_BYTES=64MiB in the environment
    mimcs.config.enable_x64()                 # or MIMCS_ENABLE_X64=1

The module deliberately mirrors :mod:`mimcs._logging`: module-level constants carrying their own
documentation, a resolver whose precedence is **an explicit setting, then the environment variable,
then the default**, and a context manager for a scoped change. What it does *not* have is a
settings object --- there is one process-wide configuration, the same way there is one ``mimcs``
logger, and threading a config object through the call graph would buy nothing.

**Values are resolved at call time, never bound as default arguments.** That is not a style
preference: a value read once at import cannot be overridden afterwards, and a test that tried
would silently compare a code path against itself. :mod:`mimcs._chunked` records the incident.

**What is deliberately not configurable.** Two neighbouring constants look like they belong here
and do not:

* :data:`mimcs.factory.regression.CHUNK_LOSS_BYTES` is a **gate, not a knob**. It decides whether
  the metric fit takes a path that reorders its accumulation, and it sits far above the largest fit
  the test suite performs so that no seed-pinned expectation can shift underneath it. Lowering it
  is a correctness change dressed as a tuning change.
* The RNG ``buffer_size`` is the largest array a sampler holds --- 219.5 MB in the
  parallel-tempered discrete case --- so it *is* a memory knob. But it is **not stream-neutral**:
  draws agree only up to the first refill, so a global default would silently move every
  seed-pinned result in the library. It stays a per-sampler argument, where the seed it belongs
  with is also written down.
"""

from __future__ import annotations

import contextlib
import os
import re

import jax

from ._logging import get_logger

log = get_logger(__name__)

#: Target working set for one chunk of a row-wise pass, in bytes, when nothing overrides it.
#:
#: A *size*, not a *gate*: it says how much data one chunk of :func:`mimcs._chunked.map_rows` or
#: :func:`~mimcs._chunked.sum_rows` should carry, and small is the whole point --- at 8 MiB a
#: 2000-coordinate row gives ~250 rows per chunk, which is the configuration the measurements in
#: :mod:`mimcs._chunked` were taken at. Below the budget a pass is one whole-array ``vmap``,
#: exactly as it was before chunking existed, so small models --- which is most models, and every
#: model in the test suite --- take no extra dispatch at all.
#:
#: **8 MiB was tuned on a 6.4 GB CPU-only box and is not a universal number.** On a GPU the
#: pressure runs the other way: :func:`~mimcs._chunked.map_rows` copies each chunk back with
#: ``np.asarray``, which is near-free on CPU and a device-to-host transfer on an accelerator, so a
#: much larger budget --- or ``None``, meaning never chunk --- is likely to be right there. That
#: asymmetry is why this is configuration rather than a literal in the source.
DEFAULT_CHUNK_BYTES = 8 * 1024 * 1024

#: Environment variable read by :func:`chunk_bytes`: a byte count, with optional base-2 suffix
#: (``"64MiB"``, ``"8M"``), or ``"none"`` to disable chunking.
CHUNK_BYTES_ENV_VAR = "MIMCS_CHUNK_BYTES"

#: Environment variable read at ``import mimcs`` to enable x64 (see :func:`enable_x64`). Setting it
#: in the environment is the only way to be *sure* the switch happens before anything is built.
X64_ENV_VAR = "MIMCS_ENABLE_X64"

#: Strings :func:`_truthy` accepts as true, shared with :class:`mimcs._logging.SafeFormatting`.
TRUE_STRINGS = ("1", "true", "yes", "on")

#: Sentinel for "nothing has been set here", so that ``None`` can mean the real setting *never
#: chunk* rather than *unset*. The two are genuinely different and the module would be ambiguous
#: without it.
_UNSET = object()

#: The process-wide settings, by name. Only :func:`set_chunk_bytes` and :func:`options` write here.
_settings: dict = {"chunk_bytes": _UNSET}

_SUFFIXES = {"": 1, "k": 1024, "m": 1024 ** 2, "g": 1024 ** 3,
             "kib": 1024, "mib": 1024 ** 2, "gib": 1024 ** 3}
_SIZE = re.compile(r"^\s*(\d+)\s*([a-z]*)\s*$", re.IGNORECASE)


def _truthy(text: str) -> bool:
    return text.strip().lower() in TRUE_STRINGS


def parse_bytes(value) -> int | None:
    """A byte count from an int, a string like ``"64MiB"``, or ``None`` / ``"none"``.

    Suffixes are **base 2** throughout (``M`` and ``MiB`` both mean 1048576), because every use of
    this number is a working-set size against a memory budget, where powers of two are what the
    hardware deals in. Decimal ``MB`` would be a second meaning for the same letter and is not
    accepted rather than being quietly rounded.

    ``None`` --- or the string ``"none"`` --- is a real setting meaning *never chunk*, not a
    missing one. A negative or zero count is rejected: it would ask for chunks of no rows.
    """
    if value is None:
        return None
    if isinstance(value, str):
        if value.strip().lower() in ("none", "off", "never"):
            return None
        m = _SIZE.match(value)
        if not m or m.group(2).lower() not in _SUFFIXES:
            raise ValueError(
                f"cannot read {value!r} as a byte count (try 8388608, '8M', '64MiB' or 'none')")
        value = int(m.group(1)) * _SUFFIXES[m.group(2).lower()]
    value = int(value)
    if value <= 0:
        raise ValueError(f"a chunk budget must be positive; got {value} "
                         f"(use None or 'none' to turn chunking off)")
    return value


def chunk_bytes() -> int | None:
    """The chunk budget in bytes, or ``None`` for "never chunk".

    Precedence: whatever :func:`set_chunk_bytes` or :func:`options` last set, then
    :data:`CHUNK_BYTES_ENV_VAR`, then :data:`DEFAULT_CHUNK_BYTES`. Read this **at call time** ---
    binding it as a default argument would make every later override inert.
    """
    if _settings["chunk_bytes"] is not _UNSET:
        return _settings["chunk_bytes"]
    env = os.environ.get(CHUNK_BYTES_ENV_VAR)
    return DEFAULT_CHUNK_BYTES if env is None else parse_bytes(env)


def set_chunk_bytes(value) -> None:
    """Set the chunk budget: an int, a string like ``"64MiB"``, or ``None`` to never chunk.

    Takes precedence over :data:`CHUNK_BYTES_ENV_VAR`. It affects the *next* row-wise pass, not
    any in flight, and never changes a result --- :func:`mimcs._chunked.map_rows` is bit-identical
    at every budget, and the one caller whose accumulation is not gates itself on a separate
    threshold.
    """
    _settings["chunk_bytes"] = parse_bytes(value)
    log.debug("chunk budget set to %s", "never chunk" if _settings["chunk_bytes"] is None
              else f"{_settings['chunk_bytes']} bytes")


def reset() -> None:
    """Forget every explicit setting, so the environment and the defaults decide again."""
    for name in _settings:
        _settings[name] = _UNSET


@contextlib.contextmanager
def options(**kwargs):
    """Apply settings for the duration of a block, then restore what was there before::

        with mimcs.config.options(chunk_bytes=64):
            ...                                    # forces the chunked path on a small array

    Restores on the way out of an exception too, which is what makes it safe in a test. Accepts
    the same names as the ``set_*`` functions; an unknown name is an error rather than a
    silently-ignored typo.
    """
    unknown = set(kwargs) - set(_settings)
    if unknown:
        raise TypeError(f"unknown option(s) {sorted(unknown)} (known: {sorted(_settings)})")
    previous = {name: _settings[name] for name in kwargs}
    for name, value in kwargs.items():
        _SETTERS[name](value)
    try:
        yield
    finally:
        _settings.update(previous)


def x64_enabled() -> bool:
    """Whether JAX is canonicalizing Python ``float`` to float64 right now."""
    return bool(jax.config.jax_enable_x64)


def enable_x64(on: bool = True) -> None:
    """Turn JAX's x64 mode on (or off), so ``float`` canonicalizes to float64.

    Worth doing for any density that catastrophically cancels in float32 --- a funnel's neck, a
    Poisson likelihood with counts around 1e7 --- where float32 overflows to ``inf`` mid-warmup and
    silently corrupts the adaptation. The library never hardcodes ``jnp.float32``, so it follows
    this switch throughout.

    Two things this **cannot** do anything about, and documents instead of pretending to solve:

    * **Arrays that already exist keep the dtype they were made with.** There is no way to find
      them, so the contract is "call this before you build a model or a sampler". Calling it
      immediately after ``import mimcs`` is safe --- no module of this library captures a dtype at
      import time, which was checked rather than assumed --- but a model built *before* the call
      stays float32 and will then disagree with everything built after it. Setting
      :data:`X64_ENV_VAR` in the environment is the only ordering-proof route, since an env var
      read during import cannot be called too late.
    * **The flag is process-global JAX state, not a mimcs setting.** It leaks into everything else
      in the process --- which is exactly why ``tests/conftest.py`` refuses to collect the scratch
      directories that enable it, since it would shift every float32-margin statistical test in the
      suite.
    """
    on = bool(on)
    if on is x64_enabled():
        return
    jax.config.update("jax_enable_x64", on)
    log.info("x64 %s: Python `float` now canonicalizes to %s. Anything built before this call "
             "keeps the dtype it was built with.", "enabled" if on else "disabled",
             "float64" if on else "float32")


def _configure_from_env() -> None:
    """Apply the environment's settings. Called once from ``mimcs/__init__.py``, deliberately
    before the subpackages are imported --- the same ordering argument ``configure_logging()``
    makes one line above it.

    Only x64 is applied here, because only x64 has to happen *early*; the chunk budget is read at
    every call, so there is nothing to apply in advance.
    """
    if _truthy(os.environ.get(X64_ENV_VAR, "")):
        enable_x64(True)


#: Name to setter, so :func:`options` can accept the same names the ``set_*`` functions take.
_SETTERS = {"chunk_bytes": set_chunk_bytes}


__all__ = ["DEFAULT_CHUNK_BYTES", "CHUNK_BYTES_ENV_VAR", "X64_ENV_VAR", "TRUE_STRINGS",
           "parse_bytes", "chunk_bytes", "set_chunk_bytes", "reset", "options",
           "x64_enabled", "enable_x64"]
