# Logging

## Motivation

Everything the library decides for the user — which mass mode a block gets, when warmup stopped,
which metric candidate won — used to be visible only by holding on to the right object
(`spec.rationale`, `sampler.warmup_mixing_stats()`, `res.converged`) and printing it. A long run
that produced a suspicious answer left no trace of *how* it got there, and a run inside a sweep or
a test left none at all.

Standard-library `logging` gives that trace at no cost when nobody is listening, so the library
can be talkative about its own reasoning without any of it having to be plumbed into return
values.

## Shape

One logger per module, `log = get_logger(__name__)`, so every record is named for where it came
from (`mimcs.factory.rules`, `mimcs.adaptation.termination`, `mimcs.hmc.nuts`) and any subtree can be
turned up or silenced on its own:

```python
import logging
logging.getLogger("mimcs.adaptation").setLevel(logging.DEBUG)   # just the adaptation mixins
```

`mimcs/_logging.py` owns the setup. Importing `mimcs` calls `configure_logging()`, which attaches
**one** stream handler to the `mimcs` logger at **INFO** — the library prints out of the box, which
is the point; a library that stays silent until the application configures logging would leave the
default path exactly as opaque as before. Two consequences are deliberate:

* the handler goes on `mimcs`, never on the root logger, and `propagate` is turned **off**, so the
  library neither reconfigures an application's logging nor double-prints into it;
* `MIMCS_LOG_LEVEL` (a level name or number) overrides the default at import, and
  `set_log_level(...)` / `with log_level("DEBUG"):` change it at runtime.

## Levels

The level is a statement about *who needs to read the message*, not about how interesting the
author found it.

| Level | Meaning | Examples |
|---|---|---|
| `ERROR` | An operation is failing; an exception is about to be raised. | a DSL compile error |
| `WARNING` | The run continues, but a result is suspect. | L-BFGS hit `max_iter`; warmup exhausted `max_warmup` without its criterion firing; post-warmup divergences; max split-R̂ > 1.1 |
| `INFO` | The few decisions and outcomes a user wants without asking. | every factory choice; every warmup-termination check; how warmup and sampling ended; the initial step size |
| `DEBUG` | The detail for when something *is* wrong. | rejected proposals and inert rules; per-candidate metric fits; frozen adaptation state; shapes and dimensions |

Log calls use `%`-style lazy formatting, so a DEBUG call near the sampling loop costs a level
comparison while DEBUG is off. Nothing logs *per transition*: the per-iteration record is
`state.diagnostics` (doc 01), and duplicating it into the log would swamp both the reader and the
loop. Phase boundaries and periodic checks are where the messages sit.

## A log call must not be able to fail a run

Lazy formatting has a cost that is easy to miss: `msg % args` happens inside whichever *handler*
emits the record, so a bad argument raises there, not at the call site. In a numerical library the
recurring mistake is a quantity that is a scalar in one configuration and a vector in another, and
this one was found the hard way. `HMC.__init__` logged `"initial step size %.4g"`; parallel
tempering with a per-rung acceptance signal hands that constructor **one step size per
temperature**, `%.4g` calls `float()` on a length-K array, and `TypeError` came out of the handler.
`logging.Handler.emit` catches it, prints `--- Logging error ---` to stderr and moves on — so in
ordinary use it was invisible — but pytest's capture handler re-raises, so it crashed four builds.
They were builds nobody ran: the four tests pass at the default level, and the suite is never run
at DEBUG.

Two pieces answer it, in `mimcs/_logging.py`.

`fmt(x, spec=".4g")` formats a scalar **or a vector** under one spec and never raises: `None` is
`"n/a"`, a scalar formats as `%` would, an array formats element by element in brackets, elided
after `FMT_MAX_ELEMENTS` with the total count. It pairs with `%s`, never with a numeric conversion.
`samplers/base.py` had a scalar-only `_fmt` of its own; that name is now an alias for this one, so
a parallel-tempered `warmup finished ... step size` prints `[0.5, 0.5, 0.5]` rather than a bare
`repr`.

`SafeFormatting` is the net for the sites nobody thought about. It is a *logger* filter — attached
by `get_logger`, so every one of the library's fifty module loggers carries it — which matters
because a logger filter runs in `Logger.handle`, **before any handler sees the record**. That is
the only hook that protects our handler, an application's, and pytest's alike. When
`record.getMessage()` raises, the message is re-rendered one conversion at a time through `fmt`, so
the record still says what it meant to:

    "step %.4g over %d rung(s)", [0.5, 0.25], 2   ->   "step [0.5, 0.25] over 2 rung(s)"

A filter on the `mimcs` logger alone would not do: logger filters do not run for records
propagating up from child loggers.

**The net would otherwise hide the defect it exists for**, which is why `MIMCS_STRICT_LOGGING=1`
makes it re-raise instead — from inside `Logger.handle`, so the traceback points at the call site
rather than at a handler. The audit is

    MIMCS_LOG_LEVEL=DEBUG MIMCS_STRICT_LOGGING=1 python -m pytest tests/ -q

and it is the only run that exercises every DEBUG line the library has. It was checked to be
non-vacuous by reintroducing the `%.4g` and confirming that strict mode fails the four tests while
the lenient DEBUG run passes them. Beyond the step-size line the audit found nothing: the library's
other array-valued log arguments already go through `%s` (`np.array2string` for the ladder's betas,
`fmt` for the step size, list comprehensions for names), and every numeric conversion is applied to
a value already reduced with `float()`, `int()`, `len()` or a numpy reduction.

## The four that carry weight

**DSL compile errors (ERROR).** `DslError` is raised from the lexer, the parser, semantic analysis
and the interpreter build. Each compiler entry point funnels its failures through
`log_compile_error(err)` — `raise log_compile_error(...)` — so the error, with its `line:col` and
caret line, is on the record *before* it propagates, even if the caller swallows it. A one-shot
`logged` flag (inherited by `DslError.with_source`, which is the same error with a better message)
keeps an error that passes several stages on its way out to a single record.

**L-BFGS hitting its cap (WARNING).** `minimize` is a single `lax.while_loop`, so it reports on
the host after the loop; under `jit` its outcome is a tracer and it says so at DEBUG instead of
inventing numbers. Reaching `max_iter` without meeting `gtol` means the returned fit is the last
iterate, not a minimiser — the failure mode that once left the metric regression's constant
baseline fitted at `b ~ 1e4` (doc 09). One caller opts out: the logistic fit behind
`ClassifierTermination` runs at every check, warm started and ridged, and routinely stops on the
cap with a gradient already near `gtol`; it passes `warn_max_iter=False` (logging the cap at
DEBUG) because it reads `converged` itself.

**Warmup ending on its budget (WARNING) and each check (INFO).** Two new cooperative hooks on
`BaseSampler` carry end-of-phase reporting, in the mixin idiom of the rest of the library:
`_warmup_end_hooks(completed, stopped)` and `_sample_end_hooks(state)`. Both are terminal no-ops,
so they change no behaviour on their own. `_WarmupTermination` uses the first to distinguish the
three ways warmup can end — criterion met (INFO), the mixin's own `max_warmup` exhausted
(WARNING: the chain never met the criterion, and the draws that follow inherit that), or a
user-supplied `n` running out before the criterion could fire (INFO, since that is the user's
instruction, not a failure). Every check logs its statistic, verdict and consecutive-pass count at
INFO, which is the criterion's trajectory as it happens rather than after the fact.

**Post-warmup divergences (WARNING).** NUTS uses `_sample_end_hooks`: warmup divergences are
common and harmless (they are reported at DEBUG), but a divergence with the adaptation frozen
means part of the target could not be integrated and the draws are biased there.

A note on names: a composed sampler class is named by concatenation
(`ClassifierTerminationRobbinsMonroStepSizeScoreMassAdaptation…NUTS`), which is unreadable in a
log line. A mixin that logs about itself reports *its own* class name — `_WarmupTermination`
finds it as the first class in the MRO declaring its own `_stat_name`.

## The factory

`arbitrate` logs each adopted proposal at INFO and each proposal it beat at DEBUG — the same
content as `spec.rationale`, available without keeping the spec, which matters because
`make_sampler` never hands the spec back. Values are formatted compactly (a block list becomes its
`names[dim,kind]` summary; anything else is truncated), since a proposal's value can be a fitted
array pytree. Rules that cannot fire say why at DEBUG (`learned_metric rule inert: no row-aligned
coordinate + gradient evidence`), which answers the most common question about the factory: why
the sophisticated choice did *not* happen.
