# Configuration

Almost every constant in this library is a statement about the mathematics — a step-size target, an
AIC margin, the Marchenko–Pastur edge — and belongs beside the code that reasons about it, where the
argument for its value can be written down next to it. A handful are not. They are statements about
the **machine**, and the right value for one box is the wrong value for another. Those live in
`mimcs/config.py`.

There are two of them today, and the module is deliberately small for the same reason the top-level
namespace is: a configuration surface grows by accretion unless each addition has to justify itself.

```python
import mimcs

mimcs.config.enable_x64()                    # before you build anything
mimcs.config.set_chunk_bytes("64MiB")        # a bigger budget suits an accelerator
```

or, from the shell, which is the ordering-proof route:

```
MIMCS_ENABLE_X64=1 MIMCS_CHUNK_BYTES=64MiB python run.py
```

## Environment variables

The library reads four in total. This is the only place they are listed together.

| Variable | Read by | When | Meaning |
|---|---|---|---|
| `MIMCS_LOG_LEVEL` | `configure_logging` (doc 12) | at import and at every `set_log_level` | a level name or number |
| `MIMCS_STRICT_LOGGING` | `SafeFormatting` (doc 12) | per log record | make a bad log call raise instead of being rescued |
| `MIMCS_CHUNK_BYTES` | `config.chunk_bytes` | at every row-wise pass | chunk budget: `8388608`, `8M`, `64MiB`, or `none` |
| `MIMCS_ENABLE_X64` | `config._configure_from_env` | at `import mimcs` | turn on float64 |

## The resolver

The precedence is **an explicit setting, then the environment variable, then the default** — the
same chain `_logging._resolve_level` uses, for the same reason: a caller who has said what they want
should not be overruled by an environment they may not control.

Values are resolved at **call time**, never bound as default arguments. That is not a style
preference. `mimcs/_chunked.py` records the incident: a budget bound at definition time makes every
later override silently inert, so a test that lowers it to force the chunked path ends up comparing
the whole-array path against itself and passing. Two tests were vacuous for exactly this reason
before it was caught. Every consumer therefore keeps `budget: int | None = None` and resolves inside
the function.

`options(**kwargs)` is the scoped form, mirroring `log_level`:

```python
with mimcs.config.options(chunk_bytes=64):
    ...                    # forces the chunked path on a small array
```

It restores on the way out of an exception, which is what makes it safe in a test, and it restores
*unset as unset* rather than freezing the environment's value in place.

## `chunk_bytes`, and why it is the one that had to move

`_chunked.map_rows` and `_chunked.sum_rows` size a chunk from a byte budget rather than a row count,
because 512 rows is 8 MB at `d = 2000` and 8 KB at `d = 2`, and no single row count is right for
both a test model and a real one (doc 09, doc 11). The default is 8 MiB.

**That number was tuned on a 6.4 GB CPU-only box**, and the pressures run the other way on an
accelerator. `map_rows` copies each chunk back with `np.asarray`:

```python
part = np.asarray(vf(*[jax.tree.map(lambda a: a[i:i + chunk], t) for t in trees]))
```

On the CPU backend that is a zero-copy view plus a memcpy into a preallocated array — cheap, and the
point of doing it is to release the device buffer as the loop moves on. On a GPU it is a
**device-to-host transfer per chunk**, which is the expensive direction, and a small budget means
many of them. So on a GPU the right setting is likely a much larger budget, or `None`.

`chunk_bytes() is None` is a real setting meaning **never chunk**, not an absent one — the
distinction matters enough that the module carries a `_UNSET` sentinel so the two cannot be
confused. Under it, every pass takes the whole-array path it took before chunking existed.

Nothing about a result depends on the value: `map_rows` is bit-identical at every budget (pinned
across `n ∈ {1, 2, 63, 997}` × budgets `{1, 64, 1024, 2³⁰}` in `tests/test_chunked.py`), and the one
caller whose accumulation is *not* bit-identical gates itself on a separate threshold.

### What is deliberately not configurable

Two neighbouring constants look like they belong here.

**`factory.regression.CHUNK_LOSS_BYTES` is a gate, not a knob.** It decides *whether* the metric fit
takes the `sum_rows` path, which reorders its accumulation and so agrees with the whole-array form
only to ~1e-13. It sits 160× above the largest fit the test suite performs, measured, so that no
seed-pinned expectation can shift underneath it. Lowering it is a correctness change wearing a
tuning change's clothes. It has to be *large* to be safe; `chunk_bytes` has to be *small* to be
useful. That opposition is why one is configuration and the other is not.

**The RNG `buffer_size` is not stream-neutral.** It is the largest array a sampler holds — 219.5 MB
in the parallel-tempered discrete case — so it genuinely is a memory knob, and it is tempting. But
draws agree only up to the first refill, so changing the global default would silently move every
seed-pinned result in the library. It stays a per-sampler argument, where the seed it has to be
quoted alongside is also written down (doc 03).

## x64

The library runs in float32 by default and is float64-capable: it never hardcodes `jnp.float32`, and
the byte-budget arithmetic reads `jnp.result_type(float)` rather than assuming a width. Enabling x64
is worth doing for any density that catastrophically cancels — a funnel's neck, a Poisson likelihood
with counts around 1e7 — where float32 overflows to `inf` mid-warmup and silently corrupts the
adaptation.

`enable_x64()` is a thin wrapper over `jax.config.update("jax_enable_x64", …)` that logs the flip at
INFO. It is a wrapper and not a reimplementation because there is nothing to reimplement: the flag
is JAX's, and pretending otherwise would suggest the library can undo its consequences. Two of those
it documents rather than solves.

**Arrays that already exist keep the dtype they were made with.** There is no way to find them, so
the contract is "call this before you build a model or a sampler". Calling it immediately after
`import mimcs` *is* safe, which was checked rather than assumed: no module of this library captures a
dtype at import time — the precision reads are all `jnp.result_type(float)` inside functions, and
the module-level `jax.jit`s key their caches on avals — so a model built after the call is float64
throughout. A model built *before* it stays float32 and will then disagree with everything built
after. `MIMCS_ENABLE_X64=1` is the only ordering-proof route, since an environment variable read
during import cannot be called too late.

**The flag is process-global JAX state.** It leaks into everything else in the process, which is why
`tests/conftest.py` refuses to collect the scratch directories that enable it: x64 in a pytest run
shifts every float32-margin statistical test in the suite. Tests that need to be precision-agnostic
scale their tolerances by `np.finfo(...).eps` instead of asserting an x64 number (see
`tests/test_chunked.py::_tol`).
