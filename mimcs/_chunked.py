"""Row-chunked passes over evidence-sized arrays.

Three places map or reduce a function over every row of an ``(n, width)`` array: the evidence
pass (:mod:`mimcs.factory.evidence`), the metric regression (:mod:`mimcs.factory.regression`) and
the sample summary (:mod:`mimcs.summary`). All three wrote that as one ``jax.vmap`` over the whole
array, which materialises **every row's intermediates at once** --- fine at the two-dimensional
scale most tests run at, and the dominant memory cost at the scale a second-round ``analyze`` runs
at. Measured on a 2000-predictor spike-and-slab with 6000 draws (x64): one metric-regression
candidate peaked at ~924 MB, ``summarize`` at +1327 MB, and the whole pipeline at **2858 MB**
against 229.8 MB of actual draws.

After chunking the pipeline peaks at 2537 MB, one candidate at 551 MB (and **5.4x faster**, the
memory traffic having dominated the arithmetic), and ``summarize`` at +1206 MB. That last one is
the weakest of the three on purpose: ``feats``, ``stein`` and the posterior matrix have to exist
whole, because ESS and split-R-hat are time-series statistics over the entire chain.

The fix is to do the same work a chunk of rows at a time. Two helpers, because the two uses have
genuinely different contracts and one function cannot be both:

* :func:`map_rows` --- eager, returns numpy, never differentiated. Used where the full
  ``(n, width)`` result is still wanted but its *intermediates* need not all exist at once.
* :func:`sum_rows` --- traced and differentiable, returns a scalar. Used inside an optimisation
  objective, where reverse-mode AD is what holds every row's residuals live.

**Chunk sizes come from a byte budget, not a row count** (:func:`mimcs.config.chunk_bytes`). A
row count means something different at every width --- 512 rows is 8 MB at ``d = 2000`` and 8 KB at
``d = 2`` --- so one constant could not be right for both a test model and a real one.

Note this is a deliberate departure from the package's stated default, which
``dsl/interpreter.py`` puts as "a ``jax.vmap``, not a ``lax.scan``. There is no carry, so the
semantics are a map". That remains right for one row's work; a chunked accumulator *is* a carry.
"""

from __future__ import annotations

import math
import sys

import numpy as np
import jax
import jax.numpy as jnp

from . import config
from ._logging import get_logger

log = get_logger(__name__)

def _budget(budget):
    """Resolve a caller's ``budget``, reading the configured one when it is ``None``.

    The budget is a *size*, not a *gate*: it says how much data one chunk should carry, and small
    is the whole point. It is deliberately **not** the number that decides *whether* to chunk.
    :func:`map_rows` chunks unconditionally because it is bit-identical, so a small budget costs
    nothing; :func:`sum_rows` is not, so its caller gates it on a separate and much larger
    threshold (:data:`mimcs.factory.regression.CHUNK_LOSS_BYTES`). Conflating the two gives either
    a gate too low to be safe or chunks too large to help.

    The value lives in :mod:`mimcs.config` rather than here because it is a statement about the
    machine: 8 MiB was tuned on a 6.4 GB CPU-only box, and an accelerator wants a very different
    number --- or none at all, since the per-chunk ``np.asarray`` below is a device-to-host
    transfer there. ``None`` from :func:`mimcs.config.chunk_bytes` is that setting --- *never
    chunk* --- and is passed through as such; the callers below treat it as one whole-array pass.
    """
    return config.chunk_bytes() if budget is None else budget


def rows_per_chunk(row_bytes: int, budget: int | None = None) -> int:
    """How many rows of ``row_bytes`` each fit in ``budget`` --- at least 1.

    The floor matters: a single row wider than the whole budget must still make progress, one row
    at a time, rather than produce a zero-length chunk and an infinite loop.

    ``budget=None`` reads :func:`mimcs.config.chunk_bytes` **at call time** rather than binding it
    as a default argument, which is what lets a test (or a user, or a GPU) lower it to force the
    chunked path on a small array. A default bound at definition time would make that override
    silently do nothing --- and the test would then pass by comparing the unchunked path against
    itself.

    A configured budget of ``None`` means *never chunk*, which is reported here as a row count
    nothing can exceed rather than as a special case every caller would have to remember.
    """
    budget = _budget(budget)
    if budget is None:
        return sys.maxsize
    return max(1, int(budget) // max(1, int(row_bytes)))


def _row_bytes(*arrays, extra: int = 0) -> int:
    """Bytes one row of every array occupies, plus a caller-declared allowance."""
    total = int(extra)
    for a in arrays:
        shape, dtype = np.shape(a), np.dtype(getattr(a, "dtype", float))
        total += int(np.prod(shape[1:], dtype=np.int64)) * dtype.itemsize
    return max(1, total)


def map_rows(fn, *trees, extra_row_bytes: int = 0, budget: int | None = None) -> np.ndarray:
    """``jax.vmap(fn)`` over the rows of ``trees``, a chunk at a time, into one numpy array.

    ``fn`` takes one row of each tree and returns one row of the result. Each tree is a pytree of
    arrays (commonly just an array, but ``_dep_data``'s ``{name: (N, dep_dim)}`` dict is one too)
    whose leaves all share the leading (row) dimension. The return is ``(n, *fn_output_shape)``.

    **Bit-identical to a single whole-array ``jax.vmap``.** Rows are independent, so chunking
    cannot change any row's value --- which is a statement about the mathematics, so it was
    *checked* rather than assumed (XLA is free to lower a batched computation differently at
    different batch sizes, and these functions do contain intra-row reductions). Verified equal
    under ``np.array_equal`` for ``Model.features``, ``stein_terms``, ``ambient_score`` (both
    branches) and ``sample_to_coordinate``, at chunk sizes 1/7/128/500/1024 over 997 rows, at
    widths 9 and 2006, under float32 and x64. ``tests/test_chunked.py`` pins it.

    Two properties beyond the memory saving:

    * **The result owns its data.** ``np.asarray`` of a JAX array is zero-copy on the CPU backend:
      it returns a non-owning view that pins the whole device buffer. Copying each chunk into a
      preallocated array releases the buffer as the loop moves on. ``Evidence.coordinates`` used to
      be exactly such a view (``owns_data=False``).
    * Only one chunk's device buffers are live at a time, which is the point.

    ``extra_row_bytes`` declares the width of the *output* row when the caller knows it and it is
    materially wider than the inputs (``Model.features`` roughly doubles the width). The budget is
    otherwise computed from the inputs alone, since the output shape is not known until ``fn`` has
    run once.
    """
    budget = _budget(budget)
    leaves = jax.tree.leaves(trees)
    n = int(np.shape(leaves[0])[0])
    vf = jax.vmap(fn)
    if n == 0:                    # nothing to chunk; let vmap produce the right empty shape
        return np.array(vf(*trees))
    chunk = min(n, rows_per_chunk(_row_bytes(*leaves, extra=extra_row_bytes), budget))
    if chunk >= n:
        # ``np.array``, not ``np.asarray``: the result must own its data even on this path, or
        # the two branches would differ in exactly the property the docstring promises.
        return np.array(vf(*trees))
    log.debug("map_rows: %d row(s) in chunks of %d", n, chunk)
    out = None
    for i in range(0, n, chunk):
        part = np.asarray(vf(*[jax.tree.map(lambda a: a[i:i + chunk], t) for t in trees]))
        if out is None:
            out = np.empty((n,) + part.shape[1:], part.dtype)
        out[i:i + len(part)] = part
    return out


def sum_rows(fn, rows, *, budget: int | None = None):
    """``sum(fn(row) for row in rows)`` as a differentiable scalar, in memory ``O(chunk)``.

    ``rows`` is a pytree whose leaves all share a leading row dimension; ``fn`` takes one row's
    pytree and returns a scalar.

    Two implementation choices are load-bearing:

    * **``jax.checkpoint`` on the scan body.** Without it the backward pass stores every
      iteration's residuals and the memory is exactly what it was --- the scan alone buys nothing.
      With it each chunk's forward work is recomputed during the backward pass, which on the
      measured case was not merely affordable but *faster*: 740 MB / 15.7 s to 90 MB / 4.3 s,
      because the memory traffic dominated the arithmetic.
    * **``lax.scan``, not a Python loop.** :func:`mimcs.optim.minimize` is a bare
      ``lax.while_loop`` called *outside* any ``jit``, so it rebuilds its jaxprs on every eager
      call (see :mod:`mimcs.adaptation._logistic`); a host-side chunk loop would recompile per
      chunk per iteration.

    Rows are padded up to a whole number of chunks by **repeating row 0** under a 0/1 weight mask,
    so the sum is exactly the sum over the real rows. Repeating a real row rather than zero-filling
    is deliberate: ``_logistic._buffered`` documents that "a non-finite pad would poison the loss
    however little weight it carried" because ``0 * nan`` is ``nan``, and a metric expression may
    be singular at the origin, so a zero row is not a safe pad here even though a zero *weight*
    would suggest it is.

    The pad-and-reshape sits **inside** this function, and therefore inside the objective an
    optimiser traces. Hoisting it out --- doing the layout once rather than per L-BFGS iteration ---
    looks like an obvious win and was measured **inert** (551.0 MB against 551.3 MB on the case
    below): XLA already lifts it out of the ``lax.while_loop`` body as loop-invariant. Recorded so
    the next reader does not spend the same afternoon splitting this in two.

    Not bit-identical to ``jnp.sum(jax.vmap(fn)(rows))`` --- the accumulation is reordered.
    Measured at 1.1e-13 on the loss and 7.4e-16 on the fitted parameters, against a decision margin
    of ``1/N`` nats; callers that cannot afford even that gate on size and keep the whole-array
    form.
    """
    budget = _budget(budget)
    leaves = jax.tree.leaves(rows)
    n = int(np.shape(leaves[0])[0])
    # ``max(1, ...)`` guards ``n == 0``, which would otherwise divide by a zero chunk. No caller
    # reaches here empty (the metric rule needs `LEARNED_METRIC_MIN_ROWS`), but a zero-row sum has
    # an obvious right answer and should not be an exception.
    chunk = max(1, min(n, rows_per_chunk(_row_bytes(*leaves), budget)))
    n_chunks = math.ceil(n / chunk)
    pad = n_chunks * chunk - n
    log.debug("sum_rows: %d row(s) in %d chunk(s) of %d (%d padded row(s))",
              n, n_chunks, chunk, pad)

    if pad:
        rows = jax.tree.map(
            lambda a: jnp.concatenate(
                [a, jnp.broadcast_to(a[:1], (pad,) + jnp.shape(a)[1:])], axis=0),
            rows)
    dtype = jnp.result_type(float)
    weight = jnp.concatenate([jnp.ones(n, dtype), jnp.zeros(pad, dtype)])
    xs = (jax.tree.map(lambda a: a.reshape(n_chunks, chunk, *jnp.shape(a)[1:]), rows),
          weight.reshape(n_chunks, chunk))

    @jax.checkpoint
    def body(carry, x):
        chunk_rows, chunk_weight = x
        vals = jax.vmap(fn)(chunk_rows)
        return carry + jnp.sum(chunk_weight * vals), None

    total, _ = jax.lax.scan(body, jnp.zeros((), dtype), xs)
    return total
