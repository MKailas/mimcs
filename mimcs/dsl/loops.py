"""The loop forms: ``scan`` and ``fori_loop``, the DSL's non-unrolling loops.

A plain ``for`` in this language **unrolls** at trace time
(:func:`mimcs.dsl.interpreter.exec_stmt`), which is right for a three-term sum and hopeless for a
thousand-step recursion: the jaxpr grows with the loop, so compile time and memory do too. These
two forms stay a *single* jaxpr equation however long they run, which is what makes state-space
likelihoods, HMM forward passes and AR recursions practical to write here.

Both keep their **lengths compile-time constant**, and that is not a limitation to work around
--- it is the whole reason they are differentiable. ``jax.lax.fori_loop`` with static bounds
lowers to a ``scan`` (reverse-mode differentiable); with *traced* bounds it lowers to a
``while_loop``, which has no reverse-mode rule and so could never carry a model's gradient. This
module makes the relationship explicit rather than relying on that choice: :func:`fori_loop` is
written *as* a ``scan``, so the differentiability is true by construction.

``scan`` keeps JAX's signature exactly, which is what the DSL's tuples are for --- the body
returns ``(carry, y)`` and the form returns ``(carry, ys)``. Because ``init`` and ``xs`` are
pytrees, a tuple ``init`` carries several values and a tuple ``xs`` scans several arrays in
step, both for free.

.. warning::
   :func:`fori_loop` runs the **inclusive** range ``[lower, upper]``, matching this language's
   ``for (i in 1:n)`` and its 1-based indexing --- *not* ``jax.lax.fori_loop``'s half-open
   ``[lower, upper)``. The same call therefore runs one more iteration here than in JAX. This is
   deliberate (an index that indexes correctly beats an index that ports silently), but it is a
   trap worth knowing about when translating JAX code.

**Closures.** A DSL function sees only its arguments, so a loop body cannot capture data the way
a JAX closure does. Instead, any arguments after the fixed ones are forwarded unchanged to every
body call: ``scan(f, init, xs, A, k)`` calls ``f(carry, x, A, k)``. That is this language's
stand-in for a closure, and it costs nothing at trace time.

This module is a leaf: it imports ``jax`` and :mod:`mimcs.dsl.errors` and nothing else from the
package, so the parser, the semantic pass and the interpreter can all read :data:`LOOP_FORMS`
without a cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp

from .errors import DslError


def _as_pair(out, span, form: str):
    """A loop body's ``(carry, y)`` return, checked before JAX sees it."""
    if not (isinstance(out, tuple) and len(out) == 2):
        raise DslError(
            f"the body of `{form}` must return a `(carry, y)` pair, e.g. "
            f"`return (new_carry, y);` --- got {_describe(out)}", span)
    return out


def _describe(value) -> str:
    if isinstance(value, tuple):
        return f"a {len(value)}-tuple"
    return "a single value"


def _is_carry_mismatch(exc: Exception) -> bool:
    """Is this JAX complaining about the carry, rather than about anything else?

    The translation below must be **narrow**. Everything the body does happens inside
    ``lax.scan``, so every error a body can raise passes through here --- a tracer leak, a shape
    clash, a bad index. Rewriting all of those as "your carry changed type" would replace a true
    error with a confident false one, which is worse than no translation at all. So: a
    ``TypeError`` that actually mentions the carry, and nothing else.
    """
    return isinstance(exc, TypeError) and "carry" in str(exc).lower()


def _carry_error(exc: Exception, span, form: str) -> DslError:
    """Translate JAX's carry-mismatch complaint into something actionable.

    By far the likeliest mistake is a carry whose dtype or shape changes between iterations ---
    writing ``0`` instead of ``0.0`` for the initial value is enough, since an integer carry then
    meets a floating-point body result. JAX reports that as a long structural ``TypeError`` from
    inside ``lax.scan``, describing pytrees rather than the program the user wrote.
    """
    return DslError(
        f"`{form}`: the carry changes type or shape between iterations. It must come back from "
        f"the body exactly as it went in --- a common cause is an integer initial value (write "
        f"`0.0`, not `0`) meeting a real-valued body. JAX reported: {exc}", span)


def scan(body: Callable, init, xs, *rest, span=None):
    """``scan(f, init, xs, ...extra)`` --- JAX's ``scan``, with ``f(carry, x, ...extra)``.

    Either end may be ``None``. An **empty carry** (``init = None``) needs nothing special ---
    ``None`` is an empty pytree, so it threads through untouched and the body returns
    ``(None, y)``; that turns a scan into a map. **Empty inputs** (``xs = None``) do need one
    thing: with no array there is nothing to take the loop length from, so the length is written
    as the next argument, ``scan(f, init, None, n)``, and the extras follow it.
    """
    if xs is None:
        if not rest:
            raise DslError(
                "`scan` with `None` inputs needs a length: write `scan(f, init, None, n)` "
                "--- with no array to scan over, there is nothing to take the length from", span)
        length, extra = rest[0], rest[1:]
    else:
        length, extra = None, rest

    def step(carry, x):
        return _as_pair(body(carry, x, *extra), span, "scan")

    try:
        return jax.lax.scan(step, init, xs, length=length)
    except Exception as exc:
        if _is_carry_mismatch(exc):
            raise _carry_error(exc, span, "scan") from exc
        raise


def fori_loop(body: Callable, lower: int, upper: int, init, *extra, span=None):
    """``fori_loop(lower, upper, f, init, ...extra)`` over the **inclusive** range.

    Implemented as a ``scan`` over the index range, so it is reverse-mode differentiable by
    construction rather than by JAX's choice of lowering. An empty range returns ``init``
    unchanged, matching ``for (i in lo:hi)`` with ``hi < lo``.
    """
    def step(carry, i):
        return body(i, carry, *extra), None

    try:
        carry, _ = jax.lax.scan(step, init, jnp.arange(lower, upper + 1))
    except Exception as exc:
        if _is_carry_mismatch(exc):
            raise _carry_error(exc, span, "fori_loop") from exc
        raise
    return carry


@dataclass(frozen=True)
class LoopForm:
    """One higher-order loop form: where its function goes, and what shape its call must take.

    Attributes:
        name: the form's name, which is also a reserved word.
        fn_arg: index (in the source argument list) of the argument naming the body function.
        n_fixed: number of arguments before the forwarded ``...extra``.
        body_arity: how many arguments the body takes before the forwarded ones.
        static_args: positions, among the *non-function* arguments, that must be compile-time
            integers. This is what keeps a loop's length static, hence differentiable.
        length_after: position, among the *non-function* arguments, of a slot that may be
            ``None``; when it is, the argument after it is a compile-time loop length and the
            forwarded extras start one later. ``scan``'s ``xs`` is the only such slot. Kept
            declarative here so the interpreter and the static check agree without either
            special-casing a form by name.
        impl: ``impl(body, *non_function_args, span=...)``, the non-function arguments in
            source order.
        signature: shown in arity errors.
    """

    name: str
    fn_arg: int
    n_fixed: int
    body_arity: int
    static_args: tuple
    impl: Callable
    signature: str
    length_after: int | None = None


#: Every higher-order loop form, keyed by name. Registering one here reserves its name
#: (:data:`mimcs.dsl.semantics.RESERVED_NAMES`) and teaches the parser which argument names a
#: function (:class:`mimcs.dsl.ast.FuncRef`).
LOOP_FORMS: dict[str, LoopForm] = {
    "scan": LoopForm(
        name="scan", fn_arg=0, n_fixed=3, body_arity=2, static_args=(), impl=scan,
        length_after=1,                            # `xs` may be None; then a length follows
        signature="scan(f, init, xs, ...extra) with f(carry, x, ...extra) -> (carry, y)"),
    "fori_loop": LoopForm(
        name="fori_loop", fn_arg=2, n_fixed=4, body_arity=2, static_args=(0, 1), impl=fori_loop,
        signature=("fori_loop(lower, upper, f, init, ...extra) with "
                   "f(i, val, ...extra) -> val")),
}
