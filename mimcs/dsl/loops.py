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


def _is_pred_error(exc: Exception) -> bool:
    """Is JAX complaining that ``cond``'s predicate is not a scalar?"""
    return isinstance(exc, TypeError) and "Pred must be a scalar" in str(exc)


def _is_branch_mismatch(exc: Exception) -> bool:
    """Is JAX complaining that the two branches disagree about what they return?

    Two phrases, not one: a dtype or shape difference gives "equal output types", while a *pytree*
    difference --- one branch returning a tuple and the other a scalar, which DSL functions make
    easy --- gives a different message entirely. Matching only the first would let the second reach
    the user as raw JAX text.

    Matched on JAX's own wording rather than on a bare "cond" or "branch", for the reason
    :func:`_is_carry_mismatch` gives: every error a branch raises passes through ``lax.cond``, so a
    loose match would relabel a true error as a confident false one.
    """
    s = str(exc)
    return isinstance(exc, TypeError) and (
        "branches must have equal output types" in s
        or "branch outputs must have the same pytree structure" in s)


def cond(true_body, false_body, pred, *operands, span=None):
    """``cond(pred, true_fn, false_fn, ...extra)`` --- run one branch or the other.

    The runtime counterpart to ``if``, whose condition must be a compile-time constant. Both
    branches are *traced*, so the graph holds both, but only the selected one **executes** --- and
    that is the point rather than an implementation detail:

    **``where`` evaluates both branches and so can poison a gradient that ``cond`` leaves clean.**
    ``where(x > 0, sqrt(x), 0.0)`` at ``x = -1`` gives the right value and a ``NaN`` derivative,
    because ``sqrt`` is differentiated at a negative argument and the ``NaN`` survives the multiply
    by zero. ``cond`` gives ``0.0``. The difference persists under ``vmap(grad(...))``, which
    matters because this library vmaps densities over draws and over discrete candidates.

    Two deliberate departures from ``jax.lax.cond``, both in the "an error beats a wrong answer"
    direction this module already takes with ``fori_loop``'s inclusive range:

    * a **float predicate is refused**. ``lax.cond`` accepts one and branches on ``pred != 0``, so
      in a language with no boolean type ``cond(x, f, g, x)`` --- a plausible slip for
      ``cond(x > 0, f, g, x)`` --- would silently take the true branch for every non-zero ``x``.
    * the two JAX complaints most likely to reach a user are translated (below).

    **What it does not do:** a concrete predicate buys no trace-time short circuit. Both branch
    bodies are interpreted while ``lax.cond`` traces them, so a compile error in the branch *not*
    taken still fires. ``cond`` guards execution, never trace-time validity.
    """
    dtype = jnp.asarray(pred).dtype
    if not (jnp.issubdtype(dtype, jnp.bool_) or jnp.issubdtype(dtype, jnp.integer)):
        raise DslError(
            f"`cond`'s predicate must be a condition, not a number: it has dtype {dtype}. JAX "
            f"would accept this and branch on `pred != 0`, which makes a slip like "
            f"`cond(x, ...)` for `cond(x > 0, ...)` a wrong answer rather than an error. Use a "
            f"comparison (`<`, `>`, `==`, ...), or `any(...)` / `all(...)` to reduce a mask.", span)
    try:
        return jax.lax.cond(pred, true_body, false_body, *operands)
    except DslError:
        # A nested `cond` has already translated and reported against its own span; re-labelling it
        # here would move the error to the outer call.
        raise
    except Exception as exc:
        if _is_pred_error(exc):
            raise DslError(
                f"`cond` chooses one branch for the whole computation, so its predicate must be a "
                f"single true/false value --- an array was given. For an elementwise choice use "
                f"`where(condition, a, b)`, which picks per element. JAX reported: {exc}",
                span) from exc
        if _is_branch_mismatch(exc):
            raise DslError(
                f"`cond`: the two branches must return the same thing --- same shape, same dtype, "
                f"and the same tuple structure --- because one value comes back whichever runs. A "
                f"common cause is an integer literal in one branch (write `0.0`, not `0`) meeting "
                f"a real-valued other. JAX reported: {exc}", span) from exc
        raise


@dataclass(frozen=True)
class LoopForm:
    """One higher-order form: where its functions go, and what shape its call must take.

    Attributes:
        name: the form's name, which is also a reserved word.
        fn_args: indices (in the source argument list) of the arguments naming functions. A tuple
            because ``cond`` takes two --- one per branch --- while the loops take one. Must be
            sorted, distinct, and every entry below ``n_fixed``: the arity arithmetic in
            ``check_loop_forms`` counts everything past ``n_fixed`` as a forwarded operand, so a
            function slot among the extras would be miscounted as one.
        slot_names: what to call each function slot in an error message, in ``fn_args`` order.
            ``("f",)`` for a loop body; ``("true_fn", "false_fn")`` for ``cond``, where "argument 3
            must name a function" is far less use than naming the branch.
        n_fixed: number of arguments before the forwarded ``...extra``.
        body_arity: how many arguments each function takes before the forwarded ones. Zero for
            ``cond``, whose branches take only the operands.
        static_args: positions, among the *non-function* arguments, that must be compile-time
            integers. This is what keeps a loop's length static, hence differentiable.
        length_after: position, among the *non-function* arguments, of a slot that may be
            ``None``; when it is, the argument after it is a compile-time loop length and the
            forwarded extras start one later. ``scan``'s ``xs`` is the only such slot. Kept
            declarative here so the interpreter and the static check agree without either
            special-casing a form by name.
        impl: ``impl(*bodies_in_slot_order, *non_function_args, span=...)``, the non-function
            arguments in source order.
        signature: shown in arity errors.
    """

    name: str
    fn_args: tuple
    n_fixed: int
    body_arity: int
    static_args: tuple
    impl: Callable
    signature: str
    length_after: int | None = None
    slot_names: tuple = ("f",)

    def source_index(self, rest_position: int, n_args: int) -> int:
        """Translate a *non-function* argument position into a **source** argument position.

        Two index spaces meet here: ``static_args`` and ``length_after`` count only the
        non-function arguments (which is what ``impl`` receives), while an AST walk indexes the
        source list. Converting by building the surviving-index list is deliberate --- the obvious
        arithmetic shortcut, "add one for each function slot at or below the position", is **wrong**
        as soon as there are two slots, because the shift from the first can carry the result past
        the second. With ``fn_args = (0, 2)`` and position 1 it yields 2, which is itself a function
        slot, where the answer is 3.
        """
        surviving = [i for i in range(n_args) if i not in self.fn_args]
        if rest_position >= len(surviving):
            raise IndexError(
                f"`{self.name}`: non-function position {rest_position} is out of range for a call "
                f"with {n_args} argument(s)")
        return surviving[rest_position]


#: Every higher-order loop form, keyed by name. Registering one here reserves its name
#: (:data:`mimcs.dsl.semantics.RESERVED_NAMES`) and teaches the parser which argument names a
#: function (:class:`mimcs.dsl.ast.FuncRef`).
LOOP_FORMS: dict[str, LoopForm] = {
    "scan": LoopForm(
        name="scan", fn_args=(0,), n_fixed=3, body_arity=2, static_args=(), impl=scan,
        length_after=1,                            # `xs` may be None; then a length follows
        signature="scan(f, init, xs, ...extra) with f(carry, x, ...extra) -> (carry, y)"),
    "fori_loop": LoopForm(
        name="fori_loop", fn_args=(2,), n_fixed=4, body_arity=2, static_args=(0, 1), impl=fori_loop,
        signature=("fori_loop(lower, upper, f, init, ...extra) with "
                   "f(i, val, ...extra) -> val")),
    "cond": LoopForm(
        name="cond", fn_args=(1, 2), n_fixed=3, body_arity=0, static_args=(), impl=cond,
        slot_names=("true_fn", "false_fn"),
        signature=("cond(pred, true_fn, false_fn, ...extra) with "
                   "true_fn(...extra) / false_fn(...extra) -> value")),
}
