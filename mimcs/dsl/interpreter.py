"""Tree-walking interpreter for the model DSL: AST -> JAX-traceable closures.

Each ``model`` block (with any ``transformed parameters`` prepended) becomes one closure
``(params: dict) -> scalar`` that seeds an environment from the ambient parameter values,
accumulates a scalar ``target``, and returns it. A program with several named ``model``
components produces one such closure per component, all over the same constants and function
table. Because JAX traces a closure exactly once, the per-node walk cost is amortized away.
``transformed data`` is run *eagerly* with concrete data to produce constants. See
``docs/design/08_model_dsl.md``.

**The evaluation context.** Everything a node needs to be evaluated travels in one
:class:`EvalContext`: the frame's variable bindings, the program's function table, the density
accumulator (``None`` where ``~`` / ``target +=`` are illegal), and the call depth. A call to a
user-defined function evaluates its arguments in the *caller's* context and runs the body in a
**fresh** one holding the arguments alone --- so a function sees neither data nor parameters
unless they were passed in, which is Stan's rule and what makes a function a self-contained
closure. ``acc=None`` in that frame is a second, structural guarantee of the purity rule that
:func:`mimcs.dsl.semantics.check_functions` enforces statically.
"""

from __future__ import annotations

import jax.numpy as jnp

from . import ast
from . import semantics
from .builtins import BUILTINS
from .distributions import DISTRIBUTIONS
from .errors import DslError
from .loops import LOOP_FORMS
from .._logging import get_logger

log = get_logger(__name__)


class _Acc:
    """The ``target`` accumulator: a host-side cell holding a JAX scalar during the trace."""

    def __init__(self):
        self.value = jnp.zeros(())


class _Return(Exception):
    """Non-local exit from a user-function body, carrying the returned value.

    Raised by a ``return`` statement and caught by :func:`_call_user_function`, which is the
    only place a function body is ever run. An exception is the right tool because
    :func:`exec_stmt` returns nothing and bodies nest (a ``return`` inside a ``for`` inside an
    ``if`` has to unwind all of it). It is raised and caught entirely on the host while the
    closure is being traced, so it never reaches JAX.
    """

    def __init__(self, value):
        super().__init__("return outside a function body")
        self.value = value


#: Cap on nested user-function calls. Each DSL frame costs several Python frames, so without a
#: cap a runaway recursion surfaces as a span-less ``RecursionError`` from deep inside the
#: interpreter --- and one that never reaches the compiler's error funnel.
MAX_CALL_DEPTH = 64


class EvalContext:
    """One evaluation frame, plus what the whole program shares.

    Attributes:
        env: this frame's variable bindings (mutable, as declarations and assignments write here).
        functions: ``{name: ast.FuncDef}`` for the program --- shared by every frame.
        acc: the density accumulator, or ``None`` where ``~`` / ``target +=`` are not allowed
            (an eager block, or a function body).
        depth: nesting depth of user-function calls, against :data:`MAX_CALL_DEPTH`.
    """

    __slots__ = ("env", "functions", "acc", "depth")

    def __init__(self, env: dict, functions: dict | None = None,
                 acc: _Acc | None = None, depth: int = 0):
        self.env = env
        self.functions = functions or {}
        self.acc = acc
        self.depth = depth

    def call_frame(self, env: dict) -> "EvalContext":
        """A fresh frame for a function body: new bindings, no accumulator, one level deeper."""
        return EvalContext(env, self.functions, acc=None, depth=self.depth + 1)


# --- expressions ------------------------------------------------------------- #

def eval_expr(node: ast.Expr, ctx: EvalContext):
    if isinstance(node, ast.IntLit):
        return node.value
    if isinstance(node, ast.RealLit):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in ctx.env:
            raise DslError(f"unknown name {node.id!r}", node.span)
        return ctx.env[node.id]
    if isinstance(node, ast.UnaryOp):
        v = eval_expr(node.operand, ctx)
        return -v if node.op == "-" else +v
    if isinstance(node, ast.Transpose):
        return jnp.transpose(eval_expr(node.operand, ctx))
    if isinstance(node, ast.BinOp):
        return _eval_binop(node, ctx)
    if isinstance(node, ast.NoneLit):
        return None
    if isinstance(node, ast.TupleLit):
        return tuple(eval_expr(e, ctx) for e in node.elements)
    if isinstance(node, ast.FuncRef):
        raise DslError(
            f"{node.name!r} is a function, not a value: it can only be passed to "
            f"{', '.join(sorted(LOOP_FORMS))}", node.span)
    if isinstance(node, ast.Call):
        form = LOOP_FORMS.get(node.fn)
        if form is not None:
            return _eval_loop_form(form, node, ctx)
        args = [eval_expr(a, ctx) for a in node.args]      # evaluated in the *caller's* frame
        fn = ctx.functions.get(node.fn)
        if fn is not None:
            return _call_user_function(fn, args, node, ctx)
        if node.fn not in BUILTINS:
            raise DslError(f"unknown function {node.fn!r}", node.span)
        return BUILTINS[node.fn](*args)
    if isinstance(node, ast.Index):
        base, key = eval_expr(node.base, ctx), _index_key(node, ctx)
        if _is_traced_key(key):
            # `data` and `transformed data` constants are *numpy* arrays, and indexing one with a
            # traced index calls `__array__` on the tracer. Inside a loop body the index is
            # exactly that, so hand JAX the array instead. Only this path converts, so the
            # ordinary static-index case keeps its numpy semantics (and its dtype) untouched.
            base = jnp.asarray(base)
        return base[key]
    raise DslError(f"cannot evaluate {type(node).__name__}", getattr(node, "span", None))


def _call_user_function(fn: ast.FuncDef, args: list, node: ast.Call, ctx: EvalContext):
    """Run a user-defined function's body in a fresh frame and return its value."""
    if len(args) != len(fn.params):
        raise DslError(f"{fn.name}() takes {len(fn.params)} argument(s), {len(args)} given",
                       node.span)
    if ctx.depth >= MAX_CALL_DEPTH:
        raise DslError(
            f"call depth exceeded {MAX_CALL_DEPTH} while calling {fn.name!r} "
            f"(infinite recursion?)", node.span)
    # A `None` parameter may have no name (`f(None, real x)`): there is nothing to refer to, so
    # it binds nothing while still occupying its argument position.
    frame = ctx.call_frame({p.name: a for p, a in zip(fn.params, args) if p.name is not None})
    try:
        for stmt in fn.body:
            exec_stmt(stmt, frame)
    except _Return as r:
        return r.value
    raise DslError(f"function {fn.name!r} finished without returning a value", fn.span)


def _eval_loop_form(form, node: ast.Call, ctx: EvalContext):
    """Evaluate a ``scan`` / ``fori_loop`` call (:mod:`mimcs.dsl.loops`).

    The body becomes a Python callable that runs the user function through the ordinary
    :func:`_call_user_function` path --- so a loop body gets the same fresh frame, the same
    ``acc=None`` purity guarantee and the same call-depth cap as any other call, and JAX traces
    it once inside ``lax.scan``. Everything else is evaluated in the caller's frame as usual.
    """
    if len(node.args) < form.n_fixed:
        raise DslError(
            f"`{form.name}` takes at least {form.n_fixed} arguments, {len(node.args)} given "
            f"--- `{form.signature}`", node.span)
    ref = node.args[form.fn_arg]
    if not isinstance(ref, ast.FuncRef):
        raise DslError(
            f"argument {form.fn_arg + 1} of `{form.name}` must name a function --- "
            f"`{form.signature}`", node.span)
    fn = ctx.functions.get(ref.name)
    if fn is None:
        raise DslError(
            f"`{form.name}`: {ref.name!r} is not a user-defined function. The body must be "
            f"defined in the `functions` block.", ref.span)

    def body(*values):
        return _call_user_function(fn, list(values), node, ctx)

    rest = [eval_expr(a, ctx) for i, a in enumerate(node.args) if i != form.fn_arg]
    for pos in form.static_args:
        rest[pos] = _static_loop_bound(rest[pos], node.span, f"a `{form.name}` bound")
    # A `None` in the length-bearing slot (`scan`'s `xs`) means the next argument is the loop
    # length, and it must be static for the same reason the bounds are: that is what lets the
    # loop lower to a `scan` and stay reverse-mode differentiable.
    length_at = form.length_after
    if length_at is not None and rest[length_at] is None and len(rest) > length_at + 1:
        rest[length_at + 1] = _static_loop_bound(
            rest[length_at + 1], node.span, f"a `{form.name}` length")
    return form.impl(body, *rest, span=node.span)


def _static_loop_bound(value, span, what: str) -> int:
    """A loop bound: compile-time constant *and* a whole number.

    Constant is what lets the loop lower to a ``scan`` and so stay reverse-mode differentiable
    (see :mod:`mimcs.dsl.loops`); a traced value fails that here rather than deep inside JAX.

    The integrality check is separate and just as necessary. ``as_static_int`` is an ``int()``
    call, which happily *truncates*: without this, ``fori_loop(1, 2.7, ...)`` would silently run
    two iterations instead of complaining, and a bound quietly off by one is precisely the bug
    that is hardest to see in a result.
    """
    n = semantics.as_static_int(value, span, what)
    if float(value) != n:
        raise DslError(f"{what} must be a whole number, got {value}", span)
    return n


def _eval_binop(node: ast.BinOp, ctx: EvalContext):
    lo, ro = eval_expr(node.lhs, ctx), eval_expr(node.rhs, ctx)
    op = node.op
    if op == "*":                                  # matmul, or scalar multiply if either scalar
        if jnp.ndim(lo) == 0 or jnp.ndim(ro) == 0:
            return lo * ro
        return jnp.matmul(lo, ro)
    return {
        "+": lambda: lo + ro, "-": lambda: lo - ro,
        ".+": lambda: lo + ro, ".-": lambda: lo - ro,
        ".*": lambda: lo * ro, "/": lambda: lo / ro, "./": lambda: lo / ro,
        "^": lambda: lo ** ro, ".^": lambda: lo ** ro,
        "<": lambda: lo < ro, ">": lambda: lo > ro, "<=": lambda: lo <= ro,
        ">=": lambda: lo >= ro, "==": lambda: lo == ro, "!=": lambda: lo != ro,
    }[op]()


def _is_traced_key(key) -> bool:
    """Does this index key contain anything that is not statically known?

    ``None`` counts as static: it is ``jnp.newaxis``, a pure reshape. Leaving it out would send
    every ``constant[:, None]`` through ``jnp.asarray`` and quietly change a numpy constant's
    dtype in an eager ``transformed data`` block.
    """
    parts = key if isinstance(key, tuple) else (key,)
    return any(not isinstance(p, (int, slice, type(None))) for p in parts)


def _index_key(node: ast.Index, ctx: EvalContext):
    keys = []
    for arg in node.args:
        if isinstance(arg, ast.NewAxis):
            keys.append(None)                      # `None` is `jnp.newaxis`
        elif isinstance(arg, ast.ScalarIndex):
            keys.append(semantics.lower_scalar_index(eval_expr(arg.expr, ctx)))
        else:                                      # Range: slice bounds must be static
            lo = (None if arg.lo is None else
                  semantics.as_static_int(eval_expr(arg.lo, ctx), node.span, "slice bound"))
            hi = (None if arg.hi is None else
                  semantics.as_static_int(eval_expr(arg.hi, ctx), node.span, "slice bound"))
            keys.append(semantics.lower_range(lo, hi))
    return tuple(keys) if len(keys) > 1 else keys[0]


# --- statements -------------------------------------------------------------- #

def exec_stmt(node: ast.Stmt, ctx: EvalContext):
    if isinstance(node, ast.VarDecl):              # local declaration
        if node.init is not None:
            ctx.env[node.name] = eval_expr(node.init, ctx)
        elif node.base_type == "None":             # `None x;` binds the empty value
            ctx.env[node.name] = None
        elif node.shape:                           # uninitialised array: zeros, for indexed fill
            sizes = tuple(semantics.as_static_int(eval_expr(d, ctx), node.span, "array size")
                          for d in node.shape)
            ctx.env[node.name] = jnp.zeros(sizes)
        return
    if isinstance(node, ast.TupleDecl):
        _bind_targets(node.targets, eval_expr(node.init, ctx), node.span, ctx)
        return
    if isinstance(node, ast.Assign):
        _exec_assign(node, ctx)
        return
    if isinstance(node, ast.TargetPlus):
        _require_acc(ctx, node).value += eval_expr(node.value, ctx)
        return
    if isinstance(node, ast.Sample):
        if node.dist not in DISTRIBUTIONS:
            raise DslError(f"unknown distribution {node.dist!r}", node.span)
        x = eval_expr(node.lhs, ctx)
        args = [eval_expr(a, ctx) for a in node.args]
        _require_acc(ctx, node).value += jnp.sum(DISTRIBUTIONS[node.dist](x, *args))
        return
    if isinstance(node, ast.Return):
        # Unwinds out of any nesting to _call_user_function. Inside an unrolled `for` this also
        # stops the unrolling, which is exactly what a runtime `return` would do.
        raise _Return(None if node.value is None else eval_expr(node.value, ctx))
    if isinstance(node, ast.For):
        lo = semantics.as_static_int(eval_expr(node.lo, ctx), node.span, "loop bound")
        hi = semantics.as_static_int(eval_expr(node.hi, ctx), node.span, "loop bound")
        for i in range(lo, hi + 1):                # 1-based inclusive; unrolled
            ctx.env[node.var] = i
            for s in node.body:
                exec_stmt(s, ctx)
        return
    if isinstance(node, ast.If):
        cond = eval_expr(node.cond, ctx)
        try:
            take = bool(cond)
        except Exception:
            raise DslError("stage 1 supports only compile-time-constant `if` conditions",
                           node.span)
        for s in (node.then_body if take else node.else_body):
            exec_stmt(s, ctx)
        return
    if isinstance(node, ast.While):
        raise DslError("`while` is not supported yet (stage 1)", node.span)
    raise DslError(f"cannot execute {type(node).__name__}", getattr(node, "span", None))


def _bind_targets(targets: tuple, value, span, ctx: EvalContext) -> None:
    """Bind a destructuring target list against a value, recursing into nested groups."""
    labels = [_target_label(t) for t in targets]
    if not isinstance(value, tuple):
        raise DslError(
            f"cannot destructure into ({', '.join(labels)}): the right-hand side is a single "
            f"value, not a tuple", span)
    if len(value) != len(targets):
        raise DslError(
            f"cannot destructure a {len(value)}-tuple into {len(targets)} name(s) "
            f"({', '.join(labels)})", span)
    for target, part in zip(targets, value):
        if isinstance(target, ast.TupleTarget):
            _bind_targets(target.targets, part, span, ctx)
        elif target.name is not None:              # a nameless `None` discards its element
            ctx.env[target.name] = part


def _target_label(target) -> str:
    if isinstance(target, ast.TupleTarget):
        return f"({', '.join(_target_label(t) for t in target.targets)})"
    return target.name if target.name is not None else "None"


def _exec_assign(node: ast.Assign, ctx: EvalContext):
    target = node.target
    value = eval_expr(node.value, ctx)
    if isinstance(target, ast.Name):
        ctx.env[target.id] = value
    elif isinstance(target, ast.Index):
        base = eval_expr(target.base, ctx)
        if not isinstance(target.base, ast.Name):
            raise DslError("indexed assignment target must be a variable", node.span)
        ctx.env[target.base.id] = base.at[_index_key(target, ctx)].set(value)
    else:
        raise DslError("invalid assignment target", node.span)


def _require_acc(ctx: EvalContext, node):
    if ctx.acc is None:
        raise DslError("sampling / `target +=` is only allowed in a model block", node.span)
    return ctx.acc


# --- closure / eager drivers ------------------------------------------------- #

def run_eager(stmts, constants: dict, functions: dict | None = None) -> dict:
    """Run a block (e.g. ``transformed data``) eagerly with concrete constants."""
    ctx = EvalContext(dict(constants), functions, acc=None)
    for s in stmts:
        exec_stmt(s, ctx)
    log.debug("ran %d eager statement(s); environment now %s", len(stmts), sorted(ctx.env))
    return ctx.env


def build_component_closure(stmts, param_names, constants, functions: dict | None = None,
                            name: str = "target"):
    """Build one model component's log-prob closure ``(params) -> scalar`` (re-run per trace).

    ``name`` is the component this closure becomes in ``Model.log_prob_fns``; a program with
    several named ``model`` blocks calls this once per component, over the same ``constants``
    and ``functions``.
    """
    log.debug("building the %r closure over parameters %s from %d statement(s)",
              name, list(param_names), len(stmts))

    def component_fn(params: dict):
        env = dict(constants)
        for pname in param_names:
            env[pname] = params[pname]
        acc = _Acc()
        ctx = EvalContext(env, functions, acc=acc)
        for s in stmts:
            exec_stmt(s, ctx)
        return acc.value

    return component_fn
