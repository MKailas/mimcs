"""Semantic helpers for the model DSL: index lowering, constant evaluation, the
``parameters`` block -> ``list[BaseParameter]`` plan, and the static checks.

The semantic layer is deliberately light: name resolution happens at interpretation time via
the environment, and JAX reports shape mismatches (attributed to source spans). The
genuinely error-prone bit --- 1-based-inclusive to 0-based-half-open index lowering --- is
isolated here as two tiny, exhaustively testable functions.

What *is* checked statically (so the error arrives from ``compile_model(source)``, before any
data is bound) are the rules a tree walk can settle on its own: that a user-defined function has
a usable name and signature, is pure, and returns something; that ``return`` appears only in a
function body; that a call to a user function has the right number of arguments; and that a
density statement is not written where it would have no component to belong to. Each check
raises a span-carrying :class:`~mimcs.dsl.errors.DslError`; the factory attaches the source.
"""

from __future__ import annotations

from . import ast
from . import parser as _parser
from .builtins import BUILTINS
from .distributions import DISTRIBUTIONS
from .errors import DslError
from .loops import LOOP_FORMS
from ..model import PARAMETER_KINDS
from .._logging import get_logger

log = get_logger(__name__)

#: Names a user-defined function (or one of its arguments) may not take. Keywords carry a
#: meaning the parser already gives them; builtins and distributions would be shadowed or
#: shadowing. Ordinary variables are deliberately *not* restricted: a variable named ``mean``
#: is legal and always has been, because ``Sample.dist`` and ``Call.fn`` are separate namespaces
#: from ``Name.id`` --- but a *function* name shares the call path with the builtins, so it has
#: to be unambiguous.
RESERVED_NAMES = (frozenset(_parser.KEYWORDS) | frozenset(BUILTINS)
                  | frozenset(DISTRIBUTIONS) | frozenset(LOOP_FORMS))


# --- tree walkers ------------------------------------------------------------- #

def iter_stmts(stmts):
    """Every statement in ``stmts``, recursing into ``for`` / ``while`` / ``if`` bodies."""
    for s in stmts:
        yield s
        if isinstance(s, (ast.For, ast.While)):
            yield from iter_stmts(s.body)
        elif isinstance(s, ast.If):
            yield from iter_stmts(s.then_body)
            yield from iter_stmts(s.else_body)


def iter_exprs(stmt):
    """Every expression subtree reachable from one statement (not its nested bodies).

    Combined with :func:`iter_stmts` this reaches every expression in a block --- which is what
    the call-arity and function-scope checks walk.
    """
    roots = []
    if isinstance(stmt, ast.VarDecl):
        roots = [*(stmt.shape or ()), *stmt.base_args, stmt.lower, stmt.upper, stmt.init]
    elif isinstance(stmt, ast.TupleDecl):
        roots = [stmt.init]
        for t in iter_targets(stmt.targets):
            roots += [*(t.shape or ()), *t.base_args, t.lower, t.upper]
    elif isinstance(stmt, ast.Assign):
        roots = [stmt.target, stmt.value]
    elif isinstance(stmt, ast.Sample):
        roots = [stmt.lhs, *stmt.args]
    elif isinstance(stmt, ast.TargetPlus):
        roots = [stmt.value]
    elif isinstance(stmt, ast.Return):
        roots = [stmt.value]
    elif isinstance(stmt, ast.For):
        roots = [stmt.lo, stmt.hi]
    elif isinstance(stmt, (ast.While, ast.If)):
        roots = [stmt.cond]
    for root in roots:
        yield from _iter_expr(root)


def _iter_expr(node):
    if node is None:
        return
    yield node
    if isinstance(node, ast.BinOp):
        yield from _iter_expr(node.lhs)
        yield from _iter_expr(node.rhs)
    elif isinstance(node, (ast.UnaryOp, ast.Transpose)):
        yield from _iter_expr(node.operand)
    elif isinstance(node, ast.Call):
        for a in node.args:
            yield from _iter_expr(a)
    elif isinstance(node, ast.TupleLit):
        for e in node.elements:
            yield from _iter_expr(e)
    elif isinstance(node, ast.Index):
        yield from _iter_expr(node.base)
        for arg in node.args:
            if isinstance(arg, ast.NewAxis):
                continue                            # `None`: a reshape, with no subexpression
            if isinstance(arg, ast.ScalarIndex):
                yield from _iter_expr(arg.expr)
            else:                                   # Range
                yield from _iter_expr(arg.lo)
                yield from _iter_expr(arg.hi)


def iter_targets(targets):
    """Every leaf :class:`~mimcs.dsl.ast.VarDecl` of a destructuring target list, recursively."""
    for t in targets:
        if isinstance(t, ast.TupleTarget):
            yield from iter_targets(t.targets)
        else:
            yield t


def _all_exprs(stmts):
    for s in iter_stmts(stmts):
        yield from iter_exprs(s)


def bound_names(stmts) -> set:
    """Every name a statement list binds itself: local declarations and loop variables.

    Collected at any depth and flow-insensitively, like every walk here --- it over-approximates
    what is in scope, which is the safe direction for both callers (a name wrongly considered
    bound is merely not reported as free).
    """
    names = set()
    for s in iter_stmts(stmts):
        if isinstance(s, ast.VarDecl):
            names.add(s.name)
        elif isinstance(s, ast.TupleDecl):
            names.update(t.name for t in iter_targets(s.targets) if t.name is not None)
        elif isinstance(s, ast.For):
            names.add(s.var)
    return names


def read_names(stmts) -> frozenset:
    """Every *free* name a statement list reads --- what it needs from its environment.

    Three details make this both necessary and sufficient. An assignment *target* is an
    ``ast.Name`` too (:func:`iter_exprs` puts it among the roots), so ``y[i] = ...`` reports
    ``y``; ``VarDecl.name`` and ``For.var`` are plain strings, so a local or loop variable is
    never *yielded* as a binding but always *read* as a name --- hence subtracting
    :func:`bound_names`; and ``Call.fn`` / ``Sample.dist`` are strings too, so no function or
    distribution name can leak in.

    Function **bodies are deliberately not descended into**: a function sees only its own
    arguments and locals (:func:`check_functions`, and structurally
    :func:`mimcs.dsl.interpreter._call_user_function`), so the argument expressions at each call
    site --- already inside ``stmts``, already reached here --- are the whole story. A transitive
    walk would also need a cycle guard, since mutual recursion is legal.
    """
    names = {e.id for e in _all_exprs(stmts) if isinstance(e, ast.Name)}
    return frozenset(names - bound_names(stmts))


# --- 1-based inclusive -> 0-based half-open index lowering -------------------- #

def lower_scalar_index(i):
    """Stan ``a[i]`` (1-based) -> JAX ``a[i - 1]``."""
    return i - 1


def lower_range(lo, hi):
    """Stan ``a[lo:hi]`` (both inclusive, 1-based) -> a JAX ``slice``.

    The lower bound shifts by -1; the (inclusive) upper bound becomes the (exclusive) JAX
    stop unchanged, since ``(hi - 1) + 1 == hi``. ``None`` means an open end.
    """
    return slice(None if lo is None else lo - 1, hi)


# --- constant evaluation (for array sizes and constant bounds) --------------- #

def as_static_int(value, span, what: str) -> int:
    """Coerce a build-time-constant value to a Python int, else a span-aware error."""
    try:
        return int(value)
    except Exception:
        raise DslError(f"{what} must be a compile-time constant integer", span)


def const_eval(expr: ast.Expr, constants: dict):
    """Evaluate a constant expression (literals, constant names, arithmetic) at build time."""
    if isinstance(expr, ast.IntLit):
        return expr.value
    if isinstance(expr, ast.RealLit):
        return expr.value
    if isinstance(expr, ast.Name):
        if expr.id not in constants:
            raise DslError(f"unknown name {expr.id!r} (not a constant)", expr.span)
        return constants[expr.id]
    if isinstance(expr, ast.UnaryOp):
        v = const_eval(expr.operand, constants)
        return -v if expr.op == "-" else +v
    if isinstance(expr, ast.BinOp):
        a, b = const_eval(expr.lhs, constants), const_eval(expr.rhs, constants)
        return {"+": a + b, "-": a - b, "*": a * b, "/": a / b, "^": a ** b}[expr.op.lstrip(".")]
    raise DslError("expected a constant expression here", expr.span)


# --- parameters block -> BaseParameter list ---------------------------------- #

def _resolve_bound(expr, param_names, constants, span):
    if expr is None:
        return None
    if isinstance(expr, (ast.IntLit, ast.RealLit)):
        return float(expr.value)
    if isinstance(expr, ast.Name):
        if expr.id in param_names:
            return expr.id                       # parent-dependent bound (a parameter name)
        if expr.id in constants:
            return float(constants[expr.id])
        raise DslError(f"unknown name {expr.id!r} in a bound", expr.span)
    raise DslError("stage 1 supports only constant or single-parameter bounds", span)


def _chart_kwargs(decl, charts: dict | None) -> dict:
    """The chart keyword arguments for one declaration, validated against its kind.

    Which options a kind accepts (``centered`` for the Euclidean/bounded standardization,
    ``adaptive`` for a unit vector's fitted chart) is the parameter type's own business, read
    off :data:`~mimcs.model.PARAMETER_KINDS`. Neither has DSL syntax --- they are supplied by a
    :class:`~mimcs.dsl.spec.ModelSpec`.
    """
    opts = dict((charts or {}).get(decl.name) or {})
    kind = PARAMETER_KINDS[decl.base_type]
    for opt in opts:
        if opt in kind.chart_options:
            continue
        hint = kind.chart_option_hints.get(opt)
        if hint is not None:
            raise DslError(f"parameter {decl.name!r} is a `{kind.name}`: {hint}", decl.span)
        takes = (", ".join(kind.chart_options) if kind.chart_options
                 else "no chart options --- its chart is fixed")
        raise DslError(
            f"{opt!r} is not a chart option for parameter {decl.name!r} (a "
            f"{decl.base_type} parameter takes {takes})", decl.span)
    return opts


def plan_parameters(decls, constants, charts: dict | None = None) -> list:
    """Turn ``parameters``-block declarations into a list of ``BaseParameter``.

    ``charts`` maps a parameter name to keyword options for its chart
    (``{"beta": {"centered": True}}``) --- the flags the grammar has no syntax for, supplied by a
    :class:`~mimcs.dsl.spec.ModelSpec`. An option that does not apply to the declared kind is a
    span-carrying error, pointing at the declaration that makes it inapplicable.

    Which class a declaration becomes is the registry's decision
    (:data:`~mimcs.model.PARAMETER_KINDS`), not this function's: everything DSL-shaped ---
    constant folding, bound resolution --- happens here, and the resolved plain values go to the
    kind's builder. A builder rejecting its arguments raises :class:`ValueError`, which becomes a
    span-carrying error against the declaration responsible.
    """
    param_names = {d.name for d in decls}
    params = []
    for d in decls:
        if d.init is not None:
            raise DslError("parameters cannot have an initializer", d.span)
        if d.base_type not in PARAMETER_KINDS:
            # `None` is a value and a signature type, never something to sample: it would have
            # no coordinates and no density. Caught here so it is a span-carrying error rather
            # than a bare KeyError out of the table below.
            raise DslError(
                f"`{d.base_type}` is not a parameter type --- a parameter must be one of "
                f"{', '.join(PARAMETER_KINDS)}", d.span)
        chart = _chart_kwargs(d, charts)
        kind = PARAMETER_KINDS[d.base_type]
        shape = tuple(as_static_int(const_eval(dim, constants), d.span, "array size")
                      for dim in d.shape)
        # An `array[n]` prefix sizes the array *of* elements; the base type's own sizes size the
        # element (`unit_vector[d]`). The parser has already enforced their count.
        base_sizes = tuple(
            as_static_int(const_eval(a, constants), d.span, f"{kind.name} size")
            for a in d.base_args)
        lower = _resolve_bound(d.lower, param_names, constants, d.span)
        upper = _resolve_bound(d.upper, param_names, constants, d.span)
        try:
            params.append(kind.build(d.name, shape, base_sizes=base_sizes,
                                     lower=lower, upper=upper, **chart))
        except ValueError as e:
            raise DslError(str(e), d.span) from e
    # A discrete parameter has no `coord_dim` -- it contributes no coordinate at all -- so report
    # its support width instead of inventing a dimension for it.
    log.debug("planned %d parameter(s): %s", len(params),
              ", ".join(f"{p.name}[{type(p).__name__}, "
                        f"{p.coord_dim}d]" if hasattr(p, "coord_dim") else
                        f"{p.name}[{type(p).__name__}, {p.size} discrete]"
                        for p in params) or "(none)")
    return params


# --- functions block -> the call table ---------------------------------------- #

def _why_reserved(name: str) -> str:
    if name in BUILTINS:
        return f"{name!r} is a builtin function"
    if name in DISTRIBUTIONS:
        return f"{name!r} is a distribution name"
    if name in LOOP_FORMS:
        return f"{name!r} is a loop form"
    return f"{name!r} is a reserved word"


def _bound_names(fn: ast.FuncDef) -> set:
    """Everything a function body can legitimately read: its arguments, its own local
    declarations and its loop variables (collected at any depth --- the check below is
    flow-insensitive on purpose, so it never reports a false error)."""
    names = {p.name for p in fn.params if p.name is not None}
    for s in iter_stmts(fn.body):
        if isinstance(s, ast.VarDecl):
            names.add(s.name)
        elif isinstance(s, ast.TupleDecl):
            names.update(t.name for t in iter_targets(s.targets) if t.name is not None)
        elif isinstance(s, ast.For):
            names.add(s.var)
    return names


def check_functions(funcdefs) -> dict:
    """Validate a ``functions`` block and return its ``{name: FuncDef}`` call table.

    The rules, each an error the user can act on:

    * the name is free (not a keyword, builtin or distribution) and defined once;
    * argument names are free and distinct;
    * the return type is not ``void`` --- a pure function with no return value could have no
      observable effect at all;
    * the body is **pure**: no ``~``, no ``target +=`` (Stan's ``_lp`` functions are not
      supported), so a function is a value of its arguments and nothing else;
    * the body returns somewhere;
    * every name the body reads is one of its own arguments or locals --- a function does not
      see data, parameters or transformed parameters.
    """
    table: dict = {}
    for fn in funcdefs:
        if fn.name in RESERVED_NAMES:
            raise DslError(f"{_why_reserved(fn.name)} and cannot be used as a function name",
                           fn.span)
        if fn.name in table:
            raise DslError(f"function {fn.name!r} is already defined", fn.span)
        seen: set = set()
        for p in fn.params:
            if p.name is None:                     # a nameless `None` argument binds nothing
                continue
            if p.name in RESERVED_NAMES:
                raise DslError(
                    f"{_why_reserved(p.name)} and cannot be used as an argument name", p.span)
            if p.name in seen:
                raise DslError(
                    f"duplicate argument name {p.name!r} in function {fn.name!r}", p.span)
            seen.add(p.name)
            kind = PARAMETER_KINDS.get(p.type.base)
            if kind is not None and kind.parameter_only:
                raise DslError(
                    f"`{kind.name}` may only be declared in the `parameters` block, not as a "
                    f"function argument", p.span)
        if fn.return_type.base == "void":
            raise DslError(
                f"a `void` function has no observable effect (function bodies are pure): give "
                f"{fn.name!r} a return type", fn.return_type.span)
        for s in iter_stmts(fn.body):
            if isinstance(s, ast.Sample):
                raise DslError(
                    "`~` is not allowed in a function body: functions are pure "
                    "(`_lp` functions are not supported)", s.span)
            if isinstance(s, ast.TargetPlus):
                raise DslError(
                    "`target +=` is not allowed in a function body: functions are pure "
                    "(`_lp` functions are not supported)", s.span)
        if not any(isinstance(s, ast.Return) for s in iter_stmts(fn.body)):
            raise DslError(f"function {fn.name!r} never returns a value", fn.span)
        bound = _bound_names(fn)
        for expr in _all_exprs(fn.body):
            if isinstance(expr, ast.Name) and expr.id not in bound:
                raise DslError(
                    f"unknown name {expr.id!r} in function {fn.name!r}: a function sees only "
                    f"its arguments and its own locals (pass {expr.id!r} in as an argument)",
                    expr.span)
        table[fn.name] = fn
    # Calls between functions are checked once the table is complete, so a function may call one
    # defined below it (and two may call each other).
    for fn in table.values():
        for expr in _all_exprs(fn.body):
            if (isinstance(expr, ast.Call) and expr.fn not in table
                    and expr.fn not in BUILTINS and expr.fn not in LOOP_FORMS):
                raise DslError(f"unknown function {expr.fn!r}", expr.span)
        check_call_arity(fn.body, table)
        check_loop_forms(fn.body, table)
    if table:
        log.debug("compiled %d user function(s): %s", len(table),
                  ", ".join(f"{f.name}/{len(f.params)}" for f in table.values()))
    return table


def check_no_return(stmts, where: str) -> None:
    """``return`` is a function's way out; anywhere else there is nothing to return from."""
    for s in iter_stmts(stmts):
        if isinstance(s, ast.Return):
            raise DslError(f"`return` is only allowed in a function body, not in {where}", s.span)


def check_call_arity(stmts, functions: dict) -> None:
    """Every call to a *user* function passes the declared number of arguments.

    Builtins are not checked: :data:`mimcs.dsl.builtins.BUILTINS` records no arity, so a wrong
    call there still surfaces as a ``TypeError`` from ``jax.numpy`` at trace time.
    """
    for expr in _all_exprs(stmts):
        if isinstance(expr, ast.Call) and expr.fn in functions:
            declared = len(functions[expr.fn].params)
            if len(expr.args) != declared:
                raise DslError(
                    f"{expr.fn}() takes {declared} argument(s), {len(expr.args)} given",
                    expr.span)


def check_loop_forms(stmts, functions: dict) -> None:
    """Every ``scan`` / ``fori_loop`` call is shaped right, and its body has a matching arity.

    Worth doing statically because the alternative is finding out from inside a JAX trace, where
    the complaint is about pytrees rather than about the program. Three things are settled here:
    the call has at least the fixed arguments; the function slot names a *user-defined* function
    (a builtin makes no sense as a loop body, and `scan(exp, ...)` should say so rather than fail
    on arity); and that function takes the body's fixed arguments plus however many extras the
    call forwards.
    """
    for expr in _all_exprs(stmts):
        if not isinstance(expr, ast.Call) or expr.fn not in LOOP_FORMS:
            continue
        form = LOOP_FORMS[expr.fn]
        if len(expr.args) < form.n_fixed:
            raise DslError(
                f"`{form.name}` takes at least {form.n_fixed} arguments, {len(expr.args)} given "
                f"--- `{form.signature}`", expr.span)
        ref = expr.args[form.fn_arg]
        if not isinstance(ref, ast.FuncRef):
            raise DslError(
                f"argument {form.fn_arg + 1} of `{form.name}` must name a function --- "
                f"`{form.signature}`", expr.span)
        if ref.name not in functions:
            what = ("is a builtin, and a builtin cannot be a loop body"
                    if ref.name in BUILTINS else "is not a user-defined function")
            raise DslError(
                f"`{form.name}`: {ref.name!r} {what}. The body must be defined in the "
                f"`functions` block.", ref.span)
        # A literal `None` in the length-bearing slot adds one fixed argument (the loop length),
        # so the forwarded extras start one later. Visible in the AST, which is what lets a
        # missing length be a compile-time error rather than a complaint from inside JAX.
        n_fixed, none_inputs = form.n_fixed, False
        if form.length_after is not None:
            slot = form.length_after + (1 if form.length_after >= form.fn_arg else 0)
            if isinstance(expr.args[slot], ast.NoneLit):
                none_inputs = True
                n_fixed += 1
                if len(expr.args) < n_fixed:
                    raise DslError(
                        f"`{form.name}` with `None` inputs needs a length: write "
                        f"`{form.name}(f, init, None, n)` --- with no array to scan over, there "
                        f"is nothing to take the length from", expr.span)
        n_extra = len(expr.args) - n_fixed
        expected = form.body_arity + n_extra
        declared = len(functions[ref.name].params)
        if declared != expected:
            extra_note = (f" ({form.body_arity} fixed + {n_extra} forwarded)" if n_extra
                          else "")
            # With `None` inputs the argument after it is the *length*, not an extra --- easy to
            # trip over, and the arity mismatch alone would not say so.
            length_note = (" --- remember that with `None` inputs the next argument is the "
                           "loop length, not a forwarded extra" if none_inputs else "")
            raise DslError(
                f"`{form.name}` calls {ref.name}() with {expected} argument(s){extra_note}, but "
                f"it declares {declared}{length_note} --- `{form.signature}`", expr.span)


def check_no_density(stmts, where: str) -> None:
    """Reject ``~`` / ``target +=`` where no model component owns the contribution.

    Used for ``transformed parameters`` once a program has more than one ``model`` component:
    the block's deterministic statements are prepended to *every* component closure (that is how
    the components see the transformed values), so a density statement there would be counted
    once per component. Attributing it to one component instead would be an arbitrary, invisible
    choice --- hence an error, with the fix always being to move the statement into the component
    it belongs to. With a single component nothing changes: this check is not run.
    """
    for s in iter_stmts(stmts):
        kind = ("`~`" if isinstance(s, ast.Sample)
                else "`target +=`" if isinstance(s, ast.TargetPlus) else None)
        if kind is not None:
            raise DslError(
                f"{kind} in {where} is ambiguous when the program has more than one `model` "
                f"component: move it into the component it belongs to", s.span)
