"""Abstract syntax tree for the model DSL (stage-1 subset).

Frozen dataclasses; every node carries a ``SourceSpan`` for diagnostics. See
``docs/design/08_model_dsl.md`` for the grammar this represents.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .tokens import SourceSpan


# --- expressions ------------------------------------------------------------- #

class Expr:
    span: SourceSpan


@dataclass(frozen=True)
class IntLit(Expr):
    value: int
    span: SourceSpan


@dataclass(frozen=True)
class RealLit(Expr):
    value: float
    span: SourceSpan


@dataclass(frozen=True)
class Name(Expr):
    id: str
    span: SourceSpan


@dataclass(frozen=True)
class NoneLit(Expr):
    """``None`` --- the same object Python's is, and useful for the same two reasons.

    ``jnp.newaxis`` *is* ``None``, so it reshapes when it appears in an index; and ``None`` is an
    empty JAX pytree, so it stands for an absent ``scan`` carry or an absent per-step input. It
    is an ordinary expression, valid wherever a value is: that is what lets a ``scan`` body write
    ``return (None, y);`` without any special case for the empty carry.
    """

    span: SourceSpan


@dataclass(frozen=True)
class BinOp(Expr):
    op: str               # "+ - * / ^" or elementwise ".+ .- .* ./ .^" or comparisons
    lhs: Expr
    rhs: Expr
    span: SourceSpan


@dataclass(frozen=True)
class UnaryOp(Expr):
    op: str               # "-" or "+"
    operand: Expr
    span: SourceSpan


@dataclass(frozen=True)
class Transpose(Expr):
    operand: Expr
    span: SourceSpan


@dataclass(frozen=True)
class Call(Expr):
    fn: str
    args: list[Expr]
    span: SourceSpan


@dataclass(frozen=True)
class TupleLit(Expr):
    """``(a, b, ...)``: a fixed-length heterogeneous group, evaluated to a Python tuple.

    Tuples exist so that :data:`mimcs.dsl.loops.LOOP_FORMS` can keep JAX's signatures --- a
    ``scan`` body returns ``(carry, y)`` and ``scan`` itself returns ``(carry, ys)``. A Python
    tuple is a JAX pytree, so it passes through ``lax.scan`` untouched. There is no 1-tuple:
    ``(x)`` is grouping, as everywhere else.
    """

    elements: tuple           # tuple[Expr, ...], at least 2
    span: SourceSpan


@dataclass(frozen=True)
class FuncRef(Expr):
    """A function named in argument position, e.g. the ``f`` of ``scan(f, init, xs)``.

    Distinct from :class:`Name` on purpose. A bare name is looked up in the environment, and
    inside a function body it must be one of that body's bound names --- neither is true of a
    function's *name*. Making it its own node means the scope checker skips it and the
    interpreter resolves it, with no special case in either walk. The parser produces it only in
    the function slot of a loop form, so it can never appear where a value is expected.
    """

    name: str
    span: SourceSpan


@dataclass(frozen=True)
class ScalarIndex:
    expr: Expr


@dataclass(frozen=True)
class Range:
    lo: Expr | None
    hi: Expr | None


@dataclass(frozen=True)
class NewAxis:
    """``None`` in an index: insert an axis of length 1, as in NumPy (``a[:, None]``)."""

    span: SourceSpan


@dataclass(frozen=True)
class Index(Expr):
    base: Expr
    args: list                # list[ScalarIndex | Range]
    span: SourceSpan


# --- statements -------------------------------------------------------------- #

class Stmt:
    span: SourceSpan


@dataclass(frozen=True)
class TypeExpr:
    """A declared type. **Recorded, never checked** --- there is no type checker; JAX reports
    real mismatches at trace time, attributed to the source span.

    ``dims`` is three-valued, because a *declaration* knows its sizes and a function *signature*
    does not:

    * ``()`` --- a scalar (``real x``);
    * a tuple of :class:`Expr` --- sized (``array[n, m] real x``);
    * a tuple whose entries are ``None`` --- ranked but unsized (``array[] real`` -> ``(None,)``,
      ``array[,] real`` -> ``(None, None)``); signatures only;
    * ``None`` --- an ``array`` with the rank left unsaid (``array real``); signatures only.
    """

    base: str                 # "int" | "real" | "unit_vector" | "ordered" | "tuple" | "void"
    dims: tuple | None        # see above
    base_args: tuple          # the base type's own sizes, e.g. `unit_vector[d]`
    span: SourceSpan
    # Bounds belong to the *type*, and are written between it and its size, as in Stan:
    # `real<lower=0>`, `ordered<lower=0, upper=1>[d]`. Only kinds whose `ParameterKind`
    # sets `takes_bounds` may carry them.
    lower: Expr | None = None
    upper: Expr | None = None
    # The element types of a `(real, array[n] real)` tuple type; empty for every other base.
    elements: tuple = ()      # tuple[TypeExpr, ...]


@dataclass(frozen=True)
class VarDecl(Stmt):
    base_type: str            # "int" | "real" | "unit_vector"
    shape: tuple              # tuple[Expr, ...] (empty for scalar) -- the `array[...]` prefix
    name: str
    lower: Expr | None
    upper: Expr | None
    init: Expr | None
    span: SourceSpan
    base_args: tuple = ()     # tuple[Expr, ...]: the base type's own size args, e.g.
                              # `unit_vector[d]`. Empty for `real` / `int`.


@dataclass(frozen=True)
class TupleTarget:
    """A parenthesised group inside a destructuring declaration, so nesting can be taken apart.

    Needed as soon as a ``scan`` carries a tuple: the result is then
    ``((carry_a, carry_b), ys)``, and unpacking it in one statement means a target list that
    nests. Elements are :class:`VarDecl` (a ``type name`` leaf) or a further ``TupleTarget``.
    """

    targets: tuple            # tuple[VarDecl | TupleTarget, ...], at least 2
    span: SourceSpan


@dataclass(frozen=True)
class TupleDecl(Stmt):
    """``(real last, array[n] real ys) = scan(...);`` --- destructuring a tuple into declarations.

    Leaf targets are ordinary :class:`VarDecl` nodes with ``init=None``, reused as the element
    descriptors so that a destructured name declares exactly what a plain declaration does; a
    parenthesised group is a :class:`TupleTarget`. Destructuring is the *only* way to take a
    tuple apart --- there is no ``t.1`` accessor (the lexer has no ``.`` token) and no
    tuple-typed local, so inside a target list ``(`` always opens a nested group and never a
    tuple type.
    """

    targets: tuple            # tuple[VarDecl | TupleTarget, ...], at least 2
    init: Expr
    span: SourceSpan


@dataclass(frozen=True)
class Assign(Stmt):
    target: Expr              # Name | Index
    value: Expr
    span: SourceSpan


@dataclass(frozen=True)
class Sample(Stmt):
    lhs: Expr
    dist: str
    args: list[Expr]
    span: SourceSpan


@dataclass(frozen=True)
class TargetPlus(Stmt):
    value: Expr
    span: SourceSpan


@dataclass(frozen=True)
class For(Stmt):
    var: str
    lo: Expr
    hi: Expr
    body: list[Stmt]
    span: SourceSpan


@dataclass(frozen=True)
class While(Stmt):
    cond: Expr
    body: list[Stmt]
    span: SourceSpan


@dataclass(frozen=True)
class If(Stmt):
    cond: Expr
    then_body: list[Stmt]
    else_body: list[Stmt]
    span: SourceSpan


@dataclass(frozen=True)
class Return(Stmt):
    """``return <expr>;`` --- legal only inside a user-defined function body."""

    value: Expr | None        # None is parseable (`return;`) but rejected: functions are pure,
                              # so a `void` function could have no observable effect
    span: SourceSpan


# --- user-defined functions -------------------------------------------------- #

@dataclass(frozen=True)
class Param:
    """One argument of a function signature."""

    type: TypeExpr
    name: str
    span: SourceSpan


@dataclass(frozen=True)
class FuncDef:
    """One definition in a ``functions`` block. Not a :class:`Stmt`: it can appear nowhere else.

    The body is an ordinary statement list, but a **pure** one --- ``~`` and ``target +=`` are
    rejected (see :func:`mimcs.dsl.semantics.check_functions`), so a function is a value-returning
    closure over its arguments alone.
    """

    name: str
    return_type: TypeExpr
    params: list              # list[Param]
    body: list                # list[VarDecl | Stmt]
    span: SourceSpan


# --- top level --------------------------------------------------------------- #

@dataclass(frozen=True)
class Block:
    kind: str                 # "data" | "transformed_data" | "parameters" |
                              # "transformed_parameters" | "model" | "functions" |
                              # "generated_quantities"
    body: list                # list[VarDecl | Stmt] --- list[FuncDef] for a `functions` block
    span: SourceSpan
    name: str | None = None   # the component name of a `model <name>` block


@dataclass(frozen=True)
class Program:
    blocks: list[Block] = field(default_factory=list)
