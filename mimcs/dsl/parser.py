"""Hand-written parser for the model DSL: tokens -> ``Program`` AST.

Recursive descent for blocks / statements / declarations, with a Pratt
(precedence-climbing) sub-parser for expressions. See ``docs/design/08_model_dsl.md``.
"""

from __future__ import annotations

from . import ast
from .errors import DslError, log_compile_error
from .lexer import tokenize
from .tokens import Token, TokenKind as T
from .loops import LOOP_FORMS
from .._logging import get_logger
from ..model import PARAMETER_KINDS

log = get_logger(__name__)

#: The word for the empty value *and* the empty type --- Python's, and for the same reasons:
#: ``jnp.newaxis`` is ``None`` (so it reshapes in an index) and ``None`` is an empty JAX pytree
#: (so it stands for an absent ``scan`` carry or input).
NONE = "None"

#: The words that open a declaration: `array`, plus every registered parameter kind. Deriving
#: this from :data:`~mimcs.model.PARAMETER_KINDS` is what makes registering a parameter type
#: reserve its keyword in the grammar --- there is no second list to keep in step.
_TYPE_KEYWORDS = {"array", NONE} | set(PARAMETER_KINDS)
_BLOCK_STARTS = {"data", "parameters", "model", "functions", "transformed", "generated"}

#: Words the language gives a meaning of its own. The lexer emits every word as an ``IDENT``
#: (keywords are contextual, decided here by string comparison), so this set exists for the
#: *semantic* pass: a user-defined function may not take one of these names
#: (:data:`mimcs.dsl.semantics.RESERVED_NAMES`). Ordinary variables are deliberately not
#: restricted --- a variable named ``mean`` or ``beta`` is legal and always has been.
KEYWORDS = frozenset(
    _BLOCK_STARTS                                  # data parameters model functions
                                                   # transformed generated
    | {"quantities"}                               # the second word of `generated quantities`
    | _TYPE_KEYWORDS                               # real int array unit_vector
    | {"void",                                     # a (rejected) function return type
       NONE,                                       # the empty value, and the empty type
       "for", "in", "while", "if", "else",         # statement keywords
       "return", "target",                         # `return expr;` / the accumulator
       "lower", "upper"})                          # constraint keys

# infix binary operators -> (left binding power, right binding power for the rhs)
_INFIX = {
    T.LT: (10, 11), T.GT: (10, 11), T.LE: (10, 11), T.GE: (10, 11),
    T.EQEQ: (10, 11), T.NE: (10, 11),
    T.PLUS: (20, 21), T.MINUS: (20, 21), T.DOTPLUS: (20, 21), T.DOTMINUS: (20, 21),
    T.STAR: (30, 31), T.SLASH: (30, 31), T.DOTSTAR: (30, 31), T.DOTSLASH: (30, 31),
    T.CARET: (50, 50), T.DOTCARET: (50, 50),       # right-associative
}
_POSTFIX_BP = 60


class Parser:
    def __init__(self, source: str):
        self.source = source
        self.toks = tokenize(source)
        self.pos = 0

    # --- token helpers ------------------------------------------------------ #

    def peek(self, k: int = 0) -> Token:
        return self.toks[min(self.pos + k, len(self.toks) - 1)]

    def at(self, kind: T) -> bool:
        return self.peek().kind is kind

    def at_ident(self, text: str, k: int = 0) -> bool:
        t = self.peek(k)
        return t.kind is T.IDENT and t.text == text

    def advance(self) -> Token:
        t = self.toks[self.pos]
        if self.pos < len(self.toks) - 1:
            self.pos += 1
        return t

    def expect(self, kind: T, what: str | None = None) -> Token:
        if not self.at(kind):
            self.error(f"expected {what or kind.value!r}, found {self.peek().text or 'end of input'!r}")
        return self.advance()

    def expect_ident(self, text: str) -> Token:
        if not self.at_ident(text):
            self.error(f"expected {text!r}, found {self.peek().text or 'end of input'!r}")
        return self.advance()

    def error(self, msg: str):
        raise DslError(msg, self.peek().span, self.source)

    # --- program / blocks --------------------------------------------------- #

    def parse_program(self) -> ast.Program:
        blocks = []
        while not self.at(T.EOF):
            blocks.append(self.parse_block())
        return ast.Program(blocks)

    def parse_block(self) -> ast.Block:
        head = self.peek()
        if not (head.kind is T.IDENT and head.text in _BLOCK_STARTS):
            self.error(f"expected a block (data/parameters/model/...), found {head.text or 'end of input'!r}")
        name = None
        if head.text == "transformed":
            self.advance()
            sub = self.advance()
            if sub.text == "data":
                kind = "transformed_data"
            elif sub.text == "parameters":
                kind = "transformed_parameters"
            else:
                raise DslError(f"expected 'data' or 'parameters', found {sub.text!r}", sub.span, self.source)
        elif head.text == "generated":
            self.advance()
            self.expect_ident("quantities")
            kind = "generated_quantities"
        else:
            self.advance()
            kind = head.text                                       # data/parameters/model/functions
            if kind == "model" and self.at(T.IDENT):               # `model prior { ... }`
                name = self.advance().text
        scan_over = None
        if kind == "model" and self.at_ident("scan"):              # `model lik scan(z, y) { ... }`
            scan_over = self.parse_scan_header(name)
        elif kind == "model" and name == "scan" and self.at(T.LPAREN):
            # `model scan(z) { ... }`: the IDENT above was taken as the component *name*, so this
            # never reaches `parse_scan_header`. Caught here, or it fails on `(` with a generic
            # "expected '{'" that points at the wrong thing entirely.
            self.parse_scan_header(None)
        self.expect(T.LBRACE)
        # A `functions` block holds definitions, never declarations or statements, so it gets its
        # own body parser: no decl-vs-definition lookahead, and a stray `real x;` in there earns a
        # message about what a functions block is for.
        if kind == "functions":
            body = self.parse_function_defs()
        else:
            body = []
            while not self.at(T.RBRACE):
                if self.at(T.EOF):
                    self.error("unterminated block: expected '}'")
                body.append(self.parse_decl_or_stmt())
        self.expect(T.RBRACE)
        return ast.Block(kind=kind, body=body, span=head.span, name=name,
                         scan_over=scan_over)

    def parse_scan_header(self, name) -> tuple:
        """``scan(a, b)`` after a component name --- the arrays the body is evaluated over.

        Entries are **plain declared names**, never slices or expressions. That restriction is what
        makes "element `i` of the scan is coordinate `i` of this parameter" a fact about the
        program rather than something inferred from an index expression.

        A name is required before ``scan``, and its absence gets its own message: ``model scan(z)``
        would otherwise parse ``scan`` as the *component name* and then fail on ``(`` with a
        generic "expected '{'", which points at the wrong thing entirely.
        """
        sp = self.advance().span                                   # 'scan'
        if name is None:
            raise DslError(
                "a scan component needs a name: write `model <name> scan(...) { ... }`. "
                "(Without one, `scan` reads as the component's name.)", sp, self.source)
        self.expect(T.LPAREN)
        names: list = []
        while not self.at(T.RPAREN):
            if names:
                self.expect(T.COMMA)
            tok = self.expect(T.IDENT, "the name of an array to scan over")
            if self.at(T.LBRACK):
                raise DslError(
                    f"`scan` takes plain array names, not an indexed expression like "
                    f"{tok.text}[...]. Scanning {tok.text!r} whole is what makes element `i` of "
                    f"the scan mean element `i` of {tok.text!r}.", tok.span, self.source)
            if tok.text in names:
                raise DslError(
                    f"`scan` names {tok.text!r} twice; each scanned array is bound once",
                    tok.span, self.source)
            names.append(tok.text)
        self.expect(T.RPAREN)
        if not names:
            raise DslError(
                "`scan()` needs at least one array to scan over --- it is what gives the "
                "component its per-element structure and its length", sp, self.source)
        return tuple(names)

    def parse_decl_or_stmt(self):
        if self.peek().kind is T.IDENT and self.peek().text in _TYPE_KEYWORDS:
            return self.parse_decl()
        if self._looks_like_destructuring():
            return self.parse_tuple_decl()
        return self.parse_statement()

    def _looks_like_destructuring(self) -> bool:
        """Is this `(real a, ...) = ...;` rather than an expression statement?

        Look past the opening parentheses --- a target list may nest --- to the first token that
        is not one. A type keyword there can only be a destructuring declaration, since no
        statement legally starts with a parenthesised type.
        """
        if self.peek().kind is not T.LPAREN:
            return False
        i = 0
        while self.peek(i).kind is T.LPAREN:
            i += 1
        tok = self.peek(i)
        return tok.kind is T.IDENT and tok.text in _TYPE_KEYWORDS

    def parse_tuple_decl(self) -> ast.TupleDecl:
        """``( <target> , <target> {, <target>} ) = <expr> ;`` --- destructure a tuple."""
        sp = self.peek().span
        group = self._parse_target_group()
        self.expect(T.ASSIGN)
        init = self.parse_expr()
        self.expect(T.SEMI)
        return ast.TupleDecl(targets=group.targets, init=init, span=sp)

    def _parse_target_group(self) -> ast.TupleTarget:
        """``( target, target {, target} )`` --- a parenthesised group of destructuring targets."""
        sp = self.peek().span
        self.expect(T.LPAREN)
        targets = [self._parse_decl_target()]
        while self.at(T.COMMA):
            self.advance()
            targets.append(self._parse_decl_target())
        self.expect(T.RPAREN)
        if len(targets) < 2:
            raise DslError("a destructuring group needs at least two names", sp, self.source)
        return ast.TupleTarget(targets=tuple(targets), span=sp)

    def _parse_decl_target(self):
        """One element of a destructuring declaration: `<type> <name>`, or a nested group.

        Inside a target list a `(` is always a nested group, never a tuple *type*: there are no
        tuple-typed locals, so nothing else it could be.
        """
        if self.at(T.LPAREN):
            return self._parse_target_group()
        sp = self.peek().span
        t = self.parse_type()
        if self.at(T.LT):
            self._reject_trailing_constraints(t)
        name = self._optional_name(t, "a variable name")
        return ast.VarDecl(base_type=t.base, shape=t.dims, name=name, lower=t.lower,
                           upper=t.upper, init=None, span=sp, base_args=t.base_args)

    # --- declarations ------------------------------------------------------- #

    def parse_type(self, *, allow_void: bool = False,
                   allow_unsized: bool = False) -> ast.TypeExpr:
        """A declared type: ``[array[dims]] ('real'|'int'|'unit_vector'[size])``, or ``void``.

        ``allow_void`` and ``allow_unsized`` are what separate a *declaration* (which knows its
        sizes) from a function *signature* (which does not): a signature writes ``array real x``
        or ``array[] real x``. ``void`` parses only to be rejected with a useful message by the
        semantic pass --- see :func:`mimcs.dsl.semantics.check_functions`.
        """
        sp = self.peek().span
        if self.at(T.LPAREN):                              # a tuple type: `(real, array[n] real)`
            self.advance()
            elements = [self.parse_type(allow_unsized=allow_unsized)]
            while self.at(T.COMMA):
                self.advance()
                elements.append(self.parse_type(allow_unsized=allow_unsized))
            self.expect(T.RPAREN)
            if len(elements) < 2:
                raise DslError("a tuple type needs at least two elements", sp, self.source)
            return ast.TypeExpr(base="tuple", dims=(), base_args=(), span=sp,
                                elements=tuple(elements))
        dims: tuple | None = ()
        if self.at_ident("array"):
            self.advance()
            if self.at(T.LBRACK):
                dims = self._parse_dims(allow_unsized=allow_unsized)
            elif allow_unsized:
                dims = None                                # `array real x`: rank left unsaid
            else:
                self.error("'array' needs a size: array[n] real")
        base = self.advance()                              # 'real' | 'int' | 'unit_vector'
        allowed = tuple(PARAMETER_KINDS) + (NONE,) + (("void",) if allow_void else ())
        if base.text not in allowed:
            raise DslError(
                f"expected element type {' or '.join(repr(a) for a in allowed)}, "
                f"found {base.text!r}", base.span, self.source)
        kind = PARAMETER_KINDS.get(base.text)
        # Bounds sit between the base type and its size, as in Stan: `real<lower=0>`,
        # `ordered<lower=0, upper=1>[d]`. They constrain the type, so they are parsed with it.
        lower = upper = None
        if self.at(T.LT):
            if kind is None or not kind.takes_bounds:
                self.error(f"a {base.text} cannot carry lower/upper bounds")
            lower, upper = self.parse_constraints()
        # An `array[...]` prefix sizes the array *of* elements; a size argument on the base type
        # sizes the element itself, so `array[n] unit_vector[d]` carries both. How many the base
        # type takes is the parameter kind's own business.
        base_args: tuple = ()
        if kind is not None and kind.n_base_sizes:
            if not self.at(T.LBRACK):
                self.error(f"'{base.text}' needs a size: {base.text}[d]")
            base_args = self._parse_dims()                 # always sized: a chart needs its d
            if len(base_args) != kind.n_base_sizes:
                expected = ("one size" if kind.n_base_sizes == 1
                            else f"{kind.n_base_sizes} sizes")
                raise DslError(f"{base.text} takes {expected}, found {len(base_args)}",
                               base.span, self.source)
        return ast.TypeExpr(base=base.text, dims=dims, base_args=base_args, span=sp,
                            lower=lower, upper=upper)

    def _reject_trailing_constraints(self, t: ast.TypeExpr) -> None:
        """``<...>`` *after* the size: either the type takes no bounds, or they are misplaced."""
        kind = PARAMETER_KINDS.get(t.base)
        if kind is None or not kind.takes_bounds:
            self.error(f"a {t.base} cannot carry lower/upper bounds")
        self.error(f"bounds go before the size: write `{t.base}<lower=...>[d]`, "
                   f"not `{t.base}[d]<lower=...>`")

    def parse_decl(self) -> ast.VarDecl:
        sp = self.peek().span
        t = self.parse_type()
        lower, upper = t.lower, t.upper
        if self.at(T.LT):
            self._reject_trailing_constraints(t)
        name = self.expect(T.IDENT, "a variable name").text
        init = None
        if self.at(T.ASSIGN):
            self.advance()
            init = self.parse_expr()
        self.expect(T.SEMI)
        return ast.VarDecl(base_type=t.base, shape=t.dims, name=name,
                           lower=lower, upper=upper, init=init, span=sp,
                           base_args=t.base_args)

    def _parse_dims(self, *, allow_unsized: bool = False) -> tuple:
        """``[ expr {, expr} ]`` --- the sizes shared by `array[...]` and `unit_vector[...]`.

        With ``allow_unsized`` an entry may be empty (``array[] real``, ``array[,] real``), which
        records the rank as ``None`` entries; that form is only legal in a function signature.
        """
        self.expect(T.LBRACK)
        dims = [self._parse_dim(allow_unsized)]
        while self.at(T.COMMA):
            self.advance()
            dims.append(self._parse_dim(allow_unsized))
        self.expect(T.RBRACK)
        return tuple(dims)

    def _parse_dim(self, allow_unsized: bool):
        if allow_unsized and (self.at(T.RBRACK) or self.at(T.COMMA)):
            return None
        if allow_unsized:
            self.error("a function argument has no declared size: write 'array real x' "
                       "or 'array[] real x'")
        return self.parse_expr()

    def parse_constraints(self):
        self.expect(T.LT)
        lower = upper = None
        while True:
            key = self.expect(T.IDENT, "'lower' or 'upper'").text
            self.expect(T.ASSIGN)
            val = self.parse_expr(11)            # above comparison bp, so '>' closes <...>
            if key == "lower":
                lower = val
            elif key == "upper":
                upper = val
            else:
                self.error(f"unknown constraint {key!r} (expected 'lower' or 'upper')")
            if self.at(T.COMMA):
                self.advance()
                continue
            break
        self.expect(T.GT)
        return lower, upper

    # --- user-defined functions --------------------------------------------- #

    def parse_function_defs(self) -> list:
        """The body of a ``functions`` block: definitions, and nothing else."""
        defs = []
        while not self.at(T.RBRACE):
            if self.at(T.EOF):
                self.error("unterminated block: expected '}'")
            defs.append(self.parse_funcdef())
        return defs

    def parse_funcdef(self) -> ast.FuncDef:
        """``type NAME ( [param {, param}] ) { statements }``."""
        sp = self.peek().span
        ret = self.parse_type(allow_void=True, allow_unsized=True)
        name = self.expect(T.IDENT, "a function name").text
        if not self.at(T.LPAREN):
            self.error("the `functions` block holds function definitions only "
                       "(expected '(' after the function name)")
        self.advance()
        params = []
        if not self.at(T.RPAREN):
            params.append(self.parse_param())
            while self.at(T.COMMA):
                self.advance()
                params.append(self.parse_param())
        self.expect(T.RPAREN)
        self.expect(T.LBRACE)
        body = []
        while not self.at(T.RBRACE):
            if self.at(T.EOF):
                self.error(f"unterminated body of function {name!r}: expected '}}'")
            body.append(self.parse_decl_or_stmt())
        self.expect(T.RBRACE)
        return ast.FuncDef(name=name, return_type=ret, params=params, body=body, span=sp)

    def parse_param(self) -> ast.Param:
        sp = self.peek().span
        t = self.parse_type(allow_unsized=True)
        if self.at(T.LT) or t.lower is not None or t.upper is not None:
            self.error("a function argument cannot carry lower/upper bounds")
        return ast.Param(type=t, name=self._optional_name(t, "an argument name"), span=sp)

    def _optional_name(self, t: ast.TypeExpr, what: str) -> str | None:
        """The declared name, which a `None` may omit --- there is nothing to refer to.

        `f(None, real x)` reads better than inventing a name for a value that is always the
        empty pytree, and a name is still allowed for anyone who wants one.
        """
        if t.base == NONE and not self.at(T.IDENT):
            return None
        return self.expect(T.IDENT, what).text

    # --- statements --------------------------------------------------------- #

    def parse_statement(self) -> ast.Stmt:
        # `target += ...` or `<component> += ...`. Any IDENT is accepted here and the name
        # recorded; `semantics.check_target_names` decides which are legal, because only it knows
        # the enclosing component.
        if self.at(T.IDENT) and self.peek(1).kind is T.PLUSEQ:
            tok = self.advance()
            self.advance()                                          # '+='
            value = self.parse_expr()
            self.expect(T.SEMI)
            return ast.TargetPlus(value=value, span=tok.span,
                                  name=None if tok.text == "target" else tok.text)
        if self.at_ident("return"):
            sp = self.advance().span
            value = None if self.at(T.SEMI) else self.parse_expr()
            self.expect(T.SEMI)
            return ast.Return(value=value, span=sp)
        if self.at_ident("for"):
            return self.parse_for()
        if self.at_ident("while"):
            return self.parse_while()
        if self.at_ident("if"):
            return self.parse_if()

        lhs = self.parse_expr()
        if self.at(T.ASSIGN):
            sp = self.advance().span
            value = self.parse_expr()
            self.expect(T.SEMI)
            return ast.Assign(target=lhs, value=value, span=sp)
        if self.at(T.TILDE):
            sp = self.advance().span
            dist = self.expect(T.IDENT, "a distribution name").text
            self.expect(T.LPAREN)
            args = self.parse_arglist()
            self.expect(T.RPAREN)
            self.expect(T.SEMI)
            return ast.Sample(lhs=lhs, dist=dist, args=args, span=sp)
        self.error("expected '=' (assignment) or '~' (sampling statement)")

    def parse_block_or_stmt(self) -> list[ast.Stmt]:
        if self.at(T.LBRACE):
            self.advance()
            body = []
            while not self.at(T.RBRACE):
                if self.at(T.EOF):
                    self.error("unterminated block: expected '}'")
                body.append(self.parse_decl_or_stmt())
            self.expect(T.RBRACE)
            return body
        return [self.parse_statement()]

    def parse_for(self) -> ast.For:
        sp = self.advance().span                                    # 'for'
        self.expect(T.LPAREN)
        var = self.expect(T.IDENT, "a loop variable").text
        self.expect_ident("in")
        lo = self.parse_expr()
        self.expect(T.COLON)
        hi = self.parse_expr()
        self.expect(T.RPAREN)
        body = self.parse_block_or_stmt()
        return ast.For(var=var, lo=lo, hi=hi, body=body, span=sp)

    def parse_while(self) -> ast.While:
        sp = self.advance().span
        self.expect(T.LPAREN)
        cond = self.parse_expr()
        self.expect(T.RPAREN)
        return ast.While(cond=cond, body=self.parse_block_or_stmt(), span=sp)

    def parse_if(self) -> ast.If:
        sp = self.advance().span
        self.expect(T.LPAREN)
        cond = self.parse_expr()
        self.expect(T.RPAREN)
        then_body = self.parse_block_or_stmt()
        else_body: list = []
        if self.at_ident("else"):
            self.advance()
            else_body = self.parse_block_or_stmt()
        return ast.If(cond=cond, then_body=then_body, else_body=else_body, span=sp)

    # --- expressions (Pratt) ------------------------------------------------ #

    def parse_expr(self, min_bp: int = 0) -> ast.Expr:
        left = self.parse_prefix()
        while True:
            tok = self.peek()
            if tok.kind in (T.LBRACK, T.LPAREN, T.TRANSPOSE) and _POSTFIX_BP >= min_bp:
                left = self.parse_postfix(left)
                continue
            bp = _INFIX.get(tok.kind)
            if bp is None or bp[0] < min_bp:
                break
            self.advance()
            right = self.parse_expr(bp[1])
            left = ast.BinOp(op=tok.text, lhs=left, rhs=right, span=tok.span)
        return left

    def parse_prefix(self) -> ast.Expr:
        tok = self.peek()
        if tok.kind in (T.MINUS, T.PLUS):
            self.advance()
            return ast.UnaryOp(op=tok.text, operand=self.parse_expr(45), span=tok.span)
        if tok.kind is T.INT:
            self.advance()
            return ast.IntLit(value=int(tok.text), span=tok.span)
        if tok.kind is T.REAL:
            self.advance()
            return ast.RealLit(value=float(tok.text), span=tok.span)
        if tok.kind is T.IDENT and tok.text == NONE:
            self.advance()
            return ast.NoneLit(span=tok.span)
        if tok.kind is T.IDENT:
            self.advance()
            return ast.Name(id=tok.text, span=tok.span)
        if tok.kind is T.LPAREN:
            self.advance()
            inner = self.parse_expr()
            if not self.at(T.COMMA):                       # plain grouping; there is no 1-tuple
                self.expect(T.RPAREN)
                return inner
            elements = [inner]
            while self.at(T.COMMA):
                self.advance()
                elements.append(self.parse_expr())
            self.expect(T.RPAREN)
            return ast.TupleLit(elements=tuple(elements), span=tok.span)
        self.error(f"unexpected {tok.text or 'end of input'!r} in expression")

    def parse_postfix(self, left: ast.Expr) -> ast.Expr:
        tok = self.peek()
        if tok.kind is T.TRANSPOSE:
            self.advance()
            return ast.Transpose(operand=left, span=tok.span)
        if tok.kind is T.LPAREN:                                    # call: f(args)
            if not isinstance(left, ast.Name):
                raise DslError("only a named function can be called", tok.span, self.source)
            self.advance()
            args = self.parse_arglist()
            self.expect(T.RPAREN)
            return ast.Call(fn=left.id, args=self._mark_function_arg(left.id, args, left.span),
                            span=left.span)
        # indexing: a[ idx (, idx)* ]
        self.advance()                                             # '['
        idx = [self.parse_index_arg()]
        while self.at(T.COMMA):
            self.advance()
            idx.append(self.parse_index_arg())
        self.expect(T.RBRACK)
        return ast.Index(base=left, args=idx, span=tok.span)

    def _mark_function_arg(self, callee: str, args: list, span) -> list:
        """In a loop form's function slot, rewrite the bare name into an :class:`ast.FuncRef`.

        Doing it here, once, is what keeps every later walk simple: a function's *name* is not a
        value, so leaving it as an :class:`ast.Name` would make the interpreter look it up in the
        environment and the function-scope check report it as unknown.
        """
        form = LOOP_FORMS.get(callee)
        if form is None or len(args) <= form.fn_arg:
            return args
        slot = args[form.fn_arg]
        if not isinstance(slot, ast.Name):
            raise DslError(
                f"argument {form.fn_arg + 1} of `{callee}` must name a function --- "
                f"write `{form.signature}`", getattr(slot, "span", span), self.source)
        args = list(args)
        args[form.fn_arg] = ast.FuncRef(name=slot.id, span=slot.span)
        return args

    def parse_index_arg(self):
        if self.at(T.COLON):                                       # [:hi] or [:]
            self.advance()
            hi = None if self.at(T.RBRACK) or self.at(T.COMMA) else self.parse_expr()
            return ast.Range(lo=None, hi=hi)
        e = self.parse_expr()
        if self.at(T.COLON):                                       # [lo:hi] or [lo:]
            if isinstance(e, ast.NoneLit):
                raise DslError("`None` cannot be a slice bound; it inserts an axis, as in "
                               "`a[:, None]`", e.span, self.source)
            self.advance()
            hi = None if self.at(T.RBRACK) or self.at(T.COMMA) else self.parse_expr()
            return ast.Range(lo=e, hi=hi)
        if isinstance(e, ast.NoneLit):                             # `a[:, None]`: a new axis
            return ast.NewAxis(span=e.span)
        return ast.ScalarIndex(expr=e)

    def parse_arglist(self) -> list[ast.Expr]:
        args: list[ast.Expr] = []
        if self.at(T.RPAREN):
            return args
        args.append(self.parse_expr())
        while self.at(T.COMMA):
            self.advance()
            args.append(self.parse_expr())
        return args


def parse(source: str) -> ast.Program:
    log.debug("parsing DSL source (%d chars, %d lines)", len(source), source.count("\n") + 1)
    try:
        program = Parser(source).parse_program()
    except DslError as e:
        raise log_compile_error(e if e.source is not None else e.with_source(source))
    log.debug("parsed %d block(s): %s", len(program.blocks),
              ", ".join(b.kind if b.name is None else f"{b.kind} {b.name}"
                        for b in program.blocks) or "(none)")
    return program
