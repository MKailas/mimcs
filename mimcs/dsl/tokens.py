"""Tokens and source spans for the model DSL (``docs/design/08_model_dsl.md``)."""

from __future__ import annotations

import enum
from dataclasses import dataclass


@dataclass(frozen=True)
class SourceSpan:
    """A half-open source region, for diagnostics (1-based line, 1-based column)."""

    line: int
    col: int

    def __str__(self) -> str:
        return f"{self.line}:{self.col}"


class TokenKind(enum.Enum):
    # literals / names
    IDENT = "identifier"
    INT = "int literal"
    REAL = "real literal"
    # delimiters
    LPAREN = "("
    RPAREN = ")"
    LBRACK = "["
    RBRACK = "]"
    LBRACE = "{"
    RBRACE = "}"
    COMMA = ","
    SEMI = ";"
    COLON = ":"
    # operators
    PLUS = "+"
    MINUS = "-"
    STAR = "*"
    SLASH = "/"
    CARET = "^"
    DOTPLUS = ".+"
    DOTMINUS = ".-"
    DOTSTAR = ".*"
    DOTSLASH = "./"
    DOTCARET = ".^"
    TRANSPOSE = "'"
    ASSIGN = "="
    PLUSEQ = "+="
    TILDE = "~"
    LT = "<"
    GT = ">"
    LE = "<="
    GE = ">="
    EQEQ = "=="
    NE = "!="
    EOF = "end of input"


@dataclass(frozen=True)
class Token:
    kind: TokenKind
    text: str
    span: SourceSpan
