"""Hand-written lexer for the model DSL: source text -> list of tokens.

Whitespace and ``//`` / ``/* */`` comments are skipped; spans are preserved. Block headers
that are two words in Stan (``transformed data``, ``transformed parameters``,
``generated quantities``) are lexed as two identifier tokens and recombined by the parser.
A leading-digit rule for numbers keeps ``.`` unambiguous: a ``.`` starting a token is always
the prefix of an elementwise operator (``.+ .- .* ./ .^``).
"""

from __future__ import annotations

from .errors import DslError
from .tokens import SourceSpan, Token, TokenKind as T
from .._logging import get_logger

log = get_logger(__name__)

_TWO_CHAR = {
    "+=": T.PLUSEQ, "<=": T.LE, ">=": T.GE, "==": T.EQEQ, "!=": T.NE,
    ".+": T.DOTPLUS, ".-": T.DOTMINUS, ".*": T.DOTSTAR, "./": T.DOTSLASH, ".^": T.DOTCARET,
}
_ONE_CHAR = {
    "+": T.PLUS, "-": T.MINUS, "*": T.STAR, "/": T.SLASH, "^": T.CARET,
    "=": T.ASSIGN, "~": T.TILDE, "'": T.TRANSPOSE, "<": T.LT, ">": T.GT,
    "(": T.LPAREN, ")": T.RPAREN, "[": T.LBRACK, "]": T.RBRACK,
    "{": T.LBRACE, "}": T.RBRACE, ",": T.COMMA, ";": T.SEMI, ":": T.COLON,
}


def tokenize(source: str) -> list[Token]:
    toks: list[Token] = []
    i, line, col, n = 0, 1, 1, len(source)

    def span() -> SourceSpan:
        return SourceSpan(line, col)

    def advance(k: int = 1):
        nonlocal i, line, col
        for _ in range(k):
            if i < n and source[i] == "\n":
                line += 1
                col = 1
            else:
                col += 1
            i += 1

    while i < n:
        c = source[i]
        if c in " \t\r\n":
            advance()
            continue
        if c == "/" and i + 1 < n and source[i + 1] == "/":          # line comment
            while i < n and source[i] != "\n":
                advance()
            continue
        if c == "/" and i + 1 < n and source[i + 1] == "*":          # block comment
            start = span()
            advance(2)
            while i < n and not (source[i] == "*" and i + 1 < n and source[i + 1] == "/"):
                advance()
            if i >= n:
                raise DslError("unterminated /* comment", start, source)
            advance(2)
            continue

        if c.isalpha() or c == "_":                                  # identifier / keyword
            sp, j = span(), i
            while i < n and (source[i].isalnum() or source[i] == "_"):
                advance()
            toks.append(Token(T.IDENT, source[j:i], sp))
            continue

        if c.isdigit():                                              # number
            sp, j = span(), i
            while i < n and source[i].isdigit():
                advance()
            is_real = False
            if i < n and source[i] == ".":
                is_real = True
                advance()
                while i < n and source[i].isdigit():
                    advance()
            if i < n and source[i] in "eE":
                is_real = True
                advance()
                if i < n and source[i] in "+-":
                    advance()
                if not (i < n and source[i].isdigit()):
                    raise DslError("malformed exponent in number", sp, source)
                while i < n and source[i].isdigit():
                    advance()
            toks.append(Token(T.REAL if is_real else T.INT, source[j:i], sp))
            continue

        two = source[i:i + 2]
        if two in _TWO_CHAR:
            sp = span()
            advance(2)
            toks.append(Token(_TWO_CHAR[two], two, sp))
            continue
        if c in _ONE_CHAR:
            sp = span()
            advance()
            toks.append(Token(_ONE_CHAR[c], c, sp))
            continue

        raise DslError(f"unexpected character {c!r}", span(), source)

    toks.append(Token(T.EOF, "", SourceSpan(line, col)))
    log.debug("tokenized %d character(s) into %d token(s)", n, len(toks))
    return toks
