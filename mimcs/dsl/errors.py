"""Span-aware compile errors for the model DSL."""

from __future__ import annotations

from .tokens import SourceSpan
from .._logging import get_logger

log = get_logger(__name__)


class DslError(Exception):
    """A compile error carrying a source location and (when available) a caret line.

    Raised by the lexer / parser / semantic analysis / interpreter-build with a
    ``SourceSpan`` so the message points at ``line:col`` in the source.
    """

    def __init__(self, message: str, span: SourceSpan | None = None, source: str | None = None):
        self.message = message
        self.span = span
        self.source = source
        super().__init__(self._format())

    def _format(self) -> str:
        if self.span is None:
            return self.message
        head = f"{self.span}: {self.message}"
        if self.source is None:
            return head
        lines = self.source.splitlines()
        if not (1 <= self.span.line <= len(lines)):
            return head
        src_line = lines[self.span.line - 1]
        caret = " " * (self.span.col - 1) + "^"
        return f"{head}\n  {src_line}\n  {caret}"

    def with_source(self, source: str) -> "DslError":
        """Attach the full source so the message can render a caret line."""
        enriched = DslError(self.message, self.span, source)
        enriched.logged = self.logged        # do not report the same error twice
        return enriched

    #: set by :func:`log_compile_error` once the error has been reported.
    logged = False


def log_compile_error(error: DslError, logger=None) -> DslError:
    """Log a compile error at ERROR level, once, and return it (to be raised by the caller).

    Every stage of the compiler funnels its failures through here --- ``raise
    log_compile_error(err)`` --- so a compilation error is *reported* before it is raised, whether
    or not the caller catches it. The one-shot ``logged`` flag keeps an error that passes several
    stages on its way out (lexer -> parser -> ``compile_model``) to a single record; a
    :meth:`DslError.with_source` copy inherits the flag, since it is the same error with a better
    message.
    """
    if not error.logged:
        error.logged = True
        (logger or log).error("DSL compilation failed: %s", error)
    return error
