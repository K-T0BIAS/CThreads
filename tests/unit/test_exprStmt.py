"""Unit tests for exprStmt translator."""

from cthreads.Thread.compile.AstTranslators import exprStmt
from helpers import make_ctx, parse_stmt


def test_expr_docstring_ignored():
    ctx = make_ctx()
    # Module-level string as Expr stmt
    node = parse_stmt('"docstring"')
    assert exprStmt.translate(node, ctx) == []


def test_expr_other_unsupported():
    ctx = make_ctx(symbols={})
    node = parse_stmt("1 + 2")
    lines = exprStmt.translate(node, ctx)
    assert "unsupported statement: Expr" in lines[0]
