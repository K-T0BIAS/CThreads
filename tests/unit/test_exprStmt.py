"""Unit tests for exprStmt lowering."""

from cthreads.compiler.translation.syntax import Syntax
from helpers import make_ctx, parse_stmt


def test_expr_docstring_ignored():
    ctx = make_ctx()
    node = parse_stmt('"docstring"')
    assert Syntax.stmt(node, ctx) == []


def test_expr_other_unsupported():
    ctx = make_ctx(symbols={})
    node = parse_stmt("1 + 2")
    lines = Syntax.stmt(node, ctx)
    assert "unsupported statement: Expr" in lines[0]
