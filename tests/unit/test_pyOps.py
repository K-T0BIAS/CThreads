"""Unit tests for operator tables + builtin whitelist."""

import ast

from cthreads.compiler.translation.syntax.Op import Op
from helpers import parse_expr


def test_tables_cover_common_ops():
    assert Op.BINOPS[ast.Add] == "+"
    assert Op.BINOPS[ast.Mult] == "*"
    assert Op.UNARYOPS[ast.USub] == "-"
    assert Op.UNARYOPS[ast.Not] == "!"
    assert Op.CMPOPS[ast.Lt] == "<"
    assert Op.BOOLOPS[ast.And] == "&&"
    assert "range" in Op.BUILTINS
    assert "len" in Op.BUILTINS
    assert "__sync_state" in Op.BUILTINS


def test_is_builtin_call():
    assert Op.is_builtin_call(parse_expr("len(xs)"), "len")
    assert Op.is_builtin_call(parse_expr("range(3)"), "range")
    assert not Op.is_builtin_call(parse_expr("len(xs)"), "range")
    assert not Op.is_builtin_call(parse_expr("foo.len(xs)"), "len")
