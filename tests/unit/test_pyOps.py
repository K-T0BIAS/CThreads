"""Unit tests for cthreads.pyOps operator tables + builtin whitelist."""

import ast

from cthreads.pyOps import BINOPS, BOOLOPS, BUILTINS, CMPOPS, UNARYOPS, is_builtin_call
from helpers import parse_expr


def test_binops_cover_common_ops():
    assert BINOPS[ast.Add] == "+"
    assert BINOPS[ast.Mult] == "*"
    assert BINOPS[ast.Mod] == "%"


def test_unary_compare_bool_ops():
    assert UNARYOPS[ast.USub] == "-"
    assert UNARYOPS[ast.Not] == "!"
    assert CMPOPS[ast.Lt] == "<"
    assert CMPOPS[ast.Eq] == "=="
    assert BOOLOPS[ast.And] == "&&"
    assert BOOLOPS[ast.Or] == "||"


def test_tables_are_typed_dicts():
    assert all(isinstance(k, type) for k in BINOPS)
    assert all(isinstance(v, str) for v in BINOPS.values())


def test_builtins_whitelist():
    assert BUILTINS == frozenset({"range", "len"})
    assert is_builtin_call(parse_expr("range(n)"), "range")
    assert is_builtin_call(parse_expr("len(xs)"), "len")
    assert not is_builtin_call(parse_expr("range(n)"), "len")
    assert not is_builtin_call(parse_expr("len(xs)"), "range")
    assert not is_builtin_call(parse_expr("math.sqrt(x)"), "len")
    assert not is_builtin_call(parse_expr("foo(xs)"), "len")
    assert not is_builtin_call(parse_expr("1 + 2"), "len")
