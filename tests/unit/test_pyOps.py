"""Unit tests for cthreads.pyOps operator tables."""

import ast

from cthreads.pyOps import BINOPS, BOOLOPS, CMPOPS, UNARYOPS


def test_binops_cover_common_ops():
    assert BINOPS[ast.Add] == "+"
    assert BINOPS[ast.Mult] == "*"
    assert BINOPS[ast.Mod] == "%"
    assert BINOPS[ast.Pow] == ""  # intentionally unsupported token


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
