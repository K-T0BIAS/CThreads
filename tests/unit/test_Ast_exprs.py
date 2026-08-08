"""Unit tests for AST expression translators."""

import ast

import pytest

from cthreads.pyTypes import PyFloat, PyInt, PyThreadable
from cthreads.Thread.compile.AstTranslators import (
    attribute,
    binOp,
    boolOp,
    compare,
    constant,
    name,
    unaryOp,
)
from cthreads.Thread.compile.AstTranslators.translate import translate_expr
from helpers import make_ctx, parse_expr


def test_constant_literals():
    ctx = make_ctx()
    assert constant.translate(parse_expr("10"), ctx) == "10"
    assert constant.translate(parse_expr("True"), ctx) == "true"
    assert constant.translate(parse_expr('"hi"'), ctx) == '"hi"'


def test_name_known_unknown_and_self():
    ctx = make_ctx(symbols={"a": PyInt()})
    assert name.translate(parse_expr("a"), ctx) == "a"
    with pytest.raises(TypeError, match="unknown name"):
        name.translate(parse_expr("missing"), ctx)

    mctx = make_ctx(owner_name="Particle", symbols={"self": PyThreadable("Particle", "x")})
    assert name.translate(parse_expr("self"), mctx) == "(*this)"


def test_attribute_self_and_plain():
    ctx = make_ctx(
        owner_name="Particle",
        symbols={
            "self": PyThreadable("Particle", "x"),
            "p": PyThreadable("Particle", "x"),
        },
    )
    assert attribute.translate(parse_expr("self.x"), ctx) == "this->x"
    assert attribute.translate(parse_expr("p.velocity"), ctx) == "p.velocity"


def test_binop_and_unsupported():
    ctx = make_ctx(symbols={"a": PyInt(), "b": PyInt()})
    assert binOp.translate(parse_expr("a + b"), ctx) == "(a + b)"
    assert binOp.translate(parse_expr("a * b"), ctx) == "(a * b)"
    with pytest.raises(TypeError, match="unsupported binary"):
        binOp.translate(parse_expr("a ** b"), ctx)


def test_unaryop():
    ctx = make_ctx(symbols={"a": PyInt(), "f": PyFloat()})
    assert unaryOp.translate(parse_expr("-a"), ctx) == "(-a)"
    assert unaryOp.translate(parse_expr("not f"), ctx) == "(!f)"


def test_compare_simple_and_chained():
    ctx = make_ctx(symbols={"a": PyInt(), "b": PyInt(), "c": PyInt()})
    assert compare.translate(parse_expr("a < b"), ctx) == "(a < b)"
    assert compare.translate(parse_expr("a < b < c"), ctx) == "((a < b) && (b < c))"


def test_boolop_and_or_and_errors():
    ctx = make_ctx(symbols={"a": PyInt(), "b": PyInt(), "c": PyInt()})
    assert boolOp.translate(parse_expr("a and b"), ctx) == "(a && b)"
    assert boolOp.translate(parse_expr("a or b or c"), ctx) == "(a || b || c)"


def test_translate_expr_dispatch_and_unsupported():
    ctx = make_ctx(symbols={"a": PyInt()})
    assert translate_expr(parse_expr("a + 1"), ctx) == "(a + 1)"
    with pytest.raises(TypeError, match="unsupported expression"):
        translate_expr(parse_expr("[1, 2]"), ctx)
