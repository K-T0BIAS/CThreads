"""Unit tests for AST expression translation (Syntax.expr)."""

import pytest

from cthreads.compiler.translation.syntax import Syntax
from cthreads.types import PyFloat, PyInt, PyList, PyThreadable
from helpers import make_ctx, parse_expr


def test_constant_literals():
    ctx = make_ctx()
    assert Syntax.expr(parse_expr("10"), ctx) == "10"
    assert Syntax.expr(parse_expr("True"), ctx) == "true"
    assert Syntax.expr(parse_expr('"hi"'), ctx) == '"hi"'


def test_name_known_unknown_and_self():
    ctx = make_ctx(symbols={"a": PyInt()})
    assert Syntax.expr(parse_expr("a"), ctx) == "a"
    with pytest.raises(TypeError, match="unknown name"):
        Syntax.expr(parse_expr("missing"), ctx)

    mctx = make_ctx(owner_name="Particle", symbols={"self": PyThreadable("Particle")})
    assert Syntax.expr(parse_expr("self"), mctx) == "(*this)"


def test_attribute_self_and_plain():
    ctx = make_ctx(
        owner_name="Particle",
        symbols={
            "self": PyThreadable("Particle"),
            "p": PyThreadable("Particle"),
        },
    )
    assert Syntax.expr(parse_expr("self.x"), ctx) == "this->x"
    assert Syntax.expr(parse_expr("p.velocity"), ctx) == "p.velocity"


def test_binop_and_pow():
    ctx = make_ctx(symbols={"a": PyInt(), "b": PyInt()})
    assert Syntax.expr(parse_expr("a + b"), ctx) == "(a + b)"
    assert Syntax.expr(parse_expr("a * b"), ctx) == "(a * b)"
    assert Syntax.expr(parse_expr("a ** b"), ctx) == "std::pow(a, b)"
    assert any("cmath" in line for line in ctx.body_includes)


def test_math_call_and_const():
    import math

    ctx = make_ctx(
        symbols={"x": PyFloat(), "y": PyFloat()},
        globals_extra={"math": math, "sqrt": math.sqrt},
    )
    assert Syntax.expr(parse_expr("math.sqrt(x)"), ctx) == "std::sqrt(x)"
    assert Syntax.expr(parse_expr("sqrt(x)"), ctx) == "std::sqrt(x)"
    assert Syntax.expr(parse_expr("math.atan2(y, x)"), ctx) == "std::atan2(y, x)"
    assert Syntax.expr(parse_expr("math.pi"), ctx) == "std::numbers::pi"
    with pytest.raises(TypeError, match="expects"):
        Syntax.expr(parse_expr("math.pow(x)"), ctx)
    with pytest.raises(TypeError, match="unsupported call"):
        Syntax.expr(parse_expr("unknown(x)"), ctx)


def test_threadable_method_call_lowers_to_cpp_member():
    ctx = make_ctx(
        owner_name="Particle",
        symbols={
            "self": PyThreadable("Particle"),
            "p": PyThreadable("Particle"),
            "ps": PyList(PyThreadable("Particle")),
            "dt": PyFloat(),
            "i": PyInt(),
        },
    )
    assert Syntax.expr(parse_expr("p.step(dt)"), ctx) == "(p).step(dt)"
    assert Syntax.expr(parse_expr("self.step(dt)"), ctx) == "this->step(dt)"
    assert Syntax.expr(parse_expr("ps[i].step(dt)"), ctx) == "((ps[i])).step(dt)"
    assert Syntax.expr(parse_expr("p.reset()"), ctx) == "(p).reset()"
    with pytest.raises(TypeError, match="keyword"):
        Syntax.expr(parse_expr("p.step(dt=1.0)"), ctx)


def test_builtin_len_call():
    ctx = make_ctx(symbols={"xs": PyList(PyInt()), "n": PyInt()})
    assert Syntax.expr(parse_expr("len(xs)"), ctx) == "(xs).size()"
    with pytest.raises(TypeError, match="len\\(\\) expects 1 arg"):
        Syntax.expr(parse_expr("len()"), ctx)
    with pytest.raises(TypeError, match="len\\(\\) expects 1 arg"):
        Syntax.expr(parse_expr("len(xs, n)"), ctx)


def test_builtin_sync_state_call():
    ctx = make_ctx()
    assert (
        Syntax.expr(parse_expr("__sync_state()"), ctx)
        == "cthreads::detail::__sync_state()"
    )
    assert any("sync/syncState.hpp" in line for line in ctx.body_includes)
    with pytest.raises(TypeError, match="__sync_state"):
        Syntax.expr(parse_expr("__sync_state(1)"), ctx)


def test_unaryop():
    ctx = make_ctx(symbols={"a": PyInt(), "f": PyFloat()})
    assert Syntax.expr(parse_expr("-a"), ctx) == "(-a)"
    assert Syntax.expr(parse_expr("not f"), ctx) == "(!f)"


def test_compare_simple_and_chained():
    ctx = make_ctx(symbols={"a": PyInt(), "b": PyInt(), "c": PyInt()})
    assert Syntax.expr(parse_expr("a < b"), ctx) == "(a < b)"
    assert Syntax.expr(parse_expr("a < b < c"), ctx) == "((a < b) && (b < c))"


def test_boolop_and_or_and_errors():
    ctx = make_ctx(symbols={"a": PyInt(), "b": PyInt(), "c": PyInt()})
    assert Syntax.expr(parse_expr("a and b"), ctx) == "(a && b)"
    assert Syntax.expr(parse_expr("a or b or c"), ctx) == "(a || b || c)"


def test_translate_expr_dispatch_and_unsupported():
    ctx = make_ctx(symbols={"a": PyInt()})
    assert Syntax.expr(parse_expr("a + 1"), ctx) == "(a + 1)"
    with pytest.raises(TypeError, match="unsupported expression"):
        Syntax.expr(parse_expr("{1: 2}"), ctx)


def test_list_literal():
    ctx = make_ctx(symbols={"x": PyInt(), "y": PyInt()})
    assert Syntax.expr(parse_expr("[1, 2, 3]"), ctx) == (
        "std::vector<int>{1, 2, 3}"
    )
    assert Syntax.expr(parse_expr("[x, y]"), ctx) == (
        "std::vector<int>{x, y}"
    )
    assert Syntax.expr(parse_expr("[]"), ctx) == "{}"
    assert Syntax.expr(parse_expr("[[1, 2], [3, 4]]"), ctx) == (
        "std::vector<std::vector<int>>{std::vector<int>{1, 2}, std::vector<int>{3, 4}}"
    )
    assert any("vector" in line for line in ctx.body_includes)

    sctx = make_ctx()
    assert Syntax.expr(parse_expr('["a", "b"]'), sctx) == (
        'std::vector<std::string>{"a", "b"}'
    )
    assert any("string" in line for line in sctx.body_includes)

    with pytest.raises(TypeError, match="mixed list element types"):
        Syntax.expr(parse_expr("[1, 2.0]"), ctx)
    with pytest.raises(TypeError, match="starred"):
        Syntax.expr(parse_expr("[*x]"), ctx)
    with pytest.raises(TypeError, match="cannot infer"):
        Syntax.expr(parse_expr("[x + y]"), ctx)

    assert Syntax.expr(parse_expr("[[1], []]"), ctx) == (
        "std::vector<std::vector<int>>{std::vector<int>{1}, {}}"
    )
