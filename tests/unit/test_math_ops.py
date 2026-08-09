"""Unit tests for math → std:: lowering (table + resolve + emit)."""

from __future__ import annotations

import math

import pytest

from cthreads.pyTypes import PyFloat, PyInt
from cthreads.Thread.compile.AstTranslators import attribute, binOp, call
from cthreads.Thread.compile.mathLibTranslators import (
    CMATH_INCLUDE,
    MATHCONSTS,
    MATHOPS,
    NUMBERS_INCLUDE,
    MathOp,
    is_math,
    resolve_math_call,
    resolve_math_const,
)
from cthreads.Thread.compile.mathLibTranslators.mathOps import MATHOPS as MATHOPS_TABLE
from helpers import make_ctx, parse_expr


def _math_ctx(**symbols):
    syms = {"x": PyFloat(), "y": PyFloat(), "z": PyFloat(), "i": PyInt(), **symbols}
    return make_ctx(symbols=syms, globals_extra={"math": math, "sqrt": math.sqrt})


def test_mathops_table_shape():
    assert MATHOPS is MATHOPS_TABLE
    assert CMATH_INCLUDE.strip() == "#include <cmath>"
    assert NUMBERS_INCLUDE.strip() == "#include <numbers>"
    for name, op in MATHOPS.items():
        assert isinstance(op, MathOp)
        assert hasattr(math, name), f"math.{name} missing"
        assert op.cpp_func.startswith("std::"), name
        assert op.arity >= 1
        assert isinstance(op.arity, int)


def test_version_gated_entries_match_hasattr():
    from cthreads.Thread.compile.mathLibTranslators.mathOps import _ALL_MATHOPS

    for name in _ALL_MATHOPS:
        if hasattr(math, name):
            assert name in MATHOPS
        else:
            assert name not in MATHOPS


def test_gamma_maps_to_tgamma():
    assert MATHOPS["gamma"].cpp_func == "std::tgamma"
    assert MATHOPS["gamma"].arity == 1


@pytest.mark.parametrize("name,op", list(MATHOPS.items()))
def test_resolve_math_call_attribute_form(name, op):
    args = ", ".join(["x", "y", "z"][: op.arity])
    node = parse_expr(f"math.{name}({args})")
    ctx = _math_ctx()
    assert is_math(node, ctx)
    assert resolve_math_call(node, ctx) == op


@pytest.mark.parametrize("name,op", list(MATHOPS.items()))
def test_call_translate_emits_std(name, op):
    args = ", ".join(["x", "y", "z"][: op.arity])
    node = parse_expr(f"math.{name}({args})")
    ctx = _math_ctx()
    got = call.translate(node, ctx)
    expected_args = ", ".join(["x", "y", "z"][: op.arity])
    assert got == f"{op.cpp_func}({expected_args})"
    assert CMATH_INCLUDE in ctx.body_includes or any(
        "cmath" in line for line in ctx.body_includes
    )


def test_resolve_from_import_style():
    ctx = _math_ctx()
    node = parse_expr("sqrt(x)")
    assert resolve_math_call(node, ctx) == MATHOPS["sqrt"]
    assert call.translate(node, ctx) == "std::sqrt(x)"


def test_resolve_rejects_wrong_arity():
    ctx = _math_ctx()
    assert resolve_math_call(parse_expr("math.pow(x)"), ctx) is None
    assert resolve_math_call(parse_expr("math.sqrt(x, y)"), ctx) is None
    assert resolve_math_call(parse_expr("math.fma(x, y)"), ctx) is None
    assert resolve_math_call(parse_expr("math.log(x, y)"), ctx) is None


def test_resolve_rejects_keywords_and_unknown():
    ctx = _math_ctx()
    assert resolve_math_call(parse_expr("math.sqrt(x=1)"), ctx) is None
    assert resolve_math_call(parse_expr("math.comb(5, 2)"), ctx) is None
    assert resolve_math_call(parse_expr("nope(x)"), ctx) is None


def test_resolve_rejects_non_math_module():
    class Fake:
        __name__ = "fake"

        @staticmethod
        def sqrt(x):
            return x

    Fake.sqrt.__module__ = "other"
    Fake.sqrt.__name__ = "sqrt"
    ctx = make_ctx(
        symbols={"x": PyFloat()},
        globals_extra={"fake": Fake},
    )
    assert resolve_math_call(parse_expr("fake.sqrt(x)"), ctx) is None


@pytest.mark.parametrize("name,cpp", list(MATHCONSTS.items()))
def test_resolve_math_consts(name, cpp):
    ctx = _math_ctx()
    node = parse_expr(f"math.{name}")
    assert resolve_math_const(node, ctx) == cpp
    assert attribute.translate(node, ctx) == cpp
    assert any("numbers" in line for line in ctx.body_includes)


def test_pow_binop_and_augassign():
    from cthreads.Thread.compile.AstTranslators import augAssign
    from helpers import parse_stmt

    ctx = _math_ctx()
    assert binOp.translate(parse_expr("x ** y"), ctx) == "std::pow(x, y)"
    lines = augAssign.translate(parse_stmt("x **= y"), ctx)
    assert lines == ["    x = std::pow(x, y);"]
