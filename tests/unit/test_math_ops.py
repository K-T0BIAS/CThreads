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


# --- cthreads.math (internal) -------------------------------------------------

from types import SimpleNamespace

from cthreads.Thread.compile.mathLibTranslators import CTHREADS_MATHOPS


def _internal_fn(name: str):
    return SimpleNamespace(__name__=name, __cthreads_internal__=True)


def _cthreads_math_ctx():
    """Globals as if `from cthreads import math as cm` (same pattern as sync)."""
    fns = {name: _internal_fn(name) for name in CTHREADS_MATHOPS}
    cm = SimpleNamespace(__name__="cthreads.math", __cthreads_internal__=True, **fns)
    return make_ctx(
        symbols={"x": PyFloat(), "y": PyFloat(), "z": PyFloat(), "i": PyInt()},
        globals_extra={"cm": cm, **fns},
    )


def _call_src(name: str, arity: int) -> str:
    args = ", ".join(["x", "y", "z"][:arity])
    return f"{name}({args})"


def test_cthreads_mathops_table_complete():
    expected = {
        "abs": ("cthreads::math::abs", 1, "math/abs.hpp"),
        "min": ("cthreads::math::min", 2, "math/clamps.hpp"),
        "max": ("cthreads::math::max", 2, "math/clamps.hpp"),
        "clamp": ("cthreads::math::clamp", 3, "math/clamps.hpp"),
        "random": ("cthreads::math::random", 0, "math/random.hpp"),
        "uniform": ("cthreads::math::uniform", 2, "math/random.hpp"),
        "randint": ("cthreads::math::randint", 2, "math/random.hpp"),
        "seed": ("cthreads::math::seed", 1, "math/random.hpp"),
    }
    assert set(CTHREADS_MATHOPS) == set(expected)
    for name, (cpp, arity, inc) in expected.items():
        op = CTHREADS_MATHOPS[name]
        assert op.cpp_func == cpp
        assert op.arity == arity
        assert op.cpp_include == inc


@pytest.mark.parametrize("name,op", list(CTHREADS_MATHOPS.items()))
def test_resolve_cthreads_math_from_import(name, op):
    ctx = _cthreads_math_ctx()
    node = parse_expr(_call_src(name, op.arity))
    assert is_math(node, ctx)
    assert resolve_math_call(node, ctx) == op


@pytest.mark.parametrize("name,op", list(CTHREADS_MATHOPS.items()))
def test_resolve_cthreads_math_attribute_form(name, op):
    ctx = _cthreads_math_ctx()
    args = ", ".join(["x", "y", "z"][: op.arity])
    node = parse_expr(f"cm.{name}({args})")
    assert resolve_math_call(node, ctx) == op


@pytest.mark.parametrize("name,op", list(CTHREADS_MATHOPS.items()))
def test_call_translate_cthreads_math(name, op):
    ctx = _cthreads_math_ctx()
    node = parse_expr(_call_src(name, op.arity))
    got = call.translate(node, ctx)
    args = ", ".join(["x", "y", "z"][: op.arity])
    assert got == f"{op.cpp_func}({args})"
    assert any(op.cpp_include in line for line in ctx.body_includes)


def test_cthreads_math_rejects_wrong_arity():
    ctx = _cthreads_math_ctx()
    assert resolve_math_call(parse_expr("abs(x, y)"), ctx) is None
    assert resolve_math_call(parse_expr("min(x)"), ctx) is None
    assert resolve_math_call(parse_expr("clamp(x, y)"), ctx) is None
    assert resolve_math_call(parse_expr("random(x)"), ctx) is None
    assert resolve_math_call(parse_expr("seed()"), ctx) is None
    assert resolve_math_call(parse_expr("uniform(x)"), ctx) is None


def test_cthreads_math_rejects_unmarked_and_unknown():
    plain = SimpleNamespace(__name__="abs")  # no module / fn mark
    # Known op name on an unmarked module must not lower.
    other = SimpleNamespace(__name__="other", abs=plain)
    ctx = make_ctx(
        symbols={"x": PyFloat()},
        globals_extra={
            "abs": plain,
            "nope": _internal_fn("nope"),
            "other": other,
        },
    )
    assert resolve_math_call(parse_expr("abs(x)"), ctx) is None
    assert resolve_math_call(parse_expr("nope(x)"), ctx) is None
    assert resolve_math_call(parse_expr("other.abs(x)"), ctx) is None


def test_cthreads_math_accepts_module_mark_without_fn_mark():
    """Runtime pybind fns have no __cthreads_internal__; module mark is enough."""
    fn = SimpleNamespace(__name__="abs")
    cm = SimpleNamespace(__name__="cthreads.math", __cthreads_internal__=True, abs=fn)
    ctx = make_ctx(symbols={"x": PyFloat()}, globals_extra={"cm": cm, "abs": fn})
    assert resolve_math_call(parse_expr("cm.abs(x)"), ctx) == CTHREADS_MATHOPS["abs"]
    # from-import without fn mark or __module__ lookup still rejected
    assert resolve_math_call(parse_expr("abs(x)"), ctx) is None
