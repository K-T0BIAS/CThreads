"""Unit tests for math → std:: lowering (stdlib Math plugin + Syntax)."""

from __future__ import annotations

import math

import pytest

from cthreads.compiler.translation.plugins.stdlib.Math import (
    MATHCONSTS,
    MATHOPS,
    _ALL_MATHOPS,
)
from cthreads.compiler.translation.syntax import Syntax
from cthreads.types import PyFloat, PyInt
from helpers import make_ctx, parse_expr


def _math_ctx(**symbols):
    syms = {"x": PyFloat(), "y": PyFloat(), "z": PyFloat(), "i": PyInt(), **symbols}
    return make_ctx(symbols=syms, globals_extra={"math": math, "sqrt": math.sqrt})


def test_mathops_table_shape():
    from cthreads.compiler.translation.Cpp import Cpp

    assert "#include <cmath>" in Cpp.CMATH
    for name, op in MATHOPS.items():
        assert hasattr(math, name), f"math.{name} missing"
        assert op.cpp_func.startswith("std::"), name
        assert op.arity >= 1


def test_version_gated_entries_match_hasattr():
    for name in _ALL_MATHOPS:
        if hasattr(math, name):
            assert name in MATHOPS
        else:
            assert name not in MATHOPS


def test_gamma_maps_to_tgamma():
    assert MATHOPS["gamma"].cpp_func == "std::tgamma"
    assert MATHOPS["gamma"].arity == 1


@pytest.mark.parametrize("name,op", list(MATHOPS.items()))
def test_call_translate_emits_std(name, op):
    args = ", ".join(["x", "y", "z"][: op.arity])
    node = parse_expr(f"math.{name}({args})")
    ctx = _math_ctx()
    got = Syntax.expr(node, ctx)
    expected_args = ", ".join(["x", "y", "z"][: op.arity])
    assert got == f"{op.cpp_func}({expected_args})"
    assert any("cmath" in line for line in ctx.body_includes)


def test_from_import_style_sqrt():
    ctx = _math_ctx()
    assert Syntax.expr(parse_expr("sqrt(x)"), ctx) == "std::sqrt(x)"


def test_math_const_pi_e_tau():
    ctx = _math_ctx()
    for name, cpp in MATHCONSTS.items():
        assert Syntax.expr(parse_expr(f"math.{name}"), ctx) == cpp
    assert any("numbers" in line for line in ctx.body_includes)


def test_math_arity_mismatch_raises():
    ctx = _math_ctx()
    with pytest.raises(TypeError, match="expects"):
        Syntax.expr(parse_expr("math.pow(x)"), ctx)


def test_binop_pow_still_uses_cmath():
    ctx = _math_ctx()
    assert Syntax.expr(parse_expr("x ** y"), ctx) == "std::pow(x, y)"
    assert any("cmath" in line for line in ctx.body_includes)
