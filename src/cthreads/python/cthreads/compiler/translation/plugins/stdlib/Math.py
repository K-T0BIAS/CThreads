"""
stdlib `math` -> C++ `<cmath>` / `<numbers>` lowering.

Handles:
  math.sqrt(x) / from math import sqrt; sqrt(x)  -> std::sqrt(x) + #include <cmath>
  math.pi                                      -> std::numbers::pi + #include <numbers>

Does not handle `cthreads.math` — see `plugins/cthreads_math`.
"""

import ast
import math
from typing import Any, NamedTuple

from ...Cpp import Cpp
from ...context import TranslationContext
from ...include import add_include
from ..base import AttrPlugin, CallPlugin, TranslateExpr


class _StdMathOp(NamedTuple):
    cpp_func: str
    arity: int


_ALL_MATHOPS: dict[str, _StdMathOp] = {
    "sqrt": _StdMathOp("std::sqrt", 1),
    "cbrt": _StdMathOp("std::cbrt", 1),
    "pow": _StdMathOp("std::pow", 2),
    "hypot": _StdMathOp("std::hypot", 2),
    "fabs": _StdMathOp("std::fabs", 1),
    "floor": _StdMathOp("std::floor", 1),
    "ceil": _StdMathOp("std::ceil", 1),
    "trunc": _StdMathOp("std::trunc", 1),
    "fmod": _StdMathOp("std::fmod", 2),
    "remainder": _StdMathOp("std::remainder", 2),
    "copysign": _StdMathOp("std::copysign", 2),
    "fma": _StdMathOp("std::fma", 3),
    "exp": _StdMathOp("std::exp", 1),
    "exp2": _StdMathOp("std::exp2", 1),
    "expm1": _StdMathOp("std::expm1", 1),
    "log": _StdMathOp("std::log", 1),
    "log2": _StdMathOp("std::log2", 1),
    "log10": _StdMathOp("std::log10", 1),
    "log1p": _StdMathOp("std::log1p", 1),
    "sin": _StdMathOp("std::sin", 1),
    "cos": _StdMathOp("std::cos", 1),
    "tan": _StdMathOp("std::tan", 1),
    "asin": _StdMathOp("std::asin", 1),
    "acos": _StdMathOp("std::acos", 1),
    "atan": _StdMathOp("std::atan", 1),
    "atan2": _StdMathOp("std::atan2", 2),
    "sinh": _StdMathOp("std::sinh", 1),
    "cosh": _StdMathOp("std::cosh", 1),
    "tanh": _StdMathOp("std::tanh", 1),
    "asinh": _StdMathOp("std::asinh", 1),
    "acosh": _StdMathOp("std::acosh", 1),
    "atanh": _StdMathOp("std::atanh", 1),
    "erf": _StdMathOp("std::erf", 1),
    "erfc": _StdMathOp("std::erfc", 1),
    "gamma": _StdMathOp("std::tgamma", 1),
    "lgamma": _StdMathOp("std::lgamma", 1),
    "ldexp": _StdMathOp("std::ldexp", 2),
    "isfinite": _StdMathOp("std::isfinite", 1),
    "isinf": _StdMathOp("std::isinf", 1),
    "isnan": _StdMathOp("std::isnan", 1),
}

MATHOPS: dict[str, _StdMathOp] = {
    name: op for name, op in _ALL_MATHOPS.items() if hasattr(math, name)
}

_ALL_MATHCONSTS: dict[str, str] = {
    "pi": "std::numbers::pi",
    "e": "std::numbers::e",
    "tau": "std::numbers::pi * 2.0",
}

MATHCONSTS: dict[str, str] = {
    name: expr for name, expr in _ALL_MATHCONSTS.items() if hasattr(math, name)
}


def _globals(ctx: TranslationContext) -> dict[str, Any]:
    g = getattr(ctx.fn, "__globals__", None)
    return g if isinstance(g, dict) else {}


def _resolve_call_obj(node: ast.Call, ctx: TranslationContext) -> Any:
    """Callable object for math.sin / from math import sin."""
    func = node.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        mod = _globals(ctx).get(func.value.id)
        if mod is None:
            return None
        return getattr(mod, func.attr, None)
    if isinstance(func, ast.Name):
        return _globals(ctx).get(func.id)
    return None


def _stdlib_math_op(obj: Any) -> _StdMathOp | None:
    if obj is None:
        return None
    if getattr(obj, "__module__", None) != "math":
        return None
    name = getattr(obj, "__name__", None)
    if not isinstance(name, str) or name not in MATHOPS:
        return None
    return MATHOPS[name]


class MathCallPlugin(CallPlugin):
    """`math.sin(x)` / `sqrt(x)` ->`std::sin(x)` + `#include <cmath>`."""

    def try_lower(
        self,
        node: ast.Call,
        ctx: TranslationContext,
        translate_expr: TranslateExpr,
    ) -> str | None:
        obj = _resolve_call_obj(node, ctx)
        op = _stdlib_math_op(obj)
        if op is None:
            return None

        if node.keywords:
            raise TypeError(
                f"Thread function {ctx.func_name}: "
                f"math.{getattr(obj, '__name__', '?')} keyword args are not supported"
            )
        if len(node.args) != op.arity:
            raise TypeError(
                f"Thread function {ctx.func_name}: "
                f"math.{getattr(obj, '__name__', '?')} expects {op.arity} "
                f"args, got {len(node.args)}"
            )

        add_include(ctx.body_includes, ctx.seen_body, Cpp.CMATH)
        args = ", ".join(translate_expr(a, ctx) for a in node.args)
        return f"{op.cpp_func}({args})"


class MathConstPlugin(AttrPlugin):
    """`math.pi` -> `std::numbers::pi` + `#include <numbers>`."""

    def try_lower(
        self,
        node: ast.Attribute,
        ctx: TranslationContext,
        translate_expr: TranslateExpr,
    ) -> str | None:
        if not isinstance(node.value, ast.Name):
            return None
        if node.attr not in MATHCONSTS:
            return None
        mod = _globals(ctx).get(node.value.id)
        if mod is None:
            return None
        if getattr(mod, "__name__", None) != "math":
            return None
        if getattr(mod, node.attr, None) is None:
            return None

        add_include(ctx.body_includes, ctx.seen_body, Cpp.NUMBERS)
        return MATHCONSTS[node.attr]
