"""
`cthreads.math` -> `cthreads::math::*` + quoted `math/*.hpp` includes.

Handles:
  cthreads.math.abs(x) / from cthreads.math import abs; abs(x)

Detection: module (or callable's module) marked `__cthreads_internal__`
(pybind binds the flag on the module, not on each function).
"""

import ast
import sys
from typing import Any, NamedTuple

from ...context import TranslationContext
from ...include import add_include
from ..base import CallPlugin, TranslateExpr


class _CthreadsMathOp(NamedTuple):
    cpp_func: str
    arity: int
    cpp_include: str  # e.g. math/abs.hpp -> #include "math/abs.hpp"


CTHREADS_MATHOPS: dict[str, _CthreadsMathOp] = {
    "abs": _CthreadsMathOp("cthreads::math::abs", 1, "math/abs.hpp"),
    "min": _CthreadsMathOp("cthreads::math::min", 2, "math/clamps.hpp"),
    "max": _CthreadsMathOp("cthreads::math::max", 2, "math/clamps.hpp"),
    "clamp": _CthreadsMathOp("cthreads::math::clamp", 3, "math/clamps.hpp"),
    "random": _CthreadsMathOp("cthreads::math::random", 0, "math/random.hpp"),
    "uniform": _CthreadsMathOp("cthreads::math::uniform", 2, "math/random.hpp"),
    "randint": _CthreadsMathOp("cthreads::math::randint", 2, "math/random.hpp"),
    "seed": _CthreadsMathOp("cthreads::math::seed", 1, "math/random.hpp"),
}


def _globals(ctx: TranslationContext) -> dict[str, Any]:
    g = getattr(ctx.fn, "__globals__", None)
    return g if isinstance(g, dict) else {}


def _resolve_call_obj(node: ast.Call, ctx: TranslationContext) -> tuple[Any, Any]:
    """(callable, parent_module_or_None)."""
    func = node.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        mod = _globals(ctx).get(func.value.id)
        if mod is None:
            return None, None
        return getattr(mod, func.attr, None), mod
    if isinstance(func, ast.Name):
        return _globals(ctx).get(func.id), None
    return None, None


def _owner_is_cthreads_math(obj: Any, parent_mod: Any) -> bool:
    if parent_mod is not None and getattr(parent_mod, "__cthreads_internal__", False):
        return True
    if getattr(obj, "__cthreads_internal__", False):
        return True
    mod_name = getattr(obj, "__module__", None)
    if isinstance(mod_name, str):
        mod = sys.modules.get(mod_name)
        if mod is not None and getattr(mod, "__cthreads_internal__", False):
            return True
    return False


def _cthreads_math_op(obj: Any, parent_mod: Any) -> _CthreadsMathOp | None:
    if obj is None:
        return None
    name = getattr(obj, "__name__", None)
    if not isinstance(name, str) or name not in CTHREADS_MATHOPS:
        return None
    if not _owner_is_cthreads_math(obj, parent_mod):
        return None
    return CTHREADS_MATHOPS[name]


class CthreadsMathCallPlugin(CallPlugin):
    """`cthreads.math.abs(x)` -> `cthreads::math::abs(x)` + `#include "math/abs.hpp"`."""

    def try_lower(
        self,
        node: ast.Call,
        ctx: TranslationContext,
        translate_expr: TranslateExpr,
    ) -> str | None:
        obj, parent = _resolve_call_obj(node, ctx)
        op = _cthreads_math_op(obj, parent)
        if op is None:
            return None

        name = getattr(obj, "__name__", "?")
        if node.keywords:
            raise TypeError(
                f"Thread function {ctx.func_name}: "
                f"cthreads.math.{name} keyword args are not supported"
            )
        if len(node.args) != op.arity:
            raise TypeError(
                f"Thread function {ctx.func_name}: "
                f"cthreads.math.{name} expects {op.arity} "
                f"args, got {len(node.args)}"
            )

        add_include(
            ctx.body_includes,
            ctx.seen_body,
            f'#include "{op.cpp_include}"\n',
        )
        args = ", ".join(translate_expr(a, ctx) for a in node.args)
        return f"{op.cpp_func}({args})"
