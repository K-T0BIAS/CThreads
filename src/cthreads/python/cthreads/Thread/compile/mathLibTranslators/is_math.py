"""
Detect ``math`` / ``cthreads.math`` calls for lowering.

Resolve names through ``ctx.fn.__globals__``. Stdlib math uses ``__module__``;
``cthreads.math`` is marked ``__cthreads_internal__`` on the *module* (pybind
bound functions cannot carry that attribute).
"""

from __future__ import annotations

import ast
import sys
from typing import Any, Optional

from ..AstTranslators.context import TranslateContext
from .mathOps import CTHREADS_MATHOPS, MATHCONSTS, MATHOPS, MathOp


def _globals(ctx: TranslateContext) -> dict[str, Any]:
    fn = getattr(ctx, "fn", None)
    g = getattr(fn, "__globals__", None)
    return g if isinstance(g, dict) else {}


def _resolve_call_obj(node: ast.Call, ctx: TranslateContext) -> tuple[Any, Any]:
    """Return ``(callable_or_none, parent_module_or_none)``."""
    func = node.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        mod = _globals(ctx).get(func.value.id)
        if mod is None:
            return None, None
        return getattr(mod, func.attr, None), mod
    if isinstance(func, ast.Name):
        return _globals(ctx).get(func.id), None
    return None, None


def _stdlib_math_op(obj: Any) -> Optional[MathOp]:
    if obj is None:
        return None
    if getattr(obj, "__module__", None) != "math":
        return None
    name = getattr(obj, "__name__", None)
    if not isinstance(name, str) or name not in MATHOPS:
        return None
    return MATHOPS[name]


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


def _cthreads_math_op(obj: Any, parent_mod: Any = None) -> Optional[MathOp]:
    if obj is None:
        return None
    name = getattr(obj, "__name__", None)
    if not isinstance(name, str) or name not in CTHREADS_MATHOPS:
        return None
    if not _owner_is_cthreads_math(obj, parent_mod):
        return None
    return CTHREADS_MATHOPS[name]


def resolve_math_call(node: ast.AST, ctx: TranslateContext) -> Optional[MathOp]:
    """
    Whitelisted ``math.*`` or ``cthreads.math.*`` call → MathOp.
    Forms: ``math.sqrt(x)``, ``from math import sqrt``, ``from cthreads import math; math.abs(x)``.
    """
    if not isinstance(node, ast.Call):
        return None
    if node.keywords:
        return None

    obj, parent = _resolve_call_obj(node, ctx)
    op = _cthreads_math_op(obj, parent) or _stdlib_math_op(obj)
    if op is None:
        return None
    if len(node.args) != op.arity:
        return None
    return op


def resolve_math_const(node: ast.AST, ctx: TranslateContext) -> Optional[str]:
    """``math.pi`` / ``math.e`` / ``math.tau`` → C++ expression."""
    if not isinstance(node, ast.Attribute):
        return None
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
    return MATHCONSTS[node.attr]


def is_math(node: ast.AST, ctx: TranslateContext) -> bool:
    return resolve_math_call(node, ctx) is not None
