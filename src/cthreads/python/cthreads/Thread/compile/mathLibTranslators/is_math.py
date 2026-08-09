"""
Detect ``math`` calls that we can lower to ``std::``.

``ctx`` comes from the active translator (same ``TranslateContext`` every
AstTranslator gets). Resolve names through ``ctx.fn.__globals__`` — the AST
only has identifiers, never live objects.
"""

import ast
from typing import Any, Optional

from ..AstTranslators.context import TranslateContext
from .mathOps import MATHCONSTS, MATHOPS, MathOp


def _globals(ctx: TranslateContext) -> dict[str, Any]:
    fn = getattr(ctx, "fn", None)
    g = getattr(fn, "__globals__", None)
    return g if isinstance(g, dict) else {}


def _math_callable(obj: Any) -> Optional[str]:
    """Return math function name if ``obj`` is a whitelisted ``math.*`` fn."""
    if obj is None:
        return None
    if getattr(obj, "__module__", None) != "math":
        return None
    name = getattr(obj, "__name__", None)
    if not isinstance(name, str) or name not in MATHOPS:
        return None
    return name


def resolve_math_call(node: ast.AST, ctx: TranslateContext) -> Optional[MathOp]:
    """
    If ``node`` is a Call to a whitelisted math function, return its MathOp.
    Handles ``math.sqrt(x)`` and ``from math import sqrt; sqrt(x)``.
    """
    if not isinstance(node, ast.Call):
        return None
    if node.keywords:
        # std:: has no Python kwargs; reject early
        return None

    func = node.func
    obj: Any = None

    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        mod = _globals(ctx).get(func.value.id)
        if mod is None:
            return None
        obj = getattr(mod, func.attr, None)
    elif isinstance(func, ast.Name):
        obj = _globals(ctx).get(func.id)
    else:
        return None

    name = _math_callable(obj)
    if name is None:
        return None

    op = MATHOPS[name]
    if len(node.args) != op.arity:
        return None
    return op


def resolve_math_const(node: ast.AST, ctx: TranslateContext) -> Optional[str]:
    """
    If ``node`` is ``math.pi`` / ``math.e`` / ``math.tau`` (or alias), return
    the C++ expression string.
    """
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
