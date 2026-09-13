"""
Minimal GLSL math CallPlugins for @Gpu (sqrt first — needed for SPH forces).
"""

from __future__ import annotations

import ast

from ..context import GpuTranslationContext
from .base import CallPlugin, TranslateExpr


class MathCallPlugin(CallPlugin):
    """
    Lower `sqrt(x)` / `math.sqrt(x)` to GLSL `sqrt(...)`.
    """

    def try_lower(
        self,
        node: ast.Call,
        ctx: GpuTranslationContext,
        translate_expr: TranslateExpr,
    ) -> str | None:
        if node.keywords:
            return None
        if len(node.args) != 1:
            return None
        fn = node.func
        is_sqrt = False
        if isinstance(fn, ast.Name) and fn.id == "sqrt":
            is_sqrt = True
        elif (
            isinstance(fn, ast.Attribute)
            and fn.attr == "sqrt"
            and isinstance(fn.value, ast.Name)
            and fn.value.id == "math"
        ):
            is_sqrt = True
        if not is_sqrt:
            return None
        arg = translate_expr(node.args[0], ctx)
        return f"sqrt({arg})"
