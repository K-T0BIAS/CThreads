"""
Minimal GLSL math CallPlugins for @Gpu (sqrt / floor / int — SPH grid + forces).
"""

from __future__ import annotations

import ast

from ..context import GpuTranslationContext
from .base import CallPlugin, TranslateExpr


class MathCallPlugin(CallPlugin):
    """
    Lower a small set of math / cast calls to GLSL.
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
        arg = translate_expr(node.args[0], ctx)

        if isinstance(fn, ast.Name) and fn.id == "sqrt":
            return f"sqrt({arg})"
        if (
            isinstance(fn, ast.Attribute)
            and fn.attr == "sqrt"
            and isinstance(fn.value, ast.Name)
            and fn.value.id == "math"
        ):
            return f"sqrt({arg})"

        # Truncate toward -inf (GLSL floor); used for cell indices.
        if isinstance(fn, ast.Name) and fn.id == "floor":
            return f"floor({arg})"
        if (
            isinstance(fn, ast.Attribute)
            and fn.attr == "floor"
            and isinstance(fn.value, ast.Name)
            and fn.value.id == "math"
        ):
            return f"floor({arg})"

        # Python int(x) on floats -> GLSL int(x) (trunc toward zero).
        if isinstance(fn, ast.Name) and fn.id == "int":
            return f"int({arg})"

        return None
