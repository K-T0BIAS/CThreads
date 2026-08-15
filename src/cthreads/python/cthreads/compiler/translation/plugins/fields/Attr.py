"""
Fallback attribute lowering (v1 attribute.py tail).

Must register **last** among AttrPlugins so math consts / linalg props win first.
"""

from __future__ import annotations

import ast

from ...context import TranslationContext
from ..base import AttrPlugin, TranslateExpr


class FieldAttrPlugin(AttrPlugin):
    """`self.x` -> `this->x`; otherwise `(base).attr`."""

    def try_lower(
        self,
        node: ast.Attribute,
        ctx: TranslationContext,
        translate_expr: TranslateExpr,
    ) -> str | None:
        if (
            isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and ctx.owner is not None
        ):
            return f"this->{node.attr}"
        base = translate_expr(node.value, ctx)
        return f"{base}.{node.attr}"
