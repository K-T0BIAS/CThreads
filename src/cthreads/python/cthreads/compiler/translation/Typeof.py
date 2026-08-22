from __future__ import annotations

import ast

from ...frontend.Registry import REGISTRY
from ...types import PyList, PyThreadable, PyType, peel_shared
from .context import TranslationContext


class Typeof:
    """Receiver type for method/attr lowering: symbols, Threadable fields, list elems."""

    @staticmethod
    def src(node: ast.AST) -> str:
        try:
            return ast.unparse(node)
        except (AttributeError, ValueError):
            return type(node).__name__

    @staticmethod
    def of(node: ast.expr, ctx: TranslationContext) -> PyType | None:
        if isinstance(node, ast.Name):
            ty = ctx.symbols.get(node.id)
            if isinstance(ty, PyType):
                return peel_shared(ty)
            return None

        if isinstance(node, ast.Attribute):
            base = Typeof.of(node.value, ctx)
            if not isinstance(base, PyThreadable):
                return None
            unit = REGISTRY.threadable_units.get(base.name)
            if unit is None:
                return None
            return unit.fields.get(node.attr)

        if isinstance(node, ast.Subscript):
            if isinstance(node.slice, ast.Slice):
                return None
            base = Typeof.of(node.value, ctx)
            if isinstance(base, PyList):
                return base.inner_type
            return None

        return None
