"""
Plugin bases for GPU Call / Attribute lowering.
"""

from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from collections.abc import Callable

from ..context import GpuTranslationContext

# Already-translated subexpressions; typically GpuSyntax.expr
TranslateExpr = Callable[[ast.expr, GpuTranslationContext], str]


class AttrPlugin(ABC):
    """
    Handles `ast.Attribute` (index builtins, props - not calls).
    """

    @abstractmethod
    def try_lower(
        self,
        node: ast.Attribute,
        ctx: GpuTranslationContext,
        translate_expr: TranslateExpr,
    ) -> str | None:
        """
        Return GLSL text if this plugin handles `node`, else None.

        #### Args:
        - node: ast.Attribute = attribute expression
        - ctx: GpuTranslationContext = current GPU translation state
        - translate_expr: TranslateExpr = nested expression lowerer

        #### Returns
        - str | None = GLSL expression, or None to try the next plugin
        """


class CallPlugin(ABC):
    """
    Handles `ast.Call` (math builtins later).
    """

    @abstractmethod
    def try_lower(
        self,
        node: ast.Call,
        ctx: GpuTranslationContext,
        translate_expr: TranslateExpr,
    ) -> str | None:
        """
        Return GLSL text if this plugin handles `node`, else None.

        #### Args:
        - node: ast.Call = call expression
        - ctx: GpuTranslationContext = current GPU translation state
        - translate_expr: TranslateExpr = nested expression lowerer

        #### Returns
        - str | None = GLSL expression, or None to try the next plugin
        """
