"""
Syntax dispatcher for @Gpu bodies.

Wires GpuOp / GpuName / GpuIndex / GpuAssign / GpuFlow / Literal.constant.
Attribute and Call go through the GPU plugin registry.
"""

from __future__ import annotations

import ast
from typing import Callable

from .....compiler.translation.syntax.Literal import Literal
from ..context import GpuTranslationContext
from ..plugins import lower_attr, lower_call
from .Assign import GpuAssign
from .Flow import GpuFlow
from .Index import GpuIndex
from .Name import GpuName
from .Op import GpuOp

ExprHandler = Callable[[ast.AST, GpuTranslationContext], str]
StmtHandler = Callable[[ast.AST, GpuTranslationContext], list[str]]


class GpuSyntax:
    """
    Dispatcher: GpuSyntax.expr / GpuSyntax.stmt to area static methods.
    """

    _EXPR: dict[type, ExprHandler] = {
        ast.Constant: Literal.constant,
        ast.Name: GpuName.name,
        ast.Subscript: GpuIndex.subscript,
        ast.BinOp: GpuOp.bin_op,
        ast.UnaryOp: GpuOp.unary_op,
        ast.Compare: GpuOp.compare,
        ast.BoolOp: GpuOp.bool_op,
    }
    _STMT: dict[type, StmtHandler] = {
        ast.AnnAssign: GpuAssign.ann_assign,
        ast.Assign: GpuAssign.assign,
        ast.AugAssign: GpuAssign.aug_assign,
        ast.Pass: GpuFlow.pass_stmt,
        ast.Break: GpuFlow.break_stmt,
        ast.Continue: GpuFlow.continue_stmt,
        ast.Return: GpuFlow.return_stmt,
        ast.Expr: GpuFlow.expr_stmt,
        ast.If: GpuFlow.if_stmt,
        ast.For: GpuFlow.for_stmt,
        ast.While: GpuFlow.while_stmt,
    }

    @staticmethod
    def expr(node: ast.expr, ctx: GpuTranslationContext) -> str:
        """
        Lower one expression AST node to GLSL text.

        #### Args:
        - node: ast.expr = expression node
        - ctx: GpuTranslationContext = current GPU translation state

        #### Returns
        - str = GLSL expression text
        """
        if isinstance(node, ast.Call):
            out = lower_call(node, ctx, GpuSyntax.expr)
            if out is not None:
                return out
            raise TypeError(
                f"GPU function {ctx.func_name}: "
                "unsupported call (no CallPlugin matched)"
            )
        if isinstance(node, ast.Attribute):
            out = lower_attr(node, ctx, GpuSyntax.expr)
            if out is not None:
                return out
            raise TypeError(
                f"GPU function {ctx.func_name}: "
                "unsupported attribute (no AttrPlugin matched)"
            )

        handler = GpuSyntax._EXPR.get(type(node))
        if handler is None:
            raise TypeError(
                f"GPU function {ctx.func_name}: "
                f"unsupported expression {type(node).__name__}"
            )
        return handler(node, ctx)

    @staticmethod
    def stmt(node: ast.stmt, ctx: GpuTranslationContext) -> list[str]:
        """
        Lower one statement AST node to indented GLSL lines.

        #### Args:
        - node: ast.stmt = statement node
        - ctx: GpuTranslationContext = current GPU translation state

        #### Returns
        - list[str] = GLSL lines (comment stub if unsupported)
        """
        handler = GpuSyntax._STMT.get(type(node))
        if handler is None:
            return [f"    // unsupported statement: {type(node).__name__}"]
        return handler(node, ctx)
