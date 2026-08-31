"""
Syntax - AST node dispatch for @Thread bodies.

Area static classes (call as Syntax.expr / Literal.constant / Op.bin_op / …):
  Literal  Constant, List
  Name     Name
  Op       BinOp, UnaryOp, Compare, BoolOp (+ operator tables)
  Assign   AnnAssign, Assign, AugAssign
  Flow     If, For, While, Break, Continue, Pass, Return, Expr
  Index    Subscript

Call / Attribute: tried via plugins.registry (empty until plugins are ported).
"""

from __future__ import annotations

import ast
from typing import Callable

from ..context import TranslationContext
from ..plugins import lower_attr, lower_call
from .Assign import Assign
from .Flow import Flow
from .Index import Index
from .Literal import Literal
from .Name import Name
from .Op import Op

ExprHandler = Callable[[ast.AST, TranslationContext], str]
StmtHandler = Callable[[ast.AST, TranslationContext], list[str]]


class Syntax:
    """Dispatcher: Syntax.expr / Syntax.stmt -> area static methods."""

    _EXPR: dict[type, ExprHandler] = {
        ast.Constant: Literal.constant,
        ast.Name: Name.name,
        ast.Subscript: Index.subscript,
        ast.BinOp: Op.bin_op,
        ast.UnaryOp: Op.unary_op,
        ast.Compare: Op.compare,
        ast.BoolOp: Op.bool_op,
        ast.List: Literal.list_display,
    }
    _STMT: dict[type, StmtHandler] = {
        ast.AnnAssign: Assign.ann_assign,
        ast.Assign: Assign.assign,
        ast.AugAssign: Assign.aug_assign,
        ast.Pass: Flow.pass_stmt,
        ast.Return: Flow.return_stmt,
        ast.Expr: Flow.expr_stmt,
        ast.If: Flow.if_stmt,
        ast.For: Flow.for_stmt,
        ast.While: Flow.while_stmt,
        ast.Break: Flow.break_stmt,
        ast.Continue: Flow.continue_stmt,
    }

    @staticmethod
    def expr(node: ast.expr, ctx: TranslationContext) -> str:
        if isinstance(node, ast.Call):
            out = lower_call(node, ctx, Syntax.expr)
            if out is not None:
                return out
            raise TypeError(
                f"Thread function {ctx.func_name}: "
                "unsupported call (no CallPlugin matched; "
                "builtins / math / sync / linalg / containers not ported yet)"
            )
        if isinstance(node, ast.Attribute):
            out = lower_attr(node, ctx, Syntax.expr)
            if out is not None:
                return out
            raise TypeError(
                f"Thread function {ctx.func_name}: "
                "unsupported attribute (no AttrPlugin matched; "
                "math const / linalg props / field access not ported yet)"
            )

        handler = Syntax._EXPR.get(type(node))
        if handler is None:
            raise TypeError(
                f"Thread function {ctx.func_name}: "
                f"unsupported expression {type(node).__name__}"
            )
        return handler(node, ctx)

    @staticmethod
    def stmt(node: ast.stmt, ctx: TranslationContext) -> list[str]:
        handler = Syntax._STMT.get(type(node))
        if handler is None:
            return [f"    // unsupported statement: {type(node).__name__}"]
        return handler(node, ctx)
