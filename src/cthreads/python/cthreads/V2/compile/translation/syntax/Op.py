from __future__ import annotations

import ast

from ..Cpp import Cpp
from ..context import TranslationContext
from ..include import add_include


class Op:
    """ast.BinOp / UnaryOp / Compare / BoolOp plus operator tables."""

    BINOPS: dict[type, str] = {
        ast.Add: "+",
        ast.Sub: "-",
        ast.Mult: "*",
        ast.Div: "/",
        ast.FloorDiv: "/",
        ast.Mod: "%",
        ast.LShift: "<<",
        ast.RShift: ">>",
        ast.BitOr: "|",
        ast.BitXor: "^",
        ast.BitAnd: "&",
    }
    UNARYOPS: dict[type, str] = {
        ast.UAdd: "+",
        ast.USub: "-",
        ast.Not: "!",
        ast.Invert: "~",
    }
    CMPOPS: dict[type, str] = {
        ast.Eq: "==",
        ast.NotEq: "!=",
        ast.Lt: "<",
        ast.LtE: "<=",
        ast.Gt: ">",
        ast.GtE: ">=",
    }
    BOOLOPS: dict[type, str] = {
        ast.And: "&&",
        ast.Or: "||",
    }
    BUILTINS: frozenset[str] = frozenset({"range", "len", "__sync_state"})

    @staticmethod
    def is_builtin_call(node: ast.AST, name: str) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == name
            and name in Op.BUILTINS
        )

    @staticmethod
    def bin_op(node: ast.BinOp, ctx: TranslationContext) -> str:
        from .Syntax import Syntax

        left = Syntax.expr(node.left, ctx)
        right = Syntax.expr(node.right, ctx)
        if isinstance(node.op, ast.Pow):
            add_include(ctx.body_includes, ctx.seen_body, Cpp.CMATH)
            return f"std::pow({left}, {right})"
        op = Op.BINOPS.get(type(node.op))
        if not op:
            raise TypeError(
                f"Thread function {ctx.func_name}: "
                f"unsupported binary operator {type(node.op).__name__}"
            )
        return f"({left} {op} {right})"

    @staticmethod
    def unary_op(node: ast.UnaryOp, ctx: TranslationContext) -> str:
        from .Syntax import Syntax

        op = Op.UNARYOPS.get(type(node.op))
        if not op:
            raise TypeError(
                f"Thread function {ctx.func_name}: "
                f"unsupported unary operator {type(node.op).__name__}"
            )
        operand = Syntax.expr(node.operand, ctx)
        return f"({op}{operand})"

    @staticmethod
    def compare(node: ast.Compare, ctx: TranslationContext) -> str:
        from .Syntax import Syntax

        if len(node.ops) != len(node.comparators):
            raise TypeError(
                f"Thread function {ctx.func_name}: malformed Compare node"
            )
        left = Syntax.expr(node.left, ctx)
        parts: list[str] = []
        prev = left
        for op_node, comparator in zip(node.ops, node.comparators):
            op = Op.CMPOPS.get(type(op_node))
            if not op:
                raise TypeError(
                    f"Thread function {ctx.func_name}: "
                    f"unsupported compare operator {type(op_node).__name__}"
                )
            right = Syntax.expr(comparator, ctx)
            parts.append(f"({prev} {op} {right})")
            prev = right
        if len(parts) == 1:
            return parts[0]
        return "(" + " && ".join(parts) + ")"

    @staticmethod
    def bool_op(node: ast.BoolOp, ctx: TranslationContext) -> str:
        from .Syntax import Syntax

        op = Op.BOOLOPS.get(type(node.op))
        if not op:
            raise TypeError(
                f"Thread function {ctx.func_name}: "
                f"unsupported bool operator {type(node.op).__name__}"
            )
        if len(node.values) < 2:
            raise TypeError(
                f"Thread function {ctx.func_name}: BoolOp needs at least two values"
            )
        parts = [Syntax.expr(v, ctx) for v in node.values]
        return "(" + f" {op} ".join(parts) + ")"
