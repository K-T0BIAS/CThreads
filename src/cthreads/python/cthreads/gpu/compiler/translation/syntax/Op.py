"""
GLSL operator lowering.

Reuses CppOp operator tables (BINOPS / UNARYOPS / CMPOPS / BOOLOPS).
Handlers are overridden so they call GPU Syntax and emit GLSL (pow, no includes).
"""
import ast

from .....compiler.translation.syntax.Op import Op as CppOp
from ..context import GpuTranslationContext


class GpuOp(CppOp):
    """ast.BinOp / UnaryOp / Compare / BoolOp for @Gpu bodies."""

    # No CPU sync builtin on the GPU path.
    BUILTINS: frozenset[str] = frozenset({"range", "len"})

    @staticmethod
    def is_builtin_call(node: ast.AST, name: str) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == name
            and name in GpuOp.BUILTINS
        )

    @staticmethod
    def bin_op(node: ast.BinOp, ctx: GpuTranslationContext) -> str:
        from .Syntax import GpuSyntax

        left = GpuSyntax.expr(node.left, ctx)
        right = GpuSyntax.expr(node.right, ctx)
        # Power stays out until math builtins mirror Python -> GLSL (like CPU).
        if isinstance(node.op, ast.Pow):
            raise TypeError(
                f"GPU function {ctx.func_name}: "
                "** / pow is not supported yet "
                "(planned via a math builtin mirror)"
            )
        op = GpuOp.BINOPS.get(type(node.op))
        if not op:
            raise TypeError(
                f"GPU function {ctx.func_name}: "
                f"unsupported binary operator {type(node.op).__name__}"
            )
        return f"({left} {op} {right})"

    @staticmethod
    def unary_op(node: ast.UnaryOp, ctx: GpuTranslationContext) -> str:
        from .Syntax import GpuSyntax

        op = GpuOp.UNARYOPS.get(type(node.op))
        if not op:
            raise TypeError(
                f"GPU function {ctx.func_name}: "
                f"unsupported unary operator {type(node.op).__name__}"
            )
        operand = GpuSyntax.expr(node.operand, ctx)
        return f"({op}{operand})"

    @staticmethod
    def compare(node: ast.Compare, ctx: GpuTranslationContext) -> str:
        from .Syntax import GpuSyntax

        if len(node.ops) != len(node.comparators):
            raise TypeError(
                f"GPU function {ctx.func_name}: malformed Compare node"
            )
        left = GpuSyntax.expr(node.left, ctx)
        parts: list[str] = []
        prev = left
        for op_node, comparator in zip(node.ops, node.comparators):
            op = GpuOp.CMPOPS.get(type(op_node))
            if not op:
                raise TypeError(
                    f"GPU function {ctx.func_name}: "
                    f"unsupported compare operator {type(op_node).__name__}"
                )
            right = GpuSyntax.expr(comparator, ctx)
            parts.append(f"({prev} {op} {right})")
            prev = right
        if len(parts) == 1:
            return parts[0]
        return "(" + " && ".join(parts) + ")"

    @staticmethod
    def bool_op(node: ast.BoolOp, ctx: GpuTranslationContext) -> str:
        from .Syntax import GpuSyntax

        op = GpuOp.BOOLOPS.get(type(node.op))
        if not op:
            raise TypeError(
                f"GPU function {ctx.func_name}: "
                f"unsupported bool operator {type(node.op).__name__}"
            )
        if len(node.values) < 2:
            raise TypeError(
                f"GPU function {ctx.func_name}: BoolOp needs at least two values"
            )
        parts = [GpuSyntax.expr(v, ctx) for v in node.values]
        return "(" + f" {op} ".join(parts) + ")"
