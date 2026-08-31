import ast

from ..context import TranslationContext


class Index:
    """ast.Subscript - list/dict indexing (not slices)."""

    @staticmethod
    def subscript(node: ast.Subscript, ctx: TranslationContext) -> str:
        from .Syntax import Syntax

        if isinstance(node.slice, ast.Slice):
            raise TypeError(
                f"Thread function {ctx.func_name}: slice syntax is not supported"
            )
        base = Syntax.expr(node.value, ctx)
        index = Syntax.expr(node.slice, ctx)
        return f"({base}[{index}])"
