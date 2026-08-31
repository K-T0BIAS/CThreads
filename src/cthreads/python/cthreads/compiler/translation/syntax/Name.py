import ast

from ..context import TranslationContext


class Name:
    """ast.Name - locals / params. Method `self` lowers to `(*this)`."""

    @staticmethod
    def name(node: ast.Name, ctx: TranslationContext) -> str:
        if node.id == "self" and ctx.owner is not None:
            return "(*this)"
        if node.id not in ctx.symbols:
            raise TypeError(
                f"Thread function {ctx.func_name}: unknown name {node.id!r}"
            )
        return node.id
