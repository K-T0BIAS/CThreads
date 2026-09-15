"""
ast.Subscript lowering for @Gpu bodies.

List indexing becomes SSBO unsized-array access: x[i] -> x.data[i].
"""

import ast

from ..context import GpuTranslationContext


class GpuIndex:
    """
    Lower ast.Subscript for GPU list SSBOs (not slices).
    """

    @staticmethod
    def subscript(node: ast.Subscript, ctx: GpuTranslationContext) -> str:
        """
        Lower list indexing to GLSL `base.data[index]`.

        #### Args:
        - node: ast.Subscript = Python subscript expression
        - ctx: GpuTranslationContext = symbols and list param set

        #### Returns
        - str = GLSL text such as `(x.data[i])`

        #### Raises
        - TypeError = slice syntax, or subscript of a non-list param

        #### Technical terms:
        - SSBO: shader storage buffer object; list params use `T data[]`
        """
        from .Syntax import GpuSyntax

        if isinstance(node.slice, ast.Slice):
            raise TypeError(
                f"GPU function {ctx.func_name}: slice syntax is not supported"
            )

        # Only list kernel params expose a `data[]` member in the preamble.
        if not isinstance(node.value, ast.Name) or node.value.id not in ctx.list_params:
            raise TypeError(
                f"GPU function {ctx.func_name}: "
                "subscript is only supported on list parameters"
            )

        base: str = GpuSyntax.expr(node.value, ctx)
        index: str = GpuSyntax.expr(node.slice, ctx)
        return f"({base}.data[{index}])"
