"""
ast.Name lowering for @Gpu bodies.

Scalar params become fields on the binding-0 Scalars SSBO.
List params stay bare buffer instance names (Index adds .data[i]).
Locals stay bare identifiers.
self is stubbed until method kernels exist.
"""
import ast

from ..context import GpuTranslationContext


class GpuName:
    """
    Lower ast.Name nodes for GPU shader bodies with SSBO-aware rewriting.
    """

    @staticmethod
    def name(node: ast.Name, ctx: GpuTranslationContext) -> str:
        """
        Lower a name to a GLSL identifier or Scalars SSBO field access.

        #### Args:
        - node: ast.Name = Python name expression
        - ctx: GpuTranslationContext = symbols and scalar/list param sets

        #### Returns
        - str = GLSL text (`scalars.n`, bare local/list id, or self hook)

        #### Technical terms:
        - SSBO: shader storage buffer object holding kernel params on device
        """
        if node.id == "self":
            # Separate hook so method kernels can land here without rewriting name().
            return GpuName.lower_self(ctx)

        if node.id not in ctx.symbols:
            raise TypeError(
                f"GPU function {ctx.func_name}: unknown name {node.id!r}"
            )

        # Scalar kernel params live in the binding-0 block instance `scalars`.
        if node.id in ctx.scalar_params:
            return f"scalars.{node.id}"

        # List params are SSBO instance names; locals are ordinary GLSL ids.
        return node.id

    @staticmethod
    def lower_self(ctx: GpuTranslationContext) -> str:
        """
        Lower `self` for method kernels (not implemented yet).

        #### Args:
        - ctx: GpuTranslationContext = current GPU translation state

        #### Returns
        - str = GLSL receiver expression (when supported)

        #### Raises
        - TypeError = self / method kernels are not supported yet
        """
        raise TypeError(
            f"GPU function {ctx.func_name}: "
            "self / method kernels are not supported yet"
        )
