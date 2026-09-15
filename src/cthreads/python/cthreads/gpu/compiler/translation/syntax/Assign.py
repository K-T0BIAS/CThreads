"""
Assignment lowering for @Gpu bodies.

Same shape as CPU Assign, without C++ includes / to_cpp / std::pow.
Annotated locals use Glsl.type_name. Power (`**`) is deferred to a future
math builtin mirror (like CPU stdlib -> C++).
"""
import ast

from .....compiler.translation.Source import Source
from .....types import PyList, hint_to_pytype
from ..Glsl import type_name
from ..context import GpuTranslationContext


class GpuAssign:
    """
    Lower ast.AnnAssign / Assign / AugAssign for GPU shader bodies.
    """

    @staticmethod
    def ann_assign(node: ast.AnnAssign, ctx: GpuTranslationContext) -> list[str]:
        """
        Declare a typed local and optionally initialize it.

        #### Args:
        - node: ast.AnnAssign = annotated assignment statement
        - ctx: GpuTranslationContext = symbols table for later Name lowering

        #### Returns
        - list[str] = one indented GLSL declaration line
        """
        from .Syntax import GpuSyntax

        if not isinstance(node.target, ast.Name):
            raise TypeError(
                f"GPU function {ctx.func_name}: "
                "AnnAssign target must be a plain name"
            )
        var_name: str = node.target.id
        if var_name in ctx.symbols:
            raise TypeError(
                f"GPU function {ctx.func_name}: redeclaration of {var_name!r}"
            )

        hint = Source.resolve_annotation(node.annotation, ctx.fn.__globals__)
        py_type = hint_to_pytype(hint)
        if isinstance(py_type, PyList):
            raise TypeError(
                f"GPU function {ctx.func_name}: "
                "local list declarations are not supported"
            )

        glsl_ty: str = type_name(py_type)
        # Locals are bare ids; do not add to scalar_params (those are SSBO fields).
        ctx.symbols[var_name] = py_type

        if node.value is None:
            return [f"    {glsl_ty} {var_name};"]
        rhs: str = GpuSyntax.expr(node.value, ctx)
        return [f"    {glsl_ty} {var_name} = {rhs};"]

    @staticmethod
    def assign(node: ast.Assign, ctx: GpuTranslationContext) -> list[str]:
        """
        Lower a single-target assignment (`lhs = rhs`).

        Name and Index already rewrite scalar/list SSBO accessors.

        #### Args:
        - node: ast.Assign = assignment statement
        - ctx: GpuTranslationContext = current GPU translation state

        #### Returns
        - list[str] = one indented GLSL assignment line
        """
        from .Syntax import GpuSyntax

        if len(node.targets) != 1:
            raise TypeError(
                f"GPU function {ctx.func_name}: "
                "only single-target assignment is supported"
            )
        target = node.targets[0]
        if isinstance(target, ast.Name):
            if target.id not in ctx.symbols:
                raise TypeError(
                    f"GPU function {ctx.func_name}: "
                    f"assign to unknown name {target.id!r} "
                    "(declare it with an annotated assignment first)"
                )
            if target.id in ctx.list_params:
                raise TypeError(
                    f"GPU function {ctx.func_name}: "
                    f"cannot assign to list parameter {target.id!r} "
                    "(assign elements via indexing)"
                )
        elif isinstance(target, ast.Subscript):
            if isinstance(target.slice, ast.Slice):
                raise TypeError(
                    f"GPU function {ctx.func_name}: "
                    "slice assignment is not supported"
                )
        else:
            raise TypeError(
                f"GPU function {ctx.func_name}: "
                f"unsupported assign target {type(target).__name__}"
            )

        lhs: str = GpuSyntax.expr(target, ctx)
        rhs: str = GpuSyntax.expr(node.value, ctx)
        return [f"    {lhs} = {rhs};"]

    @staticmethod
    def aug_assign(node: ast.AugAssign, ctx: GpuTranslationContext) -> list[str]:
        """
        Lower augmented assignment (`lhs op= rhs`). Power is not supported yet.

        #### Args:
        - node: ast.AugAssign = augmented assignment statement
        - ctx: GpuTranslationContext = current GPU translation state

        #### Returns
        - list[str] = one indented GLSL aug-assign line
        """
        from .Op import GpuOp
        from .Syntax import GpuSyntax

        if isinstance(node.op, ast.Pow):
            raise TypeError(
                f"GPU function {ctx.func_name}: "
                "** / pow is not supported yet "
                "(planned via a math builtin mirror)"
            )

        target: str = GpuSyntax.expr(node.target, ctx)
        value: str = GpuSyntax.expr(node.value, ctx)
        op = GpuOp.BINOPS.get(type(node.op))
        if not op:
            raise TypeError(
                f"GPU function {ctx.func_name}: "
                f"unsupported aug-assign operator {type(node.op).__name__}"
            )
        return [f"    {target} {op}= {value};"]
