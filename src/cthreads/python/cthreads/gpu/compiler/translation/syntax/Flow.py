"""
Control-flow lowering for @Gpu bodies.

Subclasses CPU Flow to reuse nest / pass / break / continue.
Overrides handlers that import Syntax so they use GpuSyntax.
for-in over lists (C++ auto&) is rejected; range-for is kept.
"""
import ast

from .....compiler.translation.syntax.Flow import Flow
from .....types import PyInt
from ..context import GpuTranslationContext
from .Op import GpuOp


class GpuFlow(Flow):
    """
    Lower if / while / for / return for GPU shader bodies. Reuses CPU Flow for shared helpers.
    """

    @staticmethod
    def if_stmt(node: ast.If, ctx: GpuTranslationContext) -> list[str]:
        """
        Lower an if/else statement to GLSL.

        #### Args:
        - node: ast.If = if statement
        - ctx: GpuTranslationContext = current GPU translation state

        #### Returns
        - list[str] = indented GLSL if/else lines
        """
        from .Syntax import GpuSyntax

        test: str = GpuSyntax.expr(node.test, ctx)
        lines: list[str] = [f"    if ({test}) {{"]
        for stmt in node.body:
            lines.extend(GpuFlow.nest(GpuSyntax.stmt(stmt, ctx)))
        lines.append("    }")
        if node.orelse:
            lines.append("    else {")
            for stmt in node.orelse:
                lines.extend(GpuFlow.nest(GpuSyntax.stmt(stmt, ctx)))
            lines.append("    }")
        return lines

    @staticmethod
    def while_stmt(node: ast.While, ctx: GpuTranslationContext) -> list[str]:
        """
        Lower a while loop to GLSL (no while/else).

        #### Args:
        - node: ast.While = while statement
        - ctx: GpuTranslationContext = current GPU translation state

        #### Returns
        - list[str] = indented GLSL while lines
        """
        from .Syntax import GpuSyntax

        if node.orelse:
            raise TypeError(
                f"GPU function {ctx.func_name}: while/else is not supported"
            )
        test: str = GpuSyntax.expr(node.test, ctx)
        lines: list[str] = [f"    while ({test}) {{"]
        for stmt in node.body:
            lines.extend(GpuFlow.nest(GpuSyntax.stmt(stmt, ctx)))
        lines.append("    }")
        return lines

    @staticmethod
    def for_stmt(node: ast.For, ctx: GpuTranslationContext) -> list[str]:
        """
        Lower `for i in range(...)` to a C-style GLSL for loop.

        Iteration over list parameters is not supported (no C++ range-for).

        #### Args:
        - node: ast.For = for statement
        - ctx: GpuTranslationContext = current GPU translation state

        #### Returns
        - list[str] = indented GLSL for-loop lines
        """
        from .Syntax import GpuSyntax

        if node.orelse:
            raise TypeError(
                f"GPU function {ctx.func_name}: for/else is not supported"
            )
        if not isinstance(node.target, ast.Name):
            raise TypeError(
                f"GPU function {ctx.func_name}: "
                "for-loop target must be a plain name"
            )
        loop_var: str = node.target.id
        if loop_var in ctx.symbols:
            raise TypeError(
                f"GPU function {ctx.func_name}: "
                f"for-loop rebinds existing name {loop_var!r}"
            )

        it = node.iter
        if not GpuOp.is_builtin_call(it, "range"):
            raise TypeError(
                f"GPU function {ctx.func_name}: "
                "for-iter must be range(...) "
                "(list iteration is not supported on the GPU path)"
            )
        assert isinstance(it, ast.Call)
        if it.keywords:
            raise TypeError(
                f"GPU function {ctx.func_name}: "
                "range() keyword args are not supported"
            )
        n: int = len(it.args)
        if n == 1:
            start, stop, step = "0", GpuSyntax.expr(it.args[0], ctx), "1"
        elif n == 2:
            start = GpuSyntax.expr(it.args[0], ctx)
            stop = GpuSyntax.expr(it.args[1], ctx)
            step = "1"
        elif n == 3:
            start = GpuSyntax.expr(it.args[0], ctx)
            stop = GpuSyntax.expr(it.args[1], ctx)
            step = GpuSyntax.expr(it.args[2], ctx)
        else:
            raise TypeError(
                f"GPU function {ctx.func_name}: "
                f"range() expects 1..3 args, got {n}"
            )

        ctx.symbols[loop_var] = PyInt()
        lines: list[str] = [
            f"    for (int {loop_var} = {start}; "
            f"{loop_var} < {stop}; "
            f"{loop_var} += {step}) {{"
        ]
        for stmt in node.body:
            lines.extend(GpuFlow.nest(GpuSyntax.stmt(stmt, ctx)))
        lines.append("    }")
        del ctx.symbols[loop_var]
        return lines

    @staticmethod
    def return_stmt(node: ast.Return, ctx: GpuTranslationContext) -> list[str]:
        """
        Lower return to a void early exit.

        Valued returns are not lowered yet; always emit bare `return;`.

        #### Args:
        - node: ast.Return = return statement
        - ctx: GpuTranslationContext = current GPU translation state

        #### Returns
        - list[str] = one indented `return;` line
        """
        return ["    return;"]

    @staticmethod
    def expr_stmt(node: ast.Expr, ctx: GpuTranslationContext) -> list[str]:
        """
        Lower expression statements; string doc-exprs are ignored.

        Call expressions go through GpuSyntax.expr (CallPlugins), e.g.
        `__sync_threads()` -> GLSL barrier.

        #### Args:
        - node: ast.Expr = expression statement
        - ctx: GpuTranslationContext = current GPU translation state

        #### Returns
        - list[str] = GLSL statement lines, empty for docstrings, or a comment
        """
        from .Syntax import GpuSyntax

        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return []
        if isinstance(node.value, ast.Call):
            return [f"    {GpuSyntax.expr(node.value, ctx)};"]
        return [
            f"    // unsupported statement: Expr ({type(node.value).__name__})"
        ]
