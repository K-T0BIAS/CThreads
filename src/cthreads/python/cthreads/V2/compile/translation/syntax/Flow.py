from __future__ import annotations

import ast

from ....types import PyInt, PyList
from ..context import TranslationContext
from .Op import Op


class Flow:
    """ast.If / For / While / Break / Continue / Pass / Return / Expr."""

    @staticmethod
    def nest(lines: list[str]) -> list[str]:
        return ["    " + line if line.strip() else line for line in lines]

    @staticmethod
    def if_stmt(node: ast.If, ctx: TranslationContext) -> list[str]:
        from .Syntax import Syntax

        test = Syntax.expr(node.test, ctx)
        lines = [f"    if ({test}) {{"]
        for stmt in node.body:
            lines.extend(Flow.nest(Syntax.stmt(stmt, ctx)))
        lines.append("    }")
        if node.orelse:
            lines.append("    else {")
            for stmt in node.orelse:
                lines.extend(Flow.nest(Syntax.stmt(stmt, ctx)))
            lines.append("    }")
        return lines

    @staticmethod
    def for_stmt(node: ast.For, ctx: TranslationContext) -> list[str]:
        from .Syntax import Syntax

        if node.orelse:
            raise TypeError(
                f"Thread function {ctx.func_name}: for/else is not supported"
            )
        if not isinstance(node.target, ast.Name):
            raise TypeError(
                f"Thread function {ctx.func_name}: "
                "for-loop target must be a plain name"
            )
        loop_var = node.target.id
        if loop_var in ctx.symbols:
            raise TypeError(
                f"Thread function {ctx.func_name}: "
                f"for-loop rebinds existing name {loop_var!r}"
            )

        it = node.iter
        if Op.is_builtin_call(it, "range"):
            assert isinstance(it, ast.Call)
            if it.keywords:
                raise TypeError(
                    f"Thread function {ctx.func_name}: "
                    "range() keyword args are not supported"
                )
            n = len(it.args)
            if n == 1:
                start, stop, step = "0", Syntax.expr(it.args[0], ctx), "1"
            elif n == 2:
                start = Syntax.expr(it.args[0], ctx)
                stop = Syntax.expr(it.args[1], ctx)
                step = "1"
            elif n == 3:
                start = Syntax.expr(it.args[0], ctx)
                stop = Syntax.expr(it.args[1], ctx)
                step = Syntax.expr(it.args[2], ctx)
            else:
                raise TypeError(
                    f"Thread function {ctx.func_name}: "
                    f"range() expects 1..3 args, got {n}"
                )
            ctx.symbols[loop_var] = PyInt()
            lines = [
                f"    for (int {loop_var} = {start}; "
                f"{loop_var} < {stop}; "
                f"{loop_var} += {step}) {{"
            ]
            for stmt in node.body:
                lines.extend(Flow.nest(Syntax.stmt(stmt, ctx)))
            lines.append("    }")
            del ctx.symbols[loop_var]
            return lines

        if not isinstance(it, ast.Name):
            raise TypeError(
                f"Thread function {ctx.func_name}: "
                "for-iter must be a name or range(...)"
            )
        container_ty = ctx.symbols.get(it.id)
        if not isinstance(container_ty, PyList):
            raise TypeError(
                f"Thread function {ctx.func_name}: "
                f"for-iter {it.id!r} must be a list[...], "
                f"got {type(container_ty).__name__}"
            )
        container = Syntax.expr(it, ctx)
        ctx.symbols[loop_var] = container_ty.inner_type
        lines = [f"    for (auto& {loop_var} : {container}) {{"]
        for stmt in node.body:
            lines.extend(Flow.nest(Syntax.stmt(stmt, ctx)))
        lines.append("    }")
        del ctx.symbols[loop_var]
        return lines

    @staticmethod
    def while_stmt(node: ast.While, ctx: TranslationContext) -> list[str]:
        from .Syntax import Syntax

        if node.orelse:
            raise TypeError(
                f"Thread function {ctx.func_name}: while/else is not supported"
            )
        test = Syntax.expr(node.test, ctx)
        lines = [f"    while ({test}) {{"]
        for stmt in node.body:
            lines.extend(Flow.nest(Syntax.stmt(stmt, ctx)))
        lines.append("    }")
        return lines

    @staticmethod
    def break_stmt(node: ast.Break, ctx: TranslationContext) -> list[str]:
        return ["    break;"]

    @staticmethod
    def continue_stmt(node: ast.Continue, ctx: TranslationContext) -> list[str]:
        return ["    continue;"]

    @staticmethod
    def pass_stmt(node: ast.Pass, ctx: TranslationContext) -> list[str]:
        return []

    @staticmethod
    def return_stmt(node: ast.Return, ctx: TranslationContext) -> list[str]:
        from .Syntax import Syntax

        if node.value is None:
            return ["    return;"]
        return [f"    return {Syntax.expr(node.value, ctx)};"]

    @staticmethod
    def expr_stmt(node: ast.Expr, ctx: TranslationContext) -> list[str]:
        from .Syntax import Syntax

        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return []
        if isinstance(node.value, ast.Call):
            return [f"    {Syntax.expr(node.value, ctx)};"]
        return [f"    // unsupported statement: Expr ({type(node.value).__name__})"]
