"""
Translate `ast.While` — while loop.

Python:
    while cond:
        ...

AST:
    While(test=..., body=[...], orelse=[...])

C++:
    while (<test>) {
        ...
    }

Python while/else is not supported (orelse must be empty).
"""

import ast

from .context import TranslateContext


def translate(node: ast.While, ctx: TranslateContext) -> list[str]:
    from .translate import translate_expr, translate_stmt

    if node.orelse:
        raise TypeError(
            f"Thread function {ctx.func_name}: while/else is not supported"
        )

    test = translate_expr(node.test, ctx)
    lines = [f"    while ({test}) {{"]

    for stmt in node.body:
        nested = translate_stmt(stmt, ctx)
        lines.extend(
            ["    " + line if line.strip() else line for line in nested]
        )
    lines.append("    }")
    return lines
