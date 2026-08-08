"""
Translate `ast.If` — if / elif / else.

Python:
    if cond:
        ...
    else:
        ...

AST:
    If(test=..., body=[...], orelse=[...])
    # elif is orelse=[If(...)]

C++:
    if (<test>) {
        ...
    } else {
        ...
    }
"""

import ast

from .context import TranslateContext


def _nest(lines: list[str]) -> list[str]:
    """Body stmts already have one indent level; nest one more inside {{ }}."""
    return ["    " + line if line.strip() else line for line in lines]


def translate(node: ast.If, ctx: TranslateContext) -> list[str]:
    from .translate import translate_expr, translate_stmt

    test = translate_expr(node.test, ctx)
    lines = [f"    if ({test}) {{"]

    for stmt in node.body:
        lines.extend(_nest(translate_stmt(stmt, ctx)))
    lines.append("    }")

    if node.orelse:
        lines.append("    else {")
        for stmt in node.orelse:
            lines.extend(_nest(translate_stmt(stmt, ctx)))
        lines.append("    }")

    return lines
