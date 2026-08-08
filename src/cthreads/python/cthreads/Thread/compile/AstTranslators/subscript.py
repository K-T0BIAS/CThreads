"""
Translate `ast.Subscript` — indexing (lists / vectors).

Python:  xs[i]
AST:     Subscript(value=Name('xs'), slice=Name('i'))
C++:     xs[i]
"""

import ast

from .context import TranslateContext


def translate(node: ast.Subscript, ctx: TranslateContext) -> str:
    from .translate import translate_expr

    if isinstance(node.slice, ast.Slice):
        raise TypeError(
            f"Thread function {ctx.func_name}: slice syntax is not supported"
        )

    base = translate_expr(node.value, ctx)
    index = translate_expr(node.slice, ctx)
    return f"({base}[{index}])"
