"""
Translate `ast.Return` — return statement.

Python:  return          return p.velocity > limit
AST:     Return(value=None)   Return(value=Compare(...))
C++:     return;         return (p.velocity > limit);
"""

import ast

from .context import TranslateContext


def translate(node: ast.Return, ctx: TranslateContext) -> list[str]:
    from .translate import translate_expr

    if node.value is None:
        return ["    return;"]

    value = translate_expr(node.value, ctx)
    return [f"    return {value};"]
