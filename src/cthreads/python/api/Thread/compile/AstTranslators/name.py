"""
Translate `ast.Name` — a variable / parameter reference.

Python:  a
AST:     Name(id='a', ctx=Load())
C++:     a

`self` on a method lowers to `(*this)`.
"""

import ast

from .context import TranslateContext


def translate(node: ast.Name, ctx: TranslateContext) -> str:
    if node.id == "self" and ctx.owner_name:
        return "(*this)"
    if node.id not in ctx.symbols:
        raise TypeError(
            f"Thread function {ctx.func_name}: unknown name {node.id!r}"
        )
    return node.id
