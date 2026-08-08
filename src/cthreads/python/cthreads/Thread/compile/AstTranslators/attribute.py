"""
Translate `ast.Attribute` — field access.

Python:  p.x       self.x
AST:     Attribute(...)
C++:     p.x       this->x
"""

import ast

from .context import TranslateContext


def translate(node: ast.Attribute, ctx: TranslateContext) -> str:
    from .translate import translate_expr

    if (
        isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and ctx.owner_name
    ):
        return f"this->{node.attr}"

    base = translate_expr(node.value, ctx)
    return f"{base}.{node.attr}"
