"""
Translate `ast.Attribute` — field access / math constants.

Python:  p.x       self.x       math.pi
AST:     Attribute(...)
C++:     p.x       this->x      std::numbers::pi
"""

import ast

from ..lib import add_include
from ..mathLibTranslators import NUMBERS_INCLUDE, resolve_math_const
from .context import TranslateContext


def translate(node: ast.Attribute, ctx: TranslateContext) -> str:
    from .translate import translate_expr

    const = resolve_math_const(node, ctx)
    if const is not None:
        add_include(ctx.body_includes, ctx.seen_body, NUMBERS_INCLUDE)
        return const

    if (
        isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and ctx.owner_name
    ):
        return f"this->{node.attr}"

    base = translate_expr(node.value, ctx)
    return f"{base}.{node.attr}"
