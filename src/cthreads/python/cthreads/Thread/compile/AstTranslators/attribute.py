"""
Copyright (c) 2026 Tobias Karusseit
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

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
    """
    Translates attribute expressions to a C++ expression

    #### Args
    - node: ast.Attribute - the attribute node to translate
    - ctx: TranslateContext - the translate context of the function

    #### Returns
    - str - a C++ expression that represents the translated ast.Attribute node
    """
    from .translate import translate_expr

    const = resolve_math_const(node, ctx) # ceck for math constants
    if const is not None: # if the attribute is a math constant
        add_include(ctx.body_includes, ctx.seen_body, NUMBERS_INCLUDE) # add the numbers include
        return const # return the math constant

    if ( #if the attribute is a self attribute add the this-> prefix
        isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and ctx.owner_name
    ):
        return f"this->{node.attr}" # return the attribute with the this-> prefix

    base = translate_expr(node.value, ctx) # translate the base of the attribute
    return f"{base}.{node.attr}" # return the base and attribute wrapped in a dot
