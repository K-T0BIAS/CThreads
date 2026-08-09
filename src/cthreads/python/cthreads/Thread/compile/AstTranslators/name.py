"""
Copyright (c) 2026 Tobias Karusseit
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Translate `ast.Name` — a variable / parameter reference.

Python:  a
AST:     Name(id='a', ctx=Load())
C++:     a

`self` on a method lowers to `(*this)`.
"""

import ast

from .context import TranslateContext


def translate(node: ast.Name, ctx: TranslateContext) -> str:
    """
    Translates an ast.Name node to a C++ expression (self is translated to (*this))

    #### Args
    - node: ast.Name - the ast.Name node to translate
    - ctx: TranslateContext - the translate context of the function

    #### Returns
    - str - a C++ expression that represents the translated ast.Name node
    """
    # if the name is self and the owner name is set, translate to (*this)
    if node.id == "self" and ctx.owner_name:
        return "(*this)"
    if node.id not in ctx.symbols:
        raise TypeError(
            f"Thread function {ctx.func_name}: unknown name {node.id!r}"
        )
    return node.id
