"""
Copyright (c) 2026 Tobias Karusseit
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Translate `ast.Subscript` — indexing (lists / vectors).

Python:  xs[i]
AST:     Subscript(value=Name('xs'), slice=Name('i'))
C++:     xs[i]
"""

import ast

from .context import TranslateContext


def translate(node: ast.Subscript, ctx: TranslateContext) -> str:
    """
    Translates an ast.Subscript node to a C++ expression (used for indexing lists / vectors and dicts)

    #### Args
    - node: ast.Subscript - the ast.Subscript node to translate
    - ctx: TranslateContext - the translate context of the function

    #### Returns
    - str - a C++ expression that represents the translated ast.Subscript node
    """
    from .translate import translate_expr

    if isinstance(node.slice, ast.Slice): # slice syntax is not supported
        raise TypeError(
            f"Thread function {ctx.func_name}: slice syntax is not supported"
        )

    base = translate_expr(node.value, ctx) # translate the base of the subscript
    index = translate_expr(node.slice, ctx) # translate the index of the subscript
    # return the C++ expression with brackets to ensure correct operator precedence
    return f"({base}[{index}])"
