"""
Copyright (c) 2026 Tobias Karusseit
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Translate `ast.Return` — return statement.

Python:  return          return p.velocity > limit
AST:     Return(value=None)   Return(value=Compare(...))
C++:     return;         return (p.velocity > limit);
"""

import ast

from .context import TranslateContext


def translate(node: ast.Return, ctx: TranslateContext) -> list[str]:
    """
    Translates an ast.Return node to a list of C++ lines

    #### Args
    - node: ast.Return - the ast.Return node to translate
    - ctx: TranslateContext - the translate context of the function

    #### Returns
    - list[str] - a list of C++ lines that represent the translated ast.Return node
    """
    from .translate import translate_expr

    if node.value is None:
        return ["    return;"]

    value = translate_expr(node.value, ctx)
    return [f"    return {value};"]
