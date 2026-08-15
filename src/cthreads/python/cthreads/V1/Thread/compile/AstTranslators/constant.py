"""
Copyright (c) 2026 Tobias Karusseit
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Translate `ast.Constant` - a literal leaf in the expression tree.

Python:  10          "hi"        True
AST:     Constant(10) Constant("hi") Constant(True)
C++:     10          "hi"        true
"""

import ast

from ..lib import cpp_literal
from .context import TranslateContext


def translate(node: ast.Constant, ctx: TranslateContext) -> str:
    """
    Translates a constant node to a C++ literal
    #### Args
    - node: ast.Constant - the constant node to translate
    - ctx: TranslateContext - the translate context of the function

    #### Returns
    - str - a C++ literal that represents the translated ast.Constant node
    """
    return cpp_literal(node.value)
