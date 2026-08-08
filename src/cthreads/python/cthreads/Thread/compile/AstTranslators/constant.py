"""
Translate `ast.Constant` â€” a literal leaf in the expression tree.

Python:  10          "hi"        True
AST:     Constant(10) Constant("hi") Constant(True)
C++:     10          "hi"        true
"""

import ast

from ..lib import cpp_literal
from .context import TranslateContext


def translate(node: ast.Constant, ctx: TranslateContext) -> str:
    return cpp_literal(node.value)
