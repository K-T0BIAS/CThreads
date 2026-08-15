"""
Copyright (c) 2026 Tobias Karusseit
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Translate `ast.BinOp` — binary operator expression (the usual “Op root”).

Python:  a + 10          a ** b
AST:     BinOp(...)      BinOp(..., op=Pow())
C++:     (a + 10)        std::pow(a, b)

Walk order:
  1. translate_expr(left)
  2. translate_expr(right)
  3. join with C++ op from pyOps.BINOPS, or std::pow for **
"""

import ast

from ....pyOps import BINOPS
from ..lib import add_include
from ..mathLibTranslators import CMATH_INCLUDE
from .context import TranslateContext


def translate(node: ast.BinOp, ctx: TranslateContext) -> str:
    """
    Translates binary operator expressions to a C++ expression

    #### Args
    - node: ast.BinOp - the binary operator node to translate
    - ctx: TranslateContext - the translate context of the function

    #### Returns
    - str - a C++ expression that represents the translated ast.BinOp node
    """
    from .translate import translate_expr

    left = translate_expr(node.left, ctx) # translate the left side of the binary operator
    right = translate_expr(node.right, ctx) # translate the right side of the binary operator

    if isinstance(node.op, ast.Pow): # handle ** power operator
        add_include(ctx.body_includes, ctx.seen_body, CMATH_INCLUDE) # add the math include
        return f"std::pow({left}, {right})" # return the left and right sides wrapped in std::pow

    op = BINOPS.get(type(node.op)) # get the binary operator from the BINOPS dictionary
    if not op:
        raise TypeError(
            f"Thread function {ctx.func_name}: "
            f"unsupported binary operator {type(node.op).__name__}" # unsupported binary operator
        )

    return f"({left} {op} {right})" # return the left and right sides wrapped in the binary operator
