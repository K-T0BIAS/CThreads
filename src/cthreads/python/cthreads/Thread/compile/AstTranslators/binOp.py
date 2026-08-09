"""
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
    from .translate import translate_expr

    left = translate_expr(node.left, ctx)
    right = translate_expr(node.right, ctx)

    if isinstance(node.op, ast.Pow):
        add_include(ctx.body_includes, ctx.seen_body, CMATH_INCLUDE)
        return f"std::pow({left}, {right})"

    op = BINOPS.get(type(node.op))
    if not op:
        raise TypeError(
            f"Thread function {ctx.func_name}: "
            f"unsupported binary operator {type(node.op).__name__}"
        )

    return f"({left} {op} {right})"
