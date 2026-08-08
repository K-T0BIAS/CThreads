"""
Translate `ast.BinOp` â€” binary operator expression (the usual â€œOp rootâ€).

Python:  a + 10
AST:     BinOp(left=Name('a'), op=Add(), right=Constant(10))
C++:     (a + 10)

Walk order:
  1. translate_expr(left)   -> "a"
  2. translate_expr(right)  -> "10"
  3. join with C++ op from pyOps.BINOPS
"""

import ast

from ....pyOps import BINOPS
from .context import TranslateContext


def translate(node: ast.BinOp, ctx: TranslateContext) -> str:
    from .translate import translate_expr

    op = BINOPS.get(type(node.op))
    if not op:
        raise TypeError(
            f"Thread function {ctx.func_name}: "
            f"unsupported binary operator {type(node.op).__name__}"
        )

    left = translate_expr(node.left, ctx)
    right = translate_expr(node.right, ctx)
    return f"({left} {op} {right})"
