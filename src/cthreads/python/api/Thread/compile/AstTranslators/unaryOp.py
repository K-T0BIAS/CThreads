"""
Translate `ast.UnaryOp` â€” prefix operator.

Python:  -x
AST:     UnaryOp(op=USub(), operand=Name('x'))
C++:     (-x)
"""

import ast

from ....pyOps import UNARYOPS
from .context import TranslateContext


def translate(node: ast.UnaryOp, ctx: TranslateContext) -> str:
    from .translate import translate_expr

    op = UNARYOPS.get(type(node.op))
    if not op:
        raise TypeError(
            f"Thread function {ctx.func_name}: "
            f"unsupported unary operator {type(node.op).__name__}"
        )

    operand = translate_expr(node.operand, ctx)
    return f"({op}{operand})"
