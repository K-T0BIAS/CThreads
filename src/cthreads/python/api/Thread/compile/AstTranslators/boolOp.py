"""
Translate `ast.BoolOp` — `and` / `or`.

Python:  a and b or c
AST:     BoolOp(op=And()/Or(), values=[...])
C++:     ((a) && (b) || (c))   — joined with && / ||

Note: Python returns the operand value; C++ yields bool. Fine for
conditions and bool-typed Thread results in this subset.
"""

import ast

from ....pyOps import BOOLOPS
from .context import TranslateContext


def translate(node: ast.BoolOp, ctx: TranslateContext) -> str:
    from .translate import translate_expr

    op = BOOLOPS.get(type(node.op))
    if not op:
        raise TypeError(
            f"Thread function {ctx.func_name}: "
            f"unsupported bool operator {type(node.op).__name__}"
        )
    if len(node.values) < 2:
        raise TypeError(
            f"Thread function {ctx.func_name}: BoolOp needs at least two values"
        )

    parts = [translate_expr(v, ctx) for v in node.values]
    return "(" + f" {op} ".join(parts) + ")"
