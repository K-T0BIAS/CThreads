"""
Translate `ast.Compare` — comparison expression.

Python:  a < b          a < b < c
AST:     Compare(left=..., ops=[Lt()], comparators=[...])
C++:     (a < b)        ((a < b) && (b < c))

Chained comparisons are lowered to && of pairwise ops (Python semantics).
"""

import ast

from ....pyOps import CMPOPS
from .context import TranslateContext


def translate(node: ast.Compare, ctx: TranslateContext) -> str:
    from .translate import translate_expr

    if len(node.ops) != len(node.comparators):
        raise TypeError(
            f"Thread function {ctx.func_name}: malformed Compare node"
        )

    left = translate_expr(node.left, ctx)
    parts: list[str] = []
    prev = left
    for op_node, comparator in zip(node.ops, node.comparators):
        op = CMPOPS.get(type(op_node))
        if not op:
            raise TypeError(
                f"Thread function {ctx.func_name}: "
                f"unsupported compare operator {type(op_node).__name__}"
            )
        right = translate_expr(comparator, ctx)
        parts.append(f"({prev} {op} {right})")
        prev = right

    if len(parts) == 1:
        return parts[0]
    return "(" + " && ".join(parts) + ")"
