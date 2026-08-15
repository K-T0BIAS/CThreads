"""
Copyright (c) 2026 Tobias Karusseit
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Translate `ast.Compare` - comparison expression.

Python:  a < b          a < b < c
AST:     Compare(left=..., ops=[Lt()], comparators=[...])
C++:     (a < b)        ((a < b) && (b < c))

Chained comparisons are lowered to && of pairwise ops (Python semantics).
"""

import ast

from ....pyOps import CMPOPS
from .context import TranslateContext


def translate(node: ast.Compare, ctx: TranslateContext) -> str:
    """
    Translates comparison expressions to a C++ expression

    #### Args
    - node: ast.Compare - the comparison node to translate
    - ctx: TranslateContext - the translate context of the function

    #### Returns
    - str - a C++ expression that represents the translated ast.Compare node
    """
    from .translate import translate_expr

    if len(node.ops) != len(node.comparators): # malformed Compare node
        raise TypeError(
            f"Thread function {ctx.func_name}: malformed Compare node"
        )

    # translate the left side of the comparison
    left = translate_expr(node.left, ctx)
    parts: list[str] = []
    prev = left
    # iteratively build the C++ expression for each comparison operator and comparator pair (left to right)
    for op_node, comparator in zip(node.ops, node.comparators):
        op = CMPOPS.get(type(op_node)) # get the comparison operator from the CMPOPS dictionary
        if not op: # unsupported comparison operator
            raise TypeError(
                f"Thread function {ctx.func_name}: "
                f"unsupported compare operator {type(op_node).__name__}"
            )
        # translate the right side of the comparison
        right = translate_expr(comparator, ctx)
        parts.append(f"({prev} {op} {right})") # merge the left and right sides with the operator
        prev = right # update the previous side to the right side for the next iteration

    if len(parts) == 1: # single comparison
        return parts[0]
    return "(" + " && ".join(parts) + ")" # multiple comparisons so join with && and package each in parentheses
