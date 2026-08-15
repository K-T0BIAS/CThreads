"""
Copyright (c) 2026 Tobias Karusseit
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

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
    """
    Translates boolean operations to a C++ expression

    #### Args
    - node: ast.BoolOp - the boolean operation node to translate
    - ctx: TranslateContext - the translate context of the function

    #### Returns
    - str - a C++ expression that represents the translated ast.BoolOp node
    """
    from .translate import translate_expr

    op = BOOLOPS.get(type(node.op)) # get the boolean operator from the BOOLOPS dictionary
    if not op: # unsupported boolean operator
        raise TypeError(
            f"Thread function {ctx.func_name}: "
            f"unsupported bool operator {type(node.op).__name__}"
        )
    if len(node.values) < 2: # BoolOp needs at least two values
        raise TypeError(
            f"Thread function {ctx.func_name}: BoolOp needs at least two values"
        )
    # translate the values
    parts = [translate_expr(v, ctx) for v in node.values]
    return "(" + f" {op} ".join(parts) + ")" # join the values with the operator and package in parentheses
