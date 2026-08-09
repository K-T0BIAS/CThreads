"""
Copyright (c) 2026 Tobias Karusseit
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Translate `ast.UnaryOp` prefix operator.

Python:  -x
AST:     UnaryOp(op=USub(), operand=Name('x'))
C++:     (-x)
"""

import ast

from ....pyOps import UNARYOPS
from .context import TranslateContext


def translate(node: ast.UnaryOp, ctx: TranslateContext) -> str:
    """
    Translates an ast.UnaryOp node to a C++ expression

    #### Args
    - node: ast.UnaryOp - the ast.UnaryOp node to translate
    - ctx: TranslateContext - the translate context of the function

    #### Returns
    - str - a C++ expression that represents the translated ast.UnaryOp node

    #### Example

    ```python
    -x
    ```
    
    ----

    ```cpp
    (-x) // uses brackets to ensure correct operator precedence
    ```
    """
    from .translate import translate_expr

    op = UNARYOPS.get(type(node.op)) # get the unary operation from the UNARYOPS dict
    if not op: # no unary operation found so unsupported
        raise TypeError(
            f"Thread function {ctx.func_name}: "
            f"unsupported unary operator {type(node.op).__name__}"
        )

    operand = translate_expr(node.operand, ctx) # translate the operand to c++
    # return the C++ expression with brackets to ensure correct operator precedence
    return f"({op}{operand})"
