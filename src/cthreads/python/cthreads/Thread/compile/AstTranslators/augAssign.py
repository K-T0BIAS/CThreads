"""
Copyright (c) 2026 Tobias Karusseit
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Translate `ast.AugAssign` - augmented assignment statement.

Python:  p.x += p.velocity * dt
AST:     AugAssign(
           target=Attribute(Name('p'), 'x'),
           op=Add(),
           value=BinOp(...),
         )
C++:     p.x += (p.velocity * dt);

Walk: translate_expr(target), translate_expr(value), then emit `t op= v;`.
"""

import ast

from ....pyOps import BINOPS
from .context import TranslateContext


def translate(node: ast.AugAssign, ctx: TranslateContext) -> list[str]:
    """
    Translates augmented assignment statements to a list of C++ lines

    #### Args
    - node: ast.AugAssign - the augmented assignment node to translate
    - ctx: TranslateContext - the translate context of the function
    
    #### Returns
    - list[str] - a list of C++ lines that represent the translated ast.AugAssign node
    """
    from ..lib import add_include
    from ..mathLibTranslators import CMATH_INCLUDE
    from .translate import translate_expr
    target = translate_expr(node.target, ctx) # translate the target of the augmented assignment
    value = translate_expr(node.value, ctx) # translate the value of the augmented assignment
    if isinstance(node.op, ast.Pow): # handle ** power operator
        add_include(ctx.body_includes, ctx.seen_body, CMATH_INCLUDE) # add the math include
        return [f"    {target} = std::pow({target}, {value});"] # return the target and value wrapped in std::pow
    op = BINOPS.get(type(node.op)) # get the binary operator from the BINOPS dictionary
    if not op: # unsupported binary operator
        raise TypeError(
            f"Thread function {ctx.func_name}: "
            f"unsupported aug-assign operator {type(node.op).__name__}"
        )

    return [f"    {target} {op}= {value};"] # return the target and value wrapped in the binary operator
