"""
Translate `ast.AugAssign` â€” augmented assignment statement.

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
    from .translate import translate_expr

    op = BINOPS.get(type(node.op))
    if not op:
        raise TypeError(
            f"Thread function {ctx.func_name}: "
            f"unsupported aug-assign operator {type(node.op).__name__}"
        )

    target = translate_expr(node.target, ctx)
    value = translate_expr(node.value, ctx)
    return [f"    {target} {op}= {value};"]
