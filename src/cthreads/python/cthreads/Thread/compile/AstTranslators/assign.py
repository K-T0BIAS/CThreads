"""
Translate `ast.Assign` — plain assignment.

Python:  p.velocity = limit
         x = a + 1
AST:     Assign(targets=[Attribute(...)], value=...)
C++:     p.velocity = limit;
         x = (a + 1);

New locals must still be introduced with AnnAssign (`x: int = ...`).
Assign only updates an existing name or an attribute.
"""

import ast

from .context import TranslateContext


def translate(node: ast.Assign, ctx: TranslateContext) -> list[str]:
    from .translate import translate_expr

    if len(node.targets) != 1:
        raise TypeError(
            f"Thread function {ctx.func_name}: "
            "only single-target assignment is supported"
        )

    target = node.targets[0]
    if isinstance(target, ast.Name):
        if target.id not in ctx.symbols:
            raise TypeError(
                f"Thread function {ctx.func_name}: "
                f"assign to unknown name {target.id!r} "
                "(declare it with an annotated assignment first)"
            )
    elif not isinstance(target, ast.Attribute):
        raise TypeError(
            f"Thread function {ctx.func_name}: "
            f"unsupported assign target {type(target).__name__}"
        )

    lhs = translate_expr(target, ctx)
    rhs = translate_expr(node.value, ctx)
    return [f"    {lhs} = {rhs};"]
