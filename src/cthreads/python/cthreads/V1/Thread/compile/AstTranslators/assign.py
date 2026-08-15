"""
Copyright (c) 2026 Tobias Karusseit
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Translate `ast.Assign` — plain assignment.

Python:  p.velocity = limit
         xs[i] = v
         d[k] = v
         x = a + 1
AST:     Assign(targets=[...], value=...)
C++:     p.velocity = limit;
         (xs[i]) = v;
         (d[k]) = v;
         x = (a + 1);

New locals must still be introduced with AnnAssign (`x: int = ...`).
Assign updates an existing name, attribute, or subscript (list / dict / buffer index).
"""

import ast

from .context import TranslateContext


def translate(node: ast.Assign, ctx: TranslateContext) -> list[str]:
    """
    Translates assignment statements to a list of C++ lines

    #### Args
    - node: ast.Assign - the assignment node to translate
    - ctx: TranslateContext - the translate context of the function

    #### Returns
    - list[str] - a list of C++ lines that represent the translated ast.Assign node
    """
    from .translate import translate_expr

    if len(node.targets) != 1: # Assign needs exactly one target
        raise TypeError(
            f"Thread function {ctx.func_name}: "
            "only single-target assignment is supported"
        )

    target = node.targets[0] # get the target of the assignment
    if isinstance(target, ast.Name): # if the target is a name
        if target.id not in ctx.symbols: # if the target is not in the symbols (this is not an initialize value and thus cant be assigned)
            raise TypeError( 
                f"Thread function {ctx.func_name}: "
                f"assign to unknown name {target.id!r} "
                "(declare it with an annotated assignment first)"
            )
    elif isinstance(target, ast.Subscript):
        if isinstance(target.slice, ast.Slice):
            raise TypeError(
                f"Thread function {ctx.func_name}: "
                "slice assignment is not supported"
            )
    elif not isinstance(target, ast.Attribute):
        raise TypeError(
            f"Thread function {ctx.func_name}: "
            f"unsupported assign target {type(target).__name__}"
        )

    lhs = translate_expr(target, ctx) # translate the target of the assignment
    rhs = translate_expr(node.value, ctx) # translate the value of the assignment
    return [f"    {lhs} = {rhs};"] # return the target and value wrapped in an assignment
