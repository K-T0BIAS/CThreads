"""
Copyright (c) 2026 Tobias Karusseit
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Translate `ast.AnnAssign` - annotated declaration / assignment statement.

Python:  b: int = a + 10
AST:     AnnAssign(
           target=Name('b'),
           annotation=Name('int'),
           value=BinOp(Name('a'), Add(), Constant(10)),
         )

Walk:
  1. resolve annotation -> PyType / register symbol
  2. if value: rhs = translate_expr(value)   # descends into BinOp, etc.
  3. emit `int b = (a + 10);`
"""

import ast

from ....pyTypes import hint_to_pytype
from ..lib import add_include, include_for, resolve_annotation
from .context import TranslateContext


def translate(node: ast.AnnAssign, ctx: TranslateContext) -> list[str]:
    """
    Translates annotated declaration / assignment statements to a list of C++ lines

    #### Args
    - node: ast.AnnAssign - the annotated declaration / assignment node to translate
    - ctx: TranslateContext - the translate context of the function

    #### Returns
    - list[str] - a list of C++ lines that represent the translated ast.AnnAssign node
    """
    from .translate import translate_expr

    if not isinstance(node.target, ast.Name): # AnnAssign target must be a plain name
        raise TypeError(
            f"Thread function {ctx.func_name}: AnnAssign target must be a plain name"
        )

    var_name = node.target.id # get the name of the variable
    if var_name in ctx.symbols: # if the variable is already in the symbols (this is a redeclaration)
        raise TypeError(
            f"Thread function {ctx.func_name}: redeclaration of {var_name!r}"
        )

    hint = resolve_annotation(node.annotation, ctx.fn.__globals__) # resolve the annotation
    py_type = hint_to_pytype(hint) # convert the hint to a PyType
    ctx.symbols[var_name] = py_type # add the variable to the symbols
    add_include(ctx.body_includes, ctx.seen_body, include_for(py_type)) # add the include for the variable

    if node.value is None: # if the value is None (this is a declaration only)
        decl, _ = py_type.to_cpp(var_name)
        return [f"    {decl}"]

    # Walk the RHS expression tree first (Constant, Name, BinOp, ...).
    rhs = translate_expr(node.value, ctx) # translate the value of the annotated declaration / assignment
    decl, _ = py_type.to_cpp(var_name, rhs) # convert the variable to a C++ declaration (returns type name = rhs)
    return [f"    {decl}"]
