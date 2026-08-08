"""
Translate `ast.AnnAssign` â€” annotated declaration / assignment statement.

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
    from .translate import translate_expr

    if not isinstance(node.target, ast.Name):
        raise TypeError(
            f"Thread function {ctx.func_name}: AnnAssign target must be a plain name"
        )

    var_name = node.target.id
    if var_name in ctx.symbols:
        raise TypeError(
            f"Thread function {ctx.func_name}: redeclaration of {var_name!r}"
        )

    hint = resolve_annotation(node.annotation, ctx.fn.__globals__)
    py_type = hint_to_pytype(hint)
    ctx.symbols[var_name] = py_type
    add_include(ctx.body_includes, ctx.seen_body, include_for(py_type))

    if node.value is None:
        decl, _ = py_type.to_cpp(var_name)
        return [f"    {decl}"]

    # Walk the RHS expression tree first (Constant, Name, BinOp, ...).
    rhs = translate_expr(node.value, ctx)
    decl, _ = py_type.to_cpp(var_name, rhs)
    return [f"    {decl}"]
