"""
Copyright (c) 2026 Tobias Karusseit
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Translate `ast.Expr` — expression statements.

Docstrings appear as Expr(Constant("...")). Those are ignored.
Bare calls (e.g. ``xs.append(v)``) lower via ``translate_expr``.
"""

import ast

from .context import TranslateContext


def translate(node: ast.Expr, ctx: TranslateContext) -> list[str]:
    """
    Docstrings are ignored; Call exprs become ``    <expr>;``.
    """
    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
        return []
    if isinstance(node.value, ast.Call):
        from .translate import translate_expr

        return [f"    {translate_expr(node.value, ctx)};"]
    return [f"    // unsupported statement: Expr ({type(node.value).__name__})"]
