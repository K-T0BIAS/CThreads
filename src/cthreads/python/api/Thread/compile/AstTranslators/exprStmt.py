"""
Translate `ast.Expr` — expression statements.

Docstrings appear as Expr(Constant("...")). Those are ignored.
Other bare expressions are not supported yet (need Call lowering, etc.).
"""

import ast

from .context import TranslateContext


def translate(node: ast.Expr, ctx: TranslateContext) -> list[str]:
    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
        return []
    return [f"    // unsupported statement: Expr ({type(node.value).__name__})"]
