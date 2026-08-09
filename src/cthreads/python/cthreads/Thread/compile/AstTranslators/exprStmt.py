"""
Copyright (c) 2026 Tobias Karusseit
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Translate `ast.Expr` — expression statements.

Docstrings appear as Expr(Constant("...")). Those are ignored.
Other bare expressions are not supported yet (need Call lowering, etc.).
"""

import ast

from .context import TranslateContext


def translate(node: ast.Expr, ctx: TranslateContext) -> list[str]:
    """
    ensures that docstrings are ignored
    """
    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
        return []
    return [f"    // unsupported statement: Expr ({type(node.value).__name__})"]
