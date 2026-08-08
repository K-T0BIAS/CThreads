"""
Translate `ast.Continue` — same keyword in C++.
"""

import ast

from .context import TranslateContext


def translate(node: ast.Continue, ctx: TranslateContext) -> list[str]:
    return ["    continue;"]
