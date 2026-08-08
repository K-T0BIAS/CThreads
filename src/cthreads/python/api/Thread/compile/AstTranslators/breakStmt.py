"""
Translate `ast.Break` — same keyword in C++.
"""

import ast

from .context import TranslateContext


def translate(node: ast.Break, ctx: TranslateContext) -> list[str]:
    return ["    break;"]
