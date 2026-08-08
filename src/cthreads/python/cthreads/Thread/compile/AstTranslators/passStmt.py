"""
Translate `ast.Pass` — no C++ output.
"""

import ast
from .context import TranslateContext


def translate(node: ast.Pass, ctx: TranslateContext) -> list[str]:
    return []
