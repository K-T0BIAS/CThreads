"""
Copyright (c) 2026 Tobias Karusseit
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Translate `ast.Break` — same keyword in C++.
"""

import ast

from .context import TranslateContext


def translate(node: ast.Break, ctx: TranslateContext) -> list[str]:
    """
    Turn python break -> C++ break;
    """
    return ["    break;"]
