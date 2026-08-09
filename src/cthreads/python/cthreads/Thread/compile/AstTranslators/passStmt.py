"""
Copyright (c) 2026 Tobias Karusseit
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Translate `ast.Pass` — no C++ output.
"""

import ast
from .context import TranslateContext


def translate(node: ast.Pass, ctx: TranslateContext) -> list[str]:
    """
    Translates an ast.Pass node to a list of C++ lines

    returns an empty list since pass is a no-op in c++
    """
    return []
