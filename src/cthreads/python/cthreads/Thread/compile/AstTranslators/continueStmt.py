"""
Copyright (c) 2026 Tobias Karusseit
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Translate `ast.Continue` — same keyword in C++.
"""

import ast

from .context import TranslateContext


def translate(node: ast.Continue, ctx: TranslateContext) -> list[str]:
    """
    Translate continue statements to continue;
    """
    return ["    continue;"]
