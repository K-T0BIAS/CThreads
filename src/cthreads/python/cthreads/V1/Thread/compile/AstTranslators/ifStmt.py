"""
Copyright (c) 2026 Tobias Karusseit
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Translate `ast.If` — if / elif / else.

Python:
    if cond:
        ...
    else:
        ...

AST:
    If(test=..., body=[...], orelse=[...])
    # elif is orelse=[If(...)]

C++:
    if (<test>) {
        ...
    } else {
        ...
    }
"""

import ast

from .context import TranslateContext


def _nest(lines: list[str]) -> list[str]:
    """Body stmts already have one indent level; nest one more inside {{ }}."""
    return ["    " + line if line.strip() else line for line in lines]


def translate(node: ast.If, ctx: TranslateContext) -> list[str]:
    """
    Translates if/else statements to a list of C++ lines

    #### Args
    - node: ast.If - the ast.If node to translate
    - ctx: TranslateContext - the translate context of the function

    #### Returns
    - list[str] - a list of C++ lines that represent the translated ast.If node
    """
    from .translate import translate_expr, translate_stmt

    # translate the condition of the if statement
    test = translate_expr(node.test, ctx)
    lines = [f"    if ({test}) {{"]

    # walk teh body to translate each staement at the coorect indent level
    for stmt in node.body:
        lines.extend(_nest(translate_stmt(stmt, ctx)))
    lines.append("    }") # close the if statement

    # if there is an else statement, translate it
    if node.orelse:
        lines.append("    else {")
        for stmt in node.orelse:
            lines.extend(_nest(translate_stmt(stmt, ctx)))
        lines.append("    }")

    return lines
